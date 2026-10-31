"""Bounded three-turn negotiation coordinator.

Flow (fixed, documented):
  Turn 1  ATC -> PILOT -> GROUND   establish an initial plan, aircraft limits, ground alternatives
  Turn 2  ATC -> PILOT -> GROUND   negotiate revisions / conditional offers (new proposals still allowed)
  FREEZE  deterministic rule picks ONE complete, validated candidate (see `_freeze`); no plan changes after this
  Turn 3  ATC -> PILOT -> GROUND   each may only ACCEPT or REJECT that identical frozen proposal
  Consensus = three ACCEPTs of the same proposal id + scenario version with every stated condition satisfied, AND a
  fresh independent re-validation of the immutable stored plan. Only then is AGENT_CONSENSUS recorded.

Any timeout / API error / unrecoverable response / no valid candidate / no unanimity => deterministic fallback, reported
as FALLBACK_SELECTED or NO_FEASIBLE_PLAN (never as consensus). Max 3 turns, 9 messages. A schema-repair retry is another
MODEL ATTEMPT for the same message slot, not another message. The fallback is never run first.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import ValidationError

from . import agents, eventlog
from .fallback import Outcome, run_fallback
from .llm import AgentClient, ModelAPIError, ModelRequest, ModelTimeout, sanitize
from .messages import (AgentOutput, AgentRole, Condition, MessageType, NegotiationMessage, Participant, SPEAKING_ORDER,
                       legacy_participants)
from .metrics import all_satisfied, evaluate_conditions, metric_delta, plan_metrics
from .negotiation_types import (AgentContext, DecisionRecord, FallbackReason, FinalStatus, ModelAttempt,
                                NegotiationResult, ProposalRecord, ToolCallRecord, ValidationEvent, Vote)
from .plan import RecoveryPlan, make_plan
from .reports import compare_readiness, compare_reports, departure_readiness, pilot_report
from .reference import REFERENCE_DESCRIPTION, reference_plan
from .schemas import Scenario
from .tools import SOURCE, materialise
from .validate import validate_plan

FREEZE_RULE = ("Among proposals from turns 1-2 that pass full validation, rank by (1) most agents whose latest stated "
               "conditions are all satisfied by the plan, (2) fewest agents who rejected it, (3) most agents who accepted it "
               "with conditions resolved, (4) lowest fixed-weight total cost, (5) earliest version. Pick the first.")


@dataclass
class NegotiationConfig:
    call_timeout_s: float = 90.0
    deadline_s: float = 600.0
    fallback_time_budget_s: float = 30.0
    require_pilot_report: bool = False   # PILOT agents must send a structured AircraftReport in their first message


class MessageError(Exception):
    """The reply was not an acceptable message (schema / protocol / reference error). Eligible for ONE repair retry."""


class _Abort(Exception):
    def __init__(self, reason: FallbackReason, detail: str):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


class _Invalidated(Exception):
    pass


def plan_digest(plan: RecoveryPlan) -> str:
    return hashlib.sha256(json.dumps(plan.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


def new_session_id() -> str:
    return "neg-" + uuid.uuid4().hex[:10]


class Coordinator:
    def __init__(self, scenario: Scenario, client: AgentClient, *, config: Optional[NegotiationConfig] = None,
                 session_id: Optional[str] = None, log: bool = True, log_path=None,
                 scenario_guard: Optional[Callable[[], str]] = None, clock: Callable[[], float] = time.monotonic,
                 participants: Optional[list[Participant]] = None):
        self.sc, self.client = scenario, client
        self.participants = list(participants) if participants else legacy_participants(scenario.simulated.flights[0].flight_id)
        self.subject_flight = scenario.simulated.flights[0].flight_id
        self.cfg = config or NegotiationConfig()
        self.sid = session_id or new_session_id()
        self.log, self.log_path, self.guard, self.clock = log, log_path, scenario_guard, clock
        self.version = scenario.scenario_version
        self.messages: list[NegotiationMessage] = []
        self.proposals: list[ProposalRecord] = []
        self.attempts: list[ModelAttempt] = []
        self.decisions: list[DecisionRecord] = []
        self.standing: dict[str, list[Condition]] = {p.agent_id: [] for p in self.participants}
        self.frozen: Optional[ProposalRecord] = None
        self.freeze_ranking: list[dict] = []
        self.votes: list[Vote] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.validations: list[ValidationEvent] = []
        self.reference = None
        self.reference_validation = None
        self._ref_view: Optional[dict] = None
        self._n_tool = self._n_val = 0
        self._t0 = 0.0
        self._planner = None

    # ------------------------------------------------------------------ logging
    def _event(self, etype: str, payload: dict, key: str, actor: str = "coordinator") -> None:
        if not self.log:
            return
        eventlog.log_event(self.sc.scenario_id, etype, {"session_id": self.sid, "scenario_version": self.version, **payload},
                           actor=actor, path=self.log_path or eventlog.DB_PATH, dedupe_key=f"{self.sid}:{key}")

    def _tool(self, p: Participant, turn: int, seq: int, tool: str, summary: dict) -> None:
        self._n_tool += 1
        rec = ToolCallRecord(session_id=self.sid, turn=turn, seq=seq, role=p.role, agent_id=p.agent_id, tool=tool, summary=summary)
        self.tool_calls.append(rec)
        self._event("tool_call", {"turn": turn, "seq": seq, "role": p.role.value, "agent_id": p.agent_id, "tool": tool, **summary},
                    f"tool:{self._n_tool}", actor=p.agent_id)

    def _validated(self, stage: str, proposal_id: str, report) -> None:
        self._n_val += 1
        ev = ValidationEvent(session_id=self.sid, stage=stage, proposal_id=proposal_id, passed=report.passed,
                             n_checks=len(report.results), failed_constraints=sorted({r.constraint_id for r in report.failures}))
        self.validations.append(ev)
        self._event("validation_performed", ev.model_dump(mode="json", exclude={"session_id"}), f"validation:{self._n_val}")

    def _elapsed(self) -> float:
        return self.clock() - self._t0

    def _check_guard(self) -> None:
        if self.guard is not None and self.guard() != self.version:
            raise _Invalidated(f"scenario changed from {self.version} to {self.guard()} during the session")

    # ------------------------------------------------------------------ public
    def run(self) -> NegotiationResult:
        self._t0 = self.clock()
        self._event("negotiation_started", {"mode": self.client.mode, "model": self.client.model_name,
                                           "call_timeout_s": self.cfg.call_timeout_s, "deadline_s": self.cfg.deadline_s,
                                           "freeze_rule": FREEZE_RULE}, "started")
        self._prepare_reference()
        try:
            for turn in (1, 2):
                for p in self.participants:
                    self._step(p, turn)
            frozen = self._freeze()
            if frozen is None:
                raise _Abort(FallbackReason.NO_VALID_CANDIDATE, self._no_candidate_detail())
            for p in self.participants:
                self._step(p, 3, frozen)
            return self._conclude(frozen)
        except _Invalidated as e:
            return self._finish(FinalStatus.SESSION_INVALIDATED, invalidation=str(e))
        except _Abort as e:
            return self._fallback(e.reason, e.detail)

    def _prepare_reference(self) -> None:
        """Define and validate the initial reference plan BEFORE any agent speaks; every agent sees it."""
        self.reference = reference_plan(self.sc)
        self.reference_validation = validate_plan(self.sc, self.reference)
        self._validated("reference", self.reference.proposal_id, self.reference_validation)
        self._ref_view = {
            "description": REFERENCE_DESCRIPTION, "proposal_id": self.reference.proposal_id,
            "valid": self.reference_validation.passed,
            "failed_constraints": [{"constraint": r.constraint_id, "flight": r.flight_id, "resource": r.resource_id,
                                    "why": r.explanation[:160]} for r in self.reference_validation.failures][:12],
            "plan": agents.base._plan_summary(self.reference.model_dump_json())}
        self._event("reference_plan", {"proposal_id": self.reference.proposal_id, "valid": self.reference_validation.passed,
                                       "failed_constraints": [[r.constraint_id, r.flight_id, r.resource_id] for r in self.reference_validation.failures],
                                       "plan": self.reference.model_dump(mode="json")}, "reference")

    # ------------------------------------------------------------------ one agent message
    def _context(self, p, turn, seq, allowed, frozen, repair) -> AgentContext:
        return AgentContext(agent=p, participants=self.participants, require_report=self.cfg.require_pilot_report,
                            role=p.role, turn=turn, seq=seq, session_id=self.sid, scenario=self.sc, historical=self.sc.historical,
                            messages=list(self.messages), proposals=list(self.proposals), standing=dict(self.standing),
                            allowed_types=allowed, reference=self._ref_view, frozen=frozen, repair_error=repair)

    def _record_attempt(self, p, turn, attempt, outcome, t_start, resp=None, error=None) -> None:
        a = ModelAttempt(session_id=self.sid, turn=turn, role=p.role, agent_id=p.agent_id, attempt=attempt,
                         kind="initial" if attempt == 1 else "schema_repair", outcome=outcome,
                         latency_ms=int((self.clock() - t_start) * 1000), mode=self.client.mode, model=self.client.model_name,
                         credential_mode=getattr(self.client, "credential_mode", None),
                         error=sanitize(error)[:600] if error else None,
                         input_tokens=getattr(resp, "input_tokens", None), output_tokens=getattr(resp, "output_tokens", None))
        self.attempts.append(a)
        self._event("model_attempt", a.model_dump(mode="json", exclude={"session_id"}), f"attempt:{turn}:{p.agent_id}:{attempt}",
                    actor=p.agent_id)

    def _step(self, p: Participant, turn: int, frozen: Optional[ProposalRecord] = None) -> NegotiationMessage:
        seq = len(self.messages) + 1
        if seq > 3 * len(self.participants):  # defensive: the loop structure already guarantees this
            raise _Abort(FallbackReason.RESPONSE_ERROR, "message budget exceeded")
        allowed = [MessageType.ACCEPT, MessageType.REJECT] if turn == 3 else list(MessageType)
        repair: Optional[str] = None
        for attempt in (1, 2):
            self._check_guard()
            remaining = self.cfg.deadline_s - self._elapsed()
            if remaining <= 0:
                raise _Abort(FallbackReason.DEADLINE_EXCEEDED,
                             f"total negotiation deadline of {self.cfg.deadline_s:g}s exceeded before {p.agent_id}'s turn-{turn} message")
            ctx = self._context(p, turn, seq, allowed, frozen, repair)
            timeout = min(self.cfg.call_timeout_s, remaining)
            tools = agents.tool_results(ctx)
            self._tool(p, turn, seq, "option_menu", {
                "attempt": attempt, "advisory": True, "ranked_by": f"{p.agent_id} objective",
                "options_returned": [o["proposal_id_if_proposed"] for o in tools.get("options", [])]})
            req = ModelRequest(role=p.role.value, agent_id=p.agent_id, system=agents.system_prompt(p), prompt=agents.user_prompt(ctx, tools), context=ctx,
                               turn=turn, timeout_s=timeout, attempt=attempt, repair_error=repair)
            t = self.clock()
            try:
                resp = self.client.complete(req)
            except ModelTimeout as e:
                self._record_attempt(p, turn, attempt, "timeout", t, error=str(e) or "timeout")
                raise _Abort(FallbackReason.AGENT_TIMEOUT, f"{p.agent_id} model call timed out after {timeout:g}s (turn {turn})")
            except ModelAPIError as e:
                self._record_attempt(p, turn, attempt, "api_error", t, error=str(e))
                raise _Abort(FallbackReason.API_ERROR, f"{p.agent_id} model API error (turn {turn}): {sanitize(str(e))[:200]}")
            except Exception as e:  # any other client failure is an unrecoverable call error, never a crash
                self._record_attempt(p, turn, attempt, "api_error", t, error=f"{type(e).__name__}: {e}")
                raise _Abort(FallbackReason.API_ERROR, f"{p.agent_id} client failure ({type(e).__name__}) in turn {turn}")
            if self.clock() - t > timeout:  # returned after its timeout: discard
                self._record_attempt(p, turn, attempt, "timeout", t, resp=resp, error="reply arrived after the call timeout")
                raise _Abort(FallbackReason.AGENT_TIMEOUT, f"{p.agent_id} reply exceeded the {timeout:g}s call timeout (turn {turn})")
            self._check_guard()
            try:
                msg = self._accept_reply(p, turn, seq, allowed, frozen, resp.text)
            except MessageError as e:
                self._record_attempt(p, turn, attempt, "schema_error", t, resp=resp, error=str(e))
                if attempt == 1:
                    repair = str(e)[:700]
                    continue
                raise _Abort(FallbackReason.RESPONSE_ERROR,
                             f"{p.agent_id}'s turn-{turn} reply was still invalid after the single repair retry: {str(e)[:200]}")
            self._record_attempt(p, turn, attempt, "ok", t, resp=resp)
            self._commit(msg, p)
            return msg
        raise AssertionError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ reply -> message
    def _accept_reply(self, p, turn, seq, allowed, frozen, text) -> NegotiationMessage:
        role = p.role
        from .agents.base import extract_json
        try:
            out = AgentOutput.model_validate_json(extract_json(text))
        except ValueError as e:  # json errors and pydantic ValidationError both derive from ValueError
            if isinstance(e, ValidationError):
                def one(x):
                    loc = ".".join(map(str, x["loc"]))
                    if x["type"] in ("enum", "literal_error"):   # the full allowed list is in the prompt; report the bad value
                        return f"{loc}: invalid value {x.get('input')!r} (must be exactly one of the allowed values listed in your instructions)"
                    return f"{loc}: {x['msg']}"[:220]
                raise MessageError("schema: " + "; ".join(one(x) for x in e.errors()[:5]))
            raise MessageError(f"not valid JSON: {e}")
        if out.type not in allowed:
            raise MessageError(f"{out.type.value} is not allowed in turn {turn}; allowed: {[t.value for t in allowed]}")
        by_id = {p.proposal_id: p for p in self.proposals}
        plan: Optional[RecoveryPlan] = None
        origin = None
        if out.type in (MessageType.PROPOSE, MessageType.COUNTER_PROPOSE):
            if out.type == MessageType.COUNTER_PROPOSE and out.referenced_proposal_id not in by_id:
                raise MessageError(f"referenced_proposal_id {out.referenced_proposal_id!r} is not a known proposal; "
                                   f"known: {list(by_id)}")
            if out.plan is not None:
                if out.plan.scenario_version != self.version:
                    raise MessageError(f"SCENARIO_VERSION_MISMATCH: plan.scenario_version {out.plan.scenario_version!r} "
                                       f"!= session scenario_version {self.version!r}")
                plan = make_plan(self.sc, SOURCE, out.plan.flights, out.plan.decision_time_min,
                                 declared_status=out.plan.declared_status, airport=out.plan.airport)
                origin = "agent_plan"
            else:
                try:
                    if self._planner is None:
                        from .planner import Planner
                        self._planner = Planner(self.sc)
                    plan = materialise(self.sc, out.plan_spec, self._planner)
                except ValueError as e:
                    self._tool(p, turn, seq, "materialise_plan_spec", {"ok": False, "error": str(e)[:200]})
                    raise MessageError(f"plan_spec rejected: {e}")
                self._tool(p, turn, seq, "materialise_plan_spec", {"ok": True, "proposal_id": plan.proposal_id})
                origin = "materialised_from_spec"
            pid = plan.proposal_id
            ref_metrics = by_id[out.referenced_proposal_id].metrics if out.referenced_proposal_id else None
        else:
            pid = out.proposal_id
            if pid not in by_id:
                raise MessageError(f"proposal_id {pid!r} is not a known proposal; known: {list(by_id)}")
            ref_metrics = by_id[pid].metrics

        report, report_ok, discrepancies, readiness = None, None, [], None
        if p.role == AgentRole.PILOT and p.flight_id:
            departure = self.sc.simulated.departure is not None
            first = not any(m.agent_id == p.agent_id for m in self.messages)
            if departure:
                if out.readiness is None and self.cfg.require_pilot_report and first:
                    raise MessageError("your first message must include `readiness` (next flight, scheduled departure, latest take-off by crew duty)")
                if out.readiness is not None:
                    readiness = departure_readiness(self.sc, p.flight_id)       # authoritative: computed by code
                    discrepancies = compare_readiness(out.readiness, readiness)
                    report_ok = not discrepancies
            else:
                if out.report is None and self.cfg.require_pilot_report and first:
                    raise MessageError("your first message must include `report` (arrival window, fuel margin, maximum acceptable holding time)")
                if out.report is not None:
                    report = pilot_report(self.sc, p.flight_id)                    # authoritative: computed by code
                    discrepancies = compare_reports(out.report, report)
                    report_ok = not discrepancies
        msg = NegotiationMessage(
            agent_id=p.agent_id, flight_id=p.flight_id or self.subject_flight, airport=p.airport, report=report,
            report_verified=report_ok, report_discrepancies=discrepancies, readiness=readiness, declared_status=out.declared_status,
            message_id=f"{self.sid}-m{seq}", session_id=self.sid, seq=seq, turn=turn, sender=role, scenario_version=self.version,
            type=out.type, proposal_id=pid, referenced_proposal_id=out.referenced_proposal_id, reason_codes=out.reason_codes,
            explanation=out.explanation, conditions=out.conditions, tradeoff=out.tradeoff, plan=plan, plan_origin=origin,
            plan_digest=plan_digest(plan) if plan else by_id[pid].digest)
        if out.type in (MessageType.ACCEPT, MessageType.REJECT):
            msg.condition_results = evaluate_conditions(out.conditions, ref_metrics)
            msg.conditions_resolved = all_satisfied(msg.condition_results)
            msg.outdated = bool(turn == 3 and frozen is not None and pid != frozen.proposal_id)
        return msg

    # ------------------------------------------------------------------ commit
    def _register(self, msg: NegotiationMessage) -> ProposalRecord:
        existing = next((p for p in self.proposals if p.proposal_id == msg.proposal_id), None)
        if existing:
            return existing  # identical content already registered: same immutable proposal
        plan = msg.plan
        try:
            metrics = plan_metrics(self.sc, plan)
        except Exception:
            metrics = {"flights": {}, "plan": {k: float("nan") for k in (
                "max_landing_delay_min", "total_landing_delay_min", "weighted_landing_delay", "total_service_wait_min",
                "ground_overtime_min", "gate_changes", "runway_changes", "total_cost")}}
        rec = ProposalRecord(proposal_id=plan.proposal_id, version_no=len(self.proposals) + 1, proposer=msg.sender, proposer_id=msg.agent_id, turn=msg.turn,
                             seq=msg.seq, referenced_proposal_id=msg.referenced_proposal_id,
                             plan_json=plan.model_dump_json(), digest=plan_digest(plan), origin=msg.plan_origin or "",
                             validation=validate_plan(self.sc, plan), metrics=metrics)
        self.proposals.append(rec)
        self._validated("registration", rec.proposal_id, rec.validation)
        self._event("proposal_registered", {
            "proposal_id": rec.proposal_id, "version_no": rec.version_no, "proposer": rec.proposer.value, "seq": rec.seq,
            "digest": rec.digest, "valid": rec.validation.passed, "proposer_id": rec.proposer_id,
            "failed_constraints": [[r.constraint_id, r.flight_id, r.resource_id, r.explanation[:160]] for r in rec.validation.failures],
            "plan": json.loads(rec.plan_json)}, f"proposal:{rec.proposal_id}", actor=msg.agent_id)
        return rec

    def _commit(self, msg: NegotiationMessage, p: Participant) -> None:
        role = p.role
        prev_pos = next((m for m in reversed(self.messages) if m.agent_id == p.agent_id), None)
        rec = self._register(msg) if msg.plan is not None else next(p for p in self.proposals if p.proposal_id == msg.proposal_id)
        self.messages.append(msg)
        self.standing[p.agent_id] = list(msg.conditions)
        # decision record: how this agent's position moved, with the measurable consequence recomputed by us
        if msg.referenced_proposal_id:   # a counter-offer is measured against the proposal it counters (what was on the table)
            prev_id = msg.referenced_proposal_id
            changed = True
        else:                            # otherwise against this agent's own previous position
            prev_id = prev_pos.proposal_id if prev_pos else None
            changed = bool(prev_pos and (prev_pos.proposal_id != msg.proposal_id or prev_pos.type != msg.type))
        prev_rec = next((p for p in self.proposals if p.proposal_id == prev_id), None)
        observed = metric_delta(prev_rec.metrics, rec.metrics) if (prev_rec and prev_rec.proposal_id != rec.proposal_id) else {}
        dec = DecisionRecord(session_id=self.sid, seq=msg.seq, sender=role, agent_id=p.agent_id, type=msg.type, proposal_id=msg.proposal_id,
                             previous_proposal_id=prev_id, changed_preference=changed,
                             reason_codes=[r.value for r in msg.reason_codes], explanation=msg.explanation,
                             declared_tradeoff=msg.tradeoff, observed_delta=observed)
        self.decisions.append(dec)
        self._event("agent_message", {
            "seq": msg.seq, "turn": msg.turn, "sender": role.value, "agent_id": p.agent_id, "flight_id": msg.flight_id,
            "airport": msg.airport, "type": msg.type.value, "proposal_id": msg.proposal_id,
            "report_verified": msg.report_verified, "report_discrepancies": msg.report_discrepancies,
            "declared_status": msg.declared_status.value if msg.declared_status else None,
            "referenced_proposal_id": msg.referenced_proposal_id, "reason_codes": dec.reason_codes, "explanation": msg.explanation,
            "conditions": [c.model_dump(mode="json", exclude_none=True) for c in msg.conditions],
            "conditions_resolved": msg.conditions_resolved, "outdated": msg.outdated, "plan_origin": msg.plan_origin,
            "proposal_valid": rec.validation.passed, "digest": msg.plan_digest}, f"msg:{msg.seq}", actor=p.agent_id)
        self._event("decision_record", dec.model_dump(mode="json", exclude={"session_id"}), f"decision:{msg.seq}", actor=p.agent_id)

    # ------------------------------------------------------------------ freeze
    def _no_candidate_detail(self) -> str:
        if not self.proposals:
            return "No agent proposal was registered in turns 1-2."
        bits = [f"{p.proposal_id} ({p.proposer.value}): " + ", ".join(sorted({r.constraint_id for r in p.validation.failures})[:8])
                for p in self.proposals[:4]]
        return "No proposal from turns 1-2 passed validation. Failed constraints: " + " | ".join(bits)

    def _freeze(self) -> Optional[ProposalRecord]:
        self._check_guard()
        valid = [p for p in self.proposals if p.validation.passed]
        rows = []
        for p in valid:
            support = sum(1 for q in self.participants if all_satisfied(evaluate_conditions(self.standing[q.agent_id], p.metrics)))
            rejects = len({m.agent_id for m in self.messages if m.type == MessageType.REJECT and m.proposal_id == p.proposal_id})
            accepts = len({m.agent_id for m in self.messages if m.type == MessageType.ACCEPT and m.proposal_id == p.proposal_id
                           and m.conditions_resolved})
            cost = p.metrics["plan"]["total_cost"]
            rows.append({"proposal_id": p.proposal_id, "version_no": p.version_no, "proposer": p.proposer_id or p.proposer.value,
                         "support": support, "rejects": rejects, "accepts": accepts, "total_cost": round(cost, 3),
                         "_key": (-support, rejects, -accepts, cost, p.version_no)})
        rows.sort(key=lambda r: r["_key"])
        self.freeze_ranking = [{k: v for k, v in r.items() if k != "_key"} for r in rows]
        for rec_row in rows:
            rec = next(p for p in self.proposals if p.proposal_id == rec_row["proposal_id"])
            fresh = validate_plan(self.sc, rec.plan)  # independent validation of the immutable stored plan before freezing
            self._validated("freeze", rec.proposal_id, fresh)
            if fresh.passed:
                self.frozen = rec
                self._event("candidate_frozen", {"proposal_id": rec.proposal_id, "digest": rec.digest, "rule": FREEZE_RULE,
                                                 "ranking": self.freeze_ranking, "validation_passed": True,
                                                 "n_checks": len(fresh.results)}, "frozen")
                return rec
        return None

    # ------------------------------------------------------------------ conclusion
    def _conclude(self, frozen: ProposalRecord) -> NegotiationResult:
        self._check_guard()
        t3 = {m.agent_id: m for m in self.messages if m.turn == 3}
        problems = []
        for q in self.participants:
            m = t3.get(q.agent_id)
            if m is None:
                self.votes.append(Vote(role=q.role, agent_id=q.agent_id, note="no turn-3 message")); problems.append(f"{q.agent_id}: no message"); continue
            ok = (m.type == MessageType.ACCEPT and m.proposal_id == frozen.proposal_id and m.scenario_version == self.version
                  and m.conditions_resolved is True and not m.outdated)
            note = ("accepted the frozen proposal" if ok else
                    "accepted an OUTDATED proposal " + m.proposal_id if m.type == MessageType.ACCEPT and m.outdated else
                    "ACCEPT with unresolved/unsatisfied conditions" if m.type == MessageType.ACCEPT and not m.conditions_resolved else
                    "rejected the frozen proposal")
            self.votes.append(Vote(role=q.role, agent_id=q.agent_id, seq=m.seq, type=m.type, proposal_id=m.proposal_id, counted_as_accept=ok, note=note))
            if not ok:
                problems.append(f"{q.agent_id}: {note}")
        if problems:
            raise _Abort(FallbackReason.NO_UNANIMOUS_AGREEMENT, "No unanimous acceptance of the frozen proposal - " + "; ".join(problems))
        # independent re-validation of the immutable stored plan BEFORE recording consensus
        plan = RecoveryPlan.model_validate_json(frozen.plan_json)
        if plan_digest(plan) != frozen.digest or plan.scenario_version != self.version:
            raise _Abort(FallbackReason.CONSENSUS_REVALIDATION_FAILED, "frozen plan digest / scenario version changed")
        self._check_guard()
        report = validate_plan(self.sc, plan)
        self._validated("final_revalidation", plan.proposal_id, report)
        if not report.passed:
            raise _Abort(FallbackReason.CONSENSUS_REVALIDATION_FAILED,
                         "final re-validation failed: " + ", ".join(sorted({r.constraint_id for r in report.failures})))
        self._event("AGENT_CONSENSUS", {"proposal_id": frozen.proposal_id, "digest": frozen.digest, "n_checks": len(report.results),
                                        "accepted_by": [q.agent_id for q in self.participants]}, "consensus")
        return self._finish(FinalStatus.AGENT_CONSENSUS, final_plan=plan, final_validation=report)

    def _fallback(self, reason: FallbackReason, detail: str) -> NegotiationResult:
        try:
            self._check_guard()
        except _Invalidated as e:
            return self._finish(FinalStatus.SESSION_INVALIDATED, invalidation=str(e))
        fb = run_fallback(self.sc, time_budget_s=self.cfg.fallback_time_budget_s)
        status = FinalStatus.FALLBACK_SELECTED if fb.outcome == Outcome.FALLBACK_SELECTED else FinalStatus.NO_FEASIBLE_PLAN
        self._event("fallback_invoked", {"reason": reason.value, "detail": detail, "outcome": fb.outcome.value,
                                         "search_status": fb.search_status.value, "proposal_id": fb.plan.proposal_id if fb.plan else None,
                                         "candidates_total": fb.candidates_total, "candidates_validated": fb.candidates_validated,
                                         "feasible_count": fb.feasible_count, "reasons": fb.reasons}, "fallback")
        return self._finish(status, fallback=fb, reason=reason, detail=detail, final_plan=fb.plan, final_validation=fb.validation)

    def _finish(self, status: FinalStatus, *, final_plan=None, final_validation=None, fallback=None, reason=None, detail=None,
                invalidation=None) -> NegotiationResult:
        complete_turns = sum(1 for t in (1, 2, 3) if len([m for m in self.messages if m.turn == t]) == len(self.participants))
        res = NegotiationResult(
            session_id=self.sid, mode=self.client.mode, model_name=self.client.model_name, scenario_id=self.sc.scenario_id,
            scenario_version=self.version, status=status, participants=self.participants, turns_completed=complete_turns, messages=self.messages,
            proposals=self.proposals, attempts=self.attempts, decisions=self.decisions, reference_plan=self.reference,
            reference_validation=self.reference_validation, tool_calls=self.tool_calls, validations=self.validations,
            frozen_proposal_id=self.frozen.proposal_id if self.frozen else None, freeze_ranking=self.freeze_ranking,
            votes=self.votes, final_plan=final_plan, final_validation=final_validation, fallback=fallback,
            fallback_reason=reason, fallback_detail=detail, invalidation_detail=invalidation,
            elapsed_s=round(self._elapsed(), 3), deadline_s=self.cfg.deadline_s)
        self._event("negotiation_finished", {
            "status": status.value, "consensus": res.consensus, "turns_completed": complete_turns, "messages": len(self.messages),
            "model_attempts": len(self.attempts), "frozen_proposal_id": res.frozen_proposal_id,
            "fallback_reason": reason.value if reason else None, "fallback_detail": detail,
            "fallback_outcome": fallback.outcome.value if fallback else None,
            "fallback_search_status": fallback.search_status.value if fallback else None,
            "final_proposal_id": final_plan.proposal_id if final_plan else None, "invalidation": invalidation,
            "elapsed_s": res.elapsed_s}, "finished")
        return res


def run_negotiation(scenario: Scenario, client: AgentClient, **kw) -> NegotiationResult:
    return Coordinator(scenario, client, **kw).run()
