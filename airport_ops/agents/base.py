"""Shared pieces for the three agents: protocol text, prompt assembly, JSON extraction and the scripted
negotiation engine (used only by ScriptedClient; live agents never run it)."""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from ..messages import (AgentOutput, AgentRole, Condition, ConditionField, ConditionOp, MessageType, PlanSpec,
                        ReasonCode, SPEAKING_ORDER, Tradeoff, TradeoffMetric)
from ..metrics import all_satisfied, evaluate_conditions
from ..negotiation_types import AgentContext, ProposalRecord
from ..tools import Candidate, best_for, candidate_pool

CONDITION_FIELDS_DOC = {
    "per flight, numeric (flight_id required)": [f.value for f in ConditionField][:9],
    "per flight, identifier (op == or !=, string value)": ["gate_id", "runway_id", "ground_crew_id"],
    "whole plan, numeric (omit flight_id)": [f.value for f in ConditionField][12:],
}

COMMON_PROTOCOL = f"""\
You are one of three INDEPENDENT agents negotiating a recovery plan for one disrupted flight (the 'selected flight')
and two competing flights, at one airport. The agents are ATC, PILOT and GROUND. The PILOT represents ONLY the selected
historical flight. The two competing flights are simulated traffic with fixed, deterministic constraints and have no agent.
You see only your own domain plus the structured messages the other agents have sent.

NON-NEGOTIABLE RULES
- Hard constraints (runway closures, wake separation, capacity, taxiway blockages, gate compatibility/overlap, crew duty
  and overlap, minimum turnaround, fuel reserve + contingency, sequencing) can NEVER be waived, relaxed or traded off. A plan that
  violates any is discarded by an independent validator no matter how many agents like it. Only soft preferences can be traded.
- Do not invent facts. Use only the data below and the tool results provided.
- All times are minutes from scenario start (t+N). Fuel is kg; burn rates kg per hour.

PROTOCOL
- One turn = one message from each agent, in the fixed order {' -> '.join(r.value for r in SPEAKING_ORDER)}.
  At most 3 turns (9 messages). Turns 1-2 negotiate; between turn 2 and 3 the coordinator deterministically freezes ONE
  candidate; in turn 3 you may only ACCEPT or REJECT that exact frozen proposal (no new plans).
- Reply with EXACTLY ONE JSON object and nothing else (no markdown, no prose). Fields:
  type: PROPOSE | COUNTER_PROPOSE | ACCEPT | REJECT          (only types listed in your_turn.allowed_message_types)
  reason_codes: list of codes from {[c.value for c in ReasonCode]}
  explanation: <=400 chars, concise decision explanation with numbers. Do NOT include private reasoning.
  PROPOSE / COUNTER_PROPOSE: give `plan_spec` (preferred) = {{"landing_order":[3 flight ids],"runways":{{flight:runway}},
     "gates":{{flight:gate}},"ground_crews":{{flight:crew}}}}; the coordinator builds the complete timeline deterministically.
     COUNTER_PROPOSE also needs `referenced_proposal_id` (the proposal you counter). Never set proposal_id on these.
  ACCEPT / REJECT: `proposal_id` = the exact proposal you answer. No plan. REJECT needs >=1 reason code.
  conditions: optional list of {{"field","op","value","flight_id"?}} that must hold for the proposal you accept/offer.
     They are evaluated mechanically; an ACCEPT whose conditions are not all satisfied is NOT a final acceptance.
     Allowed fields: {json.dumps(CONDITION_FIELDS_DOC)}; ops: {[o.value for o in ConditionOp]}.
  tradeoff: optional {{"metric","flight_id"?,"before","after","because"}} - the measurable change behind a revised
     position (metrics: {[m.value for m in TradeoffMetric]}). The coordinator recomputes the real numbers.
"""


def build_system(instructions: str) -> str:
    return instructions.strip() + "\n\n" + COMMON_PROTOCOL


def _plan_summary(plan_json: str) -> list[dict]:
    d = json.loads(plan_json)
    return [{"flight": f["flight_id"], "runway": f["runway_id"], "landing": round(f["landing_min"], 1),
             "gate": f["gate_id"], "gate_in": round(f["gate_in_min"], 1), "service": [round(f["service_start_min"], 1),
             round(f["service_end_min"], 1)], "ground_crew": f["ground_crew_id"]} for f in d["flights"]]


def proposals_view(ctx: AgentContext) -> list[dict]:
    """Registry as evidence: validity, failed constraints (structured) and consequences of every proposal so far."""
    out = []
    for p in ctx.proposals:
        fails = [{"constraint": r.constraint_id, "flight": r.flight_id, "resource": r.resource_id,
                  "why": r.explanation[:200]} for r in p.validation.failures][:6]
        out.append({"proposal_id": p.proposal_id, "version": p.version_no, "proposed_by": p.proposer.value, "turn": p.turn,
                    "valid": p.validation.passed, "failed_constraints": fails, "plan": _plan_summary(p.plan_json),
                    "metrics_plan": {k: round(v, 2) for k, v in p.metrics["plan"].items()},
                    "metrics_selected_flight": {k: (round(v, 2) if isinstance(v, float) else v)
                                                for k, v in p.metrics["flights"].get(ctx.scenario.simulated.flights[0].flight_id, {}).items()}})
    return out


def transcript_view(ctx: AgentContext) -> list[dict]:
    return [{"seq": m.seq, "turn": m.turn, "sender": m.sender.value, "type": m.type.value, "proposal_id": m.proposal_id,
             "referenced_proposal_id": m.referenced_proposal_id, "reason_codes": [r.value for r in m.reason_codes],
             "explanation": m.explanation, "conditions": [c.model_dump(mode="json", exclude_none=True) for c in m.conditions],
             "conditions_resolved": m.conditions_resolved, "outdated": m.outdated} for m in ctx.messages]


def build_prompt(ctx: AgentContext, domain_view: dict, tool_results: dict) -> str:
    body = {
        "scenario_version": ctx.scenario.scenario_version,
        "your_role": ctx.role.value,
        "your_turn": {"turn": ctx.turn, "of": 3, "message_seq": ctx.seq, "of_messages": 9,
                      "allowed_message_types": [t.value for t in ctx.allowed_types]},
        "historical_disruption_record": ctx.historical.model_dump(mode="json"),
        "your_domain": domain_view,
        "tool_results": tool_results,
        "other_agents_standing_conditions": {r.value: [c.model_dump(mode="json", exclude_none=True) for c in cs]
                                              for r, cs in ctx.standing.items() if r != ctx.role},
        "proposals_so_far": proposals_view(ctx),
        "messages_so_far": transcript_view(ctx),
    }
    if ctx.frozen is not None:
        body["frozen_candidate"] = {"proposal_id": ctx.frozen.proposal_id, "plan": _plan_summary(ctx.frozen.plan_json),
                                    "valid": ctx.frozen.validation.passed,
                                    "metrics_plan": {k: round(v, 2) for k, v in ctx.frozen.metrics["plan"].items()},
                                    "instruction": "Answer ACCEPT or REJECT with proposal_id = this id. No plan changes."}
    if ctx.repair_error:
        body["REPAIR_REQUIRED"] = ("Your previous reply was rejected: " + ctx.repair_error +
                                   " Reply again with ONE corrected JSON object (this is a format repair, not a new turn).")
    return json.dumps(body, indent=1, default=str)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def extract_json(text: str) -> str:
    t = _FENCE.sub("", text.strip())
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object found in the reply")
    return t[a:b + 1]


# ---------------------------------------------------------------- scripted engine
def _ok(conds: list[Condition], metrics: dict) -> bool:
    return all_satisfied(evaluate_conditions(conds, metrics))


def _metric_value(metrics: dict, metric: TradeoffMetric, flight: Optional[str]) -> float:
    if flight is not None and metric.value in metrics["flights"].get(flight, {}):
        return metrics["flights"][flight][metric.value]
    return metrics["plan"][metric.value]


def _dump(o: AgentOutput) -> str:
    return o.model_dump_json(exclude_none=True)


def _failed_text(results) -> str:
    bad = [r for r in results if r.satisfied is not True]
    return "; ".join(f"{r.condition.field.value}{('['+r.condition.flight_id+']') if r.condition.flight_id else ''} "
                     f"{r.condition.op.value} {r.condition.value} (observed {r.observed})" for r in bad[:3])


def scripted_negotiate(ctx: AgentContext, role: AgentRole, own: list[Condition], *, metrics: list[TradeoffMetric],
                       flight: Optional[str], reasons: list[ReasonCode], label: str,
                       fallback_spec: Callable[[], PlanSpec]) -> str:
    """Fixed policy: minimise this role's objective subject to the OTHER agents' stated conditions; accept an existing
    proposal when it is already the best such plan. Deterministic given the context."""
    sc = ctx.scenario
    metric = metrics[0]  # the role's primary objective; the declared trade-off uses the first metric that actually moved
    if ctx.turn == 3:
        fz = ctx.frozen
        res = evaluate_conditions(own, fz.metrics)
        if fz.validation.passed and all_satisfied(res):
            return _dump(AgentOutput(type=MessageType.ACCEPT, proposal_id=fz.proposal_id, conditions=own,
                                     reason_codes=[ReasonCode.ACCEPTABLE, ReasonCode.CONDITIONS_MET],
                                     explanation=f"{label}: frozen plan meets all my stated conditions."))
        why = "plan failed validation" if not fz.validation.passed else _failed_text(res)
        return _dump(AgentOutput(type=MessageType.REJECT, proposal_id=fz.proposal_id,
                                 reason_codes=[ReasonCode.VALIDATION_FAILED if not fz.validation.passed
                                               else ReasonCode.CONDITIONS_NOT_MET],
                                 explanation=f"{label}: frozen plan breaks my condition(s): {why}."[:400], conditions=own))

    pool = candidate_pool(sc)
    others = [c for r, cs in ctx.standing.items() if r != role for c in cs]
    latest: Optional[ProposalRecord] = ctx.proposals[-1] if ctx.proposals else None
    ids = {p.proposal_id for p in ctx.proposals}

    def pick() -> Optional[Candidate]:
        if ctx.turn == 1:
            return (best_for(role, sc, pool, where=lambda c: _ok(others, c.metrics)) if others else None) or best_for(role, sc, pool)
        return (best_for(role, sc, pool, where=lambda c: _ok(others + own, c.metrics))
                or best_for(role, sc, pool, where=lambda c: _ok(own, c.metrics)) or best_for(role, sc, pool))

    if latest is not None and ctx.turn == 1 and latest.validation.passed and _ok(own, latest.metrics):
        return _dump(AgentOutput(type=MessageType.ACCEPT, proposal_id=latest.proposal_id, conditions=own,
                                 reason_codes=[ReasonCode.ACCEPTABLE, ReasonCode.CONDITIONS_MET],
                                 explanation=f"{label}: {latest.proposal_id} is within my stated limits."))
    cand = pick()
    if cand is None:  # nothing in the finite set passes every hard constraint: propose the original assignments honestly
        spec = fallback_spec()
        if latest is None:
            return _dump(AgentOutput(type=MessageType.PROPOSE, plan_spec=spec, conditions=own, reason_codes=[ReasonCode.ACCEPTABLE],
                                     explanation=f"{label}: no option passes all constraints in my tools; proposing original assignments."))
        return _dump(AgentOutput(type=MessageType.COUNTER_PROPOSE, referenced_proposal_id=latest.proposal_id, plan_spec=spec,
                                 conditions=own, reason_codes=[ReasonCode.VALIDATION_FAILED],
                                 explanation=f"{label}: no option passes all constraints in my tools; re-proposing originals."))
    if cand.proposal_id in ids:
        return _dump(AgentOutput(type=MessageType.ACCEPT, proposal_id=cand.proposal_id, conditions=own,
                                 reason_codes=[ReasonCode.ACCEPTABLE, ReasonCode.CONDITIONS_MET],
                                 explanation=f"{label}: existing proposal {cand.proposal_id} is the best plan meeting everyone's stated conditions."))
    if latest is None:
        return _dump(AgentOutput(type=MessageType.PROPOSE, plan_spec=cand.spec, conditions=own, reason_codes=reasons,
                                 explanation=f"{label}: proposing the plan minimising my objective "
                                             f"({metric.value}={_metric_value(cand.metrics, metric, flight):.1f})."))
    moved = [m for m in metrics if abs(_metric_value(cand.metrics, m, flight) - _metric_value(latest.metrics, m, flight)) > 1e-6]
    metric = moved[0] if moved else metric
    before, after = _metric_value(latest.metrics, metric, flight), _metric_value(cand.metrics, metric, flight)
    return _dump(AgentOutput(
        type=MessageType.COUNTER_PROPOSE, referenced_proposal_id=latest.proposal_id, plan_spec=cand.spec, conditions=own,
        reason_codes=reasons, tradeoff=Tradeoff(metric=metric, flight_id=flight, before=before, after=after, because=reasons[0]),
        explanation=f"{label}: counter to {latest.proposal_id}; {metric.value} {before:.1f} -> {after:.1f} while keeping "
                    f"other agents' stated limits where possible."))
