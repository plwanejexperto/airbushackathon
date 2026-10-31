"""Shared pieces for the three agents: protocol text, prompt assembly, JSON extraction and the scripted
negotiation engine (used only by ScriptedClient; live agents never run it)."""
from __future__ import annotations

import json
import re
from functools import partial
from typing import Callable, Optional

from ..messages import (FLIGHT_ID_FIELDS, FLIGHT_NUMERIC, PLAN_NUMERIC, AgentOutput, AgentRole, Condition, ConditionField,
                        ConditionOp, MessageType, PlanSpec,
                        ReasonCode, SPEAKING_ORDER, Tradeoff, TradeoffMetric)
from ..canonical import canonical_metrics, policy_text, quantise
from ..metrics import all_satisfied, evaluate_conditions
from ..workbook import FlightStatus
from ..negotiation_types import AgentContext, ProposalRecord
from ..reports import departure_readiness, pilot_report
from ..tools import Candidate, best_for, candidate_pool

CONDITION_FIELDS_DOC = {   # derived from the schema sets (no index slicing): exactly what Condition accepts
    "per flight, numeric (flight_id required)": [f.value for f in ConditionField if f in FLIGHT_NUMERIC],
    "per flight, identifier (op == or !=, string value)": [f.value for f in ConditionField if f in FLIGHT_ID_FIELDS],
    "whole plan, numeric (omit flight_id)": [f.value for f in ConditionField if f in PLAN_NUMERIC],
}

COMMON_PROTOCOL = f"""\
You are one of three INDEPENDENT agents negotiating a recovery plan for one disrupted flight (the 'selected flight')
and two competing flights, at one airport. The agents are ATC, PILOT and GROUND. The PILOT represents ONLY the selected
historical flight. The two competing flights are simulated traffic with fixed, deterministic constraints and have no agent.
You see only your own domain plus the structured messages the other agents have sent.
The historical disruption record (a real flights.csv row) is CONTEXT ONLY: its delay minutes are not a remaining holding time and
are not something to "recover"; all negotiated times are simulated minutes from scenario start. `initial_reference_plan` is the
simulated baseline (nominal ETAs, original assignments); revisions are measured against it.

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
     They are evaluated mechanically against the EXACT proposal using the SAME per-field precision as the numbers shown to you
     ({policy_text()}); an ACCEPT whose conditions are not all satisfied is NOT a final acceptance. Before you ACCEPT, check each
     condition you attach against that proposal's numbers in proposals_so_far, and drop or loosen any it does not meet.
     Allowed fields: {json.dumps(CONDITION_FIELDS_DOC)}; ops: {[o.value for o in ConditionOp]}.
  tradeoff: optional {{"metric","flight_id"?,"before","after","because"}} (`because` MUST be exactly ONE reason code from
     the list above) - the measurable change behind a revised
     position (metrics: {[m.value for m in TradeoffMetric]}). The coordinator recomputes the real numbers.
"""


def build_system(instructions: str) -> str:
    return instructions.strip() + "\n\n" + COMMON_PROTOCOL


def _plan_summary(plan_json: str) -> list[dict]:
    d = json.loads(plan_json)
    q = quantise
    out = []
    for f in d["flights"]:
        row = {"flight": f["flight_id"], "runway": f["runway_id"], "landing_min": q("landing_min", f["landing_min"]),
               "gate": f["gate_id"], "gate_in_min": q("gate_in_min", f["gate_in_min"]),
               "service_start_min": q("service_start_min", f["service_start_min"]),
               "service_end_min": q("service_end_min", f["service_end_min"]), "ground_crew": f["ground_crew_id"]}
        if f.get("hold_area_id"):
            row["ground_wait_area"] = f["hold_area_id"]
        if f.get("takeoff_min") is not None:
            row.update({"ready_min": q("ready_min", f["ready_min"]), "dep_runway": f["dep_runway_id"], "takeoff_min": q("takeoff_min", f["takeoff_min"]),
                        "equipment": sorted({t["equipment_id"] for t in f.get("tasks", []) if t.get("equipment_id")})})
        out.append(row)
    return out


def proposals_view(ctx: AgentContext) -> list[dict]:
    """Registry as evidence: validity, failed constraints (structured) and consequences of every proposal so far."""
    out = []
    for p in ctx.proposals:
        fails = [{"constraint": r.constraint_id, "flight": r.flight_id, "resource": r.resource_id,
                  "why": r.explanation[:200]} for r in p.validation.failures][:6]
        out.append({"proposal_id": p.proposal_id, "version": p.version_no, "proposed_by": p.proposer.value, "turn": p.turn,
                    "valid": p.validation.passed, "failed_constraints": fails, "plan": _plan_summary(p.plan_json),
                    "metrics_plan": canonical_metrics(p.metrics)["plan"],
                    "metrics_selected_flight": canonical_metrics(p.metrics)["flights"].get(ctx.scenario.simulated.flights[0].flight_id, {})})
    return out


def transcript_view(ctx: AgentContext) -> list[dict]:
    return [{"seq": m.seq, "turn": m.turn, "sender": m.sender.value, "type": m.type.value, "proposal_id": m.proposal_id,
             "referenced_proposal_id": m.referenced_proposal_id, "reason_codes": [r.value for r in m.reason_codes],
             "explanation": m.explanation, "conditions": [c.model_dump(mode="json", exclude_none=True) for c in m.conditions],
             "conditions_resolved": m.conditions_resolved, "outdated": m.outdated} for m in ctx.messages]


PLAN_SPEC_FORMATS = {
    "arrival": {"landing_order": "[3 flight ids, first to land first]", "runways": "{flight: runway}", "gates": "{flight: gate or remote stand}",
                "ground_crews": "{flight: ground crew}"},
    "diversion": {"note": "the plan is the aircraft's ARRIVAL at the ALTERNATE airport (same fields as arrival)",
                  "landing_order": "[3 flight ids at the alternate]", "runways": "{flight: runway}", "gates": "{flight: gate or remote stand}",
                  "ground_crews": "{flight: ground crew}", "declared_status": "NORMAL | MINIMUM_FUEL | EMERGENCY - must equal the status computed from the fuel numbers"},
    "turnaround_departure": {"note": "arrival part (landing, runway, gate) is FIXED - do not send those fields",
                             "ground_crews": "{flight: ground crew}", "equipment_order": "[3 flight ids; priority for the shared fuel truck / tractors]",
                             "overlap": "'max' (parallel where dependencies allow) or 'sequential'", "dep_runways": "{flight: departure runway}"},
}


def build_prompt(ctx: AgentContext, domain_view: dict, tool_results: dict) -> str:
    kind = ("turnaround_departure" if ctx.scenario.simulated.departure is not None else
            "diversion" if ctx.scenario.simulated.diversion is not None else "arrival")
    body = {
        "session_kind": kind, "plan_spec_format": PLAN_SPEC_FORMATS[kind],
        "scenario_version": ctx.scenario.scenario_version,
        "your_role": ctx.role.value,
        "your_agent": {"agent_id": ctx.who.agent_id, "role": ctx.role.value, "airport": ctx.who.airport, "flight_id": ctx.who.flight_id,
                       "all_agents_in_this_session": [p.agent_id for p in ctx.participants] or [ctx.who.agent_id]},
        "your_turn": {"turn": ctx.turn, "of": 3, "message_seq": ctx.seq, "of_messages": 9,
                      "allowed_message_types": [t.value for t in ctx.allowed_types]},
        "historical_disruption_record": ctx.historical.model_dump(mode="json"),
        "initial_reference_plan": ctx.reference,
        "your_domain": domain_view,
        "tool_results": tool_results,
        "other_agents_standing_conditions": {str(getattr(r, "value", r)): [c.model_dump(mode="json", exclude_none=True) for c in cs]
                                              for r, cs in ctx.standing.items() if str(getattr(r, "value", r)) != ctx.who.agent_id},
        "proposals_so_far": proposals_view(ctx),
        "messages_so_far": transcript_view(ctx),
    }
    if ctx.frozen is not None:
        body["frozen_candidate"] = {"proposal_id": ctx.frozen.proposal_id, "plan": _plan_summary(ctx.frozen.plan_json),
                                    "valid": ctx.frozen.validation.passed,
                                    "metrics_plan": canonical_metrics(ctx.frozen.metrics)["plan"],
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


_REASON_FOR_FIELD = {
    ConditionField.CONTINGENCY_MARGIN_KG: ReasonCode.FUEL_LIMIT, ConditionField.RESERVE_MARGIN_KG: ReasonCode.FUEL_LIMIT,
    ConditionField.CONNECTION_MISS_MIN: ReasonCode.CONNECTION_PRESERVATION, ConditionField.LANDING_DELAY_MIN: ReasonCode.DELAY_MINIMISATION,
    ConditionField.WEIGHTED_LANDING_DELAY: ReasonCode.DELAY_MINIMISATION, ConditionField.GROUND_OVERTIME_MIN: ReasonCode.OVERTIME,
    ConditionField.TOTAL_SERVICE_WAIT_MIN: ReasonCode.CREW_AVAILABILITY, ConditionField.DEPARTURE_DELAY_MIN: ReasonCode.DELAY_MINIMISATION,
    ConditionField.WEIGHTED_DEPARTURE_DELAY: ReasonCode.DELAY_MINIMISATION, ConditionField.TOTAL_DEPARTURE_DELAY_MIN: ReasonCode.DELAY_MINIMISATION,
    ConditionField.READY_MIN: ReasonCode.TURNAROUND_TIME, ConditionField.TURNAROUND_MIN: ReasonCode.TURNAROUND_TIME,
    ConditionField.GATE_FUEL_MARGIN_KG: ReasonCode.FUEL_LIMIT,
}


def _violation_reasons(results) -> list[ReasonCode]:
    """Reason codes for the agent's OWN conditions that the countered plan breaks (evidence-driven, deterministic order)."""
    out: list[ReasonCode] = []
    for r in results:
        code = _REASON_FOR_FIELD.get(r.condition.field)
        if r.satisfied is not True and code and code not in out:
            out.append(code)
    return out


_METRIC_FOR_FIELD = {
    ConditionField.LANDING_DELAY_MIN: TradeoffMetric.LANDING_DELAY_MIN, ConditionField.CONNECTION_MISS_MIN: TradeoffMetric.CONNECTION_MISS_MIN,
    ConditionField.WEIGHTED_LANDING_DELAY: TradeoffMetric.WEIGHTED_LANDING_DELAY, ConditionField.GROUND_OVERTIME_MIN: TradeoffMetric.GROUND_OVERTIME_MIN,
    ConditionField.TOTAL_SERVICE_WAIT_MIN: TradeoffMetric.TOTAL_SERVICE_WAIT_MIN, ConditionField.CONTINGENCY_MARGIN_KG: TradeoffMetric.CONTINGENCY_MARGIN_KG,
    ConditionField.GATE_CHANGES: TradeoffMetric.GATE_CHANGES, ConditionField.TOTAL_COST: TradeoffMetric.TOTAL_COST,
    ConditionField.GATE_IN_MIN: TradeoffMetric.GATE_IN_MIN, ConditionField.DEPARTURE_DELAY_MIN: TradeoffMetric.DEPARTURE_DELAY_MIN,
    ConditionField.WEIGHTED_DEPARTURE_DELAY: TradeoffMetric.WEIGHTED_DEPARTURE_DELAY, ConditionField.TOTAL_DEPARTURE_DELAY_MIN: TradeoffMetric.TOTAL_DEPARTURE_DELAY_MIN,
    ConditionField.READY_MIN: TradeoffMetric.READY_MIN, ConditionField.TURNAROUND_MIN: TradeoffMetric.TURNAROUND_MIN,
}


def scripted_negotiate(ctx: AgentContext, role: AgentRole, own: list[Condition], *, metrics: list[TradeoffMetric],
                       flight: Optional[str], reasons: list[ReasonCode], label: str,
                       fallback_spec: Callable[[], PlanSpec]) -> str:
    """Fixed policy: minimise this role's objective subject to the OTHER agents' stated conditions; accept an existing
    proposal when it is already the best such plan. Deterministic given the context."""
    sc = ctx.scenario
    first_msg = not any(m.agent_id == ctx.who.agent_id for m in ctx.messages)
    is_dep = sc.simulated.departure is not None
    rep = pilot_report(sc, flight) if (role == AgentRole.PILOT and flight and first_msg and not is_dep) else None   # arrival window / fuel report
    ready = departure_readiness(sc, flight) if (role == AgentRole.PILOT and flight and first_msg and is_dep) else None   # departure readiness
    dv = sc.simulated.diversion
    declared = dv.computed_status if (dv is not None and role == AgentRole.PILOT) else None   # the Pilot DECLARES the computed status
    AO = partial(AgentOutput, report=rep, readiness=ready, declared_status=declared)
    metric = metrics[0]  # the role's primary objective; the declared trade-off uses the first metric that actually moved
    last_own = next((m for m in reversed(ctx.messages) if m.agent_id == ctx.who.agent_id), None)
    own_now = list(last_own.conditions) if last_own is not None else own      # limits as LAST stated (a concession may have relaxed them)
    if ctx.turn == 3:
        fz = ctx.frozen
        own = own_now
        res = evaluate_conditions(own, fz.metrics)
        if fz.validation.passed and all_satisfied(res):
            return _dump(AO(type=MessageType.ACCEPT, proposal_id=fz.proposal_id, conditions=own,
                                     reason_codes=[ReasonCode.ACCEPTABLE, ReasonCode.CONDITIONS_MET],
                                     explanation=f"{label}: frozen plan meets all my stated conditions."))
        why = "plan failed validation" if not fz.validation.passed else _failed_text(res)
        return _dump(AO(type=MessageType.REJECT, proposal_id=fz.proposal_id,
                                 reason_codes=[ReasonCode.VALIDATION_FAILED if not fz.validation.passed
                                               else ReasonCode.CONDITIONS_NOT_MET],
                                 explanation=f"{label}: frozen plan breaks my condition(s): {why}."[:400], conditions=own))

    pool = candidate_pool(sc)
    me = ctx.who.agent_id
    others = [c for r, cs in ctx.standing.items() if str(getattr(r, "value", r)) != me for c in cs]
    latest: Optional[ProposalRecord] = ctx.proposals[-1] if ctx.proposals else None
    ids = {p.proposal_id for p in ctx.proposals}

    others_by = {str(getattr(r, "value", r)): cs for r, cs in ctx.standing.items() if str(getattr(r, "value", r)) != me}
    state = {"conditions": own, "conceded": []}

    def pick() -> Optional[Candidate]:
        if ctx.turn == 1:
            return (best_for(role, sc, pool, where=lambda c: _ok(others, c.metrics), flight=flight) if others else None) or best_for(role, sc, pool, flight=flight)
        agreed = best_for(role, sc, pool, where=lambda c: _ok(others + own, c.metrics), flight=flight)
        if agreed is not None or not pool:
            return agreed
        # No plan satisfies EVERY stated limit: CONCEDE. Settle on the plan that satisfies the most agents' limits (ties: an existing
        # proposal, then lowest cost - a rule every agent shares so they converge) and drop only the own limits that plan breaks.
        sup = {c.proposal_id: int(_ok(own, c.metrics)) + sum(int(_ok(cs, c.metrics)) for cs in others_by.values()) for c in pool}
        top = max(sup.values())
        cand = min((c for c in pool if sup[c.proposal_id] == top), key=lambda c: (c.proposal_id not in ids, c.metrics["plan"]["total_cost"], c.proposal_id))
        res = evaluate_conditions(own, cand.metrics)
        state["conditions"] = [r.condition for r in res if r.satisfied is True]
        state["conceded"] = [r for r in res if r.satisfied is not True]
        state["support"] = (top, len(others_by) + 1)
        return cand

    def concession_fields():
        """(reason codes, Tradeoff, explanation) describing exactly which limit was dropped and by how much."""
        r = state["conceded"][0]
        cd = r.condition
        metric_ = _METRIC_FOR_FIELD.get(cd.field, TradeoffMetric.TOTAL_COST)
        before_ = float(cd.value)
        after_ = float(r.observed) if r.observed is not None else before_
        txt = "; ".join(f"{x.condition.field.value}{('['+x.condition.flight_id+']') if x.condition.flight_id else ''} "
                        f"{x.condition.op.value} {x.condition.value} (plan: {x.observed})" for x in state["conceded"])
        codes_ = _violation_reasons(state["conceded"]) or [ReasonCode.COST_TRADEOFF]
        return ([ReasonCode.COST_TRADEOFF] + [c for c in codes_ if c != ReasonCode.COST_TRADEOFF],
                Tradeoff(metric=metric_, flight_id=cd.flight_id if metric_.value in ("landing_delay_min", "connection_miss_min", "contingency_margin_kg", "gate_in_min") else None,
                         before=before_, after=after_, because=ReasonCode.COST_TRADEOFF), txt)

    if latest is not None and ctx.turn == 1 and latest.validation.passed and _ok(own, latest.metrics):
        return _dump(AO(type=MessageType.ACCEPT, proposal_id=latest.proposal_id, conditions=own,
                                 reason_codes=[ReasonCode.ACCEPTABLE, ReasonCode.CONDITIONS_MET],
                                 explanation=f"{label}: {latest.proposal_id} is within my stated limits."))
    cand = pick()
    if cand is None:  # nothing in the finite set passes every hard constraint: propose the original assignments honestly
        spec = fallback_spec()
        if latest is None:
            return _dump(AO(type=MessageType.PROPOSE, plan_spec=spec, conditions=state["conditions"], reason_codes=[ReasonCode.ACCEPTABLE],
                                     explanation=f"{label}: no option passes all constraints in my tools; proposing original assignments."))
        return _dump(AO(type=MessageType.COUNTER_PROPOSE, referenced_proposal_id=latest.proposal_id, plan_spec=spec,
                                 conditions=state["conditions"], reason_codes=[ReasonCode.VALIDATION_FAILED],
                                 explanation=f"{label}: no option passes all constraints in my tools; re-proposing originals."))
    if state["conceded"] and cand.proposal_id in ids:
        codes_, tr_, txt_ = concession_fields()
        return _dump(AO(type=MessageType.ACCEPT, proposal_id=cand.proposal_id, conditions=state["conditions"], reason_codes=codes_ + [ReasonCode.ACCEPTABLE],
                        tradeoff=tr_, explanation=(f"{label}: CONCESSION - no plan meets every agent's stated limits; I drop {txt_} to settle on "
                                                   f"{cand.proposal_id}, which satisfies {state['support'][0]} of {state['support'][1]} agents' limits.")[:600]))
    if cand.proposal_id in ids:
        return _dump(AO(type=MessageType.ACCEPT, proposal_id=cand.proposal_id, conditions=state["conditions"],
                                 reason_codes=[ReasonCode.ACCEPTABLE, ReasonCode.CONDITIONS_MET],
                                 explanation=f"{label}: existing proposal {cand.proposal_id} is the best plan meeting everyone's stated conditions."))
    if latest is None:
        return _dump(AO(type=MessageType.PROPOSE, plan_spec=cand.spec, conditions=state["conditions"], reason_codes=reasons,
                                 explanation=f"{label}: proposing the plan minimising my objective "
                                             f"({metric.value}={_metric_value(cand.metrics, metric, flight):.1f})."))
    if state["conceded"]:
        codes_, tr_, txt_ = concession_fields()
        return _dump(AO(type=MessageType.COUNTER_PROPOSE, referenced_proposal_id=latest.proposal_id, plan_spec=cand.spec, conditions=state["conditions"],
                        reason_codes=codes_, tradeoff=tr_, explanation=(f"{label}: CONCESSION - no plan meets every agent's stated limits; I drop {txt_} and "
                                                                       f"propose {cand.proposal_id}, which satisfies {state['support'][0]} of {state['support'][1]} agents' limits.")[:600]))
    moved = [m for m in metrics if abs(_metric_value(cand.metrics, m, flight) - _metric_value(latest.metrics, m, flight)) > 1e-6]
    metric = moved[0] if moved else metric
    before, after = _metric_value(latest.metrics, metric, flight), _metric_value(cand.metrics, metric, flight)
    broken = evaluate_conditions(own, latest.metrics)
    why = _violation_reasons(broken)
    codes = (why + [r for r in reasons if r not in why]) if why else (
        [ReasonCode.VALIDATION_FAILED] + reasons if not latest.validation.passed else reasons)
    cause = (f" {latest.proposal_id} breaks my limit(s): {_failed_text(broken)}." if why else
             f" {latest.proposal_id} fails validation." if not latest.validation.passed else "")
    return _dump(AO(
        type=MessageType.COUNTER_PROPOSE, referenced_proposal_id=latest.proposal_id, plan_spec=cand.spec, conditions=state["conditions"],
        reason_codes=codes, tradeoff=Tradeoff(metric=metric, flight_id=flight, before=before, after=after, because=codes[0]),
        explanation=(f"{label}: counter to {latest.proposal_id}; {metric.value} {before:.1f} -> {after:.1f}.{cause} "
                     f"Keeping other agents' stated limits where possible.")[:600]))


def is_dest_agent(ctx: AgentContext) -> bool:
    """In a diversion session the ATC/Ground of the ORIGINAL destination confirm the destination cannot take the aircraft."""
    dv = ctx.scenario.simulated.diversion
    return dv is not None and ctx.who.airport == dv.destination.airport


def scripted_destination(ctx: AgentContext, label: str, fallback_spec) -> str:
    """Original-destination agents: confirm (or contest) that the destination cannot accept the aircraft. They do not shape the alternate's plan."""
    sc = ctx.scenario
    dv = sc.simulated.diversion
    d = dv.destination
    target = ctx.frozen if ctx.turn == 3 else (ctx.proposals[-1] if ctx.proposals else None)
    if target is None:
        return _dump(AgentOutput(type=MessageType.PROPOSE, plan_spec=fallback_spec(), reason_codes=[ReasonCode.ACCEPTABLE],
                                 explanation=f"{label}: no proposal yet; proposing the original assignments at the alternate."))
    justified = (not d.feasible) or dv.computed_status != FlightStatus.NORMAL or dv.emergency_declared_input
    ok = justified and target.validation.passed
    if ok:
        return _dump(AgentOutput(type=MessageType.ACCEPT, proposal_id=target.proposal_id, reason_codes=[ReasonCode.ACCEPTABLE],
                                 explanation=f"{label}: {d.airport} cannot accept {dv.diverting_flight_id}: earliest feasible landing t+{d.earliest_feasible_landing_min:.1f} "
                                             f"vs latest by fuel t+{d.latest_landing_min_by_fuel:.1f}. Releasing to {dv.alternate_airport}; "
                                             f"declared status must be {dv.computed_status.value}."))
    why = ("the plan fails validation" if not target.validation.passed else
           f"{d.airport} can still accept the aircraft (t+{d.earliest_feasible_landing_min:.1f} <= t+{d.latest_landing_min_by_fuel:.1f})")
    return _dump(AgentOutput(type=MessageType.REJECT, proposal_id=target.proposal_id,
                             reason_codes=[ReasonCode.VALIDATION_FAILED if not target.validation.passed else ReasonCode.UNACCEPTABLE],
                             explanation=f"{label}: cannot release the aircraft: {why}."[:400]))
