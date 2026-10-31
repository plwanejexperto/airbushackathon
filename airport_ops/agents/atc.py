"""ATC agent: runway sequencing, wake separation, taxiways, movement capacity."""
from __future__ import annotations

from ..capacity import airport_limit, per_runway_limit
from ..messages import AgentRole, Condition, ConditionField, ConditionOp, PlanSpec, ReasonCode, TradeoffMetric
from ..negotiation_types import AgentContext
from ..tools import best_for, option_menu
from . import base

ROLE = AgentRole.ATC

INSTRUCTIONS = """\
ROLE: AIR TRAFFIC CONTROL (ATC) agent.
You control runway use, landing sequence, wake-turbulence separation, taxi routing and airport movement capacity.
Objective (soft): keep the whole traffic flow moving - minimise priority-weighted landing delay across all three flights,
then avoid unnecessary runway changes. You do NOT own gates, crews or aircraft fuel: ask the other agents for those.
You speak FIRST in each turn. In turn 1 you open with a PROPOSE built from your tool results. Use ONLY the
runway/taxi/separation/capacity facts in your_domain; wake rules are directional (leader -> follower).
Prefer options from tool_results.options (all already pass every hard constraint); explain trade-offs with numbers."""


def domain_view(ctx: AgentContext) -> dict:
    sc = ctx.scenario
    s, a, w = sc.simulated, sc.simulated.assumptions, sc.workbook
    sub = w.capacity_subset
    return {
        "flights": [{"flight_id": f.flight_id, "wake_category": f.wake_category.value, "nominal_eta_min": f.earliest_landing_min,
                     "priority": f.priority, "original_runway": f.original_runway_id} for f in s.flights],
        "runways": [{"id": r.id, "closures": [i.model_dump() for i in r.closures]} for r in s.runways],
        "taxi_network_inbound": [{"segment": x.id, "from": x.from_node, "to": x.to_node, "minutes": x.minutes,
                                  "blockages": [i.model_dump() for i in x.blockages]} for x in s.taxi_segments],
        "separation": {"approach_speed_kt": a.approach_speed_kt, "runway_occupancy_min": a.runway_occupancy_min,
                       "workbook_time_floor_applied": a.wake_apply_time_floor,
                       "wake_rules_ordered": [{"leader": r.leader.value, "follower": r.follower.value, "nm": r.min_separation_nm,
                                               "time_s": r.min_time_s} for r in w.wake.rules],
                       "any_other_pair": {"nm": w.wake.default_separation_nm, "time_s": w.wake.default_time_s}},
        "capacity": {"airport_wide_ops_per_hour (WORKBOOK)": airport_limit(sc),
                     "per_runway_share (ASSUMPTION, configurable)": {"mode": a.per_runway_share_mode, "ops_per_hour": per_runway_limit(sc)},
                     "airport_total_runways": sub.total_runways, "modelled_runways": sub.modelled_runways,
                     "note": "limits apply to the modelled movements only; no background traffic is simulated"},
    }


def tool_results(ctx: AgentContext) -> dict:
    return {"description": "valid options ranked by YOUR objective (ATC); each already passes every hard constraint",
            "options": option_menu(ROLE, ctx.scenario, 5)}


def system_prompt() -> str:
    return base.build_system(INSTRUCTIONS)


def standing_conditions(ctx: AgentContext) -> list[Condition]:
    best = best_for(ROLE, ctx.scenario)
    if best is None:
        return []
    w = best.metrics["plan"]["weighted_landing_delay"]
    return [Condition(field=ConditionField.WEIGHTED_LANDING_DELAY, op=ConditionOp.LE, value=round(w + max(3.0, 0.5 * w), 2))]


def scripted_policy(ctx: AgentContext, fallback_spec) -> str:
    return base.scripted_negotiate(ctx, ROLE, standing_conditions(ctx), metrics=[TradeoffMetric.WEIGHTED_LANDING_DELAY, TradeoffMetric.TOTAL_COST], flight=None,
                                   reasons=[ReasonCode.DELAY_MINIMISATION, ReasonCode.WAKE_SEPARATION], label="ATC",
                                   fallback_spec=fallback_spec)
