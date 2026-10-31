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
In TURNAROUND + DEPARTURE sessions the arrival part is fixed; you offer departure runways, take-off slots (respecting the workbook's same-runway
DEPARTURE spacing, closures, arrivals and capacity) and outbound taxi routes.
In a DIVERSION session your instances are: ATC of the ALTERNATE (offer slots/runways/taxi routes for the diverting aircraft; if it is in EMERGENCY it
lands at the earliest feasible slot and no convenience may delay it) and ATC of the original DESTINATION (confirm the destination cannot accept it).
Build your proposals from your tool results. Use ONLY the
runway/taxi/separation/capacity facts in your_domain; wake rules are directional (leader -> follower).
Prefer options from tool_results.options (all already pass every hard constraint); explain trade-offs with numbers."""


def _departure_view(ctx: AgentContext) -> dict:
    sc = ctx.scenario
    s, dep, w = sc.simulated, sc.simulated.departure, sc.workbook
    return {
        "session": "TURNAROUND + DEPARTURE: you offer departure runways, slots and outbound taxi routes",
        "fixed_arrivals (agreed, not negotiable)": [{"flight": x.flight_id, "runway": x.runway_id, "landing_min": x.landing_min} for x in dep.arrival],
        "flights": [{"flight_id": f.flight_id, "wake_category": f.wake_category.value, "priority": f.priority,
                     "next_flight_scheduled_departure_min": dep.scheduled_departure_min[f.flight_id]} for f in s.flights],
        "departure_runways": [{"id": r.id, "closures": [i.model_dump() for i in r.closures]} for r in s.runways],
        "outbound_taxi_network": [{"segment": x.id, "from": x.from_node, "to": x.to_node, "minutes": x.minutes, "blockages": [i.model_dump() for i in x.blockages]}
                                  for x in s.taxi_segments if x.id.endswith("-OUT")],
        "departure_spacing (WORKBOOK min_time_s, same runway, leader -> follower)": [{"leader": r.leader.value, "follower": r.follower.value, "seconds": r.min_time_s} for r in w.wake.rules]
                                                                                  + [{"any_other_pair_seconds": w.wake.default_time_s}],
        "assumptions": {"departure_roll_min": dep.departure_roll_min, "arrival_departure_gap_min": dep.arrival_departure_gap_min, "pushback_min": dep.turnaround.pushback_min},
        "capacity": {"airport_wide_ops_per_hour (WORKBOOK)": airport_limit(sc), "counts": "arrivals + departures of the modelled flights"},
    }


def _destination_view(ctx: AgentContext) -> dict:
    dv = ctx.scenario.simulated.diversion
    d = dv.destination
    return {"role_in_this_session": f"ATC of the ORIGINAL DESTINATION {d.airport}: confirm whether you can still accept the diverting aircraft; you do NOT control the alternate's plan",
            "diverting_aircraft": {"flight_id": dv.diverting_flight_id, "computed_status": dv.computed_status.value},
            "destination_status": d.model_dump(mode="json"), "runway_closures (SIMULATED weather)": [i.model_dump() for i in d.closures],
            "capacity_limits (WORKBOOK)": d.limits}


def domain_view(ctx: AgentContext) -> dict:
    if base.is_dest_agent(ctx):
        return _destination_view(ctx)
    if ctx.scenario.simulated.departure is not None:
        return _departure_view(ctx)
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
        "safe_ground_waiting_areas (SIMULATED)": [h.model_dump() for h in s.hold_areas],
        "weather_note": "runway closures here are SIMULATED weather; the historical weather delay is context only",
        "capacity": {"airport_wide_ops_per_hour (WORKBOOK)": airport_limit(sc),
                     "per_runway_share (ASSUMPTION, configurable)": {"mode": a.per_runway_share_mode, "ops_per_hour": per_runway_limit(sc)},
                     "airport_total_runways": sub.total_runways, "modelled_runways": sub.modelled_runways,
                     "note": "limits apply to the modelled movements only; no background traffic is simulated"},
    }


def tool_results(ctx: AgentContext) -> dict:
    if base.is_dest_agent(ctx):
        return {"description": "not applicable: you confirm the destination status, you do not build the alternate's plan", "options": []}
    return {"description": "valid options ranked by YOUR objective (ATC); each already passes every hard constraint",
            "options": option_menu(ROLE, ctx.scenario, 5)}


def system_prompt() -> str:
    return base.build_system(INSTRUCTIONS)


def standing_conditions(ctx: AgentContext) -> list[Condition]:
    best = best_for(ROLE, ctx.scenario)
    if best is None:
        return []
    if ctx.scenario.simulated.departure is not None:
        w = best.metrics["plan"]["weighted_departure_delay"]
        return [Condition(field=ConditionField.WEIGHTED_DEPARTURE_DELAY, op=ConditionOp.LE, value=round(w + max(3.0, 0.5 * w), 2))]
    w = best.metrics["plan"]["weighted_landing_delay"]
    return [Condition(field=ConditionField.WEIGHTED_LANDING_DELAY, op=ConditionOp.LE, value=round(w + max(3.0, 0.5 * w), 2))]


def scripted_policy(ctx: AgentContext, fallback_spec) -> str:
    if base.is_dest_agent(ctx):
        return base.scripted_destination(ctx, ctx.who.agent_id, fallback_spec)
    if ctx.scenario.simulated.departure is not None:
        return base.scripted_negotiate(ctx, ROLE, standing_conditions(ctx), metrics=[TradeoffMetric.WEIGHTED_DEPARTURE_DELAY, TradeoffMetric.TOTAL_DEPARTURE_DELAY_MIN, TradeoffMetric.TOTAL_COST],
                                       flight=None, reasons=[ReasonCode.DELAY_MINIMISATION, ReasonCode.WAKE_SEPARATION], label="ATC", fallback_spec=fallback_spec)
    return base.scripted_negotiate(ctx, ROLE, standing_conditions(ctx), metrics=[TradeoffMetric.WEIGHTED_LANDING_DELAY, TradeoffMetric.TOTAL_COST], flight=None,
                                   reasons=[ReasonCode.DELAY_MINIMISATION, ReasonCode.WAKE_SEPARATION], label="ATC",
                                   fallback_spec=fallback_spec)
