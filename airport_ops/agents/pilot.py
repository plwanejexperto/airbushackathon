"""Pilot agent: the SELECTED historical flight only (competing flights are deterministic traffic, no pilot agent yet)."""
from __future__ import annotations

from ..fuel import fuel_account, latest_landing_min
from ..messages import AgentRole, Condition, ConditionField, ConditionOp, ReasonCode, TradeoffMetric
from ..negotiation_types import AgentContext
from ..tools import best_for, option_menu
from . import base

ROLE = AgentRole.PILOT

INSTRUCTIONS = """\
ROLE: PILOT agent for the SELECTED historical flight only (flights[0] in your domain). You do not speak for the two
competing flights; they are simulated traffic with fixed constraints.
Objective (soft): land the selected flight as close to its nominal ETA as possible and preserve its connecting passengers
(prefer gate arrival by the preferred time), while NEVER consuming the final fuel reserve or the contingency fuel.
Your fuel numbers are exact: any landing later than `latest_landing_min_protecting_contingency` is unsafe and will be
discarded by the validator - never propose or accept it. You do not own runways, gates or ground crews; ask for what you
need and state measurable conditions (e.g. landing_delay_min of the selected flight <= X)."""


def domain_view(ctx: AgentContext) -> dict:
    sc = ctx.scenario
    sf = sc.simulated.flights[0]
    fa = fuel_account(sc, sf, sf.earliest_landing_min)
    fcw = sc.workbook.fuel_crew
    crew = next(c for c in sc.simulated.flight_crews if c.id == sf.flight_crew_id)
    conn = sc.simulated.connection
    return {
        "selected_flight": {
            "flight_id": sf.flight_id, "aircraft_class": sf.aircraft_class.value, "wake_category": sf.wake_category.value,
            "passengers": sf.passengers, "priority": sf.priority,
            "eta_window": {"nominal_eta_min": sf.earliest_landing_min,
                           "earliest_landing_min": sf.earliest_landing_min,
                           "latest_landing_min_protecting_contingency": round(latest_landing_min(sc, sf), 3),
                           "latest_landing_min_protecting_final_reserve_only (never use)": round(latest_landing_min(sc, sf, protect_contingency=False), 3)},
            "fuel_at_nominal_eta": {k: round(v, 2) if isinstance(v, float) else v for k, v in fa.as_dict().items()},
            "fuel_rules": {"final_reserve_min": fcw.final_reserve_min, "contingency_fraction_of_trip": fcw.contingency_fraction,
                           "holding_burn_kg_per_h": fa.hold_burn_kg_h},
            "flight_crew": {"id": crew.id, "duty_limit_min": crew.report_time_min + fcw.flight_crew_max_duty_h * 60,
                            "postflight_min": sc.simulated.assumptions.flight_crew_postflight_min,
                            "rule": "gate_in_min + postflight must not exceed the duty limit"},
            "aircraft_restrictions": {"gate_class_needed": "any gate" if sf.aircraft_class.value == "narrowbody" else "widebody gate only",
                                      "wake_category": sf.wake_category.value},
            "connection_preference_SIMULATED_soft": conn.model_dump() if conn else None,
        },
        "other_traffic": [{"flight_id": f.flight_id, "wake_category": f.wake_category.value, "nominal_eta_min": f.earliest_landing_min}
                          for f in sc.simulated.flights[1:]],
    }


def tool_results(ctx: AgentContext) -> dict:
    return {"description": "valid options ranked by YOUR objective (selected flight delay, connection, fuel margin)",
            "options": option_menu(ROLE, ctx.scenario, 5)}


def system_prompt() -> str:
    return base.build_system(INSTRUCTIONS)


def standing_conditions(ctx: AgentContext) -> list[Condition]:
    best = best_for(ROLE, ctx.scenario)
    if best is None:
        return []
    sel = ctx.scenario.simulated.flights[0].flight_id
    m = best.metrics["flights"][sel]
    return [Condition(field=ConditionField.LANDING_DELAY_MIN, op=ConditionOp.LE, flight_id=sel, value=round(m["landing_delay_min"] + 3, 2)),
            Condition(field=ConditionField.CONNECTION_MISS_MIN, op=ConditionOp.LE, flight_id=sel, value=round(m["connection_miss_min"] + 5, 2)),
            Condition(field=ConditionField.CONTINGENCY_MARGIN_KG, op=ConditionOp.GE, flight_id=sel, value=0.0)]


def scripted_policy(ctx: AgentContext, fallback_spec) -> str:
    sel = ctx.scenario.simulated.flights[0].flight_id
    return base.scripted_negotiate(ctx, ROLE, standing_conditions(ctx), metrics=[TradeoffMetric.LANDING_DELAY_MIN, TradeoffMetric.CONNECTION_MISS_MIN, TradeoffMetric.GATE_IN_MIN], flight=sel,
                                   reasons=[ReasonCode.DELAY_MINIMISATION, ReasonCode.CONNECTION_PRESERVATION], label="PILOT",
                                   fallback_spec=fallback_spec)
