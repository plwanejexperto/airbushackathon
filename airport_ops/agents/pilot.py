"""Pilot agent: ONE PER AIRCRAFT. Each Pilot sees only its own aircraft (fuel, simulated position, status, weather, connecting
passengers) plus the other traffic's identity, and negotiates for that aircraft alone."""
from __future__ import annotations

from ..canonical import DECIMALS, quantise
from ..fuel import fuel_account, latest_landing_min
from ..messages import AgentRole, Condition, ConditionField, ConditionOp, ReasonCode, TradeoffMetric
from ..negotiation_types import AgentContext
from ..reports import pilot_report
from ..tools import best_for, option_menu
from . import base

ROLE = AgentRole.PILOT
NM_PER_MILE = 0.868976

INSTRUCTIONS = """\
ROLE: PILOT agent for ONE aircraft (your_agent.flight_id). You speak for that aircraft only; other aircraft have their own Pilot
agents or are simulated traffic with fixed constraints.
Objective (soft): land your aircraft as close to its nominal ETA as possible, avoid unnecessary holding and turbulence, and
preserve your connecting passengers (prefer gate arrival by the preferred time) - while NEVER consuming the final fuel reserve or the
contingency fuel (unless a MINIMUM_FUEL / EMERGENCY status applies to your flight as stated in your domain view).
In a DIVERSION session include `declared_status` (and in plan_spec) equal to the COMPUTED_STATUS in your domain view: a diversion is NOT automatically an emergency;
MINIMUM_FUEL means the contingency fuel will be used (no further delay acceptable); EMERGENCY means the final reserve cannot be protected: land at the earliest
feasible slot - operational convenience must never delay safety action.
Your fuel numbers are exact: any landing later than `latest_landing_min_protecting_contingency` is unsafe and will be discarded by the
validator - never propose or accept it. You may REJECT any proposal that violates aircraft safety constraints.
In your FIRST message include `report` copied from `report_template` (arrival sessions) or `readiness` copied from `readiness_template` (turnaround/departure sessions) from your domain view (arrival window, fuel margin and maximum
acceptable holding time); the coordinator recomputes it and records any discrepancy. You do not own runways, gates or ground crews:
ask for what you need and state measurable conditions (e.g. landing_delay_min of your flight <= X)."""


def _flight(ctx: AgentContext) -> str:
    return ctx.who.flight_id or ctx.scenario.simulated.flights[0].flight_id


def _departure_view(ctx: AgentContext) -> dict:
    from ..reports import departure_readiness
    sc, fid = ctx.scenario, _flight(ctx)
    dep = sc.simulated.departure
    nf = next(n for n in dep.next_flights if n.flight_id == fid)
    sf = next(f for f in sc.simulated.flights if f.flight_id == fid)
    crew = next(c for c in sc.simulated.flight_crews if c.id == sf.flight_crew_id)
    rd = departure_readiness(sc, fid)
    af = next(x for x in dep.arrival if x.flight_id == fid)
    return {"session": "TURNAROUND + DEPARTURE: you confirm your aircraft's and crew's readiness for the NEXT flight",
            "your_aircraft": {"flight_id": fid, "aircraft_class": sf.aircraft_class.value, "wake_category": sf.wake_category.value, "priority": sf.priority,
                              "arrived_at_gate_min (fixed)": af.gate_in_min, "next_flight": nf.model_dump(),
                              "scheduled_departure_min": dep.scheduled_departure_min[fid],
                              "flight_crew": {"id": crew.id, "latest_takeoff_min_by_duty": rd.latest_takeoff_min,
                                              "rule": "take-off + next flight block + post-flight must not exceed the crew duty limit"},
                              "readiness_constraints": ["refuelling must finish before boarding", "workbook minimum turnaround", "crew duty for the next flight"]},
            "readiness_template": rd.model_dump(mode="json"),
            "other_traffic": [{"flight_id": f.flight_id, "wake_category": f.wake_category.value,
                               "scheduled_departure_min": dep.scheduled_departure_min[f.flight_id]} for f in sc.simulated.flights if f.flight_id != fid]}


def domain_view(ctx: AgentContext) -> dict:
    if ctx.scenario.simulated.departure is not None:
        return _departure_view(ctx)
    sc = ctx.scenario
    fid = _flight(ctx)
    sf = next(f for f in sc.simulated.flights if f.flight_id == fid)
    fa = fuel_account(sc, sf, sf.earliest_landing_min)
    fcw = sc.workbook.fuel_crew
    crew = next(c for c in sc.simulated.flight_crews if c.id == sf.flight_crew_id)
    conn = sc.simulated.connection_for(fid)
    a = sc.simulated.assumptions
    h = ctx.historical
    remaining_nm = sf.earliest_landing_min / 60.0 * a.cruise_speed_kt
    position = {"SIMULATED": True, "minutes_to_nominal_landing": sf.earliest_landing_min,
                "distance_to_airport_nm": round(remaining_nm, 1), "assumed_cruise_speed_kt": a.cruise_speed_kt}
    if fid == h.label and h.distance_mi:
        total_nm = h.distance_mi * NM_PER_MILE
        position["route_distance_nm (flights.csv)"] = round(total_nm, 1)
        position["distance_from_origin_nm"] = round(max(0.0, total_nm - remaining_nm), 1)
    weather = {"runway_closures (SIMULATED weather)": {r.id: [i.model_dump() for i in r.closures] for r in sc.simulated.runways if r.closures},
               "taxiway_blockages (SIMULATED)": {x.id: [i.model_dump() for i in x.blockages] for x in sc.simulated.taxi_segments if x.blockages}}
    if fid == h.label:
        weather["historical_weather_delay_min (dataset, context only)"] = h.weather_delay_min
        weather["historical_late_aircraft_delay_min (dataset, context only)"] = h.late_aircraft_delay_min
    dv = sc.simulated.diversion
    diversion = None
    if dv is not None and fid == dv.diverting_flight_id:
        diversion = {
            "session": "DIVERSION: landing at the original destination is no longer feasible; you evaluate the complete options and DECLARE your status",
            "COMPUTED_STATUS (you must declare exactly this)": dv.computed_status.value, "status_rule": dv.status_rule,
            "option_A_original_destination": dv.destination.model_dump(mode="json"),
            "option_B_alternates (screened; a historical route alone does NOT make an airport suitable)": [c.model_dump(mode="json") for c in dv.candidates],
            "chosen_alternate": dv.alternate_airport, "position (SIMULATED)": dv.position, "assumptions": dv.assumptions,
            "if_no_option_is_feasible": "the session ends with NO_FEASIBLE_PLAN and the case is ESCALATED (human action)"}
    return {
        **({"diversion": diversion} if diversion else {}),
        "your_aircraft": {
            "flight_id": fid, "aircraft_class": sf.aircraft_class.value, "wake_category": sf.wake_category.value,
            "passengers": sf.passengers, "priority": sf.priority, "status": sf.status.value,
            "eta_window": {"nominal_eta_min": sf.earliest_landing_min,
                           "earliest_landing_min": sf.earliest_landing_min,
                           "latest_landing_min_protecting_contingency": round(latest_landing_min(sc, sf), 3),
                           "latest_landing_min_protecting_final_reserve_only (never use)": round(latest_landing_min(sc, sf, protect_contingency=False), 3)},
            "fuel_at_nominal_eta": {k: (quantise(k, v) if k in DECIMALS else round(v, 2) if isinstance(v, float) else v)
                                    for k, v in fa.as_dict().items()},
            "fuel_rules": {"final_reserve_min": fcw.final_reserve_min, "contingency_fraction_of_trip": fcw.contingency_fraction,
                           "holding_burn_kg_per_h": fa.hold_burn_kg_h},
            "position": position, "weather": weather,
            "ground_fuel": {"ground_burn_kg_per_h (ASSUMPTION)": round(fa.hold_burn_kg_h * a.ground_burn_fraction_of_hold, 1),
                            "rule": "fuel at gate arrival (after taxi + ground wait) must stay >= final reserve: a long ground wait or a longer "
                                    "taxi route can be unacceptable; ground waiting is preferable to airborne holding only while this holds"},
            "flight_crew": {"id": crew.id, "duty_limit_min": crew.report_time_min + fcw.flight_crew_max_duty_h * 60,
                            "postflight_min": a.flight_crew_postflight_min,
                            "rule": "gate_in_min + postflight must not exceed the duty limit"},
            "aircraft_restrictions": {"gate_class_needed": "any gate" if sf.aircraft_class.value == "narrowbody" else "widebody gate only",
                                      "wake_category": sf.wake_category.value},
            "connection_preference_SIMULATED_soft": conn.model_dump() if conn else None,
        },
        "report_template": pilot_report(sc, fid).model_dump(mode="json"),
        "other_traffic": [{"flight_id": f.flight_id, "wake_category": f.wake_category.value, "nominal_eta_min": f.earliest_landing_min}
                          for f in sc.simulated.flights if f.flight_id != fid],
    }


def tool_results(ctx: AgentContext) -> dict:
    fid = _flight(ctx)
    return {"description": f"valid options ranked by YOUR objective (flight {fid}: its delay, connection, fuel margin)",
            "options": option_menu(ROLE, ctx.scenario, 5, flight=fid)}


def system_prompt() -> str:
    return base.build_system(INSTRUCTIONS)


def standing_conditions(ctx: AgentContext) -> list[Condition]:
    sc, fid = ctx.scenario, _flight(ctx)
    best = best_for(ROLE, sc, flight=fid)
    if best is None:
        return []
    if sc.simulated.departure is not None:
        m = best.metrics["flights"][fid]
        return [Condition(field=ConditionField.DEPARTURE_DELAY_MIN, op=ConditionOp.LE, flight_id=fid, value=round(m["departure_delay_min"] + 5, 2)),
                Condition(field=ConditionField.TURNAROUND_MIN, op=ConditionOp.LE, flight_id=fid, value=round(m["turnaround_min"] + 10, 2))]
    sf = next(f for f in sc.simulated.flights if f.flight_id == fid)
    m = best.metrics["flights"][fid]
    margin0 = fuel_account(sc, sf, sf.earliest_landing_min).contingency_margin_kg   # spare holding fuel at nominal ETA, kg
    return [Condition(field=ConditionField.LANDING_DELAY_MIN, op=ConditionOp.LE, flight_id=fid, value=round(m["landing_delay_min"] + 3, 2)),
            Condition(field=ConditionField.CONNECTION_MISS_MIN, op=ConditionOp.LE, flight_id=fid, value=round(m["connection_miss_min"] + 5, 2)),
            # fuel preference (soft, on top of the hard reserve/contingency checks): keep >= 75 % of the spare holding fuel
            Condition(field=ConditionField.CONTINGENCY_MARGIN_KG, op=ConditionOp.GE, flight_id=fid, value=round(0.75 * margin0, 1))]


def scripted_policy(ctx: AgentContext, fallback_spec) -> str:
    fid = _flight(ctx)
    if ctx.scenario.simulated.departure is not None:
        return base.scripted_negotiate(ctx, ROLE, standing_conditions(ctx), flight=fid, label=f"PILOT {fid}",
                                       metrics=[TradeoffMetric.DEPARTURE_DELAY_MIN, TradeoffMetric.READY_MIN, TradeoffMetric.TURNAROUND_MIN],
                                       reasons=[ReasonCode.DELAY_MINIMISATION, ReasonCode.TURNAROUND_TIME], fallback_spec=fallback_spec)
    return base.scripted_negotiate(ctx, ROLE, standing_conditions(ctx), flight=fid, label=f"PILOT {fid}" if ctx.who.agent_id != "PILOT" else "PILOT",
                                   metrics=[TradeoffMetric.LANDING_DELAY_MIN, TradeoffMetric.CONNECTION_MISS_MIN, TradeoffMetric.GATE_IN_MIN],
                                   reasons=[ReasonCode.DELAY_MINIMISATION, ReasonCode.CONNECTION_PRESERVATION], fallback_spec=fallback_spec)
