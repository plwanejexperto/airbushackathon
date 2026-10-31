"""Ground agent: gates, crews, turnaround. (Stands/equipment are not modelled; ground crews are the capacity proxy.)"""
from __future__ import annotations

from ..messages import AgentRole, Condition, ConditionField, ConditionOp, ReasonCode, TradeoffMetric
from ..negotiation_types import AgentContext
from ..tools import best_for, option_menu
from . import base

ROLE = AgentRole.GROUND

INSTRUCTIONS = """\
ROLE: GROUND OPERATIONS agent.
You control gate assignment, ground-crew assignment and the turnaround (deboarding, servicing).
In a DIVERSION session your instances are: GROUND of the ALTERNATE (compatible stands/gates/handling for the diverting aircraft, EMERGENCY aircraft
never wait for a gate before landing) and GROUND of the original DESTINATION (confirm it cannot handle the aircraft in time).
You can offer compatible gates, REMOTE STANDS (bussing adds deboarding time) and the designated safe GROUND WAITING AREA. NEVER make
an aircraft stay airborne just because no gate is free: if the runway is available and the gate is occupied, propose another gate, a
remote stand, or ground waiting - but check the fuel at gate arrival (gate_fuel_margin_kg): ground time burns fuel too.
In TURNAROUND + DEPARTURE sessions you schedule the tasks (they may overlap where dependencies and equipment allow), share the single fuel truck
and the baggage tractors between aircraft, and propose when each aircraft is READY for its next departure. Ready time = all tasks done and the
workbook minimum turnaround met.
Objective (soft): avoid ground-crew overtime (overtime is priced at the workbook multiplier and capped), minimise time
aircraft wait for a crew or gate, and avoid gate changes (a changed gate needs the workbook notice before gate arrival).
You do NOT own runway sequencing or fuel. Stands and ground equipment are NOT modelled in this prototype: ground crews are
the only turnaround capacity. Prefer options from tool_results.options; state measurable conditions
(e.g. ground_overtime_min <= X) rather than free-text promises."""


def _departure_view(ctx: AgentContext) -> dict:
    from ..turnaround import task_minutes
    sc = ctx.scenario
    s, dep, w, a = sc.simulated, sc.simulated.departure, sc.workbook, sc.simulated.assumptions
    fcr = w.fuel_crew
    return {
        "session": "TURNAROUND + DEPARTURE: you schedule deplaning, cleaning, refuelling, baggage, boarding and checks and propose each ready time",
        "fixed_arrivals (agreed, not negotiable)": [{"flight": x.flight_id, "gate": x.gate_id, "gate_arrival_min": x.gate_in_min} for x in dep.arrival],
        "turnaround_tasks (SIMULATED work content)": [{"task": t.task_id, "name": t.name, "base_min": t.base_min, "after": t.predecessors,
                                                       "equipment_type": t.equipment} for t in dep.turnaround.tasks],
        "task_minutes_per_flight": {x.flight_id: task_minutes(sc, x.flight_id, x.gate_id) for x in dep.arrival},
        "equipment (SIMULATED, one task at a time per unit)": [u.model_dump() for u in dep.turnaround.equipment],
        "ground_crews": [{"id": c.id, "shift_end_min": c.shift_start_min + fcr.ground_shift_h * 60, "available_from_min": c.available_from_min} for c in s.ground_crews],
        "rules": {"min_turnaround_min (WORKBOOK)": w.airport.min_turnaround_min, "pushback_min (ASSUMPTION)": dep.turnaround.pushback_min,
                  "overtime_cap_min (ASSUMPTION)": a.max_ground_overtime_min, "overtime_multiplier (WORKBOOK)": fcr.ground_overtime_multiplier,
                  "no boarding during refuelling": True, "tasks may overlap where dependencies and equipment allow": True},
        "next_flights": [n.model_dump() for n in dep.next_flights],
        "scheduled_departure_min": dep.scheduled_departure_min,
    }


def _destination_view(ctx: AgentContext) -> dict:
    dv = ctx.scenario.simulated.diversion
    d = dv.destination
    return {"role_in_this_session": f"GROUND of the ORIGINAL DESTINATION {d.airport}: confirm the destination cannot handle the diverting aircraft in time; you do NOT control the alternate's plan",
            "diverting_aircraft": {"flight_id": dv.diverting_flight_id, "computed_status": dv.computed_status.value},
            "destination_status": d.model_dump(mode="json"), "workbook_limits": d.limits}


def domain_view(ctx: AgentContext) -> dict:
    if base.is_dest_agent(ctx):
        return _destination_view(ctx)
    if ctx.scenario.simulated.departure is not None:
        return _departure_view(ctx)
    sc = ctx.scenario
    a, w = sc.simulated.assumptions, sc.workbook
    fcr, sub = w.fuel_crew, w.capacity_subset
    return {
        "gates_and_stands": [{"id": g.id, "kind": g.kind + (" (SIMULATED)" if g.kind == "remote_stand" else ""), "terminal": g.terminal,
                              "max_aircraft_class": g.max_aircraft_class.value,
                              "resident_occupancy": [i.model_dump() for i in g.resident_occupancy]} for g in sc.simulated.gates],
        "safe_ground_waiting_areas (SIMULATED)": [h.model_dump() for h in sc.simulated.hold_areas],
        "ground_fuel_rule": {"ground_burn_fraction_of_hold (ASSUMPTION)": a.ground_burn_fraction_of_hold,
                             "rule": "fuel at gate arrival must stay >= final reserve (FUEL-GROUND)"},
        "remote_stand_bus_min (ASSUMPTION)": a.remote_stand_bus_min,
        "gate_capacity": {"airport_total_gates (WORKBOOK)": sub.total_gates, "modelled_gates": sub.modelled_gates,
                          "modelled_remote_stands (SIMULATED)": sub.modelled_remote_stands,
                          "spare_gates_at_any_time (WORKBOOK)": sub.spare_gates_at_any_time,
                          "stands_and_equipment": "not modelled; ground crews are the capacity proxy"},
        "ground_crews": [{"id": c.id, "shift_start_min": c.shift_start_min, "shift_end_min": c.shift_start_min + fcr.ground_shift_h * 60,
                          "available_from_min": c.available_from_min} for c in sc.simulated.ground_crews],
        "overtime": {"workbook_shift_h": fcr.ground_shift_h, "workbook_multiplier": fcr.ground_overtime_multiplier,
                     "cap_min (ASSUMPTION)": a.max_ground_overtime_min},
        "turnaround": {"min_turnaround_min (WORKBOOK)": w.airport.min_turnaround_min, "deboard_min (ASSUMPTION)": a.deboard_min,
                       "gate_buffer_min (ASSUMPTION)": a.gate_buffer_min, "crew_transfer_min (ASSUMPTION)": a.ground_crew_transfer_min,
                       "gate_change_notice_min (WORKBOOK)": fcr.gate_change_notice_min,
                       "rule": "gate arrival -> service end must be >= min_turnaround; service starts >= deboard after gate arrival"},
        "flights": [{"flight_id": f.flight_id, "aircraft_class": f.aircraft_class.value, "passengers": f.passengers,
                     "priority": f.priority, "service_task_min": f.service_task_min, "original_gate": f.original_gate_id,
                     "original_ground_crew": f.original_ground_crew_id} for f in sc.simulated.flights],
    }


def tool_results(ctx: AgentContext) -> dict:
    if base.is_dest_agent(ctx):
        return {"description": "not applicable: you confirm the destination status", "options": []}
    return {"description": "valid options ranked by YOUR objective (overtime, waiting, gate changes)",
            "options": option_menu(ROLE, ctx.scenario, 5)}


def system_prompt() -> str:
    return base.build_system(INSTRUCTIONS)


def standing_conditions(ctx: AgentContext) -> list[Condition]:
    best = best_for(ROLE, ctx.scenario)
    if best is None:
        return []
    if ctx.scenario.simulated.departure is not None:
        p = best.metrics["plan"]
        return [Condition(field=ConditionField.GROUND_OVERTIME_MIN, op=ConditionOp.LE, value=round(p["ground_overtime_min"] + 20, 2)),
                Condition(field=ConditionField.TOTAL_SERVICE_WAIT_MIN, op=ConditionOp.LE, value=round(p["total_service_wait_min"] + 25, 2))]
    p = best.metrics["plan"]
    return [Condition(field=ConditionField.GROUND_OVERTIME_MIN, op=ConditionOp.LE, value=round(p["ground_overtime_min"] + 20, 2)),
            Condition(field=ConditionField.TOTAL_SERVICE_WAIT_MIN, op=ConditionOp.LE, value=round(p["total_service_wait_min"] + 25, 2))]


def scripted_policy(ctx: AgentContext, fallback_spec) -> str:
    if base.is_dest_agent(ctx):
        return base.scripted_destination(ctx, ctx.who.agent_id, fallback_spec)
    if ctx.scenario.simulated.departure is not None:
        return base.scripted_negotiate(ctx, ROLE, standing_conditions(ctx), metrics=[TradeoffMetric.GROUND_OVERTIME_MIN, TradeoffMetric.TOTAL_SERVICE_WAIT_MIN, TradeoffMetric.WEIGHTED_DEPARTURE_DELAY],
                                       flight=None, reasons=[ReasonCode.OVERTIME, ReasonCode.CREW_AVAILABILITY, ReasonCode.TURNAROUND_TIME], label="GROUND", fallback_spec=fallback_spec)
    return base.scripted_negotiate(ctx, ROLE, standing_conditions(ctx), metrics=[TradeoffMetric.GROUND_OVERTIME_MIN, TradeoffMetric.TOTAL_SERVICE_WAIT_MIN, TradeoffMetric.GATE_CHANGES], flight=None,
                                   reasons=[ReasonCode.OVERTIME, ReasonCode.CREW_AVAILABILITY], label="GROUND",
                                   fallback_spec=fallback_spec)
