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
Objective (soft): avoid ground-crew overtime (overtime is priced at the workbook multiplier and capped), minimise time
aircraft wait for a crew or gate, and avoid gate changes (a changed gate needs the workbook notice before gate arrival).
You do NOT own runway sequencing or fuel. Stands and ground equipment are NOT modelled in this prototype: ground crews are
the only turnaround capacity. Prefer options from tool_results.options; state measurable conditions
(e.g. ground_overtime_min <= X) rather than free-text promises."""


def domain_view(ctx: AgentContext) -> dict:
    sc = ctx.scenario
    a, w = sc.simulated.assumptions, sc.workbook
    fcr, sub = w.fuel_crew, w.capacity_subset
    return {
        "gates": [{"id": g.id, "terminal": g.terminal, "max_aircraft_class": g.max_aircraft_class.value,
                   "resident_occupancy": [i.model_dump() for i in g.resident_occupancy]} for g in sc.simulated.gates],
        "gate_capacity": {"airport_total_gates (WORKBOOK)": sub.total_gates, "modelled_gates": sub.modelled_gates,
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
    return {"description": "valid options ranked by YOUR objective (overtime, waiting, gate changes)",
            "options": option_menu(ROLE, ctx.scenario, 5)}


def system_prompt() -> str:
    return base.build_system(INSTRUCTIONS)


def standing_conditions(ctx: AgentContext) -> list[Condition]:
    best = best_for(ROLE, ctx.scenario)
    if best is None:
        return []
    p = best.metrics["plan"]
    return [Condition(field=ConditionField.GROUND_OVERTIME_MIN, op=ConditionOp.LE, value=round(p["ground_overtime_min"] + 20, 2)),
            Condition(field=ConditionField.TOTAL_SERVICE_WAIT_MIN, op=ConditionOp.LE, value=round(p["total_service_wait_min"] + 25, 2))]


def scripted_policy(ctx: AgentContext, fallback_spec) -> str:
    return base.scripted_negotiate(ctx, ROLE, standing_conditions(ctx), metrics=[TradeoffMetric.GROUND_OVERTIME_MIN, TradeoffMetric.TOTAL_SERVICE_WAIT_MIN, TradeoffMetric.GATE_CHANGES], flight=None,
                                   reasons=[ReasonCode.OVERTIME, ReasonCode.CREW_AVAILABILITY], label="GROUND",
                                   fallback_spec=fallback_spec)
