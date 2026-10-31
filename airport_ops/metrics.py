"""Deterministic plan metrics and machine evaluation of message conditions. No model involved."""
from __future__ import annotations

import operator
from typing import Optional

from .canonical import quantise
from .fallback import CostWeights, plan_cost
from .fuel import fuel_account
from .messages import FLIGHT_ID_FIELDS, Condition, ConditionOp, ConditionResult
from .plan import RecoveryPlan
from .schemas import Scenario

_OPS = {ConditionOp.LE: operator.le, ConditionOp.GE: operator.ge, ConditionOp.LT: operator.lt,
        ConditionOp.GT: operator.gt, ConditionOp.EQ: operator.eq, ConditionOp.NE: operator.ne}
_EPS = 1e-9


def plan_metrics(sc: Scenario, plan: RecoveryPlan) -> dict:
    """{'flights': {fid: {...}}, 'plan': {...}} using the same definitions as the ConditionField docs."""
    a = sc.simulated.assumptions
    sims = {f.flight_id: f for f in sc.simulated.flights}
    conn = sc.simulated.connection
    per: dict[str, dict] = {}
    for p in plan.flights:
        sf = sims.get(p.flight_id)
        if sf is None:
            continue
        fa = fuel_account(sc, sf, p.landing_min)
        per[p.flight_id] = {
            "landing_delay_min": p.landing_min - sf.earliest_landing_min, "landing_min": p.landing_min,
            "gate_in_min": p.gate_in_min, "service_start_min": p.service_start_min, "service_end_min": p.service_end_min,
            "service_wait_min": p.service_start_min - (p.gate_in_min + a.deboard_min),
            "reserve_margin_kg": fa.reserve_margin_kg, "contingency_margin_kg": fa.contingency_margin_kg,
            "connection_miss_min": (max(0.0, p.gate_in_min - conn.preferred_gate_in_by_min)
                                    if conn and conn.flight_id == p.flight_id else 0.0),
            "gate_id": p.gate_id, "runway_id": p.runway_id, "ground_crew_id": p.ground_crew_id,
            "priority": sf.priority,
        }
    fcr = sc.workbook.fuel_crew
    overtime = 0.0
    for g in sc.simulated.ground_crews:
        mine = [p for p in plan.flights if p.ground_crew_id == g.id]
        if mine:
            overtime += max(0.0, max(p.service_end_min for p in mine) - (g.shift_start_min + fcr.ground_shift_h * 60))
    delays = [v["landing_delay_min"] for v in per.values()] or [0.0]
    flights = {f.flight_id: f for f in sc.simulated.flights}
    plan_level = {
        "max_landing_delay_min": max(delays), "total_landing_delay_min": sum(delays),
        "weighted_landing_delay": sum(v["priority"] * v["landing_delay_min"] for v in per.values()),
        "total_service_wait_min": sum(v["service_wait_min"] for v in per.values()),
        "ground_overtime_min": overtime,
        "gate_changes": float(sum(p.gate_id != flights[p.flight_id].original_gate_id for p in plan.flights if p.flight_id in flights)),
        "runway_changes": float(sum(p.runway_id != flights[p.flight_id].original_runway_id for p in plan.flights if p.flight_id in flights)),
        "total_cost": sum(plan_cost(sc, plan, CostWeights()).values()) if len(per) == len(plan.flights) else float("nan"),
    }
    return {"flights": per, "plan": plan_level}


def evaluate_condition(c: Condition, metrics: dict) -> ConditionResult:
    if c.field.value in metrics["plan"]:
        observed = metrics["plan"].get(c.field.value)
    else:
        row = metrics["flights"].get(c.flight_id or "")
        observed = None if row is None else row.get(c.field.value)
    if observed is None or (isinstance(observed, float) and observed != observed):
        return ConditionResult(condition=c, satisfied=None, observed=None)
    if c.field in FLIGHT_ID_FIELDS:
        return ConditionResult(condition=c, satisfied=bool(_OPS[c.op](observed, c.value)), observed=observed)
    # Both sides use the canonical per-field quantisation (canonical.py) - the SAME representation agents are shown - so a bound
    # copied from the prompt (e.g. "<= 10.71" for an exact 10.7143) does not fail on display rounding. Fuel margins keep 6 decimals.
    # Preference conditions only: hard constraints are validated separately, at full precision, in validate.py.
    lhs, rhs = quantise(c.field.value, observed), quantise(c.field.value, c.value)
    ok = {ConditionOp.LE: lhs <= rhs + _EPS, ConditionOp.GE: lhs >= rhs - _EPS, ConditionOp.LT: lhs < rhs - _EPS,
          ConditionOp.GT: lhs > rhs + _EPS, ConditionOp.EQ: abs(lhs - rhs) <= _EPS, ConditionOp.NE: abs(lhs - rhs) > _EPS}[c.op]
    return ConditionResult(condition=c, satisfied=ok, observed=lhs, observed_exact=float(observed))


def evaluate_conditions(conds: list[Condition], metrics: dict) -> list[ConditionResult]:
    return [evaluate_condition(c, metrics) for c in conds]


def all_satisfied(results: list[ConditionResult]) -> bool:
    """True only if every condition evaluated to True. Unresolved (None) is NOT satisfied."""
    return all(r.satisfied is True for r in results)


def metric_delta(before: dict, after: dict) -> dict[str, float]:
    keys = ["max_landing_delay_min", "total_landing_delay_min", "weighted_landing_delay", "total_service_wait_min",
            "ground_overtime_min", "gate_changes", "runway_changes", "total_cost"]
    return {k: round(after["plan"][k] - before["plan"][k], 4) for k in keys}
