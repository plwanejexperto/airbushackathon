"""ONE canonical representation of preference metrics, used both for what agents are SHOWN and for how their conditions
are EVALUATED. Quantisation is per field (documented below), never blanket, and it never touches hard constraints.

  quantity                                   decimals   why
  -----------------------------------------  ---------  --------------------------------------------------------------
  minutes (delay, times, wait, overtime)         2      0.01 min = 0.6 s: finer than any displayed schedule value
  weighted delay, total cost (ranking units)     2      arbitrary units, not physical
  gate / runway change counts                    0      integers
  fuel margins (kg above reserve / contingency)  6      micro-kg = the validator's own tolerance (1e-6): effectively exact,
                                                        so a fuel preference can never be satisfied by rounding up a deficit
  identifiers (gate, runway, crew)               exact  strings

Hard constraints (validate.py, fuel.py) do NOT import this module: they use exact values with their own tolerances, so display
quantisation can never make an unsafe plan valid. A condition can pass while the plan is invalid; consensus additionally needs
independent validation, which this module cannot influence.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

DECIMALS: dict[str, int] = {
    "landing_delay_min": 2, "landing_min": 2, "gate_in_min": 2, "service_start_min": 2, "service_end_min": 2,
    "service_wait_min": 2, "connection_miss_min": 2, "max_landing_delay_min": 2, "total_landing_delay_min": 2,
    "total_service_wait_min": 2, "ground_overtime_min": 2, "weighted_landing_delay": 2, "total_cost": 2,
    "ready_min": 2, "takeoff_min": 2, "departure_delay_min": 2, "turnaround_min": 2, "total_departure_delay_min": 2,
    "weighted_departure_delay": 2, "max_departure_delay_min": 2,
    "gate_changes": 0, "runway_changes": 0, "reserve_margin_kg": 6, "contingency_margin_kg": 6, "gate_fuel_margin_kg": 6,
}


def quantise(field: str, value: float) -> float:
    """Round half-up to the field's documented decimals (decimal arithmetic, deterministic). Unknown fields: unchanged."""
    d = DECIMALS.get(field)
    v = float(value)
    if d is None or v != v:
        return v
    return float(Decimal(repr(v)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def canonical_metrics(m: dict) -> dict:
    """Same structure as plan_metrics(); every numeric field in DECIMALS quantised. Use for ALL agent-visible metric values."""
    return {
        "plan": {k: quantise(k, v) for k, v in m["plan"].items()},
        "flights": {fid: {k: (quantise(k, v) if k in DECIMALS and not isinstance(v, str) else v) for k, v in row.items()}
                    for fid, row in m["flights"].items()},
    }


def policy_text() -> str:
    """Human/LLM-readable statement of the rule (embedded in every agent prompt)."""
    by: dict[int, list[str]] = {}
    for k, d in DECIMALS.items():
        by.setdefault(d, []).append(k)
    return "; ".join(f"{'integers' if d == 0 else f'{d} decimals'}: {', '.join(v)}" for d, v in sorted(by.items(), reverse=True))
