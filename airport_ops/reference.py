"""The INITIAL REFERENCE PLAN: an ILLUSTRATIVE starting point that negotiated changes are described against. It is NOT a feasible
baseline (it normally violates hard constraints), so differences from it are explanations of what had to change, never evidence of
an operational improvement. An efficiency claim needs a comparison with a FEASIBLE plan (e.g. the deterministic fallback).

Definition: before negotiation, each flight is planned exactly as originally assigned - nominal ETA landing on its original
runway, original gate and ground crew, shortest taxi route, no conflict resolution. It is a SIMULATED planning baseline in
minutes from scenario start. It is validated like any other plan, and normally FAILS some hard constraints (that is the
disruption the agents must resolve).

It is NOT the historical flight's actual outcome: flights.csv scheduled/actual arrival times and delay minutes are dataset
context only. The negotiated times are simulated minutes (t+N) and must not be read as "recovering" the historical delay.
"""
from __future__ import annotations

from .plan import RecoveryPlan, make_plan
from .planner import Planner
from .schemas import Scenario

REFERENCE_SOURCE = "reference"
REFERENCE_DESCRIPTION = ("ILLUSTRATIVE reference plan: nominal ETA landing, original runway/gate/ground crew, shortest taxi route, "
                         "no conflict resolution. It normally FAILS hard constraints, so it is not a feasible baseline and no "
                         "efficiency or improvement is claimed against it. Not the historical actual outcome.")


def reference_plan(sc: Scenario) -> RecoveryPlan:
    base = Planner(sc).naive_plan()
    return make_plan(sc, REFERENCE_SOURCE, base.flights)
