"""Bounded, deterministic, non-AI fallback planner.

Candidate set (finite, fully enumerated in a fixed order):
    landing order            : every permutation of the 3 flights
  x runway per flight        : every modelled runway
  x gate per flight          : every modelled gate (static pre-filter: aircraft-class compatible)
  x ground crew per flight   : every modelled ground crew
Landing times are NOT free variables: for each (order, runway) the earliest landing satisfying
closures, wake separation and runway capacity is used. So "exhausted" means *this candidate set* is
exhausted, not that no plan exists in continuous time.

Pipeline: build timeline -> HARD constraints filter (validate.py, no trade-offs) -> rank feasible plans
by fixed costs -> deterministic tie-breakers -> re-validate the winner before returning it.
Safety is never scored: a plan that fails any hard constraint is discarded however cheap it is.
"""
from __future__ import annotations

import itertools
import time
from collections import Counter
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .plan import RecoveryPlan
from .planner import Planner
from .schemas import Scenario
from .validate import ValidationReport, validate_plan
from .workbook import AircraftClass


class Outcome(str, Enum):
    FALLBACK_SELECTED = "FALLBACK_SELECTED"
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"


class SearchStatus(str, Enum):
    EXHAUSTED = "EXHAUSTED"  # every candidate accounted for (validated or pruned by a shared check)
    TIMEOUT = "TIMEOUT"      # stopped early (wall-clock or evaluation budget): NOT a proof of infeasibility


class CostWeights(BaseModel):
    """Fixed ranking costs (arbitrary units). Used only AFTER hard-constraint filtering."""
    landing_delay_per_min_x_priority: float = 1.0   # (landing - nominal ETA) * priority
    exit_hold_per_min_x_priority: float = 0.5       # waiting at runway exit before taxiing * priority
    service_wait_per_min_x_priority: float = 0.5    # gate arrival + deboard -> service start (crew/gate waiting) * priority
    gate_change: float = 10.0                       # per flight not at its original gate
    runway_change: float = 2.0                      # per flight not on its original runway
    ground_overtime_per_min: float = 1.0            # x workbook overtime multiplier (1.5)


class FallbackResult(BaseModel):
    outcome: Outcome
    search_status: SearchStatus
    plan: Optional[RecoveryPlan] = None
    validation: Optional[ValidationReport] = None
    cost: Optional[float] = None
    cost_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    candidates_total: int = 0
    candidates_validated: int = 0
    candidates_pruned: int = 0
    feasible_count: int = 0
    rejection_by_constraint: dict[str, int] = Field(default_factory=dict)


def plan_cost(sc: Scenario, plan: RecoveryPlan, w: CostWeights) -> dict[str, float]:
    flights = {f.flight_id: f for f in sc.simulated.flights}
    a = sc.simulated.assumptions
    fcr = sc.workbook.fuel_crew
    bd = Counter()
    for p in plan.flights:
        sf = flights[p.flight_id]
        bd["landing_delay"] += w.landing_delay_per_min_x_priority * sf.priority * (p.landing_min - sf.earliest_landing_min)
        bd["exit_hold"] += w.exit_hold_per_min_x_priority * sf.priority * (p.taxi_start_min - (p.landing_min + a.runway_occupancy_min))
        bd["service_wait"] += w.service_wait_per_min_x_priority * sf.priority * (
            p.service_start_min - (p.gate_in_min + a.deboard_min))
        bd["gate_change"] += w.gate_change * (p.gate_id != sf.original_gate_id)
        bd["runway_change"] += w.runway_change * (p.runway_id != sf.original_runway_id)
    for g in sc.simulated.ground_crews:
        mine = [p for p in plan.flights if p.ground_crew_id == g.id]
        if mine:
            end = g.shift_start_min + fcr.ground_shift_h * 60
            ot = max(0.0, max(p.service_end_min for p in mine) - end)
            bd["ground_overtime"] += w.ground_overtime_per_min * fcr.ground_overtime_multiplier * ot
    return {k: round(v, 3) for k, v in bd.items()}


def run_fallback(scenario: Scenario, *, time_budget_s: float = 30.0, max_evaluations: Optional[int] = None,
                 weights: Optional[CostWeights] = None) -> FallbackResult:
    w = weights or CostWeights()
    t_start = time.monotonic()
    pl = Planner(scenario)
    sim = scenario.simulated
    fids = [f.flight_id for f in sim.flights]
    rw_ids = [r.id for r in sim.runways]
    gate_ids = [g.id for g in sim.gates]
    crew_ids = [c.id for c in sim.ground_crews]
    n = len(fids)

    orders = list(itertools.permutations(fids))
    rw_assigns = list(itertools.product(rw_ids, repeat=n))
    crew_assigns = list(itertools.product(crew_ids, repeat=n))
    gate_opts = []
    for f in sim.flights:
        gate_opts.append([g.id for g in sim.gates
                          if f.aircraft_class == AircraftClass.NARROWBODY or g.max_aircraft_class == AircraftClass.WIDEBODY])
    full_gate_assigns = len(gate_ids) ** n
    ok_gate_assigns = 1
    for o in gate_opts:
        ok_gate_assigns *= len(o)
    total = len(orders) * len(rw_assigns) * full_gate_assigns * len(crew_assigns)
    pruned = len(orders) * len(rw_assigns) * (full_gate_assigns - ok_gate_assigns) * len(crew_assigns)
    reject: Counter = Counter()
    res = FallbackResult(outcome=Outcome.NO_FEASIBLE_PLAN, search_status=SearchStatus.EXHAUSTED, candidates_total=total)
    reasons = res.reasons
    if pruned:
        reject["GATE-COMPAT"] += pruned
        no_gate = [fids[i] for i, o in enumerate(gate_opts) if not o]
        if no_gate:
            reasons.append(f"No class-compatible gate exists for {no_gate}.")

    best = None  # (key, plan, breakdown)
    runner_up_cost = None
    feasible = validated = 0
    timed_out = False
    gate_assigns = list(itertools.product(*gate_opts)) if ok_gate_assigns else []

    def budget_left() -> bool:
        if max_evaluations is not None and validated >= max_evaluations:
            return False
        return time.monotonic() - t_start <= time_budget_s

    for order in orders:
        if timed_out or not gate_assigns:
            break
        for rwa in rw_assigns:
            if timed_out:
                break
            rwd = dict(zip(fids, rwa))
            probe = pl.build(order, rwd, dict(zip(fids, gate_assigns[0])), dict(zip(fids, crew_assigns[0])))
            ext = len(gate_assigns) * len(crew_assigns)
            if probe is None:  # no taxi route / horizon exceeded: all extensions share the cause
                reject["TAXI-ROUTE"] += ext
                pruned += ext
                continue
            rep = validate_plan(scenario, probe, stages=("land", "fuel"), record_passes=False, fail_fast=True)
            if not rep.passed:  # landing-stage constraints ignore gate/crew: whole branch is infeasible
                reject[rep.first_failure_id] += ext
                pruned += ext
                continue
            for gts in gate_assigns:
                if timed_out:
                    break
                gd = dict(zip(fids, gts))
                for crs in crew_assigns:
                    if not budget_left():
                        timed_out = True
                        break
                    plan = pl.build(order, rwd, gd, dict(zip(fids, crs)))
                    validated += 1
                    if plan is None:
                        reject["TAXI-ROUTE"] += 1
                        continue
                    rep = validate_plan(scenario, plan, stages=("ground",), record_passes=False, fail_fast=True)
                    if not rep.passed:
                        reject[rep.first_failure_id] += 1
                        continue
                    feasible += 1
                    bd = plan_cost(scenario, plan, w)
                    cost = round(sum(bd.values()), 6)
                    max_delay = max(p.landing_min - pl.flights[p.flight_id].earliest_landing_min for p in plan.flights)
                    # ranking key: cost, worst delay, then fixed enumeration identity (lexicographic)
                    key = (cost, round(max_delay, 6), order, rwa, gts, crs)
                    if best is None or key < best[0]:
                        if best is not None:
                            runner_up_cost = best[0][0] if runner_up_cost is None else min(runner_up_cost, best[0][0])
                        best = (key, plan, bd)
                    elif runner_up_cost is None or cost < runner_up_cost:
                        runner_up_cost = cost

    res.candidates_validated, res.candidates_pruned, res.feasible_count = validated, pruned, feasible
    res.rejection_by_constraint = dict(reject.most_common())
    res.search_status = SearchStatus.TIMEOUT if timed_out else SearchStatus.EXHAUSTED
    complete = res.search_status == SearchStatus.EXHAUSTED
    stats = (f"{validated:,} candidates validated, {pruned:,} pruned by shared checks, "
             f"of {total:,} in the candidate set; {feasible:,} feasible.")

    if best is None:
        top = ", ".join(f"{k} ({v:,})" for k, v in list(reject.items())[:4])
        if complete:
            reasons += [f"Candidate set exhausted: no plan satisfies all hard constraints. {stats}",
                        f"Most common first-failing constraints: {top}.",
                        "This proves infeasibility only within this finite candidate set (earliest-landing slots per "
                        "order/runway, modelled gates and crews)."]
        else:
            reasons += [f"Search stopped before exhausting the candidate set (budget: {time_budget_s:g}s"
                        f"{'' if max_evaluations is None else f', {max_evaluations} evaluations'}); infeasibility is NOT proven. {stats}"]
        return res

    key, plan, bd = best
    final = validate_plan(scenario, plan)  # re-validate with every constraint + landing stage, recording passes
    if not final.passed:  # defensive: cannot happen unless validator stages disagree
        res.reasons = [f"Selected plan failed final re-validation ({final.first_failure_id}); refusing to return it."]
        res.validation = final
        return res
    res.outcome, res.plan, res.validation = Outcome.FALLBACK_SELECTED, plan, final
    res.cost, res.cost_breakdown = key[0], bd
    reasons.append(stats)
    reasons.append("Hard constraints were applied as filters; costs rank only plans that passed all "
                   f"{len(final.results)} checks. Final re-validation: PASS.")
    reasons.append(f"Selected the lowest total cost {key[0]:g} ("
                   + ", ".join(f"{k} {v:g}" for k, v in bd.items()) + ").")
    if runner_up_cost is not None and abs(runner_up_cost - key[0]) < 1e-9:
        reasons.append("Tied on cost with another feasible plan; broke the tie by smaller worst-case landing delay, "
                       "then by fixed enumeration order (landing order, runways, gates, crews).")
    elif runner_up_cost is not None:
        reasons.append(f"Next-best feasible cost was {runner_up_cost:g}.")
    if not complete:
        reasons.append(f"Search TIMED OUT: this is the best of the {validated:,} candidates evaluated, not proven optimal.")
    return res
