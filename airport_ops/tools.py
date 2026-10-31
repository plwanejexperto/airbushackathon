"""Deterministic tools available to agents (and used by the coordinator): evaluate a plan spec, enumerate the pool of
valid candidates with consequences, and rank them by a ROLE-SPECIFIC objective.

These tools compute consequences; they do not decide. An agent still has to choose what to propose (an LLM in live
mode, a fixed scripted policy in scripted mode). The pool is NOT the fallback planner and its ranking is not the
fallback's cost: each role ranks by its own objective, which is what creates the negotiation.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

from .canonical import canonical_metrics
from .messages import AgentRole, PlanSpec
from .metrics import plan_metrics
from .plan import RecoveryPlan
from .planner import Planner
from .schemas import Scenario
from .validate import validate_plan
from .workbook import AircraftClass

SOURCE = "negotiation"  # plan.source for every negotiated proposal -> proposal id depends on content only


@dataclass(frozen=True)
class Candidate:
    spec: PlanSpec
    plan: RecoveryPlan
    metrics: dict

    @property
    def proposal_id(self) -> str:
        return self.plan.proposal_id


def spec_of(plan: RecoveryPlan) -> PlanSpec:
    order = [p.flight_id for p in sorted(plan.flights, key=lambda p: (p.landing_min, p.flight_id))]
    return PlanSpec(landing_order=order, runways={p.flight_id: p.runway_id for p in plan.flights},
                    gates={p.flight_id: p.gate_id for p in plan.flights},
                    ground_crews={p.flight_id: p.ground_crew_id for p in plan.flights}, declared_status=plan.declared_status)


def materialise(sc: Scenario, spec: PlanSpec, planner: Optional[Planner] = None) -> RecoveryPlan:
    """Deterministically build the complete timeline for discrete choices. Raises ValueError on bad ids."""
    if sc.simulated.departure is not None:
        from .turnaround import materialise_departure
        return materialise_departure(sc, spec)
    pl = planner or Planner(sc)
    ids = [f.flight_id for f in sc.simulated.flights]
    if sorted(spec.landing_order) != sorted(ids):
        raise ValueError(f"landing_order must be a permutation of {ids}")
    for name, m, valid in (("runways", spec.runways, {r.id for r in sc.simulated.runways}),
                           ("gates", spec.gates, {g.id for g in sc.simulated.gates}),
                           ("ground_crews", spec.ground_crews, {c.id for c in sc.simulated.ground_crews})):
        if sorted(m) != sorted(ids):
            raise ValueError(f"{name} must have exactly the keys {ids}")
        bad = sorted(set(m.values()) - valid)
        if bad:
            raise ValueError(f"{name}: unknown ids {bad}; valid: {sorted(valid)}")
    plan = pl.build(tuple(spec.landing_order), dict(spec.runways), dict(spec.gates), dict(spec.ground_crews), source=SOURCE)
    if plan is None:
        raise ValueError("no taxi route / schedule horizon exceeded for these choices")
    dv = sc.simulated.diversion
    if dv is not None:   # the declared status is the AGENT's claim (validated against the computed status), not copied from the truth
        from .plan import make_plan
        plan = make_plan(sc, SOURCE, plan.flights, declared_status=spec.declared_status, airport=dv.alternate_airport)
    return plan


# ----- role objectives: lexicographic keys over plan metrics (smaller is better) -----
def role_key(role: AgentRole, m: dict, selected_id: str) -> tuple:
    """`selected_id` = the flight a PILOT agent represents (ignored by ATC / GROUND)."""
    p, sel = m["plan"], m["flights"][selected_id]
    if "weighted_departure_delay" in p:      # turnaround + departure sessions
        if role == AgentRole.ATC:            # throughput: weighted departure delay, then total
            return (round(p["weighted_departure_delay"], 6), round(p["total_departure_delay_min"], 6), round(p["total_cost"], 6))
        if role == AgentRole.PILOT:          # my aircraft: its departure delay, then its ready time
            return (round(sel.get("departure_delay_min", 0.0), 6), round(sel.get("ready_min", 0.0), 6), round(sel.get("takeoff_min", 0.0), 6))
        return (round(p["ground_overtime_min"], 6), round(p["total_service_wait_min"], 6), round(p["weighted_departure_delay"], 6),
                round(p["total_cost"], 6))
    if role == AgentRole.ATC:      # traffic flow: weighted delay, then fewer runway changes
        return (round(p["weighted_landing_delay"], 6), p["runway_changes"], round(p["total_cost"], 6))
    if role == AgentRole.PILOT:    # selected aircraft: its delay, connection, earliest gate, more fuel margin
        return (round(sel["landing_delay_min"], 6), round(sel["connection_miss_min"], 6), round(sel["gate_in_min"], 6),
                -round(sel["contingency_margin_kg"], 3))
    # GROUND: overtime, waiting for crews/gates, gate changes
    return (round(p["ground_overtime_min"], 6), round(p["total_service_wait_min"], 6), p["gate_changes"],
            round(p["total_cost"], 6))


_POOLS: dict[str, list[Candidate]] = {}


def candidate_pool(sc: Scenario) -> list[Candidate]:
    """All plans from the finite candidate set that pass EVERY hard constraint, with metrics. Cached per scenario
    version. Order is deterministic (itertools order)."""
    if sc.scenario_version in _POOLS:
        return _POOLS[sc.scenario_version]
    if sc.simulated.departure is not None:
        from .turnaround import departure_pool
        _POOLS[sc.scenario_version] = departure_pool(sc)
        return _POOLS[sc.scenario_version]
    pl = Planner(sc)
    sim = sc.simulated
    fids = [f.flight_id for f in sim.flights]
    gate_opts = [[g.id for g in sim.gates if f.aircraft_class == AircraftClass.NARROWBODY
                  or g.max_aircraft_class == AircraftClass.WIDEBODY] for f in sim.flights]
    out: list[Candidate] = []
    for order in itertools.permutations(fids):
        for rwa in itertools.product([r.id for r in sim.runways], repeat=len(fids)):
            rwd = dict(zip(fids, rwa))
            for gts in itertools.product(*gate_opts):
                for crs in itertools.product([c.id for c in sim.ground_crews], repeat=len(fids)):
                    plan = pl.build(order, rwd, dict(zip(fids, gts)), dict(zip(fids, crs)), source=SOURCE)
                    if plan is None:
                        continue
                    if not validate_plan(sc, plan, stages=("land", "fuel", "ground"), record_passes=False, fail_fast=True).passed:
                        continue
                    out.append(Candidate(spec_of(plan), plan, plan_metrics(sc, plan)))
    # identical plans reached through different choices collapse to one candidate (same content hash)
    seen: dict[str, Candidate] = {}
    for c in out:
        seen.setdefault(c.proposal_id, c)
    _POOLS[sc.scenario_version] = list(seen.values())
    if len(_POOLS) > 8:
        _POOLS.pop(next(iter(_POOLS)))
    return _POOLS[sc.scenario_version]


def best_for(role: AgentRole, sc: Scenario, pool: Optional[list[Candidate]] = None, *, where=None,
             flight: Optional[str] = None) -> Optional[Candidate]:
    pool = candidate_pool(sc) if pool is None else pool
    sel = flight or sc.simulated.flights[0].flight_id
    ok = [c for c in pool if where is None or where(c)]
    return min(ok, key=lambda c: (role_key(role, c.metrics, sel), c.proposal_id), default=None)


def option_menu(role: AgentRole, sc: Scenario, k: int = 5, flight: Optional[str] = None) -> list[dict]:
    """Top-k valid options by this role's objective, with their consequences (tool output shown to live agents)."""
    sel = flight or sc.simulated.flights[0].flight_id
    pool = sorted(candidate_pool(sc), key=lambda c: (role_key(role, c.metrics, sel), c.proposal_id))
    return [{"proposal_id_if_proposed": c.proposal_id, "plan_spec": c.spec.model_dump(), "valid": True,
             "consequences": {**canonical_metrics(c.metrics)["plan"], "selected_flight": canonical_metrics(c.metrics)["flights"][sel]}}
            for c in pool[:k]]
