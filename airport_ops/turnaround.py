"""Scenario 4: TURNAROUND + DEPARTURE.

Input: an agreed ARRIVAL plan (landing, runway, taxi, gate, gate arrival are FIXED) and each aircraft's NEXT scheduled flight.
Negotiated: ground crew per aircraft, priority for shared equipment, whether independent tasks overlap, the departure runway per
aircraft; the deterministic timeline tool then derives task times, ready times, push-back/taxi-out and take-off slots.

Work content is SIMULATED (the workbook only gives a minimum turnaround) and labelled so. Departure runway separation uses the
workbook's `min_time_s (same runway departures)` for the first time (before, it was only an arrival floor).
"""
from __future__ import annotations

import pandas as pd

from .plan import FlightPlan, RecoveryPlan, TaskPlan, make_plan
from .planner import Planner
from .scenario import rehash_scenario
from .schemas import (ArrivalFix, DepartureContext, EquipmentUnit, Interval, NextFlight, Scenario, TaskDef, TaxiSegment,
                      TurnaroundModel)
from .validate import TOL
from .workbook import AircraftClass

DEFAULT_TASKS = [
    TaskDef(task_id="DEPLANE", name="Deplaning", base_min=8),
    TaskDef(task_id="UNLOAD_BAGS", name="Baggage unload", base_min=10, predecessors=["DEPLANE"], equipment="BAGGAGE_TRACTOR"),
    TaskDef(task_id="CLEAN", name="Cabin cleaning", base_min=12, predecessors=["DEPLANE"]),
    TaskDef(task_id="REFUEL", name="Refuelling (no boarding during refuelling)", base_min=15, predecessors=["DEPLANE"], equipment="FUEL_TRUCK"),
    TaskDef(task_id="LOAD_BAGS", name="Baggage load", base_min=10, predecessors=["UNLOAD_BAGS"], equipment="BAGGAGE_TRACTOR"),
    TaskDef(task_id="BOARD", name="Boarding", base_min=15, predecessors=["CLEAN", "REFUEL"]),
    TaskDef(task_id="CHECKS", name="Final checks / doors / cabin secure", base_min=5, predecessors=["BOARD", "LOAD_BAGS"]),
]
DEFAULT_EQUIPMENT = [EquipmentUnit(id="FUEL-TRUCK-1", type="FUEL_TRUCK"), EquipmentUnit(id="TRACTOR-1", type="BAGGAGE_TRACTOR"),
                     EquipmentUnit(id="TRACTOR-2", type="BAGGAGE_TRACTOR")]


def default_turnaround_model() -> TurnaroundModel:
    return TurnaroundModel(tasks=DEFAULT_TASKS, equipment=DEFAULT_EQUIPMENT)


def task_minutes(sc: Scenario, flight_id: str, gate_id: str) -> dict[str, float]:
    """Duration of every task for this aircraft at this gate/stand (widebody scaling; bussing at a remote stand)."""
    dep, a = sc.simulated.departure, sc.simulated.assumptions
    sf = next(f for f in sc.simulated.flights if f.flight_id == flight_id)
    g = next(g for g in sc.simulated.gates if g.id == gate_id)
    k = dep.turnaround.widebody_factor if sf.aircraft_class == AircraftClass.WIDEBODY else 1.0
    bus = a.remote_stand_bus_min if g.kind == "remote_stand" else 0.0
    return {t.task_id: round(t.base_min * k + (bus if t.task_id in ("DEPLANE", "BOARD") else 0.0), 3) for t in dep.turnaround.tasks}


# ----------------------------------------------------------------------------- scenario construction
def _hhmm_to_min(v: int) -> int:
    return (int(v) // 100) * 60 + int(v) % 100


def next_flights_for(sc: Scenario, df: pd.DataFrame | None) -> list[NextFlight]:
    """The aircraft's next scheduled flight. REAL (flights.csv) for the selected flight when the same tail has a later same-day flight
    leaving from this airport; otherwise SIMULATED and labelled."""
    ap = sc.workbook.airport
    min_turn = ap.min_turnaround_min
    h = sc.historical
    out: list[NextFlight] = []
    real = None
    if df is not None and h.tail_number and h.scheduled_arrival:
        hh, mm = h.scheduled_arrival.split(":")
        sched_arr = int(hh) * 60 + int(mm)
        rows = df[(df.TAIL_NUMBER == h.tail_number) & (df.YEAR == h.flight_date.year) & (df.MONTH == h.flight_date.month)
                  & (df.DAY == h.flight_date.day) & (df.ORIGIN_AIRPORT == h.destination_code)]
        rows = rows[rows.SCHEDULED_DEPARTURE.map(_hhmm_to_min) > sched_arr].sort_values("SCHEDULED_DEPARTURE")
        if len(rows):
            r = rows.iloc[0]
            real = NextFlight(flight_id=h.label, next_flight_label=f"{r.AIRLINE}{r.FLIGHT_NUMBER}", destination=str(r.DESTINATION_AIRPORT),
                              planned_ground_min=float(max(_hhmm_to_min(r.SCHEDULED_DEPARTURE) - sched_arr, min_turn)),
                              scheduled_block_min=float(r.SCHEDULED_TIME), source="flights.csv",
                              note="same tail, same day, departs from this airport (flights.csv)")
    for i, f in enumerate(sc.simulated.flights):
        if i == 0 and real is not None:
            out.append(real)
            continue
        wide = f.aircraft_class == AircraftClass.WIDEBODY
        out.append(NextFlight(flight_id=f.flight_id, next_flight_label=f"SIM-NEXT-{i + 1}", destination=None,
                              planned_ground_min=float(min_turn + 40 + 15 * i), scheduled_block_min=150.0 if wide else 90.0,
                              source="SIMULATED", note="no usable same-day next flight in flights.csv; planned ground time and block time assumed"))
    return out


def add_outbound_taxi(s) -> None:
    """Outbound (push-back -> runway) segments mirroring the inbound graph: GATE-g -> APRON -> HUB -> DEP-<runway>. SIMULATED."""
    if any(x.id.endswith("-OUT") for x in s.taxi_segments):
        return
    out = []
    for x in list(s.taxi_segments):
        if x.from_node.startswith("EXIT-"):
            out.append(TaxiSegment(id=f"{x.id}-OUT", from_node="HUB", to_node=f"DEP-{x.from_node[5:]}", minutes=x.minutes))
        elif x.from_node == "HUB" and x.to_node == "APRON":
            out.append(TaxiSegment(id=f"{x.id}-OUT", from_node="APRON", to_node="HUB", minutes=x.minutes))
        elif x.to_node.startswith("GATE-"):
            out.append(TaxiSegment(id=f"{x.id}-OUT", from_node=x.to_node, to_node="APRON", minutes=x.minutes))
    s.taxi_segments.extend(out)


def build_departure_scenario(arr_sc: Scenario, arrival_plan: RecoveryPlan, df: pd.DataFrame | None = None, *,
                             plan_source: str = "") -> Scenario:
    """Scenario for the turnaround + departure session, built on an agreed arrival plan."""
    sc = arr_sc.model_copy(deep=True)
    s = sc.simulated
    add_outbound_taxi(s)
    ref = {p.flight_id: p for p in Planner(arr_sc).naive_plan().flights}       # nominal (pre-disruption) gate arrival
    nxt = next_flights_for(sc, df)
    sched = {n.flight_id: round(ref[n.flight_id].gate_in_min + n.planned_ground_min, 3) for n in nxt}
    fixes = [ArrivalFix(flight_id=p.flight_id, landing_min=p.landing_min, runway_id=p.runway_id, taxi_route=list(p.taxi_route),
                        taxi_start_min=p.taxi_start_min, taxi_end_min=p.taxi_end_min, gate_id=p.gate_id, gate_in_min=p.gate_in_min,
                        hold_area_id=p.hold_area_id, ground_crew_id=p.ground_crew_id) for p in arrival_plan.flights]
    fcw = sc.workbook.fuel_crew
    for c in s.flight_crews:      # SIMULATED: duty left is raised only as far as needed for the NEXT flight to be legal on the scheduled timeline
        nf = next(n for n in nxt if n.flight_id == c.home_flight_id)
        need = sched[c.home_flight_id] + nf.scheduled_block_min + s.assumptions.flight_crew_postflight_min + 45
        limit = c.report_time_min + fcw.flight_crew_max_duty_h * 60
        if limit < need:
            c.report_time_min += need - limit
    s.departure = DepartureContext(turnaround=default_turnaround_model(), next_flights=nxt, arrival=fixes, scheduled_departure_min=sched,
                                   arrival_plan_source=plan_source,
                                   note="turnaround work content, equipment and push-back/taxi-out are SIMULATED; next flights from flights.csv when available")
    return rehash_scenario(sc)


# ----------------------------------------------------------------------------- outbound routes
def out_routes(sc: Scenario) -> dict[tuple[str, str], list[tuple[list[str], float]]]:
    segs = {x.id: x for x in sc.simulated.taxi_segments if x.id.endswith("-OUT")}
    adj: dict[str, list] = {}
    for x in sorted(segs.values(), key=lambda x: x.id):
        adj.setdefault(x.from_node, []).append(x)
    routes: dict[tuple[str, str], list] = {}

    def dfs(node, path, seen, gate):
        for x in adj.get(node, []):
            if x.to_node in seen:
                continue
            p2 = path + [x.id]
            if x.to_node.startswith("DEP-"):
                routes.setdefault((gate, x.to_node[4:]), []).append((p2, sum(segs[i].minutes for i in p2)))
            else:
                dfs(x.to_node, p2, seen | {x.to_node}, gate)

    for g in sc.simulated.gates:
        dfs(f"GATE-{g.id}", [], {f"GATE-{g.id}"}, g.id)
    for k in routes:
        routes[k].sort(key=lambda r: (r[1], r[0]))
    return routes


# ----------------------------------------------------------------------------- builder
def _free_slot(busy: list[tuple[float, float]], start: float, dur: float) -> float:
    """Earliest t >= start such that [t, t+dur] overlaps nothing in `busy`."""
    t = start
    for _ in range(200):
        hit = next(((b0, b1) for b0, b1 in sorted(busy) if t < b1 - TOL and t + dur > b0 + TOL), None)
        if hit is None:
            return t
        t = hit[1]
    return t


def build_turnaround_plan(sc: Scenario, spec, source: str = "fallback") -> RecoveryPlan | None:
    """Deterministic timeline from the discrete choices in `spec` (PlanSpec: ground_crews, equipment_order, overlap, dep_runways)."""
    s, a, dep = sc.simulated, sc.simulated.assumptions, sc.simulated.departure
    tm, ap, wake = dep.turnaround, sc.workbook.airport, sc.workbook.wake
    fixed = {x.flight_id: x for x in dep.arrival}
    flights = {f.flight_id: f for f in s.flights}
    tdefs = {t.task_id: t for t in tm.tasks}
    units_of = {}
    for u in tm.equipment:
        units_of.setdefault(u.type, []).append(u.id)
    unit_busy: dict[str, list] = {u.id: [] for u in tm.equipment}
    crews = {c.id: c for c in s.ground_crews}
    crew_free = {c.id: c.available_from_min for c in s.ground_crews}
    plans: dict[str, FlightPlan] = {}
    order = list(spec.equipment_order) if spec.equipment_order else list(fixed)
    if sorted(order) != sorted(fixed):
        return None
    for fid in order:
        af, sf = fixed[fid], flights[fid]
        cid = spec.ground_crews.get(fid)
        if cid not in crews:
            return None
        durs = task_minutes(sc, fid, af.gate_id)
        base = max(af.gate_in_min, crew_free[cid])
        ends: dict[str, float] = {}
        tasks: list[TaskPlan] = []
        cursor = base
        for t in tm.tasks:
            ready = max([base] + [ends[p] for p in t.predecessors])
            if spec.overlap == "sequential":
                ready = max(ready, cursor)
            unit = None
            if t.equipment:
                best = None
                for uid in units_of.get(t.equipment, []):
                    st = _free_slot(unit_busy[uid], ready, durs[t.task_id])
                    if best is None or st < best[0] - TOL:
                        best = (st, uid)
                ready, unit = best
                unit_busy[unit].append((ready, ready + durs[t.task_id]))
            ends[t.task_id] = ready + durs[t.task_id]
            cursor = ends[t.task_id]
            tasks.append(TaskPlan(task_id=t.task_id, start_min=ready, end_min=ends[t.task_id], equipment_id=unit))
        first = min(x.start_min for x in tasks)
        ready_t = max(max(ends.values()), af.gate_in_min + ap.min_turnaround_min)
        crew_free[cid] = ready_t + a.ground_crew_transfer_min
        plans[fid] = FlightPlan(
            flight_id=fid, runway_id=af.runway_id, landing_min=af.landing_min, taxi_route=list(af.taxi_route),
            taxi_start_min=af.taxi_start_min, taxi_end_min=af.taxi_end_min, gate_id=af.gate_id, gate_in_min=af.gate_in_min,
            service_start_min=first, service_end_min=ready_t, ground_crew_id=cid, flight_crew_id=sf.flight_crew_id,
            hold_area_id=af.hold_area_id, tasks=tasks, ready_min=ready_t)

    # ---- departures: in order of ready time, on the chosen runway, taxi-out around blockages and other departures
    routes = out_routes(sc)
    segs = {x.id: x for x in s.taxi_segments}
    seg_use: dict[str, list] = {}
    roll = dep.departure_roll_min
    arr_occ = [(x.runway_id, x.landing_min, x.landing_min + a.runway_occupancy_min) for x in dep.arrival]
    movements = [x.landing_min for x in dep.arrival]
    deps_done: list[tuple[str, str, float]] = []        # (runway, flight, take-off time)
    from .capacity import airport_limit, per_runway_limit
    cap_air, cap_rw = airport_limit(sc), per_runway_limit(sc)
    for fid in sorted(plans, key=lambda f: (plans[f].ready_min, f)):
        p = plans[fid]
        rid = spec.dep_runways.get(fid)
        rw = next((r for r in s.runways if r.id == rid), None)
        opts = routes.get((p.gate_id, rid))
        if rw is None or not opts:
            return None
        best = None
        for rsegs, total in opts:
            st = p.ready_min + tm.pushback_min
            for _ in range(200):
                s0, off = st, 0.0
                for sid in rsegs:
                    m = segs[sid].minutes
                    for b in segs[sid].blockages:
                        if st + off < b.end_min - TOL and st + off + m > b.start_min + TOL:
                            st = max(st, b.end_min - off)
                    for u0, u1 in seg_use.get(sid, []):
                        if st + off < u1 - TOL and st + off + m > u0 + TOL:
                            st = max(st, u1 - off)
                    off += m
                if abs(st - s0) < TOL:
                    break
            if best is None or st + total < best[0] - TOL:
                best = (st + total, st, rsegs, total)
        end, st, rsegs, total = best
        off = 0.0
        for sid in rsegs:
            seg_use.setdefault(sid, []).append((st + off, st + off + segs[sid].minutes))
            off += segs[sid].minutes
        t = max(end, dep.scheduled_departure_min.get(fid, end))     # never before the next flight's scheduled departure
        for _ in range(200):
            t0 = t
            for cl in rw.closures:
                if t < cl.end_min - TOL and t + roll > cl.start_min + TOL:
                    t = cl.end_min
            for r2, o0, o1 in arr_occ:                        # keep clear of arrivals on this runway
                if r2 == rid and t < o1 + dep.arrival_departure_gap_min - TOL and t + roll > o0 - dep.arrival_departure_gap_min + TOL:
                    t = o1 + dep.arrival_departure_gap_min
            prev = [d for d in deps_done if d[0] == rid]
            if prev:                                          # workbook same-runway DEPARTURE spacing (seconds), leader -> follower
                _, lead, lt = max(prev, key=lambda d: d[2])
                _, secs, _ = wake.lookup(flights[lead].wake_category, flights[fid].wake_category)
                t = max(t, lt + max(secs / 60.0, roll))
            allm = sorted(movements, reverse=True)
            if cap_air and len(allm) >= cap_air:
                t = max(t, allm[cap_air - 1] + 60.0)
            if cap_rw is not None:
                same = sorted([o0 for (r2, o0, _) in arr_occ if r2 == rid] + [d[2] for d in deps_done if d[0] == rid], reverse=True)
                if len(same) >= cap_rw:
                    t = max(t, same[cap_rw - 1] + 60.0)
            if abs(t - t0) < TOL:
                break
        deps_done.append((rid, fid, t))
        movements.append(t)
        plans[fid] = p.model_copy(update={"dep_runway_id": rid, "dep_taxi_route": rsegs, "dep_taxi_start_min": st,
                                          "dep_taxi_end_min": end, "takeoff_min": t})
    ordered = [plans[f.flight_id] for f in s.flights]
    return make_plan(sc, source, ordered)


# ----------------------------------------------------------------------------- naive reference, spec materialisation, pool, fallback
def naive_departure_plan(sc: Scenario) -> RecoveryPlan:
    """ILLUSTRATIVE reference: every aircraft turns around as if alone (first unit of each equipment type, no crew queue), departs from
    its arrival runway by the shortest outbound route the moment it is ready, with no separation applied."""
    s, dep, a = sc.simulated, sc.simulated.departure, sc.simulated.assumptions
    tm, ap = dep.turnaround, sc.workbook.airport
    first_unit = {}
    for u in tm.equipment:
        first_unit.setdefault(u.type, u.id)
    routes = out_routes(sc)
    out = []
    for af in dep.arrival:
        sf = next(f for f in s.flights if f.flight_id == af.flight_id)
        durs = task_minutes(sc, af.flight_id, af.gate_id)
        ends, tasks = {}, []
        for t in tm.tasks:
            st = max([af.gate_in_min] + [ends[p] for p in t.predecessors])
            ends[t.task_id] = st + durs[t.task_id]
            tasks.append(TaskPlan(task_id=t.task_id, start_min=st, end_min=ends[t.task_id], equipment_id=first_unit.get(t.equipment) if t.equipment else None))
        ready = max(max(ends.values()), af.gate_in_min + ap.min_turnaround_min)
        rsegs, total = routes[(af.gate_id, af.runway_id)][0]
        st = ready + tm.pushback_min
        out.append(FlightPlan(flight_id=af.flight_id, runway_id=af.runway_id, landing_min=af.landing_min, taxi_route=list(af.taxi_route),
                              taxi_start_min=af.taxi_start_min, taxi_end_min=af.taxi_end_min, gate_id=af.gate_id, gate_in_min=af.gate_in_min,
                              service_start_min=min(t.start_min for t in tasks), service_end_min=ready, ground_crew_id=af.ground_crew_id,
                              flight_crew_id=sf.flight_crew_id, hold_area_id=af.hold_area_id, tasks=tasks, ready_min=ready,
                              dep_runway_id=af.runway_id, dep_taxi_route=rsegs, dep_taxi_start_min=st, dep_taxi_end_min=st + total, takeoff_min=max(st + total, dep.scheduled_departure_min[af.flight_id])))
    return make_plan(sc, "naive_baseline", out)


def materialise_departure(sc: Scenario, spec) -> RecoveryPlan:
    ids = [f.flight_id for f in sc.simulated.flights]
    if sorted(spec.equipment_order) != sorted(ids):
        raise ValueError(f"equipment_order must be a permutation of {ids}")
    if spec.overlap not in ("max", "sequential"):
        raise ValueError("overlap must be 'max' or 'sequential'")
    for name, m, valid in (("ground_crews", spec.ground_crews, {c.id for c in sc.simulated.ground_crews}),
                           ("dep_runways", spec.dep_runways, {r.id for r in sc.simulated.runways})):
        if sorted(m) != sorted(ids):
            raise ValueError(f"{name} must have exactly the keys {ids}")
        bad = sorted(set(m.values()) - valid)
        if bad:
            raise ValueError(f"{name}: unknown ids {bad}; valid: {sorted(valid)}")
    plan = build_turnaround_plan(sc, spec, source="negotiation")
    if plan is None:
        raise ValueError("no outbound route / schedule for these choices")
    return plan


def _specs(sc: Scenario):
    import itertools
    from .messages import PlanSpec
    ids = [f.flight_id for f in sc.simulated.flights]
    rws = [r.id for r in sc.simulated.runways]
    crs = [c.id for c in sc.simulated.ground_crews]
    for order in itertools.permutations(ids):
        for overlap in ("max", "sequential"):
            for cr in itertools.product(crs, repeat=len(ids)):
                for dr in itertools.product(rws, repeat=len(ids)):
                    yield PlanSpec(equipment_order=list(order), overlap=overlap, ground_crews=dict(zip(ids, cr)), dep_runways=dict(zip(ids, dr)))


def departure_pool(sc: Scenario):
    """Every spec in the finite set whose plan passes ALL hard constraints, with metrics (cached by tools.candidate_pool)."""
    from .metrics import plan_metrics
    from .tools import Candidate
    from .validate import validate_plan
    out, seen = [], {}
    for spec in _specs(sc):
        plan = build_turnaround_plan(sc, spec, source="negotiation")
        if plan is None or plan.proposal_id in seen:
            continue
        if not validate_plan(sc, plan, stages=("ground", "depart"), record_passes=False, fail_fast=True).passed:
            continue
        seen[plan.proposal_id] = Candidate(spec, plan, plan_metrics(sc, plan))
    return list(seen.values())


def run_departure_fallback(sc: Scenario, *, time_budget_s: float = 30.0, max_evaluations=None, weights=None):
    """Deterministic non-AI fallback for turnaround + departure: enumerate, filter by HARD constraints, rank by fixed costs."""
    import time
    from collections import Counter
    from .fallback import CostWeights, FallbackResult, Outcome, SearchStatus, plan_cost
    from .validate import validate_plan
    w = weights or CostWeights()
    t0 = time.monotonic()
    total = sum(1 for _ in _specs(sc))
    reject: Counter = Counter()
    best, feasible, validated, timed_out = None, 0, 0, False
    for spec in _specs(sc):
        if (max_evaluations is not None and validated >= max_evaluations) or time.monotonic() - t0 > time_budget_s:
            timed_out = True
            break
        plan = build_turnaround_plan(sc, spec, source="fallback")
        validated += 1
        if plan is None:
            reject["DEP-TAXI-ROUTE"] += 1
            continue
        rep = validate_plan(sc, plan, stages=("ground", "depart"), record_passes=False, fail_fast=True)
        if not rep.passed:
            reject[rep.first_failure_id] += 1
            continue
        feasible += 1
        bd = plan_cost(sc, plan, w)
        cost = round(sum(bd.values()), 6)
        key = (cost, tuple(spec.equipment_order), spec.overlap, tuple(sorted(spec.ground_crews.items())), tuple(sorted(spec.dep_runways.items())))
        if best is None or key < best[0]:
            best = (key, plan, bd)
    res = FallbackResult(outcome=Outcome.NO_FEASIBLE_PLAN, search_status=SearchStatus.TIMEOUT if timed_out else SearchStatus.EXHAUSTED,
                         candidates_total=total, candidates_validated=validated, candidates_pruned=0, feasible_count=feasible,
                         rejection_by_constraint=dict(reject.most_common()))
    stats = f"{validated:,} of {total:,} turnaround/departure candidates validated; {feasible:,} feasible."
    if best is None:
        res.reasons = [("Candidate set exhausted: no turnaround/departure plan satisfies all hard constraints. " if not timed_out else
                        "Search stopped early (infeasibility NOT proven). ") + stats,
                       "Most common first-failing constraints: " + ", ".join(f"{k} ({v:,})" for k, v in list(reject.most_common(4)))]
        return res
    plan = best[1].model_copy(update={"source": "fallback"})
    final = validate_plan(sc, plan)
    if not final.passed:
        res.reasons = [f"Selected plan failed final re-validation ({final.first_failure_id}); refusing to return it."]
        return res
    res.outcome, res.plan, res.validation, res.cost, res.cost_breakdown = Outcome.FALLBACK_SELECTED, plan, final, best[0][0], best[2]
    res.reasons = [stats, "Hard constraints filtered; fixed costs ranked only plans that passed all checks. Final re-validation: PASS.",
                   f"Selected the lowest total cost {best[0][0]:g} (" + ", ".join(f"{k} {v:g}" for k, v in best[2].items()) + ")."]
    if timed_out:
        res.reasons.append("Search TIMED OUT: best of the candidates evaluated, not proven optimal.")
    return res


# ----------------------------------------------------------------------------- conflict preset for the departure session
def build_departure_demo(dsc: Scenario) -> Scenario:
    """Opt-in conflict preset (all SIMULATED, labelled DEMO): tight schedule, one crew with a short shift, one runway closed around the
    departure bank, and one aircraft whose flight crew can only just fly the next flight."""
    sc = dsc.model_copy(deep=True)
    s, dep, a = sc.simulated, sc.simulated.departure, sc.simulated.assumptions
    fc = sc.workbook.fuel_crew
    ids = [f.flight_id for f in s.flights]
    # tight scheduled departures: 8-10 min of slack over the unconstrained critical path
    for fid in ids:
        af = next(x for x in dep.arrival if x.flight_id == fid)
        dep.scheduled_departure_min[fid] = round(af.gate_in_min + sc.workbook.airport.min_turnaround_min + 14, 3)
    # GATE crews: GC-1 shift ends early -> overtime if it carries the late turnarounds
    gc1 = s.ground_crews[0]
    gc1.shift_start_min = (max(dep.scheduled_departure_min.values()) - 15) - fc.ground_shift_h * 60
    # one runway closed across the departure bank
    lo, hi = min(dep.scheduled_departure_min.values()), max(dep.scheduled_departure_min.values())
    if len(s.runways) >= 2:
        s.runways[-1].closures = [Interval(start_min=lo - 5, end_min=hi + 12, reason="DEMO: runway closed (simulated weather)")]
    # Pilot constraint: the selected aircraft's crew can fly the next flight only if it departs within ~12 min of schedule
    sel = ids[0]
    nf = next(n for n in dep.next_flights if n.flight_id == sel)
    crew = next(c for c in s.flight_crews if c.id == next(f for f in s.flights if f.flight_id == sel).flight_crew_id)
    latest = dep.scheduled_departure_min[sel] + 12
    crew.report_time_min = latest + nf.scheduled_block_min + a.flight_crew_postflight_min - fc.flight_crew_max_duty_h * 60
    for other in s.flight_crews[1:]:
        other.report_time_min = (max(dep.scheduled_departure_min.values()) + 400) - fc.flight_crew_max_duty_h * 60
    dep.note += " | DEMO preset: tight schedule, short-shift crew, runway closure, tight crew duty"
    return rehash_scenario(sc)
