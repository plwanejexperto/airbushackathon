"""Deterministic hard-constraint validation of a RecoveryPlan against a Scenario.

Works on plans from any source (naive baseline, fallback planner, future agents, pasted JSON).
Units: minutes from scenario start unless stated; seconds/nm/kg named explicitly.

Every constraint is HARD. There is no scoring here; costs live only in the fallback ranking.
"""
from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel

from .plan import FlightPlan, RecoveryPlan, ResourceInterval, occupancy_intervals
from .capacity import airport_limit, per_runway_limit
from .fuel import fuel_account
from .schemas import Scenario
from .workbook import AircraftClass

TOL = 1e-6

# constraint id -> (title, basis). basis names the workbook cell(s) or the labelled SimAssumptions field.
CONSTRAINTS: dict[str, tuple[str, str]] = {
    "PLAN-VERSION": ("Plan targets this scenario version", "structural"),
    "PLAN-COMPLETE": ("Exactly one plan entry per scenario flight", "structural"),
    "LAND-EARLIEST": ("Landing not before nominal ETA", "simulated: SimFlight.earliest_landing_min"),
    "RWY-AVAIL": ("Runway exists and is not closed during roll-out", "simulated: runway closures"),
    "RWY-OCC": ("No overlapping runway occupancy", "assumption: runway_occupancy_min"),
    "WAKE-SEP": ("Wake separation leader->follower on the same runway",
                 "workbook: wake_separation (nm,s) + assumption: approach_speed_kt, wake_apply_time_floor"),
    "RWY-CAP-AIRPORT": ("Movements in any 60-min window <= max_runway_ops_per_hour",
                        "workbook: airport_capacity.max_runway_ops_per_hour (read as airport-wide)"),
    "RWY-CAP-RUNWAY": ("Movements per runway in any 60-min window <= assumed per-runway share (skipped when disabled)",
                       "assumption: per_runway_share_mode (even_split | fixed | disabled)"),
    "TAXI-ROUTE": ("Route is connected from the runway exit to the gate", "simulated: taxi graph"),
    "TAXI-TIME": ("Taxi duration >= sum of segment times", "simulated: segment minutes"),
    "TAXI-BLOCK": ("No taxi segment used while blocked", "simulated: taxiway blockages"),
    "TAXI-CONFLICT": ("One aircraft per taxi segment at a time", "assumption: single-lane segments"),
    "GATE-COMPAT": ("Gate accepts the aircraft class", "simulated: gate max_aircraft_class"),
    "GATE-OVERLAP": ("No overlapping gate occupancy (incl. resident aircraft + buffer)",
                     "simulated: resident occupancy + assumption: gate_buffer_min"),
    "GATE-NOTICE": ("Changed gate is notified >= N min before gate arrival",
                    "workbook: fuel_and_crew.Gate change notice"),
    "CREW-FLIGHT-DUTY": ("Flight crew within max duty from report time",
                         "workbook: fuel_and_crew.Flight crew max duty + assumption: flight_crew_postflight_min"),
    "CREW-GROUND-SHIFT": ("Ground-crew overtime within cap",
                          "workbook: fuel_and_crew.Ground crew shift + assumption: max_ground_overtime_min"),
    "CREW-GROUND-AVAIL": ("Ground crew exists and is available at service start", "simulated: available_from_min"),
    "CREW-OVERLAP": ("No overlapping crew assignments", "simulated + assumption: ground_crew_transfer_min"),
    "TURNAROUND-MIN": ("Gate-in to service-end >= airport minimum turnaround",
                       "workbook: airport_capacity.min_turnaround_min"),
    "TASK-TIME": ("Service starts after deboarding and lasts the task content",
                  "assumption: deboard_min + simulated service_task_min"),
    "SEQ-ORDER": ("landing < taxi < gate arrival < service ordering is consistent", "structural + assumption"),
    "FUEL-RESERVE": ("Fuel at landing >= final reserve", "workbook: Final fuel reserve, Holding fuel burn"),
    "FUEL-CONTINGENCY": ("Fuel at landing >= final reserve + contingency", "workbook: Contingency fuel (5% trip)"),
}

LANDING_STAGE = ("land", "fuel")  # checks that depend only on runway_id + landing_min
GROUND_STAGE = ("ground",)


class ConstraintResult(BaseModel):
    constraint_id: str
    passed: bool
    flight_id: Optional[str] = None
    resource_id: Optional[str] = None
    explanation: str
    basis: str = ""


class ValidationReport(BaseModel):
    proposal_id: str
    scenario_version: str
    passed: bool
    results: list[ConstraintResult]

    @property
    def failures(self) -> list[ConstraintResult]:
        return [r for r in self.results if not r.passed]

    @property
    def first_failure_id(self) -> Optional[str]:
        f = self.failures
        return f[0].constraint_id if f else None


class _Stop(Exception):
    pass


class _Collector:
    def __init__(self, record_passes: bool, fail_fast: bool):
        self.record_passes, self.fail_fast = record_passes, fail_fast
        self.results: list[ConstraintResult] = []

    def check(self, cid: str, ok: bool, msg: Callable[[], str], flight: Optional[str] = None,
              resource: Optional[str] = None) -> bool:
        if ok and not self.record_passes:
            return ok
        self.results.append(ConstraintResult(constraint_id=cid, passed=ok, flight_id=flight, resource_id=resource,
                                             explanation=msg(), basis=CONSTRAINTS[cid][1]))
        if not ok and self.fail_fast:
            raise _Stop
        return ok


def _overlap(a_start, a_end, b_start, b_end, buf=0.0) -> bool:
    return a_start < b_end + buf - TOL and b_start < a_end + buf - TOL


def max_window_count(times: list[float], window: float = 60.0) -> int:
    """Max movements inside any half-open window [t_i, t_i + window)."""
    ts = sorted(times)
    return max((sum(1 for u in ts if t - TOL <= u < t + window - TOL) for t in ts), default=0)


class _Ix:
    def __init__(self, sc: Scenario):
        s = sc.simulated
        self.sc, self.a = sc, s.assumptions
        self.ap, self.wake, self.fc = sc.workbook.airport, sc.workbook.wake, sc.workbook.fuel_crew
        self.flights = {f.flight_id: f for f in s.flights}
        self.runways = {r.id: r for r in s.runways}
        self.segs = {x.id: x for x in s.taxi_segments}
        self.seg_min = {x.id: x.minutes for x in s.taxi_segments}
        self.gates = {g.id: g for g in s.gates}
        self.fcrews = {c.id: c for c in s.flight_crews}
        self.gcrews = {c.id: c for c in s.ground_crews}


def validate_plan(scenario: Scenario, plan: RecoveryPlan, *, stages: tuple[str, ...] = ("structure", "land", "fuel", "ground"),
                  record_passes: bool = True, fail_fast: bool = False) -> ValidationReport:
    """Run the selected check stages. Default = everything, recording passes and failures."""
    ix = _Ix(scenario)
    c = _Collector(record_passes, fail_fast)
    present = [p for p in plan.flights if p.flight_id in ix.flights]
    # if a flight appears twice only its first entry is checked (PLAN-COMPLETE reports the duplicate)
    seen, fps = set(), []
    for p in present:
        if p.flight_id not in seen:
            seen.add(p.flight_id)
            fps.append(p)
    try:
        if "structure" in stages:
            _structure(ix, plan, c)
        if "land" in stages:
            _landing(ix, fps, c)
        if "fuel" in stages:
            _fuel(ix, fps, c)
        if "ground" in stages:
            _ground(ix, plan, fps, c)
    except _Stop:
        pass
    return ValidationReport(proposal_id=plan.proposal_id, scenario_version=scenario.scenario_version,
                            passed=all(r.passed for r in c.results), results=c.results)


# ----------------------------- structure -----------------------------
def _structure(ix: _Ix, plan: RecoveryPlan, c: _Collector):
    c.check("PLAN-VERSION", plan.scenario_version == ix.sc.scenario_version,
            lambda: f"plan scenario_version {plan.scenario_version!r} vs scenario {ix.sc.scenario_version!r}")
    ids = [p.flight_id for p in plan.flights]
    missing = sorted(set(ix.flights) - set(ids))
    unknown = sorted(set(ids) - set(ix.flights))
    dup = sorted({i for i in ids if ids.count(i) > 1})
    c.check("PLAN-COMPLETE", not (missing or unknown or dup),
            lambda: ("all %d flights planned once" % len(ix.flights)) if not (missing or unknown or dup) else
            f"missing={missing} unknown={unknown} duplicated={dup}")


# ----------------------------- landing stage -----------------------------
def _landing(ix: _Ix, fps: list[FlightPlan], c: _Collector):
    a = ix.a
    by_rw: dict[str, list[FlightPlan]] = {}
    for p in fps:
        sf = ix.flights[p.flight_id]
        c.check("LAND-EARLIEST", p.landing_min >= sf.earliest_landing_min - TOL,
                lambda: f"landing t+{p.landing_min:.1f} min vs earliest t+{sf.earliest_landing_min:.1f} min",
                p.flight_id)
        rw = ix.runways.get(p.runway_id)
        if rw is None:
            c.check("RWY-AVAIL", False, lambda: f"unknown runway {p.runway_id!r}", p.flight_id, p.runway_id)
            continue
        hit = [cl for cl in rw.closures if _overlap(p.landing_min, p.landing_min + a.runway_occupancy_min,
                                                    cl.start_min, cl.end_min)]
        c.check("RWY-AVAIL", not hit,
                lambda: (f"{p.runway_id} open for roll-out t+{p.landing_min:.1f}..+{p.landing_min + a.runway_occupancy_min:.1f}"
                         if not hit else f"{p.runway_id} closed t+{hit[0].start_min:.1f}..+{hit[0].end_min:.1f} "
                                         f"({hit[0].reason}); landing t+{p.landing_min:.1f}"),
                p.flight_id, p.runway_id)
        by_rw.setdefault(p.runway_id, []).append(p)

    for rid, items in by_rw.items():
        items.sort(key=lambda p: (p.landing_min, p.flight_id))
        bad = [(x, y) for i, x in enumerate(items) for y in items[i + 1:]
               if y.landing_min < x.landing_min + a.runway_occupancy_min - TOL]
        for x, y in bad:
            c.check("RWY-OCC", False, lambda: f"{x.flight_id} (t+{x.landing_min:.1f}) and {y.flight_id} "
                    f"(t+{y.landing_min:.1f}) overlap on {rid}; occupancy {a.runway_occupancy_min} min", y.flight_id, rid)
        if not bad:
            c.check("RWY-OCC", True, lambda: f"{len(items)} landing(s) on {rid} do not overlap", None, rid)
        # wake: every ordered (earlier leader, later follower) pair, rules are directional
        for i, x in enumerate(items):
            for y in items[i + 1:]:
                lc, fc_ = ix.flights[x.flight_id].wake_category, ix.flights[y.flight_id].wake_category
                nm, tmin_s, label = ix.wake.lookup(lc, fc_)
                req_s = nm / a.approach_speed_kt * 3600.0
                if a.wake_apply_time_floor:
                    req_s = max(req_s, tmin_s)
                gap_s = (y.landing_min - x.landing_min) * 60.0
                c.check("WAKE-SEP", gap_s >= req_s - TOL * 60,
                        lambda: (f"{x.flight_id}({lc.value}) -> {y.flight_id}({fc_.value}) on {rid}: gap {gap_s:.0f}s, "
                                 f"required {req_s:.0f}s [{label}: {nm:g} nm @ {a.approach_speed_kt:g} kt = "
                                 f"{nm / a.approach_speed_kt * 3600:.0f}s; workbook floor {tmin_s:g}s]"),
                        y.flight_id, rid)

    # airport-wide movement capacity: workbook max_runway_ops_per_hour (modelled flights only)
    limit = airport_limit(ix.sc)
    n = max_window_count([p.landing_min for p in fps])
    c.check("RWY-CAP-AIRPORT", n <= limit,
            lambda: f"max {n} movement(s) in any 60-min window vs airport limit {limit}/h (modelled flights only)",
            None, ix.ap.iata_code)
    # ASSUMED per-runway share (separate from the workbook value; configurable, may be disabled)
    share = per_runway_limit(ix.sc)
    if share is not None:
        for rid, items in by_rw.items():
            n = max_window_count([p.landing_min for p in items])
            c.check("RWY-CAP-RUNWAY", n <= share,
                    lambda: f"{rid}: max {n} movement(s) in any 60-min window vs assumed share {share}/h "
                            f"(mode {ix.a.per_runway_share_mode})", None, rid)


# ----------------------------- fuel -----------------------------
def _fuel(ix: _Ix, fps: list[FlightPlan], c: _Collector):
    """Thin wrapper: all arithmetic lives in fuel.fuel_account (documented there)."""
    for p in fps:
        sf = ix.flights[p.flight_id]
        fa = fuel_account(ix.sc, sf, p.landing_min)
        c.check("FUEL-RESERVE", fa.reserve_margin_kg >= -TOL,
                lambda: f"landing fuel {fa.landing_fuel_kg:.0f} kg after {fa.hold_min:.1f} min hold @ {fa.hold_burn_kg_h:.0f} kg/h "
                        f"vs final reserve {fa.final_reserve_kg:.0f} kg ({ix.fc.final_reserve_min:g} min)", p.flight_id)
        c.check("FUEL-CONTINGENCY", fa.contingency_margin_kg >= -TOL,
                lambda: f"landing fuel {fa.landing_fuel_kg:.0f} kg vs reserve+contingency "
                        f"{fa.final_reserve_kg + fa.contingency_kg:.0f} kg (contingency {fa.contingency_kg:.0f} kg = "
                        f"{ix.fc.contingency_fraction:.0%} of trip {fa.trip_fuel_kg:.0f} kg)", p.flight_id)


# ----------------------------- ground stage -----------------------------
def _ground(ix: _Ix, plan: RecoveryPlan, fps: list[FlightPlan], c: _Collector):
    a, tol = ix.a, ix.a.time_tolerance_min
    ivs: list[ResourceInterval] = []
    ok_routes: set[str] = set()
    for p in fps:
        sf = ix.flights[p.flight_id]
        f = p.flight_id
        # sequencing
        errs = []
        if p.taxi_start_min < p.landing_min + a.runway_occupancy_min - tol:
            errs.append(f"taxi start t+{p.taxi_start_min:.1f} before runway vacated t+{p.landing_min + a.runway_occupancy_min:.1f}")
        if abs(p.gate_in_min - p.taxi_end_min) > tol:
            errs.append(f"gate arrival t+{p.gate_in_min:.1f} != taxi end t+{p.taxi_end_min:.1f}")
        if p.service_start_min < p.gate_in_min - tol:
            errs.append(f"service start t+{p.service_start_min:.1f} before gate arrival t+{p.gate_in_min:.1f}")
        if p.service_end_min < p.service_start_min - tol:
            errs.append("service ends before it starts")
        c.check("SEQ-ORDER", not errs, lambda: "; ".join(errs) if errs else
                f"landing t+{p.landing_min:.1f} < taxi t+{p.taxi_start_min:.1f}..{p.taxi_end_min:.1f} < "
                f"service t+{p.service_start_min:.1f}..{p.service_end_min:.1f}", f)

        # taxi route
        segs = [ix.segs.get(s) for s in p.taxi_route]
        rt_err = None
        if not p.taxi_route:
            rt_err = "empty route"
        elif any(s is None for s in segs):
            rt_err = f"unknown segment(s) {[s for s, o in zip(p.taxi_route, segs) if o is None]}"
        elif p.runway_id not in ix.runways:
            rt_err = f"unknown runway {p.runway_id!r}"
        elif p.gate_id not in ix.gates:
            rt_err = f"unknown gate {p.gate_id!r}"
        else:
            if segs[0].from_node != f"EXIT-{p.runway_id}":
                rt_err = f"route starts at {segs[0].from_node}, expected EXIT-{p.runway_id}"
            for s1, s2 in zip(segs, segs[1:]):
                if s1.to_node != s2.from_node:
                    rt_err = f"{s1.id} ends at {s1.to_node} but {s2.id} starts at {s2.from_node}"
            if rt_err is None and segs[-1].to_node != f"GATE-{p.gate_id}":
                rt_err = f"route ends at {segs[-1].to_node}, expected GATE-{p.gate_id}"
        c.check("TAXI-ROUTE", rt_err is None, lambda: rt_err or f"{' > '.join(p.taxi_route)} connects {p.runway_id} to {p.gate_id}", f)
        if rt_err is None:
            ok_routes.add(f)
            total = sum(s.minutes for s in segs)
            dur = p.taxi_end_min - p.taxi_start_min
            c.check("TAXI-TIME", dur >= total - tol,
                    lambda: f"taxi duration {dur:.1f} min vs route time {total:.1f} min", f)
            ivs.extend(i for i in occupancy_intervals(p, ix.sc, ix.seg_min))
        else:
            ivs.extend(i for i in occupancy_intervals(p, ix.sc, ix.seg_min) if i.resource_type != "taxi_segment")

        # gate compatibility + notice
        g = ix.gates.get(p.gate_id)
        if g is None:
            c.check("GATE-COMPAT", False, lambda: f"unknown gate {p.gate_id!r}", f, p.gate_id)
        else:
            compat = sf.aircraft_class == AircraftClass.NARROWBODY or g.max_aircraft_class == AircraftClass.WIDEBODY
            c.check("GATE-COMPAT", compat, lambda: f"{sf.aircraft_class.value} ({sf.wake_category.value}) at {g.id} "
                    f"(max {g.max_aircraft_class.value})", f, p.gate_id)
        if p.gate_id != sf.original_gate_id:
            notice = p.gate_in_min - plan.decision_time_min
            need = ix.fc.gate_change_notice_min
            c.check("GATE-NOTICE", notice >= need - TOL, lambda: f"gate changed {sf.original_gate_id}->{p.gate_id}; "
                    f"notice {notice:.1f} min before gate arrival vs required {need:g}", f, p.gate_id)

        # turnaround + task
        turn = p.service_end_min - p.gate_in_min
        c.check("TURNAROUND-MIN", turn >= ix.ap.min_turnaround_min - tol,
                lambda: f"gate time {turn:.1f} min vs {ix.ap.iata_code} minimum {ix.ap.min_turnaround_min} min", f)
        task_ok = (p.service_start_min >= p.gate_in_min + a.deboard_min - tol and
                   p.service_end_min - p.service_start_min >= sf.service_task_min - tol)
        c.check("TASK-TIME", task_ok, lambda: f"service t+{p.service_start_min:.1f}..{p.service_end_min:.1f} "
                f"({p.service_end_min - p.service_start_min:.1f} min) vs deboard {a.deboard_min:g} min after gate arrival "
                f"+ task {sf.service_task_min:g} min", f)

        # flight crew duty + ground crew availability
        fcrew = ix.fcrews.get(p.flight_crew_id)
        if fcrew is None:
            c.check("CREW-FLIGHT-DUTY", False, lambda: f"unknown flight crew {p.flight_crew_id!r}", f, p.flight_crew_id)
        else:
            limit = fcrew.report_time_min + ix.fc.flight_crew_max_duty_h * 60
            end = p.gate_in_min + a.flight_crew_postflight_min
            c.check("CREW-FLIGHT-DUTY", end <= limit + TOL, lambda: f"{fcrew.id} duty ends t+{end:.1f} "
                    f"(gate arrival + {a.flight_crew_postflight_min:g} min) vs limit t+{limit:.1f} "
                    f"({ix.fc.flight_crew_max_duty_h:g} h from report)", f, fcrew.id)
        gcrew = ix.gcrews.get(p.ground_crew_id)
        if gcrew is None:
            c.check("CREW-GROUND-AVAIL", False, lambda: f"unknown ground crew {p.ground_crew_id!r}", f, p.ground_crew_id)
        else:
            c.check("CREW-GROUND-AVAIL", p.service_start_min >= gcrew.available_from_min - TOL,
                    lambda: f"{gcrew.id} service starts t+{p.service_start_min:.1f}, available from t+{gcrew.available_from_min:.1f}",
                    f, gcrew.id)

    # ground-crew overtime (per crew, last job end vs shift end)
    for gid, gcrew in ix.gcrews.items():
        mine = [p for p in fps if p.ground_crew_id == gid]
        if not mine:
            continue
        shift_end = gcrew.shift_start_min + ix.fc.ground_shift_h * 60
        last = max(p.service_end_min for p in mine)
        ot = max(0.0, last - shift_end)
        c.check("CREW-GROUND-SHIFT", ot <= a.max_ground_overtime_min + TOL,
                lambda: f"{gid} shift ends t+{shift_end:.1f}; last job ends t+{last:.1f}: overtime {ot:.1f} min "
                        f"(cap {a.max_ground_overtime_min:g}, billed {ix.fc.ground_overtime_multiplier:g}x)", None, gid)

    # overlap checks over derived occupancy + simulated external occupancy
    for rid, g in ix.gates.items():
        ivs.extend(ResourceInterval(resource_type="gate", resource_id=rid, flight_id=None, start_min=i.start_min,
                                    end_min=i.end_min, purpose=i.reason) for i in g.resident_occupancy)
    _overlaps(ix, ivs, c)

    # taxi blockages
    for p in fps:
        if p.flight_id not in ok_routes:
            continue
        hits = []
        for i in occupancy_intervals(p, ix.sc, ix.seg_min):
            if i.resource_type != "taxi_segment":
                continue
            for b in ix.segs[i.resource_id].blockages:
                if _overlap(i.start_min, i.end_min, b.start_min, b.end_min):
                    hits.append(f"{i.resource_id} used t+{i.start_min:.1f}..+{i.end_min:.1f} while blocked "
                                f"t+{b.start_min:.1f}..+{b.end_min:.1f} ({b.reason})")
        c.check("TAXI-BLOCK", not hits, lambda: "; ".join(hits) if hits else "route clear of blockages", p.flight_id)


def _overlaps(ix: _Ix, ivs: list[ResourceInterval], c: _Collector):
    a = ix.a
    spec = {  # type -> (constraint, buffer)
        "taxi_segment": ("TAXI-CONFLICT", 0.0),
        "gate": ("GATE-OVERLAP", a.gate_buffer_min),
        "ground_crew": ("CREW-OVERLAP", a.ground_crew_transfer_min),
        "flight_crew": ("CREW-OVERLAP", 0.0),
    }
    groups: dict[tuple[str, str], list[ResourceInterval]] = {}
    for i in ivs:
        if i.resource_type in spec:
            groups.setdefault((i.resource_type, i.resource_id), []).append(i)
    for (rt, rid), items in sorted(groups.items()):
        cid, buf = spec[rt]
        items.sort(key=lambda i: (i.start_min, i.end_min, i.flight_id or ""))
        conflicts = [(x, y) for k, x in enumerate(items) for y in items[k + 1:]
                     if x.flight_id != y.flight_id and _overlap(x.start_min, x.end_min, y.start_min, y.end_min, buf)]
        if not conflicts:
            c.check(cid, True, lambda: f"{rt} {rid}: {len(items)} interval(s), no overlap (buffer {buf:g} min)", None, rid)
        for x, y in conflicts:
            c.check(cid, False, lambda: f"{rt} {rid}: {x.flight_id or x.purpose} t+{x.start_min:.1f}..{x.end_min:.1f} "
                    f"overlaps {y.flight_id or y.purpose} t+{y.start_min:.1f}..{y.end_min:.1f} (buffer {buf:g} min)",
                    y.flight_id or x.flight_id, rid)
