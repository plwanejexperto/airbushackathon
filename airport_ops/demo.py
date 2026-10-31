"""Demonstration preset: a reproducible variant of any scenario with CONFLICTING domain preferences, so the three-agent
negotiation is visible. Ordinary sessions are NOT forced to disagree - this is opt-in (sidebar toggle).

Edits (all simulated, all labelled DEMO in the closure/blockage reasons; the airport's workbook limits are untouched):
  ATC     : one of two runways is closed 0-60 min -> the three flights must be sequenced on one runway. ATC's objective
            (priority-weighted delay) puts the priority-5 Heavy SIM-A101 first and the selected flight last.
  PILOT   : the selected flight has only 8 min of spare holding fuel and a connecting-passenger preference; the Pilot wants
            to keep >=75 % of that fuel margin, so a late slot is unacceptable to it.
  GROUND  : the selected flight's original gate is occupied by a resident aircraft until t+50, GC-1 starts late and its shift
            ends early (overtime), and a taxiway is blocked, so the original assignments conflict.
The preset needs >=2 modelled runways (e.g. ATL); with one runway no closure is added and the conflict is weaker.
"""
from __future__ import annotations

from .scenario import fuel_state, rehash_scenario
from .schemas import ConnectionPreference, Interval, Scenario
from .workbook import AircraftClass, WakeCategory

DEMO_LABEL = "DEMONSTRATION PRESET (conflicting preferences)"
SELECTED_SPARE_MIN = 8
OTHERS_SPARE_MIN = 40


def build_demo_scenario(base: Scenario) -> Scenario:
    sc = base.model_copy(deep=True)
    s, a, fc = sc.simulated, sc.simulated.assumptions, sc.workbook.fuel_crew
    sel, heavy, large = s.flights
    eta = float(int(sel.earliest_landing_min))

    for f, cat, cls, prio, e, spare in ((sel, WakeCategory.LARGE, AircraftClass.NARROWBODY, 2, eta, SELECTED_SPARE_MIN),
                                        (heavy, WakeCategory.HEAVY, AircraftClass.WIDEBODY, 5, eta, OTHERS_SPARE_MIN),
                                        (large, WakeCategory.LARGE, AircraftClass.NARROWBODY, 3, eta + 1, OTHERS_SPARE_MIN)):
        f.wake_category, f.aircraft_class, f.priority, f.earliest_landing_min = cat, cls, prio, e
        f.fuel = fuel_state(e, cls, spare, a, fc)
        f.original_runway_id = s.runways[0].id
    sel.original_gate_id, sel.original_ground_crew_id = "GATE-1", "GC-1"
    heavy.original_gate_id, heavy.original_ground_crew_id = "GATE-2", "GC-2"
    large.original_gate_id, large.original_ground_crew_id = "GATE-3", "GC-1"

    for r in s.runways:
        r.closures = []
    if len(s.runways) >= 2:
        s.runways[-1].closures = [Interval(start_min=0, end_min=60, reason="DEMO: runway closed (simulated weather)")]
    for x in s.taxi_segments:
        x.blockages = []
        if x.id == "TWY-C":
            x.blockages = [Interval(start_min=eta + 2, end_min=eta + 18, reason="DEMO: taxiway blocked (simulated)")]
    for g in s.gates:
        g.resident_occupancy = []
    next(g for g in s.gates if g.id == "GATE-1").resident_occupancy = [
        Interval(start_min=0, end_min=50, reason="DEMO: resident aircraft on the selected flight's original gate")]
    for c in s.ground_crews:
        c.available_from_min = 0.0
    gc1, gc2 = s.ground_crews[0], s.ground_crews[1]
    gc1.available_from_min, gc1.shift_start_min = 15.0, 90 - fc.ground_shift_h * 60      # late start, shift ends t+90
    gc2.shift_start_min = 200 - fc.ground_shift_h * 60
    for c in s.flight_crews:
        c.report_time_min = 250 - fc.flight_crew_max_duty_h * 60
    s.connection = ConnectionPreference(flight_id=sel.flight_id, connecting_passengers=30, preferred_gate_in_by_min=eta + 12)
    s.connections = [s.connection,
                     ConnectionPreference(flight_id=heavy.flight_id, connecting_passengers=80, preferred_gate_in_by_min=eta + 40),
                     ConnectionPreference(flight_id=large.flight_id, connecting_passengers=20, preferred_gate_in_by_min=eta + 45)]
    return rehash_scenario(sc)


def is_demo_scenario(sc: Scenario) -> bool:
    s = sc.simulated
    reasons = [i.reason for r in s.runways for i in r.closures] + [i.reason for x in s.taxi_segments for i in x.blockages] \
        + [i.reason for g in s.gates for i in g.resident_occupancy]
    return any(r.startswith("DEMO:") for r in reasons) or bool(sc.simulated.departure and "DEMO preset" in sc.simulated.departure.note)
