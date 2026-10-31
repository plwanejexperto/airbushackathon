import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airport_ops import data
from airport_ops.plan import FlightPlan, make_plan
from airport_ops.scenario import build_scenario
from airport_ops.workbook import AircraftClass, WakeCategory, load_workbook_rules


@pytest.fixture(scope="session")
def rules():
    res = load_workbook_rules()
    assert res.usable, [str(i) for i in res.issues]
    return res.rules


@pytest.fixture(scope="session")
def flights_df():
    return data.load_flights()


@pytest.fixture()
def scenario(rules, flights_df):
    """The default seeded scenario (DL1205 PHX->ATL, 2015-01-01)."""
    return build_scenario(data.to_historical(flights_df.loc[data.default_index(flights_df)]), rules)


@pytest.fixture()
def calm(scenario):
    """Same scenario with every simulated disruption removed and fixed, round numbers, so a valid plan
    can be written by hand: all three flights ETA t+20, Large/Heavy/Large, ample fuel/duty/shift."""
    sc = scenario.model_copy(deep=True)
    s = sc.simulated
    for r in s.runways:
        r.closures = []
    for x in s.taxi_segments:
        x.blockages = []
    for g in s.gates:
        g.resident_occupancy = []
    cats = [WakeCategory.LARGE, WakeCategory.HEAVY, WakeCategory.LARGE]
    for f, cat in zip(s.flights, cats):
        f.wake_category, f.earliest_landing_min = cat, 20.0
        f.aircraft_class = AircraftClass.WIDEBODY if cat == WakeCategory.HEAVY else AircraftClass.NARROWBODY
        f.service_task_min = 30.0
        f.fuel.spare_fuel_min_at_eta = 40.0
        rate = 6500 if cat == WakeCategory.HEAVY else 2400
        f.fuel.trip_fuel_remaining_kg = 1000.0
        f.fuel.fuel_on_board_kg = 1000 * 1.05 + rate * (30 + 40) / 60
    for c in s.flight_crews:
        c.report_time_min = -600.0   # limit t+180
    for g in s.ground_crews:
        g.shift_start_min, g.available_from_min = -50.0, 0.0  # shift ends t+430
    return sc


def route(sc, rw, gate, via="TWY-C"):
    n = rw[-1]
    return [f"TWY-A{n}", via, f"TWY-G{gate[-1]}"]


def mk(sc, fid, rw, land, gate, crew, taxi_start, via="TWY-C", service_start=None):
    """One hand-computed FlightPlan: taxi_end = start + sum(segments); service per deboard/task/turnaround."""
    s, a = sc.simulated, sc.simulated.assumptions
    segs = route(sc, rw, gate, via)
    total = sum(x.minutes for x in s.taxi_segments if x.id in segs)
    sf = next(f for f in s.flights if f.flight_id == fid)
    gate_in = taxi_start + total
    ss = gate_in + a.deboard_min if service_start is None else service_start
    se = max(ss + sf.service_task_min, gate_in + sc.workbook.airport.min_turnaround_min)
    return FlightPlan(flight_id=fid, runway_id=rw, landing_min=land, taxi_route=segs, taxi_start_min=taxi_start,
                      taxi_end_min=gate_in, gate_id=gate, gate_in_min=gate_in, service_start_min=ss, service_end_min=se,
                      ground_crew_id=crew, flight_crew_id=sf.flight_crew_id)


@pytest.fixture()
def good(calm):
    """Valid by construction, written independently of Planner/fallback. Taxi is strictly sequential so no
    segment is ever shared; F1 and F3 share RWY-1 (2 min apart) and ground crew GC-1 (back to back)."""
    sc = calm
    f1, f2, f3 = (f.flight_id for f in sc.simulated.flights)
    p1 = mk(sc, f1, "RWY-1", 20.0, "GATE-1", "GC-1", 21.5)
    p3 = mk(sc, f3, "RWY-1", 22.0, "GATE-3", "GC-1", p1.gate_in_min,
            service_start=p1.service_end_min + sc.simulated.assumptions.ground_crew_transfer_min)
    p2 = mk(sc, f2, "RWY-2", 20.0, "GATE-2", "GC-2", p3.gate_in_min)
    return sc, make_plan(sc, "test", [p1, p2, p3])


@pytest.fixture(autouse=True)
def _never_read_real_dotenv(monkeypatch, tmp_path):
    from airport_ops import llm
    monkeypatch.setenv("NEGOTIATION_ENV_FILE", str(tmp_path / "no-such.env"))
    llm.LOADED_FROM_FILE.clear()
