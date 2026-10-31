import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airport_ops import data, eventlog
from airport_ops.workbook import load_workbook_rules
from airport_ops.scenario import build_scenario


def test_default_flight_and_airport_lookup():
    df = data.load_flights()
    assert ((df.WEATHER_DELAY > 15) | (df.LATE_AIRCRAFT_DELAY > 20)).all()
    row = df.loc[data.default_index(df)]
    assert (row.FLIGHT_LABEL, row.ORIGIN_AIRPORT, row.DESTINATION_AIRPORT, str(row.FLIGHT_DATE)) == \
        ("DL1205", "PHX", "ATL", "2015-01-01")
    h = data.to_historical(row)
    assert h.airline_name is None and h.airline_code == "DL"
    assert "Phoenix" in h.origin_name and "Atlanta" in h.destination_name


def test_scenario_reproducible_and_uses_workbook_capacity():
    rules = load_workbook_rules().rules
    df = data.load_flights()
    h = data.to_historical(df.loc[data.default_index(df)])
    a, b = build_scenario(h, rules), build_scenario(h, rules)
    assert a == b and a.scenario_version == b.scenario_version
    ap = rules.airports["ATL"]
    sub = a.workbook.capacity_subset
    assert (sub.total_runways, sub.total_gates) == (ap.active_runways, ap.gates) == (4, 150)
    assert (sub.modelled_runways, sub.modelled_gates) == (2, 3)
    assert len(a.simulated.flights) == 3


def test_eventlog_dedupe():
    p = Path(tempfile.mkdtemp()) / "e.db"
    assert eventlog.log_event("x", "t", {"a": 1}, path=p, dedupe_key="k")
    assert not eventlog.log_event("x", "t", {"a": 1}, path=p, dedupe_key="k")
    assert eventlog.log_event("x", "t", {"a": 2}, path=p)  # no key -> always written
    assert len(eventlog.read_events("x", path=p)) == 2
