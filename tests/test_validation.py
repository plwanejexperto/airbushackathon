from conftest import mk
from airport_ops.schemas import Interval
from airport_ops.validate import CONSTRAINTS
from airport_ops.workbook import AircraftClass as AC
from airport_ops.plan import RecoveryPlan, make_plan
from airport_ops.validate import validate_plan
from airport_ops.workbook import WakeCategory as W


def fails(sc, plan):
    return {r.constraint_id for r in validate_plan(sc, plan).failures}


def edit(plan, fid, **kw):
    p = plan.model_copy(deep=True)
    for f in p.flights:
        if f.flight_id == fid:
            for k, v in kw.items():
                setattr(f, k, v)
    return p


def test_valid_plan_passes_every_constraint(good):
    sc, plan = good
    rep = validate_plan(sc, plan)
    assert rep.passed, [(r.constraint_id, r.flight_id, r.explanation) for r in rep.failures]
    ids = {r.constraint_id for r in rep.results}
    not_arrival = {c for c in CONSTRAINTS if c.startswith(("DEP-", "DIV-", "EMERG", "FUEL-EMERGENCY")) or c in ("TASK-SET", "TASK-DUR", "TASK-DEPS", "TASK-EQUIP", "READY-TIME")}
    # notice: only if a gate changes; hold checks: only if waiting areas exist; departure/diversion groups: only in those session kinds
    assert ids == set(CONSTRAINTS) - {"GATE-NOTICE", "HOLD-AREA", "HOLD-CAP"} - not_arrival
    assert all(r.explanation and r.basis for r in rep.results)


def test_wake_separation_is_ordered_and_uses_workbook_values(good):
    sc, plan = good
    f1, _, f3 = (f.flight_id for f in sc.simulated.flights)
    # F1 Heavy leader -> F3 Large follower, 2.0 min = 120 s apart: Heavy->Large needs 5 nm @140 kt = 128.6 s
    sc.simulated.flights[0].wake_category = W.HEAVY
    sc.simulated.flights[0].aircraft_class = AC.WIDEBODY
    assert "WAKE-SEP" in fails(sc, plan)
    # F3 lands 2.2 min (132 s) behind: passes
    assert "WAKE-SEP" not in fails(sc, edit(plan, f3, landing_min=22.2))
    # reverse order (Large leader -> Heavy follower) uses the 'Any other pair' rule: 3 nm = 77 s
    sc.simulated.flights[0].wake_category, sc.simulated.flights[2].wake_category = W.LARGE, W.HEAVY
    sc.simulated.flights[2].aircraft_class = AC.WIDEBODY
    assert "WAKE-SEP" not in fails(sc, plan)
    r = [x for x in validate_plan(sc, plan).results if x.constraint_id == "WAKE-SEP"][0]
    assert "Any other pair" in r.explanation and r.flight_id == f3


def test_runway_closure_and_unknown_runway(good):
    sc, plan = good
    f1 = sc.simulated.flights[0].flight_id
    sc.simulated.runways[0].closures.append(Interval(start_min=19, end_min=30, reason="x"))
    r = [x for x in validate_plan(sc, plan).failures if x.constraint_id == "RWY-AVAIL"]
    assert r and r[0].resource_id == "RWY-1" and "closed" in r[0].explanation
    assert "RWY-AVAIL" in fails(sc, edit(plan, f1, runway_id="RWY-9"))  # unknown id: no crash


def test_runway_movement_capacity(good):
    sc, plan = good
    sc.workbook.airport = sc.workbook.airport.model_copy(update={"max_runway_ops_per_hour": 2})
    assert "RWY-CAP-AIRPORT" in fails(sc, plan)  # 3 movements within 60 min vs 2/h
    sc.workbook.airport = sc.workbook.airport.model_copy(update={"max_runway_ops_per_hour": 4 * 1})
    assert "RWY-CAP-RUNWAY" in fails(sc, plan)   # share = 4 // 4 runways = 1/h; RWY-1 has 2 landings


def test_gate_conflict_and_resident_aircraft(good):
    sc, plan = good
    f1, f2, f3 = (f.flight_id for f in sc.simulated.flights)
    # F3 moved onto F1's gate while F1 still occupies it
    bad = edit(plan, f3, gate_id="GATE-1")
    assert "GATE-OVERLAP" in fails(sc, bad)
    # resident aircraft on GATE-2 while F2 (widebody) arrives
    sc.simulated.gates[1].resident_occupancy.append(Interval(start_min=0, end_min=60, reason="resident"))
    r = [x for x in validate_plan(sc, plan).failures if x.constraint_id == "GATE-OVERLAP"]
    assert r and r[0].resource_id == "GATE-2" and "resident" in r[0].explanation


def test_gate_compatibility_and_notice(good):
    sc, plan = good
    f1, f2, f3 = (f.flight_id for f in sc.simulated.flights)
    assert "GATE-COMPAT" in fails(sc, edit(plan, f2, gate_id="GATE-3"))      # widebody at narrowbody gate
    changed = edit(plan, f1, gate_id="GATE-3")  # gate_id differs from original; notice measured to gate arrival
    p = [x for x in changed.flights if x.flight_id == f1][0]
    changed.decision_time_min = p.gate_in_min - 10  # only 10 min notice < 20
    assert "GATE-NOTICE" in fails(sc, changed)
    changed.decision_time_min = 0.0
    assert "GATE-NOTICE" not in fails(sc, changed)


def test_crew_conflicts_and_duty(good):
    sc, plan = good
    f1, f2, f3 = (f.flight_id for f in sc.simulated.flights)
    # F3 serviced by GC-1 while GC-1 is still servicing F1
    p1 = [x for x in plan.flights if x.flight_id == f1][0]
    bad = edit(plan, f3, service_start_min=p1.service_start_min + 1, service_end_min=p1.service_end_min + 31)
    assert "CREW-OVERLAP" in fails(sc, bad)
    # same flight crew on two flights
    assert "CREW-OVERLAP" in fails(sc, edit(plan, f3, flight_crew_id=sc.simulated.flights[0].flight_crew_id))
    # duty: crew report 13 h before t+10 => limit t+10, gate arrival is later
    sc.simulated.flight_crews[0].report_time_min = 10 - 13 * 60
    r = [x for x in validate_plan(sc, plan).failures if x.constraint_id == "CREW-FLIGHT-DUTY"]
    assert r and r[0].flight_id == f1 and "limit" in r[0].explanation
    # ground-crew overtime beyond cap
    sc.simulated.ground_crews[0].shift_start_min = -480 - 80  # shift ended t-80: ~190 min overtime > 120 cap
    assert "CREW-GROUND-SHIFT" in fails(sc, plan)


def test_fuel_reserve_and_contingency(good):
    sc, plan = good
    f1 = sc.simulated.flights[0].flight_id
    spare = 40.0  # calm fixture: 40 min spare above reserve+contingency
    assert not {"FUEL-RESERVE", "FUEL-CONTINGENCY"} & fails(sc, edit(plan, f1, landing_min=20 + spare - 1))
    f = fails(sc, edit(plan, f1, landing_min=20 + spare + 0.5))
    assert "FUEL-CONTINGENCY" in f and "FUEL-RESERVE" not in f   # eats contingency but not the final reserve
    r = [x for x in validate_plan(sc, edit(plan, f1, landing_min=20 + spare + 30)).failures if x.constraint_id == "FUEL-RESERVE"]
    assert r and "kg" in r[0].explanation


def test_turnaround_and_task_timing(good):
    sc, plan = good
    f1 = sc.simulated.flights[0].flight_id
    p1 = [x for x in plan.flights if x.flight_id == f1][0]
    short = edit(plan, f1, service_end_min=p1.gate_in_min + sc.workbook.airport.min_turnaround_min - 5)
    assert "TURNAROUND-MIN" in fails(sc, short)
    assert "TASK-TIME" in fails(sc, edit(plan, f1, service_start_min=p1.gate_in_min + 1))  # before deboarding done


def test_taxi_route_blockage_and_sequencing(good):
    sc, plan = good
    f1 = sc.simulated.flights[0].flight_id
    sc.simulated.taxi_segments[[x.id for x in sc.simulated.taxi_segments].index("TWY-C")].blockages.append(
        Interval(start_min=0, end_min=200, reason="blocked"))
    assert "TAXI-BLOCK" in fails(sc, plan)
    assert "TAXI-ROUTE" in fails(sc, edit(plan, f1, taxi_route=["TWY-A1", "TWY-G1"]))     # not connected
    assert "TAXI-ROUTE" in fails(sc, edit(plan, f1, taxi_route=["TWY-A2", "TWY-C", "TWY-G1"]))  # wrong runway exit
    assert "SEQ-ORDER" in fails(sc, edit(plan, f1, taxi_start_min=20.2))  # taxiing before runway vacated


def test_structural_checks_and_no_crash_on_garbage(good):
    sc, plan = good
    short = plan.model_copy(deep=True)
    short.flights = short.flights[:2]
    assert "PLAN-COMPLETE" in fails(sc, short)
    stale = plan.model_copy(update={"scenario_version": "old"})
    assert "PLAN-VERSION" in fails(sc, stale)
    f1 = sc.simulated.flights[0].flight_id
    junk = edit(plan, f1, gate_id="NOPE", ground_crew_id="NOPE", flight_crew_id="NOPE", taxi_route=["X"])
    assert {"TAXI-ROUTE", "GATE-COMPAT", "CREW-GROUND-AVAIL", "CREW-FLIGHT-DUTY"} <= fails(sc, junk)
    assert validate_plan(sc, RecoveryPlan(proposal_id="p", scenario_version="x", flights=[])).passed is False
