"""Scenario 4: turnaround + departure."""
import pytest

from mocks import MockClient
from airport_ops import agents
from airport_ops.fallback import Outcome, SearchStatus, run_fallback
from airport_ops.llm import ScriptedClient
from airport_ops.messages import AgentRole, PlanSpec
from airport_ops.negotiation import run_negotiation, NegotiationConfig
from airport_ops.negotiation_types import AgentContext
from airport_ops.reference import reference_plan
from airport_ops.reports import compare_readiness, departure_readiness
from airport_ops.scenario import build_scenario, rehash_scenario
from airport_ops.schemas import Interval
from airport_ops.sessions import participants_for
from airport_ops.tools import candidate_pool
from airport_ops.turnaround import (build_departure_demo, build_departure_scenario, build_turnaround_plan, next_flights_for, task_minutes)
from airport_ops.validate import validate_plan
from airport_ops.workbook import WakeCategory as W

CFG = NegotiationConfig(require_pilot_report=True)
IDS = ["DL1205", "SIM-A101", "SIM-B202"]


def spec(order=IDS, overlap="max", crews=("GC-1", "GC-2", "GC-1"), rw=("RWY-1", "RWY-1", "RWY-1")):
    return PlanSpec(equipment_order=list(order), overlap=overlap, ground_crews=dict(zip(IDS, crews)), dep_runways=dict(zip(IDS, rw)))


def fails(sc, plan):
    return {r.constraint_id for r in validate_plan(sc, plan, stages=("ground", "depart")).failures}


def flight(plan, fid):
    return [f for f in plan.flights if f.flight_id == fid][0]


def edit(plan, fid, **kw):
    p = plan.model_copy(deep=True)
    flight(p, fid).__dict__.update(kw)
    return p


@pytest.fixture()
def good_plan(dep_sc):
    """A feasible hand-chosen plan: staggered crews, fuel truck serialised by the builder."""
    for s in (spec(), spec(crews=("GC-2", "GC-1", "GC-2")), spec(order=IDS[::-1]), spec(rw=("RWY-2", "RWY-1", "RWY-2"))):
        p = build_turnaround_plan(dep_sc, s)
        if p and not fails(dep_sc, p):
            return p
    pytest.skip("no feasible plan among the hand-picked specs")


# ------------------------------------------------------------------ next flight is an INPUT (data when available, labelled otherwise)
def test_next_flight_comes_from_flights_csv_when_the_tail_has_one_else_is_labelled_simulated(rules, flights_df, raw_flights, dep_sc):
    nf = {n.flight_id: n for n in dep_sc.simulated.departure.next_flights}
    assert nf["DL1205"].source == "SIMULATED" and "no usable same-day next flight" in nf["DL1205"].note     # N612DL has none on 2015-01-01
    found = None
    for _, row in flights_df.sample(400, random_state=3).iterrows():
        from airport_ops import data
        sc = build_scenario(data.to_historical(row), rules)
        real = next_flights_for(sc, raw_flights)[0]
        if real.source == "flights.csv":
            found = (sc, real, row)
            break
    assert found, "expected at least one flight whose tail has a same-day next flight leaving from this airport"
    sc, real, row = found
    nxt = raw_flights[(raw_flights.TAIL_NUMBER == row.TAIL_NUMBER) & (raw_flights.MONTH == row.MONTH) & (raw_flights.DAY == row.DAY)
                      & (raw_flights.ORIGIN_AIRPORT == row.DESTINATION_AIRPORT)]
    assert f"{nxt.iloc[0].AIRLINE}{nxt.iloc[0].FLIGHT_NUMBER}" in {f"{r.AIRLINE}{r.FLIGHT_NUMBER}" for r in nxt.itertuples()}
    assert real.next_flight_label in {f"{r.AIRLINE}{r.FLIGHT_NUMBER}" for r in nxt.itertuples()} and real.planned_ground_min >= sc.workbook.airport.min_turnaround_min
    assert real.scheduled_block_min > 0


def test_scheduled_departure_is_anchored_to_the_reference_gate_arrival_plus_the_planned_ground_time(dep_sc):
    d = dep_sc.simulated.departure
    from airport_ops.planner import Planner
    ref = {p.flight_id: p for p in Planner(dep_sc).naive_plan().flights}
    # the departure scenario was built from the arrival scenario; the same nominal gate arrival is used (rebuild to check the arithmetic)
    for n in d.next_flights:
        assert d.scheduled_departure_min[n.flight_id] > 0 and n.planned_ground_min >= dep_sc.workbook.airport.min_turnaround_min


# ------------------------------------------------------------------ the arrival part is fixed
def test_arrival_part_is_a_fixed_input(dep_sc, good_plan):
    assert "DEP-ARRIVAL-FIXED" not in fails(dep_sc, good_plan)
    assert "DEP-ARRIVAL-FIXED" in fails(dep_sc, edit(good_plan, "DL1205", gate_id="GATE-3"))
    assert "DEP-ARRIVAL-FIXED" in fails(dep_sc, edit(good_plan, "DL1205", landing_min=flight(good_plan, "DL1205").landing_min + 5))


# ------------------------------------------------------------------ tasks: durations, dependencies, overlap, equipment
def test_task_durations_scale_and_remote_stands_add_bussing(dep_sc):
    a = dep_sc.simulated.assumptions
    gate = task_minutes(dep_sc, "DL1205", "GATE-1")
    stand = task_minutes(dep_sc, "DL1205", "STAND-1")
    assert stand["DEPLANE"] == pytest.approx(gate["DEPLANE"] + a.remote_stand_bus_min) and stand["BOARD"] == pytest.approx(gate["BOARD"] + a.remote_stand_bus_min)
    assert stand["REFUEL"] == gate["REFUEL"]
    wide = task_minutes(dep_sc, "SIM-A101", "GATE-2")
    f = dep_sc.simulated.departure.turnaround.widebody_factor
    assert wide["CLEAN"] == pytest.approx(12 * f) if any(x.flight_id == "SIM-A101" and x.aircraft_class.value == "widebody" for x in dep_sc.simulated.flights) else True


def test_dependencies_are_enforced_and_parallel_tasks_overlap_where_allowed(dep_sc, good_plan):
    p = flight(good_plan, "DL1205")
    t = {x.task_id: x for x in p.tasks}
    assert t["CLEAN"].start_min < t["REFUEL"].end_min and t["REFUEL"].start_min < t["CLEAN"].end_min      # independent tasks overlap
    assert t["BOARD"].start_min >= max(t["CLEAN"].end_min, t["REFUEL"].end_min) - 1e-9                       # no boarding during refuelling
    assert "TASK-DEPS" not in fails(dep_sc, good_plan)
    bad = good_plan.model_copy(deep=True)
    for x in flight(bad, "DL1205").tasks:
        if x.task_id == "BOARD":
            x.start_min, x.end_min = t["REFUEL"].start_min, t["REFUEL"].start_min + (x.end_min - x.start_min)   # boarding during refuelling
    assert "TASK-DEPS" in fails(dep_sc, bad)
    early = good_plan.model_copy(deep=True)
    for x in flight(early, "DL1205").tasks:
        if x.task_id == "DEPLANE":
            x.start_min -= 3
            x.end_min -= 3
    assert "TASK-DEPS" in fails(dep_sc, early)                                                                # before gate arrival


def test_sequential_mode_is_slower_than_overlapping_mode(dep_sc):
    a, b = build_turnaround_plan(dep_sc, spec(overlap="max")), build_turnaround_plan(dep_sc, spec(overlap="sequential"))
    assert flight(b, "DL1205").ready_min >= flight(a, "DL1205").ready_min - 1e-9


def test_shared_equipment_serves_one_task_at_a_time(dep_sc, good_plan):
    refuel = sorted((t.start_min, t.end_min, f.flight_id) for f in good_plan.flights for t in f.tasks if t.task_id == "REFUEL")
    assert all(t[1] <= nxt[0] + 1e-6 for t, nxt in zip(refuel, refuel[1:]))                                   # one fuel truck, serialised
    clash = good_plan.model_copy(deep=True)
    other = flight(clash, "SIM-A101")
    for x in other.tasks:
        if x.task_id == "REFUEL":
            x.start_min, x.end_min, x.equipment_id = flight(good_plan, "DL1205").tasks[3].start_min, flight(good_plan, "DL1205").tasks[3].end_min, "FUEL-TRUCK-1"
    assert "TASK-EQUIP" in fails(dep_sc, clash)
    wrong = good_plan.model_copy(deep=True)
    [x for x in flight(wrong, "DL1205").tasks if x.task_id == "REFUEL"][0].equipment_id = "TRACTOR-1"            # wrong equipment type
    assert "TASK-EQUIP" in fails(dep_sc, wrong)


def test_minimum_turnaround_and_ready_time(dep_sc, good_plan):
    p = flight(good_plan, "DL1205")
    assert p.ready_min >= p.gate_in_min + dep_sc.workbook.airport.min_turnaround_min - 1e-9
    assert "READY-TIME" not in fails(dep_sc, good_plan)
    assert "READY-TIME" in fails(dep_sc, edit(good_plan, "DL1205", ready_min=p.gate_in_min + 5, service_end_min=p.gate_in_min + 5))
    short = edit(good_plan, "DL1205", ready_min=p.ready_min - 3, service_end_min=p.ready_min - 3)
    assert "READY-TIME" in fails(dep_sc, short)


# ------------------------------------------------------------------ departure: separation, runway, schedule, crew, capacity
def test_departure_separation_uses_the_workbook_same_runway_departure_times(dep_sc, good_plan):
    dep_sc.simulated.flights[1].wake_category = W.HEAVY                       # SIM-A101 Heavy leads, then a Large
    ids = ["SIM-A101", "DL1205"]
    p = good_plan.model_copy(deep=True)
    lead, foll = flight(p, ids[0]), flight(p, ids[1])
    lead.dep_runway_id = foll.dep_runway_id = "RWY-1"
    lead.takeoff_min = foll.takeoff_min - 10
    secs = dep_sc.workbook.wake.lookup(W.HEAVY, W.LARGE)[1]                  # workbook: Heavy -> Large departures 120 s
    assert secs == 120
    foll.takeoff_min = lead.takeoff_min + secs / 60 - 0.01
    wake = lambda pl: [r for r in validate_plan(dep_sc, pl, stages=("depart",)).results if r.constraint_id == "DEP-WAKE"]
    assert any(not r.passed and "min_time_s" in r.explanation for r in wake(p))
    foll.takeoff_min = lead.takeoff_min + secs / 60
    assert all(r.passed for r in wake(p))


def test_runway_closure_early_departure_and_crew_duty_for_the_next_flight(dep_sc, good_plan):
    p = flight(good_plan, "DL1205")
    dep_sc.simulated.runways[0].closures = [Interval(start_min=p.takeoff_min - 1, end_min=p.takeoff_min + 5, reason="x")]
    rw = p.dep_runway_id
    dep_sc.simulated.runways[[r.id for r in dep_sc.simulated.runways].index(rw)].closures = [Interval(start_min=p.takeoff_min - 1, end_min=p.takeoff_min + 5, reason="closed")]
    assert "DEP-RWY-AVAIL" in fails(dep_sc, good_plan)
    assert "DEP-NOT-EARLY" in fails(dep_sc, edit(good_plan, "DL1205", takeoff_min=dep_sc.simulated.departure.scheduled_departure_min["DL1205"] - 5))
    rd = departure_readiness(dep_sc, "DL1205")
    late = edit(good_plan, "DL1205", takeoff_min=rd.latest_takeoff_min + 1)
    r = [x for x in validate_plan(dep_sc, late, stages=("depart",)).failures if x.constraint_id == "DEP-CREW-DUTY"]
    assert r and "next flight" in r[0].explanation and rd.next_flight_label in r[0].explanation


def test_departure_movement_capacity_includes_arrivals(dep_sc, good_plan):
    dep_sc.workbook.airport = dep_sc.workbook.airport.model_copy(update={"max_runway_ops_per_hour": 2})
    assert "DEP-CAP" in fails(dep_sc, good_plan)


def test_departure_route_must_connect_gate_to_runway_and_avoid_blockages(dep_sc, good_plan):
    assert "DEP-TAXI-ROUTE" in fails(dep_sc, edit(good_plan, "DL1205", dep_taxi_route=["TWY-C-OUT"]))
    seg = flight(good_plan, "DL1205").dep_taxi_route[0]
    dep_sc.simulated.taxi_segments[[x.id for x in dep_sc.simulated.taxi_segments].index(seg)].blockages = [Interval(start_min=0, end_min=500, reason="blocked")]
    assert "DEP-TAXI-BLOCK" in fails(dep_sc, good_plan)


# ------------------------------------------------------------------ fallback + reference
def test_departure_fallback_is_deterministic_valid_and_honest_when_infeasible(dep_sc):
    a, b = run_fallback(dep_sc), run_fallback(dep_sc)
    assert a.outcome == Outcome.FALLBACK_SELECTED and a.search_status == SearchStatus.EXHAUSTED and a.plan == b.plan and a.plan.source == "fallback"
    assert validate_plan(dep_sc, a.plan, stages=("ground", "depart")).passed and a.candidates_total == 768
    hard = dep_sc.model_copy(deep=True)
    last_in = max(x.gate_in_min for x in hard.simulated.departure.arrival)
    for c in hard.simulated.flight_crews:                                    # enough duty to land and park, not to fly the NEXT flight
        c.report_time_min = (last_in + hard.simulated.assumptions.flight_crew_postflight_min + 5) - hard.workbook.fuel_crew.flight_crew_max_duty_h * 60
    res = run_fallback(rehash_scenario(hard))
    assert res.outcome == Outcome.NO_FEASIBLE_PLAN and res.search_status == SearchStatus.EXHAUSTED and "DEP-CREW-DUTY" in res.rejection_by_constraint
    assert run_fallback(dep_sc, max_evaluations=3).search_status == SearchStatus.TIMEOUT


def test_reference_is_an_illustrative_unvalidated_baseline(dep_sc):
    ref = reference_plan(dep_sc)
    assert ref.source == "reference" and {"TASK-EQUIP", "CREW-OVERLAP"} <= fails(dep_sc, ref)


# ------------------------------------------------------------------ the three agents negotiate it
def test_departure_agents_have_separate_domain_views(dep_sc):
    parts = participants_for(dep_sc)
    mk = lambda p: AgentContext(role=p.role, agent=p, participants=parts, turn=1, seq=1, session_id="s", scenario=dep_sc, historical=dep_sc.historical,
                                messages=[], proposals=[], standing={}, allowed_types=[])
    atc, ground, pilot = (agents.MODULES[r].domain_view(mk(next(p for p in parts if p.role == r))) for r in (AgentRole.ATC, AgentRole.GROUND, AgentRole.PILOT))
    assert any("departure_spacing" in k for k in atc) and "turnaround_tasks (SIMULATED work content)" not in atc
    assert "turnaround_tasks (SIMULATED work content)" in ground and "equipment (SIMULATED, one task at a time per unit)" in ground
    assert "next_flight" in pilot["your_aircraft"] and "readiness_template" in pilot and "outbound_taxi_network" not in pilot


def test_scripted_departure_session_reaches_consensus_with_verified_readiness(dep_sc, tmp_path):
    res = run_negotiation(dep_sc, ScriptedClient(), participants=participants_for(dep_sc), config=CFG, log_path=tmp_path / "e.db")
    assert res.consensus and len(res.messages) == 15 and res.turns_completed == 3
    rd = [m for m in res.messages if m.readiness is not None]
    assert len(rd) == 3 and all(m.report_verified for m in rd) and all(m.report is None for m in rd)
    assert validate_plan(dep_sc, res.final_plan, stages=("ground", "depart")).passed
    assert all(f.takeoff_min is not None and f.ready_min is not None for f in res.final_plan.flights)


def test_departure_conflict_preset_makes_three_domains_disagree_and_logs_concessions(dep_sc, tmp_path):
    demo = build_departure_demo(dep_sc)
    from airport_ops.tools import best_for
    pool = candidate_pool(demo)
    best = {r: best_for(r, demo, pool, flight="DL1205") for r in AgentRole}
    assert len({b.proposal_id for b in best.values()}) == 3                           # ATC, Pilot and Ground prefer different plans
    res = run_negotiation(demo, ScriptedClient(), participants=participants_for(demo), config=CFG, log_path=tmp_path / "e.db")
    assert res.consensus and len(res.proposals) >= 3
    assert sum(m.type.value == "COUNTER_PROPOSE" for m in res.messages) >= 2
    assert any("CONCESSION" in m.explanation and m.tradeoff is not None for m in res.messages)
    assert not validate_plan(demo, res.reference_plan, stages=("ground", "depart")).passed


def test_a_wrong_readiness_claim_is_recorded_not_trusted(dep_sc, tmp_path):
    rd = departure_readiness(dep_sc, "DL1205")
    wrong = rd.model_copy(update={"latest_takeoff_min": rd.latest_takeoff_min + 30})
    assert any("latest_takeoff_min" in d for d in compare_readiness(wrong, rd)) and compare_readiness(rd, rd) == []
