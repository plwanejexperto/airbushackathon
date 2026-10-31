"""Stage 1-2: per-aircraft Pilots, the Pilot's structured report, remote stands, the safe ground waiting area, ground-phase fuel,
and 'never hold an aircraft airborne just because no gate is free'."""
import json

import pytest

from conftest import mk
from mocks import MockClient, Script, accept, agree_on, frozen_id, propose
from airport_ops import agents
from airport_ops.fuel import ground_burn_kg, fuel_account
from airport_ops.llm import ScriptedClient
from airport_ops.messages import AgentRole
from airport_ops.negotiation import Coordinator, NegotiationConfig, run_negotiation
from airport_ops.negotiation_types import AgentContext, FallbackReason
from airport_ops.planner import Planner
from airport_ops.reports import compare_reports, pilot_report
from airport_ops.scenario import rehash_scenario
from airport_ops.schemas import Interval
from airport_ops.sessions import participants_for
from airport_ops.validate import validate_plan
from airport_ops.workbook import AircraftClass as AC

FULL = NegotiationConfig(require_pilot_report=True)


def failed(sc, plan):
    return {r.constraint_id for r in validate_plan(sc, plan).failures}


def edit(plan, fid, **kw):
    p = plan.model_copy(deep=True)
    [f for f in p.flights if f.flight_id == fid][0].__dict__.update(kw)
    return p


# ------------------------------------------------------------------ per-aircraft Pilots + the structured report
def test_one_independent_pilot_per_aircraft_plus_atc_and_ground(scenario):
    parts = participants_for(scenario)
    assert [p.agent_id for p in parts] == ["PILOT@DL1205", "PILOT@SIM-A101", "PILOT@SIM-B202", "ATC@ATL", "GROUND@ATL"]
    assert [p.role for p in parts][:3] == [AgentRole.PILOT] * 3 and parts[3].airport == parts[4].airport == "ATL"


def test_each_pilot_sees_only_its_own_aircraft_and_is_prompted_separately(scenario):
    parts = participants_for(scenario)
    ctx = lambda p: AgentContext(role=p.role, agent=p, participants=parts, turn=1, seq=1, session_id="s", scenario=scenario,
                                 historical=scenario.historical, messages=[], proposals=[], standing={q.agent_id: [] for q in parts},
                                 allowed_types=[])
    views = {p.flight_id: agents.pilot.domain_view(ctx(p)) for p in parts[:3]}
    for fid, v in views.items():
        assert v["your_aircraft"]["flight_id"] == fid and v["report_template"]["flight_id"] == fid
        assert fid not in [t["flight_id"] for t in v["other_traffic"]] and len(v["other_traffic"]) == 2
        assert "position" in v["your_aircraft"] and "weather" in v["your_aircraft"] and "ground_fuel" in v["your_aircraft"]
    fuel = {fid: v["your_aircraft"]["fuel_at_nominal_eta"]["fuel_on_board_kg"] for fid, v in views.items()}
    assert len(set(fuel.values())) == 3                                    # three different aircraft, three different fuel states
    prompts = {p.agent_id: json.loads(agents.user_prompt(ctx(p))) for p in parts}
    assert prompts["PILOT@SIM-A101"]["your_agent"]["flight_id"] == "SIM-A101"
    assert "YOU ARE AGENT PILOT@SIM-A101" in agents.system_prompt(parts[1]) and agents.system_prompt(parts[1]) != agents.system_prompt(parts[0])
    assert all(p["your_agent"]["all_agents_in_this_session"] == [q.agent_id for q in parts] for p in prompts.values())


def test_pilot_report_is_computed_by_code_and_discrepancies_are_recorded(scenario):
    r = pilot_report(scenario, "DL1205")
    sf = scenario.simulated.flights[0]
    assert r.earliest_landing_min == sf.earliest_landing_min and r.latest_landing_min == pytest.approx(sf.earliest_landing_min + r.max_holding_min)
    assert r.contingency_margin_kg > 0 and r.reserve_margin_kg > r.contingency_margin_kg
    assert compare_reports(r, r) == []
    bad = r.model_copy(update={"max_holding_min": r.max_holding_min + 5, "latest_landing_min": r.latest_landing_min + 5})
    assert any("max_holding_min" in d for d in compare_reports(bad, r))


def test_full_session_has_five_agents_15_messages_verified_reports_and_message_level_flight_ids(scenario, tmp_path):
    res = run_negotiation(scenario, ScriptedClient(), participants=participants_for(scenario), config=FULL, log_path=tmp_path / "e.db")
    assert res.consensus and res.turns_completed == 3 and len(res.messages) == 15 and len(res.attempts) == 15
    assert [(m.turn, m.agent_id) for m in res.messages] == [(t, p.agent_id) for t in (1, 2, 3) for p in res.participants]
    reps = [m for m in res.messages if m.report is not None]
    assert [m.agent_id for m in reps] == ["PILOT@DL1205", "PILOT@SIM-A101", "PILOT@SIM-B202"]       # first message of each Pilot only
    assert all(m.report_verified and not m.report_discrepancies for m in reps)
    assert all(m.flight_id for m in res.messages)                                                     # message-level flight id
    assert {m.flight_id for m in res.messages if m.sender == AgentRole.PILOT} == {"DL1205", "SIM-A101", "SIM-B202"}
    assert {m.airport for m in res.messages if m.sender != AgentRole.PILOT} == {"ATL"}
    assert len(res.votes) == 5 and all(v.counted_as_accept for v in res.votes)


def test_a_single_pilot_rejecting_the_frozen_plan_blocks_consensus(scenario, tmp_path):
    base = ScriptedClient()

    def handler(req):
        text = base.complete(req).text
        if req.turn == 3 and req.agent_id == "PILOT@SIM-B202":
            return {"type": "REJECT", "reason_codes": ["CONNECTION_PRESERVATION"], "explanation": "my connection is lost", "proposal_id": frozen_id(req)}
        return text

    res = run_negotiation(scenario, MockClient(handler), participants=participants_for(scenario), config=FULL, log_path=tmp_path / "e.db")
    assert not res.consensus and res.fallback_reason == FallbackReason.NO_UNANIMOUS_AGREEMENT
    assert "PILOT@SIM-B202: rejected" in res.fallback_detail and len(res.messages) == 15
    assert [v for v in res.votes if v.agent_id == "PILOT@SIM-B202"][0].counted_as_accept is False


def test_missing_or_wrong_pilot_report_is_handled(scenario, tmp_path):
    from airport_ops.tools import candidate_pool
    cand = candidate_pool(scenario)[0]
    parts = participants_for(scenario)
    # (a) report required but omitted -> repair retry -> still missing -> fallback RESPONSE_ERROR
    res = run_negotiation(scenario, MockClient(lambda r: propose(cand)), participants=parts, config=FULL, log_path=tmp_path / "a.db")
    assert res.fallback_reason == FallbackReason.RESPONSE_ERROR and "report" in res.fallback_detail and len(res.attempts) == 2
    # (b) report with wrong numbers: accepted, but the code-computed values replace it and the discrepancy is recorded
    wrong = {**propose(cand), "report": {**pilot_report(scenario, "DL1205").model_dump(mode="json"), "max_holding_min": 999.0}}
    res = run_negotiation(scenario, MockClient(lambda r: wrong if r.agent_id == "PILOT@DL1205" else accept(r.context.proposals[0].proposal_id) if r.turn < 3 and r.context.proposals else accept(frozen_id(r)) if r.turn == 3 else wrong),
                          participants=parts[:1] + parts[3:], config=FULL, log_path=tmp_path / "b.db")
    m = res.messages[0]
    assert m.report_verified is False and any("max_holding_min" in d for d in m.report_discrepancies)
    assert m.report.max_holding_min != 999.0                                   # authoritative value, not the claim


# ------------------------------------------------------------------ remote stands
def test_scenario_has_a_simulated_remote_stand_and_a_safe_waiting_area(scenario):
    stand = [g for g in scenario.simulated.gates if g.kind == "remote_stand"]
    assert len(stand) == 1 and stand[0].max_aircraft_class == AC.WIDEBODY and scenario.workbook.capacity_subset.modelled_remote_stands == 1
    assert [h.id for h in scenario.simulated.hold_areas] == ["HOLD-1"] and any(x.id == "TWY-S1" for x in scenario.simulated.taxi_segments)
    assert all(f.original_gate_id != "STAND-1" for f in scenario.simulated.flights)       # stands are never an original assignment


def test_remote_stand_is_a_valid_option_and_costs_bussing_time(scenario):
    pl = Planner(scenario)
    ids = [f.flight_id for f in scenario.simulated.flights]
    plan = pl.build(tuple(ids), {i: "RWY-1" for i in ids}, {ids[0]: "STAND-1", ids[1]: "GATE-2", ids[2]: "GATE-3"},
                    {i: "GC-1" if k % 2 == 0 else "GC-2" for k, i in enumerate(ids)})
    p = [f for f in plan.flights if f.flight_id == ids[0]][0]
    a = scenario.simulated.assumptions
    assert p.gate_id == "STAND-1" and p.service_start_min - p.gate_in_min >= a.deboard_min + a.remote_stand_bus_min - 1e-9
    rep = validate_plan(scenario, plan, stages=("ground",))
    assert "GATE-COMPAT" not in {r.constraint_id for r in rep.failures}
    assert "TASK-TIME" not in {r.constraint_id for r in rep.failures}
    quick = edit(plan, ids[0], service_start_min=p.gate_in_min + a.deboard_min)       # forgot the bus transfer
    assert "TASK-TIME" in failed(scenario, quick)


def test_a_widebody_can_use_the_remote_stand_when_every_widebody_gate_is_busy(scenario):
    sc = scenario.model_copy(deep=True)
    for f in sc.simulated.flights[1:2]:
        f.aircraft_class, f.wake_category = AC.WIDEBODY, __import__("airport_ops.workbook", fromlist=["x"]).WakeCategory.HEAVY
    sc.simulated.gates[1].resident_occupancy = [Interval(start_min=0, end_min=400, reason="busy")]       # the only widebody gate
    sc = rehash_scenario(sc)
    from airport_ops.fallback import run_fallback, Outcome
    res = run_fallback(sc)
    assert res.outcome == Outcome.FALLBACK_SELECTED
    assert [p.gate_id for p in res.plan.flights if p.flight_id == "SIM-A101"] == ["STAND-1"]


# ------------------------------------------------------------------ safe waiting area
def test_waiting_after_landing_needs_a_designated_safe_area_with_capacity(scenario):
    pl = Planner(scenario)
    ids = [f.flight_id for f in scenario.simulated.flights]
    sc = scenario.model_copy(deep=True)
    sc.simulated.gates[0].resident_occupancy = [Interval(start_min=0, end_min=60, reason="busy")]       # GATE-1 busy
    sc = rehash_scenario(sc)
    pl = Planner(sc)
    plan = pl.build(tuple(ids), {i: "RWY-1" for i in ids}, {ids[0]: "GATE-1", ids[1]: "GATE-2", ids[2]: "GATE-3"}, {i: "GC-1" for i in ids})
    waiting = [p for p in plan.flights if p.hold_area_id]
    assert waiting and waiting[0].flight_id == ids[0]                                # DL1205 lands on time and waits on the GROUND
    assert waiting[0].landing_min == sc.simulated.flights[0].earliest_landing_min or waiting[0].landing_min >= sc.simulated.flights[0].earliest_landing_min
    assert "HOLD-AREA" not in failed(sc, plan)
    assert "HOLD-AREA" in failed(sc, edit(plan, ids[0], hold_area_id=None))                   # waiting without a safe area
    assert "HOLD-AREA" in failed(sc, edit(plan, ids[0], hold_area_id="CARPARK"))
    sc.simulated.hold_areas[0].capacity = 1
    one = plan
    for fid in ids[1:]:                                                                      # force all three to wait at once
        one = edit(one, fid, hold_area_id="HOLD-1", taxi_start_min=one.flights[0].taxi_start_min)
    assert "HOLD-CAP" in failed(sc, one)


def test_an_occupied_gate_never_delays_landing_it_is_handled_on_the_ground(scenario):
    ids = [f.flight_id for f in scenario.simulated.flights]
    order, rw = tuple(ids), {i: "RWY-1" for i in ids}
    gates, crews = {ids[0]: "GATE-1", ids[1]: "GATE-2", ids[2]: "GATE-3"}, {i: "GC-1" for i in ids}
    free = Planner(scenario).build(order, rw, gates, crews)
    busy = scenario.model_copy(deep=True)
    for g in busy.simulated.gates:
        if g.kind == "gate":
            g.resident_occupancy = [Interval(start_min=0, end_min=90, reason="all gates busy")]
    busy = rehash_scenario(busy)
    held = Planner(busy).build(order, rw, gates, crews)
    land = lambda pl: {p.flight_id: p.landing_min for p in pl.flights}
    assert land(free) == land(held)                                                         # landing times identical: no airborne hold for gates
    assert all(p.gate_in_min >= 90 for p in held.flights) and all(p.hold_area_id for p in held.flights)
    assert validate_plan(busy, held, stages=("land", "fuel")).passed


# ------------------------------------------------------------------ ground-phase fuel
def test_ground_burn_uses_a_labelled_fraction_of_the_workbook_holding_rate(scenario):
    sf = scenario.simulated.flights[0]
    rate = scenario.workbook.fuel_crew.hold_burn_narrowbody_kg_h
    assert ground_burn_kg(scenario, sf, 20.0, 80.0) == pytest.approx(rate * scenario.simulated.assumptions.ground_burn_fraction_of_hold * 60 / 60)


def test_a_shorter_route_is_rejected_when_waiting_for_it_exhausts_the_fuel_reserve(good):
    sc, plan = good
    f1 = sc.simulated.flights[0].flight_id
    for x in sc.simulated.taxi_segments:
        if x.id == "TWY-C":
            x.blockages = [Interval(start_min=0, end_min=200, reason="long blockage")]
    wait = mk(sc, f1, "RWY-1", 20.0, "GATE-1", "GC-1", 200.0, via="TWY-C")          # short route, waits ~3 h on the ground
    around = mk(sc, f1, "RWY-1", 20.0, "GATE-1", "GC-1", 21.5, via="TWY-D")         # longer route, no waiting
    def with_f1(p):
        q = plan.model_copy(deep=True)
        q.flights = [p if f.flight_id == f1 else f for f in q.flights]
        return q
    rep = validate_plan(sc, with_f1(wait))
    g = [r for r in rep.failures if r.constraint_id == "FUEL-GROUND"]
    assert g and g[0].flight_id == f1 and "on the ground" in g[0].explanation
    assert "FUEL-GROUND" not in {r.constraint_id for r in validate_plan(sc, with_f1(around)).failures}
    # the plan builder itself takes the longer route that gets the aircraft to the gate sooner
    ids = [f.flight_id for f in sc.simulated.flights]
    built = Planner(sc).build(tuple(ids), {i: "RWY-1" for i in ids}, {ids[0]: "GATE-1", ids[1]: "GATE-2", ids[2]: "GATE-3"}, {i: "GC-1" for i in ids})
    assert "TWY-D" in [p for p in built.flights if p.flight_id == f1][0].taxi_route


def test_weather_position_and_status_are_visible_to_the_pilot(scenario):
    parts = participants_for(scenario)
    ctx = AgentContext(role=AgentRole.PILOT, agent=parts[0], participants=parts, turn=1, seq=1, session_id="s", scenario=scenario,
                       historical=scenario.historical, messages=[], proposals=[], standing={}, allowed_types=[])
    v = agents.pilot.domain_view(ctx)["your_aircraft"]
    assert v["position"]["SIMULATED"] and v["position"]["route_distance_nm (flights.csv)"] > 1000          # PHX -> ATL from the dataset
    assert v["weather"]["historical_weather_delay_min (dataset, context only)"] == scenario.historical.weather_delay_min
    assert v["status"] == "NORMAL"
