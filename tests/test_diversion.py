"""Scenario 3: diversion / emergency (one destination, one alternate)."""
import pytest

from mocks import MockClient
from airport_ops import agents, diversion as dvmod
from airport_ops.diversion import DiversionUnavailable, build_diversion_scenario, distance_from_position, position_of, route_distances, screen_alternates
from airport_ops.fallback import Outcome, SearchStatus, run_fallback
from airport_ops.fuel import fuel_account
from airport_ops.llm import ScriptedClient
from airport_ops.messages import AgentRole
from airport_ops.negotiation import NegotiationConfig, run_negotiation
from airport_ops.negotiation_types import AgentContext, FallbackReason, FinalStatus
from airport_ops.plan import make_plan
from airport_ops.reference import reference_plan
from airport_ops.schemas import Interval
from airport_ops.sessions import participants_for
from airport_ops.validate import validate_plan
from airport_ops.workbook import FlightStatus as S, AircraftClass as AC

CFG = NegotiationConfig(require_pilot_report=True)


@pytest.fixture(scope="session")
def divs(scenario_session, rules, flights_df):
    return {c: build_diversion_scenario(scenario_session, rules, flights_df, fuel_case=c) for c in ("normal", "minimum_fuel", "emergency")}


@pytest.fixture(scope="session")
def scenario_session(rules, flights_df):
    from airport_ops import data
    from airport_ops.scenario import build_scenario
    return build_scenario(data.to_historical(flights_df.loc[data.default_index(flights_df)]), rules)


def fails(sc, plan):
    return {r.constraint_id for r in validate_plan(sc, plan).failures}


def good(sc, **kw):
    return run_fallback(sc).plan


def with_status(plan, sc, status):
    return make_plan(sc, "test", plan.flights, declared_status=status, airport=plan.airport)


# ------------------------------------------------------------------ position + distance (not simply from the destination)
def test_distance_from_position_geometry_and_labelled_assumption():
    d, how = distance_from_position(D=200.0, s=0.0, a=100.0, b=100.0)           # aircraft still at the origin: distance = origin-alternate
    assert d == pytest.approx(100.0) and "triangle geometry" in how
    d, _ = distance_from_position(200.0, 200.0, 100.0, 100.0)                   # aircraft at the destination: distance = destination-alternate
    assert d == pytest.approx(100.0)
    d_mid, _ = distance_from_position(200.0, 100.0, 120.0, 120.0)               # on the route half way, alternate off the line
    assert d_mid < 120.0 and d_mid > 0
    d2, how2 = distance_from_position(200.0, 150.0, None, 80.0)                 # no origin-alternate route: conservative, explicitly labelled
    assert d2 == pytest.approx(50.0 + 80.0) and how2.startswith("ASSUMPTION")


def test_alternate_distance_depends_on_the_aircrafts_position_not_just_the_destination(scenario_session, rules, flights_df):
    far = scenario_session.model_copy(deep=True)
    far.simulated.flights[0].earliest_landing_min = 90.0                         # 90 min from the destination instead of ~22
    near = screen_alternates(scenario_session, rules, flights_df)
    away = screen_alternates(far, rules, flights_df)
    assert [c.airport for c in near] != [] and position_of(far)["distance_to_destination_nm"] > position_of(scenario_session)["distance_to_destination_nm"]
    by = {c.airport: c for c in near}
    for c in away:
        if c.airport in by:
            assert c.distance_from_position_nm != by[c.airport].distance_from_position_nm       # moves with the aircraft
    assert all(c.distance_from_position_nm != c.route_to_destination_nm for c in near)           # not simply the destination-alternate leg
    assert [c.distance_from_position_nm for c in near] == sorted(c.distance_from_position_nm for c in near)


def test_a_historical_route_alone_does_not_make_an_airport_suitable(scenario_session, rules, flights_df, monkeypatch):
    base = screen_alternates(scenario_session, rules, flights_df)
    assert all(c.route_to_destination_nm for c in base)                                          # every candidate HAS a route in flights.csv
    monkeypatch.setattr(dvmod, "_weather_ok", lambda airport, date: airport != base[0].airport)
    c0 = screen_alternates(scenario_session, rules, flights_df)[0]
    assert c0.airport == base[0].airport and not c0.suitable and "simulated weather unusable" in c0.reasons and c0.route_to_destination_nm
    wide = scenario_session.model_copy(deep=True)
    wide.simulated.flights[0].aircraft_class = AC.WIDEBODY
    monkeypatch.undo()
    small = [c for c in screen_alternates(wide, rules, flights_df, limit=60) if c.size_class in ("Non-hub", "Small hub")]
    assert small and all(not c.size_ok and not c.suitable for c in small)                        # workbook size class matters, route or not


def test_screened_set_is_small_and_carries_evidence(divs):
    dv = divs["normal"].simulated.diversion
    assert 1 <= len(dv.candidates) <= 3 and dv.alternate_airport in {c.airport for c in dv.candidates}
    c = next(x for x in dv.candidates if x.airport == dv.alternate_airport)
    assert c.suitable and c.reachable_normal and c.distance_method and c.trip_fuel_kg > 0 and c.flight_time_min > 0
    assert divs["normal"].workbook.airport.iata_code == dv.alternate_airport != dv.destination.airport      # plan domain = the alternate


# ------------------------------------------------------------------ status: computed, not assumed; a diversion is not an emergency
def test_status_is_computed_from_fuel_and_a_diversion_is_not_automatically_an_emergency(divs):
    st = {k: v.simulated.diversion.computed_status for k, v in divs.items()}
    assert st == {"normal": S.NORMAL, "minimum_fuel": S.MINIMUM_FUEL, "emergency": S.EMERGENCY}
    for k, sc in divs.items():
        sf = sc.simulated.flights[0]
        fa = fuel_account(sc, sf, sf.earliest_landing_min)
        assert not sc.simulated.diversion.destination.feasible                                  # all three are DIVERSIONS
        assert sf.status == st[k]
        if k == "normal":
            assert fa.contingency_margin_kg >= 0
        elif k == "minimum_fuel":
            assert fa.contingency_margin_kg < 0 <= fa.reserve_margin_kg
        else:
            assert fa.reserve_margin_kg < 0 <= fa.landing_fuel_kg


def test_a_declared_emergency_input_forces_emergency_status(scenario_session, rules, flights_df):
    sc = build_diversion_scenario(scenario_session, rules, flights_df, emergency_declared=True)
    assert sc.simulated.diversion.computed_status == S.EMERGENCY and sc.simulated.diversion.emergency_declared_input


def test_unreachable_alternates_escalate_with_per_candidate_reasons(scenario_session, rules, flights_df):
    with pytest.raises(DiversionUnavailable) as e:
        build_diversion_scenario(scenario_session, rules, flights_df, fuel_case="unreachable")
    assert "ESCALATE" in str(e.value) and e.value.candidates
    assert all(not c.suitable and any("not reachable" in r for r in c.reasons) for c in e.value.candidates)


# ------------------------------------------------------------------ validation
def test_status_must_be_declared_correctly(divs):
    sc = divs["emergency"]
    plan = good(sc)
    assert plan.declared_status == S.EMERGENCY and not fails(sc, plan)
    assert "DIV-STATUS" in fails(sc, with_status(plan, sc, S.NORMAL))                            # under-declaring an emergency is rejected
    assert "DIV-STATUS" in fails(sc, with_status(plan, sc, None))
    n = divs["normal"]
    assert "DIV-STATUS" in fails(n, with_status(good(n), n, S.EMERGENCY))                        # nor can an emergency be claimed to jump the queue


def test_the_plan_must_land_at_the_screened_alternate(divs):
    sc = divs["normal"]
    plan = good(sc)
    assert "DIV-ALT-SUITABLE" in fails(sc, make_plan(sc, "t", plan.flights, declared_status=plan.declared_status, airport="ATL"))
    assert "DIV-ALT-SUITABLE" not in fails(sc, plan)


def test_diversion_is_only_justified_when_the_destination_cannot_accept_the_aircraft(scenario_session, rules, flights_df):
    sc = build_diversion_scenario(scenario_session, rules, flights_df, dest_closed_until_min=1.0)    # destination reopens at once
    assert sc.simulated.diversion.destination.feasible and sc.simulated.diversion.computed_status == S.NORMAL
    r = [x for x in validate_plan(sc, run_fallback(sc).plan or reference_plan(sc)).results if x.constraint_id == "DIV-TRIGGER"][0]
    assert not r.passed and "not justified" in r.explanation
    assert run_fallback(sc).outcome == Outcome.NO_FEASIBLE_PLAN                                  # nothing valid: escalate


def test_emergency_lands_at_the_earliest_feasible_slot_and_ignores_convenience(divs):
    sc = divs["emergency"]
    plan = good(sc)
    sf = sc.simulated.flights[0]
    land = {f.flight_id: f for f in plan.flights}[sf.flight_id].landing_min
    assert land == pytest.approx(sf.earliest_landing_min)                                        # earliest slot, no hold
    late = plan.model_copy(deep=True)
    {f.flight_id: f for f in late.flights}[sf.flight_id].landing_min = land + 6.0
    assert "EMERG-PRIORITY" in {r.constraint_id for r in validate_plan(sc, late).failures}
    # the reserve rule is replaced (reserve already unavailable) by 'fuel stays positive'
    res = validate_plan(sc, plan).results
    mine = {r.constraint_id for r in res if r.flight_id == sf.flight_id}                         # the diverting aircraft
    others = {r.constraint_id for r in res if r.flight_id and r.flight_id != sf.flight_id}       # NORMAL traffic at the alternate keeps the reserve rule
    assert "FUEL-EMERGENCY" in mine and "FUEL-RESERVE" not in mine and "EMERG-PRIORITY" in mine
    assert "FUEL-RESERVE" in others and "FUEL-EMERGENCY" not in others
    sc2 = divs["normal"]
    assert "EMERG-PRIORITY" not in {r.constraint_id for r in validate_plan(sc2, good(sc2)).results}


def test_minimum_fuel_waives_contingency_only_and_keeps_the_final_reserve(divs):
    sc = divs["minimum_fuel"]
    plan = good(sc)
    res = {r.constraint_id: r for r in validate_plan(sc, plan).results if r.flight_id == sc.simulated.flights[0].flight_id}
    assert res["FUEL-CONTINGENCY"].passed and "waived" in res["FUEL-CONTINGENCY"].explanation and res["FUEL-RESERVE"].passed
    sf = sc.simulated.flights[0]
    fa = fuel_account(sc, sf, sf.earliest_landing_min)
    late = plan.model_copy(deep=True)
    {f.flight_id: f for f in late.flights}[sf.flight_id].landing_min = sf.earliest_landing_min + fa.max_hold_min_before_reserve + 1.0
    assert "FUEL-RESERVE" in {r.constraint_id for r in validate_plan(sc, late).failures}          # reserve is never waived


def test_enforcement_uses_the_computed_status_not_the_pilots_claim(divs):
    sc = divs["normal"]
    plan = good(sc)
    sf = sc.simulated.flights[0]
    fa = fuel_account(sc, sf, sf.earliest_landing_min)
    late = make_plan(sc, "t", [f.model_copy(update={"landing_min": sf.earliest_landing_min + fa.max_hold_min_before_contingency + 2}) if f.flight_id == sf.flight_id else f
                               for f in plan.flights], declared_status=S.MINIMUM_FUEL, airport=plan.airport)   # claims minimum fuel to dodge the contingency rule
    f = fails(sc, late)
    assert "FUEL-CONTINGENCY" in f and "DIV-STATUS" in f


# ------------------------------------------------------------------ the five agents
def test_diversion_participants_are_five_independent_agents(divs):
    parts = participants_for(divs["normal"])
    assert [p.agent_id for p in parts] == ["PILOT@DL1205", f"ATC@{divs['normal'].simulated.diversion.alternate_airport}",
                                           f"GROUND@{divs['normal'].simulated.diversion.alternate_airport}", "ATC@ATL", "GROUND@ATL"]
    assert {p.airport for p in parts if p.airport} == {divs["normal"].simulated.diversion.alternate_airport, "ATL"}


def test_agents_see_different_things_in_a_diversion(divs):
    sc = divs["emergency"]
    parts = participants_for(sc)
    mk = lambda p: AgentContext(role=p.role, agent=p, participants=parts, turn=1, seq=1, session_id="s", scenario=sc, historical=sc.historical,
                                messages=[], proposals=[], standing={}, allowed_types=[])
    pilot = agents.pilot.domain_view(mk(parts[0]))
    assert pilot["diversion"]["COMPUTED_STATUS (you must declare exactly this)"] == "EMERGENCY"
    assert pilot["diversion"]["option_A_original_destination"]["feasible"] is False and pilot["diversion"]["option_B_alternates (screened; a historical route alone does NOT make an airport suitable)"]
    alt_atc, dest_atc = agents.atc.domain_view(mk(parts[1])), agents.atc.domain_view(mk(parts[3]))
    assert "runways" in alt_atc and "role_in_this_session" in dest_atc and "runways" not in dest_atc
    assert "ORIGINAL DESTINATION" in dest_atc["role_in_this_session"] and "gates_and_stands" in agents.ground.domain_view(mk(parts[2]))
    assert "gates_and_stands" not in agents.ground.domain_view(mk(parts[4]))


@pytest.mark.parametrize("case", ["normal", "minimum_fuel", "emergency"])
def test_scripted_diversion_session_reaches_a_validated_consensus_at_the_alternate(divs, case, tmp_path):
    sc = divs[case]
    res = run_negotiation(sc, ScriptedClient(), participants=participants_for(sc), config=CFG, log_path=tmp_path / "e.db")
    assert res.consensus and res.turns_completed == 3 and len(res.messages) == 15 and len(res.votes) == 5
    assert res.final_plan.airport == sc.simulated.diversion.alternate_airport and res.final_plan.declared_status == sc.simulated.diversion.computed_status
    assert validate_plan(sc, res.final_plan).passed
    pilot = [m for m in res.messages if m.agent_id == "PILOT@DL1205"]
    assert pilot[0].report is not None and pilot[0].report_verified and pilot[0].report.fuel_status == sc.simulated.diversion.computed_status
    assert all(m.declared_status == sc.simulated.diversion.computed_status for m in pilot)
    dest = [m for m in res.messages if m.agent_id.endswith("@ATL")]
    assert len(dest) == 6 and all("cannot accept" in m.explanation for m in dest)


def test_destination_agents_contest_an_unjustified_diversion_and_the_session_escalates(scenario_session, rules, flights_df, tmp_path):
    sc = build_diversion_scenario(scenario_session, rules, flights_df, dest_closed_until_min=1.0)
    res = run_negotiation(sc, ScriptedClient(), participants=participants_for(sc), config=CFG, log_path=tmp_path / "e.db")
    assert not res.consensus and res.status == FinalStatus.NO_FEASIBLE_PLAN and res.fallback_reason == FallbackReason.NO_VALID_CANDIDATE
    assert res.fallback.search_status == SearchStatus.EXHAUSTED and res.final_plan is None
    assert any(m.type.value == "REJECT" and m.agent_id.endswith("@ATL") for m in res.messages) or all(not p.validation.passed for p in res.proposals)


def test_reference_plan_for_a_diversion_is_illustrative_and_invalid(divs):
    sc = divs["normal"]
    ref = reference_plan(sc)
    assert ref.source == "reference" and {"DIV-STATUS", "DIV-ALT-SUITABLE"} <= fails(sc, ref)
