import pytest

from airport_ops import data
from airport_ops.fallback import Outcome, SearchStatus, plan_cost, CostWeights, run_fallback
from airport_ops.planner import Planner
from airport_ops.scenario import build_scenario
from airport_ops.schemas import Interval
from airport_ops.validate import validate_plan
from airport_ops.workbook import AircraftClass as AC, WakeCategory as W


def test_fallback_selects_valid_plan_and_is_deterministic(scenario):
    a, b = run_fallback(scenario), run_fallback(scenario)
    assert a.outcome == Outcome.FALLBACK_SELECTED and a.search_status == SearchStatus.EXHAUSTED
    assert a.plan == b.plan and a.cost == b.cost and a.reasons == b.reasons
    rep = validate_plan(scenario, a.plan)  # independent re-validation
    assert rep.passed and a.validation.passed
    assert a.candidates_total == a.candidates_validated + a.candidates_pruned >= a.feasible_count > 0
    assert a.plan.proposal_id.startswith("prop-") and a.plan.scenario_version == scenario.scenario_version
    assert any("lowest total cost" in r for r in a.reasons)


def test_safety_is_a_filter_not_a_cost(scenario):
    """The naive plan has far lower delay cost but violates hard constraints: it must not be chosen."""
    naive = Planner(scenario).naive_plan()
    res = run_fallback(scenario)
    assert not validate_plan(scenario, naive).passed
    assert sum(plan_cost(scenario, naive, CostWeights()).values()) < res.cost
    assert res.plan.proposal_id != naive.proposal_id and res.validation.passed


def test_infeasible_by_gate_class_is_proven_exhausted(scenario):
    sc = scenario.model_copy(deep=True)
    sc.simulated.flights[1].wake_category = W.HEAVY
    sc.simulated.flights[1].aircraft_class = AC.WIDEBODY
    for g in sc.simulated.gates:
        g.max_aircraft_class = AC.NARROWBODY
    res = run_fallback(sc)
    assert res.outcome == Outcome.NO_FEASIBLE_PLAN and res.search_status == SearchStatus.EXHAUSTED
    assert res.plan is None and res.rejection_by_constraint["GATE-COMPAT"] == res.candidates_total
    assert any("exhausted" in r.lower() for r in res.reasons)


def test_infeasible_by_fuel_when_every_runway_stays_closed(scenario):
    sc = scenario.model_copy(deep=True)
    for r in sc.simulated.runways:
        r.closures = [Interval(start_min=0, end_min=400, reason="all closed")]
    res = run_fallback(sc)
    assert res.outcome == Outcome.NO_FEASIBLE_PLAN and res.search_status == SearchStatus.EXHAUSTED
    rej = res.rejection_by_constraint
    assert rej.get("FUEL-RESERVE", 0) + rej.get("FUEL-CONTINGENCY", 0) > 0
    assert set(rej) <= {"FUEL-RESERVE", "FUEL-CONTINGENCY", "GATE-COMPAT"}  # GATE-COMPAT = static class pre-filter
    assert sum(rej.values()) == res.candidates_total


def test_timeout_is_distinct_from_exhaustion(scenario):
    t = run_fallback(scenario, time_budget_s=-1)   # budget already spent before first candidate
    assert t.outcome == Outcome.NO_FEASIBLE_PLAN and t.search_status == SearchStatus.TIMEOUT
    assert t.candidates_validated == 0 and any("NOT proven" in r for r in t.reasons)
    p = run_fallback(scenario, max_evaluations=1)
    assert p.search_status == SearchStatus.TIMEOUT and p.candidates_validated == 1
    if p.plan:  # best-so-far is returned but flagged as not proven optimal
        assert any("TIMED OUT" in r for r in p.reasons) and p.validation.passed


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sample_of_real_flights_gets_a_valid_plan(rules, flights_df, seed):
    for _, row in flights_df.sample(8, random_state=seed).iterrows():
        sc = build_scenario(data.to_historical(row), rules)
        res = run_fallback(sc)
        assert res.outcome == Outcome.FALLBACK_SELECTED, (row.FLIGHT_LABEL, res.reasons)
        assert validate_plan(sc, res.plan).passed
