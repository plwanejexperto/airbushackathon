"""Review of the two stage-2 assumptions: fuel accounting and airport-wide vs per-runway capacity."""
import pytest

from conftest import mk
from airport_ops import data
from airport_ops.capacity import airport_limit, per_runway_limit
from airport_ops.fuel import fuel_account, latest_landing_min
from airport_ops.scenario import build_scenario
from airport_ops.schemas import SimAssumptions
from airport_ops.validate import validate_plan
from pydantic import ValidationError


def results(sc, plan, cid):
    return [r for r in validate_plan(sc, plan).results if r.constraint_id == cid]


def test_trip_fuel_is_burned_once_and_initial_state_has_no_double_counting(scenario):
    fc = scenario.workbook.fuel_crew
    for sf in scenario.simulated.flights:
        a = fuel_account(scenario, sf, sf.earliest_landing_min)
        r = a.hold_burn_kg_h
        assert a.hold_min == 0 and a.hold_burn_kg == 0
        # at nominal ETA exactly trip fuel has been burned (once): on_board - landing == trip
        assert sf.fuel.fuel_on_board_kg - a.landing_fuel_kg == pytest.approx(sf.fuel.trip_fuel_remaining_kg, abs=1e-9)
        # what is left is reserve + contingency + spare, each counted once
        spare = sf.fuel.spare_fuel_min_at_eta
        expect = r * fc.final_reserve_min / 60 + fc.contingency_fraction * a.trip_fuel_kg + r * spare / 60
        assert a.landing_fuel_kg == pytest.approx(expect, abs=1e-9)
        assert a.contingency_margin_kg == pytest.approx(r * spare / 60, abs=1e-9)
        assert a.max_hold_min_before_contingency == pytest.approx(spare, abs=1e-9)  # hold allowance == spare minutes


def test_holding_uses_workbook_rate_in_consistent_units(scenario):
    sf = scenario.simulated.flights[0]
    fc = scenario.workbook.fuel_crew
    rate = fc.hold_burn_narrowbody_kg_h  # DL1205 is narrowbody; 2400 kg per HOUR from the workbook
    a = fuel_account(scenario, sf, sf.earliest_landing_min + 30)  # 30 MINUTES of holding
    assert a.hold_burn_kg == pytest.approx(rate * 30 / 60) == pytest.approx(1200.0)
    base = fuel_account(scenario, sf, sf.earliest_landing_min)
    assert base.landing_fuel_kg - a.landing_fuel_kg == pytest.approx(1200.0)
    # trip fuel does not change with the hold
    assert a.trip_fuel_kg == base.trip_fuel_kg


def test_required_reserve_is_protected_at_the_exact_boundary(good):
    sc, plan = good
    f1 = sc.simulated.flights[0]
    acc = fuel_account(sc, f1, f1.earliest_landing_min)

    def with_landing(t):
        p = plan.model_copy(deep=True)
        p.flights[0].landing_min = t
        return p

    t_c = f1.earliest_landing_min + acc.max_hold_min_before_contingency
    t_r = f1.earliest_landing_min + acc.max_hold_min_before_reserve
    assert t_c == pytest.approx(latest_landing_min(sc, f1))
    # at the contingency boundary both pass; 0.01 min later contingency (not reserve) fails
    assert results(sc, with_landing(t_c), "FUEL-CONTINGENCY")[0].passed
    r = with_landing(t_c + 0.01)
    assert not results(sc, r, "FUEL-CONTINGENCY")[0].passed and results(sc, r, "FUEL-RESERVE")[0].passed
    # at the reserve boundary the final reserve is exactly intact; 0.01 min later it is breached
    assert fuel_account(sc, f1, t_r).reserve_margin_kg == pytest.approx(0.0, abs=1e-6)
    assert results(sc, with_landing(t_r), "FUEL-RESERVE")[0].passed
    assert not results(sc, with_landing(t_r + 0.01), "FUEL-RESERVE")[0].passed


def test_every_flight_at_nominal_eta_has_reserve_and_contingency_intact(rules, flights_df):
    for _, row in flights_df.sample(40, random_state=7).iterrows():
        sc = build_scenario(data.to_historical(row), rules)
        for sf in sc.simulated.flights:
            a = fuel_account(sc, sf, sf.earliest_landing_min)
            lo, hi = sc.simulated.assumptions.spare_fuel_min_range
            assert a.reserve_margin_kg > 0 and a.contingency_margin_kg > 0
            assert lo - 1e-9 <= a.max_hold_min_before_contingency <= hi + 1e-9


def test_airport_wide_limit_is_workbook_and_per_runway_share_is_separate_assumption(scenario):
    assert airport_limit(scenario) == 90 == scenario.workbook.airport.max_runway_ops_per_hour  # ATL, workbook value
    assert per_runway_limit(scenario) == 90 // 4                                              # default even_split
    a = scenario.simulated.assumptions
    assert a.per_runway_share_mode == "even_split"  # stage-2 default unchanged


def test_per_runway_share_is_configurable_fixed_or_disabled(good):
    sc, plan = good
    sc.workbook.airport = sc.workbook.airport.model_copy(update={"max_runway_ops_per_hour": 40})
    # disabled: no RWY-CAP-RUNWAY result at all, airport-wide check still runs
    sc.simulated.assumptions = sc.simulated.assumptions.model_copy(update={"per_runway_share_mode": "disabled"})
    assert per_runway_limit(sc) is None
    assert not results(sc, plan, "RWY-CAP-RUNWAY") and results(sc, plan, "RWY-CAP-AIRPORT")[0].passed
    # fixed 1/h per runway: RWY-1 has two landings -> fails; airport-wide (40/h) still passes
    sc.simulated.assumptions = sc.simulated.assumptions.model_copy(
        update={"per_runway_share_mode": "fixed", "per_runway_fixed_ops_per_hour": 1})
    assert per_runway_limit(sc) == 1
    assert not all(r.passed for r in results(sc, plan, "RWY-CAP-RUNWAY"))
    assert results(sc, plan, "RWY-CAP-AIRPORT")[0].passed


def test_fixed_mode_requires_a_value():
    with pytest.raises(ValidationError, match="per_runway_fixed_ops_per_hour"):
        SimAssumptions(per_runway_share_mode="fixed")
