"""Fuel accounting from scenario start (t=0) to touchdown. Single source of truth for the validator,
the Pilot agent's tools and the tests.

Units: time in MINUTES from scenario start; fuel in KG; burn rates in KG PER HOUR (so every burn is
``rate_kg_per_h * minutes / 60``).

Phases and where each term comes from
-------------------------------------
1. En route, t=0 -> nominal ETA (``SimFlight.earliest_landing_min``)
       trip_fuel_kg = ETA/60 * trip_burn_kg_h              [simulated; SimAssumptions.trip_burn_*]
   Burned exactly ONCE, whether or not the aircraft later holds. It is never re-added to holding.
2. Airborne holding, ETA -> actual touchdown
       hold_min      = max(0, landing_min - ETA)
       hold_burn_kg  = hold_burn_kg_h * hold_min / 60       [workbook: "Holding fuel burn (class)"]
   Landing before ETA is not possible (LAND-EARLIEST), so it never creates fuel credit.
3. Touchdown
       landing_fuel_kg = fuel_on_board_kg - trip_fuel_kg - hold_burn_kg

Required at touchdown (workbook fuel_and_crew)
       final_reserve_kg = hold_burn_kg_h * final_reserve_min / 60      "Final fuel reserve: 30 min of holding fuel"
       contingency_kg   = contingency_fraction * trip_fuel_kg          "Contingency fuel: 5 % of trip fuel"
   FUEL-RESERVE     : landing_fuel_kg >= final_reserve_kg                       (hard safety floor)
   FUEL-CONTINGENCY : landing_fuel_kg >= final_reserve_kg + contingency_kg      (contingency is carried, not spendable)

Initial state (scenario.py) is built so that nothing is counted twice:
       fuel_on_board = trip + contingency + final_reserve + spare_min * hold_burn_kg_h / 60
   hence at the nominal ETA the margin above reserve+contingency is exactly the simulated spare, and
       max hold before contingency is touched = spare_min          (minutes)
       max hold before the final reserve is touched = spare_min + contingency_kg / hold_burn_kg_h * 60
No rounding is applied to stored fuel values, so these boundaries are exact.

4. After touchdown, landing -> gate arrival (taxi + ground waiting)
       ground_burn_kg = hold_burn_kg_h * ground_burn_fraction_of_hold * (gate_in - landing) / 60     [ASSUMPTION: the fraction]
       fuel_at_gate   = landing_fuel_kg - ground_burn_kg
   FUEL-GROUND : fuel_at_gate >= final_reserve_kg   (assumption, NORMAL status only: the final reserve must also survive the ground phase, so a long
   ground wait or a longer taxi route that exhausts the reserve is rejected; a shorter route that must wait for a blockage can lose
   to a longer route that does not).

Not modelled (stated, not hidden): APU fuel, alternate fuel (outside diversion sessions), weight-dependent burn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .schemas import Scenario, SimFlight


@dataclass(frozen=True)
class FuelAccount:
    flight_id: str
    hold_burn_kg_h: float
    fuel_on_board_kg: float
    trip_fuel_kg: float
    hold_min: float
    hold_burn_kg: float
    landing_fuel_kg: float
    final_reserve_kg: float
    contingency_kg: float
    reserve_margin_kg: float          # landing - final_reserve         (>= 0 required)
    contingency_margin_kg: float      # landing - reserve - contingency (>= 0 required)
    max_hold_min_before_contingency: float  # hold allowed from ETA without touching contingency
    max_hold_min_before_reserve: float      # hold allowed from ETA without touching the final reserve

    def as_dict(self) -> dict:
        return asdict(self)


def fuel_account(sc: Scenario, sf: SimFlight, landing_min: float) -> FuelAccount:
    fc = sc.workbook.fuel_crew
    rate = fc.hold_burn_kg_h(sf.aircraft_class)
    trip = sf.fuel.trip_fuel_remaining_kg
    hold = max(0.0, landing_min - sf.earliest_landing_min)
    hold_burn = rate * hold / 60.0
    landing = sf.fuel.fuel_on_board_kg - trip - hold_burn
    reserve = rate * fc.final_reserve_min / 60.0
    cont = fc.contingency_fraction * trip
    usable_c = sf.fuel.fuel_on_board_kg - trip - reserve - cont   # kg of holding fuel above reserve + contingency
    usable_r = sf.fuel.fuel_on_board_kg - trip - reserve          # kg of holding fuel above the final reserve
    return FuelAccount(
        flight_id=sf.flight_id, hold_burn_kg_h=rate, fuel_on_board_kg=sf.fuel.fuel_on_board_kg, trip_fuel_kg=trip,
        hold_min=hold, hold_burn_kg=hold_burn, landing_fuel_kg=landing, final_reserve_kg=reserve, contingency_kg=cont,
        reserve_margin_kg=landing - reserve, contingency_margin_kg=landing - reserve - cont,
        max_hold_min_before_contingency=usable_c / rate * 60.0, max_hold_min_before_reserve=usable_r / rate * 60.0)


def latest_landing_min(sc: Scenario, sf: SimFlight, *, protect_contingency: bool = True) -> float:
    """Latest touchdown time (minutes from scenario start) that still passes the fuel checks."""
    a = fuel_account(sc, sf, sf.earliest_landing_min)
    return sf.earliest_landing_min + (a.max_hold_min_before_contingency if protect_contingency
                                      else a.max_hold_min_before_reserve)


def ground_burn_kg(sc: Scenario, sf: SimFlight, landing_min: float, gate_in_min: float) -> float:
    rate = sc.workbook.fuel_crew.hold_burn_kg_h(sf.aircraft_class) * sc.simulated.assumptions.ground_burn_fraction_of_hold
    return rate * max(0.0, gate_in_min - landing_min) / 60.0


def gate_fuel_margin_kg(sc: Scenario, sf: SimFlight, landing_min: float, gate_in_min: float) -> float:
    """Fuel at gate arrival above the final reserve (kg)."""
    fa = fuel_account(sc, sf, landing_min)
    return fa.reserve_margin_kg - ground_burn_kg(sc, sf, landing_min, gate_in_min)
