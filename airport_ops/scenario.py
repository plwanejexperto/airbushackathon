"""Reproducible scenario: historical flight + workbook limits + seeded simulated conditions.

The scenario airport is the selected flight's DESTINATION (the flight is the disrupted arrival).
Historical delay minutes are only *context*: they decide whether a weather closure / late
ground crew is simulated, never how long the aircraft must hold.
"""
from __future__ import annotations

import hashlib
import json
import random
import zlib
from typing import Optional

from .schemas import (CapacitySubset, ConnectionPreference, FlightCrew, FuelState, GateResource, GroundCrew, HistoricalFlight,
                      Interval, RunwayResource, Scenario, SimAssumptions, SimFlight, SimulatedConditions,
                      TaxiSegment, WorkbookLimits)
from .workbook import AircraftClass, WakeCategory, WorkbookRules

MAX_MODELLED_RUNWAYS = 2
MAX_MODELLED_GATES = 3


class ScenarioError(ValueError):
    """Raised when a scenario cannot be built (e.g. airport missing from the workbook)."""


def _class_of(cat: WakeCategory) -> AircraftClass:
    return AircraftClass.WIDEBODY if cat in (WakeCategory.SUPER, WakeCategory.HEAVY) else AircraftClass.NARROWBODY


def scenario_version(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:12]


def build_scenario(h: HistoricalFlight, rules: WorkbookRules, assumptions: Optional[SimAssumptions] = None,
                   seed: Optional[int] = None) -> Scenario:
    a = assumptions or SimAssumptions()
    ap = rules.airports.get(h.destination_code)
    if ap is None:
        why = rules.invalid_airports.get(h.destination_code)
        raise ScenarioError(f"airport {h.destination_code} has no usable row in the workbook"
                            + (f" (row invalid: {why})" if why else " (not present)"))
    fc = rules.fuel_crew
    key = f"{h.flight_date}|{h.label}|{h.origin_code}|{h.destination_code}"
    seed = zlib.crc32(key.encode()) if seed is None else seed
    rng = random.Random(seed)

    # --- named subset of the airport's workbook capacity ---
    n_r = min(MAX_MODELLED_RUNWAYS, ap.active_runways)
    n_g = min(MAX_MODELLED_GATES, ap.gates)
    rw_ids = [f"RWY-{i + 1}" for i in range(n_r)]
    g_ids = [f"GATE-{i + 1}" for i in range(n_g)]
    g_classes = [AircraftClass.NARROWBODY, AircraftClass.WIDEBODY, AircraftClass.NARROWBODY][:n_g]
    subset = CapacitySubset(total_runways=ap.active_runways, modelled_runways=n_r, total_gates=ap.gates,
                            modelled_gates=n_g, spare_gates_at_any_time=ap.spare_gates_at_any_time)

    # --- simulated closures / occupancy (historical delays only influence *whether*) ---
    runways = [RunwayResource(id=r) for r in rw_ids]
    if h.weather_delay_min > 0:
        s = rng.randint(0, 8)
        runways[0].closures.append(Interval(start_min=s, end_min=s + rng.randint(10, 25), reason="simulated weather closure"))
    gates = [GateResource(id=g, terminal="T1" if i < 2 else "T2", max_aircraft_class=c)
             for i, (g, c) in enumerate(zip(g_ids, g_classes))]
    # workbook: spare_gates_at_any_time gates are free; the other modelled gates carry a resident aircraft
    for g in rng.sample(gates, max(0, n_g - min(ap.spare_gates_at_any_time, n_g))):
        g.resident_occupancy.append(Interval(start_min=0, end_min=rng.randint(25, 70), reason="simulated resident aircraft"))

    segs = [TaxiSegment(id=f"TWY-A{i + 1}", from_node=f"EXIT-{r}", to_node="HUB", minutes=rng.randint(2, 5))
            for i, r in enumerate(rw_ids)]
    segs += [TaxiSegment(id="TWY-C", from_node="HUB", to_node="APRON", minutes=2),
             TaxiSegment(id="TWY-D", from_node="HUB", to_node="APRON", minutes=3)]
    segs += [TaxiSegment(id=f"TWY-G{i + 1}", from_node="APRON", to_node=f"GATE-{g}", minutes=rng.randint(1, 2))
             for i, g in enumerate(g_ids)]
    blocked = rng.choice([s for s in segs if s.id in ("TWY-C", "TWY-D")])
    b0 = rng.randint(5, 15)
    blocked.blockages.append(Interval(start_min=b0, end_min=b0 + rng.randint(15, 30), reason="simulated taxiway blockage"))

    # --- the three flights ---
    eta_sel = rng.randint(18, 32)
    specs = [
        (h.label, "historical_selected", WakeCategory.LARGE, eta_sel),
        ("SIM-A101", "simulated", rng.choice([WakeCategory.HEAVY, WakeCategory.HEAVY, WakeCategory.LARGE, WakeCategory.SUPER]),
         max(12, eta_sel + rng.randint(-4, 4))),
        ("SIM-B202", "simulated", rng.choice([WakeCategory.LARGE, WakeCategory.SMALL, WakeCategory.HEAVY]),
         max(12, eta_sel + rng.randint(-4, 4))),
    ]
    orig_crews = ["GC-1", "GC-2", "GC-1"]
    flights: list[SimFlight] = []
    flight_crews: list[FlightCrew] = []
    used_gates: set[str] = set()
    hi_task = max(1, int(ap.min_turnaround_min - a.deboard_min))
    for (fid, src, cat, eta), gc in zip(specs, orig_crews):
        cls = _class_of(cat)
        compat = [g for g in gates if cls == AircraftClass.NARROWBODY or g.max_aircraft_class == AircraftClass.WIDEBODY]
        og = next((g for g in compat if g.id not in used_gates), compat[0] if compat else gates[0])
        used_gates.add(og.id)
        spare = rng.randint(*a.spare_fuel_min_range)
        rate = fc.hold_burn_kg_h(cls)
        burn = a.trip_burn_widebody_kg_h if cls == AircraftClass.WIDEBODY else a.trip_burn_narrowbody_kg_h
        trip = eta / 60.0 * burn
        # on board = trip + contingency (workbook %) + final reserve (workbook min) + spare holding
        onboard = trip * (1 + fc.contingency_fraction) + rate * (fc.final_reserve_min + spare) / 60.0
        flights.append(SimFlight(
            flight_id=fid, source=src, wake_category=cat, aircraft_class=cls,
            passengers=rng.randint(100, 220) if cls == AircraftClass.NARROWBODY else rng.randint(250, 420),
            priority=rng.randint(1, 5), earliest_landing_min=float(eta), original_runway_id=rw_ids[0],
            original_gate_id=og.id, original_ground_crew_id=gc, flight_crew_id=f"FC-{fid}",
            service_task_min=float(rng.randint(min(20, hi_task), hi_task)),
            fuel=FuelState(trip_fuel_remaining_kg=trip, fuel_on_board_kg=onboard,
                           spare_fuel_min_at_eta=float(spare))))
        remaining_duty = rng.randint(180, 320)  # minutes of duty left at t=0
        flight_crews.append(FlightCrew(id=f"FC-{fid}", home_flight_id=fid,
                                       report_time_min=remaining_duty - fc.flight_crew_max_duty_h * 60))

    shift_h = fc.ground_shift_h * 60
    gcs = [GroundCrew(id="GC-1", shift_start_min=rng.randint(40, 120) - shift_h,
                      available_from_min=float(rng.randint(5, 20)) if h.late_aircraft_delay_min > 0 else 0.0),
           GroundCrew(id="GC-2", shift_start_min=rng.randint(100, 200) - shift_h)]

    sel = flights[0]
    conn = ConnectionPreference(flight_id=sel.flight_id, connecting_passengers=rng.randint(10, 40),
                                preferred_gate_in_by_min=sel.earliest_landing_min + rng.randint(20, 45))
    sim = SimulatedConditions(seed=seed, assumptions=a, flights=flights, runways=runways, taxi_segments=segs,
                              gates=gates, flight_crews=flight_crews, ground_crews=gcs, connection=conn)
    wb = WorkbookLimits(airport=ap, wake=rules.wake, fuel_crew=fc, capacity_subset=subset)
    content = {"historical": h.model_dump(mode="json"), "workbook": wb.model_dump(mode="json"),
               "simulated": sim.model_dump(mode="json")}
    return Scenario(scenario_id=f"scn-{seed:08x}", scenario_version=scenario_version(content),
                    historical=h, workbook=wb, simulated=sim)


def rehash_scenario(sc: Scenario) -> Scenario:
    """Return a copy whose scenario_version matches its (possibly edited) content. Any change to a scenario MUST go
    through this: plans, proposals and caches are bound to the version."""
    content = {"historical": sc.historical.model_dump(mode="json"), "workbook": sc.workbook.model_dump(mode="json"),
               "simulated": sc.simulated.model_dump(mode="json")}
    return sc.model_copy(update={"scenario_version": scenario_version(content)}, deep=True)
