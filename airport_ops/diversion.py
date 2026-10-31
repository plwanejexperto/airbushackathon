"""Scenario 3: DIVERSION or EMERGENCY (first prototype: ONE destination and ONE alternate airport).

How it is built (everything not from the dataset/workbook is labelled SIMULATED / ASSUMPTION):
  1. Aircraft position at decision time: remaining time to the destination is the flight's ETA; with an assumed cruise speed that is a
     distance, placed on the origin->destination route (route length from flights.csv).
  2. Alternates: airports that flights.csv connects to the destination are CANDIDATES only. Distance from the aircraft's current POSITION
     (not from the destination) comes from triangle geometry on the route distances in flights.csv (origin-dest, origin-alt, dest-alt); if
     the origin-alternate route is missing, a labelled conservative assumption is used. A historical route alone never makes an airport
     suitable: suitability also needs workbook capacity/size, fuel reach and acceptable (simulated) weather.
  3. Fuel and status are computed from distance/time/burn: NORMAL (reserve + contingency intact at the alternate), MINIMUM_FUEL (contingency
     would be consumed, reserve intact) or EMERGENCY (reserve cannot be protected). A diversion does NOT imply an emergency.
  4. The negotiated plan is the aircraft's ARRIVAL at the alternate: a normal Scenario for that airport (workbook limits, simulated runways/
     gates/crews/traffic) in which the diverting aircraft's ETA and fuel come from step 3. All existing checks therefore apply unchanged.
"""
from __future__ import annotations

import zlib

import pandas as pd

from .fuel import fuel_account, latest_landing_min
from .messages import FlightStatus
from .scenario import build_scenario, fuel_state, rehash_scenario
from .schemas import (AlternateCandidate, DestinationStatus, DiversionContext, FuelState, Interval, Scenario)
from .workbook import AircraftClass, WorkbookRules

NM_PER_MILE = 0.868976
_ROUTE_CACHE: dict[int, dict] = {}


class DiversionUnavailable(Exception):
    """No reachable and suitable alternate exists: ESCALATE (nothing to negotiate)."""

    def __init__(self, message: str, candidates: list[AlternateCandidate], detail: dict):
        super().__init__(message)
        self.candidates, self.detail = candidates, detail


def route_distances(df: pd.DataFrame) -> dict[frozenset, float]:
    """Unordered airport pair -> route distance in nautical miles (median DISTANCE in flights.csv)."""
    key = id(df)
    if key not in _ROUTE_CACHE:
        if len(_ROUTE_CACHE) >= 4:
            _ROUTE_CACHE.clear()
        g = df.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]).DISTANCE.median()
        d: dict[frozenset, float] = {}
        for (o, t), v in g.items():
            if o != t:
                d.setdefault(frozenset((o, t)), float(v) * NM_PER_MILE)
        _ROUTE_CACHE[key] = d
    return _ROUTE_CACHE[key]


def distance_from_position(D: float, s: float, a: float | None, b: float) -> tuple[float, str]:
    """Distance from the aircraft (on the origin->destination line, s nm from the origin) to an alternate.
    D = origin-destination, a = origin-alternate (may be None), b = destination-alternate (all route distances, nm)."""
    if a is None:
        return (D - s) + b, ("ASSUMPTION: no origin-alternate route in flights.csv, so the distance is bounded by flying on to the "
                             "destination area and then to the alternate (remaining + destination-alternate); conservative")
    x = (a * a - b * b + D * D) / (2 * D)                  # triangle geometry in the plane of the three airports
    y2 = a * a - x * x
    note = "triangle geometry from flights.csv route distances (origin-destination, origin-alternate, destination-alternate)"
    if y2 < 0:
        y2, note = 0.0, note + "; inconsistent triangle clamped (route distances are not great-circle)"
    return ((x - s) ** 2 + y2) ** 0.5, note


def _weather_ok(airport: str, date) -> bool:
    return zlib.crc32(f"{airport}|{date}".encode()) % 7 != 0       # SIMULATED: ~1 in 7 alternates has unusable weather that day


def position_of(dest_sc: Scenario) -> dict:
    sf, h, a = dest_sc.simulated.flights[0], dest_sc.historical, dest_sc.simulated.assumptions
    remaining = sf.earliest_landing_min / 60.0 * a.cruise_speed_kt
    total = (h.distance_mi or 0) * NM_PER_MILE
    s = max(0.0, total - remaining)
    return {"SIMULATED": True, "minutes_to_destination": sf.earliest_landing_min, "assumed_cruise_speed_kt": a.cruise_speed_kt,
            "route_distance_nm (flights.csv)": round(total, 1), "distance_to_destination_nm": round(remaining, 1),
            "distance_from_origin_nm": round(s, 1)}


def _cruise_burn(sc: Scenario, cls: AircraftClass) -> float:
    a = sc.simulated.assumptions
    return a.trip_burn_widebody_kg_h if cls == AircraftClass.WIDEBODY else a.trip_burn_narrowbody_kg_h


def screen_alternates(dest_sc: Scenario, rules: WorkbookRules, df: pd.DataFrame, onboard_kg: float | None = None, limit: int = 3
                      ) -> list[AlternateCandidate]:
    """The small set of candidate alternates, nearest to the aircraft's POSITION first, each with explicit suitability evidence."""
    sf, h = dest_sc.simulated.flights[0], dest_sc.historical
    fc = dest_sc.workbook.fuel_crew
    pos = position_of(dest_sc)
    routes = route_distances(df)
    D, s = pos["route_distance_nm (flights.csv)"], pos["distance_from_origin_nm"]
    onboard = onboard_kg if onboard_kg is not None else sf.fuel.fuel_on_board_kg
    rate = fc.hold_burn_kg_h(sf.aircraft_class)
    reserve = rate * fc.final_reserve_min / 60.0
    out = []
    for key, b in routes.items():
        if h.destination_code not in key:
            continue
        alt = next(iter(key - {h.destination_code}), None)
        if alt is None or alt == h.origin_code or alt not in rules.airports:
            continue
        a = routes.get(frozenset((h.origin_code, alt)))
        dist, method = distance_from_position(D, s, a, b) if D > 0 else (b, "ASSUMPTION: route distance unknown; destination-alternate used")
        t_min = dist / dest_sc.simulated.assumptions.cruise_speed_kt * 60.0
        trip = t_min / 60.0 * _cruise_burn(dest_sc, sf.aircraft_class)
        landing = onboard - trip
        margin = landing - reserve - fc.contingency_fraction * trip
        lim = rules.airports[alt]
        size_ok = sf.aircraft_class == AircraftClass.NARROWBODY or lim.size_class in ("Large hub", "Medium hub")
        wx = _weather_ok(alt, h.flight_date)
        reasons = []
        if not size_ok:
            reasons.append(f"{lim.size_class} is too small for a widebody (assumption: widebody needs a Medium or Large hub)")
        if not wx:
            reasons.append("simulated weather unusable")
        if margin < 0:
            reasons.append(f"reserve + contingency cannot be protected (short by {-margin:.0f} kg)")
        if landing < reserve:
            reasons.append("final reserve cannot be protected")
        if landing < 0:
            reasons.append("not reachable on the fuel on board")
        out.append(AlternateCandidate(
            airport=alt, name=lim.airport, size_class=lim.size_class, route_to_destination_nm=round(b, 1),
            route_from_origin_nm=round(a, 1) if a else None, distance_from_position_nm=round(dist, 1), distance_method=method,
            flight_time_min=round(t_min, 2), trip_fuel_kg=round(trip, 1), landing_fuel_margin_over_reserve_contingency_kg=round(margin, 1),
            reachable_normal=margin >= 0, reachable_min_fuel=landing >= reserve, reachable_emergency=landing >= 0,
            simulated_weather_ok=wx, size_ok=size_ok, suitable=size_ok and wx and landing >= 0, reasons=reasons))
    out.sort(key=lambda c: (c.distance_from_position_nm, c.airport))
    return out[:limit]


def classify_status(cand: AlternateCandidate, emergency_declared: bool = False) -> FlightStatus:
    if emergency_declared:
        return FlightStatus.EMERGENCY
    if cand.reachable_normal:
        return FlightStatus.NORMAL
    if cand.reachable_min_fuel:
        return FlightStatus.MINIMUM_FUEL
    return FlightStatus.EMERGENCY


STATUS_RULE = ("Status is computed from the projected landing fuel at the alternate with NO further delay: NORMAL = reserve + contingency intact; "
               "MINIMUM_FUEL = contingency consumed but final reserve intact (contingency waived for the diverting aircraft only); "
               "EMERGENCY = final reserve cannot be protected or an emergency is declared (rule becomes: land at the earliest feasible slot "
               "with fuel > 0; gates, crews and other flights' convenience must not delay it). The Pilot must DECLARE the computed status.")


def destination_status(dest_sc: Scenario, closed_until_min: float | None) -> DestinationStatus:
    sf = dest_sc.simulated.flights[0]
    latest = latest_landing_min(dest_sc, sf)
    closures = []
    if closed_until_min is not None:
        closures = [Interval(start_min=0, end_min=closed_until_min, reason="SIMULATED: all runways closed (weather)")]
    earliest = max(sf.earliest_landing_min, closed_until_min or 0.0)
    ap = dest_sc.workbook.airport
    return DestinationStatus(airport=ap.iata_code, name=ap.airport, closures=closures, earliest_feasible_landing_min=earliest,
                             latest_landing_min_by_fuel=latest, feasible=earliest <= latest + 1e-9, limits=ap.model_dump(),
                             gates_note="gate/stand state at the destination is irrelevant while no runway is open")


def build_diversion_scenario(dest_sc: Scenario, rules: WorkbookRules, df: pd.DataFrame, *, fuel_case: str = "normal",
                             emergency_declared: bool = False, dest_closed_until_min: float | None = None) -> Scenario:
    """The ALTERNATE airport's scenario for the diverting aircraft. fuel_case: normal | minimum_fuel | emergency | unreachable
    (only sets how much fuel is on board relative to the chosen alternate's needs; the status is then COMPUTED, not assumed)."""
    sf0, h = dest_sc.simulated.flights[0], dest_sc.historical
    a, fc = dest_sc.simulated.assumptions, dest_sc.workbook.fuel_crew
    latest = latest_landing_min(dest_sc, sf0)
    closed = dest_closed_until_min if dest_closed_until_min is not None else round(latest + 25.0, 1)
    dest = destination_status(dest_sc, closed)
    onboard0 = sf0.fuel.fuel_on_board_kg
    cands = screen_alternates(dest_sc, rules, df, onboard0)
    pos = position_of(dest_sc)
    chosen = next((c for c in cands if c.suitable and c.reachable_emergency), None)
    if fuel_case != "normal" and cands:      # re-screen with the case's fuel on board (relative to the nearest candidate's need)
        base = cands[0]
        rate = fc.hold_burn_kg_h(sf0.aircraft_class)
        reserve = rate * fc.final_reserve_min / 60.0
        cont = fc.contingency_fraction * base.trip_fuel_kg
        onboard0 = {"minimum_fuel": base.trip_fuel_kg + reserve + 0.5 * cont, "emergency": base.trip_fuel_kg + 0.5 * reserve,
                    "unreachable": base.trip_fuel_kg * 0.5}[fuel_case]
        cands = screen_alternates(dest_sc, rules, df, onboard0)
        chosen = next((c for c in cands if c.suitable and c.reachable_emergency), None)
    if chosen is None:
        raise DiversionUnavailable("no reachable and suitable alternate: ESCALATE", cands,
                                   {"position": pos, "destination": dest.model_dump(mode="json"), "onboard_fuel_kg": onboard0})
    status = classify_status(chosen, emergency_declared)

    # ---- the alternate airport's arrival scenario (same machinery as every other session)
    alt_h = h.model_copy(update={"destination_code": chosen.airport, "destination_name": chosen.name})
    alt = build_scenario(alt_h, rules, a)
    alt.historical = h                                       # the disruption record stays the ORIGINAL flight row
    s = alt.simulated
    eta_alt = round(max(chosen.flight_time_min, 5.0), 1)
    gen_eta0 = s.flights[0].earliest_landing_min
    div = s.flights[0]
    div.wake_category, div.aircraft_class, div.priority, div.status = sf0.wake_category, sf0.aircraft_class, sf0.priority, status
    div.earliest_landing_min = eta_alt
    trip = chosen.trip_fuel_kg
    spare_min = max(0.0, (onboard0 - trip - fc.hold_burn_kg_h(div.aircraft_class) * fc.final_reserve_min / 60.0 - fc.contingency_fraction * trip)
                    / fc.hold_burn_kg_h(div.aircraft_class) * 60.0)
    div.fuel = FuelState(trip_fuel_remaining_kg=trip, fuel_on_board_kg=onboard0, spare_fuel_min_at_eta=spare_min)
    for f in s.flights[1:]:                                   # ALT's own traffic keeps its relative timing around the diverting aircraft
        f.earliest_landing_min = round(eta_alt + (f.earliest_landing_min - gen_eta0), 1)
        f.fuel = fuel_state(f.earliest_landing_min, f.aircraft_class, f.fuel.spare_fuel_min_at_eta, a, fc)
    for c in s.connections:
        f = next(x for x in s.flights if x.flight_id == c.flight_id)
        c.preferred_gate_in_by_min = round(f.earliest_landing_min + 35, 1)
    if s.connection is not None:
        s.connection = next(c for c in s.connections if c.flight_id == s.connection.flight_id)
    crew = next(c for c in s.flight_crews if c.id == div.flight_crew_id)
    dcrew = next(c for c in dest_sc.simulated.flight_crews if c.id == sf0.flight_crew_id)
    crew.report_time_min = dcrew.report_time_min             # same crew, same duty clock as at the destination
    s.diversion = DiversionContext(
        diverting_flight_id=div.flight_id, origin_airport=h.origin_code, destination=dest, alternate_airport=chosen.airport, position=pos,
        candidates=cands, computed_status=status, emergency_declared_input=emergency_declared, status_rule=STATUS_RULE,
        assumptions=[f"cruise speed {a.cruise_speed_kt:g} kt converts remaining minutes to distance (SIMULATED position)",
                     "en-route burn uses trip_burn_*_kg_h (SIMULATED); holding burn and reserves from the workbook",
                     "alternate weather, slots and gate occupancy are SIMULATED; capacity/size limits come from the workbook",
                     "the diversion is decided at scenario start (no further holding at the destination)",
                     "competing arrivals at the alternate are deterministic traffic with no Pilot agent (first prototype)"])
    return rehash_scenario(alt)
