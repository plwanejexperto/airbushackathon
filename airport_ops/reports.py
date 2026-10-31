"""Pilot report: arrival window, fuel margin, maximum acceptable holding time. Computed by code (fuel.py); an agent's own claim is
compared with it and any discrepancy is recorded (the code-computed values are authoritative)."""
from __future__ import annotations

from .canonical import quantise
from .fuel import fuel_account, latest_landing_min
from .messages import AircraftReport, DepartureReadiness, FlightStatus
from .schemas import Scenario


def pilot_report(sc: Scenario, flight_id: str) -> AircraftReport:
    sf = next(f for f in sc.simulated.flights if f.flight_id == flight_id)
    fa = fuel_account(sc, sf, sf.earliest_landing_min)
    conn = sc.simulated.connection_for(flight_id)
    return AircraftReport(
        flight_id=flight_id, earliest_landing_min=sf.earliest_landing_min,
        latest_landing_min=latest_landing_min(sc, sf), max_holding_min=fa.max_hold_min_before_contingency,
        contingency_margin_kg=fa.contingency_margin_kg, reserve_margin_kg=fa.reserve_margin_kg,
        preferred_gate_in_by_min=conn.preferred_gate_in_by_min if conn else None,
        connecting_passengers=conn.connecting_passengers if conn else 0, fuel_status=sf.status)


_FIELDS = {"earliest_landing_min": "landing_min", "latest_landing_min": "landing_min", "max_holding_min": "landing_min",
           "contingency_margin_kg": "contingency_margin_kg", "reserve_margin_kg": "reserve_margin_kg",
           "preferred_gate_in_by_min": "gate_in_min"}


def compare_reports(claimed: AircraftReport, actual: AircraftReport) -> list[str]:
    """Human-readable discrepancies between a Pilot's claimed report and the code-computed one (canonical precision)."""
    out = []
    if claimed.flight_id != actual.flight_id:
        out.append(f"flight_id {claimed.flight_id!r} != {actual.flight_id!r}")
    for f, q in _FIELDS.items():
        a, b = getattr(claimed, f), getattr(actual, f)
        if a is None and b is None:
            continue
        if a is None or b is None or quantise(q, a) != quantise(q, b):
            out.append(f"{f}: claimed {a} vs computed {b}")
    if claimed.fuel_status != actual.fuel_status:
        out.append(f"fuel_status: claimed {claimed.fuel_status.value} vs computed {actual.fuel_status.value}")
    return out


def departure_readiness(sc: Scenario, flight_id: str) -> DepartureReadiness:
    dep = sc.simulated.departure
    nf = next(n for n in dep.next_flights if n.flight_id == flight_id)
    sf = next(f for f in sc.simulated.flights if f.flight_id == flight_id)
    crew = next(c for c in sc.simulated.flight_crews if c.id == sf.flight_crew_id)
    limit = crew.report_time_min + sc.workbook.fuel_crew.flight_crew_max_duty_h * 60
    latest = limit - nf.scheduled_block_min - sc.simulated.assumptions.flight_crew_postflight_min
    return DepartureReadiness(flight_id=flight_id, next_flight_label=nf.next_flight_label,
                              scheduled_departure_min=dep.scheduled_departure_min[flight_id], next_flight_block_min=nf.scheduled_block_min,
                              latest_takeoff_min=latest)


def compare_readiness(claimed: DepartureReadiness, actual: DepartureReadiness) -> list[str]:
    out = []
    for f in ("flight_id", "next_flight_label", "fuel_uplift_required"):
        if getattr(claimed, f) != getattr(actual, f):
            out.append(f"{f}: claimed {getattr(claimed, f)} vs computed {getattr(actual, f)}")
    for f in ("scheduled_departure_min", "next_flight_block_min", "latest_takeoff_min"):
        if quantise("landing_min", getattr(claimed, f)) != quantise("landing_min", getattr(actual, f)):
            out.append(f"{f}: claimed {getattr(claimed, f)} vs computed {getattr(actual, f)}")
    return out
