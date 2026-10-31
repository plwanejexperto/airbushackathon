"""Hard constraints specific to diversion sessions (run with the landing stage; they depend only on the plan and the scenario)."""
from __future__ import annotations

from .fuel import fuel_account
from .workbook import FlightStatus

TOL = 1e-6


def earliest_feasible_landing(sc, sf, runway_id: str) -> float:
    """First time >= the aircraft's ETA at which the roll-out on this runway is clear of closures (ignores other traffic)."""
    rw = next((r for r in sc.simulated.runways if r.id == runway_id), None)
    roll = sc.simulated.assumptions.runway_occupancy_min
    t = sf.earliest_landing_min
    for _ in range(100):
        t0 = t
        for cl in (rw.closures if rw else []):
            if t < cl.end_min - TOL and t + roll > cl.start_min + TOL:
                t = cl.end_min
        if abs(t - t0) < TOL:
            break
    return t


def check_diversion(ix, plan, fps, c) -> None:
    sc = ix.sc
    dv = sc.simulated.diversion
    sf = ix.flights.get(dv.diverting_flight_id)
    if sf is None:
        return
    f = sf.flight_id
    dest = dv.destination
    c.check("DIV-TRIGGER", (not dest.feasible) or dv.computed_status != FlightStatus.NORMAL or dv.emergency_declared_input,
            lambda: (f"landing at {dest.airport} is infeasible: earliest feasible landing t+{dest.earliest_feasible_landing_min:.1f} vs latest by fuel "
                     f"t+{dest.latest_landing_min_by_fuel:.1f}" if not dest.feasible else
                     f"status {dv.computed_status.value} justifies leaving {dest.airport}" if dv.computed_status != FlightStatus.NORMAL else
                     f"{dest.airport} can still accept the aircraft within its fuel limit (t+{dest.earliest_feasible_landing_min:.1f} <= "
                     f"t+{dest.latest_landing_min_by_fuel:.1f}): a diversion is not justified"), f)
    cand = next((x for x in dv.candidates if x.airport == dv.alternate_airport), None)
    ok = plan.airport == dv.alternate_airport and cand is not None and cand.suitable
    c.check("DIV-ALT-SUITABLE", ok, lambda: (f"{dv.alternate_airport} suitable: {cand.size_class}, simulated weather ok={cand.simulated_weather_ok}, "
            f"reachable on fuel" if ok else f"plan airport {plan.airport!r} vs screened alternate {dv.alternate_airport!r}"
            + ("" if cand is None or cand.suitable else f" (not suitable: {'; '.join(cand.reasons)})")), f, dv.alternate_airport)
    c.check("DIV-STATUS", plan.declared_status == dv.computed_status,
            lambda: f"declared {plan.declared_status.value if plan.declared_status else None} vs computed {dv.computed_status.value} "
                    "(the Pilot must declare the status the fuel numbers imply)", f)
    fa = fuel_account(sc, sf, sf.earliest_landing_min)
    c.check("DIV-FUEL-REACH", fa.landing_fuel_kg >= -TOL, lambda: (
        f"from the simulated position: {cand.distance_from_position_nm if cand else '?'} nm, {sf.earliest_landing_min:.1f} min, trip fuel "
        f"{fa.trip_fuel_kg:.0f} kg; landing fuel at the earliest slot {fa.landing_fuel_kg:.0f} kg"), f, dv.alternate_airport)
    if dv.computed_status == FlightStatus.EMERGENCY:
        p = next((x for x in fps if x.flight_id == f), None)
        if p is not None:
            early = earliest_feasible_landing(sc, sf, p.runway_id)
            c.check("EMERG-PRIORITY", p.landing_min <= early + ix.a.time_tolerance_min, lambda: (
                f"EMERGENCY: landing t+{p.landing_min:.1f} vs earliest feasible t+{early:.1f} on {p.runway_id}: no convenience delay allowed"), f, p.runway_id)
