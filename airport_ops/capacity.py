"""Runway movement limits. Two DIFFERENT things are kept apart:

* airport-wide limit      - workbook ``airport_capacity.max_runway_ops_per_hour`` (WORKBOOK value; the sheet does not say
                            whether it is per runway or airport-wide, it is read as airport-wide). Checked as
                            RWY-CAP-AIRPORT over the modelled movements only.
* per-runway share        - an ASSUMPTION layered on top, configured by ``SimAssumptions.per_runway_share_mode``:
                            ``even_split`` = airport limit // active_runways (default, the stage-2 behaviour),
                            ``fixed``      = ``per_runway_fixed_ops_per_hour`` movements/hour per runway,
                            ``disabled``   = no per-runway check.
                            Checked as RWY-CAP-RUNWAY.
"""
from __future__ import annotations

from typing import Optional

from .schemas import Scenario


def airport_limit(sc: Scenario) -> int:
    return sc.workbook.airport.max_runway_ops_per_hour


def per_runway_limit(sc: Scenario) -> Optional[int]:
    """Assumed movements/hour per runway, or None when the per-runway check is disabled."""
    a = sc.simulated.assumptions
    if a.per_runway_share_mode == "disabled":
        return None
    if a.per_runway_share_mode == "fixed":
        return a.per_runway_fixed_ops_per_hour
    ap = sc.workbook.airport
    return max(1, ap.max_runway_ops_per_hour // ap.active_runways)
