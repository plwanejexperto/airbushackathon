"""Load flights.csv and join airport names from the airport_capacity sheet of
challenge2_operational_constraints.xlsx (both in data/). Airport identifiers
stay strings; missing lookups become None (shown as 'Unknown' in the UI).
The workbook has no airline-name table, so airline_name is always None."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import HistoricalFlight

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FLIGHT = dict(airline="DL", number=1205, origin="PHX", dest="ATL", date="2015-01-01")
DISRUPTION_QUERY = "WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20"


DATA_DIR = ROOT / "data"
FLIGHTS_CSV = DATA_DIR / "flights.csv"
CONSTRAINTS_XLSX = DATA_DIR / "challenge2_operational_constraints.xlsx"
AIRPORT_SHEET = "airport_capacity"  # 2 preamble rows, header on the 3rd


def _hhmm(v) -> str | None:
    if pd.isna(v):
        return None
    v = int(v) % 2400
    return f"{v // 100:02d}:{v % 100:02d}"


def load_flights() -> pd.DataFrame:
    """Disruption-filtered flights joined to airline/airport names."""
    strs = {"AIRLINE": str, "ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str, "TAIL_NUMBER": str}
    flights = pd.read_csv(FLIGHTS_CSV, dtype=strs, low_memory=False)
    ap = pd.read_excel(CONSTRAINTS_XLSX, sheet_name=AIRPORT_SHEET, header=2,
                       dtype={"IATA_CODE": str}, keep_default_na=False, na_values=[""])
    ap = ap[["IATA_CODE", "AIRPORT"]].dropna(subset=["IATA_CODE"]).drop_duplicates("IATA_CODE")
    ap["IATA_CODE"] = ap["IATA_CODE"].str.strip()
    for c in ("ORIGIN_AIRPORT", "DESTINATION_AIRPORT"):
        flights[c] = flights[c].str.strip()

    df = flights.query(DISRUPTION_QUERY).copy()
    df["AIRLINE_NAME"] = None  # no airline lookup in the current datasets
    for col, new in (("ORIGIN_AIRPORT", "ORIGIN_NAME"), ("DESTINATION_AIRPORT", "DESTINATION_NAME")):
        df = df.merge(ap.rename(columns={"IATA_CODE": col, "AIRPORT": new}), on=col, how="left")
    df["FLIGHT_DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]]).dt.date
    df["FLIGHT_LABEL"] = df["AIRLINE"] + df["FLIGHT_NUMBER"].astype(str)
    return df.reset_index(drop=True)


def _none(v):
    return None if pd.isna(v) else v


def to_historical(row: pd.Series) -> HistoricalFlight:
    cancelled = bool(row.CANCELLED)
    return HistoricalFlight(
        flight_date=row.FLIGHT_DATE, airline_code=row.AIRLINE,
        airline_name=_none(row.AIRLINE_NAME), flight_number=int(row.FLIGHT_NUMBER),
        tail_number=_none(row.TAIL_NUMBER),
        origin_code=row.ORIGIN_AIRPORT, origin_name=_none(row.ORIGIN_NAME),
        destination_code=row.DESTINATION_AIRPORT, destination_name=_none(row.DESTINATION_NAME),
        scheduled_departure=_hhmm(row.SCHEDULED_DEPARTURE), actual_departure=_hhmm(row.DEPARTURE_TIME),
        scheduled_arrival=_hhmm(row.SCHEDULED_ARRIVAL), actual_arrival=_hhmm(row.ARRIVAL_TIME),
        departure_delay_min=None if pd.isna(row.DEPARTURE_DELAY) else int(row.DEPARTURE_DELAY),
        arrival_delay_min=None if pd.isna(row.ARRIVAL_DELAY) else int(row.ARRIVAL_DELAY),
        weather_delay_min=int(row.WEATHER_DELAY), late_aircraft_delay_min=int(row.LATE_AIRCRAFT_DELAY),
        air_system_delay_min=int(row.AIR_SYSTEM_DELAY), airline_delay_min=int(row.AIRLINE_DELAY),
        security_delay_min=int(row.SECURITY_DELAY),
        distance_mi=None if pd.isna(row.DISTANCE) else int(row.DISTANCE),
        cancelled=cancelled, diverted=bool(row.DIVERTED),
    )


def default_index(df: pd.DataFrame) -> int:
    d = DEFAULT_FLIGHT
    m = (df.AIRLINE == d["airline"]) & (df.FLIGHT_NUMBER == d["number"]) & \
        (df.ORIGIN_AIRPORT == d["origin"]) & (df.DESTINATION_AIRPORT == d["dest"]) & \
        (df.FLIGHT_DATE.astype(str) == d["date"])
    return int(m.idxmax()) if m.any() else 0
