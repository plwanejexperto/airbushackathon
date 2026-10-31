"""Pydantic schemas. Three data origins are kept in separate models:

  HistoricalFlight     - observed values from data/flights.csv (+ airport-name lookup)
  WorkbookLimits       - SYNTHETIC planning limits from the operational workbook
  SimulatedConditions  - invented, seeded operating conditions + configurable assumptions

All times in simulated/plan models are float MINUTES relative to scenario start (t=0).
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .workbook import AircraftClass, AirportLimits, FuelCrewRules, WakeCategory, WakeTable


# ---------- HISTORICAL: observed values from flights.csv + lookups ----------
class HistoricalFlight(BaseModel):
    """Everything here comes straight from the dataset (plus name lookups)."""

    flight_date: date
    airline_code: str
    airline_name: Optional[str] = None  # None: no airline-name table in data/
    flight_number: int
    tail_number: Optional[str] = None
    origin_code: str  # kept as str: identifiers are not numbers
    origin_name: Optional[str] = None
    destination_code: str
    destination_name: Optional[str] = None
    scheduled_departure: Optional[str] = None  # "HH:MM"
    actual_departure: Optional[str] = None
    scheduled_arrival: Optional[str] = None
    actual_arrival: Optional[str] = None
    departure_delay_min: Optional[int] = None
    arrival_delay_min: Optional[int] = None
    weather_delay_min: int = 0
    late_aircraft_delay_min: int = 0
    air_system_delay_min: int = 0
    airline_delay_min: int = 0
    security_delay_min: int = 0
    distance_mi: Optional[int] = None
    cancelled: bool = False
    diverted: bool = False

    @property
    def label(self) -> str:
        return f"{self.airline_code}{self.flight_number}"


# ---------- SIMULATED: configurable assumptions (not in dataset or workbook) ----------
class SimAssumptions(BaseModel):
    """Every number here is an assumption the workbook does not provide. Change = state it in the demo."""

    approach_speed_kt: float = Field(140, gt=0, description="final-approach groundspeed; converts wake nm -> time")
    wake_apply_time_floor: bool = Field(True, description="also enforce workbook min_time_s (labelled for departures) "
                                                          "as a conservative floor between arrivals")
    runway_occupancy_min: float = Field(1.5, gt=0, description="runway occupied from landing to vacating")
    deboard_min: float = Field(5, ge=0, description="gate arrival -> earliest service start")
    gate_buffer_min: float = Field(5, ge=0, description="gap between consecutive occupants of a gate")
    ground_crew_transfer_min: float = Field(5, ge=0, description="gap between consecutive jobs of a ground crew")
    flight_crew_postflight_min: float = Field(15, ge=0, description="flight-crew duty continues after gate arrival")
    max_ground_overtime_min: float = Field(120, ge=0, description="hard cap on ground-crew overtime "
                                                                  "(workbook only prices overtime, 1.5x)")
    trip_burn_narrowbody_kg_h: float = Field(2500, gt=0, description="en-route burn used for trip fuel (narrowbody)")
    trip_burn_widebody_kg_h: float = Field(7000, gt=0, description="en-route burn used for trip fuel (widebody)")
    spare_fuel_min_range: tuple[int, int] = Field((25, 45), description="seeded spare holding fuel at nominal ETA "
                                                                        "(workbook default 45 = upper bound)")
    time_tolerance_min: float = Field(0.01, gt=0, description="tolerance for equality / sequencing comparisons")
    search_horizon_min: float = Field(600, gt=0, description="planner gives up scheduling beyond this")


class Interval(BaseModel):
    start_min: float
    end_min: float
    reason: str = ""

    @model_validator(mode="after")
    def _ordered(self):
        if self.end_min <= self.start_min:
            raise ValueError("interval end must be after start")
        return self


class FuelState(BaseModel):
    """Simulated fuel state at scenario start. Burn during holding uses the workbook rate."""
    trip_fuel_remaining_kg: float = Field(gt=0, description="fuel to fly from t=0 to nominal landing")
    fuel_on_board_kg: float = Field(gt=0)
    spare_fuel_min_at_eta: float = Field(ge=0, description="holding minutes above reserve+contingency at nominal ETA")


class SimFlight(BaseModel):
    flight_id: str
    source: Literal["historical_selected", "simulated"]
    wake_category: WakeCategory
    aircraft_class: AircraftClass
    passengers: int = Field(gt=0)
    priority: int = Field(ge=1, le=5, description="5 = highest operational priority")
    earliest_landing_min: float = Field(description="nominal ETA from scenario start; cannot land earlier")
    original_runway_id: str
    original_gate_id: str
    original_ground_crew_id: str
    flight_crew_id: str
    service_task_min: float = Field(gt=0, description="ground servicing work content")
    fuel: FuelState


class RunwayResource(BaseModel):
    id: str
    closures: list[Interval] = Field(default_factory=list)


class TaxiSegment(BaseModel):
    """Directed inbound taxi segment. Nodes: EXIT-<runway>, HUB, APRON, GATE-<gate>."""
    id: str
    from_node: str
    to_node: str
    minutes: float = Field(gt=0)
    blockages: list[Interval] = Field(default_factory=list)


class GateResource(BaseModel):
    id: str
    terminal: str
    max_aircraft_class: AircraftClass
    resident_occupancy: list[Interval] = Field(default_factory=list)


class FlightCrew(BaseModel):
    id: str
    home_flight_id: str
    report_time_min: float = Field(description="negative: duty began before scenario start")


class GroundCrew(BaseModel):
    id: str
    shift_start_min: float = Field(description="negative: shift began before scenario start")
    available_from_min: float = 0.0


class SimulatedConditions(BaseModel):
    seed: int
    assumptions: SimAssumptions
    flights: list[SimFlight]
    runways: list[RunwayResource]
    taxi_segments: list[TaxiSegment]
    gates: list[GateResource]
    flight_crews: list[FlightCrew]
    ground_crews: list[GroundCrew]


# ---------- WORKBOOK: synthetic planning limits ----------
class CapacitySubset(BaseModel):
    """Airport totals vs the named subset modelled in the demo. No background traffic is invented."""
    total_runways: int
    modelled_runways: int
    total_gates: int
    modelled_gates: int
    spare_gates_at_any_time: int
    note: str = ("Only the named resources below are modelled; the rest of the airport's capacity is "
                 "not populated with invented traffic.")


class WorkbookLimits(BaseModel):
    airport: AirportLimits
    wake: WakeTable
    fuel_crew: FuelCrewRules
    capacity_subset: CapacitySubset


class Scenario(BaseModel):
    scenario_id: str
    scenario_version: str  # content hash of historical + workbook + simulated
    historical: HistoricalFlight
    workbook: WorkbookLimits
    simulated: SimulatedConditions
