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

from .workbook import AircraftClass, AirportLimits, FlightStatus, FuelCrewRules, WakeCategory, WakeTable


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
    per_runway_share_mode: Literal["even_split", "fixed", "disabled"] = Field(
        "even_split", description="ASSUMED per-runway movement share layered on the workbook's airport-wide limit: "
                                  "even_split = limit // active_runways; fixed = per_runway_fixed_ops_per_hour; disabled = no check")
    per_runway_fixed_ops_per_hour: Optional[int] = Field(None, ge=1, description="movements/hour per runway when mode = fixed")
    cruise_speed_kt: float = Field(450, gt=0, description="cruise groundspeed: converts remaining minutes <-> distance for the simulated aircraft position")
    remote_stand_bus_min: float = Field(8, ge=0, description="extra deboarding time at a remote stand (bussing); SIMULATED - the workbook has no stand data")
    ground_burn_fraction_of_hold: float = Field(0.25, ge=0, le=1, description="ground idle/taxi burn as a fraction of the workbook holding burn; fuel after touchdown is checked down to the final reserve")
    time_tolerance_min: float = Field(0.01, gt=0, description="tolerance for equality / sequencing comparisons")
    search_horizon_min: float = Field(600, gt=0, description="planner gives up scheduling beyond this")


    @model_validator(mode="after")
    def _fixed_needs_value(self):
        if self.per_runway_share_mode == "fixed" and self.per_runway_fixed_ops_per_hour is None:
            raise ValueError("per_runway_share_mode='fixed' requires per_runway_fixed_ops_per_hour")
        return self


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
    status: FlightStatus = FlightStatus.NORMAL
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
    kind: Literal["gate", "remote_stand"] = "gate"   # remote stands are SIMULATED (not in the workbook)
    terminal: str
    max_aircraft_class: AircraftClass
    resident_occupancy: list[Interval] = Field(default_factory=list)


class HoldArea(BaseModel):
    """Designated SAFE ground waiting area (SIMULATED). An aircraft that has landed and must wait for a gate/taxiway waits here,
    never airborne. Capacity = aircraft at the same time."""
    id: str
    capacity: int = Field(ge=1)
    note: str = "simulated safe ground waiting area"


class FlightCrew(BaseModel):
    id: str
    home_flight_id: str
    report_time_min: float = Field(description="negative: duty began before scenario start")


class GroundCrew(BaseModel):
    id: str
    shift_start_min: float = Field(description="negative: shift began before scenario start")
    available_from_min: float = 0.0


class ConnectionPreference(BaseModel):
    """SIMULATED soft preference of the selected flight (not in the dataset): connecting passengers who need
    gate arrival by a preferred time. Never a hard constraint; measured as connection_miss_min."""
    flight_id: str
    connecting_passengers: int = Field(ge=0)
    preferred_gate_in_by_min: float


class TaskDef(BaseModel):
    task_id: str
    name: str
    base_min: float = Field(gt=0)
    predecessors: list[str] = Field(default_factory=list)
    equipment: Optional[str] = None          # equipment TYPE needed (units are listed in TurnaroundModel.equipment)


class EquipmentUnit(BaseModel):
    id: str
    type: str


class TurnaroundModel(BaseModel):
    """SIMULATED turnaround work content (the workbook only gives a minimum turnaround). Widebody tasks are scaled; remote stands add
    the bussing time to deplaning and boarding."""
    tasks: list[TaskDef]
    equipment: list[EquipmentUnit]
    widebody_factor: float = Field(1.3, gt=0)
    pushback_min: float = Field(3.0, ge=0)


class NextFlight(BaseModel):
    """The aircraft's NEXT scheduled flight (an input). source = flights.csv when the same tail has a later same-day flight leaving
    from this airport, otherwise SIMULATED."""
    flight_id: str                           # the ARRIVING flight whose aircraft this is
    next_flight_label: str
    destination: Optional[str] = None
    planned_ground_min: float                # next scheduled departure - this flight's scheduled arrival
    scheduled_block_min: float               # scheduled flight time of the next flight
    source: Literal["flights.csv", "SIMULATED"]
    note: str = ""


class ArrivalFix(BaseModel):
    """The agreed ARRIVAL part of a plan, fixed as an input of the turnaround/departure session."""
    flight_id: str
    landing_min: float
    runway_id: str
    taxi_route: list[str]
    taxi_start_min: float
    taxi_end_min: float
    gate_id: str
    gate_in_min: float
    hold_area_id: Optional[str] = None
    ground_crew_id: str


class DepartureContext(BaseModel):
    turnaround: TurnaroundModel
    next_flights: list[NextFlight]
    arrival: list[ArrivalFix]
    departure_roll_min: float = Field(1.5, gt=0, description="runway occupied by a take-off roll (assumption)")
    arrival_departure_gap_min: float = Field(1.0, ge=0, description="gap between an arrival's and a departure's occupancy of one runway (assumption)")
    scheduled_departure_min: dict[str, float]    # flight_id -> scheduled departure of the NEXT flight (t+ min)
    arrival_plan_source: str = ""
    note: str = ""


class AlternateCandidate(BaseModel):
    """One screened alternate airport. A route in flights.csv is only a hint that the airport is connected; SUITABILITY is decided by
    the checks below (workbook capacity/size, fuel reach from the aircraft's simulated position, simulated weather)."""
    airport: str
    name: Optional[str] = None
    size_class: str = ""
    route_to_destination_nm: Optional[float] = None          # flights.csv (statute miles converted to nm)
    route_from_origin_nm: Optional[float] = None             # flights.csv, if that route exists
    distance_from_position_nm: float
    distance_method: str                                     # how the distance was obtained (data geometry or a labelled assumption)
    flight_time_min: float
    trip_fuel_kg: float
    landing_fuel_margin_over_reserve_contingency_kg: float
    reachable_normal: bool
    reachable_min_fuel: bool
    reachable_emergency: bool
    simulated_weather_ok: bool
    size_ok: bool
    suitable: bool
    reasons: list[str] = Field(default_factory=list)


class DestinationStatus(BaseModel):
    airport: str
    name: Optional[str] = None
    closures: list[Interval] = Field(default_factory=list)       # SIMULATED
    earliest_feasible_landing_min: float
    latest_landing_min_by_fuel: float
    feasible: bool
    limits: dict = Field(default_factory=dict)                   # workbook AirportLimits
    gates_note: str = ""


class DiversionContext(BaseModel):
    """Attached to the ALTERNATE airport's scenario: the plan domain is the aircraft's arrival at the alternate."""
    diverting_flight_id: str
    origin_airport: str
    destination: DestinationStatus
    alternate_airport: str
    position: dict                       # SIMULATED position at decision time (distances from the dataset route + assumed cruise speed)
    candidates: list[AlternateCandidate]
    computed_status: FlightStatus
    emergency_declared_input: bool = False
    decision_min: float = 0.0            # the diversion is decided at scenario start (assumption)
    assumptions: list[str] = Field(default_factory=list)
    status_rule: str = ""


class SimulatedConditions(BaseModel):
    seed: int
    assumptions: SimAssumptions
    flights: list[SimFlight]
    runways: list[RunwayResource]
    taxi_segments: list[TaxiSegment]
    gates: list[GateResource]
    flight_crews: list[FlightCrew]
    ground_crews: list[GroundCrew]
    departure: Optional[DepartureContext] = None   # set => this is a turnaround + departure session
    diversion: Optional[DiversionContext] = None   # set => this is a diversion session (plan domain = the alternate airport)
    hold_areas: list[HoldArea] = Field(default_factory=list)   # empty = legacy: waiting after landing is implicit and unbounded
    connection: Optional[ConnectionPreference] = None      # selected flight (kept for compatibility)
    connections: list[ConnectionPreference] = Field(default_factory=list)   # every flight's connecting-passenger preference

    def connection_for(self, flight_id: str) -> Optional[ConnectionPreference]:
        c = next((x for x in self.connections if x.flight_id == flight_id), None)
        return c or (self.connection if self.connection and self.connection.flight_id == flight_id else None)


# ---------- WORKBOOK: synthetic planning limits ----------
class CapacitySubset(BaseModel):
    """Airport totals vs the named subset modelled in the demo. No background traffic is invented."""
    total_runways: int
    modelled_runways: int
    total_gates: int
    modelled_gates: int
    modelled_remote_stands: int = 0
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
