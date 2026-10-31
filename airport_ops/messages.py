"""Structured negotiation messages (Pydantic). Free text is limited to a short `explanation`; everything
that carries commitment (plans, conditions, references) is typed and machine-checkable.

Two layers:
  AgentOutput        - what a model/agent returns (validated, may be repaired once)
  NegotiationMessage - what the coordinator records (adds sender, turn, seq, scenario version, resolved plan)
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .plan import RecoveryPlan
from .workbook import FlightStatus  # noqa: F401  (re-exported)


class AgentRole(str, Enum):
    ATC = "ATC"
    PILOT = "PILOT"
    GROUND = "GROUND"


# LEGACY speaking order (3 agents). Sessions may supply their own participant list (see Participant / participants_for).
SPEAKING_ORDER: tuple[AgentRole, ...] = (AgentRole.ATC, AgentRole.PILOT, AgentRole.GROUND)
MAX_TURNS = 3
MAX_AGENTS = 7                          # schema-level ceiling; each session enforces 3 x its own participant count
MAX_MESSAGES = MAX_TURNS * MAX_AGENTS   # 21


class Participant(BaseModel):
    """One independent agent in a session. `agent_id` is unique ("PILOT@DL1205", "ATC@ATL"); legacy sessions use the bare role
    name ("ATC", "PILOT", "GROUND"). One participant = one separate model call per message."""
    model_config = ConfigDict(frozen=True)
    agent_id: str
    role: AgentRole
    airport: Optional[str] = None
    flight_id: Optional[str] = None

    @property
    def label(self) -> str:
        return self.agent_id


def legacy_participants(selected_flight_id: Optional[str] = None) -> list[Participant]:
    """The original three agents: ATC -> PILOT (selected flight) -> GROUND."""
    return [Participant(agent_id=r.value, role=r, flight_id=selected_flight_id if r == AgentRole.PILOT else None)
            for r in SPEAKING_ORDER]


class MessageType(str, Enum):
    PROPOSE = "PROPOSE"
    COUNTER_PROPOSE = "COUNTER_PROPOSE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class AircraftReport(BaseModel):
    """A Pilot's structured report. The coordinator recomputes every number from code and records any discrepancy."""
    model_config = ConfigDict(extra="forbid")
    flight_id: str
    earliest_landing_min: float
    latest_landing_min: float                    # latest touchdown that keeps final reserve + contingency
    max_holding_min: float                       # holding allowed after the earliest landing
    contingency_margin_kg: float                 # spare fuel above reserve + contingency at the earliest landing
    reserve_margin_kg: float
    preferred_gate_in_by_min: Optional[float] = None   # connecting passengers (soft)
    connecting_passengers: int = 0
    fuel_status: FlightStatus = FlightStatus.NORMAL


class DepartureReadiness(BaseModel):
    """A Pilot's confirmation of aircraft / flight readiness for the NEXT flight (departure sessions). Recomputed by code."""
    model_config = ConfigDict(extra="forbid")
    flight_id: str
    next_flight_label: str
    scheduled_departure_min: float
    next_flight_block_min: float
    latest_takeoff_min: float                    # latest take-off at which the same flight crew can still fly the next flight legally
    fuel_uplift_required: bool = True            # REFUEL must finish before BOARD (no boarding during refuelling)


class ReasonCode(str, Enum):
    FUEL_LIMIT = "FUEL_LIMIT"
    CREW_DUTY_LIMIT = "CREW_DUTY_LIMIT"
    WAKE_SEPARATION = "WAKE_SEPARATION"
    RUNWAY_CLOSURE = "RUNWAY_CLOSURE"
    RUNWAY_CAPACITY = "RUNWAY_CAPACITY"
    TAXIWAY_BLOCKED = "TAXIWAY_BLOCKED"
    GATE_UNAVAILABLE = "GATE_UNAVAILABLE"
    GATE_INCOMPATIBLE = "GATE_INCOMPATIBLE"
    TURNAROUND_TIME = "TURNAROUND_TIME"
    CREW_AVAILABILITY = "CREW_AVAILABILITY"
    OVERTIME = "OVERTIME"
    DELAY_MINIMISATION = "DELAY_MINIMISATION"
    CONNECTION_PRESERVATION = "CONNECTION_PRESERVATION"
    COST_TRADEOFF = "COST_TRADEOFF"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    CONDITIONS_MET = "CONDITIONS_MET"
    CONDITIONS_NOT_MET = "CONDITIONS_NOT_MET"
    ACCEPTABLE = "ACCEPTABLE"
    UNACCEPTABLE = "UNACCEPTABLE"
    SUPERSEDED = "SUPERSEDED"


# ---------------- machine-checkable conditions ----------------
class ConditionField(str, Enum):
    # per flight, numeric (flight_id required)
    LANDING_DELAY_MIN = "landing_delay_min"          # landing - nominal ETA, minutes
    LANDING_MIN = "landing_min"
    GATE_IN_MIN = "gate_in_min"
    SERVICE_START_MIN = "service_start_min"
    SERVICE_END_MIN = "service_end_min"
    SERVICE_WAIT_MIN = "service_wait_min"            # service start - (gate-in + deboard), minutes
    RESERVE_MARGIN_KG = "reserve_margin_kg"          # landing fuel - final reserve, kg
    GATE_FUEL_MARGIN_KG = "gate_fuel_margin_kg"      # fuel at gate arrival (after taxi + ground wait) - final reserve, kg
    READY_MIN = "ready_min"                          # departure sessions: aircraft ready for departure (t+ min)
    TAKEOFF_MIN = "takeoff_min"                      # departure sessions: take-off time (t+ min)
    DEPARTURE_DELAY_MIN = "departure_delay_min"      # departure sessions: take-off minus the NEXT flight's scheduled departure
    TURNAROUND_MIN = "turnaround_min"                # departure sessions: ready time minus gate arrival
    CONTINGENCY_MARGIN_KG = "contingency_margin_kg"  # landing fuel - reserve - contingency, kg
    CONNECTION_MISS_MIN = "connection_miss_min"      # selected flight only: minutes gate-in is later than preferred
    # per flight, identifier (flight_id required; op must be == or !=; value is a string)
    GATE_ID = "gate_id"
    RUNWAY_ID = "runway_id"
    GROUND_CREW_ID = "ground_crew_id"
    # whole plan, numeric (flight_id must be omitted)
    MAX_LANDING_DELAY_MIN = "max_landing_delay_min"
    TOTAL_LANDING_DELAY_MIN = "total_landing_delay_min"
    WEIGHTED_LANDING_DELAY = "weighted_landing_delay"    # sum(priority * landing delay)
    TOTAL_SERVICE_WAIT_MIN = "total_service_wait_min"
    GROUND_OVERTIME_MIN = "ground_overtime_min"          # sum over ground crews, minutes
    GATE_CHANGES = "gate_changes"
    RUNWAY_CHANGES = "runway_changes"
    TOTAL_COST = "total_cost"                            # fallback ranking cost (fixed weights)
    TOTAL_DEPARTURE_DELAY_MIN = "total_departure_delay_min"        # departure sessions
    WEIGHTED_DEPARTURE_DELAY = "weighted_departure_delay"          # sum(priority * departure delay)
    MAX_DEPARTURE_DELAY_MIN = "max_departure_delay_min"


FLIGHT_NUMERIC = {ConditionField.LANDING_DELAY_MIN, ConditionField.LANDING_MIN, ConditionField.GATE_IN_MIN,
                  ConditionField.SERVICE_START_MIN, ConditionField.SERVICE_END_MIN, ConditionField.SERVICE_WAIT_MIN,
                  ConditionField.RESERVE_MARGIN_KG, ConditionField.CONTINGENCY_MARGIN_KG, ConditionField.GATE_FUEL_MARGIN_KG,
                  ConditionField.READY_MIN, ConditionField.TAKEOFF_MIN, ConditionField.DEPARTURE_DELAY_MIN, ConditionField.TURNAROUND_MIN,
                  ConditionField.CONNECTION_MISS_MIN}
FLIGHT_ID_FIELDS = {ConditionField.GATE_ID, ConditionField.RUNWAY_ID, ConditionField.GROUND_CREW_ID}
PLAN_NUMERIC = {f for f in ConditionField} - FLIGHT_NUMERIC - FLIGHT_ID_FIELDS


class ConditionOp(str, Enum):
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"
    EQ = "=="
    NE = "!="


class Condition(BaseModel):
    """`<field> <op> <value>` evaluated against a proposal's plan at the canonical per-field precision (canonical.py) that agents are shown. Hard constraints can NOT be expressed here:
    validity is always checked separately and cannot be waived by any condition."""
    model_config = ConfigDict(extra="forbid")
    field: ConditionField
    op: ConditionOp
    value: Union[float, str]
    flight_id: Optional[str] = None

    @model_validator(mode="after")
    def _shape(self):
        if self.field in FLIGHT_NUMERIC:
            if not self.flight_id:
                raise ValueError(f"{self.field.value} requires flight_id")
            if isinstance(self.value, str):
                raise ValueError(f"{self.field.value} needs a numeric value")
        elif self.field in FLIGHT_ID_FIELDS:
            if not self.flight_id:
                raise ValueError(f"{self.field.value} requires flight_id")
            if self.op not in (ConditionOp.EQ, ConditionOp.NE) or not isinstance(self.value, str):
                raise ValueError(f"{self.field.value} needs op == or != and a string value")
        else:
            if self.flight_id:
                raise ValueError(f"{self.field.value} is plan-level; omit flight_id")
            if isinstance(self.value, str):
                raise ValueError(f"{self.field.value} needs a numeric value")
        return self


class ConditionResult(BaseModel):
    condition: Condition
    satisfied: Optional[bool]  # None = unresolved (e.g. flight missing from the plan)
    observed: Optional[Union[float, str]] = None        # canonical (quantised) value the comparison used
    observed_exact: Optional[float] = None              # exact value, for audit


class TradeoffMetric(str, Enum):
    LANDING_DELAY_MIN = "landing_delay_min"
    TOTAL_LANDING_DELAY_MIN = "total_landing_delay_min"
    WEIGHTED_LANDING_DELAY = "weighted_landing_delay"
    CONNECTION_MISS_MIN = "connection_miss_min"
    GATE_IN_MIN = "gate_in_min"
    GROUND_OVERTIME_MIN = "ground_overtime_min"
    TOTAL_SERVICE_WAIT_MIN = "total_service_wait_min"
    GATE_CHANGES = "gate_changes"
    CONTINGENCY_MARGIN_KG = "contingency_margin_kg"
    TOTAL_COST = "total_cost"
    DEPARTURE_DELAY_MIN = "departure_delay_min"
    WEIGHTED_DEPARTURE_DELAY = "weighted_departure_delay"
    TOTAL_DEPARTURE_DELAY_MIN = "total_departure_delay_min"
    READY_MIN = "ready_min"
    TURNAROUND_MIN = "turnaround_min"


class Tradeoff(BaseModel):
    """Measurable change behind a revised preference: `metric` moves from `before` to `after` because of `because`.
    The coordinator recomputes the real before/after from the plans and logs both."""
    model_config = ConfigDict(extra="forbid")
    metric: TradeoffMetric
    flight_id: Optional[str] = None
    before: float
    after: float
    because: ReasonCode


class PlanSpec(BaseModel):
    """Discrete choices an agent makes; the coordinator's deterministic timeline tool turns them into a complete RecoveryPlan.
    ARRIVAL sessions: landing_order, runways, gates, ground_crews. TURNAROUND/DEPARTURE sessions: ground_crews, equipment_order,
    overlap, dep_runways (arrival fields are fixed inputs and must be omitted). Keys are flight ids."""
    model_config = ConfigDict(extra="forbid")
    landing_order: list[str] = Field(default_factory=list)
    runways: dict[str, str] = Field(default_factory=dict)
    gates: dict[str, str] = Field(default_factory=dict)
    ground_crews: dict[str, str] = Field(default_factory=dict)
    equipment_order: list[str] = Field(default_factory=list, description="departure sessions: priority for shared equipment (fuel truck, tractors)")
    overlap: Optional[Literal["max", "sequential"]] = Field(None, description="departure sessions: run independent tasks in parallel (max) or one after another")
    dep_runways: dict[str, str] = Field(default_factory=dict, description="departure sessions: departure runway per flight")
    declared_status: Optional[FlightStatus] = Field(None, description="DIVERSION sessions: NORMAL | MINIMUM_FUEL | EMERGENCY (must match the computed status)")


class AgentOutput(BaseModel):
    """The JSON object an agent must return."""
    model_config = ConfigDict(extra="forbid")
    type: MessageType
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    explanation: str = Field(max_length=600, description="concise decision explanation; no private reasoning")
    proposal_id: Optional[str] = Field(None, description="ACCEPT/REJECT: the exact proposal being answered")
    referenced_proposal_id: Optional[str] = Field(None, description="COUNTER_PROPOSE: the proposal being countered")
    plan_spec: Optional[PlanSpec] = None
    plan: Optional[RecoveryPlan] = None
    conditions: list[Condition] = Field(default_factory=list)
    tradeoff: Optional[Tradeoff] = None
    report: Optional[AircraftReport] = Field(None, description="PILOT agents (arrival/diversion sessions): required in your first message when the session asks for it")
    readiness: Optional[DepartureReadiness] = Field(None, description="PILOT agents (turnaround/departure sessions): required in your first message when the session asks for it")
    declared_status: Optional[FlightStatus] = Field(None, description="PILOT in a diversion: NORMAL | MINIMUM_FUEL | EMERGENCY")

    @model_validator(mode="after")
    def _type_rules(self):
        t = self.type
        has_plan = (self.plan is not None) + (self.plan_spec is not None)
        if t in (MessageType.PROPOSE, MessageType.COUNTER_PROPOSE):
            if has_plan != 1:
                raise ValueError(f"{t.value} needs exactly one of plan_spec / plan")
            if self.proposal_id:
                raise ValueError(f"{t.value} must not set proposal_id (the coordinator assigns it)")
            if t == MessageType.PROPOSE and self.referenced_proposal_id:
                raise ValueError("PROPOSE must not set referenced_proposal_id")
            if t == MessageType.COUNTER_PROPOSE and not self.referenced_proposal_id:
                raise ValueError("COUNTER_PROPOSE needs referenced_proposal_id")
        else:
            if has_plan:
                raise ValueError(f"{t.value} must not carry a plan (it answers an existing immutable proposal)")
            if not self.proposal_id:
                raise ValueError(f"{t.value} needs proposal_id")
            if self.referenced_proposal_id:
                raise ValueError(f"{t.value} must not set referenced_proposal_id")
            if t == MessageType.REJECT and not self.reason_codes:
                raise ValueError("REJECT needs at least one reason code")
        return self


class NegotiationMessage(BaseModel):
    message_id: str
    session_id: str
    seq: int = Field(ge=1, le=MAX_MESSAGES)
    turn: int = Field(ge=1, le=MAX_TURNS)
    sender: AgentRole
    agent_id: str = ""                                 # unique agent ("PILOT@DL1205"); equals the role name in legacy sessions
    flight_id: Optional[str] = None                    # the flight this agent represents, or the session's subject flight
    airport: Optional[str] = None                      # the airport this agent controls (ATC/Ground)
    scenario_version: str
    type: MessageType
    proposal_id: str                                   # proposal introduced (PROPOSE/COUNTER) or answered (ACCEPT/REJECT)
    referenced_proposal_id: Optional[str] = None       # proposal countered (COUNTER_PROPOSE)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    explanation: str
    conditions: list[Condition] = Field(default_factory=list)
    condition_results: list[ConditionResult] = Field(default_factory=list)  # evaluated by the coordinator
    conditions_resolved: Optional[bool] = None         # ACCEPT: all conditions evaluated True against the exact proposal
    tradeoff: Optional[Tradeoff] = None
    plan: Optional[RecoveryPlan] = None                # complete plan for PROPOSE / COUNTER_PROPOSE
    plan_origin: Optional[Literal["agent_plan", "materialised_from_spec"]] = None
    plan_digest: Optional[str] = None
    outdated: bool = False                             # ACCEPT/REJECT referenced a proposal other than the frozen one (turn 3)
    report: Optional[AircraftReport] = None            # authoritative (code-computed) report when a Pilot sent one
    report_verified: Optional[bool] = None             # did the agent's claimed report match the computed one?
    report_discrepancies: list[str] = Field(default_factory=list)
    readiness: Optional[DepartureReadiness] = None
    declared_status: Optional[FlightStatus] = None
