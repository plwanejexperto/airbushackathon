"""Recovery-plan models and explicit resource occupancy intervals (minutes from scenario start)."""
from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .schemas import Scenario


class FlightPlan(BaseModel):
    flight_id: str
    runway_id: str
    landing_min: float
    taxi_route: list[str] = Field(description="ordered taxi segment ids, runway exit -> gate")
    taxi_start_min: float
    taxi_end_min: float
    gate_id: str
    gate_in_min: float = Field(description="gate arrival (on-block); equals taxi_end_min")
    service_start_min: float
    service_end_min: float
    ground_crew_id: str
    flight_crew_id: str


class RecoveryPlan(BaseModel):
    proposal_id: str
    scenario_version: str
    source: str = "external"  # e.g. naive_baseline | fallback | external
    decision_time_min: float = Field(0.0, description="when the plan takes effect (gate-change notice is measured from here)")
    flights: list[FlightPlan]


ResourceType = Literal["runway", "taxi_segment", "gate", "ground_crew", "flight_crew"]


class ResourceInterval(BaseModel):
    resource_type: ResourceType
    resource_id: str
    flight_id: Optional[str]  # None = simulated external occupancy (closure, resident aircraft)
    start_min: float
    end_min: float
    purpose: str


def occupancy_intervals(fp: FlightPlan, scenario: Scenario, seg_minutes: Optional[dict[str, float]] = None
                        ) -> list[ResourceInterval]:
    """Derive every resource occupancy from the plan's primary fields (single source of truth).

    runway       : [landing, landing + runway_occupancy]
    taxi segment : back-to-back from (taxi_end - sum(route minutes)); extra taxi time is holding at the exit
    gate         : [gate_in, service_end]
    ground crew  : [service_start, service_end]
    flight crew  : [landing, gate_in + postflight]
    """
    a = scenario.simulated.assumptions
    if seg_minutes is None:
        seg_minutes = {s.id: s.minutes for s in scenario.simulated.taxi_segments}
    f = fp.flight_id
    out = [ResourceInterval(resource_type="runway", resource_id=fp.runway_id, flight_id=f, start_min=fp.landing_min,
                            end_min=fp.landing_min + a.runway_occupancy_min, purpose="landing roll-out")]
    if all(s in seg_minutes for s in fp.taxi_route):
        t = fp.taxi_end_min - sum(seg_minutes[s] for s in fp.taxi_route)
        for s in fp.taxi_route:
            out.append(ResourceInterval(resource_type="taxi_segment", resource_id=s, flight_id=f, start_min=t,
                                        end_min=t + seg_minutes[s], purpose="taxi"))
            t += seg_minutes[s]
    out.append(ResourceInterval(resource_type="gate", resource_id=fp.gate_id, flight_id=f, start_min=fp.gate_in_min,
                                end_min=fp.service_end_min, purpose="gate occupancy"))
    out.append(ResourceInterval(resource_type="ground_crew", resource_id=fp.ground_crew_id, flight_id=f,
                                start_min=fp.service_start_min, end_min=fp.service_end_min, purpose="servicing"))
    out.append(ResourceInterval(resource_type="flight_crew", resource_id=fp.flight_crew_id, flight_id=f,
                                start_min=fp.landing_min,
                                end_min=fp.gate_in_min + a.flight_crew_postflight_min, purpose="duty (landing to post-flight)"))
    return out


def make_proposal_id(scenario_version: str, source: str, flights: list[FlightPlan]) -> str:
    blob = json.dumps([f.model_dump(mode="json") for f in flights], sort_keys=True)
    return "prop-" + hashlib.sha256(f"{scenario_version}|{source}|{blob}".encode()).hexdigest()[:10]


def make_plan(scenario: Scenario, source: str, flights: list[FlightPlan], decision_time_min: float = 0.0) -> RecoveryPlan:
    return RecoveryPlan(proposal_id=make_proposal_id(scenario.scenario_version, source, flights),
                        scenario_version=scenario.scenario_version, source=source,
                        decision_time_min=decision_time_min, flights=flights)
