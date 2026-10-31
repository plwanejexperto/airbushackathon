"""Which agents take part in a session. One independent model call per participant per message.

  arrival (scenarios 1 + 2): PILOT@<flight> for EVERY aircraft  ->  ATC@<airport>  ->  GROUND@<airport>
  (departure and diversion sessions extend this list in their own modules)

The Pilots speak first (they report their arrival window, fuel margin and maximum holding time); ATC then offers slots, runways and
taxi routes; Ground checks gates/stands/crews. Later agents see earlier messages of the same turn.
"""
from __future__ import annotations

from .messages import AgentRole, Participant
from .schemas import Scenario


def arrival_participants(sc: Scenario) -> list[Participant]:
    ap = sc.workbook.airport.iata_code
    return ([Participant(agent_id=f"PILOT@{f.flight_id}", role=AgentRole.PILOT, flight_id=f.flight_id) for f in sc.simulated.flights]
            + [Participant(agent_id=f"ATC@{ap}", role=AgentRole.ATC, airport=ap),
               Participant(agent_id=f"GROUND@{ap}", role=AgentRole.GROUND, airport=ap)])


def diversion_participants(sc: Scenario) -> list[Participant]:
    """Diverting aircraft's Pilot -> ALTERNATE ATC -> ALTERNATE Ground -> original DESTINATION ATC -> DESTINATION Ground (5 agents).
    The alternate's competing arrivals are deterministic traffic (no Pilot agents in the first prototype)."""
    dv = sc.simulated.diversion
    alt, dest = dv.alternate_airport, dv.destination.airport
    return [Participant(agent_id=f"PILOT@{dv.diverting_flight_id}", role=AgentRole.PILOT, flight_id=dv.diverting_flight_id),
            Participant(agent_id=f"ATC@{alt}", role=AgentRole.ATC, airport=alt), Participant(agent_id=f"GROUND@{alt}", role=AgentRole.GROUND, airport=alt),
            Participant(agent_id=f"ATC@{dest}", role=AgentRole.ATC, airport=dest), Participant(agent_id=f"GROUND@{dest}", role=AgentRole.GROUND, airport=dest)]


def participants_for(sc: Scenario) -> list[Participant]:
    return diversion_participants(sc) if sc.simulated.diversion is not None else arrival_participants(sc)
