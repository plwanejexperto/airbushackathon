"""Three independent agents: separate role instructions, domain views and (in live mode) separate model calls."""
from __future__ import annotations

from ..messages import AgentRole, PlanSpec
from ..negotiation_types import AgentContext
from . import atc, base, ground, pilot

MODULES = {AgentRole.ATC: atc, AgentRole.PILOT: pilot, AgentRole.GROUND: ground}


def system_prompt(who) -> str:
    """`who` is an AgentRole (generic prompt) or a Participant (adds the agent's identity)."""
    role = getattr(who, "role", who)
    text = MODULES[role].system_prompt()
    if hasattr(who, "agent_id") and who.agent_id != role.value:
        text += (f"\n\nYOU ARE AGENT {who.agent_id}" + (f" (airport {who.airport})" if who.airport else "")
                 + (f" (flight {who.flight_id})" if who.flight_id else "") + ". Other agents have their own identities and instructions.")
    return text


def tool_results(ctx: AgentContext) -> dict:
    """Deterministic, advisory tool output for this role (valid options ranked by the ROLE's own objective)."""
    return MODULES[ctx.role].tool_results(ctx)


def user_prompt(ctx: AgentContext, tools: dict | None = None) -> str:
    m = MODULES[ctx.role]
    return base.build_prompt(ctx, m.domain_view(ctx), tools if tools is not None else m.tool_results(ctx))


def _originals_spec(ctx: AgentContext) -> PlanSpec:
    dep = ctx.scenario.simulated.departure
    if dep is not None:      # naive departure choices: arrival order, arrival crews/runways, tasks in parallel
        fx = sorted(dep.arrival, key=lambda x: (x.gate_in_min, x.flight_id))
        return PlanSpec(equipment_order=[x.flight_id for x in fx], overlap="max", ground_crews={x.flight_id: x.ground_crew_id for x in fx},
                        dep_runways={x.flight_id: x.runway_id for x in fx})
    fl = sorted(ctx.scenario.simulated.flights, key=lambda f: (f.earliest_landing_min, f.flight_id))
    return PlanSpec(landing_order=[f.flight_id for f in fl], runways={f.flight_id: f.original_runway_id for f in fl},
                    gates={f.flight_id: f.original_gate_id for f in fl},
                    ground_crews={f.flight_id: f.original_ground_crew_id for f in fl})


def scripted_response(ctx: AgentContext) -> str:
    """Used by ScriptedClient only."""
    return MODULES[ctx.role].scripted_policy(ctx, lambda: _originals_spec(ctx))
