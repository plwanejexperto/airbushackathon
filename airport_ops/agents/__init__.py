"""Three independent agents: separate role instructions, domain views and (in live mode) separate model calls."""
from __future__ import annotations

from ..messages import AgentRole, PlanSpec
from ..negotiation_types import AgentContext
from . import atc, base, ground, pilot

MODULES = {AgentRole.ATC: atc, AgentRole.PILOT: pilot, AgentRole.GROUND: ground}


def system_prompt(role: AgentRole) -> str:
    return MODULES[role].system_prompt()


def user_prompt(ctx: AgentContext) -> str:
    m = MODULES[ctx.role]
    return base.build_prompt(ctx, m.domain_view(ctx), m.tool_results(ctx))


def _originals_spec(ctx: AgentContext) -> PlanSpec:
    fl = sorted(ctx.scenario.simulated.flights, key=lambda f: (f.earliest_landing_min, f.flight_id))
    return PlanSpec(landing_order=[f.flight_id for f in fl], runways={f.flight_id: f.original_runway_id for f in fl},
                    gates={f.flight_id: f.original_gate_id for f in fl},
                    ground_crews={f.flight_id: f.original_ground_crew_id for f in fl})


def scripted_response(ctx: AgentContext) -> str:
    """Used by ScriptedClient only."""
    return MODULES[ctx.role].scripted_policy(ctx, lambda: _originals_spec(ctx))
