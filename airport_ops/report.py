"""Event-log payloads and display rows for validation / fallback results."""
from __future__ import annotations

from .fallback import FallbackResult
from .plan import RecoveryPlan
from .schemas import Scenario
from .validate import ValidationReport


def validation_payload(report: ValidationReport, source: str = "") -> dict:
    return {
        "proposal_id": report.proposal_id, "scenario_version": report.scenario_version, "source": source,
        "passed": report.passed, "n_checks": len(report.results),
        "results": [[r.constraint_id, r.flight_id, r.resource_id, r.passed, r.explanation] for r in report.results],
    }


def fallback_payload(res: FallbackResult, scenario: Scenario) -> dict:
    return {
        "scenario_version": scenario.scenario_version, "outcome": res.outcome.value,
        "search_status": res.search_status.value,
        "proposal_id": res.plan.proposal_id if res.plan else None,
        "cost": res.cost, "cost_breakdown": res.cost_breakdown, "reasons": res.reasons,
        "candidates_total": res.candidates_total, "candidates_validated": res.candidates_validated,
        "candidates_pruned": res.candidates_pruned, "feasible_count": res.feasible_count,
        "rejection_by_constraint": res.rejection_by_constraint,
        "validation": validation_payload(res.validation, "fallback") if res.validation else None,
    }


def assignment_rows(plan: RecoveryPlan, scenario: Scenario) -> list[dict]:
    flights = {f.flight_id: f for f in scenario.simulated.flights}
    t = lambda m: f"t+{m:.1f}"
    return [{
        "flight": p.flight_id, "runway": p.runway_id, "landing (touchdown)": t(p.landing_min),
        "taxi route": " > ".join(p.taxi_route), "taxi": f"{t(p.taxi_start_min)} - {t(p.taxi_end_min)}",
        "gate": p.gate_id + (" (changed)" if p.gate_id != flights[p.flight_id].original_gate_id else ""),
        "gate arrival (on-block)": t(p.gate_in_min), "service": f"{t(p.service_start_min)} - {t(p.service_end_min)}",
        "ground crew": p.ground_crew_id, "flight crew": p.flight_crew_id,
    } for p in plan.flights]
