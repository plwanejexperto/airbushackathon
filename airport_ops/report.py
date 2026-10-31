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
    rows = []
    for p in plan.flights:
        r = {"flight": p.flight_id, "runway": p.runway_id, "landing (touchdown)": t(p.landing_min),
             "taxi route": " > ".join(p.taxi_route), "taxi": f"{t(p.taxi_start_min)} - {t(p.taxi_end_min)}",
             "ground wait": (p.hold_area_id or "-"),
             "gate": p.gate_id + (" (changed)" if p.gate_id != flights[p.flight_id].original_gate_id else ""),
             "gate arrival (on-block)": t(p.gate_in_min), "service": f"{t(p.service_start_min)} - {t(p.service_end_min)}",
             "ground crew": p.ground_crew_id, "flight crew": p.flight_crew_id}
        if p.takeoff_min is not None:
            sched = scenario.simulated.departure.scheduled_departure_min.get(p.flight_id) if scenario.simulated.departure else None
            r.update({"ready for departure": t(p.ready_min), "dep. runway": p.dep_runway_id, "outbound taxi": " > ".join(p.dep_taxi_route),
                      "take-off": t(p.takeoff_min), "scheduled departure": t(sched) if sched is not None else "-",
                      "equipment": ", ".join(sorted({x.equipment_id for x in p.tasks if x.equipment_id})) or "-"})
        rows.append(r)
    return rows


def task_rows(plan: RecoveryPlan) -> list[dict]:
    """Turnaround task schedule (departure sessions)."""
    return [{"flight": p.flight_id, "task": x.task_id, "start": f"t+{x.start_min:.1f}", "end": f"t+{x.end_min:.1f}",
             "minutes": round(x.end_min - x.start_min, 1), "equipment": x.equipment_id or "-"} for p in plan.flights for x in p.tasks]
