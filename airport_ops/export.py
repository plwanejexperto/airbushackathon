"""Downloadable JSON of one negotiation session. Contains scenario version, reference plan, messages, proposal versions,
structured changes, votes, validation results and outcome. Credentials are never included, and the final text is passed through
`llm.sanitize` as a second guard."""
from __future__ import annotations

import json

from . import llm
from .changes import build_change_log
from .demo import is_demo_scenario

from .negotiation import FREEZE_RULE
from .negotiation_types import NegotiationResult
from .schemas import Scenario
from .validate import ValidationReport


def _val(rep: ValidationReport | None, full: bool = True) -> dict | None:
    if rep is None:
        return None
    d = {"passed": rep.passed, "n_checks": len(rep.results),
         "failed": [r.model_dump(mode="json") for r in rep.failures]}
    if full:
        d["results"] = [r.model_dump(mode="json") for r in rep.results]
    return d


def export_session(sc: Scenario, res: NegotiationResult) -> dict:
    cl = build_change_log(sc, res)
    fb = res.fallback
    return {
        "export_format": "airport-negotiation-session/1",
        "scenario": {"scenario_id": sc.scenario_id, "scenario_version": res.scenario_version, "demo_preset": is_demo_scenario(sc),
                     "kind": "diversion" if sc.simulated.diversion else "turnaround_departure" if sc.simulated.departure else "arrival",
                     "airport": sc.workbook.airport.iata_code,
                     "diversion": sc.simulated.diversion.model_dump(mode="json") if sc.simulated.diversion else None,
                     "departure": ({"next_flights": [n.model_dump(mode="json") for n in sc.simulated.departure.next_flights],
                                    "scheduled_departure_min": sc.simulated.departure.scheduled_departure_min,
                                    "arrival_plan_source": sc.simulated.departure.arrival_plan_source, "note": sc.simulated.departure.note}
                                   if sc.simulated.departure else None),
                     "note": "times are simulated minutes from scenario start (t+N); not historical actual times"},
        "session": {"session_id": res.session_id, "mode": res.mode, "model": res.model_name,
                    "live_model_calls": res.mode == "live", "elapsed_s": res.elapsed_s, "deadline_s": res.deadline_s,
                    "turns_completed": res.turns_completed, "messages": len(res.messages),
                    "participants": [p.model_dump(mode="json") for p in res.participants]},
        "selected_historical_disruption": sc.historical.model_dump(mode="json"),
        "initial_reference_plan": {"plan": res.reference_plan.model_dump(mode="json") if res.reference_plan else None,
                                   "validation": _val(res.reference_validation),
                                   "note": "illustrative reference (fails hard constraints when invalid): not a feasible baseline, not the historical actual outcome, no improvement claimed"},
        "messages": [m.model_dump(mode="json") for m in res.messages],
        "proposal_versions": [{"proposal_id": p.proposal_id, "version_no": p.version_no, "proposer": p.proposer.value, "turn": p.turn,
                               "seq": p.seq, "referenced_proposal_id": p.referenced_proposal_id, "digest": p.digest,
                               "origin": p.origin, "plan": json.loads(p.plan_json), "validation": _val(p.validation, full=False),
                               "metrics": p.metrics} for p in res.proposals],
        "structured_changes": cl.model_dump(mode="json"),
        "freeze": {"frozen_proposal_id": res.frozen_proposal_id, "rule": FREEZE_RULE, "ranking": res.freeze_ranking},
        "final_votes": [v.model_dump(mode="json") for v in res.votes],
        "final_validation": _val(res.final_validation),
        "outcome": {"status": res.status.value, "agent_consensus": res.consensus,
                    "final_plan_kind": cl.final_kind, "final_plan": res.final_plan.model_dump(mode="json") if res.final_plan else None,
                    "fallback_reason": res.fallback_reason.value if res.fallback_reason else None,
                    "fallback_detail": res.fallback_detail, "fallback_outcome": fb.outcome.value if fb else None,
                    "fallback_search_status": fb.search_status.value if fb else None,
                    "fallback_reasons": fb.reasons if fb else None, "invalidation": res.invalidation_detail,
                    "escalation_required": res.status.value == "NO_FEASIBLE_PLAN"},
        "model_attempts": [a.model_dump(mode="json") for a in res.attempts],
        "tool_calls": [t.model_dump(mode="json") for t in res.tool_calls],
        "validation_events": [v.model_dump(mode="json") for v in res.validations],
        "decision_records": [d.model_dump(mode="json") for d in res.decisions],
    }


def export_json(sc: Scenario, res: NegotiationResult) -> str:
    return llm.sanitize(json.dumps(export_session(sc, res), indent=1, default=str))
