"""Structured differences between the initial reference plan, each agent proposal and the final plan.

Only things the model genuinely represents are diffed: landing time, runway, taxi route (a modelled taxi graph), gate, gate
arrival, ground crew and service times. En-route flight-plan changes are NOT modelled and are never reported.

Times are simulated minutes from scenario start. "Landing" is touchdown; "gate arrival" is on-block (the flights.csv
ARRIVAL_TIME meaning). Both are always labelled separately.
"""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field

from .negotiation_types import NegotiationResult, ProposalRecord
from .plan import FlightPlan, RecoveryPlan
from .schemas import Scenario

TOL = 0.05  # minutes: smaller numeric differences are not reported as changes

LABELS = {
    "landing_min": "Landing / touchdown time (t+ min)",
    "gate_in_min": "Gate arrival / on-block time (t+ min)",
    "gate_id": "Gate assignment",
    "runway_id": "Runway",
    "taxi_route": "Taxi route (modelled taxi graph)",
    "ground_crew_id": "Ground crew",
    "service_start_min": "Service start (t+ min)",
    "service_end_min": "Service end (t+ min)",
}
FIELD_CONSTRAINTS = {
    "landing_min": ["WAKE-SEP", "RWY-OCC", "RWY-AVAIL", "RWY-CAP-AIRPORT", "RWY-CAP-RUNWAY", "FUEL-RESERVE", "FUEL-CONTINGENCY"],
    "runway_id": ["RWY-AVAIL", "WAKE-SEP", "RWY-OCC"],
    "taxi_route": ["TAXI-BLOCK", "TAXI-CONFLICT", "TAXI-ROUTE"],
    "gate_id": ["GATE-OVERLAP", "GATE-COMPAT", "GATE-NOTICE"],
    "gate_in_min": ["GATE-OVERLAP", "TAXI-BLOCK", "TAXI-CONFLICT"],
    "ground_crew_id": ["CREW-OVERLAP", "CREW-GROUND-SHIFT", "CREW-GROUND-AVAIL"],
    "service_start_min": ["CREW-OVERLAP", "TASK-TIME"],
    "service_end_min": ["CREW-OVERLAP", "TURNAROUND-MIN", "CREW-GROUND-SHIFT"],
}
Value = Union[float, str]


class ChangeRecord(BaseModel):
    flight_id: str
    field: str
    label: str
    before: Value
    after: Value
    delta_min: Optional[float] = None
    step: Optional[int] = Field(None, description="index in the proposal lineage (0 = first proposal vs the reference plan)")
    proposal_id: Optional[str] = None
    proposed_by: Optional[str] = None          # ATC | PILOT | GROUND | "FALLBACK PLANNER"
    message_seq: Optional[int] = None
    reason_codes: list[str] = Field(default_factory=list)
    explanation: str = ""
    evidence: list[dict] = Field(default_factory=list)   # constraint evidence: reference result vs final result
    driver: str = ""                                      # why: "constraint violation in the reference plan" | "objective trade-off"


class ChangeLog(BaseModel):
    reference_plan_id: str
    final_plan_id: Optional[str]
    final_kind: str                           # "agent consensus" | "deterministic fallback" | "none"
    lineage: list[dict] = Field(default_factory=list)
    step_changes: list[ChangeRecord] = Field(default_factory=list)   # every revision along the counter-offer chain
    final_changes: list[ChangeRecord] = Field(default_factory=list)  # reference -> final, each attributed
    selected_flight: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def _val(fp: FlightPlan, field: str) -> Value:
    v = getattr(fp, field)
    return " > ".join(v) if field == "taxi_route" else (v if isinstance(v, str) else float(v))


def diff_plans(before: RecoveryPlan, after: RecoveryPlan) -> list[tuple[str, str, Value, Value]]:
    """[(flight_id, field, before, after)] for every genuine difference (numeric tolerance TOL)."""
    b = {p.flight_id: p for p in before.flights}
    out = []
    for p in after.flights:
        q = b.get(p.flight_id)
        if q is None:
            continue
        for f in LABELS:
            x, y = _val(q, f), _val(p, f)
            if (isinstance(x, float) and abs(x - y) > TOL) or (not isinstance(x, float) and x != y):
                out.append((p.flight_id, f, x, y))
    return out


def _evidence_for(res: NegotiationResult, flight: str, field: str) -> tuple[list[dict], str]:
    ref, fin = res.reference_validation, res.final_validation
    ev: list[dict] = []
    for cid in FIELD_CONSTRAINTS.get(field, []):
        bad = [r for r in (ref.failures if ref else []) if r.constraint_id == cid and (r.flight_id == flight or flight in r.explanation)]
        if not bad:
            continue
        good = [r for r in (fin.results if fin else []) if r.constraint_id == cid and r.passed and (r.flight_id == flight or r.flight_id is None)]
        ev.append({"constraint_id": cid, "reference_result": "FAIL: " + bad[0].explanation[:180],
                   "final_result": ("pass: " + good[0].explanation[:140]) if good else "pass"})
    return ev, ("constraint violation in the reference plan" if ev else "objective trade-off between agents (no reference violation for this field)")


def _msg_for(res: NegotiationResult, pid: str):
    return next((m for m in res.messages if m.proposal_id == pid and m.plan is not None), None)


def lineage_of(res: NegotiationResult, pid: str) -> list[ProposalRecord]:
    chain, seen = [], set()
    cur = res.proposal(pid)
    while cur is not None and cur.proposal_id not in seen:
        chain.append(cur)
        seen.add(cur.proposal_id)
        cur = res.proposal(cur.referenced_proposal_id) if cur.referenced_proposal_id else None
    return list(reversed(chain))


def build_change_log(sc: Scenario, res: NegotiationResult) -> ChangeLog:
    ref, final = res.reference_plan, res.final_plan
    log = ChangeLog(reference_plan_id=ref.proposal_id if ref else "", final_plan_id=final.proposal_id if final else None,
                    final_kind="agent consensus" if res.consensus else ("deterministic fallback" if final else "none"))
    log.notes += ["En-route flight-plan adjustments are not modelled; only landing, runway, taxi route, gate, crew and service "
                  "times can change.",
                  "Reference/revised times are simulated minutes from scenario start; they are not the historical actual times.",
                  "The reference plan is illustrative and usually infeasible; changes explain what had to move, they are NOT an efficiency "
                  "improvement claim (that needs a comparison with a feasible plan)."]
    if ref is None:
        return log
    chain = lineage_of(res, final.proposal_id) if (res.consensus and final) else []
    log.lineage = [{"step": i, "proposal_id": p.proposal_id, "version_no": p.version_no, "proposer": p.proposer.value, "seq": p.seq,
                    "referenced_proposal_id": p.referenced_proposal_id, "valid": p.validation.passed} for i, p in enumerate(chain)]
    prev = ref
    for i, rec in enumerate(chain):
        m = _msg_for(res, rec.proposal_id)
        cur = rec.plan
        for fid, f, x, y in diff_plans(prev, cur):
            log.step_changes.append(ChangeRecord(
                flight_id=fid, field=f, label=LABELS[f], before=x, after=y,
                delta_min=round(y - x, 2) if isinstance(x, float) else None, step=i, proposal_id=rec.proposal_id,
                proposed_by=rec.proposer.value, message_seq=rec.seq, reason_codes=[r.value for r in (m.reason_codes if m else [])],
                explanation=m.explanation if m else ""))
        prev = cur
    if final is not None:
        for fid, f, x, y in diff_plans(ref, final):
            ev, driver = _evidence_for(res, fid, f)
            who = next((c for c in reversed(log.step_changes) if c.flight_id == fid and c.field == f and c.after == y), None)
            log.final_changes.append(ChangeRecord(
                flight_id=fid, field=f, label=LABELS[f], before=x, after=y, delta_min=round(y - x, 2) if isinstance(x, float) else None,
                step=who.step if who else None, proposal_id=who.proposal_id if who else final.proposal_id,
                proposed_by=who.proposed_by if who else ("FALLBACK PLANNER" if not res.consensus else None),
                message_seq=who.message_seq if who else None, reason_codes=who.reason_codes if who else [],
                explanation=who.explanation if who else "", evidence=ev, driver=driver))
    sel = sc.simulated.flights[0]
    rp = next((p for p in ref.flights if p.flight_id == sel.flight_id), None)
    fp = next((p for p in final.flights if p.flight_id == sel.flight_id), None) if final else None
    h = sc.historical
    log.selected_flight = {
        "flight_id": sel.flight_id, "nominal_eta_min": sel.earliest_landing_min,
        "reference": {"landing_min": rp.landing_min, "gate_arrival_min": rp.gate_in_min, "gate": rp.gate_id, "runway": rp.runway_id,
                      "taxi_route": " > ".join(rp.taxi_route)} if rp else None,
        "revised": {"landing_min": fp.landing_min, "gate_arrival_min": fp.gate_in_min, "gate": fp.gate_id, "runway": fp.runway_id,
                    "taxi_route": " > ".join(fp.taxi_route)} if fp else None,
        "changed": {c.field: [c.before, c.after] for c in log.final_changes if c.flight_id == sel.flight_id},
        "proposed_by": {c.field: c.proposed_by for c in log.final_changes if c.flight_id == sel.flight_id},
        "historical_context_only": {"scheduled_arrival": h.scheduled_arrival, "actual_arrival": h.actual_arrival,
                                    "arrival_delay_min": h.arrival_delay_min, "weather_delay_min": h.weather_delay_min,
                                    "late_aircraft_delay_min": h.late_aircraft_delay_min,
                                    "note": "dataset record; NOT the negotiated result and not 'recovered' by it"}}
    return log
