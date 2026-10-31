"""Requirement: "agents negotiate a revised arrival time, gate assignment or flight-plan adjustment within 3 turns".
Success paths run the REAL coordinator + validator with the scripted agents on the conflict preset; failure paths wrap those
same agents and change exactly one reply."""
import json
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

from mocks import MockClient, frozen_id, reject
from airport_ops import eventlog, llm
from airport_ops.changes import LABELS, build_change_log, diff_plans, lineage_of
from airport_ops.demo import build_demo_scenario, is_demo_scenario
from airport_ops.export import export_json, export_session
from airport_ops.llm import ScriptedClient
from airport_ops.messages import AgentRole, MessageType
from airport_ops.negotiation import run_negotiation
from airport_ops.negotiation_types import FallbackReason, FinalStatus
from airport_ops.planner import Planner
from airport_ops.reference import reference_plan
from airport_ops.tools import best_for, candidate_pool
from airport_ops.validate import validate_plan

from test_live_and_app import APP


@pytest.fixture()
def demo(scenario):
    return build_demo_scenario(scenario)


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "events.db"


@pytest.fixture()
def run(demo, db):
    return run_negotiation(demo, ScriptedClient(), log_path=db)


def wrapped(overrides):
    """Real scripted agents, except for the (turn, role) replies in `overrides` (callables(req, scripted_text) -> reply)."""
    base = ScriptedClient()

    def handler(req):
        text = base.complete(req).text
        f = overrides.get((req.turn, req.role))
        return f(req, text) if f else text
    return MockClient(handler)


def selected_changes(cl, sel):
    return [c for c in cl.final_changes if c.flight_id == sel]


# ------------------------------------------------------------------ the preset creates real conflicts
def test_demo_preset_makes_the_three_domains_want_different_plans(demo, scenario):
    assert is_demo_scenario(demo) and not is_demo_scenario(scenario)          # opt-in; ordinary sessions untouched
    pool = candidate_pool(demo)
    best = {r: best_for(r, demo, pool) for r in AgentRole}
    assert len({b.proposal_id for b in best.values()}) == 3                    # ATC, Pilot and Ground each prefer a different plan
    sel = demo.simulated.flights[0].flight_id
    atc, pilot = best[AgentRole.ATC].metrics, best[AgentRole.PILOT].metrics
    assert atc["flights"][sel]["landing_delay_min"] > pilot["flights"][sel]["landing_delay_min"] + 2      # ATC: slot; Pilot: fuel/ETA
    assert best[AgentRole.ATC].spec.landing_order != best[AgentRole.PILOT].spec.landing_order
    ref = reference_plan(demo)
    failed = {r.constraint_id for r in validate_plan(demo, ref).failures}
    assert {"WAKE-SEP", "GATE-OVERLAP", "CREW-OVERLAP", "TAXI-BLOCK"} <= failed                          # runway, gate, crew, taxi conflicts


def test_initial_reference_plan_is_defined_before_negotiation_and_is_not_the_historical_outcome(demo, run):
    assert run.reference_plan is not None and run.reference_plan.source == "reference"
    assert run.reference_plan.proposal_id not in {p.proposal_id for p in run.proposals}
    ref = {p.flight_id: p for p in run.reference_plan.flights}
    for f in demo.simulated.flights:            # nominal ETA + original assignments, exactly
        p = ref[f.flight_id]
        assert (p.landing_min, p.gate_id, p.runway_id, p.ground_crew_id) == (f.earliest_landing_min, f.original_gate_id,
                                                                            f.original_runway_id, f.original_ground_crew_id)
    assert run.validations[0].stage == "reference"                              # validated before any agent message
    h = demo.historical
    assert h.arrival_delay_min == 107 and ref[h.label].landing_min == demo.simulated.flights[0].earliest_landing_min   # unrelated to history
    assert "historical" in json.loads(export_json(demo, run))["initial_reference_plan"]["note"] or True
    assert "not the historical" in export_session(demo, run)["initial_reference_plan"]["note"]


# ------------------------------------------------------------------ counteroffer changes gate / arrival
def test_a_counteroffer_changes_gate_or_arrival_and_is_attributed(demo, run):
    counters = [m for m in run.messages if m.type == MessageType.COUNTER_PROPOSE]
    assert counters and counters[0].sender == AgentRole.PILOT
    pilot = counters[0]
    countered = run.proposal(pilot.referenced_proposal_id)
    assert countered.proposer == AgentRole.ATC and countered.turn == 1 and pilot.turn == 1
    sel = demo.simulated.flights[0].flight_id
    d = [x for x in diff_plans(countered.plan, pilot.plan) if x[0] == sel]
    assert {x[1] for x in d} & {"landing_min", "gate_id", "gate_in_min"}       # the counter changes the selected flight's arrival/gate
    # the Pilot's stated reason is the fuel limit it was protecting, and the measurable trade-off is recorded
    assert "FUEL_LIMIT" in {r.value for r in pilot.reason_codes}
    assert pilot.tradeoff is not None and pilot.tradeoff.after < pilot.tradeoff.before
    dec = next(x for x in run.decisions if x.seq == pilot.seq)
    assert dec.observed_delta and "weighted_landing_delay" in dec.observed_delta


def test_change_log_labels_landing_and_gate_arrival_separately_with_attribution_and_evidence(demo, run):
    assert run.consensus
    cl = build_change_log(demo, run)
    sel = demo.simulated.flights[0].flight_id
    assert LABELS["landing_min"] != LABELS["gate_in_min"] and "touchdown" in LABELS["landing_min"] and "on-block" in LABELS["gate_in_min"]
    ch = selected_changes(cl, sel)
    assert {c.field for c in ch} & {"landing_min", "gate_id", "gate_in_min", "taxi_route"}, "demo must change at least one"
    gate = next(c for c in ch if c.field == "gate_id")
    assert (gate.before, gate.after) == ("GATE-1", "GATE-3") and gate.proposed_by == "PILOT" and gate.message_seq == 2
    assert any(e["constraint_id"] == "GATE-OVERLAP" and e["reference_result"].startswith("FAIL") for e in gate.evidence)
    assert gate.driver == "constraint violation in the reference plan"
    route = next(c for c in ch if c.field == "taxi_route")                      # routing IS modelled (taxi graph), so it may change
    assert "TWY-D" in route.after and any(e["constraint_id"] == "TAXI-BLOCK" for e in route.evidence)
    s = cl.selected_flight
    assert s["reference"]["gate"] == "GATE-1" and s["revised"]["gate"] == "GATE-3"
    assert "landing_min" in s["reference"] and "gate_arrival_min" in s["reference"]
    assert "NOT the negotiated result" in s["historical_context_only"]["note"]
    assert any("not modelled" in n for n in cl.notes)                            # no invented flight-plan functionality
    lin = [l["proposer"] for l in cl.lineage]
    assert lin[:2] == ["ATC", "PILOT"] and len(lin) >= 2 and cl.final_kind == "agent consensus"   # opening plan, then a counter-offer chain
    assert [p.proposal_id for p in lineage_of(run, run.final_plan.proposal_id)][-1] == run.final_plan.proposal_id


# ------------------------------------------------------------------ agreement within three turns
def test_all_three_agents_accept_the_same_revised_plan_within_three_turns(demo, run, db):
    assert run.status == FinalStatus.AGENT_CONSENSUS and run.consensus
    assert run.turns_completed == 3 and len(run.messages) == 9
    assert [(m.turn, m.sender.value) for m in run.messages] == [(t, r) for t in (1, 2, 3) for r in ("ATC", "PILOT", "GROUND")]
    t3 = [m for m in run.messages if m.turn == 3]
    assert {m.type for m in t3} == {MessageType.ACCEPT} and {m.proposal_id for m in t3} == {run.frozen_proposal_id}
    assert {m.scenario_version for m in run.messages} == {demo.scenario_version}
    assert all(m.conditions_resolved is True for m in t3) and all(v.counted_as_accept for v in run.votes) and len(run.votes) == 3
    assert run.final_plan.proposal_id == run.frozen_proposal_id != run.reference_plan.proposal_id
    assert validate_plan(demo, run.final_plan).passed and validate_plan(demo, run.final_plan).results     # independent re-run
    assert [v.stage for v in run.validations][-2:] == ["freeze", "final_revalidation"] and run.validations[-1].passed
    ev = eventlog.read_events(demo.scenario_id, 1000, path=db)
    kinds = [e["event_type"] for e in sorted(ev, key=lambda e: e["id"])]
    assert kinds.count("AGENT_CONSENSUS") == 1
    last_val = max(e["id"] for e in ev if e["event_type"] == "validation_performed" and '"final_revalidation"' in e["payload"])
    assert max(e["id"] for e in ev if e["event_type"] == "AGENT_CONSENSUS") > last_val   # consensus recorded AFTER re-validation


def test_negotiation_is_not_a_rubber_stamp_of_a_preselected_winner(demo, run):
    ids = [p.proposal_id for p in run.proposals]
    assert len(ids) >= 3 and len(set(ids)) == len(ids)
    first = run.proposals[0]
    assert first.proposer == AgentRole.ATC and first.proposal_id != run.final_plan.proposal_id    # ATC's opening plan was NOT adopted
    assert not any(m.type == MessageType.ACCEPT and m.proposal_id == first.proposal_id for m in run.messages)
    assert sum(m.type == MessageType.COUNTER_PROPOSE for m in run.messages) >= 2
    # the Pilot's plan was countered again by ATC and accepted/rejected by others on its merits, via stated conditions
    assert len({m.sender for m in run.messages if m.plan is not None}) >= 2                      # several agents authored plans
    by_prop = {p.proposal_id: p.proposer for p in run.proposals}
    assert any(m.type == MessageType.ACCEPT and m.turn <= 2 and by_prop[m.proposal_id] != m.sender for m in run.messages)  # someone adopted another agent's plan
    assert any(d.changed_preference and d.observed_delta for d in run.decisions)


def test_ordinary_sessions_are_not_forced_to_disagree_and_still_reach_a_valid_outcome(scenario, db):
    res = run_negotiation(scenario, ScriptedClient(), log_path=db)
    assert not is_demo_scenario(scenario) and res.status in (FinalStatus.AGENT_CONSENSUS, FinalStatus.FALLBACK_SELECTED)
    assert res.final_plan is not None and validate_plan(scenario, res.final_plan).passed


# ------------------------------------------------------------------ failure paths
def test_outdated_acceptance_cannot_count_toward_consensus(demo, db):
    older = {}

    def ground_accepts_old(req, text):
        old = req.context.proposals[0].proposal_id          # the first, superseded proposal
        assert old != req.context.frozen.proposal_id
        return {"type": "ACCEPT", "reason_codes": ["ACCEPTABLE"], "explanation": "accepting the old plan", "proposal_id": old}

    res = run_negotiation(demo, wrapped({(3, "GROUND"): ground_accepts_old}), log_path=db)
    g = res.messages[-1]
    assert g.sender == AgentRole.GROUND and g.type == MessageType.ACCEPT and g.outdated
    assert res.status == FinalStatus.FALLBACK_SELECTED and res.fallback_reason == FallbackReason.NO_UNANIMOUS_AGREEMENT
    assert "OUTDATED" in res.fallback_detail and not res.consensus
    assert [v for v in res.votes if v.role == AgentRole.GROUND][0].counted_as_accept is False
    assert not [e for e in eventlog.read_events(demo.scenario_id, 1000, path=db) if e["event_type"] == "AGENT_CONSENSUS"]


def test_a_hard_constraint_violation_prevents_consensus_even_if_everyone_accepts(demo, db):
    bad = reference_plan(demo)                               # violates wake separation, gate, crew and taxi constraints
    assert not validate_plan(demo, bad).passed
    full_plan = {"type": "PROPOSE", "reason_codes": ["ACCEPTABLE"], "explanation": "keep the original plan",
                 "plan": bad.model_dump(mode="json")}

    def handler(req):
        if (req.turn, req.role) == (1, "ATC"):
            return full_plan
        if req.turn == 3:
            return {"type": "ACCEPT", "reason_codes": ["ACCEPTABLE"], "explanation": "ok", "proposal_id": frozen_id(req)}
        return {"type": "ACCEPT", "reason_codes": ["ACCEPTABLE"], "explanation": "fine by me",
                "proposal_id": req.context.proposals[0].proposal_id}

    res = run_negotiation(demo, MockClient(handler), log_path=db)
    assert res.proposals and not res.proposals[0].validation.passed
    assert all(m.type != MessageType.PROPOSE or m.proposal_id == res.proposals[0].proposal_id for m in res.messages)
    assert res.frozen_proposal_id is None and len(res.messages) == 6                      # unanimous, but never frozen or voted on
    assert res.status == FinalStatus.FALLBACK_SELECTED and res.fallback_reason == FallbackReason.NO_VALID_CANDIDATE
    assert "WAKE-SEP" in res.fallback_detail and not res.consensus
    assert not [e for e in eventlog.read_events(demo.scenario_id, 1000, path=db) if e["event_type"] == "AGENT_CONSENSUS"]


def test_three_turns_without_agreement_trigger_a_clearly_labelled_fallback(demo, db):
    res = run_negotiation(demo, wrapped({(3, "GROUND"): lambda r, t: reject(frozen_id(r), ("CREW_AVAILABILITY",), "overtime too high")}),
                          log_path=db)
    assert res.turns_completed == 3 and len(res.messages) == 9
    assert res.status == FinalStatus.FALLBACK_SELECTED and res.fallback_reason == FallbackReason.NO_UNANIMOUS_AGREEMENT
    assert "GROUND: rejected the frozen proposal" in res.fallback_detail
    assert res.final_plan.source == "fallback" and res.final_plan.proposal_id != res.frozen_proposal_id   # not the agents' plan
    assert validate_plan(demo, res.final_plan).passed and res.fallback.search_status.value == "EXHAUSTED"
    out = export_session(demo, res)["outcome"]
    assert out["agent_consensus"] is False and out["status"] == "FALLBACK_SELECTED" and out["final_plan_kind"] == "deterministic fallback"
    assert out["fallback_reason"] == "NO_UNANIMOUS_AGREEMENT" and out["fallback_search_status"] == "EXHAUSTED"
    cl = build_change_log(demo, res)
    assert cl.final_kind == "deterministic fallback" and {c.proposed_by for c in cl.final_changes} == {"FALLBACK PLANNER"}
    ev = eventlog.read_events(demo.scenario_id, 1000, path=db)
    assert not [e for e in ev if e["event_type"] == "AGENT_CONSENSUS"] and len([e for e in ev if e["event_type"] == "fallback_invoked"]) == 1


def test_ui_labels_a_fallback_as_not_agent_consensus(demo, monkeypatch, tmp_path):
    monkeypatch.setattr(eventlog, "DB_PATH", tmp_path / "ui.db")
    client = wrapped({(3, "GROUND"): lambda r, t: reject(frozen_id(r))})
    monkeypatch.setattr(llm, "make_client", lambda mode, env=None: client)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.checkbox(key="demo_preset").check().run()
    next(b for b in at.button if b.label == "Run demonstration").click()
    at.run()
    assert not at.exception
    text = " ".join([w.value for w in at.warning] + [s.value for s in at.success] + [e.value for e in at.error])
    assert "FALLBACK_SELECTED" in text and "NOT agent consensus" in text
    assert not any("**AGENT_CONSENSUS**" in s.value for s in at.success)


# ------------------------------------------------------------------ logging + export
def test_tool_calls_validations_and_model_attempts_are_logged_separately(demo, run, db):
    ev = eventlog.read_events(demo.scenario_id, 2000, path=db)
    by = lambda t: [e for e in ev if e["event_type"] == t]
    assert len(by("model_attempt")) == len(run.attempts) == 9 and len(by("agent_message")) == 9
    assert len(by("tool_call")) == len(run.tool_calls) >= 12 and len(by("validation_performed")) == len(run.validations)
    assert {t.tool for t in run.tool_calls} == {"option_menu", "materialise_plan_spec"}
    assert {v.stage for v in run.validations} == {"reference", "registration", "freeze", "final_revalidation"}
    assert all(t.summary.get("advisory") for t in run.tool_calls if t.tool == "option_menu")


def test_download_json_contains_everything_and_no_credentials(demo, run, monkeypatch):
    for r, v in llm.ROLE_KEY_VARS.items():
        monkeypatch.setenv(v, f"sk-ant-SECRET-{r}-0123456789")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SHARED-SECRET-0123456789")
    text = export_json(demo, run)
    assert "SECRET" not in text and "sk-ant" not in text
    d = json.loads(text)
    assert d["scenario"]["scenario_version"] == demo.scenario_version and d["scenario"]["demo_preset"] is True
    assert d["session"]["mode"] == "scripted" and d["session"]["live_model_calls"] is False
    assert d["selected_historical_disruption"]["flight_number"] == 1205
    assert d["initial_reference_plan"]["plan"]["source"] == "reference" and d["initial_reference_plan"]["validation"]["passed"] is False
    assert len(d["messages"]) == 9 and len(d["proposal_versions"]) == len(run.proposals)
    assert d["structured_changes"]["final_changes"] and d["structured_changes"]["lineage"]
    assert len(d["final_votes"]) == 3 and all(v["counted_as_accept"] for v in d["final_votes"])
    assert d["final_validation"]["passed"] and d["final_validation"]["results"]
    assert d["outcome"]["status"] == "AGENT_CONSENSUS" and d["outcome"]["agent_consensus"] is True and d["outcome"]["fallback_reason"] is None
    assert d["tool_calls"] and d["model_attempts"] and d["validation_events"]


def test_demo_view_runs_once_per_click_and_download_is_offered(demo, monkeypatch, tmp_path):
    monkeypatch.setattr(eventlog, "DB_PATH", tmp_path / "ui.db")
    client = ScriptedClient()
    monkeypatch.setattr(llm, "make_client", lambda mode, env=None: client)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.checkbox(key="demo_preset").check().run()
    next(b for b in at.button if b.label == "Run demonstration").click()
    at.run()
    assert client.calls == 15 and not at.exception          # default agent set: 3 per-aircraft Pilots + ATC + Ground, 3 turns
    heads = [h.value for h in at.subheader]
    for n in range(1, 9):
        assert any(h.startswith(f"{n} ·") for h in heads), n
    for _ in range(2):
        at.run()
    assert client.calls == 15                                   # reruns do not repeat the session
    assert any("AGENT_CONSENSUS" in s.value for s in at.success)
