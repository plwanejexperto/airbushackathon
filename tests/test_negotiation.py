import json
import sqlite3

import pytest

from mocks import MockClient, Script, accept, agree_on, counter, frozen_id, propose, reject
from airport_ops import eventlog
from airport_ops.fallback import Outcome, SearchStatus
from airport_ops.llm import ModelAPIError, ModelTimeout, ScriptedClient
from airport_ops.messages import (AgentOutput, Condition, ConditionField, ConditionOp, MessageType)
from airport_ops.metrics import evaluate_condition, plan_metrics
from airport_ops.negotiation import Coordinator, NegotiationConfig, run_negotiation
from airport_ops.negotiation_types import FallbackReason, FinalStatus
from airport_ops.planner import Planner
from airport_ops.scenario import rehash_scenario
from airport_ops.tools import best_for, candidate_pool
from airport_ops.workbook import AircraftClass as AC, WakeCategory as W
from airport_ops.messages import AgentRole


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "events.db"


@pytest.fixture()
def pool(scenario):
    p = candidate_pool(scenario)
    assert len(p) > 10
    return p


def events(db, sc, etype=None):
    ev = eventlog.read_events(sc.scenario_id, 1000, path=db)
    return [e for e in ev if etype is None or e["event_type"] == etype]


# ------------------------------------------------------------------ success
def test_agreement_within_three_turns_with_independent_revalidation(scenario, pool, db):
    cand = pool[0]
    client = MockClient(agree_on(cand))
    res = run_negotiation(scenario, client, log_path=db)
    assert res.status == FinalStatus.AGENT_CONSENSUS and res.consensus
    assert res.turns_completed == 3 and len(res.messages) == 9 and client.calls == 9
    assert [(m.turn, m.sender.value) for m in res.messages] == [(t, r) for t in (1, 2, 3) for r in ("ATC", "PILOT", "GROUND")]
    assert res.frozen_proposal_id == cand.proposal_id and res.final_plan.proposal_id == cand.proposal_id
    assert res.final_validation.passed and res.fallback is None and res.fallback_reason is None
    assert all(v.counted_as_accept for v in res.votes) and len(res.votes) == 3
    assert all(m.scenario_version == scenario.scenario_version for m in res.messages)
    # events: one consensus record, written after the frozen candidate, 9 messages, 9 model attempts logged separately
    assert len(events(db, scenario, "AGENT_CONSENSUS")) == 1 and len(events(db, scenario, "agent_message")) == 9
    assert len(events(db, scenario, "model_attempt")) == 9 and len(events(db, scenario, "candidate_frozen")) == 1
    assert not events(db, scenario, "fallback_invoked")


def test_fallback_is_never_run_first(scenario, pool, db, monkeypatch):
    import airport_ops.negotiation as neg
    order = []
    real = neg.run_fallback
    monkeypatch.setattr(neg, "run_fallback", lambda *a, **k: (order.append("fallback"), real(*a, **k))[1])
    client = MockClient(lambda r: (order.append(f"{r.turn}{r.role}"), agree_on(pool[0])(r))[1])
    res = run_negotiation(scenario, client, log_path=db)
    assert res.consensus and "fallback" not in order


def test_acceptance_with_unresolved_condition_is_not_final(scenario, pool, db):
    sel = scenario.simulated.flights[0].flight_id
    impossible = {"field": "landing_delay_min", "op": "<=", "value": -5, "flight_id": sel}  # can never hold
    t = agree_on(pool[0]).table
    t[(3, "PILOT")] = lambda r: accept(frozen_id(r), conditions=[impossible])
    res = run_negotiation(scenario, MockClient(Script(t)), log_path=db)
    assert res.status == FinalStatus.FALLBACK_SELECTED and res.fallback_reason == FallbackReason.NO_UNANIMOUS_AGREEMENT
    pilot = res.messages[-2]
    assert pilot.type == MessageType.ACCEPT and pilot.conditions_resolved is False
    assert not [v for v in res.votes if v.role.value == "PILOT"][0].counted_as_accept
    assert "unresolved" in res.fallback_detail
    assert not res.consensus and not events(db, scenario, "AGENT_CONSENSUS")


# ------------------------------------------------------------------ validator rejects
def test_invalid_proposal_is_rejected_by_validator_and_evidence_is_fed_back(scenario, db):
    naive = Planner(scenario).naive_plan()  # violates wake separation, taxi/crew conflicts, ...
    bad = {"type": "PROPOSE", "reason_codes": ["ACCEPTABLE"], "explanation": "original timings", "plan": naive.model_dump(mode="json")}
    seen = {}

    def handler(req):
        if (req.turn, req.role) == (1, "ATC"):
            return bad
        seen[(req.turn, req.role)] = req.prompt
        pid = req.context.proposals[0].proposal_id
        return accept(pid)

    res = run_negotiation(scenario, MockClient(handler), log_path=db)
    rec = res.proposals[0]
    assert not rec.validation.passed and rec.validation.failures
    assert res.status == FinalStatus.FALLBACK_SELECTED and res.fallback_reason == FallbackReason.NO_VALID_CANDIDATE
    assert res.frozen_proposal_id is None and res.turns_completed == 2 and len(res.messages) == 6
    assert not res.consensus and res.fallback.outcome == Outcome.FALLBACK_SELECTED
    # structured evidence reached the next agent: constraint ids of the failed proposal appear in its prompt
    ev = json.loads(seen[(1, "PILOT")])["proposals_so_far"][0]
    assert ev["valid"] is False and ev["failed_constraints"]
    assert ev["failed_constraints"][0]["constraint"] in {r.constraint_id for r in rec.validation.failures}
    assert next(e for e in events(db, scenario, "proposal_registered"))


# ------------------------------------------------------------------ no unanimity
def test_no_unanimous_agreement_triggers_fallback_not_consensus(scenario, pool, db):
    t = agree_on(pool[0]).table
    t[(3, "GROUND")] = lambda r: reject(frozen_id(r), ("CREW_AVAILABILITY",))
    res = run_negotiation(scenario, MockClient(Script(t)), log_path=db)
    assert res.status == FinalStatus.FALLBACK_SELECTED and res.fallback_reason == FallbackReason.NO_UNANIMOUS_AGREEMENT
    assert len(res.messages) == 9 and res.final_plan.proposal_id == res.fallback.plan.proposal_id
    assert res.final_plan.source == "fallback"      # clearly not the agents' plan
    assert "GROUND" in res.fallback_detail and not res.consensus
    assert not events(db, scenario, "AGENT_CONSENSUS") and len(events(db, scenario, "fallback_invoked")) == 1


def test_turn_three_cannot_modify_the_plan(scenario, pool, db):
    t = agree_on(pool[0]).table
    t[(3, "PILOT")] = lambda r: propose(pool[5], text="sneaky new plan")   # not allowed in turn 3 -> invalid reply, both tries
    client = MockClient(Script(t))
    res = run_negotiation(scenario, client, log_path=db)
    assert res.fallback_reason == FallbackReason.RESPONSE_ERROR and "turn 3" in res.fallback_detail
    assert res.frozen_proposal_id == pool[0].proposal_id and len(res.proposals) == 1  # no new proposal registered


# ------------------------------------------------------------------ malformed / timeout
def test_malformed_response_gets_exactly_one_repair_retry_logged_as_attempt(scenario, pool, db):
    base = agree_on(pool[0])
    state = {"n": 0}

    def handler(req):
        if (req.turn, req.role) == (1, "ATC"):
            state["n"] += 1
            if state["n"] == 1:
                return "I think we should land everyone ASAP!"          # not JSON
        return base(req)

    client = MockClient(handler)
    res = run_negotiation(scenario, client, log_path=db)
    assert res.consensus and len(res.messages) == 9            # the retry did NOT become another message
    assert client.calls == 10 and len(res.attempts) == 10
    first, second = res.attempts[0], res.attempts[1]
    assert (first.attempt, first.kind, first.outcome) == (1, "initial", "schema_error")
    assert (second.attempt, second.kind, second.outcome) == (2, "schema_repair", "ok")
    assert client.requests[1].repair_error and client.requests[1].attempt == 2
    assert "REPAIR_REQUIRED" in client.requests[1].prompt
    assert len(events(db, scenario, "model_attempt")) == 10 and len(events(db, scenario, "agent_message")) == 9


def test_second_malformed_response_is_unrecoverable_and_falls_back(scenario, db):
    client = MockClient(lambda r: {"type": "ACCEPT", "explanation": "x"})   # missing proposal_id/reason: schema error twice
    res = run_negotiation(scenario, client, log_path=db)
    assert client.calls == 2                                     # one try + ONE repair retry, then stop
    assert res.fallback_reason == FallbackReason.RESPONSE_ERROR and res.status == FinalStatus.FALLBACK_SELECTED
    assert [a.outcome for a in res.attempts] == ["schema_error", "schema_error"] and not res.messages
    assert not res.consensus


def test_unknown_proposal_reference_is_a_response_error(scenario, db):
    client = MockClient(lambda r: accept("prop-doesnotexist"))
    res = run_negotiation(scenario, client, log_path=db)
    assert res.fallback_reason == FallbackReason.RESPONSE_ERROR and "not a known proposal" in res.fallback_detail


def test_api_timeout_triggers_fallback_without_retry(scenario, pool, db):
    base = agree_on(pool[0])
    client = MockClient(lambda r: ModelTimeout("read timed out") if (r.turn, r.role) == (1, "PILOT") else base(r))
    res = run_negotiation(scenario, client, log_path=db)
    assert res.fallback_reason == FallbackReason.AGENT_TIMEOUT and res.status == FinalStatus.FALLBACK_SELECTED
    assert client.calls == 2 and [a.outcome for a in res.attempts] == ["ok", "timeout"]   # no retry after a timeout
    assert len(res.messages) == 1 and not res.consensus
    assert events(db, scenario, "model_attempt")[0]["payload"]  # attempt rows exist separately from messages


def test_api_error_triggers_fallback(scenario, db):
    res = run_negotiation(scenario, MockClient(lambda r: ModelAPIError("HTTP 529: overloaded")), log_path=db)
    assert res.fallback_reason == FallbackReason.API_ERROR and res.attempts[0].outcome == "api_error"


def test_total_deadline_stops_the_session(scenario, pool, db):
    class Clock:
        t = 0.0
        def __call__(self): return self.t
    clk = Clock()
    base = agree_on(pool[0])

    def handler(req):
        clk.t += 250                 # each call "takes" 250 s
        return base(req)

    res = run_negotiation(scenario, MockClient(handler), log_path=db, clock=clk,
                          config=NegotiationConfig(call_timeout_s=1000, deadline_s=500))
    assert res.fallback_reason == FallbackReason.DEADLINE_EXCEEDED and len(res.messages) == 2
    assert res.status == FinalStatus.FALLBACK_SELECTED and not res.consensus


def test_per_call_timeout_is_passed_to_the_client_and_bounded_by_the_deadline(scenario, pool, db):
    client = MockClient(agree_on(pool[0]))
    run_negotiation(scenario, client, log_path=db, config=NegotiationConfig(call_timeout_s=33, deadline_s=600))
    assert {r.timeout_s for r in client.requests} <= {33}


# ------------------------------------------------------------------ outdated / version
def test_acceptance_of_an_outdated_proposal_does_not_count(scenario, pool, db):
    A, B = pool[0], pool[1]
    t = {(1, "ATC"): propose(A), (1, "PILOT"): counter(B, A.proposal_id), (1, "GROUND"): accept(B.proposal_id),
         (2, "ATC"): accept(B.proposal_id), (2, "PILOT"): accept(B.proposal_id), (2, "GROUND"): accept(B.proposal_id),
         (3, "ATC"): lambda r: accept(frozen_id(r)), (3, "PILOT"): lambda r: accept(frozen_id(r)),
         (3, "GROUND"): accept(A.proposal_id)}                     # answers the superseded proposal
    res = run_negotiation(scenario, MockClient(Script(t)), log_path=db)
    assert res.frozen_proposal_id == B.proposal_id
    g = res.messages[-1]
    assert g.type == MessageType.ACCEPT and g.proposal_id == A.proposal_id and g.outdated
    assert res.status == FinalStatus.FALLBACK_SELECTED and res.fallback_reason == FallbackReason.NO_UNANIMOUS_AGREEMENT
    assert "OUTDATED" in res.fallback_detail and not res.consensus


def test_plan_for_another_scenario_version_is_rejected(scenario, pool, db):
    stale = pool[0].plan.model_copy(update={"scenario_version": "oldversion00"})
    bad = {"type": "PROPOSE", "reason_codes": ["ACCEPTABLE"], "explanation": "x", "plan": stale.model_dump(mode="json")}
    client = MockClient(lambda r: bad)
    res = run_negotiation(scenario, client, log_path=db)
    assert res.fallback_reason == FallbackReason.RESPONSE_ERROR and "SCENARIO_VERSION_MISMATCH" in res.fallback_detail
    assert client.calls == 2 and not res.proposals           # never registered


def test_scenario_change_mid_session_invalidates_it(scenario, pool, db):
    current = {"v": scenario.scenario_version}
    base = agree_on(pool[0])

    def handler(req):
        out = base(req)
        if (req.turn, req.role) == (1, "PILOT"):
            current["v"] = "changed-under-us"            # the active scenario changes during the session
        return out

    client = MockClient(handler)
    res = run_negotiation(scenario, client, log_path=db, scenario_guard=lambda: current["v"])
    assert res.status == FinalStatus.SESSION_INVALIDATED and "changed" in res.invalidation_detail
    assert res.fallback is None and not res.consensus
    assert client.calls == 2 and len(res.messages) == 1      # nothing from the new version was combined; no later calls
    assert not events(db, scenario, "AGENT_CONSENSUS")


# ------------------------------------------------------------------ bounds
def test_never_more_than_nine_messages_even_if_agents_keep_proposing(scenario, pool, db):
    i = {"n": 0}

    def handler(req):
        i["n"] += 1
        if req.turn == 3:
            return accept(frozen_id(req))
        return propose(pool[i["n"]])

    res = run_negotiation(scenario, MockClient(handler), log_path=db)
    assert len(res.messages) <= 9 and max(m.turn for m in res.messages) <= 3


def test_events_are_idempotent_per_session(scenario, pool, db):
    co = Coordinator(scenario, MockClient(agree_on(pool[0])), log_path=db, session_id="neg-fixed")
    co.run()
    n = len(events(db, scenario))
    Coordinator(scenario, MockClient(agree_on(pool[0])), log_path=db, session_id="neg-fixed").run()
    assert len(events(db, scenario)) == n                    # same session id: nothing duplicated
    Coordinator(scenario, MockClient(agree_on(pool[0])), log_path=db, session_id="neg-other").run()
    assert len(events(db, scenario)) == 2 * n                # a new session id is a new, separate run


# ------------------------------------------------------------------ infeasible
def test_deliberately_infeasible_scenario_is_reported_honestly(scenario, db):
    sc = scenario.model_copy(deep=True)
    sc.simulated.flights[1].wake_category = W.HEAVY
    sc.simulated.flights[1].aircraft_class = AC.WIDEBODY
    for g in sc.simulated.gates:
        g.max_aircraft_class = AC.NARROWBODY            # no widebody gate anywhere
    sc = rehash_scenario(sc)
    assert candidate_pool(sc) == []
    res = run_negotiation(sc, ScriptedClient(), log_path=db)
    assert res.status == FinalStatus.NO_FEASIBLE_PLAN and not res.consensus
    assert res.fallback_reason == FallbackReason.NO_VALID_CANDIDATE
    assert res.fallback.outcome == Outcome.NO_FEASIBLE_PLAN and res.fallback.search_status == SearchStatus.EXHAUSTED
    assert res.final_plan is None and res.frozen_proposal_id is None
    assert all(not p.validation.passed for p in res.proposals) and res.proposals
    fin = events(db, sc, "negotiation_finished")[0]["payload"]
    assert '"NO_FEASIBLE_PLAN"' in fin and '"EXHAUSTED"' in fin and not events(db, sc, "AGENT_CONSENSUS")


# ------------------------------------------------------------------ scripted mode
def test_scripted_mode_reaches_consensus_on_the_default_scenario(scenario, db):
    client = ScriptedClient()
    res = run_negotiation(scenario, client, log_path=db)
    assert res.mode == "scripted" and res.model_name == "scripted-policy"
    assert res.status == FinalStatus.AGENT_CONSENSUS and len(res.messages) == 9 and client.calls == 9
    assert any(m.type == MessageType.COUNTER_PROPOSE for m in res.messages)       # real negotiation, not a rubber stamp
    assert any(d.changed_preference and d.observed_delta for d in res.decisions)  # measurable trade-off recorded
    assert all(e["payload"].count('"mode": "scripted"') for e in events(db, scenario, "model_attempt"))


@pytest.mark.parametrize("seed", [11, 12])
def test_scripted_mode_on_real_flights_always_ends_in_a_valid_labelled_state(rules, flights_df, seed, tmp_path):
    from airport_ops import data
    from airport_ops.scenario import build_scenario
    from airport_ops.validate import validate_plan
    for _, row in flights_df.sample(4, random_state=seed).iterrows():
        sc = build_scenario(data.to_historical(row), rules)
        res = run_negotiation(sc, ScriptedClient(), log_path=tmp_path / "e.db")
        assert res.status in (FinalStatus.AGENT_CONSENSUS, FinalStatus.FALLBACK_SELECTED), res.status
        assert res.final_plan is not None and validate_plan(sc, res.final_plan).passed
        assert res.consensus == (res.fallback is None) and len(res.messages) <= 9


# ------------------------------------------------------------------ schemas & conditions
def test_message_schema_rules():
    ok = {"type": "ACCEPT", "proposal_id": "p", "explanation": "x"}
    assert AgentOutput.model_validate(ok)
    for bad in ({**ok, "plan_spec": {"landing_order": [], "runways": {}, "gates": {}, "ground_crews": {}}},   # ACCEPT with plan
                {"type": "REJECT", "proposal_id": "p", "explanation": "x"},                                   # REJECT w/o reason
                {"type": "COUNTER_PROPOSE", "explanation": "x", "plan_spec": {"landing_order": [], "runways": {}, "gates": {}, "ground_crews": {}}},
                {"type": "PROPOSE", "explanation": "x"},                                                       # no plan
                {**ok, "conditions": [{"field": "landing_delay_min", "op": "<=", "value": 3}]},                # needs flight_id
                {**ok, "conditions": [{"field": "i_waive_wake_separation", "op": "==", "value": 1}]},         # undefined field
                {**ok, "free_text_promise": "I will make it up later"}):                                       # unknown key
        with pytest.raises(Exception):
            AgentOutput.model_validate(bad)


def test_conditions_are_machine_checked_and_unresolved_when_flight_missing(scenario, pool):
    m = pool[0].metrics
    sel = scenario.simulated.flights[0].flight_id
    d = m["flights"][sel]["landing_delay_min"]
    C = Condition
    assert evaluate_condition(C(field=ConditionField.LANDING_DELAY_MIN, op=ConditionOp.LE, value=d + 0.01, flight_id=sel), m).satisfied is True
    assert evaluate_condition(C(field=ConditionField.LANDING_DELAY_MIN, op=ConditionOp.LT, value=d - 1, flight_id=sel), m).satisfied is False
    assert evaluate_condition(C(field=ConditionField.RUNWAY_ID, op=ConditionOp.EQ, value="RWY-9", flight_id=sel), m).satisfied is False
    assert evaluate_condition(C(field=ConditionField.TOTAL_COST, op=ConditionOp.GE, value=0), m).satisfied is True
    assert evaluate_condition(C(field=ConditionField.GATE_ID, op=ConditionOp.EQ, value="GATE-1", flight_id="NOPE"), m).satisfied is None


def test_agents_have_separate_prompts_and_domain_views(scenario, pool):
    from airport_ops import agents
    from airport_ops.negotiation_types import AgentContext
    from airport_ops.messages import SPEAKING_ORDER
    mk = lambda r: AgentContext(role=r, turn=1, seq=1, session_id="s", scenario=scenario, historical=scenario.historical,
                                messages=[], proposals=[], standing={x: [] for x in SPEAKING_ORDER},
                                allowed_types=list(MessageType))
    sys_ = {r: agents.system_prompt(r) for r in SPEAKING_ORDER}
    usr = {r: json.loads(agents.user_prompt(mk(r))) for r in SPEAKING_ORDER}
    assert len(set(sys_.values())) == 3
    for r in SPEAKING_ORDER:     # everyone gets the same version + the historical disruption record
        assert usr[r]["scenario_version"] == scenario.scenario_version
        assert usr[r]["historical_disruption_record"]["weather_delay_min"] == scenario.historical.weather_delay_min
    assert "wake_rules_ordered" in json.dumps(usr[AgentRole.ATC]["your_domain"]) and "fuel_at_nominal_eta" not in json.dumps(usr[AgentRole.ATC])
    assert "fuel_at_nominal_eta" in json.dumps(usr[AgentRole.PILOT]["your_domain"]) and "ground_crews" not in usr[AgentRole.PILOT]["your_domain"]
    assert "ground_crews" in usr[AgentRole.GROUND]["your_domain"] and "wake_rules_ordered" not in json.dumps(usr[AgentRole.GROUND])
    assert "ONLY the selected" in sys_[AgentRole.PILOT] or "selected historical flight only" in sys_[AgentRole.PILOT]
