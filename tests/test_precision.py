"""Precision review. Preference conditions use ONE canonical, per-field quantisation shared with what agents are shown; hard
constraints keep exact values and their own tolerances, so display rounding can never make an unsafe plan valid."""
import re
from pathlib import Path

import pytest

from conftest import mk
from mocks import MockClient, Script, accept, agree_on, frozen_id, propose
from airport_ops import agents
from airport_ops.agents.base import proposals_view
from airport_ops.canonical import DECIMALS, canonical_metrics, policy_text, quantise
from airport_ops.fuel import fuel_account
from airport_ops.messages import (FLIGHT_ID_FIELDS, AgentRole, Condition, ConditionField, ConditionOp, MessageType, ReasonCode,
                                  TradeoffMetric, SPEAKING_ORDER)
from airport_ops.metrics import evaluate_condition, plan_metrics
from airport_ops.negotiation import run_negotiation
from airport_ops.negotiation_types import AgentContext, FallbackReason
from airport_ops.validate import validate_plan
from airport_ops.workbook import WakeCategory as W

ROOT = Path(__file__).resolve().parent.parent
C = Condition


def plan_cond(field, op, v, flight=None):
    return C(field=ConditionField(field), op=op, value=v, flight_id=flight)


# ------------------------------------------------------------------ the quantisation table is per-field, documented, complete
def test_quantisation_table_is_per_field_not_blanket():
    numeric = {f.value for f in ConditionField if f not in FLIGHT_ID_FIELDS}
    assert set(DECIMALS) == numeric                                  # every numeric condition field has an explicit rule
    assert {DECIMALS[k] for k in ("weighted_landing_delay", "landing_delay_min", "total_cost")} == {2}
    assert DECIMALS["gate_changes"] == DECIMALS["runway_changes"] == 0
    assert DECIMALS["reserve_margin_kg"] == DECIMALS["contingency_margin_kg"] == 6      # fuel: effectively exact
    assert len(set(DECIMALS.values())) == 3                          # NOT one blanket precision
    assert quantise("weighted_landing_delay", 10.715) == 10.72 and quantise("weighted_landing_delay", 10.7149) == 10.71   # half-up
    assert quantise("reserve_margin_kg", -0.0004) == -0.0004         # fuel deficit survives quantisation
    assert quantise("unknown_field", 1.23456789) == 1.23456789
    for token in ("2 decimals", "6 decimals", "integers"):
        assert token in policy_text()


def test_canonical_metrics_keeps_structure_and_identifiers(scenario):
    from airport_ops.tools import candidate_pool
    m = candidate_pool(scenario)[0].metrics
    c = canonical_metrics(m)
    assert set(c) == {"plan", "flights"} and set(c["flights"]) == set(m["flights"])
    row, craw = c["flights"][scenario.simulated.flights[0].flight_id], m["flights"][scenario.simulated.flights[0].flight_id]
    assert row["gate_id"] == craw["gate_id"] and row["runway_id"] == craw["runway_id"]      # identifiers untouched
    assert row["landing_delay_min"] == quantise("landing_delay_min", craw["landing_delay_min"])


# ------------------------------------------------------------------ ONE representation: shown value == evaluated value
def test_every_value_agents_are_shown_round_trips_through_condition_evaluation(scenario):
    """For every numeric field of a proposal: a condition bounded by exactly the number the agent SAW is satisfied for <=, >=, ==
    and not satisfied for a strictly tighter bound. This is the live-run failure (<= 10.71 vs exact 10.7143) generalised."""
    from airport_ops.tools import candidate_pool
    pool = candidate_pool(scenario)
    sel = scenario.simulated.flights[0].flight_id
    for cand in (pool[0], pool[len(pool) // 2]):
        m = cand.metrics
        shown = canonical_metrics(m)
        for field, v in shown["plan"].items():
            step = 10 ** -DECIMALS[field]
            for op, ok in ((ConditionOp.LE, True), (ConditionOp.GE, True), (ConditionOp.EQ, True)):
                assert evaluate_condition(plan_cond(field, op, v), m).satisfied is ok, (field, op, v)
            assert evaluate_condition(plan_cond(field, ConditionOp.LT, v), m).satisfied is False
            assert evaluate_condition(plan_cond(field, ConditionOp.LE, v - 2 * step), m).satisfied is False
        for field, v in shown["flights"][sel].items():
            if isinstance(v, str) or field == "priority":
                continue
            assert evaluate_condition(plan_cond(field, ConditionOp.LE, v, sel), m).satisfied is True, field
            assert evaluate_condition(plan_cond(field, ConditionOp.GE, v, sel), m).satisfied is True, field


def test_prompt_values_come_from_the_canonical_representation(scenario):
    from airport_ops.tools import candidate_pool, option_menu
    pool = candidate_pool(scenario)
    ctx = AgentContext(role=AgentRole.PILOT, turn=2, seq=5, session_id="s", scenario=scenario, historical=scenario.historical,
                       messages=[], proposals=[], standing={r: [] for r in SPEAKING_ORDER}, allowed_types=list(MessageType))
    menu = option_menu(AgentRole.ATC, scenario, 3)
    for opt in menu:
        cand = next(c for c in pool if c.proposal_id == opt["proposal_id_if_proposed"])
        assert opt["consequences"]["weighted_landing_delay"] == quantise("weighted_landing_delay", cand.metrics["plan"]["weighted_landing_delay"])
    fa = fuel_account(scenario, scenario.simulated.flights[0], scenario.simulated.flights[0].earliest_landing_min)
    shown = agents.pilot.domain_view(ctx)["selected_flight"]["fuel_at_nominal_eta"]
    assert shown["contingency_margin_kg"] == quantise("contingency_margin_kg", fa.contingency_margin_kg)


# ------------------------------------------------------------------ boundary 1: weighted delay (a preference metric)
def test_weighted_delay_boundary_uses_the_documented_quantisation():
    m = {"flights": {}, "plan": {"weighted_landing_delay": 10.714285714}}
    wl = lambda op, v: evaluate_condition(plan_cond("weighted_landing_delay", op, v), m)
    assert wl(ConditionOp.LE, 10.71).satisfied is True                 # the live case: bound copied from a 2-decimal display
    assert wl(ConditionOp.LE, 10.70).satisfied is False                # a genuinely tighter bound still fails
    assert wl(ConditionOp.GE, 10.72).satisfied is False and wl(ConditionOp.GE, 10.71).satisfied is True
    assert wl(ConditionOp.LT, 10.71).satisfied is False
    r = wl(ConditionOp.LE, 10.71)
    assert (r.observed, round(r.observed_exact, 6)) == (10.71, 10.714286)     # canonical AND exact value both reported


# ------------------------------------------------------------------ boundary 2: fuel reserve (hard) - never rescued by rounding
def _late_plan(sc, landing):
    """Consistent plan where DL1205 lands at `landing` (everything downstream shifted so ONLY fuel can be violated)."""
    f1, f2, f3 = (f.flight_id for f in sc.simulated.flights)
    p3 = mk(sc, f3, "RWY-1", 22.0, "GATE-3", "GC-2", 23.5)
    p2 = mk(sc, f2, "RWY-2", 20.0, "GATE-2", "GC-1", p3.gate_in_min)
    a = sc.simulated.assumptions
    p1 = mk(sc, f1, "RWY-1", landing, "GATE-1", "GC-2", landing + a.runway_occupancy_min,
            service_start=None)
    p1 = p1.model_copy(update={"service_start_min": max(p1.gate_in_min + a.deboard_min, p3.service_end_min + a.ground_crew_transfer_min)})
    p1 = p1.model_copy(update={"service_end_min": max(p1.service_start_min + 30.0, p1.gate_in_min + sc.workbook.airport.min_turnaround_min)})
    from airport_ops.plan import make_plan
    return make_plan(sc, "test", [p1, p2, p3])


def test_fuel_reserve_deficit_of_a_fraction_of_a_gram_is_never_hidden_by_rounding(good):
    sc, _ = good
    f1 = sc.simulated.flights[0]
    acc = fuel_account(sc, f1, f1.earliest_landing_min)
    t_r = f1.earliest_landing_min + acc.max_hold_min_before_reserve         # exact moment the final reserve is touched
    at = lambda t: validate_plan(sc, _late_plan(sc, t))
    ok = at(t_r)
    assert not [r for r in ok.failures if r.constraint_id == "FUEL-RESERVE"]            # exactly at the boundary: reserve intact
    short = at(t_r + 1e-5)                                                              # 0.0004 kg (0.4 g) short of the reserve
    fa = fuel_account(sc, f1, t_r + 1e-5)
    assert -0.001 < fa.reserve_margin_kg < 0
    assert round(fa.reserve_margin_kg, 2) == 0 and round(fa.reserve_margin_kg, 3) == -0.0   # a 2- or 3-decimal display would read "0"
    assert {r.constraint_id for r in short.failures} == {"FUEL-RESERVE", "FUEL-CONTINGENCY"}   # ...but the hard check rejects it
    # the preference condition sees the same deficit (6 decimals), so a rounded display cannot be used to accept it either
    m = plan_metrics(sc, _late_plan(sc, t_r + 1e-5))
    assert evaluate_condition(plan_cond("reserve_margin_kg", ConditionOp.GE, 0.0, f1.flight_id), m).satisfied is False


def test_an_agreed_plan_that_misses_the_fuel_reserve_by_a_gram_cannot_reach_consensus(good, tmp_path):
    sc, _ = good
    f1 = sc.simulated.flights[0]
    t_r = f1.earliest_landing_min + fuel_account(sc, f1, f1.earliest_landing_min).max_hold_min_before_reserve
    bad = _late_plan(sc, t_r + 1e-5)
    reply = {"type": "PROPOSE", "reason_codes": ["ACCEPTABLE"], "explanation": "x", "plan": bad.model_dump(mode="json")}

    def handler(req):
        if (req.turn, req.role) == (1, "ATC"):
            return reply
        if req.turn == 3:
            return accept(frozen_id(req))
        return accept(req.context.proposals[0].proposal_id)

    res = run_negotiation(sc, MockClient(handler), log_path=tmp_path / "e.db")
    assert not res.proposals[0].validation.passed and res.frozen_proposal_id is None
    assert res.fallback_reason == FallbackReason.NO_VALID_CANDIDATE and not res.consensus
    assert "FUEL-RESERVE" in res.fallback_detail


# ------------------------------------------------------------------ boundary 3: wake separation (hard)
def test_wake_separation_boundary_is_exact_and_reported_precisely(good):
    sc, plan = good
    sc.simulated.flights[0].wake_category = W.HEAVY                 # Heavy -> Large: 5 nm @ 140 kt = 128.5714 s (> runway occupancy)
    f1, f2, f3 = (f.flight_id for f in sc.simulated.flights)
    req_min = 5 / sc.simulated.assumptions.approach_speed_kt * 60
    p1 = [p for p in plan.flights if p.flight_id == f1][0]

    def with_gap(gap_s):
        p = plan.model_copy(deep=True)
        [x for x in p.flights if x.flight_id == f3][0].landing_min = p1.landing_min + gap_s / 60
        return validate_plan(sc, p)

    wake = lambda rep: [r for r in rep.results if r.constraint_id == "WAKE-SEP"][0]
    req_s = req_min * 60
    assert wake(with_gap(req_s)).passed                              # exactly the required spacing
    short = wake(with_gap(req_s - 0.01))                             # 10 ms short
    assert not short.passed and f"gap {req_s - 0.01:.3f}s" in short.explanation and f"required {req_s:.3f}s" in short.explanation
    assert not wake(with_gap(req_s - 0.0002)).passed                 # even 0.2 ms short is rejected (tolerance 6e-5 s)
    assert round(req_s - 0.01) == round(req_s)                       # a whole-second display would NOT show the shortfall


def test_hard_constraint_code_never_uses_display_quantisation():
    for mod in ("validate.py", "fuel.py", "planner.py", "capacity.py", "plan.py"):
        src = (ROOT / "airport_ops" / mod).read_text()
        assert "canonical" not in src and "quantise" not in src, mod


# ------------------------------------------------------------------ message repair + acceptance integrity
def test_prompts_expose_exactly_the_supported_codes_fields_ops_and_types():
    for role in AgentRole:
        sp = agents.system_prompt(role)
        for enum in (ReasonCode, ConditionField, TradeoffMetric, MessageType, ConditionOp):
            for member in enum:
                assert member.value in sp, (role, enum.__name__, member.value)
        m = re.search(r"Allowed fields: (\{.*?\}); ops:", sp, re.S)
        import json
        listed = {x for v in json.loads(m.group(1)).values() for x in v}
        assert listed == {f.value for f in ConditionField}            # no missing, no invented field names
        rc = re.search(r"reason_codes: list of codes from (\[.*?\])", sp)
        assert set(eval(rc.group(1))) == {r.value for r in ReasonCode}
        assert "`because` MUST be exactly ONE reason code" in sp and "SAME per-field precision" in sp


def test_repair_retry_names_the_offending_value_once_and_is_not_a_message(scenario, tmp_path):
    from airport_ops.tools import candidate_pool
    cand = candidate_pool(scenario)[0]
    base = agree_on(cand)
    seen = {"n": 0}
    bad = {**propose(cand), "tradeoff": {"metric": "total_cost", "before": 1, "after": 0, "because": "WEIRD_REASON"}}

    def handler(req):
        if (req.turn, req.role) == (1, "ATC"):
            seen["n"] += 1
            return bad if seen["n"] == 1 else propose(cand)
        return base(req)

    client = MockClient(handler)
    res = run_negotiation(scenario, client, log_path=tmp_path / "e.db")
    assert res.consensus and len(res.messages) == 9 and client.calls == 10
    first = res.attempts[0]
    assert first.outcome == "schema_error" and "tradeoff.because" in first.error and "'WEIRD_REASON'" in first.error
    assert "REPAIR_REQUIRED" in client.requests[1].prompt and "WEIRD_REASON" in client.requests[1].prompt
    assert res.attempts[1].kind == "schema_repair" and res.attempts[1].outcome == "ok"
    assert sum(a.attempt == 2 for a in res.attempts) == 1             # exactly one repair attempt in the whole session


@pytest.mark.parametrize("role", ["ATC", "PILOT", "GROUND"])
def test_an_unresolved_acceptance_never_counts_whoever_makes_it(scenario, tmp_path, role):
    from airport_ops.tools import candidate_pool
    sel = scenario.simulated.flights[0].flight_id
    impossible = {"field": "landing_delay_min", "op": "<=", "value": -1, "flight_id": sel}
    t = agree_on(candidate_pool(scenario)[0]).table
    t[(3, role)] = lambda r: accept(frozen_id(r), conditions=[impossible])
    res = run_negotiation(scenario, MockClient(Script(t)), log_path=tmp_path / "e.db")
    msg = [m for m in res.messages if m.turn == 3 and m.sender.value == role][0]
    assert msg.type == MessageType.ACCEPT and msg.conditions_resolved is False
    assert not res.consensus and res.fallback_reason == FallbackReason.NO_UNANIMOUS_AGREEMENT
    assert [v for v in res.votes if v.role.value == role][0].counted_as_accept is False
