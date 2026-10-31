"""Streamlit UI: python3 -m streamlit run app.py"""
import os

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from airport_ops import agents, data, eventlog, llm
from airport_ops.fallback import Outcome, SearchStatus, run_fallback
from airport_ops.messages import SPEAKING_ORDER, AgentRole
from airport_ops.negotiation import FREEZE_RULE, Coordinator, NegotiationConfig, new_session_id
from airport_ops.negotiation_types import AgentContext, FinalStatus
from airport_ops.plan import RecoveryPlan
from airport_ops.planner import Planner
from airport_ops.report import assignment_rows, fallback_payload, validation_payload
from airport_ops.scenario import ScenarioError, build_scenario
from airport_ops.schemas import SimAssumptions
from airport_ops.validate import CONSTRAINTS, validate_plan
from airport_ops.workbook import load_workbook_rules

llm.load_env_file()  # names only; shell variables win over the .env file
ENV_LOADED = list(llm.LOADED_FROM_FILE)  # reruns find the keys already in the environment; this remembers where they came from
st.set_page_config(page_title="Airport Disruption Negotiation", layout="wide")
st.title("Airport Disruption Negotiation – Scenario, Validation & Fallback")


@st.cache_data(show_spinner="Loading flights…")
def get_flights() -> pd.DataFrame:
    return data.load_flights()


@st.cache_data(show_spinner="Reading workbook…")
def get_workbook():
    return load_workbook_rules()


df = get_flights()
wb = get_workbook()

# ---------- workbook status (errors are reported, never swallowed) ----------
if wb.issues:
    with st.expander(f"Workbook: {len(wb.errors)} error(s), {len(wb.issues) - len(wb.errors)} warning(s)", expanded=bool(wb.errors)):
        for i in wb.issues:
            (st.error if i.severity.value == "error" else st.warning)(str(i))
if not wb.usable:
    st.error("The operational workbook could not be loaded, so scenarios cannot be built. Fix the errors above.")
    st.stop()
rules = wb.rules

# ---------- flight selector ----------
with st.sidebar:
    st.header("Flight selector")
    st.caption("Filter: WEATHER_DELAY > 15 OR LATE_AIRCRAFT_DELAY > 20")
    st.write(f"{len(df):,} matching flights")
    dates = sorted(df.FLIGHT_DATE.unique())
    default_row = df.loc[data.default_index(df)]
    d = st.selectbox("Date", dates, index=dates.index(default_row.FLIGHT_DATE))
    day = df[df.FLIGHT_DATE == d]
    labels = {i: f"{r.FLIGHT_LABEL}  {r.ORIGIN_AIRPORT}→{r.DESTINATION_AIRPORT}  "
                 f"(wx {r.WEATHER_DELAY}, late-ac {r.LATE_AIRCRAFT_DELAY})"
              for i, r in day.iterrows()}
    ids = list(labels)
    idx = ids.index(default_row.name) if default_row.name in ids else 0
    sel = st.selectbox("Flight", ids, index=idx, format_func=labels.get)

h = data.to_historical(df.loc[sel])
try:
    sc = build_scenario(h, rules)
except ScenarioError as e:
    st.error(f"Cannot build a scenario for {h.label}: {e}")
    st.stop()
# idempotent: a rerun (or another session) with the same scenario version writes nothing
eventlog.log_event(sc.scenario_id, "scenario_created",
                   {"flight": h.label, "seed": sc.simulated.seed, "scenario_version": sc.scenario_version,
                    "airport": sc.workbook.airport.iata_code},
                   dedupe_key=f"scenario_created:{sc.scenario_version}")

st.caption(f"Scenario `{sc.scenario_id}` · version `{sc.scenario_version}` · seed {sc.simulated.seed} "
           f"(deterministic per flight) · scenario airport = destination **{sc.workbook.airport.iata_code}**")

# ---------- shared renderer for a plan + its validation ----------
def show_report(plan: RecoveryPlan, report, only_fail: bool, key: str):
    nfail = len(report.failures)
    if report.passed:
        st.success(f"VALID – all {len(report.results)} hard-constraint checks pass · proposal `{plan.proposal_id}` "
                   f"· scenario `{plan.scenario_version}` · source: {plan.source}")
    else:
        st.error(f"REJECTED – {nfail} of {len(report.results)} checks failed · proposal `{plan.proposal_id}` "
                 f"· scenario `{plan.scenario_version}` · source: {plan.source}")
    st.markdown("**Per-flight assignments**")
    st.dataframe(pd.DataFrame(assignment_rows(plan, sc)), hide_index=True)
    st.markdown("**Constraint results**")
    rows = [{"result": "✅ pass" if r.passed else "❌ FAIL", "constraint": r.constraint_id,
             "flight": r.flight_id or "—", "resource": r.resource_id or "—", "explanation": r.explanation,
             "basis": r.basis} for r in sorted(report.results, key=lambda r: (r.passed, r.flight_id or "~", r.constraint_id))]
    if only_fail:
        rows = [r for r in rows if r["result"].startswith("❌")]
    flights = ["(all)"] + [f.flight_id for f in sc.simulated.flights]
    pick = st.selectbox("Filter by flight", flights, key=f"flt-{key}")
    if pick != "(all)":
        rows = [r for r in rows if r["flight"] in (pick, "—")]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


tab_h, tab_w, tab_s, tab_n, tab_p = st.tabs(["📊 Historical (dataset)", "📘 Workbook limits (synthetic)",
                                         "🧪 Simulated conditions", "🤝 Negotiation (3 agents)", "🛠 Plans & validation"])
unk = lambda v: v if v else "Unknown"

# ---------- historical ----------
with tab_h:
    st.write(f"**{h.label}** · airline code **{h.airline_code}**"
             + (f" ({h.airline_name})" if h.airline_name else "") + f" · {h.flight_date}")
    st.write(f"**From:** {h.origin_code} – {unk(h.origin_name)}")
    st.write(f"**To:** {h.destination_code} – {unk(h.destination_name)}")
    st.write(f"Tail: {h.tail_number or 'Unknown'} · Distance: {h.distance_mi or '?'} mi")
    st.table(pd.DataFrame({
        "Scheduled": [h.scheduled_departure, h.scheduled_arrival],
        "Actual": [h.actual_departure or "—", h.actual_arrival or "—"],
        "Delay (min)": [h.departure_delay_min, h.arrival_delay_min]}, index=["Departure", "Arrival"]))
    st.bar_chart(pd.Series({
        "Weather": h.weather_delay_min, "Late aircraft": h.late_aircraft_delay_min,
        "Air system": h.air_system_delay_min, "Airline": h.airline_delay_min,
        "Security": h.security_delay_min}, name="minutes"))
    st.caption("Historical delay is context only: it decides whether a weather closure / late ground crew is simulated, "
               "not how long the aircraft holds.")
    if h.cancelled or h.diverted:
        st.warning(f"Cancelled: {h.cancelled} · Diverted: {h.diverted}")

# ---------- workbook ----------
with tab_w:
    ap, sub = sc.workbook.airport, sc.workbook.capacity_subset
    st.info("SYNTHETIC planning limits from the workbook (not real airport data).")
    st.markdown(f"**{ap.iata_code} – {ap.airport}** ({ap.size_class})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active runways (airport total)", ap.active_runways, help=f"Modelled in demo: {sub.modelled_runways}")
    c2.metric("Gates (airport total)", ap.gates, help=f"Modelled in demo: {sub.modelled_gates}")
    c3.metric("Airport-wide runway ops / hour (workbook)", ap.max_runway_ops_per_hour,
              help="Workbook max_runway_ops_per_hour, read as airport-wide movements per hour")
    c4.metric("Min turnaround (min)", ap.min_turnaround_min)
    st.table(pd.DataFrame([
        {"resource": "Runways", "airport total": str(sub.total_runways), "modelled subset": str(sub.modelled_runways)},
        {"resource": "Gates", "airport total": str(sub.total_gates), "modelled subset": str(sub.modelled_gates)},
        {"resource": "Spare gates at any time", "airport total": str(sub.spare_gates_at_any_time),
         "modelled subset": f"{min(sub.spare_gates_at_any_time, sub.modelled_gates)} of the modelled gates kept free"}]))
    st.caption(sub.note)
    from airport_ops.capacity import per_runway_limit
    _pr = per_runway_limit(sc)
    st.caption("Separate ASSUMPTION layered on the airport-wide limit: per-runway share mode "
               f"**{sc.simulated.assumptions.per_runway_share_mode}** → "
               + (f"{_pr} movements/hour per runway" if _pr is not None else "no per-runway check")
               + " (configure `per_runway_share_mode` / `per_runway_fixed_ops_per_hour` in `SimAssumptions`).")
    fcr = sc.workbook.fuel_crew
    st.markdown("**Wake separation** (ordered: leader → follower)")
    wk = [{"leader": r.leader.value, "follower": r.follower.value, "min_separation_nm": r.min_separation_nm,
           "min_time_s": r.min_time_s} for r in sc.workbook.wake.rules]
    wk.append({"leader": "any other pair", "follower": "", "min_separation_nm": sc.workbook.wake.default_separation_nm,
               "min_time_s": sc.workbook.wake.default_time_s})
    st.dataframe(pd.DataFrame(wk), hide_index=True)
    st.markdown("**Fuel & crew rules** (parsed)")
    st.dataframe(pd.DataFrame([{"rule": k, "value": v} for k, v in fcr.model_dump().items()]), hide_index=True)
    st.markdown("**Rules of play** (workbook)")
    st.dataframe(pd.DataFrame([{"topic": k, "detail": v} for k, v in rules.rules_of_play.topics.items()]), hide_index=True)

# ---------- simulated ----------
with tab_s:
    s = sc.simulated
    st.info("Everything here is invented for the prototype (seeded, reproducible) or a configurable assumption. "
            "No background traffic is added to occupy the rest of the airport.")
    st.markdown("**Simulation assumptions** (not in the workbook; stated, configurable in `SimAssumptions`)")
    st.dataframe(pd.DataFrame([{"assumption": k, "value": str(getattr(s.assumptions, k)),
                                "meaning": f.description or ""} for k, f in SimAssumptions.model_fields.items()]),
                 hide_index=True)
    st.markdown("**Flights** (selected historical flight + 2 simulated competitors; times = minutes from scenario start)")
    st.dataframe(pd.DataFrame([{
        "flight": f.flight_id, "origin": f.source, "wake": f.wake_category.value, "class": f.aircraft_class.value,
        "pax": f.passengers, "priority": f.priority, "ETA (t+min)": f.earliest_landing_min,
        "orig. runway": f.original_runway_id, "orig. gate": f.original_gate_id,
        "orig. ground crew": f.original_ground_crew_id, "flight crew": f.flight_crew_id,
        "service task (min)": f.service_task_min, "fuel on board (kg)": f.fuel.fuel_on_board_kg,
        "trip fuel (kg)": f.fuel.trip_fuel_remaining_kg, "spare fuel at ETA (min)": f.fuel.spare_fuel_min_at_eta}
        for f in s.flights]), hide_index=True)
    iv = lambda xs: "; ".join(f"t+{i.start_min:g}..{i.end_min:g} ({i.reason})" for i in xs) or "—"
    st.markdown("**Runways & closures**")
    st.dataframe(pd.DataFrame([{"runway": r.id, "closures": iv(r.closures)} for r in s.runways]), hide_index=True)
    st.markdown("**Taxi segments & blockages**")
    st.dataframe(pd.DataFrame([{"segment": x.id, "from": x.from_node, "to": x.to_node, "minutes": x.minutes,
                                "blockages": iv(x.blockages)} for x in s.taxi_segments]), hide_index=True)
    st.markdown("**Gates & resident occupancy**")
    st.dataframe(pd.DataFrame([{"gate": g.id, "terminal": g.terminal, "max class": g.max_aircraft_class.value,
                                "resident aircraft": iv(g.resident_occupancy)} for g in s.gates]), hide_index=True)
    st.markdown("**Crews**")
    fh = sc.workbook.fuel_crew
    st.dataframe(pd.DataFrame(
        [{"crew": c.id, "type": "flight", "report / shift start (t+min)": c.report_time_min,
          "limit (t+min)": c.report_time_min + fh.flight_crew_max_duty_h * 60, "available from": 0} for c in s.flight_crews] +
        [{"crew": c.id, "type": "ground", "report / shift start (t+min)": c.shift_start_min,
          "limit (t+min)": c.shift_start_min + fh.ground_shift_h * 60, "available from": c.available_from_min}
         for c in s.ground_crews]), hide_index=True)
    with st.expander("Scenario JSON"):
        st.json(sc.model_dump(mode="json"))


# ---------- plans ----------
runs = st.session_state.setdefault("runs", {})
run = runs.setdefault(sc.scenario_version, {})

# ---------- negotiation (3 agents) ----------
neg_runs = st.session_state.setdefault("neg_runs", {})      # session_id -> NegotiationResult (survives reruns)
neg_latest = st.session_state.setdefault("neg_latest", {})  # scenario_version -> session_id

with tab_n:
    live_cfg = llm.LiveConfig.from_env()
    st.caption(".env file: " + (f"{len(ENV_LOADED)} variable(s) read from it ({', '.join(ENV_LOADED)}); values are never shown"
                                if ENV_LOADED else "nothing read from it (missing, empty, or already set in the shell)")
               + ". Restart the app after editing it.")
    default_mode = 1 if os.environ.get("NEGOTIATION_MODE", "scripted").lower() == "live" else 0
    mode_label = st.radio("Mode", ["Scripted demo (no LLM calls)", "Live LLM (Claude API)"], index=default_mode,
                          horizontal=True, key="neg_mode")
    live = mode_label.startswith("Live")
    if live:
        st.info(f"**LIVE** mode · model `{live_cfg.model}` · per-call timeout {live_cfg.call_timeout_s:g}s · "
                f"total deadline {live_cfg.deadline_s:g}s · credentials: **{live_cfg.credential_mode}** "
                f"(" + ", ".join(f"{r}: {'set' if ok else 'missing'}" for r, ok in live_cfg.credentials_by_role.items()) + ")")
        if not live_cfg.credentials_present:
            st.error("Live mode needs one key per agent (`ANTHROPIC_API_KEY_ATC`, `_PILOT`, `_GROUND`) or one shared "
                     "`ANTHROPIC_API_KEY` in the environment"
                     + (" - per-agent keys are only partly set" if live_cfg.credential_mode == "incomplete" else "")
                     + ". A live run will NOT silently fall back to scripted mode.")
    else:
        st.warning("**SCRIPTED DEMO** – deterministic scripted policies, **no language model is called**. Same message "
                   "schemas, validator and coordinator as live mode.")
    st.caption("Scope: one Pilot agent represents the selected historical flight only; the two simulated competing flights are "
               "deterministic traffic (no Pilot agents yet).")
    if st.button("Start new negotiation", type="primary", disabled=live and not live_cfg.credentials_present,
                 help="Creates a new session id and runs ATC → Pilot → Ground for up to 3 turns (max 9 messages)."):
        try:
            client = llm.make_client("live" if live else "scripted")
        except llm.MissingCredentials as e:
            st.error(f"Cannot start live run: {e}")
        else:
            cfg = NegotiationConfig(call_timeout_s=live_cfg.call_timeout_s, deadline_s=live_cfg.deadline_s) if live else NegotiationConfig()
            sid = new_session_id()
            with st.spinner(f"Negotiating ({'live Claude calls' if live else 'scripted'})…"):
                neg_runs[sid] = Coordinator(sc, client, config=cfg, session_id=sid).run()
            neg_latest[sc.scenario_version] = sid

    stale = [r for r in neg_runs.values() if r.scenario_version != sc.scenario_version]
    res = neg_runs.get(neg_latest.get(sc.scenario_version, ""))
    if stale:
        st.caption(f"{len(stale)} earlier session(s) belong to other scenario versions and are not shown or combined.")
    if res is None:
        st.info("No negotiation has been run for this scenario version yet.")
    else:
        is_live = res.mode == "live"
        cols = st.columns(5)
        cols[0].metric("Mode", {"live": "LIVE LLM", "scripted": "SCRIPTED", "mock": "MOCK"}.get(res.mode, res.mode),
                       help=f"model: {res.model_name}")
        cols[1].metric("Scenario version", res.scenario_version)
        cols[2].metric("Turns completed", f"{res.turns_completed} / 3")
        cols[3].metric("Messages", f"{len(res.messages)} / 9")
        cols[4].metric("Session", res.session_id)
        if res.scenario_version != sc.scenario_version:
            st.error("This session belongs to a different scenario version and is invalid for the current scenario.")
        S = res.status
        if S == FinalStatus.AGENT_CONSENSUS:
            st.success(f"**AGENT_CONSENSUS** – all three agents accepted frozen proposal `{res.frozen_proposal_id}` "
                       f"(scenario `{res.scenario_version}`) and an independent re-validation passed "
                       f"({len(res.final_validation.results)} checks).")
        elif S == FinalStatus.FALLBACK_SELECTED:
            st.warning(f"**FALLBACK_SELECTED** – this is the deterministic fallback planner's plan, **NOT agent consensus**. "
                       f"Reason: `{res.fallback_reason.value}`")
        elif S == FinalStatus.NO_FEASIBLE_PLAN:
            st.error(f"**NO_FEASIBLE_PLAN** – agents did not reach consensus and the fallback found no valid plan "
                     f"(search **{res.fallback.search_status.value}**"
                     + (": candidate set exhausted" if res.fallback.search_status == SearchStatus.EXHAUSTED
                        else ": infeasibility NOT proven") + f"). Fallback trigger: `{res.fallback_reason.value}`")
        else:
            st.error(f"**SESSION_INVALIDATED** – {res.invalidation_detail}")
        if res.fallback_reason:
            st.markdown(f"**Why the fallback ran:** `{res.fallback_reason.value}` – {res.fallback_detail}")
            for r_ in res.fallback.reasons:
                st.write("• " + r_)

        # ---- three separate agent panels
        st.subheader("Agents")
        pcols = st.columns(3)
        blank = lambda role: AgentContext(role=role, turn=1, seq=1, session_id=res.session_id, scenario=sc, historical=h,
                                          messages=[], proposals=[], standing={r: [] for r in SPEAKING_ORDER}, allowed_types=[])
        for col, role in zip(pcols, SPEAKING_ORDER):
            with col:
                icon = {"ATC": "🛫", "PILOT": "👨‍✈️", "GROUND": "🧰"}[role.value]
                st.markdown(f"### {icon} {role.value}")
                st.caption({"ATC": "runways · separation · taxiways · capacity",
                            "PILOT": f"selected flight {h.label} only · fuel · ETA window · connection",
                            "GROUND": "gates · crews · turnaround (stands/equipment not modelled)"}[role.value])
                with st.expander("Domain view (what this agent sees)"):
                    st.json(agents.MODULES[role].domain_view(blank(role)))
                for m in [m for m in res.messages if m.sender == role]:
                    with st.container(border=True):
                        st.markdown(f"**Turn {m.turn} · {m.type.value}** → `{m.proposal_id}`"
                                    + (f" (counters `{m.referenced_proposal_id}`)" if m.referenced_proposal_id else ""))
                        st.write(m.explanation)
                        if m.reason_codes:
                            st.caption("reasons: " + ", ".join(r.value for r in m.reason_codes))
                        if m.conditions:
                            st.caption("conditions: " + "; ".join(
                                f"{c.field.value}{('['+c.flight_id+']') if c.flight_id else ''} {c.op.value} {c.value}" for c in m.conditions)
                                       + (f" → resolved: {m.conditions_resolved}" if m.conditions_resolved is not None else ""))
                        if m.outdated:
                            st.error("accepted an OUTDATED proposal (not the frozen one)")
                        d = next((d for d in res.decisions if d.seq == m.seq), None)
                        if d and (d.changed_preference or d.declared_tradeoff):
                            st.caption("position changed" if d.changed_preference else "position held")
                            if d.declared_tradeoff:
                                t_ = d.declared_tradeoff
                                st.caption(f"declared trade-off: {t_.metric.value} {t_.before:.1f} → {t_.after:.1f} ({t_.because.value})")
                            if d.observed_delta:
                                st.caption("measured (coordinator): " + ", ".join(f"{k} {v:+.1f}" for k, v in d.observed_delta.items()
                                                                                    if k in ("weighted_landing_delay", "ground_overtime_min", "total_cost")))

        # ---- timeline
        st.subheader("Message timeline")
        st.dataframe(pd.DataFrame([{
            "seq": m.seq, "turn": m.turn, "sender": m.sender.value, "type": m.type.value, "proposal": m.proposal_id,
            "refs": m.referenced_proposal_id or "—", "valid plan": res.proposal(m.proposal_id).validation.passed,
            "conditions resolved": "—" if m.conditions_resolved is None else str(m.conditions_resolved),
            "reasons": ", ".join(r.value for r in m.reason_codes), "explanation": m.explanation} for m in res.messages]),
            hide_index=True)

        # ---- candidate versions
        st.subheader("Candidate plan versions & validation")
        st.dataframe(pd.DataFrame([{
            "v": p.version_no, "proposal": p.proposal_id, "by": p.proposer.value, "turn": p.turn, "valid": p.validation.passed,
            "failed constraints": ", ".join(sorted({r.constraint_id for r in p.validation.failures})) or "—",
            "weighted delay": round(p.metrics["plan"]["weighted_landing_delay"], 1),
            "ground overtime": round(p.metrics["plan"]["ground_overtime_min"], 1),
            "cost": round(p.metrics["plan"]["total_cost"], 1), "origin": p.origin,
            "frozen": p.proposal_id == res.frozen_proposal_id} for p in res.proposals]), hide_index=True)
        if res.proposals:
            pick = st.selectbox("Inspect a candidate's validation", [p.proposal_id for p in res.proposals], key=f"pick-{res.session_id}")
            rec = res.proposal(pick)
            show_report(rec.plan, rec.validation, st.checkbox("Only failed checks", value=not rec.validation.passed,
                                                              key=f"of-{res.session_id}"), f"neg-{res.session_id}")

        # ---- frozen candidate
        st.subheader("Frozen final candidate")
        if res.frozen_proposal_id:
            fz = res.proposal(res.frozen_proposal_id)
            st.write(f"`{fz.proposal_id}` · version {fz.version_no} · digest `{fz.digest}` · validation "
                     f"{'PASS' if fz.validation.passed else 'FAIL'}")
            st.caption("Selection rule: " + FREEZE_RULE)
            st.dataframe(pd.DataFrame(res.freeze_ranking), hide_index=True)
            if res.votes:
                st.markdown("**Turn-3 votes on the identical frozen proposal**")
                st.dataframe(pd.DataFrame([{"agent": v.role.value, "type": v.type.value if v.type else "—",
                                            "proposal": v.proposal_id or "—", "counted": v.counted_as_accept, "note": v.note}
                                           for v in res.votes]), hide_index=True)
        else:
            st.info("No candidate was frozen (no proposal passed validation before turn 3, or the session ended earlier).")

        # ---- final plan
        if res.final_plan is not None:
            st.subheader("Final plan" + (" (agent consensus)" if res.consensus else " (deterministic fallback)"))
            show_report(res.final_plan, res.final_validation, False, f"final-{res.session_id}")

        # ---- model attempts (separate from messages)
        st.subheader("Model attempts (separate from messages)")
        if res.mode == "scripted":
            st.caption("Scripted demo: these rows are policy invocations, not language-model calls.")
        st.dataframe(pd.DataFrame([a_.model_dump(mode="json", exclude={"session_id"}) for a_ in res.attempts]), hide_index=True)

with tab_p:
    only_fail = st.checkbox("Show only failed checks", value=False)
    b1, b2 = st.columns(2)
    if b1.button("Generate & validate candidate plan", help="Naive plan: nominal ETAs, original runway/gate/crew, no conflict "
                                                           "resolution. Shows the validator rejecting a plan."):
        plan = Planner(sc).naive_plan()
        rep = validate_plan(sc, plan)
        run["candidate"] = (plan, rep)
        eventlog.log_event(sc.scenario_id, "plan_validated", validation_payload(rep, plan.source),
                           dedupe_key=f"plan_validated:{plan.proposal_id}:{sc.scenario_version}")
    if b2.button("Run fallback planner", help="Bounded deterministic search; hard constraints filter, fixed costs rank."):
        with st.spinner("Searching candidate set…"):
            res = run_fallback(sc)
        run["fallback"] = res
        eventlog.log_event(sc.scenario_id, "fallback_completed", fallback_payload(res, sc),
                           dedupe_key=f"fallback_completed:{sc.scenario_version}:{res.outcome.value}:{res.search_status.value}")

    if "candidate" in run:
        st.subheader("Candidate plan (naive baseline)")
        show_report(*run["candidate"][:2], only_fail, "cand")
    if "fallback" in run:
        res = run["fallback"]
        st.subheader("Fallback planner")
        if res.outcome == Outcome.FALLBACK_SELECTED:
            st.success(f"{res.outcome.value} · search {res.search_status.value} · cost {res.cost:g}")
        else:
            (st.warning if res.search_status == SearchStatus.TIMEOUT else st.error)(
                f"{res.outcome.value} · search {res.search_status.value}"
                + (" (infeasibility NOT proven)" if res.search_status == SearchStatus.TIMEOUT
                   else " (candidate set exhausted)"))
        for r in res.reasons:
            st.write("• " + r)
        if res.cost_breakdown:
            st.caption("Cost breakdown (ranking only; safety constraints are filters, not costs): "
                       + ", ".join(f"{k} {v:g}" for k, v in res.cost_breakdown.items()))
        if res.rejection_by_constraint:
            with st.expander("Why candidates were rejected (first failing constraint)"):
                st.dataframe(pd.DataFrame([{"constraint": k, "title": CONSTRAINTS[k][0], "candidates": v}
                                           for k, v in res.rejection_by_constraint.items()]), hide_index=True)
        if res.plan:
            show_report(res.plan, res.validation, only_fail, "fb")

    with st.expander("Validate a plan from JSON (any source)"):
        txt = st.text_area("RecoveryPlan JSON", height=160, key="plan_json")
        if st.button("Validate pasted plan"):
            try:
                plan = RecoveryPlan.model_validate_json(txt)
            except ValidationError as e:
                st.error(f"Not a valid RecoveryPlan: {e.error_count()} problem(s)\n\n{e}")
            else:
                rep = validate_plan(sc, plan)
                run["pasted"] = (plan, rep)
                eventlog.log_event(sc.scenario_id, "plan_validated", validation_payload(rep, plan.source),
                                   dedupe_key=f"plan_validated:{plan.proposal_id}:{sc.scenario_version}")
        if "pasted" in run:
            show_report(*run["pasted"][:2], only_fail, "pasted")
        if "candidate" in run:
            st.caption("Tip: copy a plan's JSON from here, edit a field, paste above.")
            st.code(run["candidate"][0].model_dump_json(indent=1), language="json")

with st.expander("Event log (latest 20)"):
    st.dataframe(pd.DataFrame(eventlog.read_events(sc.scenario_id, 20)), hide_index=True)
