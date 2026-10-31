"""Streamlit UI: python3 -m streamlit run app.py"""
import os

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from airport_ops import agents, data, eventlog, llm
from airport_ops.changes import build_change_log
from airport_ops.demo import DEMO_LABEL, build_demo_scenario
from airport_ops.export import export_json
from airport_ops.fallback import Outcome, SearchStatus, run_fallback
from airport_ops.diversion import DiversionUnavailable, build_diversion_scenario
from airport_ops.messages import SPEAKING_ORDER, AgentRole, legacy_participants
from airport_ops.sessions import participants_for
from airport_ops.turnaround import build_departure_demo, build_departure_scenario
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
    st.divider()
    KINDS = ["Landing & delay (scenarios 1+2)", "Turnaround & departure (scenario 4)", "Diversion / emergency (scenario 3)"]
    kind_label = st.radio("Scenario", KINDS, key="scenario_kind",
                          help="1+2: runway-gate-fuel conflict for a delayed arrival. 4: turnaround and next departure on an agreed arrival plan. "
                               "3: the destination cannot take the aircraft, divert to one alternate.")
    kind = {KINDS[0]: "arrival", KINDS[1]: "departure", KINDS[2]: "diversion"}[kind_label]
    div_case, div_emerg = "normal", False
    if kind == "diversion":
        div_case = st.selectbox("Fuel situation", ["normal", "minimum_fuel", "emergency", "unreachable"], key="div_case",
                                help="How much fuel is on board relative to the alternate's needs. The STATUS (normal diversion / minimum fuel / "
                                     "emergency) is then COMPUTED from distance, time and burn. 'unreachable' has no feasible alternate: ESCALATE.")
        div_emerg = st.checkbox("Emergency declared (input)", key="div_emerg", help="e.g. medical/technical: forces EMERGENCY status regardless of fuel.")
    full_agents = True
    if kind == "arrival":
        full_agents = st.radio("Agents", ["Per-aircraft Pilots + ATC + Ground (5 agents)", "Legacy: ATC, Pilot (selected flight), Ground (3 agents)"],
                               key="agent_set").startswith("Per-aircraft")
    demo_on = st.checkbox("Conflict demonstration preset", key="demo_preset",
                          help="Opt-in: simulated closure, gate, crew and fuel conditions that make ATC, Pilot and Ground "
                               "want different plans. Off = ordinary seeded scenario.")

h = data.to_historical(df.loc[sel])
_cache = st.session_state.setdefault("_scenario_cache", {})


def _arrival_plan_for(base):
    """Arrival plan used as the INPUT of a turnaround/departure session: the latest arrival session's final plan on this scenario if there
    is one, otherwise the deterministic fallback plan (a feasible baseline)."""
    prior = [r for r in st.session_state.get("neg_runs", {}).values() if r.scenario_version == base.scenario_version and r.final_plan is not None]
    if prior:
        r = prior[-1]
        return r.final_plan, f"arrival session {r.session_id} ({'agent consensus' if r.consensus else 'deterministic fallback'})"
    key = ("fb", base.scenario_version)
    if key not in _cache:
        _cache[key] = run_fallback(base).plan
    return _cache[key], "deterministic fallback arrival plan (feasible baseline)"


try:
    base_sc = build_scenario(h, rules)
    arrival_src = ""
    if kind == "arrival":
        sc = build_demo_scenario(base_sc) if demo_on else base_sc
    elif kind == "departure":
        arr_plan, arrival_src = _arrival_plan_for(base_sc)
        sc = build_departure_scenario(base_sc, arr_plan, df, plan_source=arrival_src)
        if demo_on:
            sc = build_departure_demo(sc)
    else:
        try:
            sc = build_diversion_scenario(base_sc, rules, df, fuel_case=div_case, emergency_declared=div_emerg)
        except DiversionUnavailable as e:
            st.error(f"**ESCALATE – no feasible diversion.** {e}. There is nothing to negotiate: no reachable, suitable alternate exists on the fuel on board.")
            st.dataframe(pd.DataFrame([{"alternate": c.airport, "name": c.name, "distance from position (nm)": c.distance_from_position_nm,
                                        "flight time (min)": c.flight_time_min, "trip fuel (kg)": c.trip_fuel_kg, "suitable": c.suitable,
                                        "why not": "; ".join(c.reasons)} for c in e.candidates]), hide_index=True)
            st.caption("Position, distances and fuel are computed from the aircraft's simulated position; weather is simulated.")
            st.stop()
except ScenarioError as e:
    st.error(f"Cannot build a scenario for {h.label}: {e}")
    st.stop()
# idempotent: a rerun (or another session) with the same scenario version writes nothing
eventlog.log_event(sc.scenario_id, "scenario_created",
                   {"flight": h.label, "seed": sc.simulated.seed, "scenario_version": sc.scenario_version,
                    "airport": sc.workbook.airport.iata_code},
                   dedupe_key=f"scenario_created:{sc.scenario_version}")

st.caption(f"Scenario `{sc.scenario_id}` · version `{sc.scenario_version}` · seed {sc.simulated.seed} (deterministic per flight) · "
           f"**{kind_label}** · plan airport **{sc.workbook.airport.iata_code}**"
           + (f" (diverted from {sc.simulated.diversion.destination.airport})" if sc.simulated.diversion else "")
           + (f" · arrival input: {arrival_src}" if arrival_src else ""))

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


if demo_on:
    st.warning(f"**{DEMO_LABEL}** is ON: simulated runway closure, gate/crew/fuel conditions were added so the agents' "
               "preferences conflict. Ordinary sessions are not forced to disagree.")
tab_h, tab_w, tab_s, tab_d, tab_n, tab_p = st.tabs(["📊 Historical (dataset)", "📘 Workbook limits (synthetic)", "🧪 Simulated conditions",
                                                    "🎯 Requirement demo", "🤝 Negotiation (agents)", "🛠 Plans & validation"])
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


def _fv(v) -> str:
    """Display value as text (mixed str/float columns break Arrow conversion)."""
    return v if isinstance(v, str) else f"{v:.1f}"


def run_session(live: bool) -> None:
    """Runs ONE new negotiation (new session id). Called only from a button click, never on a plain rerun."""
    live_cfg_ = llm.LiveConfig.from_env()
    try:
        client = llm.make_client("live" if live else "scripted")
    except llm.MissingCredentials as e:
        st.error(f"Cannot start live run: {e}")
        return
    legacy = kind == "arrival" and not full_agents
    kw = dict(call_timeout_s=live_cfg_.call_timeout_s, deadline_s=live_cfg_.deadline_s) if live else {}
    cfg = NegotiationConfig(require_pilot_report=not legacy, **kw)
    parts = None if legacy else participants_for(sc)
    sid = new_session_id()
    with st.spinner(f"Negotiating ({'live Claude calls' if live else 'scripted'}; {len(parts) if parts else 3} agents)…"):
        neg_runs[sid] = Coordinator(sc, client, config=cfg, session_id=sid, participants=parts).run()
    neg_latest[sc.scenario_version] = sid

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
        run_session(live)

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
        cols[3].metric("Messages", f"{len(res.messages)} / {3 * len(res.participants or SPEAKING_ORDER)}")
        cols[4].metric("Session", res.session_id)
        if res.scenario_version != sc.scenario_version:
            st.error("This session belongs to a different scenario version and is invalid for the current scenario.")
        S = res.status
        if S == FinalStatus.AGENT_CONSENSUS:
            st.success(f"**AGENT_CONSENSUS** – all {len(res.participants or SPEAKING_ORDER)} agents accepted frozen proposal `{res.frozen_proposal_id}` "
                       f"(scenario `{res.scenario_version}`) and an independent re-validation passed "
                       f"({len(res.final_validation.results)} checks).")
        elif S == FinalStatus.FALLBACK_SELECTED:
            st.warning(f"**FALLBACK_SELECTED** – this is the deterministic fallback planner's plan, **NOT agent consensus**. "
                       f"Reason: `{res.fallback_reason.value}`")
        elif S == FinalStatus.NO_FEASIBLE_PLAN:
            st.error(f"**NO_FEASIBLE_PLAN – ESCALATE** – agents did not reach consensus and the fallback found no valid plan "
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
        parts = res.participants or legacy_participants(h.label)
        blank = lambda p: AgentContext(role=p.role, agent=p, participants=parts, turn=1, seq=1, session_id=res.session_id, scenario=sc,
                                       historical=h, messages=[], proposals=[], standing={}, allowed_types=[])
        slots = [(c, p) for row in (parts[i:i + 3] for i in range(0, len(parts), 3)) for c, p in zip(st.columns(3), row)]
        for col, p in slots:
            role = p.role
            with col:
                icon = {"ATC": "🛫", "PILOT": "👨‍✈️", "GROUND": "🧰"}[role.value]
                st.markdown(f"### {icon} {p.agent_id}")
                st.caption({"ATC": "runways · separation · taxiways · capacity" + (f" · airport {p.airport}" if p.airport else ""),
                            "PILOT": f"aircraft {p.flight_id or h.label} only · fuel · ETA window · connection",
                            "GROUND": "gates · stands · crews · turnaround" + (f" · airport {p.airport}" if p.airport else "")}[role.value])
                with st.expander("Domain view (what this agent sees)"):
                    st.json(agents.MODULES[role].domain_view(blank(p)))
                for m in [m for m in res.messages if (m.agent_id or m.sender.value) == p.agent_id]:
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
            "seq": m.seq, "turn": m.turn, "agent": m.agent_id or m.sender.value, "type": m.type.value, "proposal": m.proposal_id,
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

# ---------- requirement demonstration view ----------
with tab_d:
    st.markdown("#### Requirement: agents negotiate a revised arrival time, gate assignment or flight-plan adjustment within 3 turns")
    _live_sel = st.session_state.get("neg_mode", "").startswith("Live")
    _cfg = llm.LiveConfig.from_env()
    st.caption(("Mode selected in the Negotiation tab: **LIVE LLM**" if _live_sel else "Mode selected in the Negotiation tab: **SCRIPTED DEMO (no LLM)**")
               + (" · conflict preset ON" if demo_on else " · conflict preset OFF (tick it in the sidebar for a visible negotiation)"))
    if st.button("Run demonstration", key="demo_run", type="primary", disabled=_live_sel and not _cfg.credentials_present,
                 help="Starts ONE new session with the mode chosen in the Negotiation tab. Reruns never repeat it."):
        run_session(_live_sel)
    dres = neg_runs.get(neg_latest.get(sc.scenario_version, ""))
    if dres is None:
        st.info("No session yet for this scenario version. Click **Run demonstration**.")
    else:
        cl = build_change_log(sc, dres)
        mode_name = {"live": "LIVE LLM calls", "scripted": "SCRIPTED policies (no language model)", "mock": "MOCK client"}.get(dres.mode, dres.mode)
        (st.info if dres.mode == "live" else st.warning)(f"Session `{dres.session_id}` · **{mode_name}** · scenario `{dres.scenario_version}`"
                                                         + (" · conflict preset" if demo_on else ""))
        # 0 -- scenario-specific inputs
        if sc.simulated.diversion is not None:
            dv_ = sc.simulated.diversion
            st.subheader("0 · Diversion case: position, options and computed status")
            st.write(f"The aircraft cannot land at **{dv_.destination.airport}** within its fuel limit (earliest feasible landing "
                     f"t+{dv_.destination.earliest_feasible_landing_min:.1f}, latest by fuel t+{dv_.destination.latest_landing_min_by_fuel:.1f}). "
                     f"Computed status: **{dv_.computed_status.value}**" + (" (emergency declared as an input)" if dv_.emergency_declared_input else "") + ".")
            st.caption(dv_.status_rule)
            st.dataframe(pd.DataFrame([{"item": k, "value": str(v)} for k, v in dv_.position.items()]), hide_index=True)
            st.dataframe(pd.DataFrame([{"option": "ORIGINAL DESTINATION " + dv_.destination.airport, "chosen": False, "suitable": dv_.destination.feasible,
                                        "distance from position (nm)": dv_.position["distance_to_destination_nm"], "flight time (min)": dv_.position["minutes_to_destination"],
                                        "why": f"closed until t+{dv_.destination.earliest_feasible_landing_min:.0f} (SIMULATED); fuel allows t+{dv_.destination.latest_landing_min_by_fuel:.0f}"}]
                                      + [{"option": "ALTERNATE " + c.airport + (f" ({c.name})" if c.name else ""), "chosen": c.airport == dv_.alternate_airport, "suitable": c.suitable,
                                          "distance from position (nm)": c.distance_from_position_nm, "flight time (min)": c.flight_time_min,
                                          "why": ("; ".join(c.reasons) or "reachable on fuel, size ok, simulated weather ok") + f" · {c.distance_method[:60]}"} for c in dv_.candidates]),
                         hide_index=True)
            st.caption("Alternates are CANDIDATES because flights.csv connects them to the destination; suitability is decided by workbook size/capacity, fuel reach "
                       "from the simulated position, and simulated weather. " + " ".join(dv_.assumptions[:2]))
        if sc.simulated.departure is not None:
            dp_ = sc.simulated.departure
            st.subheader("0 · Turnaround + departure inputs (arrival part is fixed)")
            st.caption(f"Arrival plan used as input: {dp_.arrival_plan_source or 'n/a'}")
            st.dataframe(pd.DataFrame([{"aircraft": n.flight_id, "next flight": n.next_flight_label, "source": n.source, "planned ground time (min)": n.planned_ground_min,
                                        "next flight block (min)": n.scheduled_block_min, "scheduled departure": f"t+{dp_.scheduled_departure_min[n.flight_id]:.1f}",
                                        "note": n.note} for n in dp_.next_flights]), hide_index=True)
        # 1 -- disruption
        st.subheader("1 · Selected historical disruption (dataset row)")
        hh = sc.historical
        st.write(f"**{hh.label}** {hh.origin_code}→{hh.destination_code} on {hh.flight_date}: scheduled arrival {hh.scheduled_arrival}, "
                 f"actual {hh.actual_arrival}, arrival delay {hh.arrival_delay_min} min (weather {hh.weather_delay_min}, late-aircraft "
                 f"{hh.late_aircraft_delay_min}).")
        st.caption("Context only, passed to every agent. The negotiation below is in simulated minutes from scenario start and does "
                   "not recover or reproduce this historical delay.")
        # 2 -- reference
        st.subheader("2 · Initial reference assignments (before any negotiation)")
        if dres.reference_plan is not None:
            rv = dres.reference_validation
            st.dataframe(pd.DataFrame(assignment_rows(dres.reference_plan, sc)), hide_index=True)
            (st.success if rv.passed else st.error)(
                f"Reference plan `{dres.reference_plan.proposal_id}`: " + ("valid" if rv.passed else
                f"INVALID - {len(rv.failures)} hard-constraint failures: " + ", ".join(sorted({r.constraint_id for r in rv.failures}))))
            st.caption("ILLUSTRATIVE reference (nominal ETA, original runway/gate/crew, no conflict resolution). Usually infeasible, so it is not a "
                       "feasible baseline, not the historical outcome, and no improvement is claimed against it.")
        # 3 -- turns
        st.subheader("3 · Turn-by-turn structured messages")
        for t_ in (1, 2, 3):
            ms = [m for m in dres.messages if m.turn == t_]
            st.markdown(f"**Turn {t_}**" + (" - vote on the frozen candidate" if t_ == 3 else "") + f" ({len(ms)}/3 messages)")
            if t_ == 3 and dres.frozen_proposal_id:
                st.caption(f"Frozen before turn 3: `{dres.frozen_proposal_id}` (no plan changes allowed in turn 3)")
            for m in ms:
                pr = dres.proposal(m.proposal_id)
                st.markdown(f"- **{m.sender.value}** · {m.type.value} · `{m.proposal_id}`"
                            + (f" (counters `{m.referenced_proposal_id}`)" if m.referenced_proposal_id else "")
                            + f" · plan {'valid' if pr.validation.passed else 'INVALID'}"
                            + (f" · conditions resolved: {m.conditions_resolved}" if m.conditions_resolved is not None else "")
                            + f"  \n  reasons: {', '.join(r.value for r in m.reason_codes) or '-'}  \n  {m.explanation}")
        # 4 -- changes
        st.subheader("4 · Proposal changes and reasons")
        st.caption(f"Final plan kind: **{cl.final_kind}**. Lineage (counter-offer chain): "
                   + (" → ".join(f"{l['proposer']} {l['proposal_id']}" for l in cl.lineage) or "none"))
        if cl.final_changes:
            st.dataframe(pd.DataFrame([{
                "flight": c.flight_id, "what changed": c.label, "reference": _fv(c.before), "final": _fv(c.after),
                "proposed by": f"{c.proposed_by} (msg {c.message_seq})" if c.message_seq else (c.proposed_by or "-"),
                "reason codes": ", ".join(c.reason_codes) or "-", "why": c.driver,
                "constraint evidence": "; ".join(f"{e['constraint_id']}: {e['reference_result'][:70]}" for e in c.evidence) or "-"}
                for c in cl.final_changes]), hide_index=True)
        else:
            st.info("No difference between the reference plan and the final plan.")
        with st.expander("Every revision along the counter-offer chain"):
            st.dataframe(pd.DataFrame([{"step": c.step, "by": c.proposed_by, "msg": c.message_seq, "flight": c.flight_id, "what": c.label,
                                        "before": _fv(c.before), "after": _fv(c.after),
                                        "reasons": ", ".join(c.reason_codes)} for c in cl.step_changes]), hide_index=True)
        # 5 -- frozen + votes
        st.subheader("5 · Frozen plan and the final votes (one per agent)")
        if dres.frozen_proposal_id:
            fz = dres.proposal(dres.frozen_proposal_id)
            st.write(f"Frozen `{fz.proposal_id}` (version {fz.version_no}, digest `{fz.digest}`) - selection rule: {FREEZE_RULE}")
            st.dataframe(pd.DataFrame([{"agent": v.role.value, "vote": v.type.value if v.type else "-", "proposal": v.proposal_id or "-",
                                        "counts toward consensus": v.counted_as_accept, "note": v.note} for v in dres.votes]),
                         hide_index=True)
        else:
            st.info("No candidate was frozen, so there was no final vote.")
        # 6 -- before/after
        st.subheader(f"6 · Reference → revised for the selected flight ({sc.simulated.flights[0].flight_id})")
        sf = cl.selected_flight
        if sf.get("reference") and sf.get("revised"):
            R, V = sf["reference"], sf["revised"]
            who = lambda *fields: ", ".join(sorted({str(sf["proposed_by"][f]) for f in fields if f in sf["proposed_by"]})) or "unchanged"
            rows = [("Landing / touchdown time", f"t+{R['landing_min']:.1f}", f"t+{V['landing_min']:.1f}", who("landing_min")),
                    ("Gate arrival / on-block time", f"t+{R['gate_arrival_min']:.1f}", f"t+{V['gate_arrival_min']:.1f}", who("gate_in_min")),
                    ("Gate assignment", R["gate"], V["gate"], who("gate_id")),
                    ("Runway", R["runway"], V["runway"], who("runway_id")),
                    ("Taxi route (modelled)", R["taxi_route"], V["taxi_route"], who("taxi_route"))]
            if "takeoff_min" in R and "takeoff_min" in V:
                rows += [("Aircraft ready for departure", f"t+{R['ready_min']:.1f}", f"t+{V['ready_min']:.1f}", who("ready_min")),
                         ("Departure runway", R["dep_runway"], V["dep_runway"], who("dep_runway_id")),
                         ("Take-off time", f"t+{R['takeoff_min']:.1f}", f"t+{V['takeoff_min']:.1f}", who("takeoff_min"))]
            st.dataframe(pd.DataFrame([{"item": a, "reference": b, "revised": c, "proposed by": d, "result": "CHANGED" if b != c else "same"}
                                       for a, b, c, d in rows]), hide_index=True)
            changed = [r[0] for r in rows if r[1] != r[2]]
            (st.success if changed and dres.consensus else st.warning)(
                ("Changed for the selected flight: " + ", ".join(changed)) if changed else
                "No arrival, gate or route change for the selected flight in this outcome.")
            st.caption("Nominal ETA is t+%.1f. Touchdown time is reported separately from gate arrival; only rows marked CHANGED changed. "
                       "En-route flight-plan changes are not modelled." % sf["nominal_eta_min"])
        else:
            st.info("No final plan, so no before/after.")
        # 7 -- validation
        st.subheader("7 · Hard-constraint validation (independent validator)")
        st.dataframe(pd.DataFrame([{"stage": v.stage, "proposal": v.proposal_id, "passed": v.passed, "checks": v.n_checks,
                                    "failed constraints": ", ".join(v.failed_constraints) or "-"} for v in dres.validations]),
                     hide_index=True)
        st.caption(f"{len(dres.tool_calls)} deterministic tool calls and {len(dres.attempts)} model attempts are logged separately "
                   "(see the Negotiation tab and the download).")
        # 8 -- outcome
        st.subheader("8 · Outcome")
        if dres.status == FinalStatus.AGENT_CONSENSUS:
            st.success(f"**AGENT_CONSENSUS** after {dres.turns_completed} turn(s) / {len(dres.messages)} messages: all {len(dres.participants or SPEAKING_ORDER)} agents accepted "
                       f"`{dres.frozen_proposal_id}` and independent re-validation passed.")
        elif dres.status == FinalStatus.FALLBACK_SELECTED:
            st.warning(f"**FALLBACK_SELECTED** - NOT agent consensus. Reason `{dres.fallback_reason.value}`: {dres.fallback_detail}")
        elif dres.status == FinalStatus.NO_FEASIBLE_PLAN:
            st.error(f"**NO_FEASIBLE_PLAN** (search {dres.fallback.search_status.value}). Reason `{dres.fallback_reason.value}`: {dres.fallback_detail}")
        else:
            st.error(f"**SESSION_INVALIDATED** - {dres.invalidation_detail}")
        st.download_button("Download session JSON (no credentials)", data=export_json(sc, dres), file_name=f"negotiation_{dres.session_id}.json",
                           mime="application/json", key=f"dl-{dres.session_id}")

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
