"""Streamlit UI: python3 -m streamlit run app.py"""
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from airport_ops import data, eventlog
from airport_ops.fallback import Outcome, SearchStatus, run_fallback
from airport_ops.plan import RecoveryPlan
from airport_ops.planner import Planner
from airport_ops.report import assignment_rows, fallback_payload, validation_payload
from airport_ops.scenario import ScenarioError, build_scenario
from airport_ops.schemas import SimAssumptions
from airport_ops.validate import CONSTRAINTS, validate_plan
from airport_ops.workbook import load_workbook_rules

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

tab_h, tab_w, tab_s, tab_p = st.tabs(["📊 Historical (dataset)", "📘 Workbook limits (synthetic)",
                                      "🧪 Simulated conditions", "🛠 Plans & validation"])
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
    c3.metric("Max runway ops / hour", ap.max_runway_ops_per_hour, help="Read as airport-wide movements per hour")
    c4.metric("Min turnaround (min)", ap.min_turnaround_min)
    st.table(pd.DataFrame([
        {"resource": "Runways", "airport total": str(sub.total_runways), "modelled subset": str(sub.modelled_runways)},
        {"resource": "Gates", "airport total": str(sub.total_gates), "modelled subset": str(sub.modelled_gates)},
        {"resource": "Spare gates at any time", "airport total": str(sub.spare_gates_at_any_time),
         "modelled subset": f"{min(sub.spare_gates_at_any_time, sub.modelled_gates)} of the modelled gates kept free"}]))
    st.caption(sub.note)
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


runs = st.session_state.setdefault("runs", {})
run = runs.setdefault(sc.scenario_version, {})

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
