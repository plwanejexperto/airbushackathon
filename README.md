# Airport disruption negotiation – prototype

Stage 2 (workbook scenarios, validator, fallback) is deterministic. Stage 3 adds three independent agents and a bounded
negotiation coordinator on top of it (live Claude calls or a scripted demo).

## Run
    pip install -r requirements.txt
    python3 -m streamlit run app.py      # UI
    python3 -m pytest -q tests           # tests (~30 s)

Data: `data/flights.csv`, `data/challenge2_operational_constraints.xlsx` (no airline-name table, so airline *codes* are shown).

## Layout
| module | role |
|---|---|
| `workbook.py` | reads all 4 sheets, parses text rules, returns `WorkbookIssue`s (sheet/row/column) instead of raising |
| `schemas.py` | `HistoricalFlight` / `WorkbookLimits` / `SimulatedConditions` (+ `SimAssumptions`) / `Scenario` (content-hash `scenario_version`) |
| `scenario.py` | seeded scenario for the selected flight at its **destination** airport |
| `plan.py` | `RecoveryPlan`, `FlightPlan`, explicit `ResourceInterval`s, proposal IDs |
| `validate.py` | 25 hard constraints, each result carries id, flight/resource, pass/fail, explanation, basis |
| `planner.py` / `fallback.py` | timeline builder, naive baseline, bounded fallback search |
| `eventlog.py` / `report.py` | SQLite log with unique `dedupe_key`; payload helpers |

All times are minutes from scenario start (`t+N`).

## Workbook rules implemented
| workbook source | where used |
|---|---|
| `airport_capacity.active_runways`, `gates` | size of the airport; demo models min(2, runways) runways and min(3, gates) gates, totals shown next to the subset |
| `airport_capacity.max_runway_ops_per_hour` | `RWY-CAP-AIRPORT` (any 60-min window, modelled movements only) |
| `airport_capacity.min_turnaround_min` | `TURNAROUND-MIN` (gate arrival → service end) |
| `airport_capacity.spare_gates_at_any_time` | that many modelled gates are free; remaining modelled gates carry a simulated resident aircraft |
| `wake_separation` | `WAKE-SEP`, ordered leader→follower, "Any other pair" default for unlisted pairs |
| `fuel_and_crew`: final reserve, contingency 5 % trip, holding burn | `FUEL-RESERVE`, `FUEL-CONTINGENCY` |
| `fuel_and_crew`: assumed spare 45 min | upper bound of the seeded spare fuel at nominal ETA |
| `fuel_and_crew`: flight crew max duty 13 h | `CREW-FLIGHT-DUTY` |
| `fuel_and_crew`: ground shift 8 h, overtime 1.5x | `CREW-GROUND-SHIFT` (cap) and fallback overtime cost |
| `fuel_and_crew`: gate change notice 20 min | `GATE-NOTICE` (changed gate, measured to gate arrival = dataset's ARRIVAL_TIME meaning) |
| `rules_of_play` | loaded and shown; 3-turn limit parsed, not used until agents exist |

## Interpretations where the workbook is silent (all stated in the UI)
* `max_runway_ops_per_hour` does not say per-runway or airport-wide. Read as **airport-wide** (ORD 90 = 4 × 22.5); an even per-runway share is also checked as `RWY-CAP-RUNWAY` (assumption).
* `min_time_s` is labelled "same runway departures"; all three flights are arrivals. nm → time uses an assumed 140 kt, and the time is applied as a conservative floor (`wake_apply_time_floor`).
* Fuel at start = trip + 5 % contingency + 30 min reserve + seeded spare (25–45 min) so holding beyond the spare eats contingency first, then reserve.
* Every other number is in `SimAssumptions` (runway occupancy, deboard time, gate buffer, crew transfer, post-flight duty, overtime cap, trip burn rates).

## Stage 3: three agents + bounded negotiation
    export ANTHROPIC_API_KEY=...            # live mode only; never stored in SQLite or logs
    export NEGOTIATION_MODEL=claude-opus-5-5 NEGOTIATION_CALL_TIMEOUT_S=90 NEGOTIATION_DEADLINE_S=600   # optional
    python3 -m streamlit run app.py         # tab "Negotiation (3 agents)": pick Scripted demo or Live LLM

* **Agents** (`airport_ops/agents/{atc,pilot,ground}.py`): separate role instructions, domain views and (live) separate model calls.
  Scope: ONE Pilot agent for the selected historical flight; the two competing flights are deterministic traffic.
  Stands/equipment are not modelled (ground crews are the turnaround capacity proxy). The selected flight's connection
  preference is SIMULATED and soft.
* **Messages** (`messages.py`): PROPOSE / COUNTER_PROPOSE / ACCEPT / REJECT with sender, turn, scenario version, proposal id,
  referenced id, reason codes, explanation and machine-checkable `Condition`s (defined fields/ops only). Proposals carry a
  complete `RecoveryPlan`; agents choose discrete assignments (`plan_spec`) and a deterministic tool builds the timeline.
* **Coordinator** (`negotiation.py`): turns 1-2 ATC -> PILOT -> GROUND; deterministic freeze rule; turn 3 accept/reject of the
  identical frozen proposal only; consensus needs 3 resolved ACCEPTs of the same id + version AND a fresh re-validation.
  Max 3 turns / 9 messages; per-call timeout + total deadline; one schema-repair retry per message (a separate *model attempt*).
  Timeout, API error, unrecoverable reply, no valid candidate, no unanimity -> deterministic fallback
  (`FALLBACK_SELECTED` / `NO_FEASIBLE_PLAN`, search `EXHAUSTED` vs `TIMEOUT`) - never reported as consensus.
  A scenario-version change mid-session -> `SESSION_INVALIDATED`.
* **Reviewed assumptions**: `fuel.py` documents the fuel arithmetic (trip fuel burned once; reserve and contingency protected;
  exact boundaries); `capacity.py` keeps the workbook airport-wide limit separate from the configurable per-runway share
  (`per_runway_share_mode`: even_split | fixed | disabled; default unchanged).
* Live credentials (environment only): either one key per agent - `ANTHROPIC_API_KEY_ATC`, `ANTHROPIC_API_KEY_PILOT`,
  `ANTHROPIC_API_KEY_GROUND` (all three required; each agent's calls use only its own key) - or one shared `ANTHROPIC_API_KEY`
  / `ANTHROPIC_AUTH_TOKEN`. Partly-set per-agent keys are refused (never mixed with a shared key). An `ant auth login` profile
  is not used. Missing credentials block a live run; it never switches to scripted. Keys are redacted from logs and errors.

### Keys in a `.env` file
Copy `.env.example` to `.env` (project root, git-ignored), fill in the three `ANTHROPIC_API_KEY_*` values (or one shared
`ANTHROPIC_API_KEY`), and restart the app. Only `ANTHROPIC_*` / `NEGOTIATION_*` names are read; variables already exported in
the shell win over the file; the app shows which names were read, never values.

## Requirement audit: "agents negotiate a revised arrival time, gate assignment or flight plan adjustment within 3 turns"
Run the demonstration: tick **Conflict demonstration preset** in the sidebar, open **Requirement demo**, click **Run demonstration**
(mode follows the Negotiation tab: scripted or live). Download the session JSON from the bottom of that tab. Saved evidence is in
`demo_output/` (see the evidence table below; no credentials).

* **Initial reference plan** (`reference.py`): nominal ETA, original runway/gate/crew, shortest taxi route, no conflict resolution.
  ILLUSTRATIVE reference, validated before turn 1 and shown to every agent. It normally FAILS hard constraints, so it is NOT a feasible
  baseline and no efficiency/improvement is claimed against it; NOT the historical outcome (flights.csv times are context only).
* **What can change** (`changes.py`): landing (touchdown) time, runway, taxi route (modelled graph), gate, gate arrival (on-block),
  ground crew, service times. En-route flight-plan changes are not modelled and never claimed.
* **Demonstration preset** (`demo.py`): runway closure + priority-5 Heavy (ATC wants it first), 8 min of spare fuel and a
  connection preference (Pilot), resident aircraft on the original gate + late/short-shift crew + blocked taxiway (Ground).
  Opt-in only; ordinary sessions are not forced to disagree.
* Logged separately: `model_attempt` (incl. schema repairs), `tool_call` (advisory option menus, plan-spec materialisation),
  `validation_performed` (reference / registration / freeze / final_revalidation).

## Precision rules (preference conditions vs hard constraints)
* `canonical.py` is the ONE representation of preference metrics, used both for what agents are shown and for evaluating their
  conditions. Per-field decimals: minutes / weighted delay / cost = 2, gate and runway change counts = integers, fuel margins = 6
  (effectively exact), identifiers exact. It is NOT a blanket rounding.
* Hard constraints (`validate.py`, `fuel.py`, `planner.py`, `capacity.py`, `plan.py`) never import it and use exact values with their
  own tolerances; a test asserts this. Boundary tests: a 0.4 g fuel-reserve deficit and a 10 ms wake-separation shortfall are rejected
  even though a rounded display would read "0" / the same whole second. A condition can pass while a plan is invalid; consensus
  additionally needs independent validation.

## Verification status (what each claim rests on)
| Claim | Evidence | Status |
|---|---|---|
| Scripted negotiation reaches `AGENT_CONSENSUS` on the conflict preset, with a counter-offer chain that changes the selected flight's **gate** (GATE-1 -> GATE-3) and **taxi route**; touchdown time and gate-arrival time of DL1205 are unchanged | `demo_output/scripted_demo.json`, `tests/test_requirement.py` | verified (scripted policies, not a language model) |
| Live negotiation was exercised | run 1: `demo_output/live_demo.json`, `live_events.db`, `live_run_1_console.txt` - ended `FALLBACK_SELECTED` / `NO_UNANIMOUS_AGREEMENT` because a condition compared an exact 10.7143 with a rounded 10.71 (software defect, fixed) | exercised; failure log preserved |
| Post-fix live consensus | run 2: `demo_output/live_run_2_postfix.json`, `live_run_2_events.db`, `live_run_2_console.txt` - `AGENT_CONSENSUS`, 3 turns, 9 messages, 9 model calls, 0 repairs, 62 s, independent validation passed (63 checks) | verified for this ONE session |

Run 2 detail (model `claude-opus-5-5`, per-agent keys, scenario `af1df3e57f82`, seed 5945874): ATC proposed first; the Pilot countered
(fuel/connection); Ground and ATC reasoned against the counter on cost and weighted delay; the Pilot conceded in turn 2 (measured
by the coordinator: weighted delay -4.71, cost -48.9, 2 fewer gate changes, accepting a 21-minute soft connection miss for 30 passengers).
The frozen plan was ATC's opening plan. For DL1205 the **revised arrival time** (touchdown t+22.0 -> t+25.64 and gate arrival
t+32.5 -> t+55.0) was what changed; gate, runway and taxi route did not. These are delays against an illustrative, infeasible reference,
not an improvement. The requirement is disjunctive (arrival time OR gate OR flight plan): scripted shows the gate/route path, run 2 the
arrival-time path. En-route flight-plan adjustment is not modelled and is never claimed.
One live session is not a reliability estimate: sessions vary run to run, and it was not repeated.

## Exact demo steps
1. `cp .env.example .env`, fill the three `ANTHROPIC_API_KEY_*` values (or one `ANTHROPIC_API_KEY`); never commit `.env`.
2. `python3 -m streamlit run app.py`.
3. Sidebar: tick **Conflict demonstration preset**.
4. Scripted (free): tab **Requirement demo** -> **Run demonstration** (default mode is scripted and clearly labelled).
5. Live: tab **Negotiation (3 agents)** -> mode **Live LLM** (banner must show `credentials: per_agent`) -> back to **Requirement demo**
   -> **Run demonstration** (about 1-2 minutes, ~9-12 model calls). Read sections 1-8 and download the session JSON.
6. Tests: `python3 -m pytest -q tests` (no keys needed, never reads `.env`, makes no network calls).

## Repository hygiene
`.gitignore` excludes `.env`, `*.db`, caches and `.DS_Store`; `.env.example` and the evidence JSON/console files are meant to be
tracked. Verified on a throwaway `git init` copy: nothing sensitive would be tracked and no real key value appears in any tracked file
or evidence database. Note: `data/flights.csv` (~38 MB) would also be tracked unless you ignore it or use Git LFS.
