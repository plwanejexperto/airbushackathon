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
