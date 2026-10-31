# Airport disruption negotiation – prototype (stage 2: workbook-backed scenarios, validator, fallback)

No LLM agents yet. Everything here is deterministic.

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
