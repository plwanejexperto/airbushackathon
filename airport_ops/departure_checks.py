"""Hard constraints specific to turnaround + departure sessions (called by validate.validate_plan, stage "depart").
Exact values, no display quantisation."""
from __future__ import annotations

from .capacity import airport_limit, per_runway_limit
from .turnaround import task_minutes

TOL = 1e-6


def _ov(a0, a1, b0, b1, buf=0.0, tol=TOL):
    return a0 < b1 + buf - tol and b0 < a1 + buf - tol


def check_departure(ix, plan, fps, c) -> None:
    from .validate import max_window_count
    sc, a = ix.sc, ix.a
    dep = sc.simulated.departure
    tol = a.time_tolerance_min
    tm, fixed = dep.turnaround, {x.flight_id: x for x in dep.arrival}
    tdefs = {t.task_id: t for t in tm.tasks}
    units = {u.id: u for u in tm.equipment}
    nxt = {n.flight_id: n for n in dep.next_flights}
    segs = ix.segs
    dep_intervals: list[tuple[str, str, float, float]] = []      # (flight, segment, start, end)
    equip: dict[str, list[tuple[float, float, str]]] = {}
    takeoffs: list[tuple[str, str, float]] = []                  # (flight, runway, time)
    for p in fps:
        f, sf = p.flight_id, ix.flights[p.flight_id]
        af = fixed.get(f)
        # --- the arrival part is an INPUT
        if af is not None:
            diffs = [n for n, same in (("landing", abs(p.landing_min - af.landing_min) <= tol), ("runway", p.runway_id == af.runway_id),
                                       ("taxi route", list(p.taxi_route) == af.taxi_route), ("gate", p.gate_id == af.gate_id),
                                       ("gate arrival", abs(p.gate_in_min - af.gate_in_min) <= tol)) if not same]
            c.check("DEP-ARRIVAL-FIXED", not diffs, lambda: ("arrival part unchanged" if not diffs else "changed: " + ", ".join(diffs)), f)
        # --- tasks
        got = {t.task_id: t for t in p.tasks}
        dup = len(got) != len(p.tasks)
        ok_set = set(got) == set(tdefs) and not dup
        c.check("TASK-SET", ok_set, lambda: f"{len(got)} of {len(tdefs)} tasks scheduled" + ("" if ok_set else f" (missing {sorted(set(tdefs) - set(got))}, unknown {sorted(set(got) - set(tdefs))}, duplicates {dup})"), f)
        if ok_set and p.gate_id in ix.gates:
            durs = task_minutes(sc, f, p.gate_id)
            short = [f"{t.task_id} {t.end_min - t.start_min:.1f} < {durs[t.task_id]:.1f}" for t in p.tasks if t.end_min - t.start_min < durs[t.task_id] - tol]
            c.check("TASK-DUR", not short, lambda: "; ".join(short) if short else "every task lasts at least its work content", f)
            bad = []
            for t in p.tasks:
                if t.start_min < p.gate_in_min - tol:
                    bad.append(f"{t.task_id} starts t+{t.start_min:.1f} before gate arrival t+{p.gate_in_min:.1f}")
                for pre in tdefs[t.task_id].predecessors:
                    if t.start_min < got[pre].end_min - tol:
                        bad.append(f"{t.task_id} starts t+{t.start_min:.1f} before {pre} ends t+{got[pre].end_min:.1f}")
            c.check("TASK-DEPS", not bad, lambda: "; ".join(bad[:3]) if bad else "all dependencies respected (parallel tasks overlap where allowed)", f)
            eq_bad = []
            for t in p.tasks:
                need = tdefs[t.task_id].equipment
                if need is None:
                    if t.equipment_id:
                        eq_bad.append(f"{t.task_id} needs no equipment but uses {t.equipment_id}")
                elif t.equipment_id not in units or units[t.equipment_id].type != need:
                    eq_bad.append(f"{t.task_id} needs {need} but has {t.equipment_id!r}")
                else:
                    equip.setdefault(t.equipment_id, []).append((t.start_min, t.end_min, f))
            c.check("TASK-EQUIP", not eq_bad, lambda: "; ".join(eq_bad) if eq_bad else "equipment valid", f)
            last_end = max(t.end_min for t in p.tasks)
            rt_ok = (p.ready_min is not None and p.ready_min >= last_end - tol and p.ready_min >= p.gate_in_min + ix.ap.min_turnaround_min - tol
                     and abs(p.service_end_min - p.ready_min) <= tol)
            c.check("READY-TIME", rt_ok, lambda: f"ready t+{p.ready_min} vs last task end t+{last_end:.1f}, gate arrival + minimum turnaround t+{p.gate_in_min + ix.ap.min_turnaround_min:.1f}, service end t+{p.service_end_min:.1f}", f)
        # --- departure taxi / take-off
        if p.takeoff_min is None or p.dep_runway_id is None or p.ready_min is None:
            c.check("DEP-SEQ", False, lambda: "departure fields missing (dep_runway_id, dep_taxi_*, takeoff_min, ready_min)", f)
            continue
        dsegs = [segs.get(s) for s in p.dep_taxi_route]
        err = None
        if not p.dep_taxi_route or any(s is None for s in dsegs):
            err = "empty or unknown outbound route"
        elif p.dep_runway_id not in ix.runways:
            err = f"unknown departure runway {p.dep_runway_id!r}"
        else:
            if dsegs[0].from_node != f"GATE-{p.gate_id}":
                err = f"route starts at {dsegs[0].from_node}, expected GATE-{p.gate_id}"
            for s1, s2 in zip(dsegs, dsegs[1:]):
                if s1.to_node != s2.from_node:
                    err = f"{s1.id} ends at {s1.to_node} but {s2.id} starts at {s2.from_node}"
            if err is None and dsegs[-1].to_node != f"DEP-{p.dep_runway_id}":
                err = f"route ends at {dsegs[-1].to_node}, expected DEP-{p.dep_runway_id}"
        c.check("DEP-TAXI-ROUTE", err is None, lambda: err or f"{' > '.join(p.dep_taxi_route)} connects {p.gate_id} to runway {p.dep_runway_id}", f)
        total = sum(s.minutes for s in dsegs) if err is None else 0.0
        seq_err = []
        if p.dep_taxi_start_min is None or p.dep_taxi_end_min is None:
            seq_err.append("dep_taxi_start/end missing")
        else:
            if p.dep_taxi_start_min < p.ready_min + tm.pushback_min - tol:
                seq_err.append(f"taxi-out starts t+{p.dep_taxi_start_min:.1f} before ready + push-back t+{p.ready_min + tm.pushback_min:.1f}")
            if err is None and p.dep_taxi_end_min - p.dep_taxi_start_min < total - tol:
                seq_err.append(f"taxi-out {p.dep_taxi_end_min - p.dep_taxi_start_min:.1f} min < route time {total:.1f} min")
            if p.takeoff_min < p.dep_taxi_end_min - tol:
                seq_err.append(f"take-off t+{p.takeoff_min:.1f} before reaching the runway t+{p.dep_taxi_end_min:.1f}")
        c.check("DEP-SEQ", not seq_err, lambda: "; ".join(seq_err) if seq_err else f"ready t+{p.ready_min:.1f} < taxi-out t+{p.dep_taxi_start_min:.1f}..{p.dep_taxi_end_min:.1f} < take-off t+{p.takeoff_min:.1f}", f)
        if err is None and p.dep_taxi_end_min is not None:
            t = p.dep_taxi_end_min - total
            hits = []
            for s in dsegs:
                dep_intervals.append((f, s.id, t, t + s.minutes))
                for b in s.blockages:
                    if _ov(t, t + s.minutes, b.start_min, b.end_min):
                        hits.append(f"{s.id} used t+{t:.1f}..{t + s.minutes:.1f} while blocked t+{b.start_min:.1f}..{b.end_min:.1f} ({b.reason})")
                t += s.minutes
            c.check("DEP-TAXI-BLOCK", not hits, lambda: "; ".join(hits) if hits else "outbound route clear of blockages", f)
        roll = dep.departure_roll_min
        rw = ix.runways.get(p.dep_runway_id)
        if rw is not None:
            hit = [cl for cl in rw.closures if _ov(p.takeoff_min, p.takeoff_min + roll, cl.start_min, cl.end_min)]
            c.check("DEP-RWY-AVAIL", not hit, lambda: (f"{rw.id} open for the take-off roll t+{p.takeoff_min:.1f}" if not hit else
                    f"{rw.id} closed t+{hit[0].start_min:.1f}..{hit[0].end_min:.1f} ({hit[0].reason}); take-off t+{p.takeoff_min:.1f}"), f, rw.id)
        sched_dep = dep.scheduled_departure_min.get(f)
        if sched_dep is not None:
            c.check("DEP-NOT-EARLY", p.takeoff_min >= sched_dep - tol, lambda: f"take-off t+{p.takeoff_min:.1f} vs scheduled departure t+{sched_dep:.1f}", f)
        takeoffs.append((f, p.dep_runway_id, p.takeoff_min))
        # --- next flight: can the same flight crew legally fly it?
        nf, crew = nxt.get(f), ix.fcrews.get(p.flight_crew_id)
        if nf is not None and crew is not None:
            limit = crew.report_time_min + ix.fc.flight_crew_max_duty_h * 60
            end = p.takeoff_min + nf.scheduled_block_min + a.flight_crew_postflight_min
            c.check("DEP-CREW-DUTY", end <= limit + TOL, lambda: f"{crew.id}: take-off t+{p.takeoff_min:.1f} + next flight {nf.next_flight_label} "
                    f"{nf.scheduled_block_min:g} min + {a.flight_crew_postflight_min:g} min post-flight = t+{end:.1f} vs duty limit t+{limit:.1f}", f, crew.id)
    # --- equipment: one task at a time per unit
    for uid, items in sorted(equip.items()):
        items.sort()
        clash = [(x, y) for i, x in enumerate(items) for y in items[i + 1:] if _ov(x[0], x[1], y[0], y[1], 0.0, a.time_tolerance_min)]
        if not clash:
            c.check("TASK-EQUIP", True, lambda: f"{uid}: {len(items)} task(s), no overlap", None, uid)
        for x, y in clash:
            c.check("TASK-EQUIP", False, lambda: f"{uid}: {x[2]} t+{x[0]:.1f}..{x[1]:.1f} overlaps {y[2]} t+{y[0]:.1f}..{y[1]:.1f}", y[2], uid)
    # --- outbound taxi single lane
    by_seg: dict[str, list] = {}
    for f, sid, s0, s1 in dep_intervals:
        by_seg.setdefault(sid, []).append((s0, s1, f))
    for sid, items in sorted(by_seg.items()):
        items.sort()
        clash = [(x, y) for i, x in enumerate(items) for y in items[i + 1:] if x[2] != y[2] and _ov(x[0], x[1], y[0], y[1], 0.0, a.time_tolerance_min)]
        if not clash:
            c.check("DEP-TAXI-CONFLICT", True, lambda: f"{sid}: no overlap", None, sid)
        for x, y in clash:
            c.check("DEP-TAXI-CONFLICT", False, lambda: f"{sid}: {x[2]} t+{x[0]:.1f}..{x[1]:.1f} overlaps {y[2]} t+{y[0]:.1f}..{y[1]:.1f}", y[2], sid)
    # --- runway: occupancy vs arrivals and other departures, workbook departure spacing
    roll, gap = dep.departure_roll_min, dep.arrival_departure_gap_min
    arr = [(x.runway_id, x.landing_min, x.landing_min + a.runway_occupancy_min, x.flight_id) for x in dep.arrival]
    for i, (f, rid, t) in enumerate(takeoffs):
        bad = [f"arrival {af_} t+{o0:.1f}..{o1:.1f}" for r2, o0, o1, af_ in arr if r2 == rid and _ov(t, t + roll, o0, o1, gap)]
        bad += [f"departure {f2} t+{t2:.1f}" for f2, r2, t2 in takeoffs[i + 1:] if r2 == rid and _ov(t, t + roll, t2, t2 + roll)]
        c.check("DEP-RWY-OCC", not bad, lambda: f"{f} take-off t+{t:.1f} overlaps " + "; ".join(bad) if bad else f"{f} take-off roll clear on {rid}", f, rid)
    for rid in sorted({r for _, r, _ in takeoffs}):
        seq = sorted((t, f) for f, r, t in takeoffs if r == rid)
        for (t0, f0), (t1, f1) in zip(seq, seq[1:]):
            _, secs, label = ix.wake.lookup(ix.flights[f0].wake_category, ix.flights[f1].wake_category)
            c.check("DEP-WAKE", (t1 - t0) * 60 >= secs - 1e-3, lambda: f"{f0}({ix.flights[f0].wake_category.value}) -> {f1}({ix.flights[f1].wake_category.value}) on {rid}: "
                    f"gap {(t1 - t0) * 60:.3f}s, required {secs:g}s [{label}: workbook min_time_s]", f1, rid)
    times = [x.landing_min for x in dep.arrival] + [t for _, _, t in takeoffs]
    lim = airport_limit(sc)
    n = max_window_count(times)
    c.check("DEP-CAP", n <= lim, lambda: f"max {n} movement(s) (arrivals + departures) in any 60-min window vs airport limit {lim}/h", None, ix.ap.iata_code)
