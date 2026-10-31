"""Timeline construction from discrete choices (landing order, runway, gate, ground crew).

`Planner.build` resolves conflicts by *delaying* (landing after closures / wake / capacity; taxi start
after blockages and other aircraft; gate arrival after the gate frees; service after the crew frees).
It never checks fuel, duty, gate class or gate-change notice: those are hard constraints enforced
only by validate.py, so a built plan can still be rejected.

`naive_plan` is the opposite: nominal ETAs, original assignments, no conflict resolution. It is the
"candidate plan from some source" that the validator is expected to catch.
"""
from __future__ import annotations

from typing import Optional

from .capacity import airport_limit, per_runway_limit
from .plan import FlightPlan, RecoveryPlan, make_plan
from .schemas import Scenario
from .validate import TOL

_MAX_ITERS = 200


class Planner:
    def __init__(self, scenario: Scenario):
        self.sc = scenario
        s = scenario.simulated
        self.a = s.assumptions
        self.flights = {f.flight_id: f for f in s.flights}
        self.runways = {r.id: r for r in s.runways}
        self.segs = {x.id: x for x in s.taxi_segments}
        self.gates = {g.id: g for g in s.gates}
        self.gcrews = {c.id: c for c in s.ground_crews}
        ap = scenario.workbook.airport
        self.cap_airport = airport_limit(scenario)
        self.cap_runway = per_runway_limit(scenario)  # None = per-runway check disabled
        self.min_turn = ap.min_turnaround_min
        self.wake = scenario.workbook.wake
        self._landing_cache: dict = {}
        self._routes = self._enumerate_routes()

    # ---- taxi graph ----
    def _enumerate_routes(self) -> dict[tuple[str, str], list[tuple[list[str], float]]]:
        out_edges: dict[str, list] = {}
        for sg in sorted(self.segs.values(), key=lambda x: x.id):
            out_edges.setdefault(sg.from_node, []).append(sg)
        routes: dict[tuple[str, str], list] = {}

        def dfs(node, path, visited, rw):
            for sg in out_edges.get(node, []):
                if sg.to_node in visited:
                    continue
                p2 = path + [sg.id]
                if sg.to_node.startswith("GATE-"):
                    routes.setdefault((rw, sg.to_node[5:]), []).append((p2, sum(self.segs[i].minutes for i in p2)))
                else:
                    dfs(sg.to_node, p2, visited | {sg.to_node}, rw)

        for rid in self.runways:
            dfs(f"EXIT-{rid}", [], {f"EXIT-{rid}"}, rid)
        for k in routes:
            routes[k].sort(key=lambda r: (r[1], r[0]))
        return routes

    def shortest_route(self, runway_id: str, gate_id: str) -> Optional[tuple[list[str], float]]:
        r = self._routes.get((runway_id, gate_id))
        return r[0] if r else None

    # ---- landing stage ----
    def landing_schedule(self, order: tuple[str, ...], runways: dict[str, str]) -> dict[str, float]:
        """Earliest landing per flight given a sequence; same-runway followers wait for wake spacing."""
        key = (order, tuple(sorted(runways.items())))
        if key in self._landing_cache:
            return self._landing_cache[key]
        placed: list[tuple[str, str, float]] = []
        a = self.a
        for fid in order:
            sf, rid = self.flights[fid], runways[fid]
            rw = self.runways[rid]
            t = sf.earliest_landing_min
            for _ in range(_MAX_ITERS):
                t0 = t
                for cl in rw.closures:
                    if t < cl.end_min - TOL and t + a.runway_occupancy_min > cl.start_min + TOL:
                        t = cl.end_min
                for pf, prid, pt in placed:
                    if prid == rid:
                        nm, ts, _ = self.wake.lookup(self.flights[pf].wake_category, sf.wake_category)
                        req = nm / a.approach_speed_kt * 3600.0
                        if a.wake_apply_time_floor:
                            req = max(req, ts)
                        t = max(t, pt + max(req / 60.0, a.runway_occupancy_min))
                same = sorted((pt for _, prid, pt in placed if prid == rid), reverse=True)
                if self.cap_runway is not None and len(same) >= self.cap_runway:
                    t = max(t, same[self.cap_runway - 1] + 60.0)
                allt = sorted((pt for _, _, pt in placed), reverse=True)
                if len(allt) >= self.cap_airport:
                    t = max(t, allt[self.cap_airport - 1] + 60.0)
                if abs(t - t0) < TOL:
                    break
            placed.append((fid, rid, t))
        res = {fid: t for fid, _, t in placed}
        self._landing_cache[key] = res
        return res

    # ---- full build ----
    def build(self, order: tuple[str, ...], runways: dict[str, str], gates: dict[str, str],
              crews: dict[str, str], source: str = "fallback") -> Optional[RecoveryPlan]:
        """Return a plan, or None if the horizon is exceeded / no taxi route exists."""
        a = self.a
        land = self.landing_schedule(order, runways)
        seg_use: dict[str, list[tuple[float, float]]] = {}
        gate_end: dict[str, float] = {}
        crew_end: dict[str, float] = {}
        for gid, g in self.gates.items():
            gate_end[gid] = max([i.end_min for i in g.resident_occupancy], default=-1e9)
        plans: list[FlightPlan] = []
        for fid in sorted(order, key=lambda f: (land[f], f)):
            sf, rid, gid, cid = self.flights[fid], runways[fid], gates[fid], crews[fid]
            routes = self._routes.get((rid, gid))
            if not routes:
                return None
            best = None
            for segs, total in routes:
                s = land[fid] + a.runway_occupancy_min
                for _ in range(_MAX_ITERS):
                    s0 = s
                    s = max(s, gate_end[gid] + a.gate_buffer_min - total if gate_end[gid] > -1e8 else s)
                    off = 0.0
                    for sid in segs:
                        m = self.segs[sid].minutes
                        for b in self.segs[sid].blockages:
                            if s + off < b.end_min - TOL and s + off + m > b.start_min + TOL:
                                s = max(s, b.end_min - off)
                        for u0, u1 in seg_use.get(sid, []):
                            if s + off < u1 - TOL and s + off + m > u0 + TOL:
                                s = max(s, u1 - off)
                        off += m
                    if abs(s - s0) < TOL:
                        break
                if s + total > a.search_horizon_min:
                    continue
                if best is None or s + total < best[0] - TOL:
                    best = (s + total, s, segs, total)
            if best is None:
                return None
            gate_in, s, segs, total = best
            off = 0.0
            for sid in segs:
                seg_use.setdefault(sid, []).append((s + off, s + off + self.segs[sid].minutes))
                off += self.segs[sid].minutes
            gc = self.gcrews[cid]
            free = max(gc.available_from_min, crew_end[cid] + a.ground_crew_transfer_min if cid in crew_end else gc.available_from_min)
            ss = max(gate_in + a.deboard_min, free)
            se = max(ss + sf.service_task_min, gate_in + self.min_turn)
            gate_end[gid] = se
            crew_end[cid] = se
            plans.append(FlightPlan(flight_id=fid, runway_id=rid, landing_min=land[fid], taxi_route=segs,
                                    taxi_start_min=s, taxi_end_min=gate_in, gate_id=gid, gate_in_min=gate_in,
                                    service_start_min=ss, service_end_min=se, ground_crew_id=cid,
                                    flight_crew_id=sf.flight_crew_id))
        plans.sort(key=lambda p: [f.flight_id for f in self.sc.simulated.flights].index(p.flight_id))
        return make_plan(self.sc, source, plans)

    # ---- naive baseline ----
    def naive_plan(self) -> RecoveryPlan:
        """Nominal ETAs + original runway/gate/crew; shortest route; no conflict resolution."""
        a, out = self.a, []
        for sf in self.sc.simulated.flights:
            r = self.shortest_route(sf.original_runway_id, sf.original_gate_id)
            segs, total = r if r else ([], 0.0)
            t = sf.earliest_landing_min
            s = t + a.runway_occupancy_min
            gate_in = s + total
            ss = gate_in + a.deboard_min
            se = max(ss + sf.service_task_min, gate_in + self.min_turn)
            out.append(FlightPlan(flight_id=sf.flight_id, runway_id=sf.original_runway_id, landing_min=t,
                                  taxi_route=segs, taxi_start_min=s, taxi_end_min=gate_in,
                                  gate_id=sf.original_gate_id, gate_in_min=gate_in, service_start_min=ss,
                                  service_end_min=se, ground_crew_id=sf.original_ground_crew_id,
                                  flight_crew_id=sf.flight_crew_id))
        return make_plan(self.sc, "naive_baseline", out)
