"""Append-only SQLite event log with idempotent writes.

`dedupe_key` is UNIQUE: re-logging the same logical event (a Streamlit rerun, a repeated click on a
deterministic action) is ignored instead of creating a duplicate row.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "events.db"


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    c = sqlite3.connect(path or DB_PATH)  # resolved at call time so DB_PATH can be redirected
    c.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
        scenario_id TEXT NOT NULL, event_type TEXT NOT NULL, actor TEXT NOT NULL,
        payload TEXT NOT NULL)""")
    cols = {r[1] for r in c.execute("PRAGMA table_info(events)")}
    if "dedupe_key" not in cols:  # migrate databases created by the first version
        c.execute("ALTER TABLE events ADD COLUMN dedupe_key TEXT")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_events_dedupe ON events(dedupe_key) WHERE dedupe_key IS NOT NULL")
    return c


def log_event(scenario_id: str, event_type: str, payload: dict, actor: str = "system",
              path: Optional[Path] = None, dedupe_key: Optional[str] = None) -> bool:
    """Insert one event. Returns False when skipped as a duplicate."""
    c = _conn(path)
    try:
        with c:
            cur = c.execute(
                "INSERT OR IGNORE INTO events (ts, scenario_id, event_type, actor, payload, dedupe_key) "
                "VALUES (?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), scenario_id, event_type, actor,
                 json.dumps(payload, default=str), dedupe_key))
            return cur.rowcount == 1
    finally:
        c.close()


def read_events(scenario_id: Optional[str] = None, limit: int = 100, path: Optional[Path] = None) -> list[dict]:
    q, args = "SELECT id, ts, scenario_id, event_type, actor, payload FROM events", []
    if scenario_id:
        q += " WHERE scenario_id = ?"
        args.append(scenario_id)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    c = _conn(path)
    try:
        cols = ["id", "ts", "scenario_id", "event_type", "actor", "payload"]
        return [dict(zip(cols, r)) for r in c.execute(q, args)]
    finally:
        c.close()
