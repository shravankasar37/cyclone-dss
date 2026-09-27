"""SQLite log of every prediction the API serves (history panel + audit trail)."""
import json
import os
import sqlite3
from datetime import datetime, timezone

from .config import DB_PATH


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, kind TEXT, subject TEXT, valid_time TEXT, summary TEXT)""")
    return c


def log(kind, subject, valid_time, summary: dict):
    try:
        with _conn() as c:
            c.execute("INSERT INTO predictions (created_at, kind, subject, valid_time, summary) VALUES (?,?,?,?,?)",
                      (datetime.now(timezone.utc).isoformat(timespec="seconds"), kind, subject, valid_time,
                       json.dumps(summary)))
    except sqlite3.Error:
        pass


def recent(limit=30):
    with _conn() as c:
        rows = c.execute("SELECT id, created_at, kind, subject, valid_time, summary FROM predictions "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r[0], "created_at": r[1], "kind": r[2], "subject": r[3], "valid_time": r[4],
             "summary": json.loads(r[5])} for r in rows]
