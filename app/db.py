"""Thin sqlite3 helpers. No ORM on purpose — the SQL is part of the lesson.

Every request/function grabs a fresh connection via get_conn(); SQLite with WAL
handles this fine at prototype scale.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "platform.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA_PATH.read_text())
        # sim_state is a singleton row; create it if absent
        conn.execute("INSERT OR IGNORE INTO sim_state (id) VALUES (1)")


def query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(sql, params).fetchall()


def query_one(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(sql, params).fetchone()


def execute(sql: str, params: tuple = ()) -> int:
    """Run a write statement; returns lastrowid."""
    with get_conn() as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid
