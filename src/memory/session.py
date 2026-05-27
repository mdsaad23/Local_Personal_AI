"""
Session lifecycle management.

Each conversation is a session with a unique ID. Sessions are persisted in
SQLite so the chat history survives restarts and feeds into episodic memory
extraction.
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from typing import Any

from config.settings import MEMORY_DB_PATH

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                title       TEXT,
                started_at  REAL NOT NULL,
                ended_at    REAL,
                turn_count  INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                has_image   INTEGER DEFAULT 0,
                timestamp   REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);
        """)
        # Migrate: add columns if upgrading from older schema
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN has_image INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass


_init_db()


def new_session() -> str:
    """Create a new session and return its ID."""
    sid = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, started_at) VALUES (?, ?)",
            (sid, time.time()),
        )
    logger.info("New session: %s", sid)
    return sid


def add_message(session_id: str, role: str, content: str, has_image: bool = False) -> None:
    """Append a message to a session. Auto-sets session title from first user message."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, has_image, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, int(has_image), time.time()),
        )
        conn.execute(
            "UPDATE sessions SET turn_count = turn_count + 1 WHERE session_id = ?",
            (session_id,),
        )
        # Auto-set title from the first user message
        if role == "user":
            existing = conn.execute(
                "SELECT title FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if existing and not existing["title"]:
                title = content[:60].strip()
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE session_id = ?",
                    (title, session_id),
                )


def get_messages(session_id: str) -> list[dict[str, Any]]:
    """Return all messages for a session, ordered by timestamp."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, has_image, timestamp FROM messages "
            "WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_sessions() -> list[dict[str, Any]]:
    """
    Return all sessions ordered by most recent activity, with preview metadata.
    Used by the chat history sidebar.
    """
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                s.session_id,
                COALESCE(s.title, 'New conversation') AS title,
                s.started_at,
                s.turn_count,
                m.content AS last_message,
                m.timestamp AS last_activity
            FROM sessions s
            LEFT JOIN messages m ON m.id = (
                SELECT id FROM messages
                WHERE session_id = s.session_id
                ORDER BY timestamp DESC LIMIT 1
            )
            ORDER BY COALESCE(m.timestamp, s.started_at) DESC
        """).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    """Delete a session and all its messages."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    logger.info("Deleted session %s", session_id)


def end_session(session_id: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
            (time.time(), session_id),
        )


def get_recent_sessions(limit: int = 10) -> list[dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
