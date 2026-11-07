"""Customer-visible conversation ledger, separate from internal execution traces."""

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "support_conversations.db"


def initialize() -> None:
    with sqlite3.connect(DB) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, sender TEXT NOT NULL,
                content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS conversation_session
                ON conversation_messages(session_id, id);
            CREATE TABLE IF NOT EXISTS conversation_modes (
                session_id TEXT PRIMARY KEY, human INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'olist-demo',
                user_id TEXT NOT NULL DEFAULT 'demo-customer',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(conversation_messages)").fetchall()}
        if "case_id" not in columns:
            conn.execute("ALTER TABLE conversation_messages ADD COLUMN case_id TEXT")
        if "event_key" not in columns:
            conn.execute("ALTER TABLE conversation_messages ADD COLUMN event_key TEXT")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS conversation_event_key "
            "ON conversation_messages(event_key) WHERE event_key IS NOT NULL"
        )


def ensure_conversation(session_id: str, tenant_id: str, user_id: str) -> None:
    initialize()
    with sqlite3.connect(DB) as conn:
        conn.execute(
            """
            INSERT INTO conversations(session_id, tenant_id, user_id)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, tenant_id, user_id),
        )


def conversation_owner(session_id: str) -> dict | None:
    initialize()
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT session_id, tenant_id, user_id, created_at, updated_at "
            "FROM conversations WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def append_message(
    session_id: str,
    sender: str,
    content: str,
    *,
    case_id: str | None = None,
    event_key: str | None = None,
) -> bool:
    initialize()
    with sqlite3.connect(DB) as conn:
        if event_key:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO conversation_messages"
                "(session_id, sender, content, case_id, event_key) VALUES (?, ?, ?, ?, ?)",
                (session_id, sender, content, case_id, event_key),
            )
            return cursor.rowcount > 0
        conn.execute(
            "INSERT INTO conversation_messages(session_id, sender, content, case_id) "
            "VALUES (?, ?, ?, ?)",
            (session_id, sender, content, case_id),
        )
        return True


def messages_with_case(session_id: str) -> list[dict]:
    initialize()
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, sender, content, created_at, case_id, event_key "
            "FROM conversation_messages WHERE session_id = ? ORDER BY id DESC LIMIT 100",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def messages(session_id: str) -> list[dict]:
    initialize()
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, sender, content, created_at FROM conversation_messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT 100", (session_id,),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def human_mode(session_id: str) -> bool:
    initialize()
    with sqlite3.connect(DB) as conn:
        row = conn.execute(
            "SELECT human FROM conversation_modes WHERE session_id = ?", (session_id,),
        ).fetchone()
    return bool(row and row[0])


def recent_sessions() -> list[dict]:
    initialize()
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT session_id, MAX(created_at) AS updated_at FROM conversation_messages "
            "GROUP BY session_id ORDER BY MAX(id) DESC LIMIT 50"
        ).fetchall()
    return [dict(row) for row in rows]


def set_human_mode(session_id: str, enabled: bool) -> None:
    initialize()
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO conversation_modes VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET human=excluded.human",
            (session_id, int(enabled)),
        )
