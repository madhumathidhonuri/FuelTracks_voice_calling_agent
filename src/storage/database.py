"""
Storage Layer — SQLite
------------------------
Provides both synchronous (legacy/test) and async (production) access to the
SQLite database.

Changes in this version:
  - Fix 4a: WAL mode + NORMAL synchronous mode enabled in init_db() for safe
             concurrent access across multiple asyncio tasks on the same call.
  - Fix 4b: Async versions of all write/read functions added using aiosqlite.
             Async functions are prefixed with `a` (e.g. acreate_call).
             Synchronous wrappers are kept for backward compatibility with
             call_manager.py and existing unit tests.
"""
import sqlite3
import logging
import os
from pathlib import Path
from datetime import datetime
from config.settings import settings

try:
    import aiosqlite
    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_db_path() -> Path:
    """Resolve the SQLite database file path from settings."""
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        path_str = db_url[len("sqlite:///"):]
    else:
        path_str = "voice_calling.db"

    db_path = Path(path_str)
    if not db_path.is_absolute():
        db_path = Path(settings.BASE_DIR) / db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection() -> sqlite3.Connection:
    """Open a synchronous SQLite connection with WAL mode enabled."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # WAL mode allows concurrent readers + 1 writer without blocking
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db():
    """
    Create tables if they don't exist and configure WAL journal mode.
    Safe to call multiple times (idempotent).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Enable WAL mode at schema init too (idempotent)
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS calls (
        call_sid TEXT PRIMARY KEY,
        from_number TEXT,
        to_number TEXT,
        start_time TEXT,
        end_time TEXT,
        duration REAL DEFAULT 0.0,
        outcome TEXT DEFAULT 'ongoing',
        call_type TEXT,
        cost_tokens INTEGER DEFAULT 0,
        cost_stt_sec REAL DEFAULT 0.0,
        cost_tts_char INTEGER DEFAULT 0
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_sid TEXT,
        role TEXT,
        text TEXT,
        detected_language TEXT,
        language_confidence REAL,
        timestamp TEXT,
        FOREIGN KEY (call_sid) REFERENCES calls (call_sid)
    )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized with WAL journal mode.")


# ---------------------------------------------------------------------------
# Synchronous functions (backward-compatible)
# ---------------------------------------------------------------------------

def create_call(
    call_sid: str, from_number: str, to_number: str,
    call_type: str, start_time: str = None
) -> bool:
    if not start_time:
        start_time = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO calls (call_sid, from_number, to_number, start_time, call_type)
        VALUES (?, ?, ?, ?, ?)
        """, (call_sid, from_number, to_number, start_time, call_type))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def update_call_end(
    call_sid: str,
    end_time: str = None,
    duration: float = 0.0,
    outcome: str = "completed",
    cost_tokens: int = 0,
    cost_stt_sec: float = 0.0,
    cost_tts_char: int = 0,
) -> bool:
    if not end_time:
        end_time = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT cost_tokens, cost_stt_sec, cost_tts_char FROM calls WHERE call_sid = ?",
            (call_sid,),
        )
        row = cursor.fetchone()
        if row:
            cost_tokens += row["cost_tokens"] or 0
            cost_stt_sec += row["cost_stt_sec"] or 0.0
            cost_tts_char += row["cost_tts_char"] or 0

        cursor.execute("""
        UPDATE calls
        SET end_time = ?, duration = ?, outcome = ?, cost_tokens = ?, cost_stt_sec = ?, cost_tts_char = ?
        WHERE call_sid = ?
        """, (end_time, duration, outcome, cost_tokens, cost_stt_sec, cost_tts_char, call_sid))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def add_transcript_turn(
    call_sid: str,
    role: str,
    text: str,
    detected_language: str = None,
    language_confidence: float = None,
    timestamp: str = None,
) -> bool:
    if not timestamp:
        timestamp = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO transcripts (call_sid, role, text, detected_language, language_confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (call_sid, role, text, detected_language, language_confidence, timestamp))
        conn.commit()
        return True
    finally:
        conn.close()


def get_call_logs(call_sid: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM calls WHERE call_sid = ?", (call_sid,))
        call_row = cursor.fetchone()
        if not call_row:
            return {}

        cursor.execute(
            "SELECT * FROM transcripts WHERE call_sid = ? ORDER BY id ASC", (call_sid,)
        )
        transcript_rows = cursor.fetchall()

        return {
            "call": dict(call_row),
            "transcript": [dict(row) for row in transcript_rows],
        }
    finally:
        conn.close()


def get_recent_calls(limit: int = 50) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT * FROM calls ORDER BY start_time DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Async functions (used by pipeline.py and API endpoints for concurrent calls)
# ---------------------------------------------------------------------------

async def acreate_call(
    call_sid: str, from_number: str, to_number: str,
    call_type: str, start_time: str = None
) -> bool:
    """Async version of create_call using aiosqlite."""
    if not _HAS_AIOSQLITE:
        return create_call(call_sid, from_number, to_number, call_type, start_time)

    if not start_time:
        start_time = datetime.now().isoformat()

    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        try:
            await db.execute("""
            INSERT INTO calls (call_sid, from_number, to_number, start_time, call_type)
            VALUES (?, ?, ?, ?, ?)
            """, (call_sid, from_number, to_number, start_time, call_type))
            await db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False


async def aupdate_call_end(
    call_sid: str,
    end_time: str = None,
    duration: float = 0.0,
    outcome: str = "completed",
    cost_tokens: int = 0,
    cost_stt_sec: float = 0.0,
    cost_tts_char: int = 0,
) -> bool:
    """Async version of update_call_end using aiosqlite."""
    if not _HAS_AIOSQLITE:
        return update_call_end(
            call_sid, end_time, duration, outcome,
            cost_tokens, cost_stt_sec, cost_tts_char
        )

    if not end_time:
        end_time = datetime.now().isoformat()

    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        cursor = await db.execute(
            "SELECT cost_tokens, cost_stt_sec, cost_tts_char FROM calls WHERE call_sid = ?",
            (call_sid,),
        )
        row = await cursor.fetchone()
        if row:
            cost_tokens += row["cost_tokens"] or 0
            cost_stt_sec += row["cost_stt_sec"] or 0.0
            cost_tts_char += row["cost_tts_char"] or 0

        cursor = await db.execute("""
        UPDATE calls
        SET end_time = ?, duration = ?, outcome = ?, cost_tokens = ?, cost_stt_sec = ?, cost_tts_char = ?
        WHERE call_sid = ?
        """, (end_time, duration, outcome, cost_tokens, cost_stt_sec, cost_tts_char, call_sid))
        await db.commit()
        return cursor.rowcount > 0


async def aadd_transcript_turn(
    call_sid: str,
    role: str,
    text: str,
    detected_language: str = None,
    language_confidence: float = None,
    timestamp: str = None,
) -> bool:
    """Async version of add_transcript_turn using aiosqlite."""
    if not _HAS_AIOSQLITE:
        return add_transcript_turn(
            call_sid, role, text, detected_language, language_confidence, timestamp
        )

    if not timestamp:
        timestamp = datetime.now().isoformat()

    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("""
        INSERT INTO transcripts (call_sid, role, text, detected_language, language_confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (call_sid, role, text, detected_language, language_confidence, timestamp))
        await db.commit()
        return True


async def aget_call_logs(call_sid: str) -> dict:
    """Async version of get_call_logs using aiosqlite."""
    if not _HAS_AIOSQLITE:
        return get_call_logs(call_sid)

    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        cursor = await db.execute("SELECT * FROM calls WHERE call_sid = ?", (call_sid,))
        call_row = await cursor.fetchone()
        if not call_row:
            return {}

        cursor = await db.execute(
            "SELECT * FROM transcripts WHERE call_sid = ? ORDER BY id ASC", (call_sid,)
        )
        transcript_rows = await cursor.fetchall()

        return {
            "call": dict(call_row),
            "transcript": [dict(row) for row in transcript_rows],
        }


async def aget_recent_calls(limit: int = 50) -> list:
    """Async version of get_recent_calls using aiosqlite."""
    if not _HAS_AIOSQLITE:
        return get_recent_calls(limit)

    async with aiosqlite.connect(get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        cursor = await db.execute(
            "SELECT * FROM calls ORDER BY start_time DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
