"""
Storage Layer — PostgreSQL & SQLite Dual Backend
------------------------------------------------
Provides connection pooling for PostgreSQL (using asyncpg) and WAL mode support for SQLite.
Exposes both async (production) and sync (testing/legacy) interfaces,
tracking latency metrics alongside call and transcript records.
"""
import sqlite3
import logging
import os
import asyncio
from pathlib import Path
from datetime import datetime
from config.settings import settings

try:
    import aiosqlite
    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

try:
    import asyncpg
    _HAS_ASYNCPG = True
except ImportError:
    _HAS_ASYNCPG = False

logger = logging.getLogger(__name__)

# PostgreSQL connection pool instance
_pg_pool = None

# ---------------------------------------------------------------------------
# Backend Selection & Connection Helpers
# ---------------------------------------------------------------------------

def is_postgres() -> bool:
    """Check if the configured database URL is a PostgreSQL connection string."""
    url = settings.DATABASE_URL
    return url.startswith("postgresql") or url.startswith("postgres")


async def get_pg_pool():
    """Lazily initialize and return the asyncpg PostgreSQL connection pool."""
    global _pg_pool
    if not _HAS_ASYNCPG:
        raise ImportError("asyncpg package is required for PostgreSQL support.")
    if _pg_pool is None:
        url = settings.DATABASE_URL
        # Replace python driver scheme if present
        pg_url = url.replace("postgresql+asyncpg://", "postgresql://")
        logger.info("Initializing asyncpg connection pool with PostgreSQL...")
        _pg_pool = await asyncpg.create_pool(pg_url, min_size=5, max_size=20)
    return _pg_pool


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
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _convert_query(query: str) -> str:
    """
    Translate SQLite-style '?' placeholders to PostgreSQL-style '$1, $2, ...' format.
    No-op if using SQLite.
    """
    if not is_postgres():
        return query
    parts = query.split('?')
    res = []
    for i, part in enumerate(parts[:-1]):
        res.append(part)
        res.append(f"${i+1}")
    res.append(parts[-1])
    return "".join(res)


def _run_sync(coro):
    """Safely run async coroutines from a synchronous caller context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(coro))
            return future.result()
    else:
        return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Schema Init
# ---------------------------------------------------------------------------

async def ainit_db():
    """Async database schema initialization for PostgreSQL or SQLite."""
    if is_postgres():
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                call_sid VARCHAR(255) PRIMARY KEY,
                from_number VARCHAR(100),
                to_number VARCHAR(100),
                start_time VARCHAR(100),
                end_time VARCHAR(100),
                duration DOUBLE PRECISION DEFAULT 0.0,
                outcome VARCHAR(100) DEFAULT 'ongoing',
                call_type VARCHAR(100),
                cost_tokens INT DEFAULT 0,
                cost_stt_sec DOUBLE PRECISION DEFAULT 0.0,
                cost_tts_char INT DEFAULT 0,
                stt_latency_ms DOUBLE PRECISION DEFAULT 0.0,
                llm_latency_ms DOUBLE PRECISION DEFAULT 0.0,
                tts_latency_ms DOUBLE PRECISION DEFAULT 0.0
            )
            """)
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id SERIAL PRIMARY KEY,
                call_sid VARCHAR(255),
                role VARCHAR(100),
                text TEXT,
                detected_language VARCHAR(100),
                language_confidence DOUBLE PRECISION,
                timestamp VARCHAR(100),
                FOREIGN KEY (call_sid) REFERENCES calls (call_sid)
            )
            """)
            
            # Auto-migrate: check and add missing columns to existing tables
            columns = [row['column_name'] for row in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'calls'"
            )]
            if columns:
                if "cost_tokens" not in columns:
                    await conn.execute("ALTER TABLE calls ADD COLUMN cost_tokens INT DEFAULT 0")
                if "cost_stt_sec" not in columns:
                    await conn.execute("ALTER TABLE calls ADD COLUMN cost_stt_sec DOUBLE PRECISION DEFAULT 0.0")
                if "cost_tts_char" not in columns:
                    await conn.execute("ALTER TABLE calls ADD COLUMN cost_tts_char INT DEFAULT 0")
                if "stt_latency_ms" not in columns:
                    await conn.execute("ALTER TABLE calls ADD COLUMN stt_latency_ms DOUBLE PRECISION DEFAULT 0.0")
                if "llm_latency_ms" not in columns:
                    await conn.execute("ALTER TABLE calls ADD COLUMN llm_latency_ms DOUBLE PRECISION DEFAULT 0.0")
                if "tts_latency_ms" not in columns:
                    await conn.execute("ALTER TABLE calls ADD COLUMN tts_latency_ms DOUBLE PRECISION DEFAULT 0.0")
            
            t_columns = [row['column_name'] for row in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'transcripts'"
            )]
            if t_columns:
                if "detected_language" not in t_columns:
                    await conn.execute("ALTER TABLE transcripts ADD COLUMN detected_language VARCHAR(100)")
                if "language_confidence" not in t_columns:
                    await conn.execute("ALTER TABLE transcripts ADD COLUMN language_confidence DOUBLE PRECISION")
                    
        logger.info("PostgreSQL database schema initialized successfully.")
    else:
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("""
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
                cost_tts_char INTEGER DEFAULT 0,
                stt_latency_ms REAL DEFAULT 0.0,
                llm_latency_ms REAL DEFAULT 0.0,
                tts_latency_ms REAL DEFAULT 0.0
            )
            """)
            await db.execute("""
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
            await db.commit()
            
            # Auto-migrate: check and add missing columns to existing tables
            async with db.execute("PRAGMA table_info(calls)") as cursor:
                columns = [row[1] for row in await cursor.fetchall()]
            if columns:
                if "cost_tokens" not in columns:
                    await db.execute("ALTER TABLE calls ADD COLUMN cost_tokens INTEGER DEFAULT 0")
                if "cost_stt_sec" not in columns:
                    await db.execute("ALTER TABLE calls ADD COLUMN cost_stt_sec REAL DEFAULT 0.0")
                if "cost_tts_char" not in columns:
                    await db.execute("ALTER TABLE calls ADD COLUMN cost_tts_char INTEGER DEFAULT 0")
                if "stt_latency_ms" not in columns:
                    await db.execute("ALTER TABLE calls ADD COLUMN stt_latency_ms REAL DEFAULT 0.0")
                if "llm_latency_ms" not in columns:
                    await db.execute("ALTER TABLE calls ADD COLUMN llm_latency_ms REAL DEFAULT 0.0")
                if "tts_latency_ms" not in columns:
                    await db.execute("ALTER TABLE calls ADD COLUMN tts_latency_ms REAL DEFAULT 0.0")
            
            async with db.execute("PRAGMA table_info(transcripts)") as cursor:
                t_columns = [row[1] for row in await cursor.fetchall()]
            if t_columns:
                if "detected_language" not in t_columns:
                    await db.execute("ALTER TABLE transcripts ADD COLUMN detected_language TEXT")
                if "language_confidence" not in t_columns:
                    await db.execute("ALTER TABLE transcripts ADD COLUMN language_confidence REAL")
            await db.commit()
            
        logger.info("SQLite database schema initialized in WAL mode.")


def init_db():
    """Synchronous database schema initialization (backward-compatible)."""
    _run_sync(ainit_db())


# ---------------------------------------------------------------------------
# Synchronous functions (backward-compatible)
# ---------------------------------------------------------------------------

def create_call(
    call_sid: str, from_number: str, to_number: str,
    call_type: str, start_time: str = None
) -> bool:
    return _run_sync(acreate_call(call_sid, from_number, to_number, call_type, start_time))


def update_call_end(
    call_sid: str,
    end_time: str = None,
    duration: float = 0.0,
    outcome: str = "completed",
    cost_tokens: int = 0,
    cost_stt_sec: float = 0.0,
    cost_tts_char: int = 0,
    stt_latency_ms: float = 0.0,
    llm_latency_ms: float = 0.0,
    tts_latency_ms: float = 0.0
) -> bool:
    return _run_sync(aupdate_call_end(
        call_sid, end_time, duration, outcome,
        cost_tokens, cost_stt_sec, cost_tts_char,
        stt_latency_ms, llm_latency_ms, tts_latency_ms
    ))


def add_transcript_turn(
    call_sid: str,
    role: str,
    text: str,
    detected_language: str = None,
    language_confidence: float = None,
    timestamp: str = None,
) -> bool:
    return _run_sync(aadd_transcript_turn(
        call_sid, role, text, detected_language, language_confidence, timestamp
    ))


def get_call_logs(call_sid: str) -> dict:
    return _run_sync(aget_call_logs(call_sid))


def get_recent_calls(limit: int = 50) -> list:
    return _run_sync(aget_recent_calls(limit))


# ---------------------------------------------------------------------------
# Async functions (used by pipeline.py and CallManager for concurrency)
# ---------------------------------------------------------------------------

async def acreate_call(
    call_sid: str, from_number: str, to_number: str,
    call_type: str, start_time: str = None
) -> bool:
    """Async record call creation in either SQLite or PostgreSQL."""
    if not start_time:
        start_time = datetime.now().isoformat()

    sql = "INSERT INTO calls (call_sid, from_number, to_number, start_time, call_type) VALUES (?, ?, ?, ?, ?)"
    params = (call_sid, from_number, to_number, start_time, call_type)

    if is_postgres():
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(_convert_query(sql), *params)
                return True
            except asyncpg.UniqueViolationError:
                return False
    else:
        async with aiosqlite.connect(get_db_path()) as db:
            try:
                await db.execute(sql, params)
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
    stt_latency_ms: float = 0.0,
    llm_latency_ms: float = 0.0,
    tts_latency_ms: float = 0.0
) -> bool:
    """Async record final call parameters and turn latencies."""
    if not end_time:
        end_time = datetime.now().isoformat()

    if is_postgres():
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT cost_tokens, cost_stt_sec, cost_tts_char FROM calls WHERE call_sid = $1",
                call_sid
            )
            if row:
                cost_tokens += row["cost_tokens"] or 0
                cost_stt_sec += row["cost_stt_sec"] or 0.0
                cost_tts_char += row["cost_tts_char"] or 0

            result = await conn.execute(
                """
                UPDATE calls
                SET end_time = $1, duration = $2, outcome = $3, cost_tokens = $4, cost_stt_sec = $5, cost_tts_char = $6,
                    stt_latency_ms = $7, llm_latency_ms = $8, tts_latency_ms = $9
                WHERE call_sid = $10
                """,
                end_time, duration, outcome, cost_tokens, cost_stt_sec, cost_tts_char,
                stt_latency_ms, llm_latency_ms, tts_latency_ms, call_sid
            )
            return result.startswith("UPDATE") and " 0" not in result
    else:
        async with aiosqlite.connect(get_db_path()) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT cost_tokens, cost_stt_sec, cost_tts_char FROM calls WHERE call_sid = ?",
                (call_sid,)
            )
            row = await cursor.fetchone()
            if row:
                cost_tokens += row["cost_tokens"] or 0
                cost_stt_sec += row["cost_stt_sec"] or 0.0
                cost_tts_char += row["cost_tts_char"] or 0

            cursor = await db.execute(
                """
                UPDATE calls
                SET end_time = ?, duration = ?, outcome = ?, cost_tokens = ?, cost_stt_sec = ?, cost_tts_char = ?,
                    stt_latency_ms = ?, llm_latency_ms = ?, tts_latency_ms = ?
                WHERE call_sid = ?
                """,
                (end_time, duration, outcome, cost_tokens, cost_stt_sec, cost_tts_char,
                 stt_latency_ms, llm_latency_ms, tts_latency_ms, call_sid)
            )
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
    """Async insert a transcript conversation turn."""
    if not timestamp:
        timestamp = datetime.now().isoformat()

    sql = """
    INSERT INTO transcripts (call_sid, role, text, detected_language, language_confidence, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    params = (call_sid, role, text, detected_language, language_confidence, timestamp)

    if is_postgres():
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(_convert_query(sql), *params)
            return True
    else:
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(sql, params)
            await db.commit()
            return True


async def aget_call_logs(call_sid: str) -> dict:
    """Async retrieve all logs and transcripts associated with a Call SID."""
    if is_postgres():
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            call_row = await conn.fetchrow("SELECT * FROM calls WHERE call_sid = $1", call_sid)
            if not call_row:
                return {}
            transcript_rows = await conn.fetch("SELECT * FROM transcripts WHERE call_sid = $1 ORDER BY id ASC", call_sid)
            return {
                "call": dict(call_row),
                "transcript": [dict(r) for r in transcript_rows]
            }
    else:
        async with aiosqlite.connect(get_db_path()) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM calls WHERE call_sid = ?", (call_sid,))
            call_row = await cursor.fetchone()
            if not call_row:
                return {}
            cursor = await db.execute("SELECT * FROM transcripts WHERE call_sid = ? ORDER BY id ASC", (call_sid,))
            transcript_rows = await cursor.fetchall()
            return {
                "call": dict(call_row),
                "transcript": [dict(r) for r in transcript_rows]
            }


async def aget_recent_calls(limit: int = 50) -> list:
    """Async fetch the list of recent call records."""
    if is_postgres():
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM calls ORDER BY start_time DESC LIMIT $1", limit)
            return [dict(r) for r in rows]
    else:
        async with aiosqlite.connect(get_db_path()) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM calls ORDER BY start_time DESC LIMIT ?", (limit,))
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
