import sqlite3
import os
from pathlib import Path
from datetime import datetime
from config.settings import settings

def get_db_path() -> Path:
    # Handle URL syntax like sqlite:///path
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        path_str = db_url[len("sqlite:///"):]
    else:
        path_str = "voice_calling.db"
    
    # If it is relative, make it absolute relative to project root
    db_path = Path(path_str)
    if not db_path.is_absolute():
        db_path = Path(settings.BASE_DIR) / db_path
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path

def get_connection():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create calls table
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
    
    # Create transcripts table
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

def create_call(call_sid: str, from_number: str, to_number: str, call_type: str, start_time: str = None) -> bool:
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
        # Call already exists
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
    cost_tts_char: int = 0
) -> bool:
    if not end_time:
        end_time = datetime.now().isoformat()
    
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Sum with existing cost metrics if any
        cursor.execute("SELECT cost_tokens, cost_stt_sec, cost_tts_char FROM calls WHERE call_sid = ?", (call_sid,))
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
    timestamp: str = None
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
        
        cursor.execute("SELECT * FROM transcripts WHERE call_sid = ? ORDER BY id ASC", (call_sid,))
        transcript_rows = cursor.fetchall()
        
        return {
            "call": dict(call_row),
            "transcript": [dict(row) for row in transcript_rows]
        }
    finally:
        conn.close()
