import sqlite3
from pathlib import Path

# Rates
EXOTEL_MIN_RATE_INR = 1.00          # ₹1.00 per minute
SARVAM_STT_HOUR_RATE_INR = 30.00   # ₹30.00 per hour of audio
SARVAM_TTS_CHAR_RATE_INR = 30.00 / 10000  # ₹30.00 per 10,000 characters

# Exchange Rate
USD_TO_INR = 84.00

# Groq Pricing (USD per million tokens) — as of 2026
# llama-3.3-70b-versatile  : $0.59 input / $0.79 output
# llama-3.1-8b-instant     : $0.05 input / $0.08 output
GROQ_70B_INPUT_USD_PM  = 0.59
GROQ_70B_OUTPUT_USD_PM = 0.79
GROQ_8B_INPUT_USD_PM   = 0.05
GROQ_8B_OUTPUT_USD_PM  = 0.08


def calculate_costs():
    db_path = Path("voice_calling.db")
    if not db_path.exists():
        print("Database not found!")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Overall statistics
    cursor.execute("""
        SELECT
            COUNT(*) as total_calls,
            SUM(duration) as total_duration_sec,
            SUM(cost_tokens) as total_tokens,
            SUM(cost_stt_sec) as total_stt_sec,
            SUM(cost_tts_char) as total_tts_char
        FROM calls
    """)
    row = cursor.fetchone()

    total_calls        = row["total_calls"] or 0
    total_duration_sec = row["total_duration_sec"] or 0.0
    total_tokens       = row["total_tokens"] or 0
    total_stt_sec      = row["total_stt_sec"] or 0.0
    total_tts_char     = row["total_tts_char"] or 0

    if total_calls == 0:
        print("No calls found in the database.")
        conn.close()
        return

    # Telephony
    total_duration_min  = total_duration_sec / 60.0
    telephony_cost_inr  = total_duration_min * EXOTEL_MIN_RATE_INR

    # Sarvam STT
    stt_cost_inr = (total_stt_sec / 3600.0) * SARVAM_STT_HOUR_RATE_INR

    # Sarvam TTS
    tts_cost_inr = total_tts_char * SARVAM_TTS_CHAR_RATE_INR

    # LLM cost — 80% input / 20% output split assumption
    blend_70b_usd = (0.80 * GROQ_70B_INPUT_USD_PM + 0.20 * GROQ_70B_OUTPUT_USD_PM) / 1_000_000
    blend_8b_usd  = (0.80 * GROQ_8B_INPUT_USD_PM  + 0.20 * GROQ_8B_OUTPUT_USD_PM)  / 1_000_000

    groq_70b_cost_usd = total_tokens * blend_70b_usd
    groq_70b_cost_inr = groq_70b_cost_usd * USD_TO_INR

    groq_8b_cost_usd  = total_tokens * blend_8b_usd
    groq_8b_cost_inr  = groq_8b_cost_usd * USD_TO_INR

    base_cost_inr = telephony_cost_inr + stt_cost_inr + tts_cost_inr

    print("==================================================")
    print("      MULTILINGUAL VOICE CALLING AGENT COST REPORT")
    print("==================================================")
    print(f"Total Calls logged:          {total_calls}")
    print(f"Total Call Duration:         {total_duration_sec:.2f}s ({total_duration_min:.2f} min)")
    print(f"Total STT Audio Processed:   {total_stt_sec:.2f}s")
    print(f"Total TTS Characters Sent:   {total_tts_char} chars")
    print(f"Total LLM Tokens Used:       {total_tokens} tokens")
    print("--------------------------------------------------")
    print("COST BREAKDOWN (INR):")
    print(f"1. Telephony (Exotel @ ₹{EXOTEL_MIN_RATE_INR:.2f}/min):      INR {telephony_cost_inr:.4f}")
    print(f"2. Speech-to-Text (Sarvam @ ₹{SARVAM_STT_HOUR_RATE_INR:.2f}/hr): INR {stt_cost_inr:.4f}")
    print(f"3. Text-to-Speech (Sarvam @ ₹30/10k chars):    INR {tts_cost_inr:.4f}")
    print("--------------------------------------------------")
    print("LLM SCENARIOS (Groq, 80/20 input/output split):")
    print()
    print(f"Option A — llama-3.3-70b-versatile (Primary, High Quality)")
    print(f"   Rate: $0.59/M input, $0.79/M output")
    print(f"   Total LLM Cost:  INR {groq_70b_cost_inr:.4f}  (${groq_70b_cost_usd:.4f})")
    print(f"   Total Call Cost: INR {base_cost_inr + groq_70b_cost_inr:.4f}")
    print(f"   Average / Call:  INR {(base_cost_inr + groq_70b_cost_inr) / total_calls:.4f}")
    print()
    print(f"Option B — llama-3.1-8b-instant (Fallback, Ultra-Fast)")
    print(f"   Rate: $0.05/M input, $0.08/M output")
    print(f"   Total LLM Cost:  INR {groq_8b_cost_inr:.4f}  (${groq_8b_cost_usd:.4f})")
    print(f"   Total Call Cost: INR {base_cost_inr + groq_8b_cost_inr:.4f}")
    print(f"   Average / Call:  INR {(base_cost_inr + groq_8b_cost_inr) / total_calls:.4f}")
    print("==================================================")

    # Character statistics
    cursor.execute("""
        SELECT role, SUM(LENGTH(text)) as total_chars
        FROM transcripts
        GROUP BY role
    """)
    print("\nTranscript Character Statistics (to verify token ratio):")
    for r in cursor.fetchall():
        chars = r["total_chars"]
        approx_tokens = chars / 4.0
        print(f"  {r['role'].capitalize()}: {chars} chars (≈ {approx_tokens:.1f} tokens)")

    conn.close()


if __name__ == "__main__":
    calculate_costs()
