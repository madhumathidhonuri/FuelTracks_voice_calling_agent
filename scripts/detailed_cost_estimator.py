import sqlite3
import os
from pathlib import Path

# Rates
EXOTEL_MIN_RATE_INR = 1.00  # ₹1.00 per minute
SARVAM_STT_HOUR_RATE_INR = 30.00  # ₹30.00 per hour of audio
SARVAM_TTS_CHAR_RATE_INR = 30.00 / 10000  # ₹30.00 per 10,000 characters

# Exchange Rate
USD_TO_INR = 84.00

# Claude 3 Haiku Pricing (USD per million tokens)
CLAUDE_INPUT_USD_PM = 0.25
CLAUDE_OUTPUT_USD_PM = 1.25

# Gemini 1.5 Flash Pricing (USD per million tokens)
GEMINI_INPUT_USD_PM = 0.075
GEMINI_OUTPUT_USD_PM = 0.30

def calculate_costs():
    db_path = Path("voice_calling.db")
    if not db_path.exists():
        print("Database not found!")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get overall statistics
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
    
    total_calls = row["total_calls"] or 0
    total_duration_sec = row["total_duration_sec"] or 0.0
    total_tokens = row["total_tokens"] or 0
    total_stt_sec = row["total_stt_sec"] or 0.0
    total_tts_char = row["total_tts_char"] or 0

    if total_calls == 0:
        print("No calls found in the database.")
        conn.close()
        return

    # Calculate Telephony
    total_duration_min = total_duration_sec / 60.0
    telephony_cost_inr = total_duration_min * EXOTEL_MIN_RATE_INR

    # Calculate Sarvam STT
    stt_cost_inr = (total_stt_sec / 3600.0) * SARVAM_STT_HOUR_RATE_INR

    # Calculate Sarvam TTS
    tts_cost_inr = total_tts_char * SARVAM_TTS_CHAR_RATE_INR

    # For LLM cost, since input and output tokens are combined in cost_tokens,
    # let's assume a typical 80% input (system prompts + history) and 20% output (agent responses) split.
    # We will show both Claude 3 Haiku (Primary) and Gemini 1.5 Flash (Fallback/Alternative) scenarios.
    
    # Claude 3 Haiku Cost
    claude_blend_usd = (0.80 * CLAUDE_INPUT_USD_PM + 0.20 * CLAUDE_OUTPUT_USD_PM) / 1000000.0
    claude_cost_usd = total_tokens * claude_blend_usd
    claude_cost_inr = claude_cost_usd * USD_TO_INR

    # Gemini 1.5 Flash Cost
    gemini_blend_usd = (0.80 * GEMINI_INPUT_USD_PM + 0.20 * GEMINI_OUTPUT_USD_PM) / 1000000.0
    gemini_cost_usd = total_tokens * gemini_blend_usd
    gemini_cost_inr = gemini_cost_usd * USD_TO_INR

    print("==================================================")
    print("      MULTILINGUAL VOICE CALLING AGENT COST REPORT")
    print("==================================================")
    print(f"Total Simulated Calls logged: {total_calls}")
    print(f"Total Call Duration:         {total_duration_sec:.2f} seconds ({total_duration_min:.2f} minutes)")
    print(f"Total STT Audio Processed:   {total_stt_sec:.2f} seconds")
    print(f"Total TTS Characters Sent:   {total_tts_char} chars")
    print(f"Total LLM Tokens Used:       {total_tokens} tokens")
    print("--------------------------------------------------")
    print("COST BREAKDOWN (INR):")
    print(f"1. Telephony (Exotel @ Rs. {EXOTEL_MIN_RATE_INR:.2f}/min):     INR {telephony_cost_inr:.4f}")
    print(f"2. Speech-to-Text (Sarvam @ Rs. {SARVAM_STT_HOUR_RATE_INR:.2f}/hr): INR {stt_cost_inr:.4f}")
    print(f"3. Text-to-Speech (Sarvam @ Rs. 30/10k chars):   INR {tts_cost_inr:.4f}")
    print("--------------------------------------------------")
    print("LLM ALTERNATIVES (Estimated 80/20 Input/Output split):")
    print(f"Option A: Claude 3 Haiku (Primary)")
    print(f"   Rate: input $0.25/M, output $1.25/M (avg $0.45/M or INR 0.0378/k tokens)")
    print(f"   Total LLM Cost:  INR {claude_cost_inr:.4f} (${claude_cost_usd:.4f})")
    print(f"   Total Call Cost: INR {telephony_cost_inr + stt_cost_inr + tts_cost_inr + claude_cost_inr:.4f}")
    print(f"   Average / Call:  INR {(telephony_cost_inr + stt_cost_inr + tts_cost_inr + claude_cost_inr) / total_calls:.4f}")
    print()
    print(f"Option B: Gemini 1.5 Flash (Fallback / Lower Cost)")
    print(f"   Rate: input $0.075/M, output $0.30/M (avg $0.12/M or INR 0.0101/k tokens)")
    print(f"   Total LLM Cost:  INR {gemini_cost_inr:.4f} (${gemini_cost_usd:.4f})")
    print(f"   Total Call Cost: INR {telephony_cost_inr + stt_cost_inr + tts_cost_inr + gemini_cost_inr:.4f}")
    print(f"   Average / Call:  INR {(telephony_cost_inr + stt_cost_inr + tts_cost_inr + gemini_cost_inr) / total_calls:.4f}")
    print("==================================================")

    # Detailed statistics on character count in transcripts to verify token ratio
    cursor.execute("""
        SELECT role, SUM(LENGTH(text)) as total_chars
        FROM transcripts
        GROUP BY role
    """)
    print("\nTranscript Character Statistics (to check token ratio):")
    for r in cursor.fetchall():
        role = r["role"]
        chars = r["total_chars"]
        approx_tokens = chars / 4.0
        print(f"  {role.capitalize()}: {chars} characters (approx. {approx_tokens:.1f} tokens)")

    conn.close()

if __name__ == "__main__":
    calculate_costs()
