import asyncio
import sys
import os
import uuid
import logging

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force Windows console to use UTF-8 so Telugu and emojis print properly
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.settings import settings
from src.storage.database import init_db, get_call_logs
from src.telephony.call_manager import call_manager
from src.tts.sarvam_tts import SarvamTTSClient

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_telugu_agent")

async def test_telugu_conversation():
    print("=" * 70)
    print("       TESTING FUELTRACKS VOICE CALLING AGENT - TELUGU SIMULATION")
    print("=" * 70)

    # 1. Initialize SQLite Database
    init_db()

    # 2. Setup simulated session info
    call_sid = f"telugu-test-{uuid.uuid4().hex[:8]}"
    from_num = "+919876543210"
    to_num = "+919513886363"
    call_type = "lead_followup"  # Outbound lead follow-up campaign

    print(f"\n[1] Creating call session: Call SID = {call_sid}")
    session = await call_manager.create_session(
        call_sid=call_sid,
        from_number=from_num,
        to_number=to_num,
        call_type=call_type
    )
    # Link stream to initialize prompt variables and load prompts for call type
    session.link_stream(f"stream-{call_sid}")

    # Add customer context
    session.conversation_manager.context["customer_name"] = "Srinivas Rao"
    session.conversation_manager.context["product_interest"] = "GPS Tracker"

    # 3. Initialize clients
    tts_client = SarvamTTSClient()

    # 4. Trigger initial greeting
    print("\n[2] Triggering Initial greeting (Agent welcomes customer)...")
    initial_greeting = (
        "హలో శ్రీనివాస్ రావు గారు, నేను ఫ్యూయల్ ట్రాక్స్ నుండి మాట్లాడుతున్నాను. "
        "మీరు మా వెబ్‌సైట్‌లో జీపీఎస్ ట్రాకర్ కోసం చూశారని తెలిసింది. "
        "దీని గురించి మీకు ఏమైనా సమాచారం కావాలా?"
    )
    print(f"[Agent]: {initial_greeting}")

    # Add agent's turn to dialog history and database
    session.conversation_manager.history.append({"role": "agent", "content": initial_greeting})
    from src.storage.database import aadd_transcript_turn
    await aadd_transcript_turn(session.call_sid, "agent", initial_greeting)

    # 5. Simulated turns in Telugu
    simulated_turns = [
        {
            "input": "అవునండి, మా బైక్‌కి జీపీఎస్ ట్రాకర్ అమర్చాలి. దీని ధర ఎంత ఉంటుంది?",
            "expected_context": "Asking about price of GPS tracker for bike"
        },
        {
            "input": "మరి దీనికి మంత్లీ సబ్‌స్క్రిప్షన్ లేదా ఇతర ఛార్జీలు ఉంటాయా?",
            "expected_context": "Asking about monthly subscription fee"
        },
        {
            "input": "సరే అండి, ఇన్స్టాలేషన్ ఎలా చేస్తారు? మా ఇంటి దగ్గరకే వస్తారా?",
            "expected_context": "Asking about installation process / door step service"
        },
        {
            "input": "చాలా సంతోషం. నేను రేపు కాల్ చేసి ఆర్డర్ కన్ఫర్మ్ చేస్తాను. ధన్యవాదాలు.",
            "expected_context": "Confirming call completion and hanging up"
        }
    ]

    for idx, turn in enumerate(simulated_turns, 1):
        print("\n" + "-" * 50)
        print(f"Turn {idx}: Customer says (Telugu): {turn['input']}")
        print(f"Context: {turn['expected_context']}")

        # Feed the customer turn to conversation manager
        await session.add_customer_turn(
            text=turn["input"],
            detected_language="te-IN",
            confidence=0.99
        )

        # Show updated language profile
        profile = session.conversation_manager.language_profile
        print(f"[Language Profile]:")
        print(f"   Primary Lang: {profile.primary_language} | Secondary: {profile.secondary_language}")
        print(f"   Mix Ratio: {profile.mix_ratio}% | Formality: {profile.formality_level}")

        # Generate Gemini Response via ConversationManager
        print("[Agent]: Thinking...")
        response_text, tokens = await session.conversation_manager.generate_agent_response()

        print(f"[Agent Response]: {response_text}")
        print(f"   Tokens used: {tokens}")

        # Test Text-to-Speech synthesis for the response
        if settings.SARVAM_API_KEY and "mock_" not in settings.SARVAM_API_KEY:
            try:
                print("[TTS] Generating Telugu audio response via Sarvam TTS...")
                pcm_data, rate, size = await tts_client.text_to_speech(
                    text=response_text,
                    language_code="te-IN"
                )
                print(f"   TTS Synthesized audio: {len(pcm_data)} bytes at {rate}Hz")
            except Exception as e:
                print(f"   TTS Error: {e}")
        else:
            print("[TTS] (Sarvam TTS skipped - using mock API key)")

    # 6. Close the session
    print("\n[3] Closing call session and saving stats...")
    await call_manager.close_session(session.stream_sid or f"stream-{call_sid}", outcome="completed")

    # 7. Query database to verify record
    print("\n[4] Querying SQLite Database Records:")
    logs = get_call_logs(call_sid)
    call_record = logs.get("call", {})
    transcript_record = logs.get("transcript", [])

    print(f"  Call SID:      {call_record.get('call_sid')}")
    print(f"  Call Type:     {call_record.get('call_type')}")
    print(f"  Duration:      {call_record.get('duration'):.2f} seconds")
    print(f"  Outcome:       {call_record.get('outcome')}")
    print(f"  Total Tokens:  {call_record.get('cost_tokens')}")

    print("\n  Recorded Transcript:")
    for t_turn in transcript_record:
        lang_str = f" [{t_turn.get('detected_language')}]" if t_turn.get("detected_language") else ""
        print(f"    - {t_turn.get('role').upper()}{lang_str}: {t_turn.get('text')}")

    print("\n" + "=" * 70)
    print("                     TELUGU SIMULATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_telugu_conversation())
