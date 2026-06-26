import asyncio
import sys
import os
import uuid
import logging
from datetime import datetime

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.storage.database import init_db, get_call_logs
from src.telephony.call_manager import call_manager
from src.orchestrator.pipeline import AudioPipeline

# Configure minimal console logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def play_pcm_on_windows(pcm_data: bytes, sample_rate: int):
    """
    Play raw PCM audio on Windows using winsound by wrapping it in WAV headers in memory.
    """
    if sys.platform != "win32" or not pcm_data:
        return
        
    import io
    import wave
    import winsound
    
    try:
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wav_file:
            wav_file.setnchannels(1)      # mono
            wav_file.setsampwidth(2)      # 16-bit PCM
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        # Play the in-memory WAV file (blocks until finished so conversation feels natural)
        winsound.PlaySound(wav_buf.getvalue(), winsound.SND_MEMORY)
    except Exception as e:
        logger.error(f"Failed to play audio locally: {e}")

def record_microphone_until_enter(sample_rate: int = 16000) -> bytes:
    """
    Record audio from the microphone until the user presses Enter.
    """
    try:
        import sounddevice as sd
        import numpy as np
        import sys
        
        chunks = []
        
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            chunks.append(indata.copy())
            
        # Start InputStream
        stream = sd.InputStream(
            samplerate=sample_rate, 
            channels=1, 
            dtype='int16', 
            callback=callback
        )
        
        with stream:
            print("🎤 [RECORDING ACTIVE] Speak now... Press [ENTER] when you are finished.")
            input()
            
        if not chunks:
            return b""
        
        audio_data = np.concatenate(chunks, axis=0)
        return audio_data.tobytes()
    except Exception as e:
        print(f"\n✗ Microphone error: {e}")
        return b""

# Mock websocket to print outgoing JSON events to console instead of network
class ConsoleMockWebSocket:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.stream_sid = None
        
    async def send_json(self, data: dict):
        event = data.get("event")
        if event == "media":
            # Just show a visual indicator of audio chunk
            pass
        elif event == "clear":
            print("\n🚨 [SYSTEM EVENT: Playback Cleared/Stopped by Barge-in] 🚨")
        elif event == "start":
            print(f"\n📞 [SYSTEM EVENT: Stream Started, Stream SID: {data.get('stream_sid')}] 📞")

async def run_simulation():
    print("=" * 60)
    print("      FUEL TRACKS VOICE AGENT - LOCAL TERMINAL SIMULATOR")
    print("=" * 60)
    
    # Initialize DB
    init_db()
    
    # Select call type
    print("\nSelect Call Direction/Purpose:")
    print("1. Outbound: Lead Follow-up (e.g., website enquiry)")
    print("2. Inbound: Support call (requires welcoming routing question first)")
    print("3. Outbound: Dealer Recruitment partner pitch")
    print("4. Outbound: New Product Marketing Pitch")
    choice = input("Enter choice (1-4): ").strip()
    
    call_sid = f"sim-{uuid.uuid4().hex[:12]}"
    from_num = "+919876543210"
    to_num = "+919000666914"
    
    context = {}
    if choice == "1":
        call_type = "lead_followup"
        context["customer_name"] = input("Customer Name [Default: Suresh Kumar]: ").strip() or "Suresh Kumar"
        context["product_interest"] = input("Product [Default: Fuel Sensor]: ").strip() or "Fuel Sensor"
    elif choice == "3":
        call_type = "dealer_recruitment"
    elif choice == "4":
        call_type = "marketing"
        context["customer_name"] = input("Customer Name [Default: Ramesh Kumar]: ").strip() or "Ramesh Kumar"
        context["product_interest"] = input("Product [Default: GPS Tracker Pro]: ").strip() or "GPS Tracker Pro"
    else:
        call_type = "inbound_routing"
        
    print(f"\nStarting call session Call SID: {call_sid}...")
    session = await call_manager.create_session(
        call_sid=call_sid,
        from_number=from_num,
        to_number=to_num,
        call_type=call_type
    )
    
    # Override pipeline's turn_manager.play_audio to show the text in console
    class ConsoleTurnManager:
        async def play_audio(self, session, pcm_data, sample_rate, text=None):
            session.is_playing = True
            session.barge_in_triggered = False
            if text:
                print(f"\n🤖 Agent: {text}")
            # Play the audio chunk asynchronously in a separate thread to avoid blocking the event loop
            await asyncio.to_thread(play_pcm_on_windows, pcm_data, sample_rate)
            session.is_playing = False
            
        async def stop_audio(self, session):
            session.is_playing = False
            print("\n🚨 [SYSTEM EVENT: Audio Playback Stopped] 🚨")
            
    # Setup WS simulator and pipeline
    mock_ws = ConsoleMockWebSocket(call_sid)
    stream_sid = f"stream-{uuid.uuid4().hex[:12]}"
    session.conversation_manager.context = context
    session.link_stream(stream_sid)
    mock_ws.stream_sid = stream_sid
    
    console_turn_mgr = ConsoleTurnManager()
    pipeline = AudioPipeline(session, console_turn_mgr, sample_rate=16000)
    
    # Trigger initial agent welcome response
    print("\n--- AGENT CONNECTS ---")
    await pipeline.trigger_initial_greeting()
    
    # Wait for the welcome greeting playback to finish
    while pipeline.session.is_playing or not pipeline.playback_queue.empty() or not pipeline.tts_queue.empty() or len(pipeline.active_tasks) > 0:
        await asyncio.sleep(0.1)
    
    # Loop conversations
    while True:
        print("\n" + "-" * 40)
        print("Speak/Type Customer Utterance (or 'hangup' to end call):")
        print(" -> Press [ENTER] (empty line) to speak via microphone")
        print(" -> Or type your message and press [ENTER]")
        customer_text = input("Customer: ").strip()
        
        if not customer_text:
            pcm_bytes = record_microphone_until_enter(16000)
            if not pcm_bytes:
                continue
            print("Transcribing your speech...")
            transcript, detected_lang, confidence, duration_sec = await pipeline.stt_client.transcribe(pcm_bytes, 16000)
            if transcript.strip():
                print(f"Customer (voice): {transcript}")
                if detected_lang == "te-IN":
                    customer_text = f"[te] {transcript}"
                elif detected_lang == "hi-IN":
                    customer_text = f"[hi] {transcript}"
                else:
                    customer_text = transcript
            else:
                print("Customer (voice): [No speech detected]")
                continue
            
        if customer_text.lower() in ["hangup", "exit", "quit", "bye"]:
            print("\nClosing call connection...")
            break
            
        # Detect language inputs (simulate STT auto-detect)
        # We allow typing language tags like [hi] or [te] prefixing words to simulate multilingual inputs
        # e.g., "[te] GPS sensor pricing entha?" -> te-IN
        detected_lang = "en-IN"
        text_clean = customer_text
        if customer_text.startswith("[hi]"):
            detected_lang = "hi-IN"
            text_clean = customer_text[4:].strip()
        elif customer_text.startswith("[te]"):
            detected_lang = "te-IN"
            text_clean = customer_text[4:].strip()
        elif customer_text.startswith("[en]"):
            detected_lang = "en-IN"
            text_clean = customer_text[4:].strip()
        else:
            # Simple keyword matching if no tag is provided
            # e.g. if typing Hindi words like 'kya', 'namaste', 'nahi', 'karo'
            hi_indicators = ["namaste", "kya", "achha", "haan", "nahi", "karo", "apna", "shubh", "shreya"]
            te_indicators = ["entha", "avunu", "ledhu", "andi", "garu", "namaskaram", "vachindi", "ekkada"]
            
            lower_words = text_clean.lower().split()
            hi_hits = sum(1 for w in lower_words if w in hi_indicators)
            te_hits = sum(1 for w in lower_words if w in te_indicators)
            
            if te_hits > hi_hits:
                detected_lang = "te-IN"
            elif hi_hits > te_hits:
                detected_lang = "hi-IN"
                
        # Simulate STT cost (assume 3 seconds for the utterance)
        session.stt_seconds_logged += 3.0
        
        # Feed text into conversation manager
        await session.add_customer_turn(
            text=text_clean,
            detected_language=detected_lang,
            confidence=0.95
        )
        
        # Display updated language profile
        profile = session.conversation_manager.language_profile
        print(f"\n📊 Rolling Language Profile:")
        print(f"   Primary: {profile.primary_language} | Secondary: {profile.secondary_language} "
              f"| Mix Ratio: {profile.mix_ratio}% | Formality: {profile.formality_level}")
        if session.call_type != call_type:
            print(f"   🔀 call_type dynamically classified as: {session.call_type}")
            
        # Trigger streaming response!
        print("🤖 Agent is thinking (streaming response)...")
        # Spin off the streaming LLM response task
        task = asyncio.create_task(pipeline._process_llm_stream())
        pipeline.active_tasks.add(task)
        task.add_done_callback(pipeline.active_tasks.discard)
        
        # Wait for the response to finish streaming and playing
        while pipeline.session.is_playing or not pipeline.playback_queue.empty() or not pipeline.tts_queue.empty() or len(pipeline.active_tasks) > 0:
            await asyncio.sleep(0.1)
            
    # Close pipeline and session
    pipeline.close()
    await call_manager.close_session(stream_sid, outcome="completed")
    print("\n" + "=" * 60)
    print("                      CALL TERMINATED")
    print("=" * 60)
    
    # Query database to show records
    print("\nRetrieving SQLite Database Records for review:")
    logs = get_call_logs(call_sid)
    call_record = logs.get("call", {})
    transcript_record = logs.get("transcript", [])
    
    print("\n[CALL METRICS LOGGED]:")
    print(f"  Call SID:    {call_record.get('call_sid')}")
    print(f"  Duration:    {call_record.get('duration'):.2f} seconds")
    print(f"  Outcome:     {call_record.get('outcome')}")
    print(f"  STT Secs:    {call_record.get('cost_stt_sec')}")
    print(f"  TTS Chars:   {call_record.get('cost_tts_char')}")
    print(f"  LLM Tokens:  {call_record.get('cost_tokens')}")
    
    print("\n[TRANSCRIPT LOGGED]:")
    for turn in transcript_record:
        lang_info = f" ({turn.get('detected_language')})" if turn.get("detected_language") else ""
        print(f"  {turn.get('role').upper()}{lang_info}: {turn.get('text')}")
        
    print("\nLocal simulation completed successfully!\n")

if __name__ == "__main__":
    asyncio.run(run_simulation())
