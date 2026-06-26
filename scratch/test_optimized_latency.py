import asyncio
import time
import httpx
import socket
from config.settings import settings
from src.stt.sarvam_stt import SarvamSTTClient
from src.tts.sarvam_tts import SarvamTTSClient
from src.conversation.llm_client import LLMClient
from src.orchestrator.pipeline import extract_sentences

async def run_pipeline_test():
    print("=" * 60)
    print("      END-TO-END VOICE AGENT OPTIMIZED LATENCY TEST")
    print("=" * 60)
    
    # Initialize clients
    stt_client = SarvamSTTClient()
    llm_client = LLMClient()
    tts_client = SarvamTTSClient()
    
    # 1. Prepare dummy PCM audio (1.5 seconds of silence/speech)
    # 16000 Hz, 16-bit mono = 32000 bytes/sec. 1.5s = 48000 bytes.
    pcm_data = b"\x00" * 48000
    
    total_start = time.time()
    
    # Phase 1: STT
    print("\n[Phase 1] Running STT Transcription...")
    stt_start = time.time()
    transcript, detected_lang, confidence, duration_sec = await stt_client.transcribe(pcm_data, 16000)
    stt_duration = time.time() - stt_start
    print(f"  STT Finished in {stt_duration:.3f}s")
    print(f"  Transcript: '{transcript.encode('ascii', errors='replace').decode('ascii')}'")
    
    # Prepare prompt/history
    system_prompt = (
        "You are Shreya, a helpful customer agent at Fuel Tracks. "
        "Keep responses very short (one short sentence)."
    )
    history = [{"role": "customer", "content": "Hello, is this Fuel Tracks?"}]
    
    # Phase 2 & 3: LLM streaming to first sentence and TTS of that first sentence
    print("\n[Phase 2 & 3] Initializing LLM stream & first sentence TTS...")
    llm_start = time.time()
    
    first_sentence = None
    text_buffer = ""
    llm_stream = llm_client.generate_response_stream(system_prompt, history)
    
    first_sentence_time = None
    tts_duration = None
    
    async for chunk, usage in llm_stream:
        if chunk:
            text_buffer += chunk
            sentences, text_buffer = extract_sentences(text_buffer)
            if sentences and first_sentence is None:
                first_sentence = sentences[0]
                first_sentence_time = time.time() - llm_start
                print(f"  LLM yielded first sentence in {first_sentence_time:.3f}s: '{first_sentence.encode('ascii', errors='replace').decode('ascii')}'")
                
                # Start TTS for the first sentence immediately
                print("  Starting concurrent TTS synthesis for first sentence...")
                tts_start = time.time()
                pcm_bytes, rate, chars = await tts_client.text_to_speech(first_sentence, language_code="en-IN")
                tts_duration = time.time() - tts_start
                print(f"  TTS finished synthesizing first sentence in {tts_duration:.3f}s")
                break
                
    total_latency = time.time() - total_start
    print("\n" + "=" * 60)
    print("                      LATENCY METRICS")
    print("=" * 60)
    print(f"  1. Speech-to-Text (STT) Latency:    {stt_duration:.3f}s")
    if first_sentence_time is not None:
        print(f"  2. Time-to-First-Sentence (LLM):   {first_sentence_time:.3f}s")
    else:
        print(f"  2. Time-to-First-Sentence (LLM):   Failed/None")
    if tts_duration is not None:
        print(f"  3. Text-to-Speech (TTS) Latency:    {tts_duration:.3f}s")
    else:
        print(f"  3. Text-to-Speech (TTS) Latency:    Failed/None")
        
    print(f"  TOTAL RESPONSE TIME (STT + LLM + TTS): {total_latency:.3f}s")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_pipeline_test())
