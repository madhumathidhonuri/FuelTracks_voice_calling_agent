import logging
import asyncio
import re
from typing import Tuple, List, Set
from src.telephony.call_manager import CallSession
from src.audio.vad import VoiceActivityDetector, calculate_rms
from src.stt.sarvam_stt import SarvamSTTClient
from src.tts.sarvam_tts import SarvamTTSClient
from src.orchestrator.turn_manager import TurnManager
from src.storage.database import add_transcript_turn

logger = logging.getLogger(__name__)

def extract_sentences(text_buffer: str) -> Tuple[List[str], str]:
    """
    Extract complete sentences/clauses from the stream buffer using standard terminators
    (., ?, !, or native Indic full-stops like । and newlines).
    Returns:
        Tuple[List[extracted_sentences], remaining_suffix]
    """
    # Match text ending with terminators (. ! ? \n or ।)
    matches = re.findall(r'([^.!?\n।]+[.!?\n।]+)', text_buffer)
    if not matches:
        return [], text_buffer
        
    total_matched_len = sum(len(m) for m in matches)
    remaining = text_buffer[total_matched_len:]
    sentences = [m.strip() for m in matches if m.strip()]
    return sentences, remaining

class AudioPipeline:
    def __init__(self, session: CallSession, turn_manager: TurnManager, sample_rate: int = 16000):
        self.session = session
        self.turn_manager = turn_manager
        self.sample_rate = sample_rate
        
        # Audio buffer for accumulating customer speech
        self.audio_buffer = bytearray()
        
        # VAD settings: 800 RMS threshold, 1.0s silence timeout
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate, 
            threshold=800.0, 
            silence_timeout_ms=1000
        )
        
        # API Clients
        self.stt_client = SarvamSTTClient()
        self.tts_client = SarvamTTSClient()
        
        # Concurrency: queues to decouple STT -> LLM stream -> TTS -> Playback
        self.stt_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()
        self.playback_queue = asyncio.Queue()
        
        # Track active generation and synthesis tasks (for cancellation on barge-in)
        self.active_tasks: Set[asyncio.Task] = set()
        
        # Spin up background workers
        self.workers = [
            asyncio.create_task(self._stt_worker()),
            asyncio.create_task(self._tts_worker()),
            asyncio.create_task(self._playback_worker())
        ]
        
        # Barge-in counter
        self.barge_in_counter = 0

    def close(self):
        """
        Cancel all background workers and active tasks to prevent resource leaks.
        """
        logger.info(f"Closing AudioPipeline for call {self.session.call_sid}")
        for worker in self.workers:
            if not worker.done():
                worker.cancel()
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()
        self.active_tasks.clear()

    async def trigger_initial_greeting(self):
        """
        Generate and play the agent's first welcome turn using LLM streaming.
        """
        try:
            logger.info(f"Triggering initial welcome streaming for call {self.session.call_sid}...")
            # Spin off the streaming LLM response task directly (since history is empty)
            task = asyncio.create_task(self._process_llm_stream())
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.discard)
        except Exception as e:
            logger.exception(f"Error during initial welcome trigger: {e}")

    async def process_inbound_audio(self, audio_chunk: bytes):
        """
        Handle incoming PCM audio chunks from Exotel.
        """
        self.session.touch()
        
        # 1. Check for barge-in if the agent is speaking
        if self.session.is_playing:
            rms = calculate_rms(audio_chunk)
            if rms > 1200.0:  # Barge-in threshold
                self.barge_in_counter += 1
                if self.barge_in_counter >= 2:  # Sustained speech (~200ms)
                    logger.info(f"Barge-in detected on call {self.session.call_sid}! Interrupting playback.")
                    self.session.barge_in_triggered = True
                    await self.turn_manager.stop_audio(self.session)
                    
                    # Stop active generation/synthesis and flush queues
                    self._clear_realtime_queues()
                    
                    self.audio_buffer.clear()
                    self.vad.reset()
                    self.barge_in_counter = 0
            else:
                self.barge_in_counter = max(0, self.barge_in_counter - 1)
            return
            
        # 2. Run VAD if the agent is listening
        is_speech_active, speech_stopped_now = self.vad.process_chunk(audio_chunk)
        
        if is_speech_active:
            self.audio_buffer.extend(audio_chunk)
            
        if speech_stopped_now:
            duration_sec = len(self.audio_buffer) / (self.sample_rate * 2)
            logger.info(f"Silence detected. Enqueuing turn of duration {duration_sec:.2f}s for STT")
            utterance_audio = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            self.vad.reset()
            
            # Put utterance on STT queue
            self.stt_queue.put_nowait(utterance_audio)

    def _clear_realtime_queues(self):
        """
        Clears all pending sentence and audio playback queues, and cancels current API requests.
        """
        logger.info(f"Clearing real-time queues for call {self.session.call_sid} due to barge-in.")
        
        # Cancel all active token streaming / synthesis tasks
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()
        self.active_tasks.clear()
        
        # Flush tts_queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break
                
        # Flush playback_queue
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

    async def _stt_worker(self):
        """
        Worker task to transcribe complete speech utterances sequentially.
        """
        while True:
            try:
                audio_data = await self.stt_queue.get()
                
                # Transcribe
                transcript, detected_lang, confidence, duration_sec = await self.stt_client.transcribe(
                    audio_data, 
                    self.sample_rate
                )
                self.session.stt_seconds_logged += duration_sec
                logger.info(f"STT: '{transcript}' [Lang: {detected_lang}, Conf: {confidence:.2f}, Dur: {duration_sec:.2f}s]")
                
                if not transcript.strip():
                    self.stt_queue.task_done()
                    continue
                    
                # Append customer turn to history
                await self.session.add_customer_turn(
                    text=transcript,
                    detected_language=detected_lang,
                    confidence=confidence
                )
                
                # Spin off streaming LLM response generation
                task = asyncio.create_task(self._process_llm_stream())
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
                
                self.stt_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in STT worker: {e}")

    async def _process_llm_stream(self):
        """
        Task to stream LLM response tokens, split on sentence boundaries, and enqueue sentences.
        """
        try:
            purpose_map = {
                "lead_followup": "lead follow-up enquiry",
                "support": "customer support",
                "dealer_recruitment": "dealer partner recruitment",
                "marketing": "marketing pitch and product introduction",
                "inbound_routing": "initial welcome routing"
            }
            call_purpose = purpose_map.get(self.session.call_type, "customer service")
            
            # Build system prompt dynamically
            system_prompt = self.session.conversation_manager.get_company_specific_instructions()
            from src.conversation.prompt_builder import build_system_prompt
            system_prompt = build_system_prompt(
                company_name=self.session.conversation_manager.company_name,
                call_purpose=call_purpose,
                language_profile=self.session.conversation_manager.language_profile,
                company_specific_instructions=system_prompt
            )
            
            text_buffer = ""
            full_response = ""
            final_token_usage = None
            
            # Stream tokens
            async for chunk, usage in self.session.conversation_manager.llm_client.generate_response_stream(
                system_prompt, 
                self.session.conversation_manager.history
            ):
                if chunk:
                    text_buffer += chunk
                    full_response += chunk
                    # Split sentences
                    sentences, text_buffer = extract_sentences(text_buffer)
                    for sentence in sentences:
                        logger.info(f"Enqueuing sentence for TTS: '{sentence}'")
                        await self.tts_queue.put(sentence)
                if usage:
                    final_token_usage = usage
            
            # Flush final sentence if any remaining text
            if text_buffer.strip():
                logger.info(f"Flushing final sentence for TTS: '{text_buffer.strip()}'")
                await self.tts_queue.put(text_buffer.strip())
                
            # Log full agent turn in history and DB
            if full_response.strip():
                self.session.conversation_manager.history.append({"role": "agent", "content": full_response.strip()})
                add_transcript_turn(self.session.call_sid, "agent", full_response.strip())
                
            # Save token usage metrics
            if final_token_usage:
                self.session.llm_tokens_logged += (
                    final_token_usage.get("prompt_tokens", 0) + final_token_usage.get("completion_tokens", 0)
                )
        except asyncio.CancelledError:
            logger.info("LLM stream generation task was cancelled.")
        except Exception as e:
            logger.exception(f"Error in LLM stream task: {e}")

    async def _tts_worker(self):
        """
        Worker task to synthesize complete sentences sequentially.
        """
        while True:
            try:
                sentence = await self.tts_queue.get()
                
                lang_profile = self.session.conversation_manager.language_profile
                tts_lang = lang_profile.primary_language
                
                logger.info(f"Synthesizing sentence: '{sentence}' in {tts_lang}...")
                
                # Run TTS as subtask so it is cancellable
                task = asyncio.create_task(
                    self.tts_client.text_to_speech(
                        text=sentence,
                        language_code=tts_lang,
                        speaker="shreya"
                    )
                )
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
                
                tts_audio, tts_sample_rate, num_chars = await task
                self.session.tts_characters_logged += num_chars
                
                if tts_audio:
                    logger.info(f"Enqueuing synthesized audio for: '{sentence}'")
                    await self.playback_queue.put((tts_audio, tts_sample_rate, sentence))
                else:
                    logger.error(f"TTS synthesis failed for sentence: '{sentence}'")
                    
                self.tts_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in TTS worker: {e}")

    async def _playback_worker(self):
        """
        Worker task to stream synthesized audio blocks sequentially.
        """
        while True:
            try:
                tts_audio, tts_sample_rate, sentence = await self.playback_queue.get()
                
                logger.info(f"Starting playback of audio segment ({len(tts_audio)} bytes)")
                await self.turn_manager.play_audio(
                    session=self.session,
                    pcm_data=tts_audio,
                    sample_rate=tts_sample_rate,
                    text=sentence
                )
                
                self.playback_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in playback worker: {e}")
