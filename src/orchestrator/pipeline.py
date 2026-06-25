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
        
        # VAD settings: 800 RMS threshold, 600ms silence timeout for faster responses
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate, 
            threshold=800.0, 
            silence_timeout_ms=600
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
        
        # Explicit turn tracking to serialize conversational states
        self.processing_turn = False
        
        # Automatic hangup state tracking
        self.session.should_hangup = False
        self.session.should_hangup_pending = False

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
        Directly queue the pre-defined welcome greeting for TTS to minimize pickup latency.
        """
        try:
            logger.info(f"Triggering initial welcoming greeting for call {self.session.call_sid}...")
            self.processing_turn = True
            
            context = self.session.conversation_manager.context
            customer_name = context.get("customer_name")
            
            sentences = []
            from datetime import datetime
            hour = datetime.now().hour
            if 5 <= hour < 12:
                tod = "good morning"
            elif 12 <= hour < 17:
                tod = "good afternoon"
            elif 17 <= hour < 22:
                tod = "good evening"
            else:
                tod = "hello"

            if self.session.call_type == "inbound_routing":
                sentences.append(f"Hello, {tod}! Thank you for calling Fuel Tracks Technologies.")
                sentences.append("My name is Shreya.")
                sentences.append("Are you calling about an existing account, or are you interested in GPS tracking for your fleet?")
            else:
                sentences.append(f"Hello, {tod}!")
                if customer_name and customer_name.strip() and customer_name != "Valued Customer":
                    sentences.append(f"Am I speaking with {customer_name.strip()}?")
                else:
                    sentences.append("How can I help you today?")
                
            full_greeting = " ".join(sentences)
            
            # Log greeting to conversation history and database
            self.session.conversation_manager.history.append({"role": "agent", "content": full_greeting})
            add_transcript_turn(self.session.call_sid, "agent", full_greeting)
            
            # Enqueue each sentence directly for TTS synthesis concurrently
            tts_lang = self.session.conversation_manager.language_profile.primary_language
            for sentence in sentences:
                logger.info(f"Starting initial welcoming TTS task for: '{sentence}'")
                task = asyncio.create_task(
                    self.tts_client.text_to_speech(
                        text=sentence,
                        language_code=tts_lang,
                        speaker="shreya"
                    )
                )
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
                await self.tts_queue.put((task, sentence))
                
        except Exception as e:
            logger.exception(f"Error during initial welcome trigger: {e}")
            self.processing_turn = False

    async def process_inbound_audio(self, audio_chunk: bytes):
        """
        Handle incoming PCM audio chunks from Exotel.
        """
        self.session.touch()
        
        # 1. Check for barge-in if the agent is active
        if self.processing_turn:
                
            # Process chunks into VAD so we don't lose the customer's interruption speech
            is_speech_active, speech_stopped_now = self.vad.process_chunk(audio_chunk)
            if is_speech_active:
                self.audio_buffer.extend(audio_chunk)
                
            rms = calculate_rms(audio_chunk)
            if rms > 2200.0:  # Raised threshold to ignore minor line noise/breathing
                self.barge_in_counter += 1
                if self.barge_in_counter >= 8:  # Requires sustained speech (~160ms at 20ms chunks)
                    logger.info(f"Barge-in detected on call {self.session.call_sid}! Interrupting playback/generation.")
                    self.session.barge_in_triggered = True
                    await self.turn_manager.stop_audio(self.session)
                    
                    # Stop active generation/synthesis and flush queues
                    self._clear_realtime_queues()
                    
                    # Note: We do NOT clear self.audio_buffer or reset VAD here.
                    # We let the VAD continue accumulating the rest of their utterance.
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
            
            self.processing_turn = True
            # Put utterance on STT queue
            self.stt_queue.put_nowait(utterance_audio)

    def _clear_realtime_queues(self):
        """
        Clears all pending sentence and audio playback queues, and cancels current API requests.
        """
        logger.info(f"Clearing real-time queues for call {self.session.call_sid} due to barge-in.")
        
        self.processing_turn = False
        
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
                
                # Reset barge-in flag before transcribing, because any barge-in during/after this point is a new interruption.
                self.session.barge_in_triggered = False
                
                # Transcribe
                transcript, detected_lang, confidence, duration_sec = await self.stt_client.transcribe(
                    audio_data, 
                    self.sample_rate
                )
                self.session.stt_seconds_logged += duration_sec
                logger.info(f"STT: '{transcript}' [Lang: {detected_lang}, Conf: {confidence:.2f}, Dur: {duration_sec:.2f}s]")
                
                # If a barge-in happened while we were transcribing (customer started speaking again), discard this old turn
                if self.session.barge_in_triggered:
                    logger.info("Barge-in triggered during STT transcription. Discarding old transcript.")
                    self.stt_queue.task_done()
                    continue
                    
                if not transcript.strip() or confidence < 0.30:
                    if confidence < 0.30 and transcript.strip():
                        logger.info(f"Ignoring low-confidence STT transcript: '{transcript}' (confidence: {confidence:.2f})")
                    self.processing_turn = False
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
                self.processing_turn = False

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
            
            # Check if call duration is approaching the 1-2 minute limit
            from datetime import datetime
            elapsed_sec = (datetime.now() - self.session.start_time).total_seconds()
            if elapsed_sec > 75.0:
                logger.info(f"Soft limit reached ({elapsed_sec:.1f}s). Appending wrap-up instructions to prompt.")
                system_prompt += (
                    "\n\n[CRITICAL DIRECTIVE: The call is reaching its duration limit. "
                    "You must immediately and politely wrap up the call. Do not start new topics. "
                    "Offer to send details/brochures on WhatsApp, provide our contact details "
                    "(+91 9000666914, info@fueltracks.in), thank them, and say goodbye now.]"
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
                    tts_lang = self.session.conversation_manager.language_profile.primary_language
                    for sentence in sentences:
                        logger.info(f"Starting concurrent TTS task for sentence: '{sentence}'")
                        task = asyncio.create_task(
                            self.tts_client.text_to_speech(
                                text=sentence,
                                language_code=tts_lang,
                                speaker="shreya"
                            )
                        )
                        self.active_tasks.add(task)
                        task.add_done_callback(self.active_tasks.discard)
                        await self.tts_queue.put((task, sentence))
                if usage:
                    final_token_usage = usage
            
            # Flush final sentence if any remaining text
            if text_buffer.strip():
                sentence = text_buffer.strip()
                logger.info(f"Flushing final sentence and starting TTS task for: '{sentence}'")
                tts_lang = self.session.conversation_manager.language_profile.primary_language
                task = asyncio.create_task(
                    self.tts_client.text_to_speech(
                        text=sentence,
                        language_code=tts_lang,
                        speaker="shreya"
                    )
                )
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
                await self.tts_queue.put((task, sentence))
                
            # Log full agent turn in history and DB
            if full_response.strip():
                self.session.conversation_manager.history.append({"role": "agent", "content": full_response.strip()})
                add_transcript_turn(self.session.call_sid, "agent", full_response.strip())
                
                # Check for goodbye/call completion to trigger automatic hangup
                lower_resp = full_response.lower()
                goodbye_keywords = ["goodbye", "bye now", "have a great day", "have a nice day", "thank you for your time", "shubhadin", "dhanyavadalu", "selavu", "thank you, bye", "phir milenge", "alvida", "dhanyawad"]
                if any(kw in lower_resp for kw in goodbye_keywords):
                    logger.info(f"Goodbye detected in agent response. Session {self.session.call_sid} marked for hangup.")
                    self.session.should_hangup_pending = True
            else:
                logger.warning("LLM response was empty. Resetting processing_turn.")
                self.processing_turn = False
                
            # Save token usage metrics
            if final_token_usage:
                self.session.llm_tokens_logged += (
                    final_token_usage.get("prompt_tokens", 0) + final_token_usage.get("completion_tokens", 0)
                )
        except asyncio.CancelledError:
            logger.info("LLM stream generation task was cancelled.")
        except Exception as e:
            logger.exception(f"Error in LLM stream task: {e}")
            self.processing_turn = False

    async def _tts_worker(self):
        """
        Worker task to process pre-spawned TTS synthesis tasks in order.
        """
        while True:
            try:
                task, sentence = await self.tts_queue.get()
                
                try:
                    tts_audio, tts_sample_rate, num_chars = await task
                except asyncio.CancelledError:
                    logger.info(f"TTS task was cancelled for sentence: '{sentence}'")
                    self.tts_queue.task_done()
                    continue
                
                self.session.tts_characters_logged += num_chars
                
                if tts_audio:
                    if tts_sample_rate != self.sample_rate:
                        logger.info(f"Resampling TTS audio from {tts_sample_rate} Hz to {self.sample_rate} Hz for Exotel...")
                        import numpy as np
                        samples = np.frombuffer(tts_audio, dtype=np.int16)
                        if len(samples) > 0:
                            if tts_sample_rate > self.sample_rate:
                                # Apply low-pass filter to prevent aliasing
                                window_size = int(round(tts_sample_rate / self.sample_rate))
                                if window_size > 1:
                                    kernel = np.ones(window_size) / window_size
                                    samples = np.convolve(samples, kernel, mode='same').astype(np.int16)
                            
                            num_samples = len(samples)
                            target_num_samples = int(num_samples * self.sample_rate / tts_sample_rate)
                            src_indices = np.arange(num_samples)
                            target_indices = np.linspace(0, num_samples - 1, target_num_samples)
                            resampled_samples = np.interp(target_indices, src_indices, samples).astype(np.int16)
                            tts_audio = resampled_samples.tobytes()
                        tts_sample_rate = self.sample_rate

                    logger.info(f"Enqueuing synthesized audio for: '{sentence}'")
                    await self.playback_queue.put((tts_audio, tts_sample_rate, sentence))
                else:
                    logger.error(f"TTS synthesis failed for sentence: '{sentence}'")
                    if self.tts_queue.empty() and self.playback_queue.empty() and len(self.active_tasks) == 0:
                        logger.info("TTS failed and no other sentences. Resetting processing_turn.")
                        self.processing_turn = False
                    
                self.tts_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in TTS worker: {e}")
                self.processing_turn = False

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
                
                if self.playback_queue.empty() and self.tts_queue.empty() and len(self.active_tasks) == 0:
                    logger.info("Agent finished speaking all queued sentences. Resetting processing_turn flag and VAD buffer.")
                    self.processing_turn = False
                    self.audio_buffer.clear()
                    self.vad.reset()
                    if getattr(self.session, "should_hangup_pending", False):
                        logger.info("Goodbye playback completed. Setting should_hangup flag to True.")
                        self.session.should_hangup = True
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in playback worker: {e}")
                self.processing_turn = False
