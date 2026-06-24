import logging
import asyncio
from src.telephony.call_manager import CallSession
from src.audio.vad import VoiceActivityDetector, calculate_rms
from src.stt.sarvam_stt import SarvamSTTClient
from src.tts.sarvam_tts import SarvamTTSClient
from src.orchestrator.turn_manager import TurnManager

logger = logging.getLogger(__name__)

class AudioPipeline:
    def __init__(self, session: CallSession, turn_manager: TurnManager, sample_rate: int = 16000):
        self.session = session
        self.turn_manager = turn_manager
        self.sample_rate = sample_rate
        
        # Audio buffer for accumulating customer speech
        self.audio_buffer = bytearray()
        
        # Energy threshold of 800 RMS, 1.0s silence timeout
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate, 
            threshold=800.0, 
            silence_timeout_ms=1000
        )
        
        # API Clients
        self.stt_client = SarvamSTTClient()
        self.tts_client = SarvamTTSClient()
        
        # Barge-in tracking
        self.barge_in_counter = 0
        
    async def trigger_initial_greeting(self):
        """
        Generate and play the agent's first welcome turn.
        """
        try:
            logger.info(f"Triggering initial greeting for call {self.session.call_sid}...")
            
            # Generate greeting text from LLM
            response_text, token_usage = await self.session.conversation_manager.generate_agent_response()
            self.session.llm_tokens_logged += (
                token_usage.get("prompt_tokens", 0) + token_usage.get("completion_tokens", 0)
            )
            
            logger.info(f"Initial Greeting LLM: '{response_text}'")
            
            # Default welcome lang is primary_language (en-IN)
            tts_lang = self.session.conversation_manager.language_profile.primary_language
            
            # Generate TTS audio
            tts_audio, tts_sample_rate, num_chars = await self.tts_client.text_to_speech(
                text=response_text,
                language_code=tts_lang,
                speaker="shreya"
            )
            self.session.tts_characters_logged += num_chars
            
            if tts_audio:
                # Play greeting
                await self.turn_manager.play_audio(
                    session=self.session,
                    pcm_data=tts_audio,
                    sample_rate=tts_sample_rate
                )
            else:
                logger.error("Failed to generate audio for initial greeting")
                
        except Exception as e:
            logger.exception(f"Error during initial greeting: {e}")

    async def process_inbound_audio(self, audio_chunk: bytes):
        """
        Handle real-time incoming PCM chunks.
        """
        self.session.touch()
        
        # 1. Check for barge-in if the agent is speaking
        if self.session.is_playing:
            rms = calculate_rms(audio_chunk)
            if rms > 1200.0:  # Slightly higher threshold for barge-in to be safe
                self.barge_in_counter += 1
                if self.barge_in_counter >= 2:  # Sustained speech (~200ms)
                    logger.info(f"Barge-in detected on call {self.session.call_sid}! Stopping playback.")
                    self.session.barge_in_triggered = True
                    await self.turn_manager.stop_audio(self.session)
                    self.audio_buffer.clear()
                    self.vad.reset()
                    self.barge_in_counter = 0
            else:
                self.barge_in_counter = max(0, self.barge_in_counter - 1)
            return
            
        # 2. Run Voice Activity Detection if the agent is listening
        is_speech_active, speech_stopped_now = self.vad.process_chunk(audio_chunk)
        
        if is_speech_active:
            # Buffer user audio while speaking
            self.audio_buffer.extend(audio_chunk)
            
        if speech_stopped_now:
            logger.info(f"Silence detected. Transcribing complete turn of duration {len(self.audio_buffer)/(self.sample_rate*2):.2f}s")
            utterance_audio = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            self.vad.reset()
            
            # Spin off turn processing in task to avoid blocking the ws thread
            asyncio.create_task(self._process_utterance(utterance_audio))
            
    async def _process_utterance(self, audio_data: bytes):
        try:
            # 1. Transcribe the audio
            transcript, detected_lang, confidence, duration_sec = await self.stt_client.transcribe(
                audio_data, 
                self.sample_rate
            )
            
            self.session.stt_seconds_logged += duration_sec
            logger.info(f"STT: '{transcript}' [Lang: {detected_lang}, Conf: {confidence:.2f}, Dur: {duration_sec:.2f}s]")
            
            if not transcript.strip():
                logger.info("Empty transcript. Resuming listening.")
                return
                
            # 2. Append customer turn
            await self.session.add_customer_turn(
                text=transcript,
                detected_language=detected_lang,
                confidence=confidence
            )
            
            # 3. Generate response text
            response_text, token_usage = await self.session.conversation_manager.generate_agent_response()
            self.session.llm_tokens_logged += (
                token_usage.get("prompt_tokens", 0) + token_usage.get("completion_tokens", 0)
            )
            
            logger.info(f"LLM Response: '{response_text}'")
            
            # 4. Generate TTS audio in the caller's language
            lang_profile = self.session.conversation_manager.language_profile
            tts_lang = lang_profile.primary_language
            
            logger.info(f"Generating TTS: language={tts_lang}")
            tts_audio, tts_sample_rate, num_chars = await self.tts_client.text_to_speech(
                text=response_text,
                language_code=tts_lang,
                speaker="shreya"
            )
            self.session.tts_characters_logged += num_chars
            
            if not tts_audio:
                logger.error("TTS generation failed")
                return
                
            # 5. Play response
            await self.turn_manager.play_audio(
                session=self.session,
                pcm_data=tts_audio,
                sample_rate=tts_sample_rate
            )
            
        except Exception as e:
            logger.exception(f"Error in _process_utterance: {e}")
