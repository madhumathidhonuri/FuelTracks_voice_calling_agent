import asyncio
import base64
import logging
from typing import Optional
from src.telephony.call_manager import CallSession

logger = logging.getLogger(__name__)

class TurnManager:
    def __init__(self, websocket):
        self.websocket = websocket
        self.playback_task: Optional[asyncio.Task] = None
        
    async def play_audio(self, session: CallSession, pcm_data: bytes, sample_rate: int = 16000, text: Optional[str] = None):
        """
        Start streaming audio back to Exotel in real time.
        """
        if text:
            logger.info(f"Playing audio for sentence: '{text}'")
        # Cancel any active playback first
        await self.stop_audio(session)
        
        session.is_playing = True
        session.barge_in_triggered = False
        
        self.playback_task = asyncio.create_task(
            self._send_audio_chunks_loop(session, pcm_data, sample_rate)
        )
        try:
            await self.playback_task
        except asyncio.CancelledError:
            logger.info(f"Playback task cancelled for Call {session.call_sid}")
        except Exception as e:
            logger.error(f"Error in playback task: {e}")
        finally:
            session.is_playing = False
            self.playback_task = None
            
    async def _send_audio_chunks_loop(self, session: CallSession, pcm_data: bytes, sample_rate: int):
        # Exotel streams are 16-bit PCM.
        # We split audio into 100ms chunks to stream.
        # 16-bit PCM mono = 2 bytes per sample.
        # Chunk size in bytes = sample_rate * 0.1 * 2
        chunk_size = int(sample_rate * 0.1 * 2)
        if chunk_size <= 0:
            chunk_size = 3200  # Fallback for 16kHz
            
        num_chunks = len(pcm_data) // chunk_size
        if len(pcm_data) % chunk_size > 0:
            num_chunks += 1
            
        stream_sid = session.stream_sid
        
        for i in range(num_chunks):
            # Check if barge-in was triggered while we were sleeping
            if session.barge_in_triggered:
                logger.info(f"Barge-in flag detected in play loop. Exiting playback for {session.call_sid}")
                break
                
            chunk = pcm_data[i * chunk_size : (i + 1) * chunk_size]
            if not chunk:
                break
                
            # Base64 encode the PCM chunk
            chunk_b64 = base64.b64encode(chunk).decode("utf-8")
            
            media_event = {
                "event": "media",
                "stream_sid": stream_sid,
                "media": {
                    "payload": chunk_b64
                }
            }
            
            try:
                await self.websocket.send_json(media_event)
            except Exception as e:
                logger.error(f"Failed to send media frame to Exotel: {e}")
                break
                
            # Stream chunk at real-time rate (every 100ms)
            await asyncio.sleep(0.100)
            
    async def stop_audio(self, session: CallSession):
        """
        Stops the playback task and sends a clear event to wipe Exotel's output buffer.
        """
        if self.playback_task and not self.playback_task.done():
            self.playback_task.cancel()
            try:
                await self.playback_task
            except Exception:
                pass
                
        session.is_playing = False
        
        # Clear Exotel's audio buffer
        if session.stream_sid:
            clear_event = {
                "event": "clear",
                "stream_sid": session.stream_sid
            }
            try:
                await self.websocket.send_json(clear_event)
                logger.info(f"Sent clear event to Exotel for stream {session.stream_sid}")
            except Exception as e:
                logger.error(f"Failed to send clear event: {e}")
