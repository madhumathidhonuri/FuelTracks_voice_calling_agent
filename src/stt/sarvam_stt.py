import io
import wave
import httpx
import logging
from typing import Dict, Any, Tuple
from config.settings import settings
from src.audio.dns_resolver import resolve_hostname_ipv4
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """
    Convert raw PCM bytes to WAV format bytes in-memory.
    """
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM = 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()

class SarvamSTTClient:
    _client = None

    def __init__(self):
        self.api_key = settings.SARVAM_API_KEY
        self.api_url = "https://api.sarvam.ai/speech-to-text"
        if SarvamSTTClient._client is None:
            SarvamSTTClient._client = httpx.AsyncClient(timeout=30.0)
        
    async def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> Tuple[str, str, float, float]:
        """
        Transcribe the audio chunk using Sarvam STT REST API with language auto-detection.
        Returns:
            Tuple[transcript, detected_language, language_probability, duration_sec]
        """
        if not pcm_data or len(pcm_data) < 320:
            return "", "en-IN", 1.0, 0.0
            
        wav_data = pcm_to_wav(pcm_data, sample_rate)
        duration_sec = len(pcm_data) / (sample_rate * 2)
        
        # Check if API key is mock/placeholder
        if not self.api_key or "mock_" in self.api_key or self.api_key == "your_sarvam_api_key":
            logger.warning("Sarvam STT called with mock/missing API key. Returning mock transcription.")
            return "Mock transcription (please configure Sarvam API key)", "en-IN", 1.0, duration_sec
            
        headers = {
            "api-subscription-key": self.api_key
        }
        
        # Files structure for httpx multipart request
        files = {
            "file": ("utterance.wav", wav_data, "audio/wav")
        }
        data = {
            "model": "saaras:v3",
            "language_code": "unknown",
            "mode": "codemix"
        }
        
        try:
            parsed_url = urlparse(self.api_url)
            hostname = parsed_url.hostname or "api.sarvam.ai"
            port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
            resolved_ip = await resolve_hostname_ipv4(hostname)
            
            response = await self._client.post(
                self.api_url, 
                headers=headers, 
                files=files, 
                data=data,
                extensions={"network_address": (resolved_ip, port)}
            )
            if response.status_code != 200:
                logger.error(f"Sarvam STT request failed with status {response.status_code}: {response.text}")
                return "", "en-IN", 0.0, duration_sec
                
            result = response.json()
            
            transcript = result.get("transcript", "")
            detected_lang = result.get("language_code", "en-IN")
            lang_probability = result.get("language_probability", 1.0)
            
            # Check metrics if available
            metrics = result.get("metrics", {})
            actual_duration = metrics.get("audio_duration", duration_sec)
            
            return transcript, detected_lang or "en-IN", lang_probability or 1.0, actual_duration
                
        except Exception as e:
            logger.exception(f"Error during Sarvam STT transcription: {e}")
            return "", "en-IN", 0.0, duration_sec
