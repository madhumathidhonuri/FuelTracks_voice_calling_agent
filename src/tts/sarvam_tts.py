import base64
import wave
import io
import httpx
import logging
from typing import Tuple
from config.settings import settings

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "hi-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN", 
    "mr-IN", "gu-IN", "pa-IN", "or-IN", "en-IN"
}

def extract_pcm_from_wav(wav_bytes: bytes) -> Tuple[bytes, int]:
    """
    Parse a WAV file in-memory and extract raw PCM bytes and its sample rate.
    If the bytes are not WAV format (no 'RIFF' header), return them directly.
    """
    if not wav_bytes:
        return b"", 16000
        
    if wav_bytes.startswith(b"RIFF"):
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                pcm_data = wav_file.readframes(wav_file.getnframes())
                return pcm_data, sample_rate
        except Exception as e:
            logger.error(f"Error parsing WAV header: {e}")
            # Fallback: strip standard 44-byte WAV header
            return wav_bytes[44:], 16000
    
    # Already raw PCM
    return wav_bytes, 16000

class SarvamTTSClient:
    def __init__(self):
        self.api_key = settings.SARVAM_API_KEY
        self.api_url = "https://api.sarvam.ai/text-to-speech"
        
    async def text_to_speech(
        self, 
        text: str, 
        language_code: str = "en-IN", 
        speaker: str = "shreya"
    ) -> Tuple[bytes, int, int]:
        """
        Convert text to speech using Sarvam TTS.
        Returns:
            Tuple[pcm_audio_bytes, sample_rate, num_characters]
        """
        num_chars = len(text)
        if not text:
            return b"", 16000, 0
            
        # Normalize language code
        normalized_lang = language_code
        if normalized_lang not in SUPPORTED_LANGUAGES:
            # Try matching language prefix e.g. "te" -> "te-IN"
            prefix = normalized_lang.split("-")[0]
            matched = False
            for supported in SUPPORTED_LANGUAGES:
                if supported.startswith(prefix):
                    normalized_lang = supported
                    matched = True
                    break
            if not matched:
                normalized_lang = "en-IN"
                
        # Handle mock API keys
        if not self.api_key or "mock_" in self.api_key or self.api_key == "your_sarvam_api_key":
            logger.warning("Sarvam TTS called with mock/missing API key. Returning empty mock audio.")
            # Generate 1 second of silent PCM (16000Hz * 2 bytes * 1s = 32000 bytes)
            return b"\x00" * 32000, 16000, num_chars
            
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key
        }
        
        payload = {
            "text": text,
            "target_language_code": normalized_lang,
            "speaker": speaker,
            "model": "bulbul:v3",
            "pace": 1.05  # Slightly faster for phone calls
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)
                
                if response.status_code != 200:
                    logger.error(f"Sarvam TTS failed with status {response.status_code}: {response.text}")
                    return b"", 16000, 0
                    
                result = response.json()
                
                # Check for "audios" array or fallback "audio_content"
                audios = result.get("audios")
                if audios and isinstance(audios, list) and len(audios) > 0:
                    audio_b64 = audios[0]
                else:
                    audio_b64 = result.get("audio_content", "")
                
                if not audio_b64:
                    logger.error(f"Sarvam TTS response empty. Keys in response: {list(result.keys())}")
                    return b"", 16000, 0
                    
                audio_bytes = base64.b64decode(audio_b64)
                pcm_bytes, sample_rate = extract_pcm_from_wav(audio_bytes)
                
                return pcm_bytes, sample_rate, num_chars
                
        except Exception as e:
            logger.exception(f"Error during Sarvam TTS generation: {e}")
            return b"", 16000, 0
