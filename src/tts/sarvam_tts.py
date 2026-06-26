"""
Sarvam AI Text-to-Speech (TTS) Client
---------------------------------------
Converts agent text to PCM audio using Sarvam's bulbul:v3 model.
Includes exponential-backoff retry via tenacity for transient HTTP failures.
"""
import base64
import wave
import io
import httpx
import logging
from typing import Tuple
from config.settings import settings
from src.audio.dns_resolver import resolve_hostname_ipv4
from urllib.parse import urlparse

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "hi-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN",
    "mr-IN", "gu-IN", "pa-IN", "or-IN", "en-IN"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


_MOCK_SILENT_AUDIO = b"\x00" * 32000  # 1s of silence at 16kHz, 16-bit mono

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

_RETRYABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)

_tts_retry = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class SarvamTTSClient:
    _client = None

    def __init__(self):
        self.api_key = settings.SARVAM_API_KEY
        self.api_url = "https://api.sarvam.ai/text-to-speech"
        if SarvamTTSClient._client is None:
            SarvamTTSClient._client = httpx.AsyncClient(timeout=30.0)

    async def text_to_speech(
        self,
        text: str,
        language_code: str = "en-IN",
        speaker: str = "shreya",
    ) -> Tuple[bytes, int, int]:
        """
        Convert text to speech using Sarvam TTS.

        Retries up to 3 times with exponential backoff on transient network errors.
        Falls back to silent audio on persistent failure to prevent conversation deadlock.

        Returns:
            Tuple[pcm_audio_bytes, sample_rate, num_characters]
        """
        num_chars = len(text)
        if not text:
            return b"", 16000, 0

        # Normalize language code
        normalized_lang = language_code
        if normalized_lang not in SUPPORTED_LANGUAGES:
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
            return _MOCK_SILENT_AUDIO, 16000, num_chars

        try:
            return await self._tts_with_retry(text, normalized_lang, speaker, num_chars)
        except Exception as e:
            logger.exception(f"Sarvam TTS failed after all retries: {e}. Returning silent audio.")
            return _MOCK_SILENT_AUDIO, 16000, num_chars

    @_tts_retry
    async def _tts_with_retry(
        self, text: str, normalized_lang: str, speaker: str, num_chars: int
    ) -> Tuple[bytes, int, int]:
        """Inner method so tenacity wraps only the network call."""
        parsed_url = urlparse(self.api_url)
        hostname = parsed_url.hostname or "api.sarvam.ai"
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
        resolved_ip = await resolve_hostname_ipv4(hostname)

        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key,
        }
        payload = {
            "text": text,
            "target_language_code": normalized_lang,
            "speaker": speaker,
            "model": "bulbul:v3",
            "pace": 1.05,  # Slightly faster for phone calls
        }

        response = await self._client.post(
            self.api_url,
            headers=headers,
            json=payload,
            extensions={"network_address": (resolved_ip, port)},
        )

        if response.status_code == 429:
            raise httpx.TimeoutException(f"Sarvam TTS rate-limited (429): {response.text}", request=response.request)
        if response.status_code != 200:
            logger.error(f"Sarvam TTS failed with status {response.status_code}: {response.text}")
            logger.warning("Sarvam TTS API failed. Falling back to silent audio to prevent conversation deadlock.")
            return _MOCK_SILENT_AUDIO, 16000, num_chars

        result = response.json()

        # Check for "audios" array or fallback "audio_content"
        audios = result.get("audios")
        if audios and isinstance(audios, list) and len(audios) > 0:
            audio_b64 = audios[0]
        else:
            audio_b64 = result.get("audio_content", "")

        if not audio_b64:
            logger.error(f"Sarvam TTS response empty. Keys in response: {list(result.keys())}")
            logger.warning("Sarvam TTS API returned empty content. Falling back to silent audio.")
            return _MOCK_SILENT_AUDIO, 16000, num_chars

        audio_bytes = base64.b64decode(audio_b64)
        pcm_bytes, sample_rate = extract_pcm_from_wav(audio_bytes)

        return pcm_bytes, sample_rate, num_chars
