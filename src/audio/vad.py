"""
Voice Activity Detection (VAD)
-------------------------------
Configurable Voice Activity Detection with three backends:
  - energy  : Fast RMS energy thresholding (legacy fallback, always available)
  - webrtc  : Google WebRTC VAD (webrtcvad-wheels package, aggressiveness=3)
  - silero  : Silero VAD via ONNX Runtime (~2MB model, high accuracy)

Backend is chosen at startup from the VAD_MODE env variable (default: silero).
If the chosen backend fails to load, it automatically falls back to energy mode.

The calculate_rms() helper is kept for the fast barge-in pre-screen in pipeline.py,
which needs microsecond-latency energy checks during agent speech playback.
"""
import math
import struct
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_rms(audio_chunk: bytes) -> float:
    """
    Calculate the Root Mean Square (RMS) energy of a 16-bit mono PCM audio chunk.
    Used by pipeline.py for fast barge-in detection during agent playback.
    """
    if not audio_chunk:
        return 0.0
    num_samples = len(audio_chunk) // 2
    if num_samples == 0:
        return 0.0
    format_str = f"<{num_samples}h"
    try:
        samples = struct.unpack(format_str, audio_chunk[:num_samples * 2])
    except struct.error:
        return 0.0
    sum_squares = sum(float(s) * float(s) for s in samples)
    return math.sqrt(sum_squares / num_samples)


def _pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float32 numpy array in range [-1, 1]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# VoiceActivityDetector — configurable multi-backend
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """
    Multi-backend Voice Activity Detector.

    Backends (controlled by VAD_MODE env variable):
      - 'energy'  : RMS energy threshold (no dependencies, always works)
      - 'webrtc'  : Google WebRTC VAD via webrtcvad-wheels (low-latency, robust)
      - 'silero'  : Silero VAD (ONNX, ~2MB model, highest accuracy for Indian speech)

    All backends share the same state-machine for silence-timeout tracking.
    """

    # Class-level shared model to avoid reloading on every call session
    _silero_model = None
    _webrtc_vad = None
    _model_sample_rate = 16000

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_timeout_ms: int = 500,
        vad_mode: str = None,
    ):
        """
        :param sample_rate:        Input PCM sample rate (8000 or 16000 Hz).
        :param threshold:          Confidence threshold for Silero [0, 1], or RMS energy level.
                                   If threshold > 1.0, always use energy mode regardless of vad_mode.
        :param silence_timeout_ms: Consecutive silence (ms) before speech is declared ended.
        :param vad_mode:           Override VAD_MODE env variable. Options: energy|webrtc|silero.
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_timeout_ms = silence_timeout_ms

        self.is_speech_active = False
        self.silence_duration_ms = 0

        # Float32 audio buffer for Silero (processes fixed windows)
        self._audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        # Raw PCM buffer for WebRTC (processes fixed 20ms frames)
        self._webrtc_buffer: bytes = b""

        # Silero window size: 512 samples @ 16kHz or 256 @ 8kHz
        self._window_size_samples = 512 if sample_rate == 16000 else 256

        # Determine backend
        if threshold > 1.0:
            # Legacy compatibility: high threshold = energy mode
            self._backend = "energy"
            logger.info(f"Threshold {threshold} > 1.0, using energy-based VAD.")
        else:
            # Read from parameter or env variable
            if vad_mode is None:
                try:
                    from config.settings import settings
                    vad_mode = settings.VAD_MODE
                except Exception:
                    vad_mode = "silero"
            self._backend = vad_mode.lower()

        # Initialize the selected backend
        if self._backend == "silero":
            VoiceActivityDetector._load_silero()
            if VoiceActivityDetector._silero_model is None:
                logger.warning("Silero unavailable, falling back to energy VAD.")
                self._backend = "energy"
        elif self._backend == "webrtc":
            VoiceActivityDetector._load_webrtc(sample_rate)
            if VoiceActivityDetector._webrtc_vad is None:
                logger.warning("WebRTC VAD unavailable, falling back to energy VAD.")
                self._backend = "energy"

        logger.info(f"VoiceActivityDetector initialized with backend='{self._backend}', "
                    f"sample_rate={sample_rate}, silence_timeout={silence_timeout_ms}ms")

    # -----------------------------------------------------------------------
    # Model Loaders (class-level singletons)
    # -----------------------------------------------------------------------

    @classmethod
    def _load_silero(cls):
        """Lazy-load Silero VAD once per process using ONNX backend (no torchaudio needed)."""
        if cls._silero_model is not None:
            return
        try:
            import sys
            import unittest.mock as mock
            sys.modules['torchaudio'] = mock.Mock()
            from silero_vad import load_silero_vad  # type: ignore
            cls._silero_model = load_silero_vad(onnx=True)
            logger.info("Silero VAD model loaded successfully (ONNX backend).")
        except ImportError:
            logger.error(
                "silero-vad package not found. "
                "Install: pip install silero-vad onnxruntime"
            )
            cls._silero_model = None
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            cls._silero_model = None

    @classmethod
    def _load_webrtc(cls, sample_rate: int):
        """Lazy-load Google WebRTC VAD once per process."""
        if cls._webrtc_vad is not None:
            return
        try:
            import webrtcvad  # type: ignore
            vad = webrtcvad.Vad()
            vad.set_mode(3)  # 3 = most aggressive filtering
            cls._webrtc_vad = vad
            logger.info("WebRTC VAD initialized successfully (aggressiveness=3).")
        except ImportError:
            logger.error(
                "webrtcvad-wheels package not found. "
                "Install: pip install webrtcvad-wheels"
            )
            cls._webrtc_vad = None
        except Exception as e:
            logger.error(f"Failed to initialize WebRTC VAD: {e}")
            cls._webrtc_vad = None

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self):
        """Reset VAD state between utterances."""
        self.is_speech_active = False
        self.silence_duration_ms = 0
        self._audio_buffer = np.array([], dtype=np.float32)
        self._webrtc_buffer = b""
        # Reset Silero internal LSTM state
        if self._backend == "silero" and self._silero_model is not None:
            try:
                self._silero_model.reset_states()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------------

    def process_chunk(self, audio_chunk: bytes) -> Tuple[bool, bool]:
        """
        Process a raw PCM audio chunk (16-bit mono, little-endian).

        Returns:
            (is_speech_active, speech_stopped_now)
        """
        if not audio_chunk:
            return self.is_speech_active, False

        num_samples_chunk = len(audio_chunk) // 2
        chunk_duration_ms = (num_samples_chunk / self.sample_rate) * 1000

        if self._backend == "silero" and self._silero_model is not None:
            return self._process_silero(audio_chunk, chunk_duration_ms)
        elif self._backend == "webrtc" and self._webrtc_vad is not None:
            return self._process_webrtc(audio_chunk, chunk_duration_ms)
        else:
            return self._process_energy(audio_chunk, chunk_duration_ms)

    def _process_silero(self, audio_chunk: bytes, chunk_duration_ms: float) -> Tuple[bool, bool]:
        """Run Silero ONNX inference on the accumulated audio buffer."""
        new_samples = _pcm_bytes_to_float32(audio_chunk)

        # Resample 8kHz → 16kHz for Silero if needed
        if self.sample_rate == 8000:
            new_samples = np.repeat(new_samples, 2)

        self._audio_buffer = np.concatenate([self._audio_buffer, new_samples])
        speech_detected_in_chunk = False

        # Run inference in fixed windows
        while len(self._audio_buffer) >= self._window_size_samples:
            window = self._audio_buffer[: self._window_size_samples]
            self._audio_buffer = self._audio_buffer[self._window_size_samples :]

            try:
                import torch
                tensor_window = torch.from_numpy(window)
                confidence = self._silero_model(tensor_window, self._model_sample_rate)
                if hasattr(confidence, 'item'):
                    confidence = confidence.item()
                confidence = float(confidence)
                if confidence >= self.threshold:
                    speech_detected_in_chunk = True
            except Exception as e:
                logger.warning(f"Silero inference error: {e}. Using energy fallback for this chunk.")
                rms = calculate_rms(audio_chunk)
                speech_detected_in_chunk = rms > 800.0

        return self._update_state(speech_detected_in_chunk, chunk_duration_ms)

    def _process_webrtc(self, audio_chunk: bytes, chunk_duration_ms: float) -> Tuple[bool, bool]:
        """
        Run Google WebRTC VAD on 20ms frames of 16-bit PCM audio.
        WebRTC requires exact 10, 20, or 30ms frames.
        """
        # WebRTC operates on exactly 20ms of 16-bit PCM at the configured sample rate
        # At 16000 Hz: 20ms = 320 samples = 640 bytes
        # At 8000 Hz:  20ms = 160 samples = 320 bytes
        frame_ms = 20
        frame_samples = int(self.sample_rate * frame_ms / 1000)
        frame_bytes = frame_samples * 2  # 16-bit = 2 bytes per sample

        self._webrtc_buffer += audio_chunk
        speech_detected_in_chunk = False

        while len(self._webrtc_buffer) >= frame_bytes:
            frame = self._webrtc_buffer[:frame_bytes]
            self._webrtc_buffer = self._webrtc_buffer[frame_bytes:]
            try:
                is_speech = self._webrtc_vad.is_speech(frame, self.sample_rate)
                if is_speech:
                    speech_detected_in_chunk = True
            except Exception as e:
                logger.warning(f"WebRTC VAD frame error: {e}. Using energy fallback.")
                rms = calculate_rms(frame)
                if rms > 800.0:
                    speech_detected_in_chunk = True

        return self._update_state(speech_detected_in_chunk, chunk_duration_ms)

    def _process_energy(self, audio_chunk: bytes, chunk_duration_ms: float) -> Tuple[bool, bool]:
        """Energy-based VAD using RMS threshold."""
        rms = calculate_rms(audio_chunk)
        # Use self.threshold if it's a valid energy value (> 1.0), else default to 800.0
        energy_threshold = self.threshold if self.threshold > 1.0 else 800.0
        speech_detected = rms > energy_threshold
        return self._update_state(speech_detected, chunk_duration_ms)

    def _update_state(self, speech_detected: bool, chunk_duration_ms: float) -> Tuple[bool, bool]:
        """Shared state-machine: updates is_speech_active and silence counter."""
        speech_stopped_now = False

        if speech_detected:
            if not self.is_speech_active:
                self.is_speech_active = True
            self.silence_duration_ms = 0
        else:
            if self.is_speech_active:
                self.silence_duration_ms += chunk_duration_ms
                if self.silence_duration_ms >= self.silence_timeout_ms:
                    self.is_speech_active = False
                    self.silence_duration_ms = 0
                    speech_stopped_now = True

        return self.is_speech_active, speech_stopped_now
