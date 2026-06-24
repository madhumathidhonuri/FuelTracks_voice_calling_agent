import math
import struct

def calculate_rms(audio_chunk: bytes) -> float:
    """
    Calculate the Root Mean Square (RMS) energy of a 16-bit mono PCM audio chunk.
    """
    if not audio_chunk:
        return 0.0
        
    # Each sample is 2 bytes (16-bit)
    num_samples = len(audio_chunk) // 2
    if num_samples == 0:
        return 0.0
    
    # Form formatting string for little-endian 16-bit signed shorts '<h'
    format_str = f"<{num_samples}h"
    try:
        # Unpack binary data into integers
        samples = struct.unpack(format_str, audio_chunk[:num_samples * 2])
    except struct.error:
        return 0.0
        
    sum_squares = sum(float(s) * float(s) for s in samples)
    mean_square = sum_squares / num_samples
    return math.sqrt(mean_square)

class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000, threshold: float = 800.0, silence_timeout_ms: int = 1000):
        """
        Energy-based Voice Activity Detector.
        :param sample_rate: 8000 or 16000
        :param threshold: RMS energy threshold above which speech is detected
        :param silence_timeout_ms: Duration of consecutive silence to trigger end-of-speech
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_timeout_ms = silence_timeout_ms
        self.is_speech_active = False
        self.silence_duration_ms = 0
        
    def reset(self):
        self.is_speech_active = False
        self.silence_duration_ms = 0

    def process_chunk(self, audio_chunk: bytes) -> tuple[bool, bool]:
        """
        Process a chunk of audio.
        Returns:
            (is_speech_active, speech_stopped_now)
        """
        rms = calculate_rms(audio_chunk)
        
        # Calculate duration of the chunk in milliseconds
        # 16-bit mono PCM is 2 bytes per sample
        bytes_per_sample = 2
        num_samples = len(audio_chunk) // bytes_per_sample
        if num_samples == 0:
            return self.is_speech_active, False
            
        chunk_duration_ms = (num_samples / self.sample_rate) * 1000
        
        speech_stopped_now = False
        
        if rms > self.threshold:
            # User is speaking
            if not self.is_speech_active:
                # Speech started
                self.is_speech_active = True
            self.silence_duration_ms = 0
        else:
            # Silence detected in this chunk
            if self.is_speech_active:
                self.silence_duration_ms += chunk_duration_ms
                if self.silence_duration_ms >= self.silence_timeout_ms:
                    # Speech stopped
                    self.is_speech_active = False
                    self.silence_duration_ms = 0
                    speech_stopped_now = True
                    
        return self.is_speech_active, speech_stopped_now
