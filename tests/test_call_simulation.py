import unittest
import asyncio
import sys
import os
import sqlite3
import struct
from datetime import datetime

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.audio.vad import calculate_rms, VoiceActivityDetector
from src.stt.language_profile import LanguageProfile
from src.conversation.conversation_manager import ConversationManager
from src.storage.database import init_db, get_connection, create_call, add_transcript_turn, get_call_logs

class TestCallSimulation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Force a temporary test DB path
        settings.DATABASE_URL = "sqlite:///test_voice_calling.db"
        init_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up test DB
        db_path = settings.BASE_DIR / "test_voice_calling.db"
        if db_path.exists():
            try:
                os.remove(db_path)
            except Exception:
                pass

    def test_database_logging(self):
        call_sid = "test-call-123"
        # Test creation
        created = create_call(call_sid, "+123456", "+654321", "support")
        self.assertTrue(created)
        
        # Test double insert (should fail gracefully)
        created_again = create_call(call_sid, "+123456", "+654321", "support")
        self.assertFalse(created_again)
        
        # Test transcript turns
        add_transcript_turn(call_sid, "customer", "Hello Fuel Tracks", "en-IN", 0.99)
        add_transcript_turn(call_sid, "agent", "How can I help you today?")
        
        # Retrieve logs
        logs = get_call_logs(call_sid)
        self.assertIn("call", logs)
        self.assertIn("transcript", logs)
        self.assertEqual(logs["call"]["call_type"], "support")
        self.assertEqual(len(logs["transcript"]), 2)
        self.assertEqual(logs["transcript"][0]["role"], "customer")
        self.assertEqual(logs["transcript"][0]["text"], "Hello Fuel Tracks")
        self.assertEqual(logs["transcript"][1]["role"], "agent")

    def test_audio_rms_calculation(self):
        # Silence chunk (all zeros)
        silence_chunk = b"\x00" * 3200
        rms_silence = calculate_rms(silence_chunk)
        self.assertEqual(rms_silence, 0.0)
        
        # Pure sine/square wave simulation (alternating numbers)
        # We write signed 16-bit values (shorts)
        samples = []
        for i in range(1600):
            # Alternating between 2000 and -2000
            samples.append(2000 if i % 2 == 0 else -2000)
        
        audio_chunk = struct.pack("<1600h", *samples)
        rms = calculate_rms(audio_chunk)
        self.assertAlmostEqual(rms, 2000.0, delta=1.0)

    def test_energy_based_vad(self):
        # Initialize VAD
        vad = VoiceActivityDetector(sample_rate=16000, threshold=800.0, silence_timeout_ms=500)
        
        # 1. Feed silence chunks
        silence = b"\x00" * 3200  # 100ms
        for _ in range(3):
            active, stopped = vad.process_chunk(silence)
            self.assertFalse(active)
            self.assertFalse(stopped)
            
        # 2. Feed speech chunks (RMS = 2000, which exceeds threshold 800)
        samples = [2000 if i % 2 == 0 else -2000 for i in range(1600)]
        speech = struct.pack("<1600h", *samples)
        
        # First chunk starts speech
        active, stopped = vad.process_chunk(speech)
        self.assertTrue(active)
        self.assertFalse(stopped)
        
        # Consecutive speech chunks keep it active
        for _ in range(3):
            active, stopped = vad.process_chunk(speech)
            self.assertTrue(active)
            self.assertFalse(stopped)
            
        # 3. Feed silence chunks. Silence timeout is 500ms.
        # Each chunk is 100ms. We need 5 chunks of silence to trigger stop.
        for i in range(4):
            active, stopped = vad.process_chunk(silence)
            self.assertTrue(active)  # still active due to hangover time
            self.assertFalse(stopped)
            
        # 5th chunk should trigger speech stopped
        active, stopped = vad.process_chunk(silence)
        self.assertFalse(active)
        self.assertTrue(stopped)

    def test_silero_based_vad(self):
        # Initialize VAD with a threshold for Silero (e.g. 0.5)
        vad = VoiceActivityDetector(sample_rate=16000, threshold=0.5, silence_timeout_ms=500)
        self.assertTrue(vad._use_silero)
        
        # Mock the Silero ONNX model call
        from unittest.mock import MagicMock
        original_model = VoiceActivityDetector._model
        mock_model = MagicMock()
        VoiceActivityDetector._model = mock_model
        
        try:
            # Silence (confidence 0.01 < threshold 0.5)
            mock_model.return_value = 0.01
            silence = b"\x00" * 1024  # 512 samples = 32ms at 16kHz
            active, stopped = vad.process_chunk(silence)
            self.assertFalse(active)
            
            # Speech starts (confidence 0.99 >= threshold 0.5)
            mock_model.return_value = 0.99
            speech = b"\x00" * 1024
            active, stopped = vad.process_chunk(speech)
            self.assertTrue(active)
            self.assertFalse(stopped)
            
            # Speech continues
            active, stopped = vad.process_chunk(speech)
            self.assertTrue(active)
            
            # Silence starts
            mock_model.return_value = 0.01
            # Each chunk is 32ms. Silence timeout is 500ms.
            # We need at least 500ms / 32ms = 16 chunks of silence to trigger stop.
            for _ in range(15):
                active, stopped = vad.process_chunk(silence)
                self.assertTrue(active)
                self.assertFalse(stopped)
                
            active, stopped = vad.process_chunk(silence)
            self.assertFalse(active)
            self.assertTrue(stopped)
        finally:
            # Restore
            VoiceActivityDetector._model = original_model

    def test_language_profiling(self):
        profile = LanguageProfile()
        
        # 1. Update with purely English turn
        profile.update("en-IN", 0.99, "Hello, can you help me check if my GPS tracker is working?")
        self.assertEqual(profile.primary_language, "en-IN")
        self.assertIsNone(profile.secondary_language)
        self.assertEqual(profile.mix_ratio, 0)
        
        # 2. Update with Telugu mixed with English (using Roman script)
        # 'GPS' and 'tracker' are ASCII (English), rest are Romanized Telugu.
        profile.update("te-IN", 0.90, "nenu GPS tracker details check cheyyali andi")
        self.assertEqual(profile.primary_language, "te-IN")
        self.assertEqual(profile.secondary_language, "en-IN")
        # 2 ASCII words (GPS, tracker) out of 7 total words = ~28% mix
        self.assertGreater(profile.mix_ratio, 0)
        self.assertEqual(profile.formality_level, "formal")  # Contains "andi"

    def test_inbound_routing_classification(self):
        manager = ConversationManager("test-routing-sid")
        
        # Start in inbound routing mode
        manager.initialize_call("inbound_routing")
        self.assertEqual(manager.call_type, "inbound_routing")
        
        # Create a mock run using keyword heuristic for support
        asyncio.run(manager.add_customer_turn("My fuel sensor readings are incorrect, please check", "en-IN", 0.95))
        self.assertEqual(manager.call_type, "support")
        
        # Reset and test sales keywords
        manager.initialize_call("inbound_routing")
        asyncio.run(manager.add_customer_turn("Interested in purchasing a GPS tracker for my truck fleet", "en-IN", 0.95))
        self.assertEqual(manager.call_type, "lead_followup")
        
        # Reset and test dealer keywords
        manager.initialize_call("inbound_routing")
        asyncio.run(manager.add_customer_turn("I want to become a dealer and sell your products in Hyderabad", "en-IN", 0.95))
        self.assertEqual(manager.call_type, "dealer_recruitment")

if __name__ == "__main__":
    unittest.main()
