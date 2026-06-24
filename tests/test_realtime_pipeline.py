import unittest
import asyncio
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.orchestrator.pipeline import extract_sentences, AudioPipeline
from src.telephony.call_manager import CallSession
from src.storage.database import init_db

class TestRealtimePipeline(unittest.IsolatedAsyncioTestCase):
    
    @classmethod
    def setUpClass(cls):
        # Force test database
        settings.DATABASE_URL = "sqlite:///test_voice_calling_realtime.db"
        init_db()

    @classmethod
    def tearDownClass(cls):
        db_path = settings.BASE_DIR / "test_voice_calling_realtime.db"
        if db_path.exists():
            try:
                os.remove(db_path)
            except Exception:
                pass

    def test_sentence_extraction_basic(self):
        # Basic English
        sentences, remaining = extract_sentences("Hello there. How are you? I am")
        self.assertEqual(sentences, ["Hello there.", "How are you?"])
        self.assertEqual(remaining, " I am")

        # Native Hindi (using । full-stop)
        sentences, remaining = extract_sentences("आपका स्वागत है। आप कैसे हैं? अभी")
        self.assertEqual(sentences, ["आपका स्वागत है।", "आप कैसे हैं?"])
        self.assertEqual(remaining, " अभी")

        # Newlines and multiple spaces
        sentences, remaining = extract_sentences("First sentence\nSecond sentence. Third")
        self.assertEqual(sentences, ["First sentence", "Second sentence."])
        self.assertEqual(remaining, " Third")

        # No terminators
        sentences, remaining = extract_sentences("Hello world")
        self.assertEqual(sentences, [])
        self.assertEqual(remaining, "Hello world")

    @patch("src.stt.sarvam_stt.SarvamSTTClient.transcribe", new_callable=AsyncMock)
    @patch("src.tts.sarvam_tts.SarvamTTSClient.text_to_speech", new_callable=AsyncMock)
    @patch("src.conversation.llm_client.LLMClient.generate_response_stream")
    async def test_pipeline_queue_flow(self, mock_llm_stream, mock_tts, mock_stt):
        # 1. Setup mocks
        mock_stt.return_value = ("hello gps tracker", "en-IN", 0.99, 1.5)
        
        async def mock_stream(*args, **kwargs):
            # Yield chunks of response text
            yield "Sure. I can help. ", None
            yield "Here is the details.", {"prompt_tokens": 10, "completion_tokens": 10}
            
        mock_llm_stream.side_effect = mock_stream
        mock_tts.return_value = (b"\x00" * 3200, 16000, 10) # 100ms dummy PCM audio

        # 2. Initialize CallSession & AudioPipeline
        session = CallSession("realtime-call-123", "+919876543210", "+919000666914", "lead_followup")
        
        mock_turn = MagicMock()
        mock_turn.play_audio = AsyncMock()
        mock_turn.stop_audio = AsyncMock()
        
        pipeline = AudioPipeline(session, mock_turn, sample_rate=16000)

        # 3. Simulate customer turn audio enqueued
        pipeline.stt_queue.put_nowait(b"\x00" * 32000)

        # Run loop briefly to let workers process
        await asyncio.sleep(0.5)
        
        # Verify barge-in clearing
        pipeline._clear_realtime_queues()
        self.assertEqual(pipeline.tts_queue.qsize(), 0)
        self.assertEqual(pipeline.playback_queue.qsize(), 0)
        self.assertEqual(len(pipeline.active_tasks), 0)

        pipeline.close()

if __name__ == "__main__":
    unittest.main()
