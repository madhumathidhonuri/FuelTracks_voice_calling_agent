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

    @patch("src.stt.sarvam_stt.SarvamSTTClient.transcribe", new_callable=AsyncMock)
    @patch("src.tts.sarvam_tts.SarvamTTSClient.text_to_speech", new_callable=AsyncMock)
    @patch("src.conversation.llm_client.LLMClient.generate_response_stream")
    async def test_pipeline_downsampling_flow(self, mock_llm_stream, mock_tts, mock_stt):
        # 1. Setup mocks
        mock_stt.return_value = ("hello gps tracker", "en-IN", 0.99, 1.5)
        
        async def mock_stream(*args, **kwargs):
            # Yield chunks of response text
            yield "Sure. I can help. ", None
            yield "Here is the details.", {"prompt_tokens": 10, "completion_tokens": 10}
            
        mock_llm_stream.side_effect = mock_stream
        # Make a mock 16kHz PCM audio stream: 1600 samples = 3200 bytes
        import struct
        samples = [1000 if i % 2 == 0 else -1000 for i in range(1600)]
        pcm_16k = struct.pack("<1600h", *samples)
        mock_tts.return_value = (pcm_16k, 16000, 10)

        # 2. Initialize CallSession & AudioPipeline at 8000 Hz
        session = CallSession("realtime-call-456", "+919876543210", "+919000666914", "lead_followup")
        
        mock_turn = MagicMock()
        mock_turn.play_audio = AsyncMock()
        mock_turn.stop_audio = AsyncMock()
        
        pipeline = AudioPipeline(session, mock_turn, sample_rate=8000)

        # 3. Simulate customer turn audio enqueued
        pipeline.stt_queue.put_nowait(b"\x00" * 16000)

        # Run loop briefly to let workers process
        await asyncio.sleep(0.5)
        
        # Verify play_audio was called with sample_rate = 8000 and downsampled audio size
        self.assertTrue(mock_turn.play_audio.called)
        call_args = mock_turn.play_audio.call_args_list[0]
        _, kwargs = call_args
        self.assertEqual(kwargs["sample_rate"], 8000)
        self.assertEqual(len(kwargs["pcm_data"]), 1600) # 3200 bytes halved to 1600 bytes

        pipeline.close()

    @patch("src.stt.sarvam_stt.SarvamSTTClient.transcribe", new_callable=AsyncMock)
    @patch("src.tts.sarvam_tts.SarvamTTSClient.text_to_speech", new_callable=AsyncMock)
    @patch("src.conversation.llm_client.LLMClient.generate_response_stream")
    async def test_pipeline_tts_worker_survival_on_barge_in(self, mock_llm_stream, mock_tts, mock_stt):
        # 1. Setup slow TTS mock
        mock_stt.return_value = ("hello", "en-IN", 0.99, 1.0)
        
        async def mock_stream(*args, **kwargs):
            yield "Sure.", None
            
        mock_llm_stream.side_effect = mock_stream
        
        async def slow_tts(*args, **kwargs):
            await asyncio.sleep(2.0)
            return b"\x00" * 3200, 16000, 10
        mock_tts.side_effect = slow_tts

        # 2. Initialize CallSession & AudioPipeline
        session = CallSession("repro-call-123", "+919876543210", "+919000666914", "lead_followup")
        
        mock_turn = MagicMock()
        mock_turn.play_audio = AsyncMock()
        mock_turn.stop_audio = AsyncMock()
        
        pipeline = AudioPipeline(session, mock_turn, sample_rate=16000)
        
        # Enqueue customer audio to trigger flow
        pipeline.stt_queue.put_nowait(b"\x00" * 32000)
        
        # Let LLM run and queue sentence to TTS, and wait for TTS to start running
        await asyncio.sleep(0.5)
        
        # Verify TTS worker is active
        tts_worker = pipeline.workers[1]
        self.assertFalse(tts_worker.done())
        
        # Trigger barge-in queue clearing
        pipeline._clear_realtime_queues()
        
        # Settle any asyncio events
        await asyncio.sleep(0.5)
        
        # Verify TTS worker survived the cancellation of its subtask
        self.assertFalse(tts_worker.done(), "TTS worker died during queue clearing/cancellation!")
        
        pipeline.close()

if __name__ == "__main__":
    unittest.main()
