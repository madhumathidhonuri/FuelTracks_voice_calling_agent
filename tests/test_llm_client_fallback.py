import unittest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from src.conversation.llm_client import LLMClient

class TestLLMClientFallback(unittest.IsolatedAsyncioTestCase):
    def test_get_healthy_models_default(self):
        client = LLMClient()
        models = client.get_healthy_models()
        self.assertEqual(models[0], "gemini-3.5-flash")
        self.assertIn("gemini-3.1-flash-lite", models)

    def test_model_deprioritization_on_failure(self):
        client = LLMClient()
        
        # Initial call order
        initial_order = client.get_healthy_models().copy()
        self.assertEqual(initial_order[0], "gemini-3.5-flash")
        
        # Simulate a failure on gemini-3.5-flash
        now = time.time()
        client.model_failures["gemini-3.5-flash"] = now
        
        # Call order after failure
        new_order = client.get_healthy_models()
        self.assertNotEqual(new_order[0], "gemini-3.5-flash")
        self.assertEqual(new_order[-1], "gemini-3.5-flash") # Should be deprioritized to the very end
        self.assertEqual(new_order[0], "gemini-flash-lite-latest")

    def test_cooldown_recovery(self):
        client = LLMClient()
        
        # Simulate a failure in the past (more than 5 minutes ago)
        client.model_failures["gemini-3.5-flash"] = time.time() - 301
        
        # It should recover and be at the front again
        order = client.get_healthy_models()
        self.assertEqual(order[0], "gemini-3.5-flash")

if __name__ == "__main__":
    unittest.main()
