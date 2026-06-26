import unittest
import time
from src.conversation.llm_client import LLMClient


class TestLLMClientFallback(unittest.IsolatedAsyncioTestCase):

    def test_get_healthy_models_default(self):
        """Cascade starts with Gemini (primary) then Groq models."""
        client = LLMClient()
        models = client.get_healthy_models()

        # get_healthy_models returns (provider, model_name) tuples
        providers = [p for p, _ in models]
        names     = [m for _, m in models]

        # Gemini must be first if the key is configured
        self.assertEqual(models[0][0], "gemini")
        self.assertEqual(models[0][1], "gemini-2.0-flash")

        # Groq fallbacks must be present
        self.assertIn("llama-3.3-70b-versatile", names)
        self.assertIn("llama-3.1-8b-instant",    names)

        # Old Gemini model names that were never valid should not appear
        self.assertNotIn("gemini-3.5-flash",           names)
        self.assertNotIn("gemini-flash-lite-latest",   names)
        self.assertNotIn("gemini-flash-latest",        names)
        self.assertNotIn("gemini-3.1-flash-lite",      names)

    def test_model_deprioritization_on_failure(self):
        """A failed model is moved to the back of the cascade."""
        client = LLMClient()

        initial_order = client.get_healthy_models().copy()

        # Primary should be Gemini first
        self.assertEqual(initial_order[0], ("gemini", "gemini-2.0-flash"))

        # Simulate a failure on the primary Gemini model
        client.model_failures["gemini-2.0-flash"] = time.time()

        new_order = client.get_healthy_models()
        names = [m for _, m in new_order]

        # gemini-2.0-flash should now be deprioritized to the very end
        self.assertNotEqual(new_order[0], ("gemini", "gemini-2.0-flash"))
        self.assertEqual(new_order[-1], ("gemini", "gemini-2.0-flash"))

        # Next healthy model should be gemini-2.0-flash-lite (if key set)
        # or the first Groq model
        self.assertIn(new_order[0][1], ["gemini-2.0-flash-lite", "llama-3.3-70b-versatile"])

    def test_cooldown_recovery(self):
        """A model that failed more than 5 minutes ago recovers to front."""
        client = LLMClient()

        # Simulate a failure more than 5 minutes ago
        client.model_failures["gemini-2.0-flash"] = time.time() - 301

        # It should recover and appear at the front of the cascade again
        order = client.get_healthy_models()
        self.assertEqual(order[0], ("gemini", "gemini-2.0-flash"))


if __name__ == "__main__":
    unittest.main()
