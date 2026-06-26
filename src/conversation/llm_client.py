# LLM CASCADE ORDER (Gemini-primary, Groq-fallback):
# 1. gemini-2.0-flash          (primary — fastest Gemini, if key is set & quota OK)
# 2. llama-3.3-70b-versatile   (Groq fallback #1 — high quality)
# 3. llama-3.1-8b-instant      (Groq fallback #2 — ultra-fast)
# If all fail → graceful apology response

"""
LLM Client
-----------
Generates conversational responses using a two-tier cascade:
  Primary  : Google Gemini (gemini-2.0-flash) via REST API
  Fallback : Groq API (Llama 3 models) via OpenAI-compatible endpoint

If GEMINI_API_KEY is set and working, Gemini is used first.
If Gemini fails (quota, network, etc.), the client transparently falls through
to the Groq models in order, with no interruption to the call.

If GEMINI_API_KEY is not set, the cascade starts directly from Groq.
"""
import logging
import asyncio
import time
import json
from typing import Tuple, Dict, Any, Union
from config.settings import settings
from src.conversation.prompt_builder import SystemPrompt

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None


# ---------------------------------------------------------------------------
# Helper: normalise system prompt input
# ---------------------------------------------------------------------------

def _prompt_text(prompt: Union[str, SystemPrompt]) -> str:
    """Accept both plain str and SystemPrompt dataclass."""
    if isinstance(prompt, SystemPrompt):
        return prompt.text
    return str(prompt)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_STREAM_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Models tried in order (each is a (provider, model_name) tuple)
# The list is rebuilt at runtime so failed models are deprioritized.
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
GROQ_MODELS   = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self):
        self.gemini_key = settings.GEMINI_API_KEY
        self.groq_key   = settings.GROQ_API_KEY
        self.model_failures: Dict[str, float] = {}   # model_name -> last-failure timestamp

        # Log what's configured
        has_gemini = bool(self.gemini_key and "mock_" not in str(self.gemini_key) and "your_" not in str(self.gemini_key))
        has_groq   = bool(self.groq_key   and "mock_" not in str(self.groq_key)   and "your_" not in str(self.groq_key))

        if has_gemini and has_groq:
            logger.info("LLM cascade: Gemini (primary) → Groq (fallback).")
        elif has_gemini:
            logger.info("LLM cascade: Gemini only (GROQ_API_KEY not set).")
        elif has_groq:
            logger.info("LLM cascade: Groq only (GEMINI_API_KEY not set or quota exhausted).")
        else:
            logger.warning("No LLM keys configured — all calls will return the offline fallback.")

    # -----------------------------------------------------------------------
    # Build ordered model list (healthy models first)
    # -----------------------------------------------------------------------

    def get_healthy_models(self) -> list:
        """
        Returns (provider, model_name) pairs in cascade order, with recently-failed
        models moved to the back after a 5-minute cooldown.
        """
        cooldown = 300  # seconds
        now = time.time()

        has_gemini = bool(self.gemini_key and "mock_" not in str(self.gemini_key) and "your_" not in str(self.gemini_key))
        has_groq   = bool(self.groq_key   and "mock_" not in str(self.groq_key)   and "your_" not in str(self.groq_key))

        candidates = []
        if has_gemini:
            candidates += [("gemini", m) for m in GEMINI_MODELS]
        if has_groq:
            candidates += [("groq", m) for m in GROQ_MODELS]

        healthy   = [(p, m) for p, m in candidates if now - self.model_failures.get(m, 0) > cooldown]
        unhealthy = [(p, m) for p, m in candidates if now - self.model_failures.get(m, 0) <= cooldown]
        return healthy + unhealthy

    # -----------------------------------------------------------------------
    # Message format converters
    # -----------------------------------------------------------------------

    def _to_groq_messages(self, system_text: str, messages: list) -> list:
        """Convert dialog history to OpenAI/Groq chat format."""
        out = [{"role": "system", "content": system_text}]
        for msg in messages:
            role = msg.get("role", "user")
            if role == "customer":
                role = "user"
            elif role in ("agent", "assistant"):
                role = "assistant"
            elif role == "system":
                continue
            out.append({"role": role, "content": msg.get("content", "")})
        return out

    def _to_gemini_body(self, system_text: str, messages: list) -> dict:
        """Convert dialog history to Gemini generateContent format."""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            if role in ("customer", "user"):
                g_role = "user"
            elif role in ("agent", "assistant"):
                g_role = "model"
            else:
                continue
            contents.append({"role": g_role, "parts": [{"text": msg.get("content", "")}]})

        # Gemini requires the conversation to start with a user turn
        if not contents or contents[0]["role"] != "user":
            contents.insert(0, {"role": "user", "parts": [{"text": "Hello"}]})

        return {
            "system_instruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000},
        }

    # -----------------------------------------------------------------------
    # Per-provider call helpers (non-streaming)
    # -----------------------------------------------------------------------

    async def _call_gemini(self, client: "httpx.AsyncClient", model: str, body: dict) -> Tuple[str, dict]:
        url = GEMINI_API_URL.format(model=model) + f"?key={self.gemini_key}"
        response = await client.post(url, json=body, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            return text, {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
            }
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:200]}")

    async def _call_groq(self, client: "httpx.AsyncClient", model: str, messages: list) -> Tuple[str, dict]:
        headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
        response = await client.post(
            GROQ_API_URL,
            headers=headers,
            json={"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 1000},
            timeout=10.0,
        )
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return text, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        raise RuntimeError(f"Groq HTTP {response.status_code}: {response.text[:200]}")

    # -----------------------------------------------------------------------
    # Non-streaming public API
    # -----------------------------------------------------------------------

    async def generate_response(
        self,
        system_prompt: Union[str, SystemPrompt],
        messages: list,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response using Gemini (primary) → Groq (fallback).
        Returns: (response_text, token_usage_dict)
        """
        system_text = _prompt_text(system_prompt)
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        ordered = self.get_healthy_models()

        if not ordered:
            logger.warning("No LLM API keys configured — returning offline fallback.")
            return self._fallback_text(), token_usage

        gemini_body  = self._to_gemini_body(system_text, messages)
        groq_messages = self._to_groq_messages(system_text, messages)
        start = time.monotonic()

        async with httpx.AsyncClient() as client:
            for provider, model_name in ordered:
                if time.monotonic() - start > 14.0:
                    logger.warning("LLM total timeout (14s) reached.")
                    break
                try:
                    logger.info(f"Calling {provider.upper()} model: {model_name} ...")
                    if provider == "gemini":
                        text, usage = await asyncio.wait_for(
                            self._call_gemini(client, model_name, gemini_body), timeout=9.0
                        )
                    else:
                        text, usage = await asyncio.wait_for(
                            self._call_groq(client, model_name, groq_messages), timeout=9.0
                        )
                    token_usage.update(usage)
                    logger.info(f"[{provider.upper()}] {model_name} succeeded. Tokens: {token_usage}")
                    return text, token_usage

                except Exception as e:
                    logger.error(f"[{provider.upper()}] {model_name} failed: {e}. Trying next...")
                    self.model_failures[model_name] = time.time()

        logger.warning("All LLM providers failed — returning offline fallback.")
        return self._fallback_text(), token_usage

    # -----------------------------------------------------------------------
    # Streaming public API
    # -----------------------------------------------------------------------

    async def generate_response_stream(
        self,
        system_prompt: Union[str, SystemPrompt],
        messages: list,
    ):
        """
        Streaming response using Gemini (primary) → Groq (fallback).
        Yields: (text_chunk, Optional[token_usage_dict])
        """
        system_text  = _prompt_text(system_prompt)
        token_usage  = {"prompt_tokens": 0, "completion_tokens": 0}
        ordered      = self.get_healthy_models()

        if not ordered:
            logger.warning("No LLM API keys configured — returning offline fallback stream.")
            async for chunk in self._fallback_stream(token_usage):
                yield chunk
            return

        gemini_body   = self._to_gemini_body(system_text, messages)
        groq_messages = self._to_groq_messages(system_text, messages)
        start = time.monotonic()

        for provider, model_name in ordered:
            if time.monotonic() - start > 14.0:
                logger.warning("LLM total timeout (14s) reached.")
                break
            try:
                logger.info(f"Calling {provider.upper()} stream: {model_name} ...")

                if provider == "gemini":
                    # Gemini streaming (alt=sse)
                    url = GEMINI_STREAM_URL.format(model=model_name) + f"?key={self.gemini_key}&alt=sse"
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        async with client.stream("POST", url, json=gemini_body) as response:
                            if response.status_code != 200:
                                err = await response.aread()
                                raise RuntimeError(f"Gemini stream HTTP {response.status_code}: {err[:150]}")
                            async for line in response.aiter_lines():
                                if not line.startswith("data:"):
                                    continue
                                payload = line[5:].strip()
                                if payload in ("[DONE]", ""):
                                    continue
                                try:
                                    data = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue
                                parts = (data.get("candidates", [{}])[0]
                                             .get("content", {})
                                             .get("parts", []))
                                for part in parts:
                                    chunk = part.get("text", "")
                                    if chunk:
                                        yield chunk, None
                                # Collect usage metadata from final chunk
                                usage = data.get("usageMetadata")
                                if usage:
                                    token_usage["prompt_tokens"]     = usage.get("promptTokenCount", 0)
                                    token_usage["completion_tokens"]  = usage.get("candidatesTokenCount", 0)
                    logger.info(f"[GEMINI] {model_name} stream done. Tokens: {token_usage}")
                    yield "", token_usage
                    return

                else:
                    # Groq streaming (OpenAI SSE)
                    headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        async with client.stream(
                            "POST", GROQ_API_URL, headers=headers,
                            json={"model": model_name, "messages": groq_messages,
                                  "temperature": 0.7, "max_tokens": 1000, "stream": True},
                            timeout=httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=5.0),
                        ) as response:
                            if response.status_code != 200:
                                err = await response.aread()
                                raise RuntimeError(f"Groq stream HTTP {response.status_code}: {err[:150]}")
                            async for line in response.aiter_lines():
                                if not line.startswith("data:"):
                                    continue
                                payload = line[5:].strip()
                                if payload == "[DONE]":
                                    break
                                try:
                                    data = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue
                                text_piece = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if text_piece:
                                    yield text_piece, None
                                usage = data.get("usage")
                                if usage:
                                    token_usage["prompt_tokens"]    = usage.get("prompt_tokens", 0)
                                    token_usage["completion_tokens"] = usage.get("completion_tokens", 0)
                    logger.info(f"[GROQ] {model_name} stream done. Tokens: {token_usage}")
                    yield "", token_usage
                    return

            except Exception as e:
                logger.error(f"[{provider.upper()}] {model_name} stream failed: {e}. Trying next...")
                self.model_failures[model_name] = time.time()

        # All providers exhausted
        async for chunk in self._fallback_stream(token_usage):
            yield chunk

    # -----------------------------------------------------------------------
    # Offline fallback helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _fallback_text() -> str:
        return (
            "I'm sorry, I'm experiencing a minor connection issue right now. "
            "Please call us back at +91 9000666914 or send an email to info@fueltracks.in, "
            "and we'll help you immediately. Thank you for your patience."
        )

    async def _fallback_stream(self, token_usage: dict):
        logger.warning("All LLM providers failed — returning graceful offline stream.")
        for word in self._fallback_text().split():
            yield word + " ", None
            await asyncio.sleep(0.04)
        yield "", token_usage
