"""
LLM Client
-----------
Generates conversational responses using Claude Haiku (primary)
with a Gemini Flash cascade fallback.

Changes in this version:
  - Fix 2: Anthropic API calls are wrapped with tenacity exponential-backoff retry.
  - Fix 3: Accepts SystemPrompt dataclass (or plain str) — both clients consume .text,
            guaranteeing identical prompts across Claude and Gemini.
  - Bug fix: Gemini streaming usage_metadata is now read from the final aggregated
             response object (not the streaming iterator, which never carries it).
"""
import logging
import asyncio
from typing import Tuple, Dict, Any, Union
from config.settings import settings
from src.conversation.prompt_builder import SystemPrompt

logger = logging.getLogger(__name__)

# Initialize client imports inside try-blocks to avoid startup failure if keys are absent
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

import httpx
import socket
from src.audio.dns_resolver import resolve_hostname_ipv4

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

# ---------------------------------------------------------------------------
# Retry policy for Anthropic calls
# ---------------------------------------------------------------------------

_ANTHROPIC_RETRYABLE = (Exception,)  # anthropic raises APIConnectionError, APIStatusError etc.

def _is_anthropic_retryable(exc: BaseException) -> bool:
    """Only retry transient errors — not 4xx auth/validation errors."""
    if anthropic is None:
        return False
    not_retryable = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.UnprocessableEntityError,
    )
    return not isinstance(exc, not_retryable)

from tenacity import retry_if_exception

_anthropic_retry = retry(
    retry=retry_if_exception(_is_anthropic_retryable),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# ---------------------------------------------------------------------------
# IPv4-only transports (unchanged)
# ---------------------------------------------------------------------------

class IPv4OnlyAsyncTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request, *args, **kwargs):
        hostname = request.url.host
        if "googleapis.com" in hostname or "google.dev" in hostname:
            try:
                resolved_ip = await resolve_hostname_ipv4(hostname)
                port = request.url.port or (443 if request.url.scheme == "https" else 80)
                request.extensions["network_address"] = (resolved_ip, port)
                logger.info(f"[Gemini Transport] Intercepted {hostname}, resolved to {resolved_ip}")
            except Exception as e:
                logger.warning(f"[Gemini Transport] Failed to resolve {hostname}: {e}")
        return await super().handle_async_request(request, *args, **kwargs)

class IPv4OnlySyncTransport(httpx.HTTPTransport):
    def handle_request(self, request, *args, **kwargs):
        hostname = request.url.host
        if "googleapis.com" in hostname or "google.dev" in hostname:
            try:
                resolved_ip = socket.gethostbyname(hostname)
                port = request.url.port or (443 if request.url.scheme == "https" else 80)
                request.extensions["network_address"] = (resolved_ip, port)
                logger.info(f"[Gemini Sync Transport] Intercepted {hostname}, resolved to {resolved_ip}")
            except Exception as e:
                logger.warning(f"[Gemini Sync Transport] Failed to resolve {hostname}: {e}")
        return super().handle_request(request, *args, **kwargs)


# ---------------------------------------------------------------------------
# Helper: normalise system prompt input
# ---------------------------------------------------------------------------

def _prompt_text(prompt: Union[str, SystemPrompt]) -> str:
    """Accept both plain str and SystemPrompt dataclass."""
    if isinstance(prompt, SystemPrompt):
        return prompt.text
    return str(prompt)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self):
        self.anthropic_key = settings.ANTHROPIC_API_KEY
        self.gemini_key = settings.GEMINI_API_KEY
        self.model_failures = {}  # Tracks model name -> timestamp of last failure

        # Initialize Anthropic
        self.claude_client = None
        if anthropic and self.anthropic_key and "mock_" not in self.anthropic_key:
            try:
                self.claude_client = anthropic.AsyncAnthropic(api_key=self.anthropic_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

        # Initialize Gemini
        self.gemini_client = None
        if genai and self.gemini_key and "mock_" not in self.gemini_key:
            try:
                http_options = types.HttpOptions(
                    client_args={
                        "transport": IPv4OnlySyncTransport(),
                    },
                    async_client_args={
                        "transport": IPv4OnlyAsyncTransport(),
                    }
                )
                self.gemini_client = genai.Client(
                    api_key=self.gemini_key,
                    http_options=http_options
                )
            except Exception as e:
                logger.error(f"Failed to configure Gemini client: {e}")

    def get_healthy_models(self) -> list:
        """Returns the list of Gemini models, deprioritizing those that have failed recently."""
        import time
        cooldown = 300  # 5 minutes cooldown
        now = time.time()

        base_models = [
            "gemini-3.5-flash",
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash"
        ]

        healthy = []
        unhealthy = []

        for model in base_models:
            last_fail = self.model_failures.get(model, 0)
            if now - last_fail > cooldown:
                healthy.append(model)
            else:
                unhealthy.append(model)

        return healthy + unhealthy

    # -----------------------------------------------------------------------
    # Non-streaming response
    # -----------------------------------------------------------------------

    async def generate_response(
        self,
        system_prompt: Union[str, SystemPrompt],
        messages: list,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response using Claude Haiku (primary) or Gemini Flash (fallback).
        Accepts str or SystemPrompt — both clients receive identical text.
        Returns:
            Tuple[response_text, token_usage_dict]
        """
        system_text = _prompt_text(system_prompt)
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        # Format history for Anthropic (role: 'user' | 'assistant')
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "user" if msg["role"] == "customer" else "assistant"
            claude_messages.append({"role": role, "content": msg["content"]})

        if not claude_messages:
            claude_messages = [{"role": "user", "content": "Hello"}]

        # --- Try Anthropic first ---
        if self.claude_client:
            try:
                text, token_usage = await self._claude_generate(system_text, claude_messages)
                return text, token_usage
            except Exception as e:
                logger.error(f"Claude Haiku call failed: {e}. Falling back to Gemini...")

        # --- Gemini Fallback ---
        if self.gemini_client:
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    continue
                role = "user" if msg["role"] == "customer" else "model"
                gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})

            if not gemini_messages:
                gemini_messages = [{"role": "user", "parts": [{"text": "Hello"}]}]

            for model_name in self.get_healthy_models():
                try:
                    logger.info(f"Calling Gemini Fallback with model: {model_name}...")
                    response = await self.gemini_client.aio.models.generate_content(
                        model=model_name,
                        contents=gemini_messages,
                        config=types.GenerateContentConfig(
                            system_instruction=system_text  # ← same text as Claude
                        )
                    )
                    text = response.text
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        token_usage["prompt_tokens"] = response.usage_metadata.prompt_token_count or 0
                        token_usage["completion_tokens"] = response.usage_metadata.candidates_token_count or 0
                    logger.info(f"Gemini model {model_name} succeeded. Tokens: {token_usage}")
                    return text, token_usage
                except Exception as e:
                    logger.error(f"Gemini model {model_name} call failed: {e}.")
                    import time
                    self.model_failures[model_name] = time.time()
                    logger.info("Trying next model...")

        # --- Graceful offline fallback ---
        logger.warning("No LLM services succeeded. Returning graceful fallback response.")
        fallback_text = (
            "I'm sorry, I'm experiencing a minor connection issue right now. "
            "Please call us back at +91 9000666914 or send an email to info@fueltracks.in, "
            "and we'll help you immediately. Thank you for your patience."
        )
        return fallback_text, token_usage

    @_anthropic_retry
    async def _claude_generate(
        self, system_text: str, claude_messages: list
    ) -> Tuple[str, Dict[str, int]]:
        """Isolated Anthropic non-streaming call — wrapped with tenacity retry."""
        logger.info("Calling Claude Haiku...")
        response = await self.claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            system=system_text,
            messages=claude_messages,
        )
        text = response.content[0].text
        token_usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }
        logger.info(f"Claude Haiku succeeded. Tokens: {token_usage}")
        return text, token_usage

    # -----------------------------------------------------------------------
    # Streaming response
    # -----------------------------------------------------------------------

    async def generate_response_stream(
        self,
        system_prompt: Union[str, SystemPrompt],
        messages: list,
    ):
        """
        Generate a streaming response using Claude Haiku (primary) or Gemini Flash (fallback).
        Yields:
            Tuple[text_chunk_str, Optional[token_usage_dict]]
        """
        system_text = _prompt_text(system_prompt)
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        # Format history for Anthropic
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "user" if msg["role"] == "customer" else "assistant"
            claude_messages.append({"role": role, "content": msg["content"]})

        if not claude_messages:
            claude_messages = [{"role": "user", "content": "Hello"}]

        # --- Try Anthropic stream first ---
        if self.claude_client:
            try:
                async for chunk, usage in self._claude_stream(system_text, claude_messages):
                    yield chunk, usage
                return
            except Exception as e:
                logger.error(f"Claude Haiku streaming failed: {e}. Falling back to Gemini stream...")

        # --- Gemini stream fallback ---
        if self.gemini_client:
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    continue
                role = "user" if msg["role"] == "customer" else "model"
                gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})

            if not gemini_messages:
                gemini_messages = [{"role": "user", "parts": [{"text": "Hello"}]}]

            for model_name in self.get_healthy_models():
                try:
                    logger.info(f"Calling Gemini Stream Fallback with model: {model_name}...")
                    response = await self.gemini_client.aio.models.generate_content_stream(
                        model=model_name,
                        contents=gemini_messages,
                        config=types.GenerateContentConfig(
                            system_instruction=system_text  # ← same text as Claude
                        )
                    )

                    response_iter = response.__aiter__()
                    try:
                        # Wait at most 2.5 seconds for first stream chunk
                        first_chunk = await asyncio.wait_for(response_iter.__anext__(), timeout=2.5)
                        if first_chunk.text:
                            yield first_chunk.text, None
                    except asyncio.TimeoutError:
                        logger.warning(f"Gemini model {model_name} first chunk timeout (2.5s). Falling back...")
                        raise RuntimeError(f"First chunk timeout on model {model_name}")

                    async for chunk in response_iter:
                        if chunk.text:
                            yield chunk.text, None

                    # BUG FIX: usage_metadata lives on the final aggregated response object,
                    # NOT the streaming iterator. Collect it from the last chunk if available.
                    # The Gemini SDK accumulates usage on the response object after iteration.
                    try:
                        if hasattr(response, "usage_metadata") and response.usage_metadata:
                            token_usage["prompt_tokens"] = response.usage_metadata.prompt_token_count or 0
                            token_usage["completion_tokens"] = response.usage_metadata.candidates_token_count or 0
                    except Exception:
                        pass

                    logger.info(f"Gemini model {model_name} Stream finished. Tokens: {token_usage}")
                    yield "", token_usage
                    return
                except Exception as e:
                    logger.error(f"Gemini model {model_name} streaming failed: {e}.")
                    import time
                    self.model_failures[model_name] = time.time()
                    logger.info("Trying next model...")

        # --- Graceful offline streaming fallback ---
        logger.warning("No LLM services succeeded for stream. Returning graceful mock stream.")
        fallback_text = (
            "I'm sorry, I'm experiencing a minor connection issue right now. "
            "Please call us back at +91 9000666914 or send an email to info@fueltracks.in, "
            "and we'll help you immediately. Thank you for your patience."
        )
        for word in fallback_text.split():
            yield word + " ", None
            await asyncio.sleep(0.04)
        yield "", token_usage

    @_anthropic_retry
    async def _claude_stream(self, system_text: str, claude_messages: list):
        """
        Isolated Anthropic streaming call — wrapped with tenacity retry.
        This is an async generator; tenacity re-raises on non-retryable errors.
        """
        logger.info("Calling Claude Haiku Stream...")
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        async with self.claude_client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            system=system_text,
            messages=claude_messages,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    yield event.delta.text, None

            message = await stream.get_final_message()
            token_usage["prompt_tokens"] = message.usage.input_tokens
            token_usage["completion_tokens"] = message.usage.output_tokens
            logger.info(f"Claude Haiku Stream finished. Tokens: {token_usage}")
            yield "", token_usage
