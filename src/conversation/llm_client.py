import logging
import asyncio
from typing import Tuple, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

# Initialize client imports inside try-blocks to avoid startup failure if keys are absent
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class LLMClient:
    def __init__(self):
        self.anthropic_key = settings.ANTHROPIC_API_KEY
        self.gemini_key = settings.GEMINI_API_KEY
        
        # Initialize Anthropic
        self.claude_client = None
        if anthropic and self.anthropic_key and "mock_" not in self.anthropic_key:
            try:
                self.claude_client = anthropic.AsyncAnthropic(api_key=self.anthropic_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                
        # Initialize Gemini
        if genai and self.gemini_key and "mock_" not in self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")

    async def generate_response(
        self, 
        system_prompt: str, 
        messages: list
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response using Claude Haiku (primary) or Gemini Flash (fallback).
        Returns:
            Tuple[response_text, token_usage_dict]
            where token_usage_dict has {"prompt_tokens": int, "completion_tokens": int}
        """
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
        # Format history for Anthropic
        # Anthropic expects roles 'user' and 'assistant'.
        claude_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "customer" else "assistant"
            # Anthropic messages cannot be system messages
            if msg["role"] == "system":
                continue
            claude_messages.append({"role": role, "content": msg["content"]})

        # Anthropic and Gemini require at least one message in history
        if not claude_messages:
            claude_messages = [{"role": "user", "content": "Hello"}]

        # Try Anthropic first
        if self.claude_client:
            try:
                logger.info("Calling Claude Haiku...")
                response = await self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=150,
                    system=system_prompt,
                    messages=claude_messages
                )
                text = response.content[0].text
                token_usage["prompt_tokens"] = response.usage.input_tokens
                token_usage["completion_tokens"] = response.usage.output_tokens
                logger.info(f"Claude Haiku succeeded. Tokens: {token_usage}")
                return text, token_usage
            except Exception as e:
                logger.error(f"Claude Haiku call failed: {e}. Falling back to Gemini...")
        
        # Try Gemini Fallback
        if genai and self.gemini_key and "mock_" not in self.gemini_key:
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    continue
                role = "user" if msg["role"] == "customer" else "model"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})
            
            if not gemini_messages:
                gemini_messages = [{"role": "user", "parts": ["Hello"]}]
            
            gemini_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
            for model_name in gemini_models:
                try:
                    logger.info(f"Calling Gemini Fallback with model: {model_name}...")
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_prompt
                    )
                    
                    response = await model.generate_content_async(gemini_messages)
                    text = response.text
                    
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        token_usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
                        token_usage["completion_tokens"] = response.usage_metadata.candidates_token_count
                        
                    logger.info(f"Gemini model {model_name} succeeded. Tokens: {token_usage}")
                    return text, token_usage
                except Exception as e:
                    logger.error(f"Gemini model {model_name} call failed: {e}. Trying next model...")

        # Fallback to Mock / Offline Response if both clients are unavailable/failed
        logger.warning("No LLM services succeeded. Returning graceful fallback response.")
        fallback_text = (
            "I'm sorry, I'm experiencing a minor connection issue right now. "
            "Please call us back at +91 9000666914 or send an email to info@fueltracks.in, "
            "and we'll help you immediately. Thank you for your patience."
        )
        return fallback_text, token_usage

    async def generate_response_stream(
        self, 
        system_prompt: str, 
        messages: list
    ):
        """
        Generate a streaming response using Claude Haiku (primary) or Gemini Flash (fallback).
        Yields:
            Tuple[text_chunk_str, Optional[token_usage_dict]]
        """
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
        # Format history for Anthropic
        claude_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "customer" else "assistant"
            if msg["role"] == "system":
                continue
            claude_messages.append({"role": role, "content": msg["content"]})

        # Anthropic and Gemini require at least one message in history
        if not claude_messages:
            claude_messages = [{"role": "user", "content": "Hello"}]

        # Try Anthropic first
        if self.claude_client:
            try:
                logger.info("Calling Claude Haiku Stream...")
                async with self.claude_client.messages.stream(
                    model="claude-3-haiku-20240307",
                    max_tokens=150,
                    system=system_prompt,
                    messages=claude_messages
                ) as stream:
                    async for event in stream:
                        if event.type == "content_block_delta":
                            yield event.delta.text, None
                    
                    message = await stream.get_final_message()
                    token_usage["prompt_tokens"] = message.usage.input_tokens
                    token_usage["completion_tokens"] = message.usage.output_tokens
                    logger.info(f"Claude Haiku Stream finished. Tokens: {token_usage}")
                    yield "", token_usage
                    return
            except Exception as e:
                logger.error(f"Claude Haiku streaming failed: {e}. Falling back to Gemini stream...")
        
        # Try Gemini Fallback
        if genai and self.gemini_key and "mock_" not in self.gemini_key:
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    continue
                role = "user" if msg["role"] == "customer" else "model"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})
            
            if not gemini_messages:
                gemini_messages = [{"role": "user", "parts": ["Hello"]}]
            
            gemini_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
            for model_name in gemini_models:
                try:
                    logger.info(f"Calling Gemini Stream Fallback with model: {model_name}...")
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_prompt
                    )
                    
                    response = await model.generate_content_async(gemini_messages, stream=True)
                    async for chunk in response:
                        if chunk.text:
                            yield chunk.text, None
                    
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        token_usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
                        token_usage["completion_tokens"] = response.usage_metadata.candidates_token_count
                    
                    logger.info(f"Gemini model {model_name} Stream finished. Tokens: {token_usage}")
                    yield "", token_usage
                    return
                except Exception as e:
                    logger.error(f"Gemini model {model_name} streaming failed: {e}. Trying next model...")
        
        # Fallback to Mock / Offline Streaming Response
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
