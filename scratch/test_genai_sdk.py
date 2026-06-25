import asyncio
import time
from config.settings import settings
from google import genai
from google.genai import types

async def test_chat_manual_dict(client, model_name):
    t0 = time.time()
    try:
        messages = [
            {"role": "user", "parts": [{"text": "Hello, my name is John."}]},
            {"role": "model", "parts": [{"text": "Hello John! How can I help you today?"}]},
            {"role": "user", "parts": [{"text": "What is my name?"}]}
        ]
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
                max_output_tokens=20
            )
        )
        print(f"Chat manual dict with {model_name} succeeded: {response.text.strip()} in {time.time() - t0:.3f}s")
        return True
    except Exception as e:
        print(f"Chat manual dict with {model_name} failed: {e} in {time.time() - t0:.3f}s")
        return False

async def test_chat_stream_manual_dict(client, model_name):
    t0 = time.time()
    try:
        messages = [
            {"role": "user", "parts": [{"text": "Hello, my name is John."}]},
            {"role": "model", "parts": [{"text": "Hello John! How can I help you today?"}]},
            {"role": "user", "parts": [{"text": "What is my name?"}]}
        ]
        response = await client.aio.models.generate_content_stream(
            model=model_name,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
                max_output_tokens=20
            )
        )
        chunks = []
        async for chunk in response:
            if chunk.text:
                chunks.append(chunk.text)
        print(f"Chat stream manual dict with {model_name} succeeded: {''.join(chunks).strip()} in {time.time() - t0:.3f}s")
        return True
    except Exception as e:
        print(f"Chat stream manual dict with {model_name} failed: {e} in {time.time() - t0:.3f}s")
        return False

async def run():
    print("--- Testing google-genai Manual Dict History ---")
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    # We know gemini-2.5-flash-lite was working and not fully exhausted
    model = "gemini-2.5-flash-lite"
    await test_chat_manual_dict(client, model)
    await test_chat_stream_manual_dict(client, model)

if __name__ == "__main__":
    asyncio.run(run())
