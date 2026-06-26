import asyncio
import time
from google import genai
from google.genai import types
from config.settings import settings

async def test_stream(client, model_name):
    t0 = time.time()
    try:
        messages = [
            {"role": "user", "parts": [{"text": "Hello, my name is John."}]},
            {"role": "model", "parts": [{"text": "Hello John! How can I help you today?"}]},
            {"role": "user", "parts": [{"text": "What is my name? Answer in one short sentence."}]}
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
        first_chunk_time = None
        async for chunk in response:
            if chunk.text:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - t0
                chunks.append(chunk.text)
        total_time = time.time() - t0
        print(f"[{model_name}] Stream succeeded: {''.join(chunks).strip()}")
        first_chunk_str = f"{first_chunk_time:.3f}s" if first_chunk_time is not None else "N/A"
        print(f"  First chunk: {first_chunk_str}, Total: {total_time:.3f}s")
    except Exception as e:
        print(f"[{model_name}] Stream failed: {e} in {time.time() - t0:.3f}s")

async def run():
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    models = ["gemini-3.1-flash-lite"]
    for m in models:
        await test_stream(client, m)

if __name__ == "__main__":
    asyncio.run(run())

