import asyncio
import time
from google import genai
from config.settings import settings

async def run():
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    t0 = time.time()
    try:
        response = await client.aio.models.generate_content_stream(
            model="gemini-3.5-flash",
            contents="Say hello in three words.",
        )
        first_chunk_time = None
        chunks = []
        async for chunk in response:
            if chunk.text:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - t0
                chunks.append(chunk.text)
        total_time = time.time() - t0
        print(f"Text: {''.join(chunks)}")
        print(f"First chunk latency: {first_chunk_time:.3f}s")
        print(f"Total stream duration: {total_time:.3f}s")
    except Exception as e:
        print(f"Failed: {e} in {time.time() - t0:.3f}s")

if __name__ == "__main__":
    asyncio.run(run())
