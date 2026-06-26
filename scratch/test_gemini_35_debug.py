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
        print("Iterating over stream:")
        async for chunk in response:
            print(f"Chunk: {type(chunk)}")
            # Try printing attributes of chunk
            print(f"  text: {getattr(chunk, 'text', None)}")
            print(f"  candidates: {getattr(chunk, 'candidates', None)}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(run())
