import asyncio
from google import genai
from config.settings import settings

async def run():
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    print("Listing models:")
    try:
        # client.models.list() returns models
        for m in client.models.list():
            print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    asyncio.run(run())
