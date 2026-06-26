import asyncio
from google import genai
from google.genai import types
from config.settings import settings

async def run():
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    # 1. Test using Content objects
    messages = [
        types.Content(role="user", parts=[types.Part.from_text(text="Hello, my name is John.")]),
        types.Content(role="model", parts=[types.Part.from_text(text="Hello John! How can I help you today?")]),
        types.Content(role="user", parts=[types.Part.from_text(text="What is my name? Answer in one word.")])
    ]
    
    print("Testing with structured Content objects:")
    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
            )
        )
        print(f"Response text: '{response.text}'")
        print(f"Full response object: {response}")
    except Exception as e:
        print(f"Failed with Content objects: {e}")

    # 2. Test using dict format
    messages_dict = [
        {"role": "user", "parts": [{"text": "Hello, my name is John."}]},
        {"role": "model", "parts": [{"text": "Hello John! How can I help you today?"}]},
        {"role": "user", "parts": [{"text": "What is my name? Answer in one word."}]}
    ]
    print("\nTesting with raw dicts:")
    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.5-flash",
            contents=messages_dict,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
            )
        )
        print(f"Response text: '{response.text}'")
    except Exception as e:
        print(f"Failed with dicts: {e}")

if __name__ == "__main__":
    asyncio.run(run())
