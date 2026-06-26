import asyncio
import os
import sys

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

try:
    import httpx
except ImportError:
    print("httpx is not installed. Run: pip install httpx")
    sys.exit(1)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

async def test_key():
    key = settings.GROQ_API_KEY
    print(f"Loading GROQ_API_KEY from .env: {key[:10]}...{key[-5:] if len(key) > 5 else ''}")

    if not key or "mock_" in key:
        print("[ERROR] No valid GROQ_API_KEY found in .env.")
        sys.exit(1)

    print("\nAttempting to call Groq API with model 'llama-3.1-8b-instant'...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": "Say hello in one word."}],
                    "max_tokens": 10,
                }
            )

        if response.status_code == 200:
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            print(f"\n[SUCCESS] Connected to Groq API successfully!")
            print(f"  Model reply  : {reply}")
            print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  Output tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"\nYour GROQ_API_KEY is working. Update .env and restart the server.")
        elif response.status_code == 401:
            print(f"\n[FAILED] Authentication error (401). Your GROQ_API_KEY is invalid.")
            print("  Get a valid key from: https://console.groq.com/keys")
        elif response.status_code == 429:
            print(f"\n[FAILED] Rate limit hit (429). The key works but is temporarily throttled.")
            print("  Wait 60 seconds and try again.")
        else:
            print(f"\n[FAILED] Groq API returned HTTP {response.status_code}:")
            print(f"  {response.text[:500]}")

    except Exception as e:
        print(f"\n[FAILED] Could not connect to Groq API:")
        print(f"  {e}")

if __name__ == "__main__":
    asyncio.run(test_key())
