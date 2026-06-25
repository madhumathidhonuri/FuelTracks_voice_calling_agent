import httpx
import time
import asyncio
from config.settings import settings
from src.audio.dns_resolver import resolve_hostname_ipv4
from urllib.parse import urlparse
import google.generativeai as genai

async def test_gemini():
    genai.configure(api_key=settings.GEMINI_API_KEY)
    t0 = time.time()
    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        response = model.generate_content("Say hello in one word.")
        print(f"Gemini response: {response.text.strip()} in {time.time() - t0:.3f}s")
    except Exception as e:
        print(f"Gemini failed: {e} in {time.time() - t0:.3f}s")

async def test_sarvam_tts():
    api_key = settings.SARVAM_API_KEY
    url = "https://api.sarvam.ai/text-to-speech"
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or "api.sarvam.ai"
    port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    resolved_ip = await resolve_hostname_ipv4(hostname)
    
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": api_key
    }
    payload = {
        "text": "Hello, good afternoon!",
        "target_language_code": "en-IN",
        "speaker": "shreya",
        "model": "bulbul:v3",
        "pace": 1.05
    }
    
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url, 
            headers=headers, 
            json=payload,
            extensions={"network_address": (resolved_ip, port)}
        )
        print(f"Sarvam TTS status: {response.status_code} in {time.time() - t0:.3f}s")

async def run():
    print("--- Starting Latency Test ---")
    await test_gemini()
    await test_sarvam_tts()

asyncio.run(run())
