import asyncio
import time
import httpx
import socket
from google import genai
from google.genai import types
from config.settings import settings

async def resolve_ipv4_fast(hostname: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, socket.gethostbyname, hostname)

class IPv4OnlyAsyncTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request, *args, **kwargs):
        hostname = request.url.host
        if "googleapis.com" in hostname or "google.dev" in hostname:
            try:
                resolved_ip = await resolve_ipv4_fast(hostname)
                port = request.url.port or (443 if request.url.scheme == "https" else 80)
                request.extensions["network_address"] = (resolved_ip, port)
                print(f"[Transport] Intercepted {hostname}, resolved to {resolved_ip}")
            except Exception as e:
                print(f"[Transport] Failed to resolve {hostname}: {e}")
        return await super().handle_async_request(request, *args, **kwargs)

class IPv4OnlySyncTransport(httpx.HTTPTransport):
    def handle_request(self, request, *args, **kwargs):
        hostname = request.url.host
        if "googleapis.com" in hostname or "google.dev" in hostname:
            try:
                resolved_ip = socket.gethostbyname(hostname)
                port = request.url.port or (443 if request.url.scheme == "https" else 80)
                request.extensions["network_address"] = (resolved_ip, port)
                print(f"[Transport Sync] Intercepted {hostname}, resolved to {resolved_ip}")
            except Exception as e:
                print(f"[Transport Sync] Failed to resolve {hostname}: {e}")
        return super().handle_request(request, *args, **kwargs)

async def run():
    print("Initializing GenAI Client with FAST IPv4 transport...")
    
    http_options = types.HttpOptions(
        client_args={"transport": IPv4OnlySyncTransport()},
        async_client_args={"transport": IPv4OnlyAsyncTransport()}
    )
    
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=http_options
    )
    
    models = ["gemini-flash-latest", "gemini-flash-lite-latest"]
    for model_name in models:
        print(f"\n--- Testing {model_name} ---")
        t0 = time.time()
        try:
            response = await client.aio.models.generate_content_stream(
                model=model_name,
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
