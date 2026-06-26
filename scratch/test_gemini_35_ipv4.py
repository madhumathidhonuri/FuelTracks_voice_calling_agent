import asyncio
import time
import httpx
from google import genai
from google.genai import types
from config.settings import settings
from src.audio.dns_resolver import resolve_hostname_ipv4

class IPv4OnlyAsyncTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request, *args, **kwargs):
        hostname = request.url.host
        # Route both vertex/google endpoints if any
        if "googleapis.com" in hostname or "google.dev" in hostname:
            try:
                resolved_ip = await resolve_hostname_ipv4(hostname)
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
                # Run async resolver in sync environment helper
                # For simplicity, resolve synchronously via socket
                import socket
                resolved_ip = socket.gethostbyname(hostname)
                port = request.url.port or (443 if request.url.scheme == "https" else 80)
                request.extensions["network_address"] = (resolved_ip, port)
                print(f"[Transport Sync] Intercepted {hostname}, resolved to {resolved_ip}")
            except Exception as e:
                print(f"[Transport Sync] Failed to resolve {hostname}: {e}")
        return super().handle_request(request, *args, **kwargs)

async def run():
    print("Initializing GenAI Client with IPv4 transport...")
    
    # Configure options
    http_options = types.HttpOptions(
        client_args={
            "transport": IPv4OnlySyncTransport(),
        },
        async_client_args={
            "transport": IPv4OnlyAsyncTransport(),
        }
    )
    
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=http_options
    )
    
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
