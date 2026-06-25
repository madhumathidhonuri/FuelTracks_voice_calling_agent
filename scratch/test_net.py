import httpx
import time
import asyncio

async def run():
    async with httpx.AsyncClient(timeout=30.0) as client:
        t0 = time.time()
        try:
            r = await client.get('https://api.sarvam.ai/')
            print('GET 1 time:', time.time() - t0, r.status_code)
        except Exception as e:
            print('GET 1 error:', time.time() - t0, str(e))
            
        t1 = time.time()
        try:
            r = await client.get('https://api.sarvam.ai/')
            print('GET 2 time:', time.time() - t1, r.status_code)
        except Exception as e:
            print('GET 2 error:', time.time() - t1, str(e))

asyncio.run(run())
