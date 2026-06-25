import socket
import time

def test_conn(host, port):
    t0 = time.time()
    try:
        s = socket.create_connection((host, port), timeout=5.0)
        print(f"Connected to {host}:{port} in {time.time() - t0:.3f}s")
        s.close()
    except Exception as e:
        print(f"Failed to connect to {host}:{port}: {e} in {time.time() - t0:.3f}s")

test_conn("20.235.220.20", 443)  # Sarvam IP
test_conn("1.1.1.1", 53)          # Cloudflare DNS
test_conn("8.8.8.8", 53)          # Google DNS
test_conn("api.sarvam.ai", 443)   # Sarvam domain (with DNS)

