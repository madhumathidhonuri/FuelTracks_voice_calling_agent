"""
scripts/update_ngrok_url.py
-----------------------------
Startup helper that auto-detects the active ngrok tunnel URL and rewrites
WEBSOCKET_URL in the project's .env file — so Exotel always receives the
correct public WebSocket endpoint without manual edits.

Usage:
    python scripts/update_ngrok_url.py            # Detect and patch .env
    python scripts/update_ngrok_url.py --dry-run  # Print detected URL only

Requirements:
    - ngrok must already be running (ngrok http 8000)
    - The local ngrok API defaults to http://127.0.0.1:4040
"""
import sys
import re
import time
import argparse
import logging
import urllib.request
import urllib.error
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("update_ngrok_url")

# ---------------------------------------------------------------------------
# Project root (one level up from this script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
NGROK_API_URL = "http://127.0.0.1:4040/api/tunnels"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _fetch_ngrok_tunnels(api_url: str = NGROK_API_URL, retries: int = 5, delay: float = 1.5) -> dict:
    """
    Poll the ngrok local API until tunnels are available or retries exhausted.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(api_url, timeout=3) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionRefusedError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt}/{retries}: ngrok API not reachable ({e}). Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"Could not reach ngrok API at {api_url} after {retries} attempts. Last error: {last_error}")


def detect_ngrok_https_url(api_url: str = NGROK_API_URL) -> str:
    """
    Inspect ngrok tunnels and return the public HTTPS URL.
    Raises RuntimeError if no HTTPS tunnel is found.
    """
    data = _fetch_ngrok_tunnels(api_url)
    tunnels = data.get("tunnels", [])

    if not tunnels:
        raise RuntimeError("ngrok is running but no tunnels found. Did you run 'ngrok http 8000'?")

    for tunnel in tunnels:
        public_url: str = tunnel.get("public_url", "")
        if public_url.startswith("https://"):
            logger.info(f"Found ngrok HTTPS tunnel: {public_url}")
            return public_url

    raise RuntimeError(
        f"No HTTPS tunnel found. Available tunnels: {[t.get('public_url') for t in tunnels]}"
    )


def https_to_wss(https_url: str) -> str:
    """Convert https://xxx.ngrok-free.app → wss://xxx.ngrok-free.app/ws/media"""
    wss_url = "wss://" + https_url.removeprefix("https://") + "/ws/media"
    return wss_url


# ---------------------------------------------------------------------------
# .env patcher
# ---------------------------------------------------------------------------

def patch_env_file(env_path: Path, key: str, new_value: str) -> bool:
    """
    Rewrite `key=...` line in the .env file in-place.
    Appends a new line if the key is not already present.

    Returns True if the file was modified, False if value was already correct.
    """
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")

    content = env_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    replacement_line = f"{key}={new_value}"

    if pattern.search(content):
        current_line = pattern.search(content).group(0)
        if current_line == replacement_line:
            logger.info(f".env already has correct value: {replacement_line}")
            return False
        new_content = pattern.sub(replacement_line, content)
    else:
        # Key not present — append it
        new_content = content.rstrip("\n") + f"\n{replacement_line}\n"

    env_path.write_text(new_content, encoding="utf-8")
    logger.info(f"Patched .env: {replacement_line}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-detect ngrok tunnel and patch .env WEBSOCKET_URL")
    parser.add_argument("--dry-run", action="store_true", help="Print detected URL without modifying .env")
    parser.add_argument("--ngrok-api", default=NGROK_API_URL, help="ngrok local API URL")
    parser.add_argument("--env-file", default=str(ENV_FILE), help="Path to .env file")
    args = parser.parse_args()

    try:
        https_url = detect_ngrok_https_url(api_url=args.ngrok_api)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    wss_url = https_to_wss(https_url)
    logger.info(f"Derived WebSocket URL: {wss_url}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would set WEBSOCKET_URL={wss_url}")
        return

    env_path = Path(args.env_file)
    try:
        modified = patch_env_file(env_path, "WEBSOCKET_URL", wss_url)
        if modified:
            print(f"\n✅ .env updated: WEBSOCKET_URL={wss_url}")
        else:
            print(f"\n✅ .env already up-to-date: WEBSOCKET_URL={wss_url}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
