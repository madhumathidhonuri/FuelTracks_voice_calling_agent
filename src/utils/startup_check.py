import sys
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

def check_env_vars():
    """
    Validate that required environment variables are present and non-empty.
    Exits immediately if any required variable is missing.
    """
    required_keys = {
        "GROQ_API_KEY": settings.GROQ_API_KEY,
        "SARVAM_API_KEY": settings.SARVAM_API_KEY,
        "EXOTEL_ACCOUNT_SID": settings.EXOTEL_ACCOUNT_SID,
        "EXOTEL_API_KEY": settings.EXOTEL_API_KEY,
        "EXOTEL_API_TOKEN": settings.EXOTEL_API_TOKEN,
    }

    for key, value in required_keys.items():
        if not value or "mock_" in str(value):
            print(f"[STARTUP ERROR] {key} is missing from .env\nPlease add it and restart the server.", file=sys.stderr)
            sys.exit(1)
