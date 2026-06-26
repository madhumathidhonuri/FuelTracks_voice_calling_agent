import os
from pathlib import Path
from dotenv import load_dotenv

# Load env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
loaded = load_dotenv(dotenv_path=env_path, override=True)

class Settings:
    EXOTEL_ACCOUNT_SID: str = os.getenv("EXOTEL_ACCOUNT_SID", "")
    EXOTEL_API_KEY: str = os.getenv("EXOTEL_API_KEY", "")
    EXOTEL_API_TOKEN: str = os.getenv("EXOTEL_API_TOKEN", "")
    EXOTEL_CALLER_ID: str = os.getenv("EXOTEL_CALLER_ID", "")
    EXOTEL_API_URL: str = os.getenv("EXOTEL_API_URL", "https://api.in.exotel.com")
    
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///voice_calling.db")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WEBSOCKET_URL: str = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws/media")
    VAD_MODE: str = os.getenv("VAD_MODE", "silero")
    
    # Path configuration
    BASE_DIR: Path = BASE_DIR
    PROMPTS_DIR: Path = BASE_DIR / "config" / "prompts"
    STYLE_TEMPLATES_DIR: Path = PROMPTS_DIR / "style_templates"

settings = Settings()
