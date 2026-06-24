import sys
import os
# Add project root directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from src.storage.database import init_db, get_call_logs
from src.telephony.webhook_routes import router as webhook_router
from src.telephony.exotel_handler import router as ws_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Multilingual Voice Calling Agent API",
    description="Production-ready voice agent backend for Exotel + Sarvam AI",
    version="1.0.0"
)

# Enable CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database on Startup
@app.on_event("startup")
def startup_event():
    logger.info("Initializing SQLite pilot database...")
    init_db()
    logger.info("Database initialized successfully.")

# Root health check endpoint
@app.get("/")
async def root_health_check():
    return {
        "status": "online",
        "service": "Voice Calling Agent Backend",
        "telephony": "Exotel (AgentStream WebSocket)",
        "languages": "Multilingual (Indian Code-Mixing natively supported)",
        "stt_engine": "Sarvam AI (saaras:v3)",
        "tts_engine": "Sarvam AI (bulbul:v3)"
    }

# Endpoint for QA review of call transcripts and outcomes
@app.get("/calls/{call_sid}")
async def get_call_history(call_sid: str):
    """
    Get full transcript, detected languages, duration, and metrics for a specific call.
    """
    logs = get_call_logs(call_sid)
    if not logs:
        raise HTTPException(status_code=404, detail="Call record not found.")
    return logs

# Register Router Modules
app.include_router(webhook_router)
app.include_router(ws_router)

if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "src.api.main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=True
    )
