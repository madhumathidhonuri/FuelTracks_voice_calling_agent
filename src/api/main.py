import sys
import os
# Add project root directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uvicorn
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from src.storage.database import init_db, get_call_logs
from src.telephony.webhook_routes import router as webhook_router
from src.telephony.exotel_handler import router as ws_router
import shutil
import os
import asyncio
from pathlib import Path

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

# Global campaign state
campaign_status = {
    "status": "idle", # "idle", "running", "completed"
    "total": 0,
    "processed": 0,
    "success": 0,
    "failed": 0,
    "details": []
}

async def run_campaign_task(customers: list):
    global campaign_status
    campaign_status["status"] = "running"
    campaign_status["total"] = len(customers)
    campaign_status["processed"] = 0
    campaign_status["success"] = 0
    campaign_status["failed"] = 0
    campaign_status["details"] = []
    
    from src.telephony.outbound_caller import OutboundCaller
    caller = OutboundCaller()
    
    for cust in customers:
        if campaign_status["status"] != "running":
            break
            
        phone = cust.get("phone")
        name = cust.get("name", "Customer")
        prod = cust.get("product_interest")
        c_type = cust.get("call_type", "lead_followup")
        
        try:
            result = await caller.initiate_call(
                customer_number=phone,
                customer_name=name,
                product_interest=prod,
                call_type=c_type
            )
            if result.get("success"):
                campaign_status["success"] += 1
                campaign_status["details"].append({
                    "phone": phone,
                    "name": name,
                    "status": "success",
                    "call_sid": result["call_sid"]
                })
            else:
                campaign_status["failed"] += 1
                campaign_status["details"].append({
                    "phone": phone,
                    "name": name,
                    "status": "failed",
                    "error": result.get("error")
                })
        except Exception as e:
            campaign_status["failed"] += 1
            campaign_status["details"].append({
                "phone": phone,
                "name": name,
                "status": "error",
                "error": str(e)
            })
        
        campaign_status["processed"] += 1
        await asyncio.sleep(1.0)
        
    campaign_status["status"] = "completed"

# Serve the dashboard page
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    if not template_path.exists():
        # Fallback inline minimal HTML if file is missing
        return HTMLResponse("<h1>Dashboard Not Found</h1><p>Ensure template exists in templates/dashboard.html</p>", status_code=404)
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

# API Health endpoint
@app.get("/api/health")
async def api_health():
    return {
        "status": "online",
        "service": "Voice Calling Agent Backend",
        "telephony": "Exotel (AgentStream WebSocket)",
        "languages": "Multilingual (Indian Code-Mixing natively supported)",
        "stt_engine": "Sarvam AI (saaras:v3)",
        "tts_engine": "Sarvam AI (bulbul:v3)"
    }

# Parse campaign excel file
@app.post("/api/campaign/parse")
async def parse_campaign_excel(
    file: UploadFile = File(...),
    default_call_type: str = Form("lead_followup"),
    default_product: str = Form(None)
):
    try:
        # Save file temporarily
        temp_dir = Path(settings.BASE_DIR) / "scratch"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Infer default call type from file name if default_call_type was not customized
        filename_lower = file.filename.lower()
        inferred_type = default_call_type
        if default_call_type == "lead_followup":
            if "marketing" in filename_lower or "promo" in filename_lower:
                inferred_type = "marketing"
            elif "support" in filename_lower or "service" in filename_lower:
                inferred_type = "support"
            elif "dealer" in filename_lower or "recruit" in filename_lower or "partner" in filename_lower:
                inferred_type = "dealer_recruitment"
                
        # Parse Excel using helper
        from scripts.bulk_outbound_caller import parse_excel_file
        customers = parse_excel_file(str(temp_path), default_call_type=inferred_type, default_product=default_product)
        
        # Clean up file
        if temp_path.exists():
            os.remove(temp_path)
            
        return {
            "success": True,
            "filename": file.filename,
            "inferred_call_type": inferred_type,
            "count": len(customers),
            "customers": customers
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {str(e)}")

# Start campaign execution
@app.post("/api/campaign/start")
async def start_campaign(
    payload: dict,
    background_tasks: BackgroundTasks
):
    global campaign_status
    if campaign_status["status"] == "running":
        raise HTTPException(status_code=400, detail="A campaign is already running.")
        
    customers = payload.get("customers", [])
    if not customers:
        raise HTTPException(status_code=400, detail="No customer list provided.")
        
    # Start campaign task in the background
    background_tasks.add_task(run_campaign_task, customers)
    return {"success": True, "message": "Campaign started in background."}

# Get campaign status
@app.get("/api/campaign/status")
async def get_campaign_status():
    global campaign_status
    return campaign_status

# Get recent calls list
@app.get("/api/calls/recent")
async def get_recent_calls_api(limit: int = 50):
    from src.storage.database import get_recent_calls
    return get_recent_calls(limit=limit)

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
