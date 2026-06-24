import logging
from typing import Optional
from fastapi import APIRouter, Form, Request, Query, HTTPException
from fastapi.responses import JSONResponse
from config.settings import settings
from src.telephony.call_manager import call_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice")

@router.post("/inbound")
async def handle_inbound_call(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
    Direction: Optional[str] = Form(None),
    call_type: Optional[str] = Form(None),
    customer_name: Optional[str] = Form(None),
    product_interest: Optional[str] = Form(None)
):
    """
    Exotel webhook for call initialization (both inbound and outbound calls).
    Returns the WebSocket stream URL that Exotel will connect to.
    """
    logger.info(f"Incoming call webhook: CallSid={CallSid}, From={From}, To={To}, Direction={Direction}")
    
    # 1. Determine call type
    # Outbound calls are usually initiated with a pre-configured call_type.
    # Inbound calls default to 'inbound_routing' to ask a routing question.
    resolved_call_type = call_type or "inbound_routing"
    
    # Check if there are query parameters in the request URL that override
    qp = dict(request.query_params)
    if "call_type" in qp:
        resolved_call_type = qp["call_type"]
        
    # 2. Create the session
    session = call_manager.create_session(
        call_sid=CallSid,
        from_number=From,
        to_number=To,
        call_type=resolved_call_type
    )
    
    # 3. Store context parameters if provided
    if customer_name:
        session.conversation_manager.context["customer_name"] = customer_name
    if product_interest:
        session.conversation_manager.context["product_interest"] = product_interest
        
    # Also save any context passed via query parameters
    for k, v in qp.items():
        if k not in ["call_sid", "call_type"]:
            session.conversation_manager.context[k] = v
            
    # 4. Generate the WebSocket URL
    # We append the call_sid so the WebSocket connection can identify this call session
    ws_url = f"{settings.WEBSOCKET_URL}?call_sid={CallSid}"
    
    logger.info(f"Returning WebSocket URL: {ws_url} for CallSid={CallSid}")
    return JSONResponse(content={"url": ws_url})


@router.post("/outbound-status")
async def handle_outbound_status(
    CallSid: str = Form(...),
    Status: str = Form(...),
    DetailedStatus: Optional[str] = Form(None)
):
    """
    Exotel webhook tracking call progress (ringing, answered, failed).
    """
    logger.info(f"Call {CallSid} status update: {Status} (Detail: {DetailedStatus})")
    
    session = call_manager.get_session_by_call(CallSid)
    if session:
        session.touch()
        if Status in ["failed", "no-answer", "busy"]:
            # Close the session if the call failed
            stream_sid = session.stream_sid or f"failed_{CallSid}"
            call_manager.close_session(stream_sid, outcome=Status)
            
    return JSONResponse(content={"status": "ok"})


@router.post("/event")
async def handle_call_event(
    CallSid: str = Form(...),
    Event: str = Form(...),
    Reason: Optional[str] = Form(None)
):
    """
    Exotel webhook for general call events (e.g. call end, hangup).
    """
    logger.info(f"Call {CallSid} event: {Event} (Reason: {Reason})")
    
    session = call_manager.get_session_by_call(CallSid)
    if session:
        session.touch()
        if Event in ["completed", "terminal", "hangup"]:
            # If the session is still active in memory, close it
            stream_sid = session.stream_sid
            if stream_sid:
                call_manager.close_session(stream_sid, outcome=Event)
            else:
                # If stream never connected
                call_manager.close_session(f"no_stream_{CallSid}", outcome=Event)
                
    return JSONResponse(content={"status": "ok"})
