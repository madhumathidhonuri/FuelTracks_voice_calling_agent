import logging
import base64
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from src.telephony.call_manager import call_manager
from src.orchestrator.turn_manager import TurnManager
from src.orchestrator.pipeline import AudioPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/media")
async def websocket_endpoint(
    websocket: WebSocket,
    call_sid: Optional[str] = Query(None),
    CallSid: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for bidirectional audio streaming with Exotel.
    """
    await websocket.accept()
    
    qp = dict(websocket.query_params)
    logger.info(f"WebSocket connection accepted. Query params: {qp}")
    logger.info(f"WebSocket connection accepted. Headers: {dict(websocket.headers)}")
    
    resolved_call_sid = call_sid or CallSid or qp.get("call_sid") or qp.get("CallSid")
    session = None
    pipeline = None
    stream_sid = None
    turn_manager = TurnManager(websocket)
    
    # Exotel standard sample rate is 8000 Hz or 16000 Hz.
    sample_rate_str = websocket.query_params.get("sample-rate", "16000")
    try:
        sample_rate = int(sample_rate_str)
    except ValueError:
        sample_rate = 16000

    try:
        while True:
            if session and getattr(session, "should_hangup", False):
                logger.info(f"Call {session.call_sid} was marked for hangup. Terminating WebSocket connection.")
                break

            # Exotel sends JSON text envelopes
            message = await websocket.receive_json()
            event_type = message.get("event")
            
            # Check call duration limit (hard limit at 110 seconds)
            if session:
                from datetime import datetime
                elapsed = (datetime.now() - session.start_time).total_seconds()
                if elapsed > 110.0:
                    logger.warning(f"Call {session.call_sid} exceeded duration limit of 110 seconds. Terminating.")
                    break
            
            if event_type == "start":
                logger.info(f"Start event message content: {message}")
                stream_sid = message.get("stream_sid")
                start_data = message.get("start", {})
                
                # Dynamically resolve Sample Rate from start payload (under start -> media_format -> sample_rate)
                media_format = start_data.get("media_format") or start_data.get("mediaFormat") or {}
                media_sample_rate = media_format.get("sample_rate") or media_format.get("sampleRate")
                if media_sample_rate:
                    try:
                        sample_rate = int(media_sample_rate)
                        logger.info(f"Dynamically negotiated sample rate from Exotel media_format: {sample_rate} Hz")
                    except ValueError:
                        logger.warning(f"Invalid sample rate in media_format: {media_sample_rate}")
                        
                # Dynamically resolve Call SID from start payload (both case options)
                event_call_sid = start_data.get("callSid") or start_data.get("call_sid") or resolved_call_sid
                if not event_call_sid:
                    logger.error(f"Cannot resolve Call SID from start message: {message}. Closing WebSocket.")
                    break
                    
                resolved_call_sid = event_call_sid
                logger.info(f"Start event received. Stream SID: {stream_sid}, Call SID: {resolved_call_sid}")

                
                # 1. Retrieve or create the call session dynamically
                session = call_manager.get_session_by_call(resolved_call_sid)
                if not session:
                    logger.warning(f"No active session found for Call SID {resolved_call_sid}. Creating dynamic/outbound session on the fly.")
                    custom_params = start_data.get("custom_parameters") or start_data.get("customParameters") or {}
                    from_number = custom_params.get("from_number") or custom_params.get("from;_number") or qp.get("from_number") or qp.get("From") or "unknown"
                    to_number = custom_params.get("to_number") or custom_params.get("to;_number") or qp.get("to_number") or qp.get("To") or "unknown"
                    call_type = custom_params.get("call_type", qp.get("call_type", "inbound_routing"))
                    
                    # Strip spaces if any
                    from_number = str(from_number).strip()
                    to_number = str(to_number).strip()
                    
                    session = await call_manager.create_session(
                        call_sid=resolved_call_sid,
                        from_number=from_number,
                        to_number=to_number,
                        call_type=call_type
                    )
                    
                    # Populate context from custom params or handshake query params
                    customer_name = custom_params.get("customer_name", qp.get("customer_name"))
                    if customer_name:
                        session.conversation_manager.context["customer_name"] = customer_name
                        
                    product_interest = custom_params.get("product_interest", qp.get("product_interest"))
                    if product_interest:
                        session.conversation_manager.context["product_interest"] = product_interest
                
                # Link stream SID to call session
                call_manager.link_stream(resolved_call_sid, stream_sid)
                
                # Initialize pipeline
                pipeline = AudioPipeline(session, turn_manager, sample_rate=sample_rate)
                
                # Trigger the initial agent greeting to welcome the caller in a background task
                import asyncio
                asyncio.create_task(pipeline.trigger_initial_greeting())
                
            elif event_type == "media":
                if pipeline:
                    media = message.get("media", {})
                    payload = media.get("payload", "")
                    if payload:
                        # Decode base64 PCM frames
                        audio_bytes = base64.b64decode(payload)
                        # Feed audio into our interactive conversation pipeline
                        await pipeline.process_inbound_audio(audio_bytes)
                    
            elif event_type == "dtmf":
                dtmf = message.get("dtmf", {})
                digit = dtmf.get("digit", "")
                logger.info(f"DTMF key pressed on stream {stream_sid}: {digit}")
                
            elif event_type == "stop":
                logger.info(f"Stop event received for stream {stream_sid}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for stream {stream_sid or resolved_call_sid}")
    except Exception as e:
        logger.exception(f"Error in WebSocket handler loop: {e}")
    finally:
        # Stop any running playback
        if session:
            await turn_manager.stop_audio(session)
        # Close pipeline queue tasks
        if pipeline:
            pipeline.close()
            
        # Close the call session and commit metrics to SQLite
        if stream_sid and resolved_call_sid:
            await call_manager.close_session(stream_sid, outcome="completed")
        elif resolved_call_sid:
            await call_manager.close_session(f"no_stream_{resolved_call_sid}", outcome="abandoned")
        
        try:
            await websocket.close()
        except Exception:
            pass
