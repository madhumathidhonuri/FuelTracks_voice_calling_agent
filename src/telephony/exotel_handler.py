import logging
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from src.telephony.call_manager import call_manager
from src.orchestrator.turn_manager import TurnManager
from src.orchestrator.pipeline import AudioPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/media")
async def websocket_endpoint(
    websocket: WebSocket,
    call_sid: str = Query(...)
):
    """
    WebSocket endpoint for bidirectional audio streaming with Exotel.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for Call SID {call_sid}")
    
    # 1. Retrieve or create the call session
    session = call_manager.get_session_by_call(call_sid)
    if not session:
        logger.warning(f"No active session for Call SID {call_sid} on WS connection. Creating default/outbound dynamic session.")
        qp = dict(websocket.query_params)
        from_number = qp.get("from_number", qp.get("From", "unknown"))
        to_number = qp.get("to_number", qp.get("To", "unknown"))
        call_type = qp.get("call_type", "inbound_routing")
        
        session = call_manager.create_session(
            call_sid=call_sid,
            from_number=from_number,
            to_number=to_number,
            call_type=call_type
        )
        
        # Populate context if passed via query params
        customer_name = qp.get("customer_name")
        if customer_name:
            session.conversation_manager.context["customer_name"] = customer_name
            
        product_interest = qp.get("product_interest")
        if product_interest:
            session.conversation_manager.context["product_interest"] = product_interest
        
    turn_manager = TurnManager(websocket)
    
    # Exotel standard sample rate is 8000 Hz or 16000 Hz.
    # We query the query params for sample rate or default to 16000.
    sample_rate_str = websocket.query_params.get("sample-rate", "16000")
    try:
        sample_rate = int(sample_rate_str)
    except ValueError:
        sample_rate = 16000
        
    pipeline = AudioPipeline(session, turn_manager, sample_rate=sample_rate)
    stream_sid = None
    
    try:
        while True:
            # Exotel sends JSON text envelopes
            message = await websocket.receive_json()
            event_type = message.get("event")
            
            if event_type == "start":
                stream_sid = message.get("stream_sid")
                logger.info(f"Start event received. Stream SID: {stream_sid}")
                
                # Link stream SID to call session
                call_manager.link_stream(call_sid, stream_sid)
                
                # Trigger the initial agent greeting to welcome the caller
                # Run this in a background task so it doesn't block receiving media frames
                import asyncio
                asyncio.create_task(pipeline.trigger_initial_greeting())
                
            elif event_type == "media":
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
        logger.info(f"WebSocket disconnected for stream {stream_sid or call_sid}")
    except Exception as e:
        logger.exception(f"Error in WebSocket handler loop: {e}")
    finally:
        # Stop any running playback
        await turn_manager.stop_audio(session)
        # Close pipeline queue tasks
        if 'pipeline' in locals():
            pipeline.close()
            
        # Close the call session and commit metrics to SQLite
        if stream_sid:
            call_manager.close_session(stream_sid, outcome="completed")
        else:
            call_manager.close_session(f"no_stream_{call_sid}", outcome="abandoned")
        
        try:
            await websocket.close()
        except Exception:
            pass
