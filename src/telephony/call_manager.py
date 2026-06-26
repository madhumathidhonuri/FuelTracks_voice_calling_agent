import time
from datetime import datetime
import logging
from typing import Dict, Any, Optional
from src.conversation.conversation_manager import ConversationManager
from src.storage.database import acreate_call, aupdate_call_end

logger = logging.getLogger(__name__)

class CallSession:
    def __init__(self, call_sid: str, from_number: str, to_number: str, call_type: str):
        self.call_sid = call_sid
        self.stream_sid: Optional[str] = None
        self.from_number = from_number
        self.to_number = to_number
        self.call_type = call_type
        
        # Timing
        self.start_time = datetime.now()
        self.last_activity = time.time()
        
        # State
        self.is_playing = False
        self.barge_in_triggered = False
        
        # Cost Metrics
        self.stt_seconds_logged: float = 0.0
        self.tts_characters_logged: int = 0
        self.llm_tokens_logged: int = 0
        
        # Latency Metrics (stored as list of float milliseconds)
        self.stt_latencies = []
        self.llm_latencies = []
        self.tts_latencies = []
        
        # Conversation state
        self.conversation_manager = ConversationManager(call_sid)
        
    def link_stream(self, stream_sid: str):
        self.stream_sid = stream_sid
        # Set up prompts/context, preserving existing context
        self.conversation_manager.initialize_call(self.call_type, self.conversation_manager.context)
        
    def touch(self):
        self.last_activity = time.time()

    async def add_customer_turn(self, text: str, detected_language: str = "en-IN", confidence: float = 1.0):
        """
        Delegates registering a customer turn to the conversation manager.
        """
        await self.conversation_manager.add_customer_turn(text, detected_language, confidence)


class CallManager:
    def __init__(self):
        self.sessions_by_stream: Dict[str, CallSession] = {}
        self.sessions_by_call: Dict[str, CallSession] = {}
        
    async def create_session(
        self, 
        call_sid: str, 
        from_number: str, 
        to_number: str, 
        call_type: str
    ) -> CallSession:
        """
        Creates a call session and writes the initial record to the database.
        """
        session = CallSession(call_sid, from_number, to_number, call_type)
        self.sessions_by_call[call_sid] = session
        
        # Log session start to DB asynchronously
        await acreate_call(
            call_sid=call_sid,
            from_number=from_number,
            to_number=to_number,
            call_type=call_type,
            start_time=session.start_time.isoformat()
        )
        logger.info(f"Session created for Call SID {call_sid} (Type: {call_type})")
        return session
        
    def link_stream(self, call_sid: str, stream_sid: str) -> Optional[CallSession]:
        session = self.sessions_by_call.get(call_sid)
        if session:
            session.link_stream(stream_sid)
            self.sessions_by_stream[stream_sid] = session
            logger.info(f"Linked Stream SID {stream_sid} to Call SID {call_sid}")
            return session
        logger.error(f"Cannot link stream: Call SID {call_sid} not found")
        return None
        
    def get_session_by_stream(self, stream_sid: str) -> Optional[CallSession]:
        return self.sessions_by_stream.get(stream_sid)
        
    def get_session_by_call(self, call_sid: str) -> Optional[CallSession]:
        return self.sessions_by_call.get(call_sid)
        
    async def close_session(self, stream_sid: str, outcome: str = "completed"):
        session = self.sessions_by_stream.pop(stream_sid, None)
        if session:
            # Remove from call map
            self.sessions_by_call.pop(session.call_sid, None)
            
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - session.start_time).total_seconds()
            
            # Compute average latencies (in milliseconds)
            avg_stt = sum(session.stt_latencies) / len(session.stt_latencies) if session.stt_latencies else 0.0
            avg_llm = sum(session.llm_latencies) / len(session.llm_latencies) if session.llm_latencies else 0.0
            avg_tts = sum(session.tts_latencies) / len(session.tts_latencies) if session.tts_latencies else 0.0
            
            # Update DB with final metrics and latencies asynchronously
            await aupdate_call_end(
                call_sid=session.call_sid,
                end_time=end_time.isoformat(),
                duration=duration,
                outcome=outcome,
                cost_tokens=session.llm_tokens_logged,
                cost_stt_sec=session.stt_seconds_logged,
                cost_tts_char=session.tts_characters_logged,
                stt_latency_ms=avg_stt,
                llm_latency_ms=avg_llm,
                tts_latency_ms=avg_tts
            )
            logger.info(
                f"Session closed: Call {session.call_sid}, Stream {stream_sid}, "
                f"Duration: {duration:.2f}s, STT Sec: {session.stt_seconds_logged:.2f}, "
                f"TTS Char: {session.tts_characters_logged}, LLM Tokens: {session.llm_tokens_logged}, "
                f"Avg STT Latency: {avg_stt:.1f}ms, Avg LLM Latency: {avg_llm:.1f}ms, Avg TTS Latency: {avg_tts:.1f}ms"
            )
            
call_manager = CallManager()
