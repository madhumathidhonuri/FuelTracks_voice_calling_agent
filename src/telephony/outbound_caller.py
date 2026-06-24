import logging
import urllib.parse
import httpx
from typing import Dict, Any, Optional
from config.settings import settings
from src.storage.database import create_call

logger = logging.getLogger(__name__)

class OutboundCaller:
    def __init__(self):
        self.account_sid = settings.EXOTEL_ACCOUNT_SID
        self.api_key = settings.EXOTEL_API_KEY
        self.api_token = settings.EXOTEL_API_TOKEN
        self.caller_id = settings.EXOTEL_CALLER_ID
        self.api_url = settings.EXOTEL_API_URL

    def format_phone_number(self, phone: str) -> str:
        """
        Formats a phone number to E.164. Assumes +91 (India) if no country code.
        """
        # Remove whitespace and special characters
        clean = "".join(c for c in phone if c.isdigit() or c == "+")
        if not clean:
            return ""
        
        # Handle leading zero for 11-digit numbers (like 09876543210)
        if clean.startswith("0") and len(clean) == 11:
            clean = clean[1:]
        
        # If it doesn't start with '+', add country code
        if not clean.startswith("+"):
            if len(clean) == 10:
                clean = "+91" + clean
            elif len(clean) == 12 and clean.startswith("91"):
                clean = "+" + clean
                
        return clean

    async def initiate_call(
        self, 
        customer_number: str, 
        customer_name: str, 
        product_interest: Optional[str] = None, 
        call_type: str = "lead_followup"
    ) -> Dict[str, Any]:
        """
        Triggers an outbound call using Exotel's Connect API.
        Connects the dialed customer to our WebSocket StreamUrl.
        """
        formatted_number = self.format_phone_number(customer_number)
        if not formatted_number:
            raise ValueError(f"Invalid phone number format: {customer_number}")

        if not self.account_sid or not self.api_key or not self.api_token or not self.caller_id:
            raise ValueError("Exotel credentials and caller ID must be configured in settings.")

        # Construct stream URL with customer context
        params = {
            "customer_name": customer_name,
            "call_type": call_type,
            "from_number": formatted_number,
            "to_number": self.caller_id
        }
        if product_interest:
            params["product_interest"] = product_interest

        query_str = urllib.parse.urlencode(params)
        stream_url = settings.WEBSOCKET_URL
        if "?" in stream_url:
            stream_url += f"&{query_str}"
        else:
            stream_url += f"?{query_str}"

        # Derive status callback URL from WEBSOCKET_URL if public
        status_callback = None
        if stream_url.startswith("wss://"):
            status_callback = stream_url.replace("wss://", "https://").split("/ws/")[0] + "/voice/outbound-status"
        elif stream_url.startswith("ws://") and "localhost" not in stream_url:
            status_callback = stream_url.replace("ws://", "http://").split("/ws/")[0] + "/voice/outbound-status"

        # Exotel API URL endpoint
        url = f"{self.api_url.rstrip('/')}/v1/Accounts/{self.account_sid}/Calls/connect.json"

        # POST Form data
        data = {
            "From": formatted_number,
            "CallerId": self.caller_id,
            "StreamUrl": stream_url,
            "StreamType": "bidirectional"
        }
        
        if status_callback:
            data["StatusCallback"] = status_callback
            data["StatusCallbackEvents[]"] = "terminal"

        logger.info(f"Triggering Exotel Outbound Call to {formatted_number} ({customer_name})...")
        logger.debug(f"Request payload data: {data}")

        # Basic Auth credentials
        auth = (self.api_key, self.api_token)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, data=data, auth=auth, timeout=10.0)
                
                # Check for HTTP Errors
                if response.status_code != 200:
                    logger.error(f"Exotel API returned HTTP {response.status_code}: {response.text}")
                    return {
                        "success": False,
                        "status_code": response.status_code,
                        "error": response.text
                    }
                
                res_data = response.json()
                call_info = res_data.get("Call", {})
                call_sid = call_info.get("Sid")
                
                if not call_sid:
                    logger.error(f"Exotel API response did not contain Call.Sid: {res_data}")
                    return {
                        "success": False,
                        "error": "No Call SID in Exotel response",
                        "response": res_data
                    }
                
                logger.info(f"Outbound call successfully queued with Exotel. Call SID: {call_sid}")
                
                # Pre-log call in local SQLite database
                create_call(
                    call_sid=call_sid,
                    from_number=formatted_number,
                    to_number=self.caller_id,
                    call_type=call_type
                )
                
                return {
                    "success": True,
                    "call_sid": call_sid,
                    "status": call_info.get("Status", "queued"),
                    "response": res_data
                }
                
            except httpx.RequestError as e:
                logger.exception(f"Connection error requesting Exotel Connect API: {e}")
                return {
                    "success": False,
                    "error": f"Connection error: {str(e)}"
                }
