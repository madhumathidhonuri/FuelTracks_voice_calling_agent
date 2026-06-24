import unittest
import sys
import os
import sqlite3
import openpyxl
from unittest.mock import patch, MagicMock, AsyncMock

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.storage.database import init_db, get_connection, get_call_logs
from src.telephony.outbound_caller import OutboundCaller
from scripts.bulk_outbound_caller import parse_excel_file

class TestOutboundCalling(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Force a temporary test DB path
        settings.DATABASE_URL = "sqlite:///test_voice_calling_outbound.db"
        init_db()

    @classmethod
    def tearDownClass(cls):
        # Clean up test DB
        db_path = settings.BASE_DIR / "test_voice_calling_outbound.db"
        if db_path.exists():
            try:
                os.remove(db_path)
            except Exception:
                pass
                
        # Clean up temporary Excel test files
        for f in ["temp_test_ok.xlsx", "temp_test_bad_headers.xlsx"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_phone_number_formatting(self):
        caller = OutboundCaller()
        
        # Test basic cases
        self.assertEqual(caller.format_phone_number("9876543210"), "+919876543210")
        self.assertEqual(caller.format_phone_number("+91 98765 43210"), "+919876543210")
        self.assertEqual(caller.format_phone_number("09876543210"), "+919876543210")
        self.assertEqual(caller.format_phone_number("+1234567890"), "+1234567890")
        self.assertEqual(caller.format_phone_number(""), "")

    @patch("httpx.AsyncClient.post")
    async def _test_initiate_call_mocked(self, mock_post):
        # Setup settings with test credentials
        settings.EXOTEL_ACCOUNT_SID = "test_sid"
        settings.EXOTEL_API_KEY = "test_key"
        settings.EXOTEL_API_TOKEN = "test_token"
        settings.EXOTEL_CALLER_ID = "+919999999999"
        settings.WEBSOCKET_URL = "ws://localhost:8000/ws/media"
        
        caller = OutboundCaller()
        
        # Mock Response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Call": {
                "Sid": "outbound-call-sid-123",
                "Status": "ringing"
            }
        }
        mock_post.return_value = mock_response

        # Execute
        result = await caller.initiate_call(
            customer_number="9876543210",
            customer_name="Test Person",
            product_interest="GPS Tracker",
            call_type="lead_followup"
        )
        
        # Assertions
        self.assertTrue(result["success"])
        self.assertEqual(result["call_sid"], "outbound-call-sid-123")
        self.assertEqual(result["status"], "ringing")
        
        # Verify call is in DB
        logs = get_call_logs("outbound-call-sid-123")
        self.assertIsNotNone(logs.get("call"))
        self.assertEqual(logs["call"]["from_number"], "+919876543210")
        self.assertEqual(logs["call"]["to_number"], "+919999999999")
        self.assertEqual(logs["call"]["call_type"], "lead_followup")
        
        # Check mock post arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        post_data = kwargs["data"]
        self.assertEqual(post_data["From"], "+919876543210")
        self.assertEqual(post_data["CallerId"], "+919999999999")
        self.assertEqual(post_data["StreamType"], "bidirectional")
        self.assertIn("customer_name=Test+Person", post_data["StreamUrl"])
        self.assertIn("product_interest=GPS+Tracker", post_data["StreamUrl"])

    def test_initiate_call(self):
        # Run async test using helper
        import asyncio
        asyncio.run(self._test_initiate_call_mocked())

    def test_excel_parsing(self):
        # Create a temp Excel file with correct headers
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["Name", "Phone", "Product", "Purpose"])
        ws.append(["Alice", "9876543210", "Fuel Sensor", "support"])
        ws.append(["Bob", "+918765432109", "GPS Tracker", "invalid_type"])
        wb.save("temp_test_ok.xlsx")
        
        # Parse it
        customers = parse_excel_file("temp_test_ok.xlsx")
        
        self.assertEqual(len(customers), 2)
        self.assertEqual(customers[0]["name"], "Alice")
        self.assertEqual(customers[0]["phone"], "9876543210")
        self.assertEqual(customers[0]["product_interest"], "Fuel Sensor")
        self.assertEqual(customers[0]["call_type"], "support")
        
        # Test fallback / default of invalid type
        self.assertEqual(customers[1]["name"], "Bob")
        self.assertEqual(customers[1]["call_type"], "lead_followup") # fell back to lead_followup

    def test_excel_parsing_bad_headers(self):
        # Create a temp Excel file with missing Phone header
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["Name", "Email Address", "Address"])
        ws.append(["Alice", "alice@example.com", "Hyderabad"])
        wb.save("temp_test_bad_headers.xlsx")
        
        with self.assertRaises(ValueError) as context:
            parse_excel_file("temp_test_bad_headers.xlsx")
            
        self.assertIn("Could not identify the phone number column", str(context.exception))

if __name__ == "__main__":
    unittest.main()
