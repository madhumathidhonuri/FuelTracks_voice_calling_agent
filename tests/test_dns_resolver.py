import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from src.audio.dns_resolver import resolve_hostname_ipv4, _resolved_ip_cache

class TestDNSResolver(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Clear the global cache before each test
        _resolved_ip_cache.clear()

    @patch("httpx.AsyncClient.get")
    async def test_resolve_doh_cloudflare_success(self, mock_get):
        # Mock response from Cloudflare DoH (synchronous methods on response)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Answer": [{"type": 1, "data": "1.2.3.4"}]
        }
        mock_get.return_value = mock_response

        ip = await resolve_hostname_ipv4("api.sarvam.ai")
        self.assertEqual(ip, "1.2.3.4")
        self.assertIn("api.sarvam.ai", _resolved_ip_cache)

    @patch("httpx.AsyncClient.get")
    async def test_resolve_doh_google_fallback(self, mock_get):
        # Cloudflare fails, Google succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "Answer": [{"type": 1, "data": "5.6.7.8"}]
        }

        mock_get.side_effect = [mock_response_fail, mock_response_success]

        ip = await resolve_hostname_ipv4("api.sarvam.ai")
        self.assertEqual(ip, "5.6.7.8")

    @patch("httpx.AsyncClient.get")
    @patch("socket.getaddrinfo")
    async def test_resolve_socket_fallback(self, mock_getaddrinfo, mock_get):
        # DoH both fail
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_get.return_value = mock_response_fail

        # socket.getaddrinfo succeeds
        mock_getaddrinfo.return_value = [
            (None, None, None, None, ("9.10.11.12", 80))
        ]

        ip = await resolve_hostname_ipv4("api.sarvam.ai")
        self.assertEqual(ip, "9.10.11.12")

    @patch("httpx.AsyncClient.get")
    @patch("socket.getaddrinfo")
    async def test_resolve_hardcoded_fallback(self, mock_getaddrinfo, mock_get):
        # DoH both fail
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_get.return_value = mock_response_fail

        # socket fails
        mock_getaddrinfo.side_effect = Exception("socket error")

        ip = await resolve_hostname_ipv4("api.sarvam.ai")
        self.assertEqual(ip, "20.235.220.20")
        
        # Clear cache to force resolution again for exotel
        _resolved_ip_cache.clear()
        ip_exotel = await resolve_hostname_ipv4("api.exotel.com")
        self.assertEqual(ip_exotel, "3.0.70.209")

if __name__ == "__main__":
    unittest.main()
