import httpx
import logging
import socket
import asyncio

logger = logging.getLogger(__name__)

# Cached IP to avoid re-resolving on every single request
_resolved_ip_cache = {}

async def resolve_hostname_ipv4(hostname: str) -> str:
    """
    Resolve a hostname to its IPv4 address using DNS-over-HTTPS (DoH) to bypass
    buggy local DNS / IPv6 resolution issues on Windows.
    Falls back to socket resolution and then to hardcoded defaults if all fail.
    """
    if hostname in _resolved_ip_cache:
        return _resolved_ip_cache[hostname]

    # Standard known fallback IPs
    fallbacks = {
        "api.sarvam.ai": "20.235.220.20",
        "api.exotel.com": "3.0.70.209"
    }

    # Try DoH Cloudflare via raw IP
    try:
        url = f"https://1.1.1.1/dns-query?name={hostname}&type=A"
        headers = {"accept": "application/dns-json"}
        async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                data = r.json()
                answers = data.get("Answer", [])
                for ans in answers:
                    if ans.get("type") == 1: # A record
                        ip = ans.get("data")
                        if ip:
                            logger.info(f"Resolved {hostname} to {ip} via Cloudflare DoH")
                            _resolved_ip_cache[hostname] = ip
                            return ip
    except Exception as e:
        logger.warning(f"Failed to resolve {hostname} via Cloudflare DoH: {e}")

    # Try DoH Google via raw IP
    try:
        url = f"https://8.8.8.8/resolve?name={hostname}&type=A"
        headers = {"accept": "application/json"}
        async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                data = r.json()
                answers = data.get("Answer", [])
                for ans in answers:
                    if ans.get("type") == 1: # A record
                        ip = ans.get("data")
                        if ip:
                            logger.info(f"Resolved {hostname} to {ip} via Google DoH")
                            _resolved_ip_cache[hostname] = ip
                            return ip
    except Exception as e:
        logger.warning(f"Failed to resolve {hostname} via Google DoH: {e}")

    # Try standard socket getaddrinfo with AF_INET (IPv4 only)
    try:
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        addr_info = await loop.run_in_executor(
            None,
            lambda: socket.getaddrinfo(hostname, 80, family=socket.AF_INET)
        )
        if addr_info:
            ip = addr_info[0][4][0]
            logger.info(f"Resolved {hostname} to {ip} via socket.getaddrinfo (AF_INET)")
            _resolved_ip_cache[hostname] = ip
            return ip
    except Exception as e:
        logger.warning(f"Failed to resolve {hostname} via socket.getaddrinfo (AF_INET): {e}")

    # Hardcoded fallback
    default_ip = fallbacks.get(hostname)
    if default_ip:
        logger.warning(f"Using default fallback IP for {hostname}: {default_ip}")
        _resolved_ip_cache[hostname] = default_ip
        return default_ip

    return hostname
