"""ICE (STUN/TURN) server configuration for WebRTC.

STUN alone works locally, but cloud hosts sit behind NATs that block direct
WebRTC connections, so a TURN relay is needed. Cloudflare issues short-lived
TURN credentials from a long-lived key; without one we fall back to STUN.
"""
import json
import logging
import urllib.request

from .config import CLOUDFLARE_TURN_URL, STUN_SERVERS, TURN_CREDENTIAL_TTL_S

logger = logging.getLogger(__name__)


def fetch_cloudflare_ice_servers(key_id: str, api_token: str, timeout: float = 10) -> list[dict]:
    req = urllib.request.Request(
        CLOUDFLARE_TURN_URL.format(key_id=key_id),
        data=json.dumps({"ttl": TURN_CREDENTIAL_TTL_S}).encode(),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            # Cloudflare rejects urllib's default User-Agent (error 1010).
            "User-Agent": "hand-tracking-app/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["iceServers"]


def get_ice_servers(key_id: str | None, api_token: str | None) -> list[dict]:
    """Returns TURN+STUN servers when credentials are given, else STUN only."""
    if not key_id or not api_token:
        return STUN_SERVERS
    try:
        return fetch_cloudflare_ice_servers(key_id, api_token)
    except Exception as e:
        logger.warning("TURN credential fetch failed, falling back to STUN: %s", e)
        return STUN_SERVERS
