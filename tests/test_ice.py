from hand_tracking import ice
from hand_tracking.config import STUN_SERVERS


def test_no_credentials_uses_stun():
    assert ice.get_ice_servers(None, None) == STUN_SERVERS
    assert ice.get_ice_servers("", "") == STUN_SERVERS


def test_fetch_failure_falls_back_to_stun(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(ice, "fetch_cloudflare_ice_servers", boom)
    assert ice.get_ice_servers("id", "token") == STUN_SERVERS


def test_credentials_return_turn_servers(monkeypatch):
    turn = [{"urls": ["turn:turn.cloudflare.com:3478"], "username": "u", "credential": "c"}]
    monkeypatch.setattr(ice, "fetch_cloudflare_ice_servers", lambda *a, **k: turn)
    assert ice.get_ice_servers("id", "token") == turn
