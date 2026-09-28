"""Regression coverage for Tailscale service URLs and proxy bypass."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from toolrecap_v4.endpoints import (
    is_tailscale_endpoint,
    normalize_http_endpoint,
    use_environment_proxy,
)
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.voice_studio import VoiceStudioAdapter


TAILSCALE_URL = "https://desktop-t5c9b90.tail7b66e0.ts.net:18443"


def test_tailscale_https_hostname_with_port_is_preserved() -> None:
    assert normalize_http_endpoint(f"  {TAILSCALE_URL}/  ", "Remote URL") == TAILSCALE_URL
    assert is_tailscale_endpoint(TAILSCALE_URL)
    assert use_environment_proxy(TAILSCALE_URL) is False
    assert use_environment_proxy("https://gateway.example.com:8443") is True


@pytest.mark.parametrize("bad", [
    "https://desktop.tailnet.ts.net:123456",
    "`https://desktop.tailnet.ts.net`:8443)",
    "https://user:password@desktop.tailnet.ts.net:8443",
])
def test_malformed_tailscale_urls_fail_with_actionable_validation(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_http_endpoint(bad, "Remote URL")


def test_gateway_and_voice_clients_bypass_environment_proxy_for_tailnet() -> None:
    with patch("toolrecap_v4.gateway.httpx.Client") as gateway_client:
        GatewayClient(base_url=TAILSCALE_URL)._get_client()
        assert gateway_client.call_args.kwargs["trust_env"] is False

    with patch("toolrecap_v4.voice_studio.httpx.Client") as voice_client:
        VoiceStudioAdapter(mode="remote", remote_url=TAILSCALE_URL)._get_client(TAILSCALE_URL)
        assert voice_client.call_args.kwargs["trust_env"] is False
