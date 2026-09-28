"""Shared validation and transport policy for local and Tailscale services."""
from __future__ import annotations

from urllib.parse import urlsplit


def normalize_http_endpoint(
    value: str,
    field_name: str = "Endpoint",
    *,
    allow_empty: bool = False,
) -> str:
    """Validate an HTTP(S) service base URL, including ``.ts.net:PORT`` URLs."""
    endpoint = str(value or "").strip()
    if not endpoint:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} cannot be empty.")
    if any(char in endpoint for char in ("`", "\r", "\n", "\t")) or " " in endpoint:
        raise ValueError(
            f"{field_name} contains whitespace or quote markers. "
            "Use a plain URL such as https://desktop-name.tailnet.ts.net:8443"
        )
    parsed = urlsplit(endpoint)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            f"{field_name} must be a valid http:// or https:// URL "
            "(phải bắt đầu bằng 'http://' hoặc 'https://')."
        )
    if parsed.username or parsed.password:
        raise ValueError(
            f"{field_name} must not contain embedded credentials "
            "(không được chứa thông tin xác thực trong URL)."
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} port must be an integer from 1 to 65535.") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{field_name} port must be from 1 to 65535.")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain a query string or fragment.")
    return endpoint.rstrip("/")


def is_tailscale_endpoint(value: str) -> bool:
    """Return true for a valid Tailscale HTTPS/MagicDNS ``.ts.net`` hostname."""
    try:
        hostname = urlsplit(str(value or "")).hostname
    except ValueError:
        return False
    return bool(hostname and hostname.casefold().endswith(".ts.net"))


def use_environment_proxy(value: str) -> bool:
    """Tailnet traffic must not be diverted through a machine's HTTP proxy."""
    return not is_tailscale_endpoint(value)


def tailscale_connection_hint(value: str) -> str:
    """Give an actionable hint without weakening TLS verification."""
    if not is_tailscale_endpoint(value):
        return ""
    return (
        " Tailscale: use https://host.tailnet.ts.net for Tailscale Serve; "
        "for a direct non-TLS backend port use http://host.tailnet.ts.net:PORT. "
        "PORT must be 1-65535, and both devices must be connected to the same tailnet."
    )
