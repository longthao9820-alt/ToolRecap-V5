"""Tests for VoiceStudioAdapter."""

import io
from pathlib import Path
import struct
import wave

import httpx
import pytest

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    CancelledError,
    InvalidAudioError,
    VoiceStudioError,
    VoiceStudioUnavailableError,
)
from toolrecap_v4.voice_studio import VoiceStudioAdapter, validate_wav_bytes


def make_dummy_wav(duration_s: float = 1.0, framerate: int = 24000, channels: int = 1) -> bytes:
    """Generate valid PCM WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(framerate)
        nframes = int(duration_s * framerate)
        w.writeframes(b"\x00\x00" * channels * nframes)
    return buf.getvalue()


def test_wav_validation_acceptance():
    """Verify WAV validation accepts valid WAV and rejects corrupt/zero-duration audio."""
    valid_wav = make_dummy_wav(duration_s=1.5, framerate=24000, channels=1)
    duration = validate_wav_bytes(valid_wav)
    assert abs(duration - 1.5) < 0.01

    # Empty payload
    with pytest.raises(InvalidAudioError, match="too small"):
        validate_wav_bytes(b"")

    # Garbage bytes
    with pytest.raises(InvalidAudioError, match="Failed to parse WAV"):
        validate_wav_bytes(b"RIFF" + b"X" * 50)

    # Valid header but 0 frames
    zero_frames_wav = make_dummy_wav(duration_s=0.0, framerate=24000, channels=1)
    with pytest.raises(InvalidAudioError, match="zero frames"):
        validate_wav_bytes(zero_frames_wav)


def test_health_and_voices():
    """Verify health and voices querying via adapter."""
    mock_voices = {"voices": [{"voice_id": "alloy", "name": "Alloy"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json=mock_voices)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = VoiceStudioAdapter(client=client)

    health = adapter.check_health("http://127.0.0.1:3900")
    assert health.get("status") == "ok"

    voices = adapter.get_voices("http://127.0.0.1:3900")
    assert len(voices) == 1
    assert voices[0]["voice_id"] == "alloy"


def test_auto_routing_local_first():
    """Verify Auto mode selects Local when Local is healthy."""
    valid_wav = make_dummy_wav(duration_s=1.0)
    endpoints_called = []

    def handler(request: httpx.Request) -> httpx.Response:
        endpoints_called.append(str(request.url))
        if "3900/health" in str(request.url):
            return httpx.Response(200, json={"status": "ok"})
        if "3900/v1/audio/speech" in str(request.url):
            return httpx.Response(200, content=valid_wav)
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = VoiceStudioAdapter(mode="auto", remote_url="https://voice.example.ts.net:8443", client=client)

    wav = adapter.synthesize("Test input")
    assert wav == valid_wav
    # Verify local was used, remote was never called
    assert any("3900" in u for u in endpoints_called)
    assert not any("ts.net" in u for u in endpoints_called)


def test_auto_routing_fallback_to_remote():
    """Verify Auto mode selects Remote when Local is down."""
    valid_wav = make_dummy_wav(duration_s=1.0)
    endpoints_called = []

    def handler(request: httpx.Request) -> httpx.Response:
        endpoints_called.append(str(request.url))
        if "3900" in str(request.url):
            return httpx.Response(503, text="Service Unavailable")
        if "ts.net:8443/health" in str(request.url):
            return httpx.Response(200, json={"status": "ok"})
        if "ts.net:8443/v1/audio/speech" in str(request.url):
            return httpx.Response(200, content=valid_wav)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = VoiceStudioAdapter(mode="auto", remote_url="https://voice.example.ts.net:8443", client=client)

    wav = adapter.synthesize("Test input")
    assert wav == valid_wav
    assert any("3900" in u for u in endpoints_called)
    assert any("ts.net" in u for u in endpoints_called)


def test_auto_routing_both_unavailable():
    """Verify Auto mode raises VoiceStudioUnavailableError when both fail."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = VoiceStudioAdapter(mode="auto", client=client)

    with pytest.raises(VoiceStudioUnavailableError, match="neither Local .* nor Remote .* is reachable"):
        adapter.synthesize("Test input")


def test_auth_separation_and_credential_sanitization():
    """Verify auth is applied only to Remote and redacted in exceptions."""
    secret_key = "super_secret_bearer_token_xyz"
    captured_auth_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_auth_headers.append(request.headers.get("Authorization"))
        if "/health" in str(request.url):
            return httpx.Response(200, json={"status": "ok"})
        # Fail with error body mentioning the secret
        return httpx.Response(400, text=f"Error occurred with token {secret_key}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = VoiceStudioAdapter(
        mode="remote", remote_url="https://voice.example.test", remote_api_key=secret_key, client=client,
    )

    with pytest.raises(VoiceStudioError) as exc_info:
        adapter.synthesize("Test input")

    # Authorization header was sent to Remote
    assert f"Bearer {secret_key}" in captured_auth_headers
    # Credential must be redacted in exception
    assert secret_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_remote_key_is_never_sent_to_local_or_lookalike_origin():
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("Authorization")))
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    empty_remote = VoiceStudioAdapter(
        mode="local", remote_url="", remote_api_key="remote-secret", client=client,
    )
    empty_remote.check_health("http://127.0.0.1:3900")
    configured_remote = VoiceStudioAdapter(
        mode="remote", remote_url="https://voice.example.test", remote_api_key="remote-secret", client=client,
    )
    configured_remote.check_health("https://voice.example.test.attacker.invalid")

    assert seen == [
        ("http://127.0.0.1:3900/health", None),
        ("https://voice.example.test.attacker.invalid/health", None),
    ]


def test_transient_vs_logical_retries():
    """Verify transient errors (502) are retried; logical errors (400) fail immediately."""
    valid_wav = make_dummy_wav(duration_s=1.0)
    attempts = 0

    def transient_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if "/health" in str(request.url):
            return httpx.Response(200, json={"status": "ok"})
        attempts += 1
        if attempts < 3:
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(200, content=valid_wav)

    client = httpx.Client(transport=httpx.MockTransport(transient_handler))
    adapter = VoiceStudioAdapter(mode="local", max_retries=3, backoff_factor=0.01, client=client)
    res = adapter.synthesize("Test")
    assert res == valid_wav
    assert attempts == 3

    # Logical error (400) should fail immediately with 1 attempt
    logical_attempts = 0

    def logical_handler(request: httpx.Request) -> httpx.Response:
        nonlocal logical_attempts
        if "/health" in str(request.url):
            return httpx.Response(200, json={"status": "ok"})
        logical_attempts += 1
        return httpx.Response(400, text="Bad Request: invalid voice")

    client2 = httpx.Client(transport=httpx.MockTransport(logical_handler))
    adapter2 = VoiceStudioAdapter(mode="local", max_retries=3, backoff_factor=0.01, client=client2)
    with pytest.raises(VoiceStudioError, match="rejected .*HTTP 400"):
        adapter2.synthesize("Test")
    assert logical_attempts == 1


def test_cancellation():
    """Verify cancellation token stops synthesis and raises CancelledError."""
    token = CancellationToken()
    token.cancel()

    adapter = VoiceStudioAdapter(mode="local")
    with pytest.raises(CancelledError):
        adapter.synthesize("Test input", cancellation_token=token)


def test_actual_v2_preset_payload_all_12_and_piper_reject():
    """Verify all 12 V2 presets produce exact archetype instruct payloads and reject Piper."""
    import json
    from toolrecap_v4.voice_studio import V2_VOICE_PRESETS

    # Exact archetype instructions retained from the audited V2 voice catalog.
    # and omnivoice_adapter.py (OFFICIAL_VOICE_INSTRUCTS)
    v2_catalog_exact_instructs = {
        "voicestudio.en.neighbor": "female, young adult, moderate pitch, american accent",
        "voicestudio.en.companion": "female, middle-aged, moderate pitch, canadian accent",
        "voicestudio.en.teacher": "female, middle-aged, moderate pitch, american accent",
        "voicestudio.en.anchor": "male, middle-aged, moderate pitch, american accent",
        "voicestudio.en.documentarian": "male, middle-aged, low pitch, american accent",
        "voicestudio.en.promo": "male, middle-aged, low pitch",
        "voicestudio.en.librarian": "female, middle-aged, low pitch, british accent",
        "voicestudio.en.podcaster": "female, young adult, high pitch, australian accent",
        "voicestudio.en.luxe": "female, middle-aged, moderate pitch, british accent",
        "voicestudio.en.storyteller": "male, elderly, low pitch, british accent",
        "voicestudio.en.commentator": "male, middle-aged, high pitch, british accent",
        "voicestudio.en.explainer": "male, young adult, moderate pitch, british accent",
    }

    assert len(v2_catalog_exact_instructs) == 12
    assert set(v2_catalog_exact_instructs.keys()) == set(V2_VOICE_PRESETS.keys())

    valid_wav = make_dummy_wav(duration_s=1.0)
    captured_payloads = []

    def mock_speech_handler(request: httpx.Request) -> httpx.Response:
        if "/health" in str(request.url):
            return httpx.Response(200, json={"status": "ok"})
        if "/speech" in str(request.url):
            payload = json.loads(request.read())
            captured_payloads.append(payload)
            return httpx.Response(200, content=valid_wav)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(mock_speech_handler))
    adapter = VoiceStudioAdapter(mode="local", client=client)

    # 1. Test all 12 presets: verify exact payload, voice="default", and style doesn't overwrite instruct
    for voice_id, expected_instruct in v2_catalog_exact_instructs.items():
        captured_payloads.clear()

        # Invariant: V2 preset instruct must match V2 catalog exactly
        preset_info = V2_VOICE_PRESETS[voice_id]
        assert preset_info["instruct"] == expected_instruct

        # Synthesize with explicit style to verify style does NOT overwrite archetype instruct
        result_wav = adapter.synthesize(
            "Test narration text",
            voice=voice_id,
            style="film_recap",
        )
        assert result_wav == valid_wav
        assert len(captured_payloads) == 1
        payload = captured_payloads[0]

        assert payload["voice"] == "default", f"Preset {voice_id} must map voice to 'default'"
        assert payload["model"] == "omnivoice"
        assert payload["instruct"] == expected_instruct, (
            f"Preset {voice_id} instruct mismatch. Expected '{expected_instruct}', got '{payload['instruct']}'"
        )
        # Style is placed into description; archetype instruct must NOT be overwritten by style
        assert payload["description"] == "film_recap"
        assert payload["instruct"] != "film_recap"

    # 2. Test without explicit style: description defaults to preset's description
    for voice_id, expected_instruct in v2_catalog_exact_instructs.items():
        captured_payloads.clear()
        adapter.synthesize("Another text", voice=voice_id)
        assert len(captured_payloads) == 1
        payload = captured_payloads[0]
        assert payload["voice"] == "default"
        assert payload["instruct"] == expected_instruct
        assert payload["description"] == V2_VOICE_PRESETS[voice_id]["description"]

    # 3. Piper rejection: all Piper voices must raise VoiceStudioUnavailableError
    piper_voices = [
        "piper.en_US-lessac-medium",
        "piper.en_US-ryan-medium",
        "piper.en_GB-alba-medium",
        "piper.en_GB-alan-medium",
        "piper.other_unknown_voice",
    ]
    for piper_id in piper_voices:
        with pytest.raises(VoiceStudioUnavailableError, match="unavailable in VoiceStudio or V2 presets"):
            adapter.synthesize("Piper attempt", voice=piper_id)
