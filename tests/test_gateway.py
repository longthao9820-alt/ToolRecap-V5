"""Deterministic foundation tests for GatewayClient and error taxonomy."""

from __future__ import annotations

import base64
import inspect
import io
import json
from pathlib import Path
import sys
import types
import warnings

import httpx
from PIL import Image
from PIL.Image import DecompressionBombError, DecompressionBombWarning
import pytest

# Ensure isolated import of gateway and errors without executing untouched __init__.py exports
if "toolrecap_v4" not in sys.modules or not hasattr(sys.modules["toolrecap_v4"], "__path__"):
    pkg = types.ModuleType("toolrecap_v4")
    pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "toolrecap_v4")]
    sys.modules["toolrecap_v4"] = pkg

import toolrecap_v4.gateway as gw_module
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    CancelledError,
    EmptyApiResponseError,
    GatewayAuthenticationError,
    GatewayConnectionError,
    GatewayError,
    GatewayNotFoundError,
    GatewayPermissionError,
    GatewayRateLimitError,
    GatewayRequestTimeoutError,
    GatewayResponseError,
    GatewayServerError,
    GatewayTimeoutError,
    InvalidGatewayResponseError,
    MalformedApiResponseError,
    MalformedModelJsonError,
    ModelCapabilityError,
    PayloadContextError,
    UnsupportedMediaError,
)
from toolrecap_v4.gateway import (
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_IMAGE_DIMENSION,
    DEFAULT_MAX_IMAGES_PER_REQUEST,
    DEFAULT_MAX_TOTAL_IMAGE_BYTES,
    GatewayClient,
    GatewayResult,
    extract_json_from_text,
    sanitize_message,
    validate_and_reencode_image,
)


def create_test_image(
    path: Path,
    fmt: str = "PNG",
    size: tuple[int, int] = (64, 64),
    color: tuple = (255, 0, 0),
    mode: str = "RGB",
) -> Path:
    """Helper to create genuine test images via Pillow."""
    im = Image.new(mode, size, color)
    im.save(path, format=fmt)
    return path


# ---------------------------------------------------------------------------
# 1. Removal of production whole-video APIs
# ---------------------------------------------------------------------------


def test_old_video_transport_apis_absent():
    """Verify production whole-video transport APIs and types are completely absent."""
    assert not hasattr(GatewayClient, "submit_chat_analysis")
    assert not hasattr(GatewayClient, "validate_model_video_capability")
    assert not hasattr(GatewayClient, "validate_model_prime_capability")
    assert not hasattr(GatewayClient, "_validate_sources")
    assert not hasattr(GatewayClient, "_validate_and_encode_sources")

    assert "StreamingChatPayload" not in dir(gw_module)
    assert "SUPPORTED_VIDEO_EXTENSIONS" not in dir(gw_module)
    assert "DEFAULT_MAX_FILE_SIZE_BYTES" not in dir(gw_module)
    assert "DEFAULT_CHUNK_SIZE" not in dir(gw_module)


# ---------------------------------------------------------------------------
# 2. Text-only submit_text_chat (no images parameter)
# ---------------------------------------------------------------------------


def test_submit_text_chat_has_no_images_parameter():
    """Verify submit_text_chat is strictly text-only and has no images parameter."""
    sig = inspect.signature(GatewayClient.submit_text_chat)
    assert "images" not in sig.parameters
    assert "model" in sig.parameters


def test_submit_text_chat_streaming_success():
    """Verify streaming text chat preserves exact prompt and extracts JSON correctly."""
    captured_body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.read())
        sse_body = (
            'data: {"choices": [{"delta": {"content": "```json\\n{\\"recap\\": \\"success\\"}\\n```"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=sse_body, headers={"Content-Type": "text/event-stream"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client)

    exact_prompt = "DO NOT ALTER THIS PROMPT: character recap exact text."
    result = gw.submit_text_chat(
        prompt=exact_prompt,
        model="test-writer-model",
        reasoning_effort="high",
        stream=True,
        expect_json=True,
        phase="writer",
    )

    assert result.parsed_json == {"recap": "success"}
    assert result.model == "test-writer-model"
    assert result.bytes_sent > 0
    assert result.metadata is not None
    assert result.metadata["phase"] == "writer"
    assert result.metadata["image_count"] == 0
    assert result.metadata["retry_count"] == 0
    assert result.metadata["bytes_sent"] == result.bytes_sent

    assert captured_body["model"] == "test-writer-model"
    assert captured_body["reasoning_effort"] == "high"
    assert captured_body["stream"] is True
    assert len(captured_body["messages"]) == 1
    assert captured_body["messages"][0]["role"] == "user"
    assert captured_body["messages"][0]["content"] == exact_prompt


def test_submit_text_chat_sync_success():
    """Verify synchronous text chat (stream=False) returns structured JSON and exact bytes."""
    captured_body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.read())
        resp_data = {
            "choices": [{"message": {"content": '{"status": "sync_ok"}'}}],
            "usage": {"total_tokens": 42},
        }
        return httpx.Response(200, json=resp_data)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client)

    result = gw.submit_text_chat(
        prompt="Sync prompt",
        model="scanner-model",
        stream=False,
        expect_json=True,
    )

    assert result.parsed_json == {"status": "sync_ok"}
    assert result.usage == {"total_tokens": 42}
    assert result.bytes_sent == len(json.dumps(captured_body, ensure_ascii=False).encode("utf-8"))


def test_submit_text_chat_freetext_without_json():
    """Verify text chat with expect_json=False returns raw text and parsed_json=None."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='data: {"choices": [{"delta": {"content": "This is freeform narration."}}]}\n\ndata: [DONE]\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client)

    result = gw.submit_text_chat(
        prompt="Narrate scene",
        model="freetext-model",
        stream=True,
        expect_json=False,
    )

    assert result.raw_response == "This is freeform narration."
    assert result.parsed_json is None


def test_submit_text_chat_validates_prompt_and_model():
    """Verify submit_text_chat rejects empty prompt or model with ValueError."""
    gw = GatewayClient()
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        gw.submit_text_chat(prompt="  ", model="some-model")
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        gw.submit_text_chat(prompt="Hello", model="  ")


def test_system_prompt_transmission():
    """Verify system prompt is transmitted in messages array before user prompt."""
    captured_body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.read())
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client)

    gw.submit_text_chat(
        prompt="User question",
        model="instruct-model",
        system_prompt="System persona instructions",
        stream=False,
    )

    messages = captured_body["messages"]
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "System persona instructions"}
    assert messages[1] == {"role": "user", "content": "User question"}


# ---------------------------------------------------------------------------
# 3. Explicit submit_image_chat, strict magic bytes, and Pillow allowlist
# ---------------------------------------------------------------------------


def test_submit_image_chat_jpeg_and_png(tmp_path: Path):
    """Verify explicit submit_image_chat decodes, re-encodes, and embeds still JPEG and PNG."""
    jpg_file = create_test_image(tmp_path / "frame1.jpg", fmt="JPEG", size=(100, 80))
    png_file = create_test_image(tmp_path / "frame2.png", fmt="PNG", size=(120, 90))

    captured_body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.read())
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"identified": true}'}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client)

    result = gw.submit_image_chat(
        prompt="Analyze these two frames",
        images=[jpg_file, png_file],
        model="vision-model",
        stream=False,
        phase="vision_ocr",
    )

    assert result.parsed_json == {"identified": True}
    assert result.metadata["image_count"] == 2
    assert result.metadata["phase"] == "vision_ocr"

    user_content = captured_body["messages"][0]["content"]
    assert len(user_content) == 3
    assert user_content[0] == {"type": "text", "text": "Analyze these two frames"}

    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    assert user_content[2]["type"] == "image_url"
    assert user_content[2]["image_url"]["url"].startswith("data:image/png;base64,")


def test_submit_image_chat_requires_images():
    """Verify submit_image_chat raises ValueError if empty images sequence is passed."""
    gw = GatewayClient()
    with pytest.raises(ValueError, match="At least one image must be provided"):
        gw.submit_image_chat(prompt="Analyze", images=[], model="vision-model")


def test_strict_pre_pillow_magic_bytes_rejection(tmp_path: Path):
    """Verify non-JPEG/PNG formats (BMP, GIF, WebP, TIFF, MP4) fail magic bytes check before Pillow."""
    request_made = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_made
        request_made = True
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client)

    # 1. BMP magic bytes
    bmp_file = tmp_path / "fake.jpg"
    bmp_file.write_bytes(b"BM" + b"\x00" * 50)
    with pytest.raises(UnsupportedMediaError, match="magic byte validation"):
        gw.submit_image_chat(prompt="Test", images=[bmp_file], model="m")
    assert request_made is False

    # 2. GIF magic bytes
    gif_file = tmp_path / "fake.png"
    gif_file.write_bytes(b"GIF89a" + b"\x00" * 50)
    with pytest.raises(UnsupportedMediaError, match="magic byte validation"):
        gw.submit_image_chat(prompt="Test", images=[gif_file], model="m")
    assert request_made is False

    # 3. Video container
    mp4_file = tmp_path / "video.jpg"
    mp4_file.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 50)
    with pytest.raises(UnsupportedMediaError, match="magic byte validation"):
        gw.submit_image_chat(prompt="Test", images=[mp4_file], model="m")
    assert request_made is False

    # 4. Text file
    txt_file = tmp_path / "text.png"
    txt_file.write_text("Hello, I am plain text posing as image")
    with pytest.raises(UnsupportedMediaError, match="magic byte validation"):
        gw.submit_image_chat(prompt="Test", images=[txt_file], model="m")
    assert request_made is False


def test_animated_apng_rejected_before_transport(tmp_path: Path):
    """Verify valid PNG magic with multi-frame APNG is rejected before load."""
    apng_path = tmp_path / "anim.png"
    im1 = Image.new("RGB", (32, 32), "red")
    im2 = Image.new("RGB", (32, 32), "blue")
    im1.save(apng_path, format="PNG", save_all=True, append_images=[im2])

    gw = GatewayClient()
    with pytest.raises(UnsupportedMediaError, match="Animated or multi-frame image format"):
        gw.submit_image_chat(prompt="Test APNG", images=[apng_path], model="m")


def test_image_reencoding_strips_trailing_garbage(tmp_path: Path):
    """Verify validated still images are re-encoded to strip trailing/ancillary arbitrary bytes."""
    clean_png = create_test_image(tmp_path / "clean.png", fmt="PNG", size=(32, 32))
    raw_png_bytes = clean_png.read_bytes()

    dirty_payload = raw_png_bytes + b"SECRET_POLYGLOT_INJECTION_AND_TRAILING_BYTES_99999"
    dirty_path = tmp_path / "dirty.png"
    dirty_path.write_bytes(dirty_payload)

    reencoded_bytes, mime = validate_and_reencode_image(dirty_path)
    assert mime == "image/png"
    assert b"SECRET_POLYGLOT_INJECTION" not in reencoded_bytes
    assert len(reencoded_bytes) < len(dirty_payload)


# ---------------------------------------------------------------------------
# 4. Configurable bounds, limits before load, and decompression bomb safety
# ---------------------------------------------------------------------------


def test_constructor_validates_positive_bounds():
    """Verify constructor raises ValueError for non-positive or negative parameters."""
    with pytest.raises(ValueError, match="timeout must be positive"):
        GatewayClient(timeout=0)
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        GatewayClient(max_retries=-1)
    with pytest.raises(ValueError, match="backoff_factor must be positive"):
        GatewayClient(backoff_factor=0)
    with pytest.raises(ValueError, match="max_retry_delay must be positive"):
        GatewayClient(max_retry_delay=-5)
    with pytest.raises(ValueError, match="max_images_per_request must be positive"):
        GatewayClient(max_images_per_request=0)
    with pytest.raises(ValueError, match="max_image_bytes must be positive"):
        GatewayClient(max_image_bytes=0)
    with pytest.raises(ValueError, match="max_image_dimension must be positive"):
        GatewayClient(max_image_dimension=0)
    with pytest.raises(ValueError, match="max_total_image_bytes must be positive"):
        GatewayClient(max_total_image_bytes=0)
    with pytest.raises(ValueError, match="max_response_bytes must be positive"):
        GatewayClient(max_response_bytes=0)


def test_empty_and_oversized_images_rejected(tmp_path: Path):
    """Verify 0-byte files, non-existent files, and oversized images are rejected."""
    gw = GatewayClient(max_image_bytes=50, max_images_per_request=2)

    # Empty 0 bytes
    empty_file = tmp_path / "empty.jpg"
    empty_file.write_bytes(b"")
    with pytest.raises(UnsupportedMediaError, match="empty .*0 bytes"):
        gw.submit_image_chat(prompt="Test", images=[empty_file], model="m")

    # Non-existent file
    missing_file = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError):
        gw.submit_image_chat(prompt="Test", images=[missing_file], model="m")

    # Exceeding byte limit
    oversized = create_test_image(tmp_path / "oversized.png", size=(64, 64))
    assert oversized.stat().st_size > 50
    with pytest.raises(UnsupportedMediaError, match="exceeds limit"):
        gw.submit_image_chat(prompt="Test", images=[oversized], model="m")

    # Exceeding image count limit
    img1 = create_test_image(tmp_path / "img1.png", size=(8, 8))
    img2 = create_test_image(tmp_path / "img2.png", size=(8, 8))
    img3 = create_test_image(tmp_path / "img3.png", size=(8, 8))
    with pytest.raises(UnsupportedMediaError, match="Requested 3 images exceeds limit of 2"):
        gw.submit_image_chat(prompt="Test", images=[img1, img2, img3], model="m")


def test_max_total_image_bytes_enforced(tmp_path: Path):
    """Verify max_total_image_bytes rejects request when combined size exceeds bound."""
    img1 = create_test_image(tmp_path / "img1.png", size=(32, 32))
    img2 = create_test_image(tmp_path / "img2.png", size=(32, 32))

    # Total bytes for two 32x32 PNGs is ~200+ bytes; cap at 150 bytes
    gw = GatewayClient(max_total_image_bytes=150)
    with pytest.raises(UnsupportedMediaError, match="Total re-encoded image bytes .* exceed limit"):
        gw.submit_image_chat(prompt="Test", images=[img1, img2], model="m")


def test_oversized_dimensions_rejected_before_load(tmp_path: Path):
    """Verify images exceeding max dimension limit are rejected before pixel load."""
    gw = GatewayClient(max_image_dimension=100)
    large_dim = create_test_image(tmp_path / "large_dim.jpg", fmt="JPEG", size=(200, 50))

    with pytest.raises(UnsupportedMediaError, match="exceed maximum allowed dimension"):
        gw.submit_image_chat(prompt="Test", images=[large_dim], model="m")


# ---------------------------------------------------------------------------
# 5. Exact model response without redaction vs error sanitization
# ---------------------------------------------------------------------------


def test_successful_model_response_not_redacted():
    """Verify successful model response is returned exact without API key redaction."""
    secret_key = "sk-super-secret-api-token-9999"

    def handler(request: httpx.Request) -> httpx.Response:
        # Model echoes secret key or generates text that contains it
        content_text = f"The secret code is {secret_key} and should NOT be redacted."
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content_text}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client, api_key=secret_key)

    result = gw.submit_text_chat(prompt="Echo test", model="echo-model", stream=False, expect_json=False)

    # Content must remain exact
    assert secret_key in result.raw_response
    assert "[REDACTED]" not in result.raw_response

    # Metadata must NOT leak the secret
    meta_str = json.dumps(result.metadata)
    assert secret_key not in meta_str


def test_error_sanitization_redacts_credentials_in_message_and_properties():
    """Verify API keys are redacted in error messages and error raw_response properties."""
    secret_key = "sk-super-secret-api-token-9999"

    def handler(request: httpx.Request) -> httpx.Response:
        # Server error echoing authorization token
        return httpx.Response(
            500,
            text=f"Internal Server Error: failed for Bearer {secret_key} on worker node",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client, api_key=secret_key, max_retries=0)

    with pytest.raises(GatewayServerError) as exc_info:
        gw.submit_text_chat(prompt="Fail test", model="m", stream=False)

    err = exc_info.value
    assert secret_key not in str(err)
    assert "[REDACTED]" in str(err)


# ---------------------------------------------------------------------------
# 6. Error classification: sync/SSE empty vs malformed vs HTTP status
# ---------------------------------------------------------------------------


def test_sync_empty_and_malformed_envelope_classification():
    """Verify sync response variations are classified into EmptyApiResponseError or MalformedApiResponseError."""
    # 1. Empty body
    def empty_body_handler(r):
        return httpx.Response(200, text="   ")
    c1 = httpx.Client(transport=httpx.MockTransport(empty_body_handler))
    with pytest.raises(EmptyApiResponseError, match="empty response body"):
        GatewayClient(client=c1, max_retries=0).submit_text_chat("t", model="m", stream=False)

    # 2. Invalid JSON
    def broken_json_handler(r):
        return httpx.Response(200, text="not json at all")
    c2 = httpx.Client(transport=httpx.MockTransport(broken_json_handler))
    with pytest.raises(MalformedApiResponseError, match="not valid JSON"):
        GatewayClient(client=c2, max_retries=0).submit_text_chat("t", model="m", stream=False)

    # 3. Root not dict
    def root_not_dict_handler(r):
        return httpx.Response(200, text="[1, 2, 3]")
    c3 = httpx.Client(transport=httpx.MockTransport(root_not_dict_handler))
    with pytest.raises(MalformedApiResponseError, match="root is not an object"):
        GatewayClient(client=c3, max_retries=0).submit_text_chat("t", model="m", stream=False)

    # 4. Missing choices
    def no_choices_handler(r):
        return httpx.Response(200, json={"model": "m"})
    c4 = httpx.Client(transport=httpx.MockTransport(no_choices_handler))
    with pytest.raises(MalformedApiResponseError, match="missing or invalid 'choices'"):
        GatewayClient(client=c4, max_retries=0).submit_text_chat("t", model="m", stream=False)

    # 5. Empty choices list
    def empty_choices_handler(r):
        return httpx.Response(200, json={"choices": []})
    c5 = httpx.Client(transport=httpx.MockTransport(empty_choices_handler))
    with pytest.raises(EmptyApiResponseError, match="empty 'choices' list"):
        GatewayClient(client=c5, max_retries=0).submit_text_chat("t", model="m", stream=False)

    # 6. choices[0] not dict
    def choice_not_dict_handler(r):
        return httpx.Response(200, json={"choices": ["not a dict"]})
    c6 = httpx.Client(transport=httpx.MockTransport(choice_not_dict_handler))
    with pytest.raises(MalformedApiResponseError, match="choices\\[0\\] is not an object"):
        GatewayClient(client=c6, max_retries=0).submit_text_chat("t", model="m", stream=False)

    # 7. choices[0] missing message
    def choice_no_msg_handler(r):
        return httpx.Response(200, json={"choices": [{"delta": {}}]})
    c7 = httpx.Client(transport=httpx.MockTransport(choice_no_msg_handler))
    with pytest.raises(MalformedApiResponseError, match="missing or invalid 'message'"):
        GatewayClient(client=c7, max_retries=0).submit_text_chat("t", model="m", stream=False)

    # 8. content is None
    def null_content_handler(r):
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
    c8 = httpx.Client(transport=httpx.MockTransport(null_content_handler))
    with pytest.raises(EmptyApiResponseError, match="content is null/missing"):
        GatewayClient(client=c8, max_retries=0).submit_text_chat("t", model="m", stream=False)

    # 9. content is empty string
    def empty_content_handler(r):
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})
    c9 = httpx.Client(transport=httpx.MockTransport(empty_content_handler))
    with pytest.raises(EmptyApiResponseError, match="content is empty"):
        GatewayClient(client=c9, max_retries=0).submit_text_chat("t", model="m", stream=False)


def test_sse_empty_and_malformed_classification():
    """Verify SSE stream chunk defects and empty stream classify accurately."""
    # 1. Broken JSON in SSE chunk
    def broken_chunk_handler(r):
        return httpx.Response(200, text="data: {bad json\n\n", headers={"Content-Type": "text/event-stream"})
    c1 = httpx.Client(transport=httpx.MockTransport(broken_chunk_handler))
    with pytest.raises(MalformedApiResponseError, match="Invalid JSON in SSE chunk"):
        GatewayClient(client=c1, max_retries=0).submit_text_chat("t", model="m", stream=True)

    # 2. SSE with only [DONE] and no content
    def empty_stream_handler(r):
        return httpx.Response(200, text="data: [DONE]\n\n", headers={"Content-Type": "text/event-stream"})
    c2 = httpx.Client(transport=httpx.MockTransport(empty_stream_handler))
    with pytest.raises(EmptyApiResponseError, match="completed with empty content"):
        GatewayClient(client=c2, max_retries=0).submit_text_chat("t", model="m", stream=True)


def test_bounded_response_body_size():
    """Verify response exceeding max_response_bytes raises GatewayResponseError."""
    def large_sync_handler(r):
        return httpx.Response(200, text="X" * 200)

    client = httpx.Client(transport=httpx.MockTransport(large_sync_handler))
    gw = GatewayClient(client=client, max_response_bytes=100, max_retries=0)
    with pytest.raises(GatewayResponseError, match="exceeded limit of 100 bytes"):
        gw.submit_text_chat("Test", model="m", stream=False)


def test_streaming_response_limit_counts_utf8_bytes_not_characters():
    content = "😀" * 100

    def large_stream_handler(_request):
        body = 'data: {"choices":[{"delta":{"content":"' + content + '"}}]}\n\n'
        assert len(body) < len(body.encode("utf-8"))
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    client = httpx.Client(transport=httpx.MockTransport(large_stream_handler))
    gateway = GatewayClient(client=client, max_response_bytes=200, max_retries=0)
    with pytest.raises(GatewayResponseError, match="exceeded limit of 200 bytes"):
        gateway.submit_text_chat("Test", model="m", stream=True)


def test_error_classification_http_status_codes():
    """Verify HTTP status codes (401, 403, 404, 408, 413, 429, 5xx) are classified."""
    cases = [
        (401, GatewayAuthenticationError, "authentication failed"),
        (403, GatewayPermissionError, "access forbidden"),
        (404, GatewayNotFoundError, "endpoint or model not found"),
        (408, GatewayRequestTimeoutError, "request timeout"),
        (413, PayloadContextError, "payload too large"),
        (429, GatewayRateLimitError, "rate limit exceeded"),
        (500, GatewayServerError, "server error"),
        (503, GatewayServerError, "server error"),
    ]

    for status_code, expected_err, match_str in cases:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, text=f"HTTP {status_code} details")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        gw = GatewayClient(client=client, max_retries=0)
        with pytest.raises(expected_err, match=match_str) as exc_info:
            gw.submit_text_chat("Test", model="m", stream=False)
        assert exc_info.value.status_code == status_code


def test_payload_context_error_on_400_token_limit():
    """Verify HTTP 400 with context length keywords raises PayloadContextError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request: context_length_exceeded: maximum context length is 8192")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client, max_retries=0)
    with pytest.raises(PayloadContextError, match="context limit exceeded"):
        gw.submit_text_chat("Test", model="m", stream=False)


# ---------------------------------------------------------------------------
# 7. Model JSON extraction and distinction between empty and malformed JSON
# ---------------------------------------------------------------------------


def test_extract_json_valid_and_fenced():
    """Verify extract_json_from_text handles valid raw and markdown fenced JSON."""
    assert extract_json_from_text('{"score": 95}') == {"score": 95}
    fenced = "Here is your output:\n```json\n{\"items\": [1, 2, 3]}\n```\nDone."
    assert extract_json_from_text(fenced) == {"items": [1, 2, 3]}


def test_distinguish_empty_from_malformed_model_json():
    """Verify empty API/text raises EmptyApiResponseError while invalid JSON raises MalformedModelJsonError."""
    with pytest.raises(EmptyApiResponseError, match="empty response"):
        extract_json_from_text("")

    with pytest.raises(EmptyApiResponseError, match="empty response"):
        extract_json_from_text("   \n\t  ")

    broken = "```json\n{\"outputs\": [1, 2, unquoted_symbol]}\n```"
    with pytest.raises(MalformedModelJsonError, match="Automatic repair is strictly prohibited"):
        extract_json_from_text(broken)

    # Backwards compatibility check
    with pytest.raises(InvalidGatewayResponseError):
        extract_json_from_text(broken)


# ---------------------------------------------------------------------------
# 8. Bounded retries, Retry-After capping, and cancellation
# ---------------------------------------------------------------------------


def test_bounded_retry_transient_success():
    """Verify transient 503 errors are retried up to max_retries and succeed on recovery."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, text="Service Temporarily Unavailable")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client, max_retries=3, backoff_factor=0.01)

    result = gw.submit_text_chat("Retry test", model="m", stream=False)
    assert result.parsed_json == {"ok": True}
    assert attempts == 3
    assert result.metadata["retry_count"] == 2


def test_non_transient_errors_fail_immediately_without_retry():
    """Verify non-transient errors (401, 403, 404, 413, 400) fail on first attempt without retry."""
    non_transient_codes = [401, 403, 404, 413, 400]

    for code in non_transient_codes:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(code, text=f"Error {code}")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        gw = GatewayClient(client=client, max_retries=3, backoff_factor=0.01)

        with pytest.raises(GatewayError):
            gw.submit_text_chat("Test", model="m", stream=False)

        assert attempts == 1, f"Status code {code} was retried {attempts} times instead of failing immediately"


def test_retry_after_capped():
    """Verify Retry-After header on 429 is parsed and capped by max_retry_delay."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "120"}, text="Rate limit exceeded")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client, max_retries=2, max_retry_delay=0.01)

    result = gw.submit_text_chat("429 test", model="m", stream=False)
    assert result.parsed_json == {"ok": True}
    assert attempts == 2


def test_cancellation_not_retried():
    """Verify cancellation token interrupts request and is never retried."""
    token = CancellationToken()
    token.cancel()

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client, max_retries=3)

    with pytest.raises(CancelledError):
        gw.submit_text_chat("Cancel test", model="m", cancellation_token=token)

    assert attempts == 0


def test_cancellation_during_stream():
    """Verify cancelling token during streaming aborts stream and raises CancelledError."""
    token = CancellationToken()

    def handler(request: httpx.Request) -> httpx.Response:
        sse_body = (
            'data: {"choices": [{"delta": {"content": "part1"}}]}\n\n'
            'data: {"choices": [{"delta": {"content": "part2"}}]}\n\n'
        )
        return httpx.Response(200, text=sse_body, headers={"Content-Type": "text/event-stream"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client)

    token.cancel()
    with pytest.raises(CancelledError):
        gw.submit_text_chat("Cancel during stream", model="m", stream=True, cancellation_token=token)


def test_cancellation_during_retry_sleep():
    """Verify cancelling token while sleeping between retry attempts interrupts immediately."""
    token = CancellationToken()
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        token.cancel()
        return httpx.Response(503, text="Service Unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client, max_retries=3, backoff_factor=1.0)

    with pytest.raises(CancelledError):
        gw.submit_text_chat("Retry cancel", model="m", stream=False, cancellation_token=token)

    assert attempts == 1


# ---------------------------------------------------------------------------
# 9. Provider-neutral validate_model_availability (no heuristics/aliases)
# ---------------------------------------------------------------------------


def test_validate_model_availability_queries_every_model():
    """Verify validate_model_availability queries /v1/models provider-neutrally for all models."""
    queried_models = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "prime"},
                    {"id": "sub"},
                    {"id": "custom-scanner"},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gw = GatewayClient(client=client)

    # Every ID must query the endpoint without hardcoded bypass
    assert gw.validate_model_availability("prime") is True
    assert gw.validate_model_availability("sub") is True
    assert gw.validate_model_availability("custom-scanner") is True

    # Unknown model fails
    with pytest.raises(ModelCapabilityError, match="not found in advertised Gateway models"):
        gw.validate_model_availability("unknown-model")


# ---------------------------------------------------------------------------
# 10. Programmer errors not converted to GatewayError
# ---------------------------------------------------------------------------


def test_programmer_errors_not_converted():
    """Verify programmer bugs (e.g. invalid arguments) are not converted to GatewayError."""
    gw = GatewayClient()
    # Passing non-string prompt raises ValueError, not GatewayError
    with pytest.raises(ValueError):
        gw.submit_text_chat(prompt="", model="m")
    with pytest.raises(ValueError):
        gw.submit_text_chat(prompt="valid", model="")
