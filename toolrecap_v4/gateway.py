"""GatewayClient for 9router AI Gateway.

Invariants:
- Safe provider-neutral text/JSON and explicit selected still-image transport only.
- Production whole-video transport APIs and video capability validation are completely removed.
- Validates genuine still JPEG/PNG via strict pre-Pillow magic bytes, formats allowlist,
  and Pillow image decode.
- Animated, multi-frame (GIF, APNG, multi-frame TIFF), video, and arbitrary binary formats are strictly rejected.
- Catches Image.DecompressionBombError / DecompressionBombWarning safely and enforces limits before load.
- Validated still images are re-encoded to clean buffers to strip ancillary, trailing, or polyglot bytes.
- submit_text_chat is text-only (no images parameter); submit_image_chat is the sole vetted media path.
- Enforces strict configurable positive bounds on images per request, byte sizes, dimensions, and total image bytes.
- No raw caller request escape hatch: caller cannot supply raw messages or unverified media.
- Measures exact serialized bytes sent over the HTTP transport.
- Metadata is sanitized without prompts, bodies, or secrets.
- Successful model response text is returned exact without API key redaction.
  Sanitization applies strictly to errors, metadata, and error properties to prevent credential leakage.
- Full error taxonomy classification: connection, timeout, 401, 403, 404, 408, 429, 5xx,
  payload context, malformed API envelope, empty API response, model JSON defects, cancel.
- Bounded cancellation-aware retries with capped Retry-After delays; cancellation is never retried.
- Provider-neutral model availability check queries advertised models for every ID without name heuristics.
- No generic 'except Exception' converting programmer defects.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import httpx
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError, DecompressionBombWarning

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.endpoints import (
    normalize_http_endpoint,
    tailscale_connection_hint,
    use_environment_proxy,
)
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

SUPPORTED_IMAGE_FORMATS = ("JPEG", "PNG")
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

DEFAULT_MAX_IMAGES_PER_REQUEST = 16
DEFAULT_MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MiB
DEFAULT_MAX_IMAGE_DIMENSION = 4096  # 4096 x 4096 px
DEFAULT_MAX_TOTAL_IMAGE_BYTES = 16 * 1024 * 1024  # 16 MiB
DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MiB
DEFAULT_MAX_RETRY_DELAY_SECONDS = 60.0
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def sanitize_message(msg: str, secrets: Optional[List[Optional[str]]] = None) -> str:
    """Redact secret strings from message to prevent credential leakage in logs and errors."""
    if not secrets:
        return msg
    sanitized = msg
    for s in secrets:
        if s and len(s) >= 4 and s in sanitized:
            sanitized = sanitized.replace(s, "[REDACTED]")
    return sanitized


def validate_and_reencode_image(
    source: Union[Path, str, bytes],
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
) -> Tuple[bytes, str]:
    """Validate image is genuine still JPEG or PNG and re-encode to strip metadata/trailing bytes.

    Applies:
    1. Independent pre-Pillow magic byte detection.
    2. Pillow formats allowlist restricting parser to ['JPEG', 'PNG'].
    3. Safe DecompressionBombWarning/Error handling.
    4. Dimension and multi-frame animation checks before pixel decode.
    5. Clean re-encoding discarding trailing bytes, polyglots, and ancillary metadata.

    Returns:
        Tuple of (reencoded_bytes, mime_type).

    Raises:
        UnsupportedMediaError: If image fails magic bytes, decoding, has unsupported format,
            is animated/multi-frame, exceeds byte/dimension limits, or decompression bomb.
        FileNotFoundError: If a file path is provided that does not exist.
    """
    if isinstance(source, (Path, str)):
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        file_size = path.stat().st_size
        if file_size <= 0:
            raise UnsupportedMediaError(f"Image file '{path.name}' is empty (0 bytes).")
        if file_size > max_bytes:
            raise UnsupportedMediaError(
                f"Image file '{path.name}' ({file_size} bytes) exceeds limit ({max_bytes} bytes)."
            )
        raw_bytes = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw_bytes = bytes(source)
        if len(raw_bytes) == 0:
            raise UnsupportedMediaError("Image bytes are empty (0 bytes).")
        if len(raw_bytes) > max_bytes:
            raise UnsupportedMediaError(
                f"Image bytes ({len(raw_bytes)} bytes) exceed limit ({max_bytes} bytes)."
            )
    else:
        raise UnsupportedMediaError(f"Invalid image source type: {type(source).__name__}")

    # Strict pre-Pillow magic bytes verification
    is_jpeg = raw_bytes.startswith(JPEG_MAGIC)
    is_png = raw_bytes.startswith(PNG_MAGIC)
    if not (is_jpeg or is_png):
        raise UnsupportedMediaError(
            "Image failed magic byte validation. Only genuine still JPEG and PNG images are supported."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=DecompressionBombWarning)
        try:
            bio = io.BytesIO(raw_bytes)
            with Image.open(bio, formats=["JPEG", "PNG"]) as img:
                fmt = (img.format or "").upper()
                if fmt not in SUPPORTED_IMAGE_FORMATS:
                    raise UnsupportedMediaError(
                        f"Unsupported media format '{fmt}'. Only genuine still JPEG and PNG images are supported."
                    )

                # Dimension limits before load
                width, height = img.size
                if width <= 0 or height <= 0:
                    raise UnsupportedMediaError(f"Invalid image dimensions: {width}x{height}.")
                if width > max_dimension or height > max_dimension:
                    raise UnsupportedMediaError(
                        f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({max_dimension}px)."
                    )

                # Multi-frame / animation check before load
                if getattr(img, "is_animated", False) or getattr(img, "n_frames", 1) > 1:
                    raise UnsupportedMediaError(
                        f"Animated or multi-frame image format ({fmt}) is strictly prohibited."
                    )

                # Force decoding of pixel data to catch truncated or corrupted streams
                img.load()

                # Re-encode to clean buffer to discard any trailing garbage or unwanted metadata
                out_buf = io.BytesIO()
                if fmt == "JPEG":
                    save_img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
                    save_img.save(out_buf, format="JPEG", quality=90)
                    mime_type = "image/jpeg"
                else:  # PNG
                    save_img = img if img.mode in ("RGB", "RGBA", "L") else img.convert("RGBA")
                    save_img.save(out_buf, format="PNG")
                    mime_type = "image/png"

                reencoded = out_buf.getvalue()

        except UnsupportedMediaError:
            raise
        except (DecompressionBombError, DecompressionBombWarning) as e:
            raise UnsupportedMediaError(f"Image decompression bomb detected: {e}") from e
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            raise UnsupportedMediaError(f"Image decode failed or format spoofed: {e}") from e

    if len(reencoded) > max_bytes:
        raise UnsupportedMediaError(
            f"Re-encoded image ({len(reencoded)} bytes) exceeds limit ({max_bytes} bytes)."
        )

    return reencoded, mime_type


def extract_json_from_text(text: str) -> Any:
    """Extract and parse JSON from text without repairing editorial content.

    Handles plain JSON or fenced ```json ... ``` blocks.
    Raises:
        EmptyApiResponseError: If response text is empty or contains no content.
        MalformedModelJsonError: If response text cannot be parsed as JSON.
    """
    stripped = text.strip()
    if not stripped:
        raise EmptyApiResponseError(
            "Gateway returned an empty response; cannot parse JSON.",
            raw_response=text,
        )

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        candidate = stripped

    if not candidate:
        raise EmptyApiResponseError(
            "Gateway returned an empty JSON code fence; cannot parse JSON.",
            raw_response=text,
        )

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise MalformedModelJsonError(
            f"Gateway model output failed JSON parsing ({e}). Automatic repair is strictly prohibited.",
            raw_response=text,
        ) from e


def _parse_retry_after(header_val: Optional[str]) -> Optional[float]:
    """Parse HTTP Retry-After header into seconds."""
    if not header_val:
        return None
    try:
        val = float(header_val.strip())
        if val >= 0:
            return val
    except ValueError:
        pass
    return None


@dataclass
class GatewayResult:
    """Container for Gateway output."""

    raw_response: str
    parsed_json: Optional[Union[Dict[str, Any], List[Any]]] = None
    model: str = ""
    usage: Optional[Dict[str, Any]] = None
    bytes_sent: int = 0
    metadata: Optional[Dict[str, Any]] = None


class GatewayClient:
    """Client for communicating with the 9router AI Gateway."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:20128",
        api_key: Optional[str] = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        max_retry_delay: float = DEFAULT_MAX_RETRY_DELAY_SECONDS,
        max_images_per_request: int = DEFAULT_MAX_IMAGES_PER_REQUEST,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
        max_total_image_bytes: int = DEFAULT_MAX_TOTAL_IMAGE_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        client: Optional[httpx.Client] = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")
        if backoff_factor <= 0:
            raise ValueError(f"backoff_factor must be positive, got {backoff_factor}")
        if max_retry_delay <= 0:
            raise ValueError(f"max_retry_delay must be positive, got {max_retry_delay}")
        if max_images_per_request <= 0:
            raise ValueError(f"max_images_per_request must be positive, got {max_images_per_request}")
        if max_image_bytes <= 0:
            raise ValueError(f"max_image_bytes must be positive, got {max_image_bytes}")
        if max_image_dimension <= 0:
            raise ValueError(f"max_image_dimension must be positive, got {max_image_dimension}")
        if max_total_image_bytes <= 0:
            raise ValueError(f"max_total_image_bytes must be positive, got {max_total_image_bytes}")
        if max_response_bytes <= 0:
            raise ValueError(f"max_response_bytes must be positive, got {max_response_bytes}")

        self.base_url = normalize_http_endpoint(base_url, "Gateway Endpoint")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_retry_delay = max_retry_delay
        self.max_images_per_request = max_images_per_request
        self.max_image_bytes = max_image_bytes
        self.max_image_dimension = max_image_dimension
        self.max_total_image_bytes = max_total_image_bytes
        self.max_response_bytes = max_response_bytes
        self._external_client = client

    def _get_client(self) -> httpx.Client:
        if self._external_client is not None:
            return self._external_client
        return httpx.Client(
            timeout=self.timeout,
            trust_env=use_environment_proxy(self.base_url),
        )

    def _sanitize(self, text: str) -> str:
        return sanitize_message(text, [self.api_key])

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _classify_http_error(self, resp: httpx.Response, sanitized_body: str) -> GatewayError:
        """Classify HTTP response status code into specific GatewayError hierarchy."""
        status = resp.status_code
        if status == 401:
            return GatewayAuthenticationError(
                f"Gateway authentication failed (HTTP 401): {sanitized_body}",
                status_code=401,
            )
        if status == 403:
            return GatewayPermissionError(
                f"Gateway access forbidden (HTTP 403): {sanitized_body}",
                status_code=403,
            )
        if status == 404:
            return GatewayNotFoundError(
                f"Gateway endpoint or model not found (HTTP 404): {sanitized_body}",
                status_code=404,
            )
        if status == 408:
            return GatewayRequestTimeoutError(
                f"Gateway request timeout (HTTP 408): {sanitized_body}",
                status_code=408,
            )
        if status == 413:
            return PayloadContextError(
                f"Gateway payload too large (HTTP 413): {sanitized_body}",
                status_code=413,
            )
        if status == 400:
            lower_body = sanitized_body.lower()
            if any(
                kw in lower_body
                for kw in (
                    "context_length",
                    "context window",
                    "maximum context",
                    "too many tokens",
                    "prompt is too long",
                )
            ):
                return PayloadContextError(
                    f"Gateway context limit exceeded (HTTP 400): {sanitized_body}",
                    status_code=400,
                )
            return GatewayResponseError(
                f"Gateway returned HTTP 400: {sanitized_body}",
                raw_response=sanitized_body,
                status_code=400,
            )
        if status == 429:
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            return GatewayRateLimitError(
                f"Gateway rate limit exceeded (HTTP 429): {sanitized_body}",
                retry_after=retry_after,
                status_code=429,
            )
        if 500 <= status <= 599:
            return GatewayServerError(
                f"Gateway server error (HTTP {status}): {sanitized_body}",
                status_code=status,
            )
        return GatewayResponseError(
            f"Gateway returned HTTP {status}: {sanitized_body}",
            raw_response=sanitized_body,
            status_code=status,
        )

    def validate_model_availability(
        self,
        model: str,
        cancellation_token: Optional[CancellationToken] = None,
        client: Optional[httpx.Client] = None,
    ) -> bool:
        """Validate that target model is available on the Gateway without vendor heuristics.

        Queries GET /v1/models provider-neutrally for every model ID.
        """
        if not model or not model.strip():
            raise ValueError("Model identifier cannot be empty.")

        if cancellation_token:
            cancellation_token.check_cancelled()

        url = f"{self.base_url}/v1/models"
        owned = False
        if client is None:
            if self._external_client is not None:
                client = self._external_client
            else:
                client = httpx.Client(
                    timeout=min(10.0, self.timeout),
                    trust_env=use_environment_proxy(self.base_url),
                )
                owned = True

        try:
            resp = client.get(url, headers=self._headers(), timeout=min(10.0, self.timeout))
            if resp.status_code != 200:
                body = resp.text[:300]
                clean_body = self._sanitize(body)
                raise self._classify_http_error(resp, clean_body)

            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                raise MalformedApiResponseError(
                    "Gateway models endpoint returned invalid JSON",
                    raw_response=self._sanitize(resp.text[:300]),
                ) from e

            models_list = data.get("data", []) if isinstance(data, dict) else []
            for m in models_list:
                if isinstance(m, dict) and m.get("id") == model:
                    return True

            raise ModelCapabilityError(f"Model '{model}' not found in advertised Gateway models.")
        except GatewayError:
            raise
        except httpx.TimeoutException as e:
            clean_err = self._sanitize(str(e))
            raise GatewayTimeoutError(f"Timeout checking model availability: {clean_err}") from e
        except (httpx.ConnectError, httpx.NetworkError) as e:
            clean_err = self._sanitize(str(e))
            raise GatewayConnectionError(
                f"Connection error checking model availability: {clean_err}"
                f"{tailscale_connection_hint(self.base_url)}"
            ) from e
        except (httpx.TransportError, httpx.HTTPError) as e:
            clean_err = self._sanitize(str(e))
            raise GatewayConnectionError(
                f"Transport error checking model availability: {clean_err}"
                f"{tailscale_connection_hint(self.base_url)}"
            ) from e
        finally:
            if owned:
                client.close()

    def submit_text_chat(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        stream: bool = True,
        expect_json: bool = True,
        cancellation_token: Optional[CancellationToken] = None,
        phase: Optional[str] = None,
    ) -> GatewayResult:
        """Submit text-only chat request in a provider-neutral boundary.

        Invariant: Strictly text-only. No images parameter or media bypass.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not model or not model.strip():
            raise ValueError("Model identifier cannot be empty.")

        messages: List[Dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        payload_dict: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if reasoning_effort and reasoning_effort.strip():
            payload_dict["reasoning_effort"] = reasoning_effort.strip().lower()

        return self._execute_chat_request(
            payload_dict=payload_dict,
            model=model,
            image_count=0,
            stream=stream,
            expect_json=expect_json,
            cancellation_token=cancellation_token,
            phase=phase,
        )

    def submit_image_chat(
        self,
        prompt: str,
        images: Sequence[Union[Path, str, bytes]],
        model: str,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        stream: bool = True,
        expect_json: bool = True,
        cancellation_token: Optional[CancellationToken] = None,
        phase: Optional[str] = None,
    ) -> GatewayResult:
        """Submit prompt with explicit validated still JPEG/PNG images.

        Invariant: Sole public path for media transport. Validates and re-encodes images.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not model or not model.strip():
            raise ValueError("Model identifier cannot be empty.")
        if not images:
            raise ValueError("At least one image must be provided for submit_image_chat.")
        if len(images) > self.max_images_per_request:
            raise UnsupportedMediaError(
                f"Requested {len(images)} images exceeds limit of {self.max_images_per_request} images per request."
            )

        messages: List[Dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        user_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        total_bytes = 0

        for img_src in images:
            encoded_bytes, mime_type = validate_and_reencode_image(
                img_src,
                max_bytes=self.max_image_bytes,
                max_dimension=self.max_image_dimension,
            )
            total_bytes += len(encoded_bytes)
            if total_bytes > self.max_total_image_bytes:
                raise UnsupportedMediaError(
                    f"Total re-encoded image bytes ({total_bytes} bytes) exceed limit ({self.max_total_image_bytes} bytes)."
                )
            b64_str = base64.b64encode(encoded_bytes).decode("ascii")
            user_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_str}"},
            })

        messages.append({"role": "user", "content": user_parts})

        payload_dict: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if reasoning_effort and reasoning_effort.strip():
            payload_dict["reasoning_effort"] = reasoning_effort.strip().lower()

        return self._execute_chat_request(
            payload_dict=payload_dict,
            model=model,
            image_count=len(images),
            stream=stream,
            expect_json=expect_json,
            cancellation_token=cancellation_token,
            phase=phase,
        )

    def _execute_chat_request(
        self,
        payload_dict: Dict[str, Any],
        model: str,
        image_count: int,
        stream: bool,
        expect_json: bool,
        cancellation_token: Optional[CancellationToken],
        phase: Optional[str],
    ) -> GatewayResult:
        """Execute chat completion request with exact measurement and bounded retries."""
        if cancellation_token:
            cancellation_token.check_cancelled()

        serialized_bytes = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
        exact_bytes_sent = len(serialized_bytes)

        owned_client = False
        if self._external_client is not None:
            client = self._external_client
        else:
            client = httpx.Client(
                timeout=self.timeout,
                trust_env=use_environment_proxy(self.base_url),
            )
            owned_client = True

        url = f"{self.base_url}/v1/chat/completions"
        headers = self._headers()
        headers["Content-Length"] = str(exact_bytes_sent)

        last_transient_error: Optional[GatewayError] = None

        try:
            for attempt in range(self.max_retries + 1):
                if cancellation_token:
                    cancellation_token.check_cancelled()

                t0 = time.monotonic()
                try:
                    if stream:
                        raw_text, usage = self._execute_streaming_request(
                            client, url, serialized_bytes, headers, cancellation_token
                        )
                    else:
                        raw_text, usage = self._execute_sync_request(
                            client, url, serialized_bytes, headers, cancellation_token
                        )

                    duration_ms = (time.monotonic() - t0) * 1000.0

                    # Exact model response returned; no redaction applied to valid content
                    if expect_json:
                        parsed = extract_json_from_text(raw_text)
                    else:
                        try:
                            parsed = extract_json_from_text(raw_text)
                        except (MalformedModelJsonError, EmptyApiResponseError):
                            parsed = None

                    metadata: Dict[str, Any] = {
                        "phase": phase,
                        "model": model,
                        "retry_count": attempt,
                        "image_count": image_count,
                        "bytes_sent": exact_bytes_sent,
                        "status_code": 200,
                        "duration_ms": round(duration_ms, 2),
                    }

                    return GatewayResult(
                        raw_response=raw_text,
                        parsed_json=parsed,
                        model=model,
                        usage=usage,
                        bytes_sent=exact_bytes_sent,
                        metadata=metadata,
                    )

                except CancelledError:
                    raise
                except (
                    UnsupportedMediaError,
                    GatewayAuthenticationError,
                    GatewayPermissionError,
                    GatewayNotFoundError,
                    PayloadContextError,
                    MalformedApiResponseError,
                    EmptyApiResponseError,
                    MalformedModelJsonError,
                ):
                    raise
                except GatewayResponseError as e:
                    if e.status_code not in TRANSIENT_STATUS_CODES:
                        raise
                    last_transient_error = e
                except (GatewayServerError, GatewayRateLimitError, GatewayRequestTimeoutError) as e:
                    last_transient_error = e
                except httpx.TimeoutException as e:
                    last_transient_error = GatewayTimeoutError(
                        f"Gateway request timed out: {self._sanitize(str(e))}"
                    )
                except (httpx.ConnectError, httpx.NetworkError) as e:
                    last_transient_error = GatewayConnectionError(
                        f"Gateway connection error: {self._sanitize(str(e))}"
                        f"{tailscale_connection_hint(self.base_url)}"
                    )
                except (httpx.TransportError, httpx.HTTPError) as e:
                    clean_err = self._sanitize(str(e))
                    last_transient_error = GatewayConnectionError(
                        f"Gateway transport error: {clean_err}"
                        f"{tailscale_connection_hint(self.base_url)}"
                    )

                # Retry transient error with bounded backoff
                if attempt < self.max_retries:
                    if (
                        isinstance(last_transient_error, GatewayRateLimitError)
                        and last_transient_error.retry_after is not None
                    ):
                        sleep_time = min(last_transient_error.retry_after, self.max_retry_delay)
                    else:
                        sleep_time = min(self.backoff_factor * (2 ** attempt), self.max_retry_delay)

                    if cancellation_token:
                        end_time = time.monotonic() + sleep_time
                        while time.monotonic() < end_time:
                            cancellation_token.check_cancelled()
                            time.sleep(min(0.05, max(0.0, end_time - time.monotonic())))
                    else:
                        time.sleep(sleep_time)
                else:
                    break

            if last_transient_error:
                raise last_transient_error
            raise GatewayError("Gateway request failed: retries exhausted.")
        finally:
            if owned_client:
                client.close()

    def _execute_streaming_request(
        self,
        client: httpx.Client,
        url: str,
        content_bytes: bytes,
        headers: Dict[str, str],
        cancellation_token: Optional[CancellationToken],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Execute cancel-aware streaming request handling SSE chunks."""
        collected_chunks: List[str] = []
        usage: Optional[Dict[str, Any]] = None
        total_streamed_chars = 0

        with client.stream("POST", url, headers=headers, content=content_bytes, timeout=self.timeout) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")[:500]
                clean_body = self._sanitize(body)
                raise self._classify_http_error(resp, clean_body)

            for line in resp.iter_lines():
                if cancellation_token and cancellation_token.is_cancelled:
                    resp.close()
                    raise CancelledError("Gateway request cancelled during stream.")

                line_str = line.strip()
                if not line_str or line_str.startswith(":"):
                    continue

                total_streamed_chars += len(line_str.encode("utf-8"))
                if total_streamed_chars > self.max_response_bytes:
                    resp.close()
                    raise GatewayResponseError(
                        f"Gateway streaming response exceeded limit of {self.max_response_bytes} bytes.",
                        status_code=resp.status_code,
                    )

                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError as e:
                        raise MalformedApiResponseError(
                            f"Invalid JSON in SSE chunk: {self._sanitize(data_str[:200])}",
                            raw_response=self._sanitize(data_str[:500]),
                        ) from e

                    if not isinstance(chunk, dict):
                        raise MalformedApiResponseError(
                            "SSE chunk is not a JSON object.",
                            raw_response=self._sanitize(data_str[:500]),
                        )

                    choices = chunk.get("choices")
                    if choices is not None:
                        if not isinstance(choices, list):
                            raise MalformedApiResponseError(
                                "SSE chunk 'choices' is not a list.",
                                raw_response=self._sanitize(data_str[:500]),
                            )
                        if len(choices) > 0:
                            c0 = choices[0]
                            if not isinstance(c0, dict):
                                raise MalformedApiResponseError(
                                    "SSE chunk choices[0] is not an object.",
                                    raw_response=self._sanitize(data_str[:500]),
                                )
                            delta = c0.get("delta")
                            if delta is not None and isinstance(delta, dict):
                                piece = delta.get("content")
                                if piece is not None:
                                    if not isinstance(piece, str):
                                        raise MalformedApiResponseError(
                                            "SSE chunk delta content is not a string.",
                                            raw_response=self._sanitize(data_str[:500]),
                                        )
                                    collected_chunks.append(piece)

                    if "usage" in chunk and isinstance(chunk["usage"], dict):
                        usage = chunk["usage"]

        full_text = "".join(collected_chunks)
        if not full_text.strip():
            raise EmptyApiResponseError(
                "Gateway streaming response completed with empty content.",
                raw_response="",
            )

        return full_text, usage

    def _execute_sync_request(
        self,
        client: httpx.Client,
        url: str,
        content_bytes: bytes,
        headers: Dict[str, str],
        cancellation_token: Optional[CancellationToken],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Execute synchronous (stream=False) request with cancellation awareness."""
        if cancellation_token:
            cancellation_token.check_cancelled()

        resp = client.post(url, headers=headers, content=content_bytes, timeout=self.timeout)

        if cancellation_token:
            cancellation_token.check_cancelled()

        if resp.status_code != 200:
            body = resp.text[:500]
            clean_body = self._sanitize(body)
            raise self._classify_http_error(resp, clean_body)

        if len(resp.content) > self.max_response_bytes:
            raise GatewayResponseError(
                f"Gateway response size ({len(resp.content)} bytes) exceeded limit of {self.max_response_bytes} bytes.",
                status_code=resp.status_code,
            )

        if not resp.text or not resp.text.strip():
            raise EmptyApiResponseError(
                "Gateway returned an empty response body.",
                raw_response="",
                status_code=resp.status_code,
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise MalformedApiResponseError(
                f"Gateway response is not valid JSON: {self._sanitize(resp.text[:200])}",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            ) from e

        if not isinstance(data, dict):
            raise MalformedApiResponseError(
                "Gateway response JSON root is not an object.",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            )

        if "choices" not in data or not isinstance(data["choices"], list):
            raise MalformedApiResponseError(
                "Gateway response missing or invalid 'choices' list.",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            )

        choices = data["choices"]
        if len(choices) == 0:
            raise EmptyApiResponseError(
                "Gateway returned response with empty 'choices' list.",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            )

        choice_0 = choices[0]
        if not isinstance(choice_0, dict):
            raise MalformedApiResponseError(
                "Gateway response choices[0] is not an object.",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            )

        if "message" not in choice_0 or not isinstance(choice_0["message"], dict):
            raise MalformedApiResponseError(
                "Gateway response choices[0] missing or invalid 'message' object.",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            )

        content = choice_0["message"].get("content")
        if content is None:
            raise EmptyApiResponseError(
                "Gateway response choices[0].message.content is null/missing.",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            )

        if not isinstance(content, str):
            raise MalformedApiResponseError(
                f"Gateway response content must be string, got {type(content).__name__}.",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            )

        if not content.strip():
            raise EmptyApiResponseError(
                "Gateway response choices[0].message.content is empty.",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            )

        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise MalformedApiResponseError(
                "Gateway response 'usage' field is invalid.",
                raw_response=self._sanitize(resp.text[:500]),
                status_code=resp.status_code,
            )

        return content, usage
