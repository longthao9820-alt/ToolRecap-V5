"""Single VoiceStudio adapter supporting Local, Remote (via Tailscale), and Auto routing.

Invariants:
- Health, voices, and speech WAV handled via the SAME adapter.
- Auto mode: checks Local, then Remote only before synthesis.
- No engine fallback under any circumstances.
- Actual WAV validation: rejects corrupt WAV or zero/invalid duration.
- Auth separated (remote bearer token isolated from local).
- Bounded retries for transient errors only; logical errors fail immediately.
- Cancellation closes request immediately and raises CancelledError.
- Credential leakage prevented in all exceptions and logs.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple
import wave

import httpx

from toolrecap_v4.endpoints import (
    normalize_http_endpoint,
    tailscale_connection_hint,
    use_environment_proxy,
)

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    CancelledError,
    InvalidAudioError,
    VoiceStudioError,
    VoiceStudioUnavailableError,
)

TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Official 12 designed voice archetypes from ToolRecap V2 (VoiceStudio v0.5.3 backend/core/archetypes.py)
V2_VOICE_PRESETS: Dict[str, Dict[str, Any]] = {
    "voicestudio.en.neighbor": {
        "voice_id": "voicestudio.en.neighbor",
        "name": "Neighbor",
        "display_name": "Neighbor — Nữ — ToolRecap Local",
        "gender": "Female",
        "language": "en-US",
        "description": "Film recap / Storytelling",
        "style": "film_recap",
        "instruct": "female, young adult, moderate pitch, american accent",
    },
    "voicestudio.en.companion": {
        "voice_id": "voicestudio.en.companion",
        "name": "Companion",
        "display_name": "Companion — Nữ — ToolRecap Local",
        "gender": "Female",
        "language": "en-US",
        "description": "Film recap / Storytelling",
        "style": "film_recap",
        "instruct": "female, middle-aged, moderate pitch, canadian accent",
    },
    "voicestudio.en.teacher": {
        "voice_id": "voicestudio.en.teacher",
        "name": "Teacher",
        "display_name": "Teacher — Nữ — ToolRecap Local",
        "gender": "Female",
        "language": "en-US",
        "description": "Film recap / Storytelling",
        "style": "film_recap",
        "instruct": "female, middle-aged, moderate pitch, american accent",
    },
    "voicestudio.en.anchor": {
        "voice_id": "voicestudio.en.anchor",
        "name": "Anchor",
        "display_name": "Anchor — Nam — ToolRecap Local",
        "gender": "Male",
        "language": "en-US",
        "description": "Documentary / News",
        "style": "documentary",
        "instruct": "male, middle-aged, moderate pitch, american accent",
    },
    "voicestudio.en.documentarian": {
        "voice_id": "voicestudio.en.documentarian",
        "name": "Documentarian",
        "display_name": "Documentarian — Nam — ToolRecap Local",
        "gender": "Male",
        "language": "en-US",
        "description": "Documentary / Recap (Default US)",
        "style": "documentary",
        "instruct": "male, middle-aged, low pitch, american accent",
    },
    "voicestudio.en.promo": {
        "voice_id": "voicestudio.en.promo",
        "name": "Promo",
        "display_name": "Promo — Nam — ToolRecap Local",
        "gender": "Male",
        "language": "en-US",
        "description": "Energetic / Trailer",
        "style": "energetic",
        "instruct": "male, middle-aged, low pitch",
    },
    "voicestudio.en.librarian": {
        "voice_id": "voicestudio.en.librarian",
        "name": "Librarian",
        "display_name": "Librarian — Nữ — ToolRecap Local",
        "gender": "Female",
        "language": "en-GB",
        "description": "Storytelling / Drama",
        "style": "storytelling",
        "instruct": "female, middle-aged, low pitch, british accent",
    },
    "voicestudio.en.podcaster": {
        "voice_id": "voicestudio.en.podcaster",
        "name": "Podcaster",
        "display_name": "Podcaster — Nữ — ToolRecap Local",
        "gender": "Female",
        "language": "en-GB",
        "description": "Narrative / Conversation",
        "style": "storytelling",
        "instruct": "female, young adult, high pitch, australian accent",
    },
    "voicestudio.en.luxe": {
        "voice_id": "voicestudio.en.luxe",
        "name": "Luxe",
        "display_name": "Luxe — Nữ — ToolRecap Local",
        "gender": "Female",
        "language": "en-GB",
        "description": "Emotional / Drama",
        "style": "drama",
        "instruct": "female, middle-aged, moderate pitch, british accent",
    },
    "voicestudio.en.storyteller": {
        "voice_id": "voicestudio.en.storyteller",
        "name": "Storyteller",
        "display_name": "Storyteller — Nam — ToolRecap Local",
        "gender": "Male",
        "language": "en-GB",
        "description": "Storytelling / Drama",
        "style": "storytelling",
        "instruct": "male, elderly, low pitch, british accent",
    },
    "voicestudio.en.commentator": {
        "voice_id": "voicestudio.en.commentator",
        "name": "Commentator",
        "display_name": "Commentator — Nam — ToolRecap Local",
        "gender": "Male",
        "language": "en-GB",
        "description": "Commentary / Recap (Default UK)",
        "style": "storytelling",
        "instruct": "male, middle-aged, high pitch, british accent",
    },
    "voicestudio.en.explainer": {
        "voice_id": "voicestudio.en.explainer",
        "name": "Explainer",
        "display_name": "Explainer — Nam — ToolRecap Local",
        "gender": "Male",
        "language": "en-GB",
        "description": "Documentary / Explainer",
        "style": "documentary",
        "instruct": "male, young adult, moderate pitch, british accent",
    },
}

DEFAULT_V2_VOICE_ID = "voicestudio.en.documentarian"


def get_v2_voice_presets() -> Dict[str, Dict[str, Any]]:
    """Return immutable copy of supported ToolRecap V2 voice presets."""
    return dict(V2_VOICE_PRESETS)


def get_voice_display_name(voice_id: str) -> str:
    """Return human-readable display name for a voice ID, or the voice ID itself if unmapped."""
    if voice_id in V2_VOICE_PRESETS:
        return V2_VOICE_PRESETS[voice_id].get("display_name", voice_id)
    return voice_id


def resolve_voice_id(display_or_id: str) -> str:
    """Resolve human-readable display name or name back to stored voice ID."""
    if not display_or_id:
        return ""
    if display_or_id in V2_VOICE_PRESETS:
        return display_or_id
    for vid, p in V2_VOICE_PRESETS.items():
        if p.get("display_name") == display_or_id or p.get("name") == display_or_id:
            return vid
    return display_or_id


def sanitize_message(msg: str, secrets: Optional[List[Optional[str]]] = None) -> str:
    """Redact secrets from error messages to prevent credential leakage."""
    if not secrets:
        return msg
    sanitized = msg
    for secret in secrets:
        if secret and len(secret) >= 4 and secret in sanitized:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    return sanitized


def validate_wav_bytes(data: bytes) -> float:
    """Validate WAV audio bytes and return duration in seconds.
    
    Raises InvalidAudioError if:
    - Data is empty or not a valid RIFF/WAVE container
    - Channel count <= 0
    - Framerate <= 0
    - Total frames <= 0
    - Duration <= 0
    """
    if not data or len(data) < 44:
        raise InvalidAudioError(f"Audio payload too small to be a valid WAV ({len(data)} bytes).")

    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels = w.getnchannels()
            framerate = w.getframerate()
            nframes = w.getnframes()

            if channels <= 0:
                raise InvalidAudioError(f"WAV has invalid channel count: {channels}")
            if framerate <= 0:
                raise InvalidAudioError(f"WAV has invalid framerate: {framerate}")
            if nframes <= 0:
                raise InvalidAudioError(f"WAV has zero frames: {nframes}")

            duration = nframes / float(framerate)
            if duration <= 0.0:
                raise InvalidAudioError(f"WAV has invalid duration: {duration:.3f}s")
            return duration
    except wave.Error as e:
        raise InvalidAudioError(f"Failed to parse WAV header: {e}") from e
    except Exception as e:
        if isinstance(e, InvalidAudioError):
            raise
        raise InvalidAudioError(f"Unexpected error validating WAV: {e}") from e


class VoiceStudioAdapter:
    """Unified adapter for VoiceStudio synthesis across Local, Remote, and Auto modes."""

    def __init__(
        self,
        mode: str = "auto",
        local_url: str = "http://127.0.0.1:3900",
        remote_url: str = "",
        remote_api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        client: Optional[httpx.Client] = None,
    ) -> None:
        mode_lower = mode.strip().lower()
        if mode_lower not in {"auto", "local", "remote"}:
            raise ValueError(f"Invalid VoiceStudio mode '{mode}'. Must be 'auto', 'local', or 'remote'.")

        self.mode = mode_lower
        self.local_url = normalize_http_endpoint(local_url, "VoiceStudio Local URL")
        self.remote_url = normalize_http_endpoint(
            remote_url, "VoiceStudio Remote URL", allow_empty=True,
        )
        self.remote_api_key = remote_api_key
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff_factor = backoff_factor
        self._external_client = client

    def close(self) -> None:
        """Close external client if owned or provided."""
        if self._external_client is not None:
            self._external_client.close()

    def __enter__(self) -> "VoiceStudioAdapter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _get_client(self, target_url: str | None = None) -> httpx.Client:
        if self._external_client is not None:
            return self._external_client
        endpoint = target_url or self.local_url
        return httpx.Client(
            timeout=self.timeout,
            trust_env=use_environment_proxy(endpoint),
        )

    def _sanitize(self, msg: str) -> str:
        return sanitize_message(msg, [self.remote_api_key])

    def _headers_for_url(self, url: str) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        # Auth is separate: apply the remote key only to the exact configured
        # remote origin.  An empty remote URL must never match every URL, and
        # string-prefix matching must not trust lookalike hosts.
        from urllib.parse import urlsplit

        def _origin(value: str) -> tuple[str, str, int | None] | None:
            if not value.strip():
                return None
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return None
            default_port = 443 if parsed.scheme == "https" else 80
            return parsed.scheme, parsed.hostname.casefold(), parsed.port or default_port

        if self.remote_api_key and _origin(self.remote_url) is not None and _origin(url) == _origin(self.remote_url):
            headers["Authorization"] = f"Bearer {self.remote_api_key}"
        return headers

    def check_health(self, target_url: Optional[str] = None) -> Dict[str, Any]:
        """Check /health on target URL or default resolved endpoint."""
        if target_url is None:
            if self.mode == "remote":
                target_url = self.remote_url
            elif self.mode == "auto":
                target_url, _ = self._resolve_target()
            else:
                target_url = self.local_url
        url = target_url.rstrip("/") + "/health"
        headers = self._headers_for_url(url)
        owned = False
        if self._external_client is not None:
            client = self._external_client
        else:
            client = httpx.Client(
                timeout=min(5.0, self.timeout),
                trust_env=use_environment_proxy(target_url),
            )
            owned = True

        try:
            resp = client.get(url, headers=headers, timeout=min(5.0, self.timeout))
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    return {"status": "ok"}
            raise VoiceStudioError(
                self._sanitize(f"Health check failed at {url} with status {resp.status_code}: {resp.text[:200]}")
            )
        except Exception as e:
            if isinstance(e, VoiceStudioError):
                raise
            clean_err = self._sanitize(str(e))
            raise VoiceStudioError(f"Health check failed for {url}: {clean_err}") from e
        finally:
            if owned:
                client.close()

    def get_voices(self, target_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve available voices from /v1/audio/voices."""
        base = target_url or self._resolve_target()[0]
        url = base.rstrip("/") + "/v1/audio/voices"
        headers = self._headers_for_url(url)
        owned = False
        if self._external_client is not None:
            client = self._external_client
        else:
            client = httpx.Client(
                timeout=self.timeout,
                trust_env=use_environment_proxy(base),
            )
            owned = True

        try:
            resp = client.get(url, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                raise VoiceStudioError(
                    self._sanitize(f"Failed to fetch voices from {url}: status {resp.status_code}")
                )
            data = resp.json()
            if isinstance(data, dict) and "voices" in data:
                return data["voices"]
            elif isinstance(data, list):
                return data
            return []
        except Exception as e:
            if isinstance(e, VoiceStudioError):
                raise
            clean_err = self._sanitize(str(e))
            raise VoiceStudioError(f"Failed to get voices from {url}: {clean_err}") from e
        finally:
            if owned:
                client.close()

    def _resolve_target(self) -> Tuple[str, bool]:
        """Resolve target base URL based on mode.
        
        Auto mode: checks Local first; if unavailable, checks Remote.
        Returns (base_url, is_remote).
        Raises VoiceStudioUnavailableError if no target is available.
        """
        if self.mode == "local":
            try:
                self.check_health(self.local_url)
                return self.local_url, False
            except Exception as e:
                raise VoiceStudioUnavailableError(
                    f"Local VoiceStudio unavailable at {self.local_url}: {e}"
                ) from e

        if self.mode == "remote":
            try:
                self.check_health(self.remote_url)
                return self.remote_url, True
            except Exception as e:
                raise VoiceStudioUnavailableError(
                    f"Remote VoiceStudio unavailable at {self.remote_url}: {self._sanitize(str(e))}"
                    f"{tailscale_connection_hint(self.remote_url)}"
                ) from e

        # Auto mode: try Local first, then Remote
        try:
            self.check_health(self.local_url)
            return self.local_url, False
        except Exception:
            pass

        try:
            self.check_health(self.remote_url)
            return self.remote_url, True
        except Exception as e:
            raise VoiceStudioUnavailableError(
                f"VoiceStudio unavailable: neither Local ({self.local_url}) nor Remote ({self.remote_url}) is reachable."
                f"{tailscale_connection_hint(self.remote_url)}"
            ) from e

    def synthesize(
        self,
        input_text: str,
        voice: str = "alloy",
        model: str = "omnivoice",
        language: str = "en",
        response_format: str = "wav",
        style: Optional[str] = None,
        instruct: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> bytes:
        """Synthesize text into WAV audio using the resolved VoiceStudio endpoint.
        
        Invariants:
        - Auto resolution happens ONCE before synthesis. No engine or target fallback during/after.
        - Exact V2 voice presets map to real matching profile/instruct definitions, never rename to alloy.
        - Unavailable legacy/old voices raise VoiceStudioUnavailableError (exact blocker, no fake aliases).
        - Validates returned bytes as valid WAV with non-zero duration. Rejects invalid audio.
        - Retries only transient errors with bounded count.
        - Cancellation immediately aborts request and raises CancelledError.
        - Sensitive credentials never leak into exceptions.
        """
        if not input_text or not input_text.strip():
            raise ValueError("Input text for synthesis cannot be empty.")

        if cancellation_token:
            cancellation_token.check_cancelled()

        # Resolve voice preset mapping (No silent Piper mapping; exact V2 presets or error)
        preset = V2_VOICE_PRESETS.get(voice)

        target_voice = voice
        target_instruct = instruct
        target_description = style

        if preset:
            # Map exact V2 preset definition to real instruct/description
            target_instruct = instruct or preset["instruct"]
            target_description = style or preset["description"]
            # For VoiceStudio server with OmniVoice engine, passing voice="default"
            # with instruct invokes the exact archetype instruction.
            target_voice = "default"
        elif voice.startswith(("voicestudio.", "piper.")):
            raise VoiceStudioUnavailableError(
                f"Requested voice '{voice}' is unavailable in VoiceStudio or V2 presets."
            )

        # Step 1: Resolve endpoint strictly before synthesis
        target_url, is_remote = self._resolve_target()
        speech_url = f"{target_url}/v1/audio/speech"
        headers = self._headers_for_url(speech_url)
        payload: Dict[str, Any] = {
            "model": model,
            "input": input_text,
            "voice": target_voice,
            "response_format": response_format,
            "language": language,
        }
        if target_instruct:
            payload["instruct"] = target_instruct
        if target_description:
            payload["description"] = target_description

        # Step 2: Bounded retry loop for transient errors
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            if cancellation_token:
                cancellation_token.check_cancelled()

            client = self._get_client(target_url)

            try:
                # Cancel-aware streaming request
                with client.stream("POST", speech_url, json=payload, headers=headers, timeout=self.timeout) as resp:
                    if resp.status_code != 200:
                        body = resp.read().decode("utf-8", errors="replace")[:300]
                        clean_body = self._sanitize(body)
                        # Distinguish transient vs logical errors
                        if resp.status_code in TRANSIENT_STATUS_CODES:
                            raise httpx.HTTPStatusError(
                                f"Transient HTTP {resp.status_code}: {clean_body}",
                                request=resp.request,
                                response=resp,
                            )
                        # Logical error: fail immediately, do not retry
                        raise VoiceStudioError(
                            f"VoiceStudio synthesis rejected (HTTP {resp.status_code}): {clean_body}"
                        )

                    # Stream bytes while checking cancellation
                    chunks: List[bytes] = []
                    for chunk in resp.iter_bytes():
                        if cancellation_token and cancellation_token.is_cancelled:
                            resp.close()
                            raise CancelledError("VoiceStudio synthesis cancelled during stream.")
                        if chunk:
                            chunks.append(chunk)

                    wav_bytes = b"".join(chunks)

                # Step 3: Validate WAV format and duration
                validate_wav_bytes(wav_bytes)
                return wav_bytes

            except CancelledError:
                raise
            except VoiceStudioError as e:
                # Logical error, do not retry
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                # Transient error: retry if attempts remain
                last_exception = e
                clean_err = self._sanitize(str(e))
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    if cancellation_token:
                        # Sleep in small increments to remain cancel-aware
                        end_time = time.monotonic() + sleep_time
                        while time.monotonic() < end_time:
                            cancellation_token.check_cancelled()
                            time.sleep(0.05)
                    else:
                        time.sleep(sleep_time)
                    continue
                else:
                    raise VoiceStudioUnavailableError(
                        f"VoiceStudio synthesis failed after {self.max_retries + 1} attempts: {clean_err}"
                    ) from e
            except Exception as e:
                # Catch any unexpected failure, sanitize and re-raise without credential leakage
                clean_err = self._sanitize(str(e))
                raise VoiceStudioError(f"Unexpected error during synthesis: {clean_err}") from e

        if last_exception:
            raise VoiceStudioUnavailableError(
                f"VoiceStudio synthesis failed after retries: {self._sanitize(str(last_exception))}"
            )
        raise VoiceStudioUnavailableError("VoiceStudio synthesis failed: retries exhausted.")
