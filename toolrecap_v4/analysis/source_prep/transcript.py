"""Transcript normalization, strict timestamp validation, cue sequencing, and provenance hashing."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from toolrecap_v4.analysis.models import Transcript, TranscriptCue
from toolrecap_v4.analysis.source_prep.subtitles.models import SubtitleCue
from toolrecap_v4.analysis.source_prep.subtitles.parsers import (
    MAX_ROUNDING_JITTER_MS,
    normalize_subtitle_text,
)


def validate_and_normalize_cue(
    start_ms: Any,
    end_ms: Any,
    text: str,
    *,
    source_duration_ms: int | None = None,
    max_jitter_ms: int = MAX_ROUNDING_JITTER_MS,
) -> tuple[tuple[int, int, str] | None, str | None]:
    """Validate cue boundaries and normalize minor start jitter.

    Invariants:
    - Rejects boolean and non-integer timestamp values explicitly with diagnostics.
    - Rejects cues where end_ms exceeds source_duration_ms (no silent clamping!).
    - Minor negative start jitter within max_jitter_ms is normalized to 0.
    - Materially negative timestamps (< -max_jitter_ms) are rejected with diagnostics.
    - Non-positive durations (end_ms <= start_ms) are rejected with diagnostics.
    - Formatting tags stripped; empty text rejected with diagnostics.
    """
    if type(start_ms) is bool or type(end_ms) is bool:
        return None, f"Timestamp rejected: start_ms ({start_ms!r}) or end_ms ({end_ms!r}) is boolean, not integer"

    if not isinstance(start_ms, int) or not isinstance(end_ms, int):
        return None, f"Timestamp rejected: start_ms ({start_ms!r}) or end_ms ({end_ms!r}) is not integer"

    # Reject cues extending beyond source duration (no silent clamping)
    if source_duration_ms is not None and end_ms > source_duration_ms:
        return None, (
            f"Cue rejected: end_ms ({end_ms}) exceeds source duration ({source_duration_ms} ms); "
            "material clamping forbidden"
        )

    # Minor negative start jitter normalization
    if start_ms < 0:
        if start_ms >= -max_jitter_ms:
            start_ms = 0
        else:
            return None, f"Cue rejected: start_ms ({start_ms}) materially negative (< -{max_jitter_ms} ms)"

    # Reject non-positive duration
    if end_ms <= start_ms:
        return None, f"Cue rejected: non-positive duration (end_ms {end_ms} <= start_ms {start_ms})"

    clean_text = normalize_subtitle_text(str(text or ""))
    if not clean_text:
        return None, "Cue rejected: text is empty after tag stripping"

    return (start_ms, end_ms, clean_text), None


def compute_transcript_hash(
    episode_id: str,
    source_type: str,
    source_format: str,
    cues: Sequence[TranscriptCue],
    language: str = "und",
) -> str:
    """Compute deterministic provenance hash for transcript cues and metadata."""
    payload = {
        "episode_id": episode_id,
        "source_type": source_type,
        "source_format": source_format,
        "language": language,
        "cue_count": len(cues),
        "cues": [
            [c.start_ms, c.end_ms, c.text, round(c.confidence, 4) if c.confidence is not None else None]
            for c in cues
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_transcript(
    episode_id: str,
    source_type: str,
    source_format: str,
    raw_cues: Sequence[SubtitleCue | TranscriptCue | dict[str, Any]],
    *,
    language: str = "und",
    has_speech: bool = True,
    id_prefix: str = "",
    source_duration_ms: int | None = None,
    initial_diagnostics: Sequence[str] | None = None,
) -> Transcript:
    """Build a sanitized, sequentially indexed Transcript from raw cue items.

    Invariants:
    - Drops materially invalid cues and logs diagnostic explanation.
    - Preserves exact timestamps without material clamping to arbitrary durations.
    - Rejects cues extending past source_duration_ms without silent clamp.
    - Sequentially assigns unique deterministic cue_ids.
    - Computes deterministic SHA-256 provenance hash.
    """
    prefix = id_prefix or episode_id or "EP"
    sanitized: list[tuple[int, int, str, float | None]] = []
    diagnostics: list[str] = list(initial_diagnostics or [])

    for raw in raw_cues:
        if isinstance(raw, (SubtitleCue, TranscriptCue)):
            s_ms = raw.start_ms
            e_ms = raw.end_ms
            txt = raw.text
            conf = raw.confidence
        elif isinstance(raw, dict):
            s_ms = raw.get("start_ms", 0)
            e_ms = raw.get("end_ms", 0)
            txt = str(raw.get("text", ""))
            conf = float(raw["confidence"]) if raw.get("confidence") is not None else None
        else:
            diagnostics.append(f"Invalid cue item type: {type(raw).__name__}")
            continue

        result_tuple, diag = validate_and_normalize_cue(
            s_ms,
            e_ms,
            txt,
            source_duration_ms=source_duration_ms,
        )
        if result_tuple is None:
            if diag:
                diagnostics.append(diag)
            continue

        norm_s, norm_e, clean_t = result_tuple
        sanitized.append((norm_s, norm_e, clean_t, conf))

    # Sort strictly by start_ms, then end_ms
    sanitized.sort(key=lambda item: (item[0], item[1]))

    # Construct immutable cues with sequential cue_id
    cues_out: list[TranscriptCue] = []
    for idx, (s_ms, e_ms, txt, conf) in enumerate(sanitized, start=1):
        c_id = f"{prefix}-CUE-{idx:04d}"
        cues_out.append(
            TranscriptCue(
                cue_id=c_id,
                start_ms=s_ms,
                end_ms=e_ms,
                text=txt,
                confidence=conf,
            )
        )

    prov_hash = compute_transcript_hash(
        episode_id=episode_id,
        source_type=source_type,
        source_format=source_format,
        cues=cues_out,
        language=language,
    )

    return Transcript(
        episode_id=episode_id,
        source_type=source_type,
        source_format=source_format,
        cues=tuple(cues_out),
        has_speech=has_speech if cues_out else False,
        language=language,
        provenance_hash=prov_hash,
        dropped_cues_count=len(diagnostics),
        diagnostics=tuple(diagnostics),
    )
