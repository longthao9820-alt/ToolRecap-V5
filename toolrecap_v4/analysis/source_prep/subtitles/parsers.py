"""Direct subtitle parsers for SRT, ASS/SSA, and WebVTT with tag normalization and strict ms validation."""
from __future__ import annotations

import html
from pathlib import Path
import re
from typing import Sequence

from .models import SubtitleCue

# Documented constant for container start jitter / minor rounding normalization (in ms)
MAX_ROUNDING_JITTER_MS: int = 50


def strip_formatting_tags(text: str) -> str:
    """Strip HTML tags, ASS override tags, and WebVTT styling tags from subtitle text."""
    if not text:
        return ""

    # Replace ASS line breaks \N, \n, and hard space \h
    cleaned = text.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")

    # Strip ASS curly brace override tags: {\an8}, {\pos(1,2)}, {\c&H...&}, etc.
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned)

    # Strip HTML / WebVTT angle bracket tags: <i>, <b>, <font...>, <v Speaker>, <c.color>, etc.
    cleaned = re.sub(r"</?[a-zA-Z0-9_\-.:]+(?:\s+[^>]*)?>", "", cleaned)

    # Unescape HTML entities
    cleaned = html.unescape(cleaned)

    # Normalize line breaks and whitespace
    lines = [line.strip() for line in cleaned.splitlines()]
    filtered_lines: list[str] = []
    prev_blank = False
    for line in lines:
        if line:
            filtered_lines.append(line)
            prev_blank = False
        elif not prev_blank:
            filtered_lines.append("")
            prev_blank = True

    return "\n".join(filtered_lines).strip()


def normalize_subtitle_text(text: str) -> str:
    """Normalize subtitle text, removing tags and trimming whitespace."""
    return strip_formatting_tags(text)


def parse_timestamp_srt(ts_str: str) -> int:
    """Parse SRT timestamp 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' to milliseconds."""
    clean = ts_str.strip().replace(",", ".")
    match = re.match(r"^(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{1,3})$", clean)
    if not match:
        raise ValueError(f"Invalid SRT timestamp format: '{ts_str}'")

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis_str = match.group(4).ljust(3, "0")[:3]
    millis = int(millis_str)
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def parse_timestamp_vtt(ts_str: str) -> int:
    """Parse WebVTT timestamp 'HH:MM:SS.mmm' or 'MM:SS.mmm' to milliseconds."""
    clean = ts_str.strip()
    match = re.match(r"^(?:(?:(\d+):)?(\d{2}):)?(\d{2})[.,](\d{1,3})$", clean)
    if not match:
        raise ValueError(f"Invalid WebVTT timestamp format: '{ts_str}'")

    h_str, m_str, s_str, ms_str = match.groups()
    hours = int(h_str) if h_str is not None else 0
    minutes = int(m_str) if m_str is not None else 0
    seconds = int(s_str)
    millis = int(ms_str.ljust(3, "0")[:3])
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def parse_timestamp_ass(ts_str: str) -> int:
    """Parse ASS/SSA timestamp 'H:MM:SS.cc' (centiseconds) to milliseconds."""
    clean = ts_str.strip()
    match = re.match(r"^(\d+):(\d{2}):(\d{2})[.,](\d{1,3})$", clean)
    if not match:
        raise ValueError(f"Invalid ASS timestamp format: '{ts_str}'")

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    cs_str = match.group(4)
    if len(cs_str) == 2:
        millis = int(cs_str) * 10
    elif len(cs_str) == 1:
        millis = int(cs_str) * 100
    else:
        millis = int(cs_str[:3].ljust(3, "0"))
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def _sanitize_cue_timing(
    start_ms: int,
    end_ms: int,
    *,
    source_duration_ms: int | None = None,
) -> tuple[int, int] | None:
    """Validate and normalize cue timestamps.
    
    Invariants:
    - Rejects boolean values.
    - Rejects cues exceeding source_duration_ms without silent clamp.
    - Minor negative start timestamps within MAX_ROUNDING_JITTER_MS are normalized to 0.
    - Materially negative timestamps (< -MAX_ROUNDING_JITTER_MS) are rejected.
    - Non-positive durations (end_ms <= start_ms) are rejected as materially invalid.
    - Strict ms: no material clamp to arbitrary bounds.
    """
    if type(start_ms) is bool or type(end_ms) is bool:
        return None

    if source_duration_ms is not None and end_ms > source_duration_ms:
        return None

    if start_ms < 0:
        if start_ms >= -MAX_ROUNDING_JITTER_MS:
            start_ms = 0
        else:
            return None

    if end_ms <= start_ms:
        return None

    return start_ms, end_ms


def parse_srt(
    content_or_path: str | Path,
    *,
    source_type: str = "sidecar",
    source_format: str = "srt",
    stream_index: int | None = None,
    source_file: str | None = None,
    language: str = "und",
    episode_id: str = "",
    source_video: str = "",
    source_duration_ms: int | None = None,
) -> list[SubtitleCue]:
    """Parse SubRip (.srt) subtitle content or file into normalized SubtitleCue items."""
    if isinstance(content_or_path, Path) or (
        isinstance(content_or_path, str) and "\n" not in content_or_path and Path(content_or_path).is_file()
    ):
        p = Path(content_or_path)
        source_file = source_file or str(p.resolve())
        content = p.read_text(encoding="utf-8-sig", errors="replace")
    else:
        content = str(content_or_path)

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", content.strip())
    cues: list[SubtitleCue] = []

    time_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})"
    )

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        time_line_idx = -1
        time_match = None
        for idx, line in enumerate(lines):
            m = time_pattern.search(line)
            if m:
                time_line_idx = idx
                time_match = m
                break

        if not time_match or time_line_idx == -1:
            continue

        start_str, end_str = time_match.group(1), time_match.group(2)
        try:
            start_ms = parse_timestamp_srt(start_str)
            end_ms = parse_timestamp_srt(end_str)
        except ValueError:
            continue

        timing = _sanitize_cue_timing(start_ms, end_ms, source_duration_ms=source_duration_ms)
        if timing is None:
            continue
        start_ms, end_ms = timing

        raw_text = "\n".join(lines[time_line_idx + 1:])
        clean_text = normalize_subtitle_text(raw_text)
        if not clean_text:
            continue

        cues.append(
            SubtitleCue(
                start_ms=start_ms,
                end_ms=end_ms,
                text=clean_text,
                source_type=source_type,
                source_format=source_format,
                stream_index=stream_index,
                source_file=source_file,
                language=language,
                confidence=1.0,
                episode_id=episode_id,
                source_video=source_video,
            )
        )

    return cues


def parse_vtt(
    content_or_path: str | Path,
    *,
    source_type: str = "sidecar",
    source_format: str = "vtt",
    stream_index: int | None = None,
    source_file: str | None = None,
    language: str = "und",
    episode_id: str = "",
    source_video: str = "",
    source_duration_ms: int | None = None,
) -> list[SubtitleCue]:
    """Parse WebVTT (.vtt) subtitle content or file into normalized SubtitleCue items."""
    if isinstance(content_or_path, Path) or (
        isinstance(content_or_path, str) and "\n" not in content_or_path and Path(content_or_path).is_file()
    ):
        p = Path(content_or_path)
        source_file = source_file or str(p.resolve())
        content = p.read_text(encoding="utf-8-sig", errors="replace")
    else:
        content = str(content_or_path)

    content = content.replace("\r\n", "\n").replace("\r", "\n")

    if content.startswith("WEBVTT"):
        parts = re.split(r"\n\s*\n+", content, maxsplit=1)
        content = parts[1] if len(parts) > 1 else ""

    blocks = re.split(r"\n\s*\n+", content.strip())
    cues: list[SubtitleCue] = []

    time_pattern = re.compile(
        r"((?:\d{1,2}:)?\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*((?:\d{1,2}:)?\d{2}:\d{2}[,\.]\d{1,3})"
    )

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if lines[0].startswith("NOTE"):
            continue

        time_line_idx = -1
        time_match = None
        for idx, line in enumerate(lines):
            m = time_pattern.search(line)
            if m:
                time_line_idx = idx
                time_match = m
                break

        if not time_match or time_line_idx == -1:
            continue

        start_str, end_str = time_match.group(1), time_match.group(2)
        try:
            start_ms = parse_timestamp_vtt(start_str)
            end_ms = parse_timestamp_vtt(end_str)
        except ValueError:
            continue

        timing = _sanitize_cue_timing(start_ms, end_ms, source_duration_ms=source_duration_ms)
        if timing is None:
            continue
        start_ms, end_ms = timing

        raw_text = "\n".join(lines[time_line_idx + 1:])
        clean_text = normalize_subtitle_text(raw_text)
        if not clean_text:
            continue

        cues.append(
            SubtitleCue(
                start_ms=start_ms,
                end_ms=end_ms,
                text=clean_text,
                source_type=source_type,
                source_format=source_format,
                stream_index=stream_index,
                source_file=source_file,
                language=language,
                confidence=1.0,
                episode_id=episode_id,
                source_video=source_video,
            )
        )

    return cues


def parse_ass(
    content_or_path: str | Path,
    *,
    source_type: str = "sidecar",
    source_format: str = "ass",
    stream_index: int | None = None,
    source_file: str | None = None,
    language: str = "und",
    episode_id: str = "",
    source_video: str = "",
    source_duration_ms: int | None = None,
) -> list[SubtitleCue]:
    """Parse Advanced SubStation Alpha (.ass / .ssa) content or file into normalized SubtitleCue items."""
    if isinstance(content_or_path, Path) or (
        isinstance(content_or_path, str) and "\n" not in content_or_path and Path(content_or_path).is_file()
    ):
        p = Path(content_or_path)
        source_file = source_file or str(p.resolve())
        content = p.read_text(encoding="utf-8-sig", errors="replace")
    else:
        content = str(content_or_path)

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.splitlines()

    in_events = False
    format_fields: list[str] = []
    cues: list[SubtitleCue] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        if stripped.lower().startswith("[events]"):
            in_events = True
            continue
        elif stripped.startswith("[") and in_events:
            break

        if not in_events:
            continue

        if stripped.lower().startswith("format:"):
            header_vals = stripped[7:].strip().split(",")
            format_fields = [h.strip().lower() for h in header_vals]
            continue

        if stripped.lower().startswith("dialogue:"):
            payload = stripped[9:].strip()
            if not format_fields:
                format_fields = [
                    "layer", "start", "end", "style", "name",
                    "marginl", "marginr", "marginv", "effect", "text"
                ]

            parts = payload.split(",", len(format_fields) - 1)
            if len(parts) < len(format_fields):
                continue

            field_map = dict(zip(format_fields, parts))
            start_str = field_map.get("start", "").strip()
            end_str = field_map.get("end", "").strip()
            text_str = field_map.get("text", "")

            try:
                start_ms = parse_timestamp_ass(start_str)
                end_ms = parse_timestamp_ass(end_str)
            except ValueError:
                continue

            timing = _sanitize_cue_timing(start_ms, end_ms, source_duration_ms=source_duration_ms)
            if timing is None:
                continue
            start_ms, end_ms = timing

            clean_text = normalize_subtitle_text(text_str)
            if not clean_text:
                continue

            cues.append(
                SubtitleCue(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=clean_text,
                    source_type=source_type,
                    source_format=source_format,
                    stream_index=stream_index,
                    source_file=source_file,
                    language=language,
                    confidence=1.0,
                    episode_id=episode_id,
                    source_video=source_video,
                )
            )

    return cues
