"""Generate short local previews around a selected source cut boundary."""
from __future__ import annotations

import hashlib
from pathlib import Path

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.media import find_binary, run_command


def boundary_preview(source: Path, start_ms: int, end_ms: int, *, at_end: bool,
                     storage_root: Path, source_hash: str = "",
                     cancellation_token: CancellationToken | None = None) -> Path:
    if start_ms < 0 or end_ms <= start_ms or not source.is_file():
        raise ValueError("Preview requires a real source and valid clip boundaries")
    duration_ms = min(5000, end_ms - start_ms)
    preview_start = end_ms - duration_ms if at_end else start_ms
    key = hashlib.sha256(
        f"{source.resolve()}|{source_hash}|{source.stat().st_mtime_ns}|{start_ms}|{end_ms}|{at_end}".encode()
    ).hexdigest()[:24]
    directory = Path(storage_root) / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{key}.mp4"
    if target.is_file() and target.stat().st_size > 0:
        return target
    temporary = directory / f"{key}.partial.mp4"
    try:
        run_command([
            str(find_binary("ffmpeg")), "-hide_banner", "-loglevel", "error", "-y",
            "-accurate_seek", "-ss", f"{preview_start / 1000:.3f}",
            "-i", str(source), "-t", f"{duration_ms / 1000:.3f}",
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
            "-preset", "ultrafast", "-crf", "25", "-c:a", "aac", str(temporary),
        ], timeout=120.0, cancellation_token=cancellation_token, check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("Source boundary preview is empty")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)
