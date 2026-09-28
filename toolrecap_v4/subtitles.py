"""Subtitle generation, timing offsets, and SRT file handling for ToolRecap V4.

Invariants:
- Cues within segments are relative to segment start (0 <= start_ms < end_ms <= segment_duration).
- Timeline SRT cues are offset by accumulated segment durations.
- Generates exactly two SRT files per output:
    - {title}.narration.srt (narration cues)
    - {title}.original.srt (original dialogue cues)
- Preserves UTF-8 encoding without BOM.
- Path escaping for FFmpeg subtitles filter on Windows and POSIX.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Sequence


@dataclass(frozen=True)
class SubtitleCue:
    """Individual subtitle cue with timeline timestamps in milliseconds."""

    start_ms: int
    end_ms: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleCue:
        return cls(
            start_ms=int(data["start_ms"]),
            end_ms=int(data["end_ms"]),
            text=str(data.get("text", "")).strip(),
        )


def format_srt_time(ms: int) -> str:
    """Format milliseconds integer to SRT timestamp string HH:MM:SS,mmm."""
    if ms < 0:
        ms = 0
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_srt_time(ts_str: str) -> int:
    """Parse SRT timestamp string HH:MM:SS,mmm (or .mmm) to milliseconds integer."""
    clean = ts_str.strip().replace(".", ",")
    parts = clean.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid SRT timestamp format: '{ts_str}'")
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split(",")
    seconds = int(sec_parts[0])
    millis = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def build_srt(cues: Sequence[SubtitleCue | dict[str, Any]]) -> str:
    """Build standardized SRT file content from ordered cues."""
    if not cues:
        return ""

    blocks: list[str] = []
    for idx, cue in enumerate(cues, start=1):
        if isinstance(cue, dict):
            start_ms = int(cue["start_ms"])
            end_ms = int(cue["end_ms"])
            text = str(cue.get("text", "")).strip()
        else:
            start_ms = cue.start_ms
            end_ms = cue.end_ms
            text = cue.text.strip()

        if end_ms <= start_ms or not text:
            continue

        start_str = format_srt_time(start_ms)
        end_str = format_srt_time(end_ms)
        blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")

    return "\n".join(blocks)


def parse_srt(content: str) -> list[SubtitleCue]:
    """Parse SRT formatted text into a list of SubtitleCue objects."""
    if not content or not content.strip():
        return []

    cues: list[SubtitleCue] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        # Look for the line with the timestamp separator '-->'
        time_line_idx = -1
        for idx, line in enumerate(lines):
            if "-->" in line:
                time_line_idx = idx
                break

        if time_line_idx == -1:
            continue

        time_line = lines[time_line_idx]
        start_str, _, end_str = time_line.partition("-->")
        try:
            start_ms = parse_srt_time(start_str)
            # End string may have coordinates or extra tags; take first token
            end_token = end_str.strip().split()[0]
            end_ms = parse_srt_time(end_token)
        except ValueError:
            continue

        text_lines = lines[time_line_idx + 1 :]
        text = "\n".join(text_lines)
        if text:
            cues.append(SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text))

    return cues


def write_srt(
    path: Path | str,
    cues: Sequence[SubtitleCue | dict[str, Any]],
) -> Path:
    """Write SRT file atomically with UTF-8 encoding (no BOM)."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    content = build_srt(cues)
    tmp = p.with_suffix(f"{p.suffix}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)
    return p


def extract_subtitles_for_output(
    output_def: dict[str, Any],
) -> tuple[list[SubtitleCue], list[SubtitleCue]]:
    """Extract timeline-offset narration and original dialogue cues for an output.
    
    Returns:
        (narration_cues, original_dialogue_cues)
    """
    narration_cues: list[SubtitleCue] = []
    original_cues: list[SubtitleCue] = []

    timeline_offset_ms = 0

    segments = output_def.get("segments", [])
    from toolrecap_v4.narration_groups import narration_groups
    group_lengths = {
        name: sum(member["end_ms"] - member["start_ms"] for member in members)
        for name, members in narration_groups(segments).items()
    }
    for seg in segments:
        start_ms = int(seg.get("start_ms", 0))
        end_ms = int(seg.get("end_ms", 0))
        seg_duration_ms = max(0, end_ms - start_ms)
        seg_type = seg.get("type", "narration")
        subtitles = seg.get("subtitles", [])
        narration_text = str(seg.get("narration", "")).strip()

        if seg_type == "narration":
            if subtitles:
                for cue in subtitles:
                    c_start = int(cue.get("start_ms", 0))
                    c_end = int(cue.get("end_ms", 0))
                    c_text = str(cue.get("text", "")).strip()
                    if c_text and c_end > c_start:
                        narration_cues.append(
                            SubtitleCue(
                                start_ms=timeline_offset_ms + c_start,
                                end_ms=timeline_offset_ms + c_end,
                                text=c_text,
                            )
                        )
            elif narration_text:
                # Synthesize cue if no explicit cues provided
                narr_dur = seg.get("narration_duration_ms")
                if seg.get("narration_group"):
                    group = seg["narration_group"]
                    dur = min(group_lengths[group], output_def.get("_narration_group_durations", {}).get(group, group_lengths[group]))
                else:
                    dur = min(seg_duration_ms, narr_dur) if narr_dur is not None else seg_duration_ms
                if dur > 0:
                    narration_cues.append(
                        SubtitleCue(
                            start_ms=timeline_offset_ms,
                            end_ms=timeline_offset_ms + dur,
                            text=narration_text,
                        )
                    )
        elif seg_type == "original_dialogue":
            if subtitles:
                for cue in subtitles:
                    c_start = int(cue.get("start_ms", 0))
                    c_end = int(cue.get("end_ms", 0))
                    c_text = str(cue.get("text", "")).strip()
                    if c_text and c_end > c_start:
                        original_cues.append(
                            SubtitleCue(
                                start_ms=timeline_offset_ms + c_start,
                                end_ms=timeline_offset_ms + c_end,
                                text=c_text,
                            )
                        )

        timeline_offset_ms += seg_duration_ms

    return narration_cues, original_cues


def generate_subtitles(
    output_def: dict[str, Any],
    output_dir: Path | str,
    title: str | None = None,
    *,
    original_cues: Sequence[SubtitleCue | dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Generate both {title}.narration.srt and {title}.original.srt in output_dir.
    
    Both files are guaranteed to be created.
    Returns (narration_srt_path, original_srt_path).
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t = title or output_def.get("title") or output_def.get("render_id") or "output"
    narr_cues, legacy_orig_cues = extract_subtitles_for_output(output_def)
    orig_cues = list(original_cues) if original_cues is not None else legacy_orig_cues

    narr_path = out_dir / f"{t}.narration.srt"
    orig_path = out_dir / f"{t}.original.srt"

    write_srt(narr_path, narr_cues)
    write_srt(orig_path, orig_cues)

    return narr_path, orig_path


def escape_ffmpeg_subtitles_path(path: Path | str) -> str:
    """Escape filesystem path for FFmpeg's subtitles filter.
    
    On Windows:
    - Normalizes backslashes to forward slashes
    - Escapes colon after drive letter: C:/path -> C\\:/path
    - Escapes single quotes
    """
    p_str = Path(path).resolve().as_posix()
    # Replace drive colon C: with C\:
    if len(p_str) >= 2 and p_str[1] == ":" and p_str[0].isalpha():
        p_str = p_str[0] + r"\:" + p_str[2:]
    # Escape single quotes and colons inside path if any
    p_str = p_str.replace("'", r"\'")
    return p_str
