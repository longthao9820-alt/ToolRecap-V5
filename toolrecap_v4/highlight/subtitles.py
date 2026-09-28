"""Verified original-dialogue SRT creation for Highlight Mode."""
from __future__ import annotations

from toolrecap_v4.original_dialogue import OriginalDialogueSubtitleMapper, ResolvedSourceClip
from toolrecap_v4.subtitles import SubtitleCue, build_srt
from .models import HighlightOutput


def highlight_srt(output: HighlightOutput) -> str:
    """Map verified source-global cues through the shared authoritative mapper."""
    source_cues: dict[str, tuple[SubtitleCue, ...]] = {}
    for source_file in dict.fromkeys(clip.source_file for clip in output.clips):
        source_cues[source_file] = tuple(
            SubtitleCue(cue.start_ms, cue.end_ms, cue.text)
            for clip in output.clips if clip.source_file == source_file
            for cue in clip.subtitle_cues
        )
    timeline_offset = 0
    resolved: list[ResolvedSourceClip] = []
    for clip in output.clips:
        duration = clip.end_ms - clip.start_ms
        resolved.append(ResolvedSourceClip(
            segment_id=clip.clip_id,
            source_file=clip.source_file,
            source_start_ms=clip.start_ms,
            source_end_ms=clip.end_ms,
            final_start_ms=timeline_offset,
            final_duration_ms=duration,
            source_audio=True,
        ))
        timeline_offset += duration
    mapping = OriginalDialogueSubtitleMapper(source_cues).map(tuple(resolved))
    if mapping.retained_verified_source_cue_count and not mapping.cues:
        raise ValueError("Verified Highlight dialogue could not be mapped")
    return build_srt(mapping.cues)
