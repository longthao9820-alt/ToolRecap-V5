"""Original-audio multi-clip Highlight renderer."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.original_dialogue import NO_ORIGINAL_DIALOGUE
from toolrecap_v4.renderer import RenderResult, render_output
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.subtitles import SubtitleCue
from .models import HighlightOutput


@dataclass(frozen=True)
class HighlightRenderResult:
    output_id: str
    video_path: Path
    subtitle_path: Path
    video_sha256: str
    subtitle_sha256: str
    original_subtitle_state: str = NO_ORIGINAL_DIALOGUE
    original_subtitle_cue_count: int = 0


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _publish_pair(files: tuple[tuple[Path, Path], ...]) -> None:
    """Publish MP4/SRT as a recoverable pair."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for _source, destination in files:
            if destination.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.", suffix=".backup", dir=destination.parent,
                )
                os.close(descriptor)
                backup = Path(backup_name)
                shutil.copy2(destination, backup)
                backups[destination] = backup
        for source, destination in files:
            descriptor, staging_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
            )
            os.close(descriptor)
            staging = Path(staging_name)
            try:
                shutil.copy2(source, staging)
                os.replace(staging, destination)
                published.append(destination)
            finally:
                staging.unlink(missing_ok=True)
    except Exception:
        for destination in reversed(published):
            backup = backups.get(destination)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def render_highlight(
    output: HighlightOutput,
    *,
    sources: Mapping[str, Path | str],
    publication_dir: Path | str,
    ffmpeg_path: Path | str | None = None,
    settings: AppSettings | None = None,
    cancellation_token: CancellationToken | None = None,
    allow_owned_overwrite: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    render_output_fn: Callable[..., RenderResult] = render_output,
    publication_owner: str | None = None,
    publication_journal_root: Path | None = None,
    audio_stream_indices: Mapping[str, int] | None = None,
    **_legacy_ignored: Any,
) -> HighlightRenderResult:
    """Render one coherent story from one or more chronological source clips.

    Every clip retains original source audio. No narration or VoiceStudio path
    is used. The shared normalized renderer makes clips from different episodes
    compatible before concatenation and produces frame-stable CFR output.
    """
    if not output.original_audio:
        raise ValueError("Highlight outputs must preserve original source audio")
    if not output.clips:
        raise ValueError("Highlight output requires at least one source clip")

    resolved_sources: dict[str, Path] = {}
    for clip in output.clips:
        if clip.source_file not in sources:
            raise ValueError(f"Missing Highlight source: {clip.source_file}")
        source = Path(sources[clip.source_file]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        resolved_sources[clip.source_file] = source

    target_dir = Path(publication_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    video = target_dir / f"{output.title}.mp4"
    subtitle = target_dir / f"{output.title}.srt"
    if any(source in (video.resolve(), subtitle.resolve()) for source in resolved_sources.values()):
        raise ValueError("Highlight publication collides with its source")
    if not allow_owned_overwrite:
        existing = next((path for path in (video, subtitle) if path.exists()), None)
        if existing is not None:
            raise FileExistsError(
                f"Highlight publication target already exists without a matching checkpoint: {existing}"
            )
    if cancellation_token:
        cancellation_token.check_cancelled()

    source_dialogue: dict[str, tuple[SubtitleCue, ...]] = {}
    for source_file in resolved_sources:
        unique: dict[tuple[int, int, str], SubtitleCue] = {}
        for clip in output.clips:
            if clip.source_file != source_file:
                continue
            for cue in clip.subtitle_cues:
                unique[(cue.start_ms, cue.end_ms, cue.text)] = SubtitleCue(
                    cue.start_ms, cue.end_ms, cue.text,
                )
        source_dialogue[source_file] = tuple(
            unique[key] for key in sorted(unique, key=lambda item: (item[0], item[1], item[2]))
        )

    output_def = {
        "render_id": output.output_id,
        "title": output.title,
        "segments": [{
            "segment_id": clip.clip_id,
            "source_file": clip.source_file,
            "start_ms": clip.start_ms,
            "end_ms": clip.end_ms,
            "type": "original_dialogue",
            "narration": "",
            "source_audio": True,
            "audio_intent": "ORIGINAL_DIALOGUE",
            "subtitles": [],
        } for clip in output.clips],
    }
    cfg = replace(settings or AppSettings(), auto_duck=False)
    with tempfile.TemporaryDirectory(prefix="toolrecap_highlight_story_", dir=target_dir) as temp_name:
        temporary_publication = Path(temp_name)
        rendered = render_output_fn(
            output_def=output_def,
            source_paths=resolved_sources,
            output_dir=temporary_publication,
            settings=cfg,
            voice_adapter=None,
            ffmpeg_path=ffmpeg_path,
            source_dialogue_map=source_dialogue,
            audio_stream_indices=dict(audio_stream_indices) if audio_stream_indices is not None else None,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
        if cancellation_token:
            cancellation_token.check_cancelled()
        if publication_owner:
            from toolrecap_v4.publication import publish
            publish([(rendered.output_path, video), (rendered.original_srt_path, subtitle)],
                    publication_owner, cancellation_token, publication_journal_root,
                    allow_existing=allow_owned_overwrite)
        else:
            _publish_pair(((rendered.output_path, video), (rendered.original_srt_path, subtitle)))

    cue_count = sum(len(clip.subtitle_cues) for clip in output.clips)
    return HighlightRenderResult(
        output.output_id,
        video,
        subtitle,
        _sha(video),
        _sha(subtitle),
        rendered.original_subtitle_state,
        cue_count,
    )
