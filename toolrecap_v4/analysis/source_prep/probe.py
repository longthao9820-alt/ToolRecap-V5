"""Media probing for episode sources with video, audio, subtitle streams, and auto-canvas resolution."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from toolrecap_v4.analysis.source_prep.subtitles.models import SubtitleStreamInfo, normalize_language_code
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError
from toolrecap_v4.media import (
    AudioStreamInfo,
    MediaProbeResult,
    VideoStreamInfo,
    calculate_auto_canvas,
    find_binary,
    probe_media,
    run_command,
)


@dataclass(frozen=True)
class EpisodeProbeResult:
    """Probed episode metadata with video, audio, subtitle streams, and canvas bounds."""

    path: Path
    duration_ms: int
    container: str
    video_streams: tuple[VideoStreamInfo, ...]
    audio_streams: tuple[AudioStreamInfo, ...]
    subtitle_streams: tuple[SubtitleStreamInfo, ...]
    canvas_width: int
    canvas_height: int
    primary_video: VideoStreamInfo | None
    primary_audio: AudioStreamInfo | None
    subtitle_probe_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration_ms": self.duration_ms,
            "container": self.container,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "video_streams": [v.to_dict() for v in self.video_streams],
            "audio_streams": [a.to_dict() for a in self.audio_streams],
            "subtitle_streams": [s.to_dict() for s in self.subtitle_streams],
            "primary_video": self.primary_video.to_dict() if self.primary_video else None,
            "primary_audio": self.primary_audio.to_dict() if self.primary_audio else None,
            "subtitle_probe_error": self.subtitle_probe_error,
        }


def probe_episode_source(
    path: Path | str,
    *,
    timeout: float = 60.0,
    cancellation_token: CancellationToken | None = None,
) -> EpisodeProbeResult:
    """Probe an episode media source file for video, audio, and subtitle streams.

    Invariants:
    - Never swallows CancelledError.
    - Captures and distinguishes subtitle stream probe failures instead of reporting fake empty list silently.
    """
    if cancellation_token:
        cancellation_token.check_cancelled()

    p = Path(path).resolve()
    base_probe = probe_media(p, timeout=timeout, cancellation_token=cancellation_token)
    canvas_w, canvas_h = calculate_auto_canvas(base_probe)

    duration_ms = max(0, int(round(base_probe.duration * 1000.0)))
    primary_video = base_probe.video_streams[0] if base_probe.video_streams else None
    primary_audio = base_probe.audio_streams[0] if base_probe.audio_streams else None

    sub_streams: list[SubtitleStreamInfo] = []
    subtitle_probe_error: str | None = None

    try:
        if cancellation_token:
            cancellation_token.check_cancelled()

        binary = find_binary("ffprobe")
        cmd = [
            str(binary),
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "s",
            str(p),
        ]
        res = run_command(cmd, timeout=timeout, cancellation_token=cancellation_token)
        if res.exit_code == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            streams_raw = data.get("streams", [])
            bitmap_codecs = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvdsub", "pgs"}

            for ordinal, s in enumerate(streams_raw):
                codec = str(s.get("codec_name", "")).lower()
                tags = s.get("tags") or {}
                norm_tags = {str(k).lower(): str(v) for k, v in tags.items()}
                raw_lang = norm_tags.get("language") or norm_tags.get("lang") or ""
                lang = normalize_language_code(raw_lang)
                title = norm_tags.get("title") or ""
                disp = s.get("disposition") or {}
                default = bool(disp.get("default", 0))
                forced = bool(disp.get("forced", 0))
                is_bitmap = codec in bitmap_codecs

                sub_streams.append(
                    SubtitleStreamInfo(
                        index=int(s.get("index", ordinal)),
                        subtitle_index=ordinal,
                        codec=codec,
                        language=lang,
                        title=title,
                        default=default,
                        forced=forced,
                        is_bitmap=is_bitmap,
                    )
                )
        elif res.exit_code != 0:
            subtitle_probe_error = f"ffprobe subtitle query exit {res.exit_code}: {res.stderr.strip()}"
    except CancelledError:
        raise
    except Exception as e:
        subtitle_probe_error = f"Subtitle stream probing failed: {e}"
        sub_streams = []

    return EpisodeProbeResult(
        path=p,
        duration_ms=duration_ms,
        container=base_probe.container,
        video_streams=base_probe.video_streams,
        audio_streams=base_probe.audio_streams,
        subtitle_streams=tuple(sub_streams),
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        primary_video=primary_video,
        primary_audio=primary_audio,
        subtitle_probe_error=subtitle_probe_error,
    )
