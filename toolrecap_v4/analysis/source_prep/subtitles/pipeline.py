"""Subtitle extraction, demuxing, OCR execution, and caching pipeline for ToolRecap V4."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Sequence
import uuid

from toolrecap_v4.analysis.source_prep.subtitles.cache import (
    SubtitleCacheManager,
    compute_subtitle_cache_key,
)
from toolrecap_v4.analysis.source_prep.subtitles.discovery import select_best_english_subtitles
from toolrecap_v4.analysis.source_prep.subtitles.models import (
    PgsSubtitleEvent,
    SubtitleCue,
    SubtitleTrack,
    VobSubEvent,
)
from toolrecap_v4.analysis.source_prep.subtitles.ocr import OcrAdapter, OcrResult
from toolrecap_v4.analysis.source_prep.subtitles.parsers import parse_ass, parse_srt, parse_vtt
from toolrecap_v4.analysis.source_prep.subtitles.pgs import parse_pgs_sup
from toolrecap_v4.analysis.source_prep.subtitles.vobsub import extract_vobsub_events
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, ToolRecapError
from toolrecap_v4.media import CommandResult, find_binary, run_command


def extract_embedded_subtitle_stream(
    video_path: Path | str,
    stream_index: int,
    output_path: Path | str,
    *,
    source_format: str | None = None,
    codec: str | None = None,
    timeout: float = 120.0,
    cancellation_token: CancellationToken | None = None,
    run_command_fn: Any = run_command,
) -> Path:
    """Demux an embedded subtitle stream from container using container global stream mapping (0:<index>).

    Invariants:
    - Uses exact container global stream mapping: -map 0:<stream_index>.
    - Finite timeout and cancellation safety.
    - Explicit failure on non-zero exit code or missing output (never silent).
    """
    if cancellation_token:
        cancellation_token.check_cancelled()

    p_video = Path(video_path).resolve()
    if not p_video.is_file():
        raise FileNotFoundError(f"Source video file not found: {p_video}")

    p_out = Path(output_path).resolve()
    p_out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = find_binary("ffmpeg")
    fmt = (source_format or p_out.suffix.lstrip(".").lower()).lower()
    codec_lower = (codec or "").lower()

    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-v", "error",
        "-i", str(p_video),
        "-map", f"0:{stream_index}",
    ]

    if fmt in ("srt", "subrip", "mov_text", "tx3g") or codec_lower in ("subrip", "srt", "mov_text", "tx3g"):
        cmd.extend(["-c:s", "srt", str(p_out)])
    elif fmt in ("ass", "ssa") or codec_lower in ("ass", "ssa"):
        cmd.extend(["-c:s", "ass", str(p_out)])
    elif fmt in ("vtt", "webvtt") or codec_lower in ("webvtt", "vtt"):
        cmd.extend(["-c:s", "webvtt", str(p_out)])
    elif fmt in ("pgs", "sup") or codec_lower in ("hdmv_pgs_subtitle", "pgs"):
        cmd.extend(["-c:s", "copy", str(p_out)])
    elif fmt in ("vobsub", "idx") or codec_lower in ("dvd_subtitle", "dvdsub"):
        cmd.extend(["-c:s", "copy", str(p_out)])
    else:
        cmd.extend(["-c:s", "copy", str(p_out)])

    res = run_command_fn(cmd, timeout=timeout, cancellation_token=cancellation_token)
    if res.exit_code != 0:
        err_msg = res.stderr.strip() or res.stdout.strip()
        raise ToolRecapError(
            f"FFmpeg failed to demux subtitle stream #{stream_index} from {p_video.name} "
            f"(exit code {res.exit_code}): {err_msg}"
        )

    if fmt in ("vobsub", "idx") or codec_lower in ("dvd_subtitle", "dvdsub"):
        p_sub = p_out.with_suffix(".sub")
        if not p_out.is_file() or not p_sub.is_file():
            raise ToolRecapError(
                f"FFmpeg demux did not produce expected paired VobSub files: {p_out} and {p_sub}"
            )
    elif not p_out.is_file():
        raise ToolRecapError(
            f"FFmpeg demux did not produce expected output file: {p_out}"
        )

    return p_out


@dataclass(frozen=True)
class SubtitlePipelineResult:
    """Result of subtitle extraction, parsing, and OCR processing."""

    track: SubtitleTrack | None = None
    cues: tuple[SubtitleCue, ...] = field(default_factory=tuple)
    status: str = "success"  # 'success' | 'empty' | 'failed' | 'fallback_required' | 'no_track'
    source_type: str = "none"  # 'sidecar' | 'embedded' | 'none'
    source_format: str = "none"  # 'srt' | 'ass' | 'vtt' | 'pgs' | 'vobsub' | 'none'
    language: str = "und"
    diagnostics: tuple[str, ...] = field(default_factory=tuple)
    cached: bool = False
    dropped_cues_count: int = 0

    @property
    def cue_count(self) -> int:
        return len(self.cues)

    @property
    def has_cues(self) -> bool:
        return len(self.cues) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track.to_dict() if self.track else None,
            "cues": [c.to_dict() for c in self.cues],
            "status": self.status,
            "source_type": self.source_type,
            "source_format": self.source_format,
            "language": self.language,
            "diagnostics": list(self.diagnostics),
            "cached": self.cached,
            "dropped_cues_count": self.dropped_cues_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitlePipelineResult:
        raw_track = data.get("track")
        track = SubtitleTrack(**raw_track) if isinstance(raw_track, dict) else raw_track
        raw_cues = data.get("cues", [])
        cues = tuple(
            SubtitleCue.from_dict(c) if isinstance(c, dict) else c
            for c in raw_cues
        )
        return cls(
            track=track,
            cues=cues,
            status=str(data.get("status", "success")),
            source_type=str(data.get("source_type", "none")),
            source_format=str(data.get("source_format", "none")),
            language=str(data.get("language", "und")),
            diagnostics=tuple(str(d) for d in data.get("diagnostics", [])),
            cached=bool(data.get("cached", False)),
            dropped_cues_count=int(data.get("dropped_cues_count", 0)),
        )


class SubtitlePipeline:
    """Orchestrates subtitle extraction, embedded demuxing, bitmap OCR, and atomic caching.

    Invariants:
    - Checks atomic cache before expensive parsing or OCR.
    - Demuxes embedded streams safely using -map 0:<stream_index>.
    - Cleans up temporary demuxed files in all execution paths.
    - Quality gates OCR results and records dropped cues and diagnostics.
    - Explicit status errors vs legitimate empty; never swallows CancelledError.
    """

    def __init__(
        self,
        cache_manager: SubtitleCacheManager | None = None,
        ocr_adapter: OcrAdapter | None = None,
        run_command_fn: Any = run_command,
        temp_dir: Path | str | None = None,
    ) -> None:
        self.cache_manager = cache_manager or SubtitleCacheManager()
        self.ocr_adapter = ocr_adapter or OcrAdapter()
        self.run_command_fn = run_command_fn
        self.temp_dir = Path(temp_dir) if temp_dir else None

    def extract_cues(
        self,
        track: SubtitleTrack,
        source_video: Path | str,
        episode_id: str = "",
        *,
        source_fingerprint: str | None = None,
        source_duration_ms: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> SubtitlePipelineResult:
        """Extract and normalize subtitle cues from a discovered or embedded SubtitleTrack."""
        if cancellation_token:
            cancellation_token.check_cancelled()

        # 1. Check cache first
        cached_cues = self.cache_manager.load_cues(
            episode_id, source_video, track, source_fingerprint=source_fingerprint
        )
        if cached_cues is not None:
            return SubtitlePipelineResult(
                track=track,
                cues=tuple(cached_cues),
                status="success" if cached_cues else "empty",
                source_type=track.source_type,
                source_format=track.source_format,
                language=track.language,
                diagnostics=("Loaded from verified atomic cache",),
                cached=True,
                dropped_cues_count=0,
            )

        # 2. Cache miss: extract or parse
        temp_cleanup_paths: list[Path] = []
        try:
            cues: list[SubtitleCue] = []
            dropped_cues = 0
            diag: list[str] = []

            fmt = track.source_format.lower()
            is_embedded = (track.source_type == "embedded")

            # Resolve file path for parsing
            file_to_parse: Path
            if is_embedded:
                if track.stream_index is None:
                    raise ToolRecapError(f"Embedded track {track.track_id} has no stream_index")

                # Setup temp output path
                base_tmp = self.temp_dir or (Path(tempfile.gettempdir()) / "kilo" / "toolrecap_subs")
                base_tmp.mkdir(parents=True, exist_ok=True)
                nonce = uuid.uuid4().hex[:8]

                if fmt in ("srt", "subrip", "mov_text", "tx3g"):
                    ext = "srt"
                elif fmt in ("ass", "ssa"):
                    ext = "ass"
                elif fmt in ("vtt", "webvtt"):
                    ext = "vtt"
                elif fmt in ("pgs", "sup"):
                    ext = "sup"
                elif fmt in ("vobsub", "idx"):
                    ext = "idx"
                else:
                    ext = "sub"

                tmp_demux = base_tmp / f"ep_{episode_id}_{track.stream_index}_{nonce}.{ext}"
                temp_cleanup_paths.append(tmp_demux)
                if ext == "idx":
                    temp_cleanup_paths.append(tmp_demux.with_suffix(".sub"))

                extract_embedded_subtitle_stream(
                    video_path=source_video,
                    stream_index=track.stream_index,
                    output_path=tmp_demux,
                    source_format=fmt,
                    cancellation_token=cancellation_token,
                    run_command_fn=self.run_command_fn,
                )
                file_to_parse = tmp_demux
            else:
                if not track.source_file:
                    raise FileNotFoundError(f"Sidecar track {track.track_id} missing source_file")
                file_to_parse = Path(track.source_file).resolve()
                if not file_to_parse.is_file():
                    raise FileNotFoundError(f"Sidecar file does not exist: {file_to_parse}")

            if cancellation_token:
                cancellation_token.check_cancelled()

            # Format-specific parsing & OCR
            if fmt in ("srt", "subrip", "mov_text", "tx3g"):
                cues = parse_srt(
                    file_to_parse,
                    source_type=track.source_type,
                    source_format="srt",
                    stream_index=track.stream_index,
                    source_file=str(file_to_parse),
                    language=track.language,
                    episode_id=episode_id,
                    source_video=str(source_video),
                    source_duration_ms=source_duration_ms,
                )
            elif fmt in ("ass", "ssa"):
                cues = parse_ass(
                    file_to_parse,
                    source_type=track.source_type,
                    source_format=fmt,
                    stream_index=track.stream_index,
                    source_file=str(file_to_parse),
                    language=track.language,
                    episode_id=episode_id,
                    source_video=str(source_video),
                    source_duration_ms=source_duration_ms,
                )
            elif fmt in ("vtt", "webvtt"):
                cues = parse_vtt(
                    file_to_parse,
                    source_type=track.source_type,
                    source_format="vtt",
                    stream_index=track.stream_index,
                    source_file=str(file_to_parse),
                    language=track.language,
                    episode_id=episode_id,
                    source_video=str(source_video),
                    source_duration_ms=source_duration_ms,
                )
            elif fmt in ("pgs", "sup"):
                pgs_events = parse_pgs_sup(file_to_parse)
                cues, dropped_cues, ocr_diagnostics = self._ocr_events(
                    pgs_events,
                    track=track,
                    episode_id=episode_id,
                    source_video=source_video,
                    source_duration_ms=source_duration_ms,
                    cancellation_token=cancellation_token,
                )
                diag.append(f"Decoded {len(pgs_events)} PGS events; OCR produced {len(cues)} cues")
                diag.extend(ocr_diagnostics)
            elif fmt in ("vobsub", "idx"):
                vob_events = extract_vobsub_events(file_to_parse)
                cues, dropped_cues, ocr_diagnostics = self._ocr_events(
                    vob_events,
                    track=track,
                    episode_id=episode_id,
                    source_video=source_video,
                    source_duration_ms=source_duration_ms,
                    cancellation_token=cancellation_token,
                )
                diag.append(f"Decoded {len(vob_events)} VobSub events; OCR produced {len(cues)} cues")
                diag.extend(ocr_diagnostics)
            else:
                raise ToolRecapError(f"Unsupported subtitle format: '{fmt}'")

            if cancellation_token:
                cancellation_token.check_cancelled()

            # 3. Save to atomic cache
            if cues or not track.is_bitmap:
                # A failed OCR runtime/quality pass can recover later. Never
                # persist an empty bitmap result as a durable cache hit.
                self.cache_manager.save_cues(
                    episode_id=episode_id,
                    source_video=source_video,
                    track=track,
                    cues=cues,
                    source_fingerprint=source_fingerprint,
                    cancellation_token=cancellation_token,
                )

            status = "success" if cues else "empty"
            return SubtitlePipelineResult(
                track=track,
                cues=tuple(cues),
                status=status,
                source_type=track.source_type,
                source_format=track.source_format,
                language=track.language,
                diagnostics=tuple(diag),
                cached=False,
                dropped_cues_count=dropped_cues,
            )

        except CancelledError:
            raise
        except Exception as exc:
            return SubtitlePipelineResult(
                track=track,
                cues=(),
                status="failed",
                source_type=track.source_type,
                source_format=track.source_format,
                language=track.language,
                diagnostics=(f"Subtitle extraction error: {exc}",),
                cached=False,
                dropped_cues_count=0,
            )
        finally:
            for p in temp_cleanup_paths:
                try:
                    if p.is_file():
                        p.unlink()
                except OSError:
                    pass

    def _ocr_events(
        self,
        events: Sequence[PgsSubtitleEvent | VobSubEvent],
        track: SubtitleTrack,
        episode_id: str,
        source_video: Path | str,
        source_duration_ms: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[list[SubtitleCue], int, list[str]]:
        """Run bounded local OCR across bitmap subtitle events."""
        cues: list[SubtitleCue] = []
        dropped = 0
        diagnostics: list[str] = []

        for ev in events:
            if cancellation_token:
                cancellation_token.check_cancelled()

            # Boundary validation
            if source_duration_ms is not None and ev.end_ms > source_duration_ms:
                dropped += 1
                continue
            if ev.end_ms <= ev.start_ms:
                dropped += 1
                continue

            if ev.image is None:
                dropped += 1
                continue

            ocr_res: OcrResult = self.ocr_adapter.ocr_image(
                ev.image,
                cancellation_token=cancellation_token,
            )

            if ocr_res.is_valid and ocr_res.text:
                cue = SubtitleCue(
                    start_ms=ev.start_ms,
                    end_ms=ev.end_ms,
                    text=ocr_res.text,
                    source_type=track.source_type,
                    source_format=track.source_format,
                    stream_index=track.stream_index,
                    source_file=track.source_file,
                    language=track.language,
                    confidence=ocr_res.confidence,
                    episode_id=episode_id,
                    source_video=str(source_video),
                )
                cues.append(cue)
            else:
                dropped += 1
                if ocr_res.reason and ocr_res.reason not in diagnostics and len(diagnostics) < 3:
                    diagnostics.append(ocr_res.reason)

        return cues, dropped, diagnostics

    def process_subtitles(
        self,
        tracks: Sequence[SubtitleTrack],
        source_video: Path | str,
        episode_id: str = "",
        *,
        source_fingerprint: str | None = None,
        source_duration_ms: int | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> SubtitlePipelineResult:
        """Select best candidate English Full track and extract cues."""
        if cancellation_token:
            cancellation_token.check_cancelled()

        disc = select_best_english_subtitles(
            list(tracks),
            video_path=str(source_video),
            episode_id=episode_id,
        )

        if disc.best_english_full is None:
            return SubtitlePipelineResult(
                track=None,
                cues=(),
                status="fallback_required",
                source_type="none",
                source_format="none",
                language="und",
                diagnostics=(disc.selection_reason,),
            )

        return self.extract_cues(
            disc.best_english_full,
            source_video=source_video,
            episode_id=episode_id,
            source_fingerprint=source_fingerprint,
            source_duration_ms=source_duration_ms,
            cancellation_token=cancellation_token,
        )
