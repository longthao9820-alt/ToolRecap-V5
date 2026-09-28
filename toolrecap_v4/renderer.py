"""Video rendering pipeline for ToolRecap V4.

Invariants:
- Exact JSON ordered multi-source timestamps.
- Video normalized to common even canvas with aspectfit padding and uniform fps.
- Audio normalized to stereo 48kHz; generates silence for missing audio.
- Narration synthesis unchanged; fails on oversize (NarrationFitError); never trims clips.
- Audio gains exact dB: ducking applied only while narration overlaps source_audio=true.
- Original dialogue segments never ducked.
- Final continuous TWO PASS EBU R128 loudnorm and peak limiting.
- GPU acceleration via actual tested media primitives (NVENC / AMF / QSV) with CPU fallback.
- Internal temporary directory isolated; only {title}.mp4, {title}.narration.srt, and
  {title}.original.srt published atomically.
- Source files and project JSON strictly immutable.
- Cancellation and failure cleanup guaranteed.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    CancelledError,
    CancelledError,
    InvalidAudioError,
    JsonSourceNotFoundError,
    NarrationFitError,
    TimestampBoundaryError,
    ToolRecapError,
    VoiceStudioUnavailableError,
    WindowsCollisionError,
    WindowsNameError,
)
from toolrecap_v4.media import (
    EncoderStatus,
    MediaProbeResult,
    calculate_auto_canvas,
    detect_gpu_encoder,
    find_binary,
    get_video_encode_args,
    nvidia_decode_args,
    nvidia_scale_pad_filter,
    probe_nvidia_source_pipeline,
    probe_duration,
    probe_media,
    run_command,
    select_audio_for_source,
)
from toolrecap_v4.persistence import get_storage_root
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.subtitles import (
    escape_ffmpeg_subtitles_path,
    extract_subtitles_for_output,
    generate_subtitles,
    write_srt,
)
from toolrecap_v4.validator import validate_windows_name
from toolrecap_v4.voice_studio import VoiceStudioAdapter, validate_wav_bytes
from toolrecap_v4.progress import ActivityState, WorkflowStage, safe_emit
from toolrecap_v4.narration_fit import resolve_narration_fit
from toolrecap_v4.original_dialogue import (
    ORIGINAL_DIALOGUE_MAPPED,
    ORIGINAL_SUBTITLE_GENERATION_FAILED,
    NO_ORIGINAL_DIALOGUE,
    OriginalDialogueMapping,
    OriginalDialogueSubtitleMapper,
    ResolvedSourceClip,
)


class RenderError(ToolRecapError):
    """Base exception for render failures."""


class SourceCollisionError(RenderError):
    """Raised when publication path collides with source files."""


class RenderValidationError(RenderError):
    """Raised when rendered output fails technical validation."""


@dataclass(frozen=True)
class RenderResult:
    """Result of rendering a single output title."""

    output_path: Path
    narration_srt_path: Path
    original_srt_path: Path
    duration: float
    title: str
    render_id: str
    video_codec: str
    audio_codec: str
    width: int
    height: int
    fps: float
    original_subtitle_state: str = "NO_ORIGINAL_DIALOGUE"
    original_subtitle_cue_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "narration_srt_path": str(self.narration_srt_path),
            "original_srt_path": str(self.original_srt_path),
            "duration": self.duration,
            "title": self.title,
            "render_id": self.render_id,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "original_subtitle_state": self.original_subtitle_state,
            "original_subtitle_cue_count": self.original_subtitle_cue_count,
        }


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _stage_and_publish_files(
    files: Sequence[tuple[Path, Path]],
    *,
    cancellation_token: CancellationToken | None = None,
) -> None:
    """Stage every publication file beside its target and always remove staging files."""
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for source, destination in files:
            file_descriptor, staging_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
            )
            os.close(file_descriptor)
            staging = Path(staging_name)
            staged.append((staging, destination))
            shutil.copy2(source, staging)
            if destination.exists():
                backup_descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.", suffix=".bak", dir=destination.parent,
                )
                os.close(backup_descriptor)
                backup = Path(backup_name)
                shutil.copy2(destination, backup)
                backups[destination] = backup
        if cancellation_token:
            cancellation_token.check_cancelled()
        for staging, destination in staged:
            staging.replace(destination)
            published.append(destination)
    except Exception:
        # Multi-file publication is not atomically swappable as a group.  Roll
        # back every destination already replaced so Resume never encounters a
        # half-published MP4/SRT set.
        for destination in reversed(published):
            backup = backups.get(destination)
            if backup is not None and backup.exists():
                backup.replace(destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    finally:
        for staging, _destination in staged:
            staging.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def calculate_ducking_gains(
    original_db: float,
    commentary_db: float,
    ducking_amount_db: float,
    auto_duck: bool,
) -> tuple[float, float, float]:
    """Calculate linear gain multipliers: (normal_orig, ducked_orig, commentary)."""
    normal_orig = 10.0 ** (original_db / 20.0)
    if auto_duck:
        duck_delta = ducking_amount_db if ducking_amount_db <= 0 else -ducking_amount_db
        ducked_db = original_db + duck_delta
        ducked_orig = 10.0 ** (ducked_db / 20.0)
    else:
        ducked_orig = normal_orig
    commentary_linear = 10.0 ** (commentary_db / 20.0)
    return normal_orig, ducked_orig, commentary_linear


def measure_loudnorm(
    media_path: Path | str,
    *,
    target_lufs: float = -14.0,
    true_peak: float = -1.0,
    ffmpeg_path: Path | str | None = None,
    timeout: float = 60.0,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, str]:
    """Run pass 1 loudnorm measurement on media file and return parsed JSON parameters."""
    binary = find_binary("ffmpeg", ffmpeg_path)
    null_sink = "NUL" if sys.platform == "win32" else "/dev/null"
    cmd = [
        str(binary),
        "-hide_banner",
        "-i", str(Path(media_path).resolve()),
        "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:print_format=json",
        "-f", "null",
        null_sink,
    ]

    res = run_command(cmd, timeout=timeout, cancellation_token=cancellation_token)
    if res.exit_code != 0:
        raise RenderError(f"Pass 1 loudnorm measurement failed (exit {res.exit_code}): {res.stderr.strip()}")

    matches = re.findall(r"\{\s*\"input_i\"[\s\S]*?\}", res.stderr)
    if not matches:
        return {}

    try:
        data = json.loads(matches[-1])
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def build_two_pass_loudnorm_filter(
    measured: dict[str, str],
    *,
    target_lufs: float = -14.0,
    true_peak: float = -1.0,
) -> str:
    """Build FFmpeg loudnorm filter string for pass 2 from pass 1 measurements."""
    input_i = measured.get("input_i", "")
    target_offset = measured.get("target_offset", "")

    if not input_i:
        return f"loudnorm=I={target_lufs}:TP={true_peak}"

    try:
        val = float(input_i)
        if not math.isfinite(val):
            return "anull"
        if val <= -70.0:
            return f"loudnorm=I={target_lufs}:TP={true_peak}"
    except ValueError:
        return f"loudnorm=I={target_lufs}:TP={true_peak}"

    if target_offset:
        try:
            if not math.isfinite(float(target_offset)):
                return "anull"
        except ValueError:
            return f"loudnorm=I={target_lufs}:TP={true_peak}"

    input_lra = measured.get("input_lra", "7.0")
    input_tp = measured.get("input_tp", "-1.0")
    input_thresh = measured.get("input_thresh", "-24.0")
    offset = measured.get("target_offset", "0.0")

    return (
        f"loudnorm=I={target_lufs}:TP={true_peak}:"
        f"measured_I={input_i}:measured_LRA={input_lra}:measured_TP={input_tp}:"
        f"measured_thresh={input_thresh}:offset={offset}:linear=true"
    )


def get_render_work_root(override_dir: Optional[str | Path] = None) -> Path:
    """Return render scratch directory strictly in LOCALAPPDATA/ToolRecapV4/render-work."""
    base = get_storage_root(override_dir) / "render-work"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_sources(
    source_names: Sequence[str],
    source_paths: dict[str, Path | str],
) -> dict[str, Path]:
    """Map source basenames to verified absolute Path objects.
    
    Exact mapping ONLY. Directory lookup and traversal are strictly rejected.
    """
    if not isinstance(source_paths, dict):
        raise TypeError(f"source_paths must be an exact mapping dict, got {type(source_paths).__name__}")

    resolved: dict[str, Path] = {}
    for name in source_names:
        if not isinstance(name, str):
            raise JsonSourceNotFoundError(f"Source file name must be a string, got {type(name).__name__}")
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise JsonSourceNotFoundError(
                f"Source file name '{name}' contains directory traversal or path separators."
            )
        if name not in source_paths:
            raise JsonSourceNotFoundError(f"Source file '{name}' not provided in source_paths mapping.")
        candidate = Path(source_paths[name]).resolve()
        if not candidate.is_file():
            raise JsonSourceNotFoundError(f"Source file '{name}' at '{candidate}' does not exist.")
        resolved[name] = candidate

    return resolved


def render_output(
    output_def: dict[str, Any],
    source_paths: dict[str, Path | str],
    output_dir: Path | str,
    *,
    settings: AppSettings | None = None,
    voice_adapter: VoiceStudioAdapter | None = None,
    cancellation_token: CancellationToken | None = None,
    narration_audio_map: dict[str, Path | str] | None = None,
    source_dialogue_map: dict[str, Sequence[Any]] | None = None,
    ffmpeg_path: Path | str | None = None,
    work_dir: Path | str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    publication_owner: str | None = None,
    publication_journal_root: Path | None = None,
    allow_publication_overwrite: bool = False,
    audio_stream_indices: dict[str, int] | None = None,
    render_mode: str = "RECAP",
    _hardware_fallback_attempted: bool = False,
) -> RenderResult:
    """Render a single output specification according to all technical invariants."""
    if cancellation_token:
        cancellation_token.check_cancelled()

    cfg = settings or AppSettings()
    is_highlight = str(render_mode).upper() == "HIGHLIGHT"
    ffmpeg = find_binary("ffmpeg", ffmpeg_path)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    title = output_def.get("title")
    validate_windows_name(title, "title")

    render_id = output_def.get("render_id")
    if render_id is not None:
        validate_windows_name(render_id, "render_id")
    else:
        render_id = title

    segments = output_def.get("segments", [])
    if not segments:
        raise RenderError(f"Output '{render_id}' contains no segments.")
    if is_highlight and any(
        segment.get("type") != "original_dialogue"
        or str(segment.get("narration") or "").strip()
        for segment in segments
    ):
        raise RenderValidationError(
            f"Highlight output '{render_id}' must contain original-dialogue segments without narration."
        )

    def emit(stage: WorkflowStage, **payload: Any) -> None:
        safe_emit(progress_callback, stage=stage.value, output_id=render_id, **payload)

    # Collect source files
    source_names = [seg["source_file"] for seg in segments]
    resolved_sources = _resolve_sources(source_names, source_paths)

    # Collision check: publication files must not collide with any source files (including unused)
    target_mp4 = out_dir / f"{title}.mp4"
    target_narr_srt = out_dir / f"{title}.narration.srt"
    target_orig_srt = out_dir / f"{title}.original.srt"
    publication_targets = {target_mp4.resolve(), target_narr_srt.resolve(), target_orig_srt.resolve()}

    for src_name, src_val in source_paths.items():
        src_path = Path(src_val).resolve()
        if src_path in publication_targets:
            raise SourceCollisionError(
                f"Publication target collides with source file '{src_name}' at '{src_path}'."
            )

    # Probe all source files
    probes: dict[str, MediaProbeResult] = {}
    for name, path in resolved_sources.items():
        probes[name] = probe_media(path, cancellation_token=cancellation_token)

    # Canvas dimensions and FPS (guarantee even numbers for H.264/yuv420p)
    first_seg = segments[0]
    first_src_name = first_seg["source_file"]
    first_probe = probes[first_src_name]

    if getattr(cfg, "canvas_auto", True):
        canvas_w, canvas_h = calculate_auto_canvas(
            first_probe,
            fallback_w=int(cfg.canvas_width),
            fallback_h=int(cfg.canvas_height),
        )
    else:
        canvas_w = int(cfg.canvas_width)
        canvas_h = int(cfg.canvas_height)
        if canvas_w % 2 != 0:
            canvas_w += 1
        if canvas_h % 2 != 0:
            canvas_h += 1
    canvas_fps = float(cfg.canvas_fps)

    # Prepare isolated temporary working directory in LOCALAPPDATA/ToolRecapV4/render-work
    render_work_base = Path(work_dir).resolve() if work_dir is not None else get_render_work_root()
    render_work_base.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="render_", dir=render_work_base)).resolve()
    segments_dir = temp_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    narration_dir = temp_dir / "narration"
    narration_dir.mkdir(parents=True, exist_ok=True)

    try:
        segment_mkv_paths: list[Path] = []
        from toolrecap_v4.narration_groups import narration_groups, slice_group_audio
        grouped_audio: dict[str, Path] = {}
        group_durations: dict[str, int] = {}
        for group_id, members in narration_groups(segments).items():
            leader = members[0]
            recording = (narration_audio_map or {}).get(leader["segment_id"])
            if recording is None:
                if voice_adapter is None:
                    raise VoiceStudioUnavailableError(f"Missing speech for narration group {group_id}")
                from toolrecap_v4.downstream import apply_commentary_speed
                speech = voice_adapter.synthesize(
                    leader["narration"], voice=cfg.voice_id, model=cfg.voice_model,
                    language=cfg.voice_language, style=cfg.voice_style,
                    cancellation_token=cancellation_token,
                )
                recording = narration_dir / f"group_{leader['segment_id']}.wav"
                recording.write_bytes(apply_commentary_speed(speech, cfg.commentary_reading_speed, ffmpeg_path=ffmpeg))
            parts, duration = slice_group_audio(Path(recording), members, narration_dir)
            grouped_audio.update(parts)
            group_durations[group_id] = duration
        effective_segment_durations: dict[str, int] = {}
        resolved_source_clips: list[ResolvedSourceClip] = []
        output_audio_languages: set[str] = set()
        final_timeline_offset_ms = 0
        if is_highlight:
            # Highlight keeps source-only audio. Recap gain/duck/commentary
            # controls must not alter its original-dialogue segments.
            normal_orig_linear = ducked_orig_linear = comm_linear = 1.0
        else:
            normal_orig_linear, ducked_orig_linear, comm_linear = calculate_ducking_gains(
                cfg.original_audio_db,
                cfg.commentary_audio_db,
                cfg.ducking_amount_db,
                cfg.auto_duck,
            )

        # Video encode args for segment intermediate (GPU or CPU)
        v_args, enc_status = get_video_encode_args(
            quality=cfg.quality,
            use_gpu=cfg.use_gpu,
            ffmpeg_path=ffmpeg,
        )
        if enc_status.encoder == "h264_nvenc":
            source_starts = {
                name: min(int(segment["start_ms"]) for segment in segments if segment["source_file"] == name)
                for name in resolved_sources
            }
            failures = []
            for name, source_path in resolved_sources.items():
                source_probe = probes[name]
                video = source_probe.video_streams[0] if source_probe.video_streams else None
                if video is None:
                    failures.append(f"{name}: no video stream")
                    continue
                usable, detail = probe_nvidia_source_pipeline(
                    ffmpeg, source_path, source_codec=video.codec,
                    pixel_format=video.pixel_format, canvas_width=canvas_w,
                    canvas_height=canvas_h, fps=canvas_fps,
                    start_seconds=source_starts[name] / 1000.0,
                )
                if not usable:
                    failures.append(f"{name}: {detail[-300:]}")
            if failures:
                v_args, enc_status = get_video_encode_args(
                    quality=cfg.quality, use_gpu=False, ffmpeg_path=ffmpeg,
                )
                emit(
                    WorkflowStage.RENDER,
                    state=ActivityState.RETRYING.value,
                    activity_text="Source is incompatible with the NVIDIA processing path; using CPU for this output.",
                    current_item=render_id,
                    gpu_fallback_reason=" | ".join(failures)[:1000],
                )
        emit(
            WorkflowStage.AUDIO_MIX,
            state=ActivityState.LOCAL_PROCESSING.value,
            activity_text=(
                "Processing original source audio..."
                if is_highlight else "Mixing narration with original audio..."
            ),
            current_item=render_id, completed=0, total=len(segments), unit="segments",
        )

        for seg_idx, seg in enumerate(segments):
            if cancellation_token:
                cancellation_token.check_cancelled()

            seg_id = seg.get("segment_id")
            validate_windows_name(seg_id, "segment_id")
            src_name = seg["source_file"]
            src_path = resolved_sources[src_name]
            src_probe = probes[src_name]

            start_ms = int(seg["start_ms"])
            end_ms = int(seg["end_ms"])
            if start_ms < 0:
                raise TimestampBoundaryError(
                    f"Segment '{seg_id}' start_ms ({start_ms}) cannot be negative."
                )
            if end_ms <= start_ms:
                raise TimestampBoundaryError(
                    f"Segment '{seg_id}' end_ms ({end_ms}) <= start_ms ({start_ms})."
                )

            planned_seg_dur_ms = end_ms - start_ms
            seg_dur_ms = planned_seg_dur_ms
            seg_dur_s = seg_dur_ms / 1000.0
            seg_start_s = start_ms / 1000.0

            # Source duration validation: strict actual duration, no 100ms bypass
            src_dur_ms = int(round(src_probe.duration * 1000.0))
            if src_dur_ms > 0 and end_ms > src_dur_ms:
                raise TimestampBoundaryError(
                    f"Segment '{seg_id}' end_ms ({end_ms}ms) exceeds source duration ({src_dur_ms}ms) for '{src_name}'."
                )

            seg_type = seg.get("type", "narration")
            narration_text = seg.get("narration")
            if narration_text is None:
                narration_text = ""
            # Legacy generated Final JSON used source_audio=false for every
            # narration segment.  Absence of an explicit audio_intent therefore
            # migrates to the product-wide SOURCE FOOTAGE -> SOURCE AUDIO rule.
            audio_intent = str(seg.get("audio_intent") or "NARRATION_WITH_ORIGINAL_AUDIO")
            source_audio_enabled = audio_intent != "SILENT_SOURCE"

            # Narration synthesis / retrieval
            narr_wav: Optional[Path] = None
            narr_dur_s = 0.0

            if seg.get("narration_group"):
                narr_wav = grouped_audio.get(seg_id)
                narr_dur_s = validate_wav_bytes(narr_wav.read_bytes()) if narr_wav else 0.0
                source_visual_s = seg_dur_s
                hold_s = 0.0
            elif seg_type == "narration" and narration_text:
                if narration_audio_map and seg_id in narration_audio_map:
                    narr_wav = Path(narration_audio_map[seg_id]).resolve()
                    if not narr_wav.is_file():
                        raise RenderError(
                            f"Narration audio file for segment '{seg_id}' not found at '{narr_wav}'."
                        )
                    narr_dur_s = validate_wav_bytes(narr_wav.read_bytes())
                elif voice_adapter is not None:
                    wav_bytes = voice_adapter.synthesize(
                        narration_text,
                        voice=cfg.voice_id,
                        model=cfg.voice_model,
                        language=cfg.voice_language,
                        style=cfg.voice_style,
                        cancellation_token=cancellation_token,
                    )
                    from toolrecap_v4.downstream import apply_commentary_speed
                    wav_bytes = apply_commentary_speed(wav_bytes, cfg.commentary_reading_speed, ffmpeg_path=ffmpeg)
                    narr_dur_s = validate_wav_bytes(wav_bytes)
                    narr_wav = narration_dir / f"narr_{seg_idx:04d}.wav"
                    narr_wav.write_bytes(wav_bytes)
                else:
                    raise VoiceStudioUnavailableError(
                        f"Segment '{seg_id}' specifies narration text, but no voice adapter or narration audio was provided."
                    )

                narr_dur_ms = int(round(narr_dur_s * 1000.0))
                fit=resolve_narration_fit(segment_id=seg_id,start_ms=start_ms,end_ms=end_ms,source_duration_ms=src_dur_ms,narration_duration_ms=narr_dur_ms)
                end_ms=fit.source_end_ms;seg_dur_ms=fit.render_duration_ms;seg_dur_s=seg_dur_ms/1000.0
                effective_segment_durations[seg_id] = seg_dur_ms
                source_visual_s=(fit.source_end_ms-fit.start_ms)/1000.0;hold_s=fit.hold_ms/1000.0
                emit(WorkflowStage.NARRATION_FIT,state=ActivityState.LOCAL_PROCESSING.value,activity_text=f"Narration {narr_dur_s:.1f}s; visual {planned_seg_dur_ms/1000.0:.1f}s; strategy: {', '.join(fit.strategy)}",current_item=seg_id,completed=seg_idx+1,total=len(segments),unit="segments",stage_status="complete" if seg_idx+1==len(segments) else None)
            else:
                source_visual_s=seg_dur_s;hold_s=0.0

            resolved_source_clips.append(ResolvedSourceClip(
                segment_id=seg_id,
                source_file=src_name,
                source_start_ms=start_ms,
                source_end_ms=end_ms,
                final_start_ms=final_timeline_offset_ms,
                final_duration_ms=seg_dur_ms,
                source_audio=source_audio_enabled,
            ))
            final_timeline_offset_ms += seg_dur_ms

            # Build FFmpeg command for normalized intermediate segment MKV
            seg_mkv = segments_dir / f"seg_{seg_idx:04d}.mkv"
            cmd: list[str] = [
                str(ffmpeg),
                "-y",
                "-accurate_seek",
                "-ss", f"{seg_start_s:.6f}",
                "-t", f"{source_visual_s:.6f}",
            ]
            source_codec = src_probe.video_streams[0].codec if src_probe.video_streams else ""
            source_pixel_format = src_probe.video_streams[0].pixel_format if src_probe.video_streams else ""
            decode_args = nvidia_decode_args(source_codec, enc_status, source_pixel_format)
            cmd.extend(decode_args)
            cmd.extend(["-i", str(src_path)])

            input_count = 1
            has_usable_source_audio = src_probe.has_audio and source_audio_enabled

            # Silence generator if no source audio
            silence_input_idx: Optional[int] = None
            if not has_usable_source_audio:
                cmd.extend([
                    "-f", "lavfi",
                    "-i", f"anullsrc=r=48000:cl=stereo:d={seg_dur_s:.6f}",
                ])
                silence_input_idx = input_count
                input_count += 1

            # Narration audio input
            narr_input_idx: Optional[int] = None
            if narr_wav is not None and narr_wav.is_file():
                cmd.extend(["-i", str(narr_wav)])
                narr_input_idx = input_count
                input_count += 1

            # Video filter: aspect-fit into canvas, pad borders, set FPS and yuv420p
            # Normalize anamorphic pixels to square pixels via ih*dar before aspect-fit scale
            if enc_status.encoder == "h264_nvenc":
                v_filter = (
                    "[0:v]" + nvidia_scale_pad_filter(
                        canvas_width=canvas_w, canvas_height=canvas_h, fps=canvas_fps,
                        source_duration=source_visual_s, hold_duration=hold_s,
                        hardware_input=bool(decode_args),
                    ) + "[vout]"
                )
            else:
                hold_filter=f",tpad=stop_mode=clone:stop_duration={hold_s:.6f}" if hold_s>0 else ""
                v_filter = (
                    f"[0:v]scale=ih*dar:ih,scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                    f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
                    f"fps={canvas_fps},format=yuv420p,"
                    f"trim=0:{source_visual_s:.6f},setpts=PTS-STARTPTS{hold_filter}[vout]"
                )

            filter_chains = [v_filter]

            # Original audio filter
            if has_usable_source_audio:
                selection = select_audio_for_source(src_probe)
                stream_idx = (audio_stream_indices or {}).get(src_name)
                if stream_idx is None:
                    stream_idx = selection.selected_index if selection.selected_index is not None else 0
                if not any(stream.index == stream_idx for stream in src_probe.audio_streams):
                    raise RenderValidationError(f"Selected audio stream {stream_idx} is unavailable in {src_name}")
                chosen_stream = next(stream for stream in src_probe.audio_streams if stream.index == stream_idx)
                if chosen_stream.language:
                    output_audio_languages.add(chosen_stream.language)
                a_spec = f"0:{stream_idx}"

                if seg_type == "original_dialogue":
                    # Invariant: original dialogue never ducked
                    gain_clause = f"volume={normal_orig_linear:.6f}:eval=once"
                elif seg_type == "narration":
                    if cfg.auto_duck and narr_dur_s > 0:
                        # Duck only while narration overlaps
                        gain_clause = f"volume='if(between(t,0,{narr_dur_s:.6f}),{ducked_orig_linear:.6f},{normal_orig_linear:.6f})':eval=frame"
                    else:
                        gain_clause = f"volume={normal_orig_linear:.6f}:eval=once"
                else:
                    gain_clause = f"volume={normal_orig_linear:.6f}:eval=once"

                a_orig_chain = (
                    f"[{a_spec}]aresample=48000,aformat=channel_layouts=stereo,{gain_clause},"
                    f"apad=whole_dur={seg_dur_s:.6f},atrim=0:{seg_dur_s:.6f},asetpts=PTS-STARTPTS[aorig]"
                )
            else:
                a_orig_chain = f"[{silence_input_idx}:a]atrim=0:{seg_dur_s:.6f},asetpts=PTS-STARTPTS[aorig]"

            filter_chains.append(a_orig_chain)

            # Narration commentary mix
            final_audio_label = "[aorig]"
            if narr_input_idx is not None:
                a_narr_chain = (
                    f"[{narr_input_idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
                    f"volume={comm_linear:.6f}:eval=once,"
                    f"apad=whole_dur={seg_dur_s:.6f},atrim=0:{seg_dur_s:.6f},asetpts=PTS-STARTPTS[anarr]"
                )
                filter_chains.append(a_narr_chain)
                filter_chains.append("[aorig][anarr]amix=inputs=2:duration=first:normalize=0[aout]")
                final_audio_label = "[aout]"

            filter_complex = ";".join(filter_chains)
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-map", final_audio_label,
                *v_args,
                "-c:a", "pcm_s16le",
                str(seg_mkv),
            ])

            try:
                run_command(cmd, timeout=max(120.0, seg_dur_s * 20.0), cancellation_token=cancellation_token, check=True)
            except CancelledError:
                raise
            except Exception as exc:
                if enc_status.encoder == "h264_nvenc" and not _hardware_fallback_attempted:
                    emit(WorkflowStage.RENDER,state=ActivityState.RETRYING.value,
                         activity_text="NVIDIA segment processing failed; retrying the complete output on CPU.",
                         current_item=seg_id,gpu_fallback_reason=str(exc)[-1000:])
                    shutil.rmtree(temp_dir,ignore_errors=True)
                    return render_output(
                        output_def,source_paths,output_dir,settings=replace(cfg,use_gpu=False),
                        voice_adapter=voice_adapter,cancellation_token=cancellation_token,
                        narration_audio_map=narration_audio_map,source_dialogue_map=source_dialogue_map,
                        ffmpeg_path=ffmpeg,work_dir=work_dir,progress_callback=progress_callback,
                        publication_owner=publication_owner,
                        publication_journal_root=publication_journal_root,
                        allow_publication_overwrite=allow_publication_overwrite,
                        audio_stream_indices=audio_stream_indices,_hardware_fallback_attempted=True,
                        render_mode=render_mode,
                    )
                raise
            if not seg_mkv.is_file() or seg_mkv.stat().st_size == 0:
                raise RenderError(f"Failed to produce intermediate segment MKV for segment '{seg_id}'.")
            segment_mkv_paths.append(seg_mkv)
            emit(
                WorkflowStage.AUDIO_MIX,
                state=ActivityState.LOCAL_PROCESSING.value,
                activity_text=(
                    f"Original-audio segment {seg_idx + 1} / {len(segments)} prepared."
                    if is_highlight else f"Mixed segment {seg_idx + 1} / {len(segments)}."
                ),
                current_item=seg_id, completed=seg_idx + 1, total=len(segments), unit="segments",
            )

        emit(
            WorkflowStage.AUDIO_MIX,
            state=ActivityState.LOCAL_PROCESSING.value,
            activity_text=(
                "Original-audio segment assembly ready."
                if is_highlight else "Audio mix and segment assembly ready."
            ),
            current_item=render_id, completed=len(segments), total=len(segments),
            unit="segments", stage_status="complete",
        )

        # Concatenate all segment MKVs via concat demuxer
        concat_list_file = temp_dir / "concat_list.txt"
        concat_lines = ["file '" + p.as_posix().replace("'", "'\\''") + "'" for p in segment_mkv_paths]
        concat_list_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        assembled_mkv = temp_dir / "assembled.mkv"
        run_command(
            [
                str(ffmpeg),
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                str(assembled_mkv),
            ],
            timeout=max(120.0, final_timeline_offset_ms / 1000 * 4.0),
            cancellation_token=cancellation_token,
            check=True,
        )

        # Subtitle generation into temp_dir
        subtitle_output = copy.deepcopy(output_def)
        subtitle_output["_narration_group_durations"] = group_durations
        for subtitle_segment in subtitle_output.get("segments", []):
            duration = effective_segment_durations.get(subtitle_segment.get("segment_id"))
            if duration is not None:
                subtitle_segment["end_ms"] = int(subtitle_segment.get("start_ms", 0)) + duration
        emit(
            WorkflowStage.SUBTITLES,
            state=ActivityState.LOCAL_PROCESSING.value,
            activity_text="Mapping original dialogue...",
            current_item=render_id, completed=0, total=1, unit="outputs",
        )
        if source_dialogue_map is None:
            legacy_original_cues = tuple(extract_subtitles_for_output(subtitle_output)[1])
            original_mapping = OriginalDialogueMapping(
                legacy_original_cues,
                ORIGINAL_DIALOGUE_MAPPED if legacy_original_cues else NO_ORIGINAL_DIALOGUE,
                len(legacy_original_cues),
            )
        else:
            original_mapping = OriginalDialogueSubtitleMapper(source_dialogue_map).map(resolved_source_clips)
        if original_mapping.state == ORIGINAL_SUBTITLE_GENERATION_FAILED:
            raise RenderValidationError(
                f"Verified source dialogue exists but no publication cues were generated for '{render_id}'."
            )
        temp_narr_srt, temp_orig_srt = generate_subtitles(
            subtitle_output, temp_dir, title=title, original_cues=original_mapping.cues,
        )
        if original_mapping.state == ORIGINAL_DIALOGUE_MAPPED and (
            not temp_orig_srt.is_file() or not temp_orig_srt.read_text(encoding="utf-8").strip()
        ):
            raise RenderValidationError(
                f"Original subtitle publication is empty despite verified retained dialogue for '{render_id}'."
            )
        narration_cue_count = len(extract_subtitles_for_output(subtitle_output)[0])
        narration_expected = any(
            segment.get("type") == "narration" and str(segment.get("narration", "")).strip()
            for segment in subtitle_output.get("segments", [])
        )
        if narration_expected and (
            narration_cue_count == 0 or not temp_narr_srt.read_text(encoding="utf-8").strip()
        ):
            raise RenderValidationError(f"Narration exists but narration SRT is empty for '{render_id}'.")
        emit(
            WorkflowStage.SUBTITLES,
            state=ActivityState.LOCAL_PROCESSING.value,
            activity_text=(f"Mapping original dialogue... Narration cues: {narration_cue_count}; "
                           f"Original dialogue cues: {len(original_mapping.cues)}"),
            current_item=render_id, completed=1, total=1, unit="outputs", stage_status="complete",
        )

        # Recap uses publication loudness normalization. Highlight preserves
        # source loudness unless the creator explicitly enables normalization.
        normalize_loudness = not is_highlight or bool(cfg.highlight_normalize_loudness)
        loudnorm_filter: str | None = None
        if normalize_loudness:
            measured_params = measure_loudnorm(
                assembled_mkv,
                target_lufs=cfg.target_loudness_lufs,
                true_peak=cfg.true_peak_db,
                ffmpeg_path=ffmpeg,
                cancellation_token=cancellation_token,
            )
            loudnorm_filter = build_two_pass_loudnorm_filter(
                measured_params,
                target_lufs=cfg.target_loudness_lufs,
                true_peak=cfg.true_peak_db,
            )

        # Subtitle burning check
        burn_subtitles = bool(cfg.burn_subtitles)
        video_final_args: list[str] = []

        if burn_subtitles:
            # Check which subtitle file to burn
            narr_has_cues = temp_narr_srt.is_file() and len(temp_narr_srt.read_text(encoding="utf-8").strip()) > 0
            orig_has_cues = temp_orig_srt.is_file() and len(temp_orig_srt.read_text(encoding="utf-8").strip()) > 0
            burn_file = temp_narr_srt if narr_has_cues else (temp_orig_srt if orig_has_cues else None)

            if burn_file:
                esc_path = escape_ffmpeg_subtitles_path(burn_file)
                video_final_args = ["-vf", f"subtitles='{esc_path}'", *v_args]
            else:
                video_final_args = list(v_args)
        else:
            video_final_args = list(v_args)

        # Pass 2: Final encode to temp_final_mp4
        temp_final_mp4 = temp_dir / f"{title}.mp4"
        encoder_label = getattr(enc_status, "label", None) or getattr(enc_status, "encoder", None) or "FFmpeg"
        emit(
            WorkflowStage.RENDER,
            state=ActivityState.LOCAL_PROCESSING.value,
            activity_text=f"Encoding final video with {encoder_label}...",
            current_item=render_id,
        )
        final_decode_args = (
            nvidia_decode_args("h264", enc_status)
            if enc_status.encoder == "h264_nvenc" and not burn_subtitles else []
        )
        try:
            audio_metadata_args = (
                ["-metadata:s:a:0", f"language={next(iter(output_audio_languages))}"]
                if len(output_audio_languages) == 1 else []
            )
            audio_filter_args = ["-af", loudnorm_filter] if loudnorm_filter else []
            run_command([
                str(ffmpeg),
                "-y",
                *final_decode_args,
                "-i", str(assembled_mkv),
                *video_final_args,
                *audio_filter_args,
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                *audio_metadata_args,
                str(temp_final_mp4),
            ], timeout=max(180.0, final_timeline_offset_ms / 1000 * 20.0), cancellation_token=cancellation_token, check=True)
        except CancelledError:
            raise
        except Exception as exc:
            if enc_status.encoder == "h264_nvenc" and not _hardware_fallback_attempted:
                emit(WorkflowStage.RENDER,state=ActivityState.RETRYING.value,
                     activity_text="NVIDIA final encoding failed; retrying the complete output on CPU.",
                     current_item=render_id,gpu_fallback_reason=str(exc)[-1000:])
                shutil.rmtree(temp_dir,ignore_errors=True)
                return render_output(
                    output_def,source_paths,output_dir,settings=replace(cfg,use_gpu=False),
                    voice_adapter=voice_adapter,cancellation_token=cancellation_token,
                    narration_audio_map=narration_audio_map,source_dialogue_map=source_dialogue_map,
                    ffmpeg_path=ffmpeg,work_dir=work_dir,progress_callback=progress_callback,
                    publication_owner=publication_owner,
                    publication_journal_root=publication_journal_root,
                    allow_publication_overwrite=allow_publication_overwrite,
                    audio_stream_indices=audio_stream_indices,_hardware_fallback_attempted=True,
                    render_mode=render_mode,
                )
            raise

        # Technical validation of rendered output before publication
        if not temp_final_mp4.is_file() or temp_final_mp4.stat().st_size == 0:
            raise RenderValidationError("Rendered output MP4 is missing or empty.")

        final_probe = probe_media(temp_final_mp4, cancellation_token=cancellation_token)
        if final_probe.duration <= 0.0:
            raise RenderValidationError(f"Rendered output has invalid duration: {final_probe.duration}s")
        if not final_probe.has_video:
            raise RenderValidationError("Rendered output has no video stream.")
        if not final_probe.has_audio:
            raise RenderValidationError("Rendered output has no audio stream.")
        if final_probe.width != canvas_w or final_probe.height != canvas_h:
            raise RenderValidationError(
                f"Rendered canvas mismatch: expected {canvas_w}x{canvas_h}, got {final_probe.width}x{final_probe.height}"
            )

        # Stage all publication files in the target directory, then atomically replace each target.
        published_files: list[tuple[Path, Path]] = [
            (temp_final_mp4, target_mp4),
            (temp_narr_srt, target_narr_srt),
            (temp_orig_srt, target_orig_srt),
        ]
        emit(
            WorkflowStage.PUBLISH,
            state=ActivityState.LOCAL_PROCESSING.value,
            activity_text="Publishing output files atomically...",
            current_item=render_id,
        )
        if publication_owner:
            from toolrecap_v4.publication import publish
            publish(published_files, publication_owner, cancellation_token, publication_journal_root,
                    allow_existing=allow_publication_overwrite)
        else:
            _stage_and_publish_files(published_files, cancellation_token=cancellation_token)
        emit(
            WorkflowStage.PUBLISH,
            state=ActivityState.LOCAL_PROCESSING.value,
            activity_text="Output published.",
            current_item=render_id, completed=1, total=1, unit="outputs", stage_status="complete",
        )

        v_codec = final_probe.video_streams[0].codec if final_probe.video_streams else "h264"
        a_codec = final_probe.audio_streams[0].codec if final_probe.audio_streams else "aac"

        return RenderResult(
            output_path=target_mp4,
            narration_srt_path=target_narr_srt,
            original_srt_path=target_orig_srt,
            duration=final_probe.duration,
            title=title,
            render_id=render_id,
            video_codec=v_codec,
            audio_codec=a_codec,
            width=final_probe.width,
            height=final_probe.height,
            fps=final_probe.fps,
            original_subtitle_state=original_mapping.state,
            original_subtitle_cue_count=len(original_mapping.cues),
        )

    finally:
        # Guarantee cleanup of internal temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def render_project(
    project_data: dict[str, Any],
    source_paths: dict[str, Path | str],
    output_dir: Path | str,
    *,
    settings: AppSettings | None = None,
    voice_adapter: VoiceStudioAdapter | None = None,
    cancellation_token: CancellationToken | None = None,
    narration_audio_map: dict[str, Path | str] | None = None,
    source_dialogue_map: dict[str, Sequence[Any]] | None = None,
    ffmpeg_path: Path | str | None = None,
    work_dir: Path | str | None = None,
) -> list[RenderResult]:
    """Render all outputs defined in a project JSON specification.
    
    Guarantees input project_data is completely unchanged.
    """
    if cancellation_token:
        cancellation_token.check_cancelled()

    # Invariant: JSON unchanged
    json_snapshot = json.dumps(project_data, sort_keys=True)

    outputs = project_data.get("outputs", [])
    if not outputs:
        raise RenderError("Project data contains no outputs to render.")

    results: list[RenderResult] = []
    for out_def in outputs:
        res = render_output(
            output_def=out_def,
            source_paths=source_paths,
            output_dir=output_dir,
            settings=settings,
            voice_adapter=voice_adapter,
            cancellation_token=cancellation_token,
            narration_audio_map=narration_audio_map,
            source_dialogue_map=source_dialogue_map,
            ffmpeg_path=ffmpeg_path,
            work_dir=work_dir,
        )
        results.append(res)

    # Invariant verification: JSON unchanged
    if json.dumps(project_data, sort_keys=True) != json_snapshot:
        raise RenderError("Input project JSON was mutated during render_project execution!")

    return results
