"""Media processing primitives: process runner, probe, audio selector, and GPU detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Sequence

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, ToolRecapError
from toolrecap_v4.runtime import application_root, bundled_runtime_root


class MediaError(ToolRecapError):
    """Base exception for media operations."""


class BinaryNotFoundError(MediaError):
    """Raised when required external binary (FFmpeg, FFprobe) is not found."""


class ProbeError(MediaError):
    """Raised when media probing fails."""


class SubprocessError(MediaError):
    """Raised when external subprocess execution fails."""


class SubprocessTimeoutError(SubprocessError):
    """Raised when subprocess execution exceeds specified timeout."""


@dataclass(frozen=True)
class CommandResult:
    """Result of subprocess command execution."""

    exit_code: int
    stdout: str
    stderr: str


def run_command(
    args: Sequence[str],
    *,
    timeout: float | None = 60.0,
    cancellation_token: CancellationToken | None = None,
    creationflags: int | None = None,
    check: bool = False,
) -> CommandResult:
    """Execute a subprocess command with finite timeout, cancellation, and cleanup.
    
    Invariants:
    - Safe argument array: accepts strictly a sequence of string arguments.
    - Finite timeout: enforces timeout and terminates child on expiry.
    - Cooperative cancellation: registers with CancellationToken and cleans up child tree.
    """
    if isinstance(args, (str, bytes)):
        raise TypeError("args must be a sequence of strings, not a string or bytes")

    arg_list = [str(arg) for arg in args]
    if not arg_list:
        raise ValueError("args sequence cannot be empty")

    if cancellation_token is not None:
        cancellation_token.check_cancelled()

    if creationflags is None and sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0
        )
    elif creationflags is None:
        creationflags = 0

    process = subprocess.Popen(
        arg_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )

    def _cleanup_child() -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        except Exception:
            pass

    if cancellation_token is not None:
        cancellation_token.register_callback(_cleanup_child)

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _cleanup_child()
        if cancellation_token is not None and cancellation_token.is_cancelled:
            raise CancelledError("Subprocess cancelled.") from exc
        raise SubprocessTimeoutError(
            f"Process timed out after {timeout}s: {' '.join(arg_list[:5])}"
        ) from exc
    except Exception as exc:
        _cleanup_child()
        if cancellation_token is not None and cancellation_token.is_cancelled:
            raise CancelledError("Subprocess cancelled.") from exc
        raise

    if cancellation_token is not None and cancellation_token.is_cancelled:
        raise CancelledError("Subprocess cancelled.")

    result = CommandResult(
        exit_code=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
    )

    if check and result.exit_code != 0:
        raise SubprocessError(
            f"Command failed with exit code {result.exit_code}: {result.stderr.strip()}"
        )

    return result


def find_binary(
    name: str,
    custom_path: Path | str | None = None,
) -> Path:
    """Find executable binary via custom path, env var, bundled runtime, or system PATH.
    
    Discovers binaries dynamically without hardcoded developer paths.
    """
    if custom_path is not None:
        p = Path(custom_path)
        if p.is_file():
            return p.resolve()
        if p.is_dir():
            candidate = p / name
            if candidate.is_file():
                return candidate.resolve()
            candidate_exe = p / f"{name}.exe"
            if candidate_exe.is_file():
                return candidate_exe.resolve()

    # Environment variables check
    env_keys = [
        f"TOOLRECAP_{name.upper()}_PATH",
        f"{name.upper()}_PATH",
    ]
    for key in env_keys:
        env_val = os.environ.get(key)
        if env_val:
            ep = Path(env_val)
            if ep.is_file():
                return ep.resolve()

    # Bundled runtime lookup is relocation-safe and never trusts process CWD.
    base_dirs = [application_root(), bundled_runtime_root()]
    suffix = ".exe" if sys.platform == "win32" else ""
    for base in base_dirs:
        candidates = [
            base / f"{name}{suffix}",
            base / "bin" / f"{name}{suffix}",
            base / "runtime" / "ffmpeg" / "bin" / f"{name}{suffix}",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

    # System PATH lookup
    system_match = shutil.which(name)
    if not system_match and sys.platform == "win32":
        system_match = shutil.which(f"{name}.exe")
    if system_match:
        return Path(system_match).resolve()

    raise BinaryNotFoundError(
        f"Required binary '{name}' not found. Ensure FFmpeg is installed or in PATH/TOOLRECAP_{name.upper()}_PATH."
    )


def calculate_aspect_ratio(width: int, height: int) -> str:
    """Calculate simplified aspect ratio string (e.g. 16:9)."""
    if width <= 0 or height <= 0:
        return "unknown"
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _parse_rotation(stream: dict[str, Any]) -> int:
    """Extract rotation angle in degrees from stream side data or tags."""
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            try:
                return int(float(side_data["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360
        except (TypeError, ValueError):
            pass
    return 0


def _parse_frame_rate(stream: dict[str, Any]) -> tuple[float, str]:
    """Extract frame rate as (fps_float, fps_text) from stream."""
    rate_text = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "25/1")
    if "/" in rate_text:
        num_str, _, den_str = rate_text.partition("/")
        try:
            num, den = float(num_str), float(den_str)
            if den > 0 and num > 0:
                return num / den, rate_text
        except ValueError:
            pass
    return 25.0, rate_text


@dataclass(frozen=True)
class VideoStreamInfo:
    """Metadata for a probed video stream."""

    index: int
    codec: str
    width: int
    height: int
    fps: float
    fps_text: str
    duration: float
    aspect_ratio: str
    rotation: int = 0
    bitrate: int | None = None
    sar: str = ""
    dar: str = ""
    pixel_format: str = ""
    avg_frame_rate: str = ""
    real_frame_rate: str = ""
    time_base: str = ""
    start_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioStreamInfo:
    """Metadata for a probed audio stream."""

    index: int
    codec: str
    channels: int
    sample_rate: int
    language: str = ""
    title: str = ""
    handler_name: str = ""
    disposition: dict[str, int] = field(default_factory=dict)
    duration: float | None = None
    bitrate: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MediaProbeResult:
    """Probe result for a single media source file."""

    path: Path
    duration: float
    container: str
    video_streams: tuple[VideoStreamInfo, ...]
    audio_streams: tuple[AudioStreamInfo, ...]
    width: int = 0
    height: int = 0
    fps: float = 0.0
    aspect_ratio: str = ""
    sar: str = ""
    dar: str = ""
    has_video: bool = False
    has_audio: bool = False
    start_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration": self.duration,
            "container": self.container,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "aspect_ratio": self.aspect_ratio,
            "sar": self.sar,
            "dar": self.dar,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "start_time": self.start_time,
            "video_streams": [s.to_dict() for s in self.video_streams],
            "audio_streams": [s.to_dict() for s in self.audio_streams],
        }

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def probe_media(
    path: Path | str,
    *,
    ffprobe_path: Path | str | None = None,
    timeout: float = 60.0,
    cancellation_token: CancellationToken | None = None,
) -> MediaProbeResult:
    """Probe a single media source independently using ffprobe."""
    media_path = Path(path).resolve()
    if not media_path.is_file():
        raise MediaError(f"Media source not found: {media_path}")

    binary = find_binary("ffprobe", ffprobe_path)
    cmd = [
        str(binary),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(media_path),
    ]

    result = run_command(cmd, timeout=timeout, cancellation_token=cancellation_token)
    if result.exit_code != 0:
        raise ProbeError(f"ffprobe failed (exit {result.exit_code}): {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"Failed to parse ffprobe output as JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ProbeError("ffprobe JSON output is not a dictionary")

    fmt = data.get("format") or {}
    container = str(fmt.get("format_name", "unknown"))

    raw_duration = fmt.get("duration")
    duration = 0.0
    if raw_duration:
        try:
            duration = float(raw_duration)
        except (ValueError, TypeError):
            duration = 0.0
    try:
        format_start_time = float(fmt.get("start_time") or 0.0)
    except (TypeError, ValueError):
        format_start_time = 0.0

    raw_streams = data.get("streams") or []
    video_streams_list: list[VideoStreamInfo] = []
    audio_streams_list: list[AudioStreamInfo] = []

    for s in raw_streams:
        if not isinstance(s, dict):
            continue
        codec_type = s.get("codec_type")
        stream_index = int(s.get("index", 0))
        codec = str(s.get("codec_name", "unknown"))

        stream_dur = s.get("duration")
        s_duration = duration
        if stream_dur:
            try:
                s_duration = float(stream_dur)
            except (ValueError, TypeError):
                pass

        if codec_type == "video":
            w = int(s.get("width", 0))
            h = int(s.get("height", 0))
            fps, fps_text = _parse_frame_rate(s)
            ar = calculate_aspect_ratio(w, h)
            rot = _parse_rotation(s)
            bitrate = int(s["bit_rate"]) if "bit_rate" in s and str(s["bit_rate"]).isdigit() else None
            sar = str(s.get("sample_aspect_ratio") or "")
            dar = str(s.get("display_aspect_ratio") or "")
            pixel_format = str(s.get("pix_fmt") or "")
            try:
                stream_start_time = float(s.get("start_time") or format_start_time)
            except (TypeError, ValueError):
                stream_start_time = format_start_time
            video_streams_list.append(
                VideoStreamInfo(
                    index=stream_index,
                    codec=codec,
                    width=w,
                    height=h,
                    fps=fps,
                    fps_text=fps_text,
                    duration=s_duration,
                    aspect_ratio=ar,
                    rotation=rot,
                    bitrate=bitrate,
                    sar=sar,
                    dar=dar,
                    pixel_format=pixel_format,
                    avg_frame_rate=str(s.get("avg_frame_rate") or ""),
                    real_frame_rate=str(s.get("r_frame_rate") or ""),
                    time_base=str(s.get("time_base") or ""),
                    start_time=stream_start_time,
                )
            )
        elif codec_type == "audio":
            channels = int(s.get("channels", 0))
            sample_rate = int(s.get("sample_rate", 0))
            tags = s.get("tags") or {}
            normalized_tags = {str(k).lower(): str(v) for k, v in tags.items()}
            lang = normalized_tags.get("language") or normalized_tags.get("lang") or ""
            title = normalized_tags.get("title") or normalized_tags.get("name") or ""
            handler = normalized_tags.get("handler_name") or ""
            disposition = s.get("disposition") or {}
            bitrate = int(s["bit_rate"]) if "bit_rate" in s and str(s["bit_rate"]).isdigit() else None
            audio_streams_list.append(
                AudioStreamInfo(
                    index=stream_index,
                    codec=codec,
                    channels=channels,
                    sample_rate=sample_rate,
                    language=lang,
                    title=title,
                    handler_name=handler,
                    disposition=disposition,
                    duration=s_duration,
                    bitrate=bitrate,
                )
            )

    primary_w = video_streams_list[0].width if video_streams_list else 0
    primary_h = video_streams_list[0].height if video_streams_list else 0
    primary_fps = video_streams_list[0].fps if video_streams_list else 0.0
    primary_ar = video_streams_list[0].aspect_ratio if video_streams_list else ""
    primary_sar = video_streams_list[0].sar if video_streams_list else ""
    primary_dar = video_streams_list[0].dar if video_streams_list else ""

    if duration <= 0.0 and video_streams_list:
        duration = video_streams_list[0].duration
    elif duration <= 0.0 and audio_streams_list:
        duration = audio_streams_list[0].duration or 0.0

    return MediaProbeResult(
        path=media_path,
        duration=duration,
        container=container,
        video_streams=tuple(video_streams_list),
        audio_streams=tuple(audio_streams_list),
        width=primary_w,
        height=primary_h,
        fps=primary_fps,
        aspect_ratio=primary_ar,
        sar=primary_sar,
        dar=primary_dar,
        has_video=len(video_streams_list) > 0,
        has_audio=len(audio_streams_list) > 0,
        start_time=format_start_time,
    )


def _rate_value(value: str) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            number = float(numerator)
            divisor = float(denominator)
            return number / divisor if number > 0 and divisor > 0 else None
        number = float(value)
        return number if number > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


@dataclass(frozen=True)
class SourceTimelineAssessment:
    """Editing-compatibility assessment for frame-accurate cutting."""

    status: str
    requires_cfr_output: bool
    reasons: tuple[str, ...]
    recommended_fps: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_source_timeline(probe: MediaProbeResult) -> SourceTimelineAssessment:
    """Detect timestamp/frame-rate traits that can cause one-frame edit jumps.

    ToolRecap always preserves the original source.  Risky sources are marked so
    downstream Highlight publication can force a normalized CFR timeline.
    """
    if not probe.video_streams:
        return SourceTimelineAssessment("no_video", False, ("No video stream was found.",), "")
    stream = probe.video_streams[0]
    reasons: list[str] = []
    avg = _rate_value(stream.avg_frame_rate or stream.fps_text)
    real = _rate_value(stream.real_frame_rate)
    if avg and real and abs(avg - real) > max(0.01, avg * 0.001):
        reasons.append(
            f"Frame-rate metadata differs (avg={stream.avg_frame_rate}, r={stream.real_frame_rate}); VFR is unconfirmed without frame timestamp inspection."
        )
    if abs(float(stream.start_time or probe.start_time)) > 0.001:
        reasons.append(f"Video timeline starts at {stream.start_time or probe.start_time:.6f}s instead of zero.")
    recommended = stream.avg_frame_rate or stream.fps_text or stream.real_frame_rate
    return SourceTimelineAssessment(
        "needs_frame_inspection" if reasons else "compatible",
        bool(abs(float(stream.start_time or probe.start_time)) > 0.001),
        tuple(reasons),
        recommended,
    )


def inspect_frame_timing(
    path: Path | str, duration_ms: int, *, ffprobe_path: Path | str | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """Sample actual frame timestamps at the start, middle, and end.

    Sampling can identify irregular timing but cannot certify an uninspected
    portion of a long episode as CFR.
    """
    binary = find_binary("ffprobe", ffprobe_path)
    starts = sorted({0, max(0, duration_ms // 2 - 5000), max(0, duration_ms - 10_000)})
    windows = []
    variable = False
    for start_ms in starts:
        result = run_command([
            str(binary), "-v", "error", "-select_streams", "v:0",
            "-read_intervals", f"{start_ms / 1000:.3f}%+10",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "csv=p=0", str(path),
        ], timeout=60.0, cancellation_token=cancellation_token)
        if result.exit_code != 0:
            raise ProbeError(f"Frame timestamp inspection failed: {result.stderr.strip()}")
        stamps = []
        for line in result.stdout.splitlines():
            try:
                stamps.append(float(line.split(",", 1)[0]))
            except ValueError:
                continue
        intervals = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
        typical = median(intervals) if intervals else None
        irregular = bool(typical and any(abs(delta - typical) > typical * 0.2 for delta in intervals))
        variable |= irregular
        windows.append({"start_ms": start_ms, "frames": len(stamps),
                        "median_frame_ms": round(typical * 1000, 3) if typical else None,
                        "irregular": irregular})
    return {"sampled_windows": windows, "irregular_timing_observed": variable,
            "coverage": "sampled only; other frame timestamps were not inspected"}


def calculate_auto_canvas(
    probe: MediaProbeResult,
    fallback_w: int = 1920,
    fallback_h: int = 1080,
) -> tuple[int, int]:
    """Calculate auto canvas dimensions from media probe.
    
    Invariants:
    - Accounts for DAR (Display Aspect Ratio) and non-square pixels (SAR) for square pixels.
    - Accounts for stream rotation (90/270 degrees swap width and height).
    - Guarantees even dimensions (divisible by 2) for H.264/yuv420p encoding.
    - Fallback to fallback_w/h if probe has no valid video.
    """
    if not probe.has_video or not probe.video_streams:
        w = fallback_w + (fallback_w % 2)
        h = fallback_h + (fallback_h % 2)
        return w, h

    v_stream = probe.video_streams[0]
    w = v_stream.width
    h = v_stream.height
    if w <= 0 or h <= 0:
        w = fallback_w + (fallback_w % 2)
        h = fallback_h + (fallback_h % 2)
        return w, h

    # 1. Square-pixel normalization from DAR or SAR
    dar = v_stream.dar or probe.dar
    sar = v_stream.sar or probe.sar
    w_disp = float(w)
    h_disp = float(h)

    if dar and dar not in ("0:1", "0/1", "unknown"):
        sep = "/" if "/" in dar else (":" if ":" in dar else None)
        if sep:
            try:
                num_s, den_s = dar.split(sep, 1)
                num, den = float(num_s), float(den_s)
                if num > 0 and den > 0:
                    dar_val = num / den
                    w_disp = round(h_disp * dar_val)
            except (ValueError, ZeroDivisionError):
                pass
    elif sar and sar not in ("0:1", "0/1", "1:1", "1/1", "unknown"):
        sep = "/" if "/" in sar else (":" if ":" in sar else None)
        if sep:
            try:
                num_s, den_s = sar.split(sep, 1)
                num, den = float(num_s), float(den_s)
                if num > 0 and den > 0:
                    sar_val = num / den
                    w_disp = round(w_disp * sar_val)
            except (ValueError, ZeroDivisionError):
                pass

    # 2. Rotation handling (90 and 270 degrees swap width and height)
    rot = v_stream.rotation % 360
    if rot in (90, 270):
        canvas_w = int(round(h_disp))
        canvas_h = int(round(w_disp))
    else:
        canvas_w = int(round(w_disp))
        canvas_h = int(round(h_disp))

    # 3. Even dimensions for H.264 / yuv420p
    if canvas_w % 2 != 0:
        canvas_w += 1
    if canvas_h % 2 != 0:
        canvas_h += 1

    return canvas_w, canvas_h


def probe_duration(
    path: Path | str,
    *,
    ffprobe_path: Path | str | None = None,
    timeout: float = 30.0,
    cancellation_token: CancellationToken | None = None,
) -> float:
    """Fast probe of format duration only."""
    media_path = Path(path).resolve()
    if not media_path.is_file():
        raise MediaError(f"Media source not found: {media_path}")

    binary = find_binary("ffprobe", ffprobe_path)
    cmd = [
        str(binary),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    result = run_command(cmd, timeout=timeout, cancellation_token=cancellation_token)
    if result.exit_code != 0:
        raise ProbeError(f"ffprobe duration failed (exit {result.exit_code}): {result.stderr.strip()}")
    try:
        return float(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError) as exc:
        raise ProbeError(f"Unable to parse duration from ffprobe: {result.stdout}") from exc


@dataclass(frozen=True)
class AudioSelectionResult:
    """Result of audio stream selection."""

    selected_stream: AudioStreamInfo | None
    selected_index: int | None
    warning: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_stream": self.selected_stream.to_dict() if self.selected_stream else None,
            "selected_index": self.selected_index,
            "warning": self.warning,
            "reason": self.reason,
        }


def is_english_language(lang: str | None) -> bool:
    """Return True if language code represents English (eng, en, en-US, en-GB, etc.)."""
    if not lang:
        return False
    clean = lang.strip().lower().replace("_", "-")
    if clean in ("en", "eng", "english"):
        return True
    if clean.startswith("en-"):
        return True
    return False


def is_commentary_or_descriptive(stream: AudioStreamInfo | dict[str, Any]) -> bool:
    """Return True if audio stream represents commentary or descriptive audio."""
    if isinstance(stream, AudioStreamInfo):
        disp = stream.disposition
        title = stream.title
        handler = stream.handler_name
    else:
        disp = stream.get("disposition") or {}
        tags = stream.get("tags") or {}
        norm_tags = {str(k).lower(): str(v) for k, v in tags.items()}
        title = norm_tags.get("title") or stream.get("title") or ""
        handler = norm_tags.get("handler_name") or stream.get("handler_name") or ""

    if (
        disp.get("commentary", 0) == 1
        or disp.get("descriptions", 0) == 1
        or disp.get("visual_impaired", 0) == 1
        or disp.get("hearing_impaired", 0) == 1
    ):
        return True

    combined = f"{title} {handler}".lower()
    keywords = [
        "commentary",
        "director",
        "descriptive",
        "description",
        "visual impaired",
        "audio description",
        "dvs",
    ]
    for kw in keywords:
        if kw in combined:
            return True

    return False


def is_main_audio(stream: AudioStreamInfo | dict[str, Any]) -> bool:
    """Return True if stream title/handler indicates main/primary dialogue."""
    if isinstance(stream, AudioStreamInfo):
        title = stream.title
        handler = stream.handler_name
    else:
        tags = stream.get("tags") or {}
        norm_tags = {str(k).lower(): str(v) for k, v in tags.items()}
        title = norm_tags.get("title") or stream.get("title") or ""
        handler = norm_tags.get("handler_name") or stream.get("handler_name") or ""

    combined = f"{title} {handler}".lower()
    main_keywords = ["main", "dialogue", "primary", "feature"]
    for kw in main_keywords:
        if kw in combined:
            return True
    return False


def _to_audio_stream_info(s: AudioStreamInfo | dict[str, Any]) -> AudioStreamInfo:
    if isinstance(s, AudioStreamInfo):
        return s
    tags = s.get("tags") or {}
    norm_tags = {str(k).lower(): str(v) for k, v in tags.items()}
    return AudioStreamInfo(
        index=int(s.get("index", 0)),
        codec=str(s.get("codec_name", s.get("codec", "unknown"))),
        channels=int(s.get("channels", 0)),
        sample_rate=int(s.get("sample_rate", 0)),
        language=norm_tags.get("language") or norm_tags.get("lang") or s.get("language", ""),
        title=norm_tags.get("title") or norm_tags.get("name") or s.get("title", ""),
        handler_name=norm_tags.get("handler_name") or s.get("handler_name", ""),
        disposition=s.get("disposition") or {},
        duration=float(s["duration"]) if "duration" in s and s["duration"] is not None else None,
        bitrate=int(s["bit_rate"]) if "bit_rate" in s and str(s["bit_rate"]).isdigit() else None,
    )


def select_audio_stream(
    streams: Sequence[AudioStreamInfo | dict[str, Any]],
) -> AudioSelectionResult:
    """Select the best audio stream using metadata-only inspection.
    
    Priority:
    1. English main excluding commentary/descriptive
    2. English normal/default excluding commentary/descriptive
    3. Default or first audio track with warning
    """
    if not streams:
        return AudioSelectionResult(
            selected_stream=None,
            selected_index=None,
            warning="No audio streams found in media source.",
            reason="no_audio",
        )

    parsed_streams = [_to_audio_stream_info(s) for s in streams]

    english_candidates = [
        s for s in parsed_streams
        if is_english_language(s.language) and not is_commentary_or_descriptive(s)
    ]

    if english_candidates:
        main_english = [s for s in english_candidates if is_main_audio(s)]
        if main_english:
            default_main = [s for s in main_english if s.disposition.get("default", 0) == 1]
            selected = default_main[0] if default_main else main_english[0]
            return AudioSelectionResult(
                selected_stream=selected,
                selected_index=selected.index,
                warning=None,
                reason="english_main",
            )

        default_english = [s for s in english_candidates if s.disposition.get("default", 0) == 1]
        if default_english:
            selected = default_english[0]
            return AudioSelectionResult(
                selected_stream=selected,
                selected_index=selected.index,
                warning=None,
                reason="english_default",
            )

        selected = english_candidates[0]
        return AudioSelectionResult(
            selected_stream=selected,
            selected_index=selected.index,
            warning=None,
            reason="english_normal",
        )

    default_tracks = [s for s in parsed_streams if s.disposition.get("default", 0) == 1]
    if default_tracks:
        selected = default_tracks[0]
        lang_str = selected.language or "unknown"
        warning = (
            f"No suitable English audio track found; falling back to default audio track "
            f"(index {selected.index}, language '{lang_str}')."
        )
        return AudioSelectionResult(
            selected_stream=selected,
            selected_index=selected.index,
            warning=warning,
            reason="fallback_default",
        )

    selected = parsed_streams[0]
    lang_str = selected.language or "unknown"
    warning = (
        f"No suitable English or default audio track found; falling back to first audio track "
        f"(index {selected.index}, language '{lang_str}')."
    )
    return AudioSelectionResult(
        selected_stream=selected,
        selected_index=selected.index,
        warning=warning,
        reason="fallback_first",
    )


def select_audio_for_source(
    probe_result: MediaProbeResult | dict[str, Any],
) -> AudioSelectionResult:
    """Select the best audio stream for a probed media source."""
    if isinstance(probe_result, MediaProbeResult):
        streams = probe_result.audio_streams
    elif isinstance(probe_result, dict):
        streams = probe_result.get("audio_streams", [])
    else:
        raise TypeError(f"Expected MediaProbeResult or dict, got {type(probe_result)}")
    return select_audio_stream(streams)


def resolve_sources_audio(
    probe_results: Sequence[MediaProbeResult | dict[str, Any]],
) -> list[AudioSelectionResult]:
    """Resolve audio selection for a list of probed sources independently."""
    return [select_audio_for_source(p) for p in probe_results]


@dataclass(frozen=True)
class EncoderStatus:
    """Status of GPU / CPU video encoder."""

    available: bool
    gpu_name: str
    encoder: str
    label: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_cached_encoder_status: EncoderStatus | None = None
_encoder_cache_lock = threading.Lock()
_nvidia_source_probe_cache: dict[tuple[Any, ...], tuple[bool, str]] = {}
_nvidia_source_probe_lock = threading.Lock()


def get_gpu_name() -> str:
    """Detect GPU name via nvidia-smi or system query; fallback to generic label."""
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            res = subprocess.run(
                [nvidia, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass

    if sys.platform == "win32":
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
                if lines:
                    return lines[0]
        except Exception:
            pass

    return "System GPU"


def probe_encoder_usable(
    ffmpeg_binary: Path | str,
    encoder: str,
    *,
    timeout: float = 15.0,
) -> bool:
    """Perform actual encode test using FFmpeg on dummy input to verify encoder usability."""
    null_sink = "NUL" if sys.platform == "win32" else "/dev/null"
    cmd = [
        str(ffmpeg_binary),
        "-hide_banner",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=c=black:s=640x360:d=0.1:r=25",
        "-frames:v", "2",
        "-c:v", encoder,
        "-f", "null",
        null_sink,
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.returncode == 0
    except Exception:
        return False


test_encoder = probe_encoder_usable


def detect_gpu_encoder(
    ffmpeg_path: Path | str | None = None,
    *,
    refresh: bool = False,
) -> EncoderStatus:
    """Detect available and working GPU H.264 encoder via actual encode test."""
    global _cached_encoder_status
    with _encoder_cache_lock:
        if _cached_encoder_status is not None and not refresh:
            return _cached_encoder_status

        try:
            ffmpeg = find_binary("ffmpeg", ffmpeg_path)
        except BinaryNotFoundError as exc:
            status = EncoderStatus(
                available=False,
                gpu_name="unknown",
                encoder="libx264",
                label="CPU",
                reason=f"FFmpeg binary not found: {exc}",
            )
            _cached_encoder_status = status
            return status

        try:
            res = subprocess.run(
                [str(ffmpeg), "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            encoders_text = (res.stdout or "") + (res.stderr or "")
        except Exception as exc:
            status = EncoderStatus(
                available=False,
                gpu_name="unknown",
                encoder="libx264",
                label="CPU",
                reason=f"Failed to query FFmpeg encoders: {exc}",
            )
            _cached_encoder_status = status
            return status

        gpu_name = get_gpu_name()
        lowered_gpu = gpu_name.casefold()

        candidates: list[tuple[str, str]] = []
        if "nvidia" in lowered_gpu:
            candidates.append(("h264_nvenc", "NVIDIA NVENC"))
        elif "amd" in lowered_gpu or "radeon" in lowered_gpu:
            candidates.append(("h264_amf", "AMD AMF"))
        elif "intel" in lowered_gpu:
            candidates.append(("h264_qsv", "Intel Quick Sync"))

        candidates.extend([
            ("h264_nvenc", "NVIDIA NVENC"),
            ("h264_amf", "AMD AMF"),
            ("h264_qsv", "Intel Quick Sync"),
        ])

        checked: set[str] = set()
        for encoder_name, label in candidates:
            if encoder_name in checked:
                continue
            checked.add(encoder_name)
            if encoder_name not in encoders_text:
                continue
            if test_encoder(ffmpeg, encoder_name):
                status = EncoderStatus(
                    available=True,
                    gpu_name=gpu_name,
                    encoder=encoder_name,
                    label=label,
                    reason="",
                )
                _cached_encoder_status = status
                return status

        status = EncoderStatus(
            available=False,
            gpu_name=gpu_name,
            encoder="libx264",
            label="CPU",
            reason="No working GPU H.264 encoder found.",
        )
        _cached_encoder_status = status
        return status


def get_video_encode_args(
    quality: str = "high",
    *,
    use_gpu: bool = True,
    encoder_status: EncoderStatus | None = None,
    ffmpeg_path: Path | str | None = None,
) -> tuple[list[str], EncoderStatus]:
    """Return FFmpeg video encoding arguments matching desired quality and GPU availability."""
    status = encoder_status or detect_gpu_encoder(ffmpeg_path)
    if use_gpu and not status.available:
        raise MediaError(
            f"GPU video render was selected but no working hardware encoder is available: {status.reason}"
        )
    if use_gpu and status.available:
        if status.encoder == "h264_nvenc":
            cq, preset = {"standard": (23, "p4"), "high": (19, "p5"), "source": (17, "p6")}.get(
                quality, (19, "p5")
            )
            return [
                "-c:v", status.encoder,
                "-preset", preset,
                "-tune", "hq",
                "-rc", "vbr",
                "-cq", str(cq),
                "-b:v", "0",
            ], status
        if status.encoder == "h264_amf":
            qp = {"standard": "23", "high": "19", "source": "17"}.get(quality, "19")
            return [
                "-c:v", status.encoder,
                "-quality", "quality",
                "-rc", "cqp",
                "-qp_i", qp,
                "-qp_p", qp,
            ], status
        if status.encoder == "h264_qsv":
            quality_val = {"standard": "23", "high": "19", "source": "17"}.get(quality, "19")
            return [
                "-c:v", status.encoder,
                "-preset", "medium",
                "-global_quality", quality_val,
            ], status

    crf, preset = {"standard": (22, "veryfast"), "high": (18, "medium"), "source": (16, "slow")}.get(
        quality, (18, "medium")
    )
    cpu_status = EncoderStatus(
        available=False,
        gpu_name=status.gpu_name if status else "CPU",
        encoder="libx264",
        label="CPU",
        reason="CPU encoding selected",
    )
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)], cpu_status


NVIDIA_CUVID_DECODERS = {
    "h264": "h264_cuvid", "avc": "h264_cuvid",
    "hevc": "hevc_cuvid", "h265": "hevc_cuvid",
    "av1": "av1_cuvid", "vp9": "vp9_cuvid",
    "mpeg2video": "mpeg2_cuvid", "vc1": "vc1_cuvid",
}


def nvidia_decode_args(
    source_codec: str, encoder_status: EncoderStatus, pixel_format: str = "",
) -> list[str]:
    """Return explicit NVIDIA decode flags when the source codec supports CUVID."""
    if not encoder_status.available or encoder_status.encoder != "h264_nvenc":
        return []
    decoder = NVIDIA_CUVID_DECODERS.get(str(source_codec).casefold())
    if decoder is None:
        return []
    # NVDEC/CUVID does not support common H.264 4:4:4 profiles.  These inputs
    # decode on CPU, then upload once for CUDA scale/pad and NVENC encode.
    if "444" in str(pixel_format).casefold():
        return []
    return [
        "-hwaccel", "cuda", "-hwaccel_device", "0",
        "-hwaccel_output_format", "cuda", "-c:v", decoder,
    ]


def nvidia_scale_pad_filter(
    *, canvas_width: int, canvas_height: int, fps: float,
    source_duration: float, hold_duration: float = 0.0,
    hardware_input: bool = True,
) -> str:
    """GPU scale followed by one deliberate download for lightweight timing/pad filters."""
    hold = f",tpad=stop_mode=clone:stop_duration={hold_duration:.6f}" if hold_duration > 0 else ""
    upload = "" if hardware_input else "format=nv12,hwupload_cuda,"
    return (
        upload +
        "scale_cuda=w=trunc(ih*dar/2)*2:h=ih,"
        f"scale_cuda=w={canvas_width}:h={canvas_height}:force_original_aspect_ratio=decrease:"
        # A CUVID-decoded 10-bit source stays p010 unless scale_cuda performs
        # an explicit conversion. hwdownload cannot download that frame as
        # nv12 directly, which previously stopped HEVC Main 10 projects.
        f"force_divisible_by=2:format=nv12:passthrough=0,"
        "hwdownload,format=nv12,"
        f"pad={canvas_width}:{canvas_height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        f"fps={fps},format=yuv420p,trim=0:{source_duration:.6f},setpts=PTS-STARTPTS{hold}"
    )


def probe_nvidia_source_pipeline(
    ffmpeg_binary: Path | str,
    source: Path | str,
    *,
    source_codec: str,
    pixel_format: str,
    canvas_width: int,
    canvas_height: int,
    fps: float,
    start_seconds: float = 0.0,
    timeout: float = 30.0,
    command_runner: Any = run_command,
) -> tuple[bool, str]:
    """Verify the real NVDEC → CUDA filter → NVENC path for one source.

    The general encoder check uses an 8-bit generated image and cannot prove
    that a particular 10-bit/profile source can pass through CUDA filters.
    Results are cached by file identity and render geometry for this process.
    """
    source_path = Path(source).resolve()
    stat = source_path.stat()
    key = (
        str(source_path).casefold(), stat.st_size, stat.st_mtime_ns,
        str(source_codec).casefold(), str(pixel_format).casefold(),
        canvas_width, canvas_height, round(float(fps), 6),
    )
    with _nvidia_source_probe_lock:
        cached = _nvidia_source_probe_cache.get(key)
    if cached is not None:
        return cached
    status = EncoderStatus(True, "NVIDIA", "h264_nvenc", "NVIDIA NVENC")
    decode = nvidia_decode_args(source_codec, status, pixel_format)
    null_sink = "NUL" if sys.platform == "win32" else "/dev/null"
    filter_graph = nvidia_scale_pad_filter(
        canvas_width=canvas_width, canvas_height=canvas_height, fps=fps,
        source_duration=0.25, hardware_input=bool(decode),
    )
    command = [
        str(ffmpeg_binary), "-hide_banner", "-loglevel", "error", "-y",
        "-accurate_seek", "-ss", f"{max(0.0, start_seconds):.3f}",
        *decode, "-i", str(source_path), "-an", "-frames:v", "2",
        "-vf", filter_graph, "-c:v", "h264_nvenc", "-f", "null", null_sink,
    ]
    try:
        result = command_runner(command, timeout=timeout)
        ok = result.exit_code == 0
        detail = "" if ok else (result.stderr or "NVIDIA source pipeline returned no diagnostic")[-2000:]
    except Exception as exc:
        ok, detail = False, str(exc)[-2000:]
    outcome = (ok, detail)
    with _nvidia_source_probe_lock:
        _nvidia_source_probe_cache[key] = outcome
    return outcome
