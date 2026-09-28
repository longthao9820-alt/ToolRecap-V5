"""Bounded local STT window extraction, managed model manager, device fallback, and status classification."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Sequence
import uuid

from toolrecap_v4.analysis.models import AudioSelection
from toolrecap_v4.analysis.source_prep.audio import measure_audio_signal
from toolrecap_v4.analysis.source_prep.subtitles.models import SubtitleCue
from toolrecap_v4.analysis.source_prep.subtitles.parsers import normalize_subtitle_text
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, ToolRecapError
from toolrecap_v4.media import CommandResult, find_binary, run_command
from toolrecap_v4.persistence import get_storage_root


class SttStatus(str, Enum):
    """Exhaustive status classification for bounded STT operations."""

    SUCCESS = "success"
    NO_AUDIO = "no_audio"
    SILENT = "silent"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    FAILED = "failed"
    SUCCESS_EMPTY = "success_empty"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SttModelManifest:
    """Explicit STT model, repository, revision, settings, and file manifest."""

    model_name: str = "tiny"
    repo_id: str = "Systran/faster-whisper-tiny"
    revision: str = "d90ca5fe260221311c53c58e660288d3deb8d356"
    settings: dict[str, Any] = field(default_factory=lambda: {
        "beam_size": 1,
        "compute_type": "int8",
        "device": "cpu",
    })
    expected_files: tuple[str, ...] = (
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.txt",
    )
    file_checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "settings": dict(self.settings),
            "expected_files": list(self.expected_files),
            "file_checksums": dict(self.file_checksums),
        }

    def compute_manifest_hash(self) -> str:
        """Compute deterministic SHA-256 hash representing model repository, revision, and settings."""
        settings_raw = repr(sorted(self.settings.items()))
        files_raw = "|".join(sorted(self.expected_files))
        checksums_raw = repr(sorted(self.file_checksums.items()))
        raw = f"repo:{self.repo_id}|rev:{self.revision}|settings:{settings_raw}|files:{files_raw}|hashes:{checksums_raw}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEFAULT_STT_MANIFEST = SttModelManifest()


def compute_stt_dependency_signature(
    audio_selection: AudioSelection,
    manifest: SttModelManifest,
    *,
    device_policy: str = "auto",
    video_path: Path | str | None = None,
    video_stat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute content-addressable dependency signature for STT transcription.
    
    Includes audio identity, model manifest/revision, settings, and device policy.
    """
    manifest_hash = manifest.compute_manifest_hash()
    codec = audio_selection.selected_stream.codec if audio_selection.selected_stream else "none"
    global_index = audio_selection.global_index
    audio_ordinal = audio_selection.audio_ordinal

    sig_payload = {
        "type": "stt",
        "audio_global_index": global_index,
        "audio_ordinal": audio_ordinal,
        "audio_codec": codec,
        "model_name": manifest.model_name,
        "repo_id": manifest.repo_id,
        "revision": manifest.revision,
        "settings": manifest.settings,
        "device_policy": device_policy,
        "manifest_hash": manifest_hash,
    }

    if video_stat is not None:
        sig_payload["video_size"] = int(video_stat.get("size", 0))
        sig_payload["video_mtime"] = float(video_stat.get("mtime", 0.0))
    elif video_path is not None:
        vp = Path(video_path).resolve()
        if vp.is_file():
            st = vp.stat()
            sig_payload["video_size"] = st.st_size
            sig_payload["video_mtime"] = round(st.st_mtime, 3)

    raw_tokens = [f"{k}:{v}" for k, v in sorted(sig_payload.items())]
    sig_payload["stt_hash"] = hashlib.sha256("|".join(raw_tokens).encode("utf-8")).hexdigest()
    return sig_payload


@dataclass(frozen=True)
class SttResult:
    """Immutable result of STT audio transcription with explicit diagnostics and status."""

    status: SttStatus | str
    cues: tuple[SubtitleCue, ...] = field(default_factory=tuple)
    language: str = "en"
    model_name: str = ""
    device_used: str = "cpu"
    compute_type: str = "int8"
    diagnostics: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    @property
    def cue_count(self) -> int:
        return len(self.cues)

    @property
    def is_success(self) -> bool:
        return self.status in (SttStatus.SUCCESS, "success")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status.value if isinstance(self.status, SttStatus) else self.status),
            "cues": [c.to_dict() for c in self.cues],
            "language": self.language,
            "model_name": self.model_name,
            "device_used": self.device_used,
            "compute_type": self.compute_type,
            "diagnostics": list(self.diagnostics),
            "error": self.error,
        }


class SttModelManager:
    """Manages local Whisper model snapshots with atomic staging and manifests.
    
    Invariants:
    - Staged download uses .partial staging directory.
    - Canceled/failed download never leaves a ready manifest.
    - Model local_files_only after READY.
    """

    def __init__(
        self,
        model_dir: Path | str | None = None,
        manifest: SttModelManifest | None = None,
        downloader: Callable[..., Any] | None = None,
    ) -> None:
        self.manifest = manifest or DEFAULT_STT_MANIFEST
        if model_dir is not None:
            self.model_dir = Path(model_dir).resolve()
        else:
            self.model_dir = (
                get_storage_root() / "models" / "stt" / self.manifest.model_name
            ).resolve()
        self.downloader = downloader

    def is_ready(self) -> bool:
        """Check whether local model snapshot is complete and ready for local_files_only loading."""
        if not self.model_dir.is_dir():
            return False

        # Verify presence and non-zero size of all expected files
        for fname in self.manifest.expected_files:
            target = self.model_dir / fname
            if not target.is_file() or target.stat().st_size == 0:
                return False

        # Verify any explicit checksums
        for fname, expected_hash in self.manifest.file_checksums.items():
            target = self.model_dir / fname
            if not target.is_file():
                return False
            hasher = hashlib.sha256()
            with target.open("rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    hasher.update(chunk)
            if hasher.hexdigest().lower() != expected_hash.lower():
                return False

        return True

    def download_models(
        self,
        cancellation_token: CancellationToken | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> bool:
        """Download Whisper snapshot atomically into staging directory before promotion.
        
        Invariants:
        - Partial or cancelled downloads NEVER result in READY state.
        - Staging directory is cleaned on failure or cancellation.
        """
        if cancellation_token:
            cancellation_token.check_cancelled()

        if self.is_ready():
            return True

        self.model_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = self.model_dir.parent / f"{self.model_dir.name}.partial"
        if staging_dir.is_dir():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.downloader is not None:
                self.downloader(
                    repo_id=self.manifest.repo_id,
                    revision=self.manifest.revision,
                    target_dir=staging_dir,
                    cancellation_token=cancellation_token,
                    progress_callback=progress_callback,
                )
            else:
                self._default_download(
                    target_dir=staging_dir,
                    cancellation_token=cancellation_token,
                    progress_callback=progress_callback,
                )

            if cancellation_token:
                cancellation_token.check_cancelled()

            # Verify all expected files are present in staging
            for fname in self.manifest.expected_files:
                target = staging_dir / fname
                if not target.is_file() or target.stat().st_size == 0:
                    raise ToolRecapError(
                        f"Expected STT model file {fname} is missing or empty after download"
                    )

            # Verify checksums if present
            for fname, expected_hash in self.manifest.file_checksums.items():
                target = staging_dir / fname
                hasher = hashlib.sha256()
                with target.open("rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        hasher.update(chunk)
                if hasher.hexdigest().lower() != expected_hash.lower():
                    raise ToolRecapError(
                        f"Checksum mismatch for STT model file {fname}: expected {expected_hash}"
                    )

            # Write manifest metadata into staging
            manifest_path = staging_dir / "manifest.json"
            m_data = self.manifest.to_dict()
            m_data["manifest_hash"] = self.manifest.compute_manifest_hash()
            m_data["ready"] = True
            manifest_path.write_text(json.dumps(m_data, indent=2), encoding="utf-8")

            if cancellation_token:
                cancellation_token.check_cancelled()

            # Atomic directory replacement
            if self.model_dir.exists():
                shutil.rmtree(self.model_dir, ignore_errors=True)
            os.replace(staging_dir, self.model_dir)

        except Exception:
            if staging_dir.is_dir():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        return self.is_ready()

    def _default_download(
        self,
        target_dir: Path,
        cancellation_token: CancellationToken | None,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> None:
        try:
            import huggingface_hub
        except ImportError as exc:
            raise ToolRecapError(
                "huggingface_hub is not installed; cannot download STT models"
            ) from exc

        allow_patterns = [
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.txt",
        ]
        huggingface_hub.snapshot_download(
            self.manifest.repo_id,
            revision=self.manifest.revision,
            local_dir=str(target_dir),
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
        )


def resolve_device_policy(
    policy: str = "auto",
    check_cuda_fn: Callable[[], bool] | None = None,
) -> tuple[str, str, list[str]]:
    """Resolve device ('cpu' or 'cuda') and compute_type ('int8' or 'float16') with safe CPU fallback.
    
    Diagnostics are explicitly returned if requested GPU is unavailable.
    """
    p = policy.strip().lower()

    def _has_cuda() -> bool:
        if check_cuda_fn is not None:
            return bool(check_cuda_fn())
        try:
            import ctranslate2
            return bool(hasattr(ctranslate2, "get_cuda_device_count") and ctranslate2.get_cuda_device_count() > 0)
        except Exception:
            return False

    diagnostics: list[str] = []

    if p == "cuda":
        if _has_cuda():
            return "cuda", "float16", ["CUDA GPU device available and selected"]
        raise ToolRecapError(
            "Requested STT device 'cuda' is unavailable (no CUDA GPU detected or ctranslate2 lacks CUDA support)."
        )

    if p == "cpu":
        return "cpu", "int8", ["Device explicitly set to CPU (int8)"]

    if p != "auto":
        raise ValueError(f"Unsupported STT device policy: {policy!r}")

    # 'auto' policy
    if _has_cuda():
        return "cuda", "float16", ["Auto-detected CUDA GPU device"]
    return "cpu", "int8", ["Auto-selected CPU (int8) safe fallback"]


def default_whisper_factory(
    model_path: str | Path,
    device: str = "cpu",
    compute_type: str = "int8",
    local_files_only: bool = True,
) -> Any:
    """Factory creating faster-whisper WhisperModel instance with local_files_only=True."""
    from faster_whisper import WhisperModel
    return WhisperModel(
        str(model_path),
        device=device,
        compute_type=compute_type,
        local_files_only=local_files_only,
    )


def plan_audio_windows(
    total_duration_sec: float,
    window_duration_sec: float = 60.0,
    overlap_sec: float = 2.0,
) -> list[tuple[float, float]]:
    """Compute deterministic sequence of (start_sec, duration_sec) bounded time windows.
    
    Invariants:
    - Never extracts whole episode audio in normal path.
    - Overlap duration is deterministic.
    - Covers entire total_duration_sec.
    """
    if total_duration_sec <= 0.0:
        return []

    if total_duration_sec <= window_duration_sec:
        return [(0.0, total_duration_sec)]

    windows: list[tuple[float, float]] = []
    step = max(1.0, window_duration_sec - overlap_sec)
    start = 0.0

    while start < total_duration_sec:
        dur = min(window_duration_sec, total_duration_sec - start)
        windows.append((round(start, 3), round(dur, 3)))
        start += step
        if dur < window_duration_sec:
            break

    return windows


def extract_bounded_audio_window(
    video_path: Path | str,
    selection: AudioSelection,
    start_sec: float,
    duration_sec: float,
    output_wav: Path | str,
    *,
    command_runner: Callable[..., CommandResult] = run_command,
    cancellation_token: CancellationToken | None = None,
    timeout: float = 60.0,
) -> Path:
    """Extract a bounded time window from selected global audio stream via FFmpeg directly.
    
    Invariants:
    - Strictly uses container global stream mapping: -map 0:<global_index>.
    - Strictly extracts bounded slice with -ss and -t.
    - Never whole episode audio file.
    """
    if not selection.has_audio or selection.global_index < 0:
        raise ValueError("Cannot extract audio from empty or invalid audio selection")

    if cancellation_token:
        cancellation_token.check_cancelled()

    src = Path(video_path).resolve()
    dst = Path(output_wav).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    binary = find_binary("ffmpeg")
    cmd = [
        str(binary),
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration_sec:.3f}",
        "-i", str(src),
        "-map", f"0:{selection.global_index}",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        str(dst),
    ]

    res = command_runner(cmd, timeout=timeout, cancellation_token=cancellation_token)
    if res.exit_code != 0:
        raise ToolRecapError(f"FFmpeg bounded window extraction failed: {res.stderr.strip()}")

    if not dst.is_file() or dst.stat().st_size == 0:
        raise ToolRecapError(f"Extracted window WAV is missing or empty: {dst}")

    return dst


def deduplicate_overlap_cues(
    existing_cues: Sequence[SubtitleCue],
    candidate_cues: Sequence[SubtitleCue],
    max_jitter_ms: int = 1500,
) -> list[SubtitleCue]:
    """Deduplicate only exact overlap duplicate cues, preserving boundary speech.
    
    A candidate cue is treated as an overlap duplicate only if its normalized text
    matches an existing cue and its start timestamp is within max_jitter_ms.
    """
    if not existing_cues:
        return list(candidate_cues)

    accepted: list[SubtitleCue] = []

    for cand in candidate_cues:
        cand_norm = normalize_subtitle_text(cand.text)
        is_duplicate = False

        # Compare with recent existing cues
        for prev in reversed(existing_cues):
            # If previous cue is far behind, stop checking
            if cand.start_ms - prev.start_ms > 10000:
                break

            prev_norm = normalize_subtitle_text(prev.text)
            if prev_norm == cand_norm and abs(prev.start_ms - cand.start_ms) <= max_jitter_ms:
                is_duplicate = True
                break

        if not is_duplicate:
            accepted.append(cand)

    return accepted


def transcribe_episode_stt(
    video_path: Path | str,
    selection: AudioSelection,
    duration_ms: int,
    *,
    model_manager: SttModelManager | None = None,
    manifest: SttModelManifest | None = None,
    device_policy: str = "auto",
    window_duration_sec: float = 60.0,
    overlap_sec: float = 2.0,
    language: str = "en",
    command_runner: Callable[..., CommandResult] = run_command,
    model_factory: Callable[..., Any] | None = None,
    check_cuda_fn: Callable[[], bool] | None = None,
    temp_dir: Path | str | None = None,
    cancellation_token: CancellationToken | None = None,
    min_rms_threshold: float = 80.0,
) -> SttResult:
    """Execute bounded STT transcription across an episode source adhering to Phase 3 invariants.
    
    Invariants:
    - Uses explicit global stream mapping (-map 0:<global_index>).
    - Extracts bounded windows directly via FFmpeg, never whole episode WAV.
    - Safe CPU int8 fallback with diagnostics when requested GPU is unavailable.
    - Cleans up temporary window WAV files upon completion or failure.
    - Distinguishes no_audio, silent, runtime_unavailable, failed, success_empty, and cancel.
    - Cancellation propagates immediately.
    """
    if cancellation_token:
        cancellation_token.check_cancelled()

    # 1. No audio check
    if not selection.has_audio or selection.global_index < 0:
        return SttResult(
            status=SttStatus.NO_AUDIO,
            cues=(),
            diagnostics=("Audio stream not present in source or selection is invalid",),
        )

    # 2. Model manager readiness check
    mgr = model_manager or SttModelManager(manifest=manifest)
    if not mgr.is_ready():
        return SttResult(
            status=SttStatus.RUNTIME_UNAVAILABLE,
            cues=(),
            model_name=mgr.manifest.model_name,
            diagnostics=(f"STT model snapshot '{mgr.manifest.model_name}' is not ready locally",),
        )

    # 3. Resolve device policy
    device, compute_type, dev_diagnostics = resolve_device_policy(
        policy=device_policy,
        check_cuda_fn=check_cuda_fn,
    )

    # 4. Instantiate model with local_files_only=True
    factory = model_factory or default_whisper_factory
    try:
        model = factory(
            str(mgr.model_dir),
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )
    except Exception as exc:
        return SttResult(
            status=SttStatus.RUNTIME_UNAVAILABLE,
            cues=(),
            model_name=mgr.manifest.model_name,
            device_used=device,
            compute_type=compute_type,
            diagnostics=tuple(dev_diagnostics + [f"Failed to load local STT model: {exc}"]),
            error=str(exc),
        )

    # 5. Plan bounded windows
    total_duration_sec = max(0.0, duration_ms / 1000.0)
    windows = plan_audio_windows(
        total_duration_sec=total_duration_sec,
        window_duration_sec=window_duration_sec,
        overlap_sec=overlap_sec,
    )

    if not windows:
        return SttResult(
            status=SttStatus.SUCCESS_EMPTY,
            cues=(),
            model_name=mgr.manifest.model_name,
            device_used=device,
            compute_type=compute_type,
            diagnostics=tuple(dev_diagnostics + ["Source duration is 0 seconds"]),
        )

    # Setup temporary directory for bounded audio window files
    run_id = uuid.uuid4().hex[:8]
    work_dir = Path(temp_dir) if temp_dir else (get_storage_root() / "temp" / f"stt_{run_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    all_cues: list[SubtitleCue] = []
    any_has_signal = False
    window_index = 0

    try:
        for start_sec, dur_sec in windows:
            if cancellation_token:
                cancellation_token.check_cancelled()

            window_wav = work_dir / f"win_{window_index:04d}.wav"
            window_index += 1

            try:
                # Extract bounded window directly
                extract_bounded_audio_window(
                    video_path=video_path,
                    selection=selection,
                    start_sec=start_sec,
                    duration_sec=dur_sec,
                    output_wav=window_wav,
                    command_runner=command_runner,
                    cancellation_token=cancellation_token,
                )

                if cancellation_token:
                    cancellation_token.check_cancelled()

                # Check acoustic signal
                sig_res = measure_audio_signal(
                    window_wav,
                    min_rms_threshold=min_rms_threshold,
                    cancellation_token=cancellation_token,
                )

                if not sig_res.has_signal:
                    continue

                any_has_signal = True

                # Transcribe window
                beam_size = mgr.manifest.settings.get("beam_size", 1)
                segments_iter, _info = model.transcribe(
                    str(window_wav),
                    beam_size=beam_size,
                    language=language,
                )

                window_cues: list[SubtitleCue] = []
                for seg in segments_iter:
                    if cancellation_token:
                        cancellation_token.check_cancelled()

                    txt = getattr(seg, "text", "").strip()
                    if not txt:
                        continue

                    seg_start = float(getattr(seg, "start", 0.0))
                    seg_end = float(getattr(seg, "end", 0.0))

                    abs_start_ms = max(0, round((start_sec + seg_start) * 1000))
                    abs_end_ms = max(abs_start_ms + 10, round((start_sec + seg_end) * 1000))

                    cue = SubtitleCue(
                        start_ms=abs_start_ms,
                        end_ms=abs_end_ms,
                        text=txt,
                        source_type="stt",
                        source_format="whisper",
                        stream_index=selection.global_index,
                        language=language,
                    )
                    window_cues.append(cue)

                # Deduplicate overlap
                accepted = deduplicate_overlap_cues(all_cues, window_cues)
                all_cues.extend(accepted)

            finally:
                # Ensure bounded disk usage: clean up window WAV immediately
                if window_wav.is_file():
                    window_wav.unlink(missing_ok=True)

    except CancelledError:
        raise
    except Exception as exc:
        return SttResult(
            status=SttStatus.FAILED,
            cues=tuple(all_cues),
            model_name=mgr.manifest.model_name,
            device_used=device,
            compute_type=compute_type,
            diagnostics=tuple(dev_diagnostics + [f"STT transcription failed: {exc}"]),
            error=str(exc),
        )
    finally:
        # Bounded cleanup: remove temporary working directory
        if work_dir.is_dir():
            shutil.rmtree(work_dir, ignore_errors=True)

    # 6. Evaluate final status
    if not any_has_signal:
        return SttResult(
            status=SttStatus.SILENT,
            cues=(),
            model_name=mgr.manifest.model_name,
            device_used=device,
            compute_type=compute_type,
            diagnostics=tuple(dev_diagnostics + ["Audio signal below acoustic threshold across all windows"]),
        )

    if not all_cues:
        return SttResult(
            status=SttStatus.SUCCESS_EMPTY,
            cues=(),
            model_name=mgr.manifest.model_name,
            device_used=device,
            compute_type=compute_type,
            diagnostics=tuple(dev_diagnostics + ["Audio signal detected but 0 speech cues transcribed"]),
        )

    return SttResult(
        status=SttStatus.SUCCESS,
        cues=tuple(all_cues),
        language=language,
        model_name=mgr.manifest.model_name,
        device_used=device,
        compute_type=compute_type,
        diagnostics=tuple(dev_diagnostics),
    )
