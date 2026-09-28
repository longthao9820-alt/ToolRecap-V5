"""Deterministic audio stream selection, mapping helpers, and bounded-memory audio signal checks."""
from __future__ import annotations

import array
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence
import wave

from toolrecap_v4.analysis.models import AudioSelection
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import ToolRecapError
from toolrecap_v4.media import (
    AudioStreamInfo,
    find_binary,
    is_commentary_or_descriptive,
    is_english_language,
    run_command,
)


@dataclass(frozen=True)
class AudioSignalResult:
    """Technical acoustic measurement of PCM audio signal without anthropomorphic claims."""

    has_signal: bool
    rms_peak: float
    rms_mean: float
    active_chunks: int
    total_chunks: int
    threshold: float


def select_episode_audio_stream(audio_streams: Sequence[AudioStreamInfo]) -> AudioSelection:
    """Select the best audio stream adhering strictly to Phase 3 invariants.

    Invariants:
    - Stores both container global_index and audio_ordinal (0-based among audio streams).
    - Unknown/undefined language fallback is explicitly marked uncertain; NEVER claims English.
    - Commentary and descriptive audio tracks are deprioritized.
    - If no audio stream is present, returns AudioSelection with global_index=-1 and selected_stream=None.
    - Priority:
        1. English non-commentary default
        2. English non-commentary any
        3. Default non-commentary
        4. First non-commentary
        5. First overall stream
    """
    if not audio_streams:
        return AudioSelection(
            selected_stream=None,
            global_index=-1,
            audio_ordinal=-1,
            warning="No audio streams found in source",
            reason="Source has no audio streams",
        )

    # Annotate streams with their 0-based audio ordinal
    annotated = list(enumerate(audio_streams))

    clean_streams = [
        (ord_idx, s) for ord_idx, s in annotated if not is_commentary_or_descriptive(s)
    ]
    candidate_pool = clean_streams if clean_streams else annotated

    selected_ordinal: int
    selected: AudioStreamInfo
    reason_prefix: str

    # 1. English default
    eng_default = [
        (ord_idx, s) for ord_idx, s in candidate_pool
        if is_english_language(s.language) and s.disposition.get("default", 0) == 1
    ]
    if eng_default:
        selected_ordinal, selected = eng_default[0]
        reason_prefix = "English default audio stream"
    else:
        # 2. English any
        eng_any = [
            (ord_idx, s) for ord_idx, s in candidate_pool
            if is_english_language(s.language)
        ]
        if eng_any:
            selected_ordinal, selected = eng_any[0]
            reason_prefix = "English audio stream"
        else:
            # 3. Default stream
            def_any = [
                (ord_idx, s) for ord_idx, s in candidate_pool
                if s.disposition.get("default", 0) == 1
            ]
            if def_any:
                selected_ordinal, selected = def_any[0]
                reason_prefix = "Default audio stream fallback"
            else:
                # 4. First candidate
                selected_ordinal, selected = candidate_pool[0]
                reason_prefix = "First available audio stream fallback"

    # Evaluate language certainty
    lang_raw = (selected.language or "").strip().lower()
    is_und = not lang_raw or lang_raw in ("und", "unknown", "undefined")

    if is_und:
        warning = "Selected audio stream has undefined language; uncertain whether English"
        reason = (
            f"{reason_prefix} #{selected.index} (audio ordinal {selected_ordinal}) "
            "has undefined/uncertain language; not confirmed English"
        )
    elif is_english_language(selected.language):
        warning = None
        reason = f"{reason_prefix} #{selected.index} (audio ordinal {selected_ordinal}, lang='{selected.language}')"
    else:
        warning = f"Selected audio stream has non-English language '{selected.language}'"
        reason = (
            f"{reason_prefix} #{selected.index} (audio ordinal {selected_ordinal}, lang='{selected.language}')"
        )

    return AudioSelection(
        selected_stream=selected,
        global_index=selected.index,
        audio_ordinal=selected_ordinal,
        warning=warning,
        reason=reason,
    )


def get_audio_map_arg(selection: AudioSelection) -> list[str]:
    """Return explicit FFmpeg stream mapping argument using global stream index (0:<global_index>).
    
    Invariants:
    - Never generates '0:a:<ordinal>' which fails if container global indexing diverges.
    - Raises ValueError if selection represents no-audio.
    """
    if not selection.has_audio or selection.global_index < 0:
        raise ValueError("Cannot map audio for empty or invalid audio selection")
    return ["-map", f"0:{selection.global_index}"]


def extract_audio_for_stt(
    video_path: Path | str,
    output_wav: Path | str,
    selection: AudioSelection,
    *,
    timeout: float = 300.0,
    cancellation_token: CancellationToken | None = None,
) -> Path:
    """Extract selected audio stream to 16kHz mono WAV using explicit container global stream mapping."""
    src = Path(video_path).resolve()
    dst = Path(output_wav).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if cancellation_token:
        cancellation_token.check_cancelled()

    binary = find_binary("ffmpeg")
    map_args = get_audio_map_arg(selection)

    cmd = [
        str(binary),
        "-y",
        "-i", str(src),
        *map_args,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        str(dst),
    ]

    res = run_command(cmd, timeout=timeout, cancellation_token=cancellation_token)
    if res.exit_code != 0:
        raise ToolRecapError(f"Audio extraction failed for {src}: {res.stderr.strip()}")

    if not dst.is_file() or dst.stat().st_size == 0:
        raise ToolRecapError(f"Extracted audio file is missing or empty: {dst}")

    return dst


def measure_audio_signal(
    wav_path: Path | str,
    *,
    min_rms_threshold: float = 80.0,
    chunk_frames: int = 32768,
    min_active_chunks: int = 2,
    cancellation_token: CancellationToken | None = None,
) -> AudioSignalResult:
    """Measure acoustic signal levels in 16-bit PCM WAV using streaming chunked reading.
    
    Invariants:
    - Memory bounded: reads only 64KB (32768 16-bit frames) per chunk, never buffering entire file in RAM.
    - Cooperative cancellation checked between chunks.
    """
    p = Path(wav_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"WAV audio file not found: {p}")

    active_chunks = 0
    total_chunks = 0
    rms_peak = 0.0
    rms_sum = 0.0

    try:
        with wave.open(str(p), "rb") as wf:
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                # Non-16-bit audio: cannot unpack as int16
                return AudioSignalResult(
                    has_signal=True,
                    rms_peak=0.0,
                    rms_mean=0.0,
                    active_chunks=0,
                    total_chunks=0,
                    threshold=min_rms_threshold,
                )

            while True:
                if cancellation_token:
                    cancellation_token.check_cancelled()

                raw = wf.readframes(chunk_frames)
                if not raw:
                    break

                samples = array.array("h")
                samples.frombytes(raw)
                if not samples:
                    continue

                total_chunks += 1
                sum_sq = sum(s * s for s in samples)
                rms = math.sqrt(sum_sq / len(samples))
                rms_sum += rms
                if rms > rms_peak:
                    rms_peak = rms

                if rms >= min_rms_threshold:
                    active_chunks += 1

    except (wave.Error, EOFError):
        return AudioSignalResult(
            has_signal=True,
            rms_peak=0.0,
            rms_mean=0.0,
            active_chunks=0,
            total_chunks=0,
            threshold=min_rms_threshold,
        )

    rms_mean = (rms_sum / total_chunks) if total_chunks > 0 else 0.0
    required_chunks = min(min_active_chunks, total_chunks) if total_chunks > 0 else 1
    has_signal = active_chunks >= required_chunks and active_chunks > 0

    return AudioSignalResult(
        has_signal=has_signal,
        rms_peak=round(rms_peak, 2),
        rms_mean=round(rms_mean, 2),
        active_chunks=active_chunks,
        total_chunks=total_chunks,
        threshold=min_rms_threshold,
    )


def check_audio_has_signal(
    wav_path: Path | str,
    *,
    min_rms_threshold: float = 80.0,
    chunk_frames: int = 32768,
    min_active_chunks: int = 2,
    cancellation_token: CancellationToken | None = None,
) -> bool:
    """Technical check whether audio file has acoustic signal above threshold."""
    result = measure_audio_signal(
        wav_path=wav_path,
        min_rms_threshold=min_rms_threshold,
        chunk_frames=chunk_frames,
        min_active_chunks=min_active_chunks,
        cancellation_token=cancellation_token,
    )
    return result.has_signal


def check_audio_has_speech(
    wav_path: Path | str,
    *,
    min_rms_threshold: float = 80.0,
    chunk_frames: int = 32768,
    min_speech_chunks: int = 2,
    cancellation_token: CancellationToken | None = None,
) -> bool:
    """Legacy alias for check_audio_has_signal."""
    return check_audio_has_signal(
        wav_path=wav_path,
        min_rms_threshold=min_rms_threshold,
        chunk_frames=chunk_frames,
        min_active_chunks=min_speech_chunks,
        cancellation_token=cancellation_token,
    )
