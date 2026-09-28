"""Source media preparation, stream probing, audio selection, and subtitle extraction for ToolRecap V4."""
from __future__ import annotations

from .audio import (
    check_audio_has_speech,
    extract_audio_for_stt,
    get_audio_map_arg,
    select_episode_audio_stream,
)
from .pipeline import (
    SourcePreparationPipeline,
    compute_pipeline_cache_key,
    prepare_episode_source,
)
from .probe import EpisodeProbeResult, probe_episode_source
from .stt import (
    DEFAULT_STT_MANIFEST,
    SttModelManifest,
    SttModelManager,
    SttResult,
    SttStatus,
    compute_stt_dependency_signature,
    deduplicate_overlap_cues,
    extract_bounded_audio_window,
    plan_audio_windows,
    resolve_device_policy,
    transcribe_episode_stt,
)
from .transcript import build_transcript, compute_transcript_hash, validate_and_normalize_cue

__all__ = [
    "DEFAULT_STT_MANIFEST",
    "EpisodeProbeResult",
    "SourcePreparationPipeline",
    "SttModelManifest",
    "SttModelManager",
    "SttResult",
    "SttStatus",
    "build_transcript",
    "check_audio_has_speech",
    "compute_pipeline_cache_key",
    "compute_stt_dependency_signature",
    "compute_transcript_hash",
    "deduplicate_overlap_cues",
    "extract_audio_for_stt",
    "extract_bounded_audio_window",
    "get_audio_map_arg",
    "plan_audio_windows",
    "prepare_episode_source",
    "probe_episode_source",
    "resolve_device_policy",
    "select_episode_audio_stream",
    "transcribe_episode_stt",
    "validate_and_normalize_cue",
]
