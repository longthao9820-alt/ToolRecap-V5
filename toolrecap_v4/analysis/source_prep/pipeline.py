"""Deterministic local episode source preparation orchestrator for ToolRecap V4."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Sequence
import uuid

from toolrecap_v4.analysis.cache import AnalysisCacheManager, default_analysis_cache_dir
from toolrecap_v4.analysis.dependencies import (
    NORMALIZATION_VERSION,
    PGS_DECODER_VERSION,
    PIPELINE_VERSION,
    STT_EXTRACTION_POLICY,
    STT_WINDOW_POLICY,
    SUBTITLE_PARSER_VERSION,
    VOBSUB_DECODER_VERSION,
    compute_embedded_track_signature,
    compute_inventory_hash,
    compute_sidecar_signature,
    compute_source_fingerprint,
    compute_source_signature,
    compute_stt_dependency_signature,
    compute_vobsub_signature,
    hash_file_content,
)
from toolrecap_v4.analysis.models import AudioSelection, PreparedEpisode, Transcript
from toolrecap_v4.analysis.source_prep.audio import select_episode_audio_stream
from toolrecap_v4.analysis.source_prep.probe import EpisodeProbeResult, probe_episode_source
from toolrecap_v4.analysis.source_prep.stt import (
    DEFAULT_STT_MANIFEST,
    SttModelManager,
    SttResult,
    SttStatus,
    transcribe_episode_stt,
)
from toolrecap_v4.analysis.source_prep.subtitles.discovery import (
    build_embedded_tracks,
    discover_sidecars,
    select_best_english_subtitles,
)
from toolrecap_v4.analysis.source_prep.subtitles.models import SubtitleTrack
from toolrecap_v4.analysis.source_prep.subtitles.ocr import (
    DEFAULT_OCR_MANIFEST,
    OcrAdapter,
)
from toolrecap_v4.analysis.source_prep.subtitles.pipeline import (
    SubtitlePipeline,
    SubtitlePipelineResult,
)
from toolrecap_v4.analysis.source_prep.transcript import build_transcript
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, ToolRecapError
from toolrecap_v4.media import MediaProbeResult, assess_source_timeline, inspect_frame_timing, run_command


def compute_pipeline_cache_key(
    episode_id: str,
    source_fingerprint: str,
    inventory_hash: str,
    selected_track_id: str,
    selected_method: str,
    target_hash: str,
    *,
    pipeline_version: str = PIPELINE_VERSION,
    parser_version: str = SUBTITLE_PARSER_VERSION,
    normalization_version: str = NORMALIZATION_VERSION,
    pgs_decoder_version: str = PGS_DECODER_VERSION,
    vobsub_decoder_version: str = VOBSUB_DECODER_VERSION,
    ocr_manifest_hash: str = "none",
    stt_manifest_hash: str = "none",
    device_policy: str = "auto",
) -> str:
    """Compute content-addressable cache key strictly incorporating inventory, selection, versions, and policy."""
    raw = (
        f"ep:{episode_id}|src:{source_fingerprint}|inv:{inventory_hash}|"
        f"sel:{selected_track_id}|method:{selected_method}|target:{target_hash}|"
        f"v_pipe:{pipeline_version}|v_parse:{parser_version}|v_norm:{normalization_version}|"
        f"v_pgs:{pgs_decoder_version}|v_vob:{vobsub_decoder_version}|"
        f"ocr:{ocr_manifest_hash}|stt:{stt_manifest_hash}|dev:{device_policy}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SourcePreparationPipeline:
    """Orchestrates local episode source preparation into PreparedEpisode with strict fallback.

    Invariants:
    - Fallback exact order: sidecar English text -> embedded English text -> bitmap local OCR -> STT selected audio.
    - No expensive lower fallback when valid higher artifact or cache exists.
    - Fresh inventory scan each run: new better sidecar automatically invalidates old STT.
    - Strict transcript bounds and no clamp/truncation.
    - Errors not silence: explicit status classification ('ready', 'no_audio', 'silent', 'failed').
    - Cancellation never promotes cache; all temp files cleaned up.
    - Fully injectable components for deterministic testing without external binaries.
    """

    def __init__(
        self,
        cache_manager: AnalysisCacheManager | None = None,
        subtitle_pipeline: SubtitlePipeline | None = None,
        stt_model_manager: SttModelManager | None = None,
        ocr_adapter: OcrAdapter | None = None,
        probe_fn: Any = probe_episode_source,
        transcribe_fn: Any = transcribe_episode_stt,
        run_command_fn: Any = run_command,
        temp_dir: Path | str | None = None,
        allow_stt: bool = False,
        allow_bitmap_ocr: bool = True,
    ) -> None:
        self.cache_manager = cache_manager or AnalysisCacheManager()
        self.ocr_adapter = ocr_adapter or (subtitle_pipeline.ocr_adapter if subtitle_pipeline else OcrAdapter())
        self.subtitle_pipeline = subtitle_pipeline or SubtitlePipeline(
            ocr_adapter=self.ocr_adapter,
            run_command_fn=run_command_fn,
            temp_dir=temp_dir,
        )
        self.stt_model_manager = stt_model_manager
        self.probe_fn = probe_fn
        self.transcribe_fn = transcribe_fn
        self.run_command_fn = run_command_fn
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.allow_stt = allow_stt
        self.allow_bitmap_ocr = allow_bitmap_ocr

    def prepare_episode(
        self,
        source_path: Path | str,
        episode_id: str = "",
        source_id: str = "",
        *,
        source_fingerprint: str | None = None,
        force_refresh: bool = False,
        probe_timeout: float = 60.0,
        device_policy: str = "auto",
        on_progress: Callable[[str, float, str], None] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> PreparedEpisode:
        """Prepare an episode source into a deterministic, immutable PreparedEpisode."""
        if cancellation_token:
            cancellation_token.check_cancelled()

        def _report(phase: str, pct: float, msg: str) -> None:
            if on_progress:
                on_progress(phase, pct, msg)

        # 1. Source verification & full content fingerprinting
        p_video = Path(source_path).resolve()
        if not p_video.is_file():
            raise FileNotFoundError(f"Source episode video file not found: {p_video}")

        ep_id = episode_id or p_video.stem
        src_id = source_id or ep_id

        # Use supplied SHA-256 or compute full content hash (never path/mtime only)
        src_fingerprint = compute_source_fingerprint(p_video, supplied_hash=source_fingerprint)

        _report("probe", 0.05, f"Probing container and media streams for {p_video.name}")
        probe_res: EpisodeProbeResult = self.probe_fn(
            p_video,
            timeout=probe_timeout,
            cancellation_token=cancellation_token,
        )

        if cancellation_token:
            cancellation_token.check_cancelled()

        timeline_assessment = assess_source_timeline(MediaProbeResult(
            path=probe_res.path,
            duration=probe_res.duration_ms / 1000.0,
            container=probe_res.container,
            video_streams=probe_res.video_streams,
            audio_streams=probe_res.audio_streams,
            has_video=bool(probe_res.video_streams),
            has_audio=bool(probe_res.audio_streams),
            start_time=probe_res.primary_video.start_time if probe_res.primary_video else 0.0,
        ))
        timing_details: dict[str, Any] = {}
        if timeline_assessment.status == "needs_frame_inspection" and self.probe_fn is probe_episode_source:
            timing_details = inspect_frame_timing(
                p_video, probe_res.duration_ms, cancellation_token=cancellation_token,
            )
        if timeline_assessment.requires_cfr_output:
            _report(
                "compatibility",
                0.08,
                "Source has irregular frame timing; Highlight publication will normalize it to CFR.",
            )

        # 2. Audio stream selection
        audio_sel: AudioSelection = select_episode_audio_stream(probe_res.audio_streams)

        # 3. Fresh inventory scan
        _report("inventory", 0.15, "Scanning directory for sidecars and matching tracks")
        sidecars = discover_sidecars(p_video, episode_id=ep_id)
        inv_hash, inv_items = compute_inventory_hash(sidecars, cancellation_token=cancellation_token)
        embedded = build_embedded_tracks(probe_res.subtitle_streams, source_video=str(p_video))
        all_tracks = list(sidecars) + list(embedded)
        blocked_bitmap_tracks = [
            track for track in all_tracks
            if track.is_bitmap and track.language == "eng" and not track.is_forced
        ]
        selection_tracks = (
            all_tracks if self.allow_bitmap_ocr
            else [track for track in all_tracks if not track.is_bitmap]
        )

        # 4. Selection policy decision. V5 disables bitmap OCR and therefore
        # considers only directly extractable text tracks.
        disc_res = select_best_english_subtitles(
            selection_tracks,
            video_path=str(p_video),
            episode_id=ep_id,
        )

        # Determine target asset identity and hash for dependency key
        chosen_track = disc_res.best_english_full
        eligible_tracks = [
            track for track in selection_tracks
            if track.language == "eng" and not track.is_forced
            and track.track_id != (chosen_track.track_id if chosen_track else None)
        ]
        eligible_tracks.sort(key=lambda track: (
            0 if not track.is_bitmap and track.source_type == "sidecar" else
            1 if not track.is_bitmap and track.source_type == "embedded" else
            2 if track.source_type == "sidecar" else 3,
            -track.score, track.track_id,
        ))
        candidate_tracks = ([chosen_track] if chosen_track else []) + eligible_tracks
        chosen_track_id: str
        chosen_method: str
        chosen_hash: str

        if chosen_track is not None:
            chosen_track_id = chosen_track.track_id
            chosen_method = "ocr" if chosen_track.is_bitmap else chosen_track.source_type
            if chosen_track.source_type == "sidecar" and chosen_track.source_file:
                p_side = Path(chosen_track.source_file).resolve()
                if p_side.suffix.lower() == ".idx":
                    chosen_hash = compute_vobsub_signature(p_side)["pair_hash"]
                else:
                    chosen_hash = compute_sidecar_signature(p_side)["content_hash"]
            else:
                chosen_hash = compute_embedded_track_signature(
                    p_video,
                    chosen_track.stream_index or 0,
                    chosen_track.source_format,
                )["stream_hash"]
        else:
            chosen_track_id = "stt" if self.allow_stt else "none"
            chosen_method = chosen_track_id
            if self.allow_stt and audio_sel.has_audio:
                chosen_hash = compute_stt_dependency_signature(
                    audio_sel,
                    DEFAULT_STT_MANIFEST,
                    device_policy=device_policy,
                    video_path=p_video,
                )["stt_hash"]
            else:
                chosen_hash = "no_audio"

        # All plausible English tracks affect a fallback result. A new or
        # changed alternative must invalidate the prepared-episode cache.
        candidate_identity = []
        for track in candidate_tracks:
            if track.source_type == "sidecar" and track.source_file:
                path = Path(track.source_file).resolve()
                signature = (
                    compute_vobsub_signature(path)["pair_hash"] if path.suffix.lower() == ".idx"
                    else compute_sidecar_signature(path)["content_hash"]
                )
            else:
                signature = compute_embedded_track_signature(
                    p_video, track.stream_index or 0, track.source_format,
                )["stream_hash"]
            candidate_identity.append((track.track_id, track.source_format, track.language, track.is_bitmap, signature))
        chosen_hash = hashlib.sha256(json.dumps(
            {
                "primary_hash": chosen_hash,
                "candidates": candidate_identity,
                "allow_stt": self.allow_stt,
                "allow_bitmap_ocr": self.allow_bitmap_ocr,
            },
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()

        ocr_manifest_hash = DEFAULT_OCR_MANIFEST.compute_manifest_hash()
        stt_manifest_hash = DEFAULT_STT_MANIFEST.compute_manifest_hash()

        dep_sig = {
            "pipeline_version": PIPELINE_VERSION,
            "parser_version": SUBTITLE_PARSER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "pgs_decoder_version": PGS_DECODER_VERSION,
            "vobsub_decoder_version": VOBSUB_DECODER_VERSION,
            "source_fingerprint": src_fingerprint,
            "source_path": str(p_video),
            "inventory_hash": inv_hash,
            "inventory_items": inv_items,
            "selected_track_id": chosen_track_id,
            "selected_method": chosen_method,
            "subtitle_candidates": [track.track_id for track in candidate_tracks],
            "allow_stt": self.allow_stt,
            "allow_bitmap_ocr": self.allow_bitmap_ocr,
            "target_hash": chosen_hash,
            "has_audio": audio_sel.has_audio,
            "audio_global_index": audio_sel.global_index,
            "ocr_version": ocr_manifest_hash if chosen_method == "ocr" else "none",
            "stt_version": stt_manifest_hash if chosen_method == "stt" else "none",
            "device_policy": device_policy,
            "stt_window_policy": STT_WINDOW_POLICY,
            "stt_extraction_policy": STT_EXTRACTION_POLICY,
        }

        cache_key = compute_pipeline_cache_key(
            episode_id=ep_id,
            source_fingerprint=src_fingerprint,
            inventory_hash=inv_hash,
            selected_track_id=chosen_track_id,
            selected_method=chosen_method,
            target_hash=chosen_hash,
            pipeline_version=PIPELINE_VERSION,
            parser_version=SUBTITLE_PARSER_VERSION,
            normalization_version=NORMALIZATION_VERSION,
            pgs_decoder_version=PGS_DECODER_VERSION,
            vobsub_decoder_version=VOBSUB_DECODER_VERSION,
            ocr_manifest_hash=ocr_manifest_hash,
            stt_manifest_hash=stt_manifest_hash,
            device_policy=device_policy,
        )

        # 5. Check cache against fresh inventory
        if not force_refresh:
            _report("cache_check", 0.25, "Checking verified atomic cache")
            cached_data = self.cache_manager.load_artifact(cache_key)
            if cached_data is not None:
                try:
                    cached_ep = PreparedEpisode.from_dict(cached_data)
                    cached_dep = cached_ep.dependency_signature

                    # Verify inventory validation conditions
                    is_src_valid = (cached_ep.source_fingerprint == src_fingerprint)
                    is_dep_valid = cached_dep == dep_sig

                    # The cache key includes the complete ordered subtitle
                    # candidate list and each candidate's content identity.
                    if is_src_valid and is_dep_valid:
                        _report("complete", 1.0, "Loaded prepared episode from verified cache")
                        return cached_ep
                except Exception:
                    # Corruption / schema decode failure: treat as cache miss
                    pass

        if cancellation_token:
            cancellation_token.check_cancelled()

        # 6. Execute Subtitle Extraction / OCR or STT Fallback
        transcript: Transcript
        transcript_method: str
        selected_subtitle: SubtitleTrack | None = None
        fallback: bool = False
        uncertainty: bool = False
        status: str = "ready"
        transcript_diagnostics: list[str] = []

        if chosen_track is not None:
            # Case A: English Full subtitle candidate available
            failed_tracks: list[str] = []
            for candidate in candidate_tracks:
                if cancellation_token:
                    cancellation_token.check_cancelled()
                _report("subtitles", 0.45, f"Processing subtitle track: {candidate.track_id}")
                if candidate.is_bitmap and not self.allow_bitmap_ocr:
                    failed_tracks.append(f"{candidate.track_id}: bitmap OCR is disabled")
                    continue
                if candidate.is_bitmap and type(self.ocr_adapter) is OcrAdapter:
                    if not self.ocr_adapter.is_package_installed():
                        failed_tracks.append(f"{candidate.track_id}: local OCR runtime is unavailable")
                        continue
                    if not self.ocr_adapter.is_engine_ready():
                        try:
                            _report("ocr_models", 0.45, "Preparing verified local OCR models for bitmap subtitles...")
                            self.ocr_adapter.model_manager.download_models(cancellation_token=cancellation_token)
                        except CancelledError:
                            raise
                        except Exception as exc:
                            failed_tracks.append(f"{candidate.track_id}: local OCR model unavailable: {exc}")
                            continue
                sub_res: SubtitlePipelineResult = self.subtitle_pipeline.extract_cues(
                    candidate,
                    source_video=p_video,
                    episode_id=ep_id,
                    source_fingerprint=src_fingerprint,
                    source_duration_ms=probe_res.duration_ms,
                    cancellation_token=cancellation_token,
                )
                if sub_res.status == "success" and sub_res.has_cues:
                    chosen_track = candidate
                    break
                failed_tracks.append(
                    f"{candidate.track_id}: {sub_res.status}: "
                    f"{'; '.join(sub_res.diagnostics) or 'no valid subtitle cues'}"
                )
            else:
                if not self.allow_stt:
                    raise ToolRecapError(
                        "No usable English subtitle track. " + "; ".join(failed_tracks)
                    )
                sub_res = SubtitlePipelineResult(
                    track=chosen_track, status="failed", diagnostics=tuple(failed_tracks),
                )

            if sub_res.status == "success" and sub_res.has_cues:
                selected_subtitle = chosen_track
                transcript_method = "ocr" if chosen_track.is_bitmap else chosen_track.source_type
                transcript = build_transcript(
                    episode_id=ep_id,
                    source_type=sub_res.source_type,
                    source_format=sub_res.source_format,
                    raw_cues=sub_res.cues,
                    language=sub_res.language,
                    has_speech=True,
                    source_duration_ms=probe_res.duration_ms,
                    initial_diagnostics=sub_res.diagnostics,
                )
                status = "ready"
                fallback = False
                uncertainty = False
            else:
                # Subtitle extraction produced 0 cues or failed
                if sub_res.status == "failed":
                    diag_err = "; ".join(sub_res.diagnostics) if sub_res.diagnostics else "Unknown error"
                else:
                    diag_err = "Subtitle track yielded 0 valid cues"
                transcript_diagnostics.append(f"Subtitle extraction fallback: {diag_err}")

                # Check if audio is available for STT fallback
                if audio_sel.has_audio:
                    _report("audio_stt", 0.65, f"{diag_err}; falling back to audio STT")
                    stt_res: SttResult = self.transcribe_fn(
                        video_path=p_video,
                        selection=audio_sel,
                        duration_ms=probe_res.duration_ms,
                        model_manager=self.stt_model_manager,
                        device_policy=device_policy,
                        cancellation_token=cancellation_token,
                    )
                    transcript_method = "stt"
                    fallback = True

                    if stt_res.status in (SttStatus.CANCELLED, "cancelled"):
                        raise CancelledError("STT transcription was cancelled")
                    elif stt_res.status in (SttStatus.NO_AUDIO, "no_audio"):
                        status = "no_audio"
                        transcript_diagnostics.extend(stt_res.diagnostics)
                        transcript = build_transcript(
                            episode_id=ep_id,
                            source_type="stt",
                            source_format="whisper",
                            raw_cues=[],
                            language="eng",
                            has_speech=False,
                            source_duration_ms=probe_res.duration_ms,
                            initial_diagnostics=transcript_diagnostics,
                        )
                        uncertainty = True
                    elif stt_res.status in (SttStatus.SILENT, "silent"):
                        status = "silent"
                        transcript_diagnostics.extend(stt_res.diagnostics)
                        transcript = build_transcript(
                            episode_id=ep_id,
                            source_type="stt",
                            source_format="whisper",
                            raw_cues=[],
                            language="eng",
                            has_speech=False,
                            source_duration_ms=probe_res.duration_ms,
                            initial_diagnostics=transcript_diagnostics,
                        )
                        uncertainty = False
                    elif stt_res.status in (SttStatus.SUCCESS_EMPTY, "success_empty"):
                        status = "empty_transcript"
                        transcript_diagnostics.extend(stt_res.diagnostics)
                        transcript = build_transcript(
                            episode_id=ep_id,
                            source_type="stt",
                            source_format="whisper",
                            raw_cues=[],
                            language="eng",
                            has_speech=False,
                            source_duration_ms=probe_res.duration_ms,
                            initial_diagnostics=transcript_diagnostics,
                        )
                        uncertainty = False
                    elif stt_res.status in (SttStatus.SUCCESS, "success"):
                        status = "ready"
                        transcript_diagnostics.extend(stt_res.diagnostics)
                        if audio_sel.has_warning:
                            transcript_diagnostics.append(audio_sel.warning or audio_sel.reason)
                        transcript = build_transcript(
                            episode_id=ep_id,
                            source_type="stt",
                            source_format="whisper",
                            raw_cues=stt_res.cues,
                            language="eng",
                            has_speech=bool(stt_res.cues),
                            source_duration_ms=probe_res.duration_ms,
                            initial_diagnostics=transcript_diagnostics,
                        )
                        uncertainty = bool(audio_sel.has_warning)
                    elif stt_res.status in (SttStatus.FAILED, SttStatus.RUNTIME_UNAVAILABLE, "failed", "runtime_unavailable"):
                        status = "failed"
                        err_msg = stt_res.error or "; ".join(stt_res.diagnostics) or "STT failed"
                        raise ToolRecapError(
                            f"Subtitle extraction and STT fallback both failed: {err_msg}"
                        )
                    else:
                        status = "ready"
                        transcript_diagnostics.extend(stt_res.diagnostics)
                        transcript = build_transcript(
                            episode_id=ep_id,
                            source_type="stt",
                            source_format="whisper",
                            raw_cues=stt_res.cues,
                            language="eng",
                            has_speech=bool(stt_res.cues),
                            source_duration_ms=probe_res.duration_ms,
                            initial_diagnostics=transcript_diagnostics,
                        )
                        uncertainty = True
                else:
                    # Subtitle had no cues and no audio stream exists
                    status = "no_audio"
                    transcript_method = "none"
                    transcript_diagnostics.append("No audio streams available for STT fallback")
                    transcript = build_transcript(
                        episode_id=ep_id,
                        source_type="empty",
                        source_format="none",
                        raw_cues=[],
                        language="und",
                        has_speech=False,
                        source_duration_ms=probe_res.duration_ms,
                        initial_diagnostics=transcript_diagnostics,
                    )
                    fallback = False
                    uncertainty = True

        else:
            # Case B: No English Full subtitle track discovered -> direct STT fallback
            selected_subtitle = None
            if self.allow_stt is False and audio_sel.has_audio:
                inventory = ", ".join(
                    f"{track.track_id} (language={track.language}, format={track.source_format}, forced={track.is_forced})"
                    for track in all_tracks
                ) or "none"
                probe_error = f" Probe error: {probe_res.subtitle_probe_error}." if probe_res.subtitle_probe_error else ""
                if blocked_bitmap_tracks:
                    blocked = ", ".join(track.track_id for track in blocked_bitmap_tracks)
                    raise ToolRecapError(
                        f"No usable English text subtitle track was found. Bitmap track(s) {blocked} "
                        "require OCR, which is disabled in V5."
                    )
                raise ToolRecapError(
                    f"No English full subtitle track with extractable text was found. Available tracks: {inventory}.{probe_error} "
                    "Source preparation cannot continue without extractable subtitles."
                )
            if not audio_sel.has_audio:
                # No audio streams in source container
                status = "no_audio"
                transcript_method = "none"
                transcript = build_transcript(
                    episode_id=ep_id,
                    source_type="empty",
                    source_format="none",
                    raw_cues=[],
                    language="und",
                    has_speech=False,
                    source_duration_ms=probe_res.duration_ms,
                    initial_diagnostics=("No subtitle tracks and container has no audio streams",),
                )
                fallback = False
                uncertainty = True
            else:
                _report("audio_stt", 0.65, "Transcribing audio via STT fallback")
                stt_res = self.transcribe_fn(
                    video_path=p_video,
                    selection=audio_sel,
                    duration_ms=probe_res.duration_ms,
                    model_manager=self.stt_model_manager,
                    device_policy=device_policy,
                    cancellation_token=cancellation_token,
                )
                transcript_method = "stt"
                fallback = True

                if stt_res.status in (SttStatus.CANCELLED, "cancelled"):
                    raise CancelledError("STT transcription was cancelled")
                elif stt_res.status in (SttStatus.NO_AUDIO, "no_audio"):
                    status = "no_audio"
                    transcript = build_transcript(
                        episode_id=ep_id,
                        source_type="stt",
                        source_format="whisper",
                        raw_cues=[],
                        language="eng",
                        has_speech=False,
                        source_duration_ms=probe_res.duration_ms,
                        initial_diagnostics=stt_res.diagnostics,
                    )
                    uncertainty = True
                elif stt_res.status in (SttStatus.SILENT, "silent"):
                    status = "silent"
                    transcript = build_transcript(
                        episode_id=ep_id,
                        source_type="stt",
                        source_format="whisper",
                        raw_cues=[],
                        language="eng",
                        has_speech=False,
                        source_duration_ms=probe_res.duration_ms,
                        initial_diagnostics=stt_res.diagnostics,
                    )
                    uncertainty = False
                elif stt_res.status in (SttStatus.SUCCESS_EMPTY, "success_empty"):
                    status = "empty_transcript"
                    transcript = build_transcript(
                        episode_id=ep_id,
                        source_type="stt",
                        source_format="whisper",
                        raw_cues=[],
                        language="eng",
                        has_speech=False,
                        source_duration_ms=probe_res.duration_ms,
                        initial_diagnostics=stt_res.diagnostics,
                    )
                    uncertainty = False
                elif stt_res.status in (SttStatus.SUCCESS, "success"):
                    status = "ready"
                    initial_diag = list(stt_res.diagnostics)
                    if audio_sel.has_warning:
                        initial_diag.append(audio_sel.warning or audio_sel.reason)
                    transcript = build_transcript(
                        episode_id=ep_id,
                        source_type="stt",
                        source_format="whisper",
                        raw_cues=stt_res.cues,
                        language="eng",
                        has_speech=bool(stt_res.cues),
                        source_duration_ms=probe_res.duration_ms,
                        initial_diagnostics=initial_diag,
                    )
                    uncertainty = bool(audio_sel.has_warning)
                elif stt_res.status in (SttStatus.FAILED, SttStatus.RUNTIME_UNAVAILABLE, "failed", "runtime_unavailable"):
                    status = "failed"
                    err_msg = stt_res.error or "; ".join(stt_res.diagnostics) or "STT failed"
                    raise ToolRecapError(f"STT fallback failed ({stt_res.status}): {err_msg}")
                else:
                    status = "ready"
                    transcript = build_transcript(
                        episode_id=ep_id,
                        source_type="stt",
                        source_format="whisper",
                        raw_cues=stt_res.cues,
                        language="eng",
                        has_speech=bool(stt_res.cues),
                        source_duration_ms=probe_res.duration_ms,
                        initial_diagnostics=stt_res.diagnostics,
                    )
                    uncertainty = True

        if cancellation_token:
            cancellation_token.check_cancelled()

        # 7. Compute deterministic artifact hash
        artifact_raw = (
            f"ep:{ep_id}|src:{src_fingerprint}|t:{transcript.provenance_hash}|"
            f"m:{transcript_method}|status:{status}"
        )
        artifact_hash = hashlib.sha256(artifact_raw.encode("utf-8")).hexdigest()

        # 8. Instantiate PreparedEpisode
        prepared_ep = PreparedEpisode(
            episode_id=ep_id,
            source_id=src_id,
            source_path=p_video,
            duration_ms=probe_res.duration_ms,
            canvas_width=probe_res.canvas_width,
            canvas_height=probe_res.canvas_height,
            source_basename=p_video.name,
            source_fingerprint=src_fingerprint,
            video_streams=probe_res.video_streams,
            audio_streams=probe_res.audio_streams,
            subtitle_streams=probe_res.subtitle_streams,
            primary_video=probe_res.primary_video,
            primary_audio=probe_res.primary_audio,
            video_info=probe_res.primary_video,
            audio_info=audio_sel.selected_stream if audio_sel.has_audio else None,
            audio_selection=audio_sel,
            selected_audio=audio_sel.selected_stream if audio_sel.has_audio else None,
            audio_ordinal=audio_sel.audio_ordinal if audio_sel.has_audio else None,
            selected_subtitle=selected_subtitle,
            transcript=transcript,
            transcript_method=transcript_method,
            artifact_hash=artifact_hash,
            dependency_signature=dep_sig,
            status=status,
            uncertainty=uncertainty,
            fallback=fallback,
            source_hash=src_fingerprint,
            source_timeline={**timeline_assessment.to_dict(), **timing_details},
        )

        # 9. Atomic persistence to cache
        if status != "failed":
            _report("caching", 0.95, "Saving atomic prepared episode artifact")
            self.cache_manager.save_artifact(
                key=cache_key,
                data=prepared_ep.to_dict(),
                meta={
                    "episode_id": ep_id,
                    "source_fingerprint": src_fingerprint,
                    "method": transcript_method,
                    "status": status,
                },
                cancellation_token=cancellation_token,
            )

        _report("complete", 1.0, "Source preparation complete")
        return prepared_ep


def prepare_episode_source(
    source_path: Path | str,
    episode_id: str = "",
    source_id: str = "",
    *,
    source_fingerprint: str | None = None,
    cache_manager: AnalysisCacheManager | None = None,
    subtitle_pipeline: SubtitlePipeline | None = None,
    stt_model_manager: SttModelManager | None = None,
    ocr_adapter: OcrAdapter | None = None,
    probe_fn: Any = probe_episode_source,
    transcribe_fn: Any = transcribe_episode_stt,
    on_progress: Callable[[str, float, str], None] | None = None,
    cancellation_token: CancellationToken | None = None,
    force_refresh: bool = False,
) -> PreparedEpisode:
    """Convenience functional wrapper for SourcePreparationPipeline.prepare_episode."""
    pipeline = SourcePreparationPipeline(
        cache_manager=cache_manager,
        subtitle_pipeline=subtitle_pipeline,
        stt_model_manager=stt_model_manager,
        ocr_adapter=ocr_adapter,
        probe_fn=probe_fn,
        transcribe_fn=transcribe_fn,
    )
    return pipeline.prepare_episode(
        source_path=source_path,
        episode_id=episode_id,
        source_id=source_id,
        source_fingerprint=source_fingerprint,
        force_refresh=force_refresh,
        on_progress=on_progress,
        cancellation_token=cancellation_token,
    )
