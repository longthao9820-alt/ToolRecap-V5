"""Content-addressable subtitle cue cache with atomic persistence and dependency verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from toolrecap_v4.analysis.cache import AnalysisCacheManager, default_analysis_cache_dir
from toolrecap_v4.analysis.dependencies import (
    PGS_DECODER_VERSION,
    SUBTITLE_PARSER_VERSION,
    VOBSUB_DECODER_VERSION,
    compute_sidecar_signature,
    compute_vobsub_signature,
    hash_file_content,
)
from toolrecap_v4.analysis.source_prep.subtitles.ocr import DEFAULT_OCR_MANIFEST
from toolrecap_v4.cancellation import CancellationToken
from .models import SubtitleCue, SubtitleTrack


def compute_subtitle_cache_key(
    episode_id: str,
    source_video: str | Path,
    track: SubtitleTrack,
    *,
    source_fingerprint: str | None = None,
    parser_version: str = SUBTITLE_PARSER_VERSION,
    pgs_decoder_version: str = PGS_DECODER_VERSION,
    vobsub_decoder_version: str = VOBSUB_DECODER_VERSION,
    ocr_manifest: Any = None,
) -> str:
    """Compute deterministic cache key content-addressed to actual sidecar/source bytes and semantic versions.

    Invariants:
    - Never relies solely on mtime/size for sidecars or embedded; hashes actual file content.
    - Hashes both .idx and .sub bytes for VobSub sidecars.
    - Incorporates parser/decoder/OCR versions.
    """
    payload: dict[str, Any] = {
        "episode_id": episode_id,
        "source_video": str(Path(source_video).resolve()) if source_video else "",
        "source_type": track.source_type,
        "source_format": track.source_format,
        "language": track.language,
        "stream_index": track.stream_index,
        "is_forced": track.is_forced,
        "parser_version": parser_version,
    }

    if track.is_bitmap:
        if track.source_format in ("pgs", "sup"):
            payload["decoder_version"] = pgs_decoder_version
        elif track.source_format in ("vobsub", "idx"):
            payload["decoder_version"] = vobsub_decoder_version
        manifest = ocr_manifest or DEFAULT_OCR_MANIFEST
        payload["ocr_version"] = (
            manifest.compute_manifest_hash()
            if hasattr(manifest, "compute_manifest_hash")
            else str(manifest)
        )

    if track.source_type == "sidecar" and track.source_file:
        p_sidecar = Path(track.source_file).resolve()
        if p_sidecar.suffix.lower() == ".idx":
            # VobSub pair
            try:
                sig = compute_vobsub_signature(p_sidecar)
                payload["vobsub_pair_hash"] = sig["pair_hash"]
            except FileNotFoundError:
                payload["vobsub_missing"] = True
        else:
            try:
                sig = compute_sidecar_signature(p_sidecar)
                payload["sidecar_content_hash"] = sig["content_hash"]
            except FileNotFoundError:
                payload["sidecar_missing"] = True
    elif track.source_type == "embedded":
        if source_fingerprint and len(source_fingerprint) >= 32:
            payload["source_content_hash"] = source_fingerprint
        elif source_video:
            p_vid = Path(source_video).resolve()
            if p_vid.is_file():
                payload["source_content_hash"] = hash_file_content(p_vid)

    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SubtitleCacheManager:
    """Manages independent atomic caching of SubtitleCue lists per episode."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        sub_dir = Path(cache_dir) if cache_dir else (default_analysis_cache_dir() / "subtitles")
        self._cache = AnalysisCacheManager(sub_dir)

    def save_cues(
        self,
        episode_id: str,
        source_video: str | Path,
        track: SubtitleTrack,
        cues: Sequence[SubtitleCue],
        *,
        source_fingerprint: str | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Path:
        """Atomically persist parsed cues for an episode track."""
        key = compute_subtitle_cache_key(
            episode_id, source_video, track, source_fingerprint=source_fingerprint
        )
        data = {
            "episode_id": episode_id,
            "source_video": str(Path(source_video).resolve()) if source_video else "",
            "track_id": track.track_id,
            "source_type": track.source_type,
            "source_format": track.source_format,
            "language": track.language,
            "cues": [c.to_dict() for c in cues],
        }
        meta = {
            "cue_count": len(cues),
            "track_id": track.track_id,
        }
        return self._cache.save_artifact(
            key=key,
            data=data,
            meta=meta,
            cancellation_token=cancellation_token,
        )

    def load_cues(
        self,
        episode_id: str,
        source_video: str | Path,
        track: SubtitleTrack,
        *,
        source_fingerprint: str | None = None,
    ) -> list[SubtitleCue] | None:
        """Load cached cues if verified manifest is COMPLETE and content hash matches.
        
        Returns None on cache miss or corrupted/tampered cache.
        """
        key = compute_subtitle_cache_key(
            episode_id, source_video, track, source_fingerprint=source_fingerprint
        )
        data = self._cache.load_artifact(key)
        if data is None:
            return None

        cues_raw = data.get("cues", [])
        return [SubtitleCue.from_dict(c) for c in cues_raw]

    def has_cues(
        self,
        episode_id: str,
        source_video: str | Path,
        track: SubtitleTrack,
        *,
        source_fingerprint: str | None = None,
    ) -> bool:
        """Check if uncorrupted cache exists for the specified track."""
        return self.load_cues(
            episode_id, source_video, track, source_fingerprint=source_fingerprint
        ) is not None

    def invalidate(
        self,
        episode_id: str,
        source_video: str | Path,
        track: SubtitleTrack,
        *,
        source_fingerprint: str | None = None,
    ) -> bool:
        """Remove cache artifact and manifest for track."""
        key = compute_subtitle_cache_key(
            episode_id, source_video, track, source_fingerprint=source_fingerprint
        )
        return self._cache.invalidate(key)
