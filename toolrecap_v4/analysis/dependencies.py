"""Dependency fingerprinting, content-addressable file hashing, and cache invalidation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError


def hash_file_content(path: Path | str, chunk_size: int = 65536) -> str:
    """Compute deterministic SHA-256 hash of a file's content in bounded streaming chunks."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found for content hashing: {p}")

    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_bytes_content(data: bytes) -> str:
    """Compute deterministic SHA-256 hash of byte buffer."""
    return hashlib.sha256(data).hexdigest()


def compute_sidecar_signature(sidecar_path: Path | str) -> dict[str, Any]:
    """Compute content-addressable dependency signature for a text sidecar subtitle file.
    
    Invariants:
    - Never keys purely on mtime/size; hashes actual file bytes.
    """
    p = Path(sidecar_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Sidecar subtitle file not found: {p}")

    stat = p.stat()
    content_hash = hash_file_content(p)

    return {
        "type": "sidecar",
        "path": str(p),
        "name": p.name,
        "size": stat.st_size,
        "mtime": round(stat.st_mtime, 3),
        "content_hash": content_hash,
    }


def compute_vobsub_signature(
    idx_path: Path | str,
    sub_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute content-addressable dependency signature for a paired VobSub (.idx / .sub) source.
    
    Invariants:
    - Hashes BOTH .idx and .sub file bytes.
    - Yields a combined pair_hash representing the composite identity of both files.
    """
    p_idx = Path(idx_path).resolve()
    if not p_idx.is_file():
        raise FileNotFoundError(f"VobSub index file (.idx) not found: {p_idx}")

    p_sub = Path(sub_path).resolve() if sub_path else p_idx.with_suffix(".sub")
    if not p_sub.is_file():
        raise FileNotFoundError(f"Paired VobSub sub-picture file (.sub) not found: {p_sub}")

    idx_stat = p_idx.stat()
    sub_stat = p_sub.stat()

    idx_hash = hash_file_content(p_idx)
    sub_hash = hash_file_content(p_sub)

    pair_hasher = hashlib.sha256()
    pair_hasher.update(f"idx:{idx_hash}|sub:{sub_hash}".encode("utf-8"))
    pair_hash = pair_hasher.hexdigest()

    return {
        "type": "vobsub_pair",
        "idx_path": str(p_idx),
        "sub_path": str(p_sub),
        "idx_size": idx_stat.st_size,
        "sub_size": sub_stat.st_size,
        "idx_mtime": round(idx_stat.st_mtime, 3),
        "sub_mtime": round(sub_stat.st_mtime, 3),
        "idx_content_hash": idx_hash,
        "sub_content_hash": sub_hash,
        "pair_hash": pair_hash,
    }


def compute_embedded_track_signature(
    video_path: Path | str,
    stream_index: int,
    codec: str,
    *,
    video_stat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute dependency signature for an embedded subtitle or audio stream."""
    p_video = Path(video_path).resolve()
    if video_stat is not None:
        size = int(video_stat.get("size", 0))
        mtime = float(video_stat.get("mtime", 0.0))
    elif p_video.is_file():
        st = p_video.stat()
        size = st.st_size
        mtime = round(st.st_mtime, 3)
    else:
        size = 0
        mtime = 0.0

    raw = f"video:{p_video}|size:{size}|mtime:{mtime}|stream:{stream_index}|codec:{codec}"
    stream_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return {
        "type": "embedded",
        "video_path": str(p_video),
        "stream_index": stream_index,
        "codec": codec,
        "video_size": size,
        "video_mtime": mtime,
        "stream_hash": stream_hash,
    }


def compute_source_signature(
    video_path: Path | str,
    *,
    full_content_hash: bool = False,
) -> dict[str, Any]:
    """Compute source video file signature based on stats and optional content hash."""
    p = Path(video_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Source video file not found: {p}")

    stat = p.stat()
    sig: dict[str, Any] = {
        "path": str(p),
        "name": p.name,
        "size": stat.st_size,
        "mtime": round(stat.st_mtime, 3),
    }

    if full_content_hash:
        sig["content_hash"] = hash_file_content(p)
    else:
        # Fast composite hash of path + size + mtime
        raw = f"{p}|{stat.st_size}|{sig['mtime']}"
        sig["source_hash"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return sig


def compute_ocr_dependency_signature(
    manifest: Any,
    extra_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute content-addressable dependency signature for OCR runtime and models."""
    settings = dict(extra_settings or {})
    manifest_hash = manifest.compute_manifest_hash() if hasattr(manifest, "compute_manifest_hash") else str(manifest)
    settings_raw = repr(sorted(settings.items()))
    combined = f"ocr_manifest:{manifest_hash}|settings:{settings_raw}"
    ocr_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

    return {
        "type": "ocr",
        "runtime": getattr(manifest, "runtime", "rapidocr"),
        "version": getattr(manifest, "version", "unknown"),
        "manifest_hash": manifest_hash,
        "settings": settings,
        "ocr_hash": ocr_hash,
    }


def compute_stt_dependency_signature(
    audio_selection: Any,
    manifest: Any,
    *,
    device_policy: str = "auto",
    video_path: Path | str | None = None,
    video_stat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute content-addressable dependency signature for STT transcription."""
    manifest_hash = manifest.compute_manifest_hash() if hasattr(manifest, "compute_manifest_hash") else str(manifest)
    codec = "none"
    if hasattr(audio_selection, "selected_stream") and audio_selection.selected_stream is not None:
        codec = getattr(audio_selection.selected_stream, "codec", "none")
    global_index = getattr(audio_selection, "global_index", -1)
    audio_ordinal = getattr(audio_selection, "audio_ordinal", -1)

    sig_payload = {
        "type": "stt",
        "audio_global_index": global_index,
        "audio_ordinal": audio_ordinal,
        "audio_codec": codec,
        "model_name": getattr(manifest, "model_name", "whisper"),
        "repo_id": getattr(manifest, "repo_id", ""),
        "revision": getattr(manifest, "revision", ""),
        "settings": getattr(manifest, "settings", {}),
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


def is_signature_valid(expected_sig: dict[str, Any], current_sig: dict[str, Any]) -> bool:
    """Compare two dependency signatures for exact identity match.
    
    If content hashes are present in both signatures, hashes must match.
    Otherwise size and mtime must match.
    """
    if not expected_sig or not current_sig:
        return False

    # Check OCR hash
    if "ocr_hash" in expected_sig and "ocr_hash" in current_sig:
        return expected_sig["ocr_hash"] == current_sig["ocr_hash"]

    # Check STT hash
    if "stt_hash" in expected_sig and "stt_hash" in current_sig:
        return expected_sig["stt_hash"] == current_sig["stt_hash"]

    # Check pair hash for VobSub
    if "pair_hash" in expected_sig and "pair_hash" in current_sig:
        return expected_sig["pair_hash"] == current_sig["pair_hash"]

    # Check content hash if available
    if "content_hash" in expected_sig and "content_hash" in current_sig:
        return expected_sig["content_hash"] == current_sig["content_hash"]

    # Fallback to size and mtime comparison
    if "size" in expected_sig and "size" in current_sig:
        if expected_sig["size"] != current_sig["size"]:
            return False

    if "mtime" in expected_sig and "mtime" in current_sig:
        return abs(expected_sig["mtime"] - current_sig["mtime"]) < 0.001

    # Check stream hash for embedded
    if "stream_hash" in expected_sig and "stream_hash" in current_sig:
        return expected_sig["stream_hash"] == current_sig["stream_hash"]

    return False


PIPELINE_VERSION = "1.0.0"
SUBTITLE_PARSER_VERSION = "1.0.0"
NORMALIZATION_VERSION = "1.0.0"
PGS_DECODER_VERSION = "1.0.0"
VOBSUB_DECODER_VERSION = "1.0.0"

STT_EXTRACTION_POLICY = {
    "sample_rate": 16000,
    "channels": 1,
    "format": "wav",
}
STT_WINDOW_POLICY = {
    "window_duration_sec": 60.0,
    "overlap_sec": 2.0,
    "min_rms_threshold": 80.0,
}


def compute_source_fingerprint(
    video_path: Path | str,
    supplied_hash: str | None = None,
) -> str:
    """Compute content-addressable SHA-256 fingerprint for source video.

    Invariants:
    - If supplied full SHA-256 is passed, uses it directly.
    - Otherwise hashes full content bytes.
    - Never uses path/mtime only.
    """
    if supplied_hash and len(supplied_hash) >= 32:
        return supplied_hash
    p = Path(video_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Source video file not found for fingerprint: {p}")
    return hash_file_content(p)


def compute_inventory_hash(
    sidecars: Sequence[Any],
    *,
    cancellation_token: CancellationToken | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Compute deterministic composite inventory hash from matching sidecar subtitle tracks.

    Invariants:
    - Never swallows CancelledError.
    - Malformed or missing pairs remain in inventory signature as explicit diagnostic entries.
    """
    if cancellation_token:
        cancellation_token.check_cancelled()

    items: list[dict[str, Any]] = []
    for s in sidecars:
        if cancellation_token:
            cancellation_token.check_cancelled()

        file_path = getattr(s, "source_file", None) or getattr(s, "path", None)
        if not file_path:
            continue
        p = Path(file_path).resolve()
        fmt = getattr(s, "source_format", None) or p.suffix.lstrip(".").lower()
        score = float(getattr(s, "score", 0.0))

        if not p.is_file():
            items.append({
                "name": p.name,
                "hash": "missing_file",
                "format": fmt,
                "error": f"File not found: {p}",
                "score": score,
            })
            continue

        if fmt == "vobsub" or p.suffix.lower() == ".idx":
            try:
                sig = compute_vobsub_signature(p)
                items.append({
                    "name": p.name,
                    "hash": sig["pair_hash"],
                    "format": "vobsub",
                    "score": score,
                })
            except CancelledError:
                raise
            except Exception as e:
                items.append({
                    "name": p.name,
                    "hash": "malformed_pair",
                    "format": "vobsub",
                    "error": str(e),
                    "score": score,
                })
        else:
            try:
                sig = compute_sidecar_signature(p)
                items.append({
                    "name": p.name,
                    "hash": sig["content_hash"],
                    "format": fmt,
                    "score": score,
                })
            except CancelledError:
                raise
            except Exception as e:
                items.append({
                    "name": p.name,
                    "hash": "malformed_sidecar",
                    "format": fmt,
                    "error": str(e),
                    "score": score,
                })

    items.sort(key=lambda x: x["name"])
    raw = json.dumps(items, sort_keys=True)
    inv_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return inv_hash, items

