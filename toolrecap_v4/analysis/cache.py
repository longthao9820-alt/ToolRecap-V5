"""Atomic, content-verified cache manager with cancellation safety and corruption detection."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, PersistenceError
from toolrecap_v4.persistence import get_storage_root
from toolrecap_v4.validator import check_for_secrets


def default_analysis_cache_dir() -> Path:
    """Return default directory for analysis and source prep caches in LocalAppData."""
    cache_dir = get_storage_root() / "cache" / "analysis"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _ensure_local_absolute_path(path: Path | str) -> Path:
    """Ensure path is absolute and local (non-UNC / non-network)."""
    p = Path(path).resolve()
    if not p.is_absolute():
        raise ValueError(f"Path must be absolute: {p}")
    if os.name == "nt":
        if str(p).startswith(("\\\\", "//")):
            raise ValueError(f"Network UNC paths are forbidden for local cache: {p}")
        if not p.drive:
            raise ValueError(f"Path must contain a local drive letter on Windows: {p}")
    return p


def safe_cache_filename(key: str) -> str:
    """Deterministic, collision-resistant filename incorporating key SHA-256 digest."""
    key_bytes = key.encode("utf-8")
    digest = hashlib.sha256(key_bytes).hexdigest()[:24]
    clean_prefix = "".join(c if c.isalnum() else "_" for c in key[:20]).strip("_")
    return f"{clean_prefix}_{digest}" if clean_prefix else digest


class AnalysisCacheManager:
    """Atomic, content-verified cache manager.
    
    Invariants:
    - Collision-resistant file naming with key digests.
    - Writes data first, verifies data hash and size, then writes manifest marked COMPLETE.
    - Cooperative cancellation checked before promotion; aborts and cleans up temp files.
    - Local absolute paths only.
    - Corrupt, missing, or mismatched manifest/data is detected and treated as a cache miss.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        raw_dir = cache_dir or default_analysis_cache_dir()
        self.cache_dir = _ensure_local_absolute_path(raw_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _data_path(self, safe_name: str) -> Path:
        return self.cache_dir / f"{safe_name}.data.json"

    def _manifest_path(self, safe_name: str) -> Path:
        return self.cache_dir / f"{safe_name}.manifest.json"

    def save_artifact(
        self,
        key: str,
        data: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Path:
        """Atomically persist artifact data and verification manifest with cancellation safety."""
        if cancellation_token:
            cancellation_token.check_cancelled()

        check_for_secrets(data)
        if meta:
            check_for_secrets(meta)

        safe_name = safe_cache_filename(key)
        target_data = self._data_path(safe_name)
        target_manifest = self._manifest_path(safe_name)

        nonce = uuid.uuid4().hex[:8]
        pid = os.getpid()
        tmp_data = self.cache_dir / f"{safe_name}.data.tmp.{pid}.{nonce}"
        tmp_manifest = self.cache_dir / f"{safe_name}.manifest.tmp.{pid}.{nonce}"

        try:
            # 1. Serialize and write data file with fsync
            data_str = json.dumps(data, indent=2, ensure_ascii=False)
            data_bytes = data_str.encode("utf-8")
            data_hash = hashlib.sha256(data_bytes).hexdigest()
            data_size = len(data_bytes)

            with tmp_data.open("wb") as f:
                f.write(data_bytes)
                f.flush()
                os.fsync(f.fileno())

            if cancellation_token:
                cancellation_token.check_cancelled()

            # 2. Serialize and write manifest marked COMPLETE with fsync
            manifest_payload = {
                "status": "COMPLETE",
                "cache_key": key,
                "data_hash": data_hash,
                "data_size": data_size,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "meta": meta or {},
            }
            manifest_bytes = json.dumps(manifest_payload, indent=2, ensure_ascii=False).encode("utf-8")

            with tmp_manifest.open("wb") as f:
                f.write(manifest_bytes)
                f.flush()
                os.fsync(f.fileno())

            # 3. Final cancellation check immediately before atomic promotion
            if cancellation_token:
                cancellation_token.check_cancelled()

            # 4. Atomic promotion: data first, manifest second
            os.replace(tmp_data, target_data)
            os.replace(tmp_manifest, target_manifest)
            return target_data

        except CancelledError:
            for tmp in (tmp_data, tmp_manifest):
                try:
                    if tmp.is_file():
                        tmp.unlink()
                except OSError:
                    pass
            raise

        except Exception as e:
            for tmp in (tmp_data, tmp_manifest):
                try:
                    if tmp.is_file():
                        tmp.unlink()
                except OSError:
                    pass
            raise PersistenceError(f"Failed to save atomic artifact for key '{key}': {e}") from e

    def load_artifact(self, key: str) -> dict[str, Any] | None:
        """Load artifact only if verified manifest is COMPLETE, cache_key matches, size matches, and hash matches.
        
        Returns None on cache miss or any detected corruption/incompletion.
        """
        safe_name = safe_cache_filename(key)
        target_data = self._data_path(safe_name)
        target_manifest = self._manifest_path(safe_name)

        if not target_manifest.is_file() or not target_data.is_file():
            return None

        try:
            # 1. Read and verify manifest
            manifest_raw = target_manifest.read_text(encoding="utf-8")
            manifest = json.loads(manifest_raw)

            if manifest.get("status") != "COMPLETE":
                return None

            # Verify exact cache_key match to guard against safe name collisions
            if manifest.get("cache_key") != key:
                return None

            expected_hash = manifest.get("data_hash")
            if not expected_hash:
                return None

            # 2. Read data and verify size and SHA-256 against manifest
            data_bytes = target_data.read_bytes()

            expected_size = manifest.get("data_size")
            if expected_size is not None and len(data_bytes) != expected_size:
                return None

            actual_hash = hashlib.sha256(data_bytes).hexdigest()
            if actual_hash != expected_hash:
                # Content hash mismatch: corrupted artifact
                return None

            return json.loads(data_bytes.decode("utf-8"))

        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def has_artifact(self, key: str) -> bool:
        """Check if a valid, uncorrupted artifact exists for key."""
        return self.load_artifact(key) is not None

    def invalidate(self, key: str) -> bool:
        """Remove cached artifact and manifest for key if present."""
        safe_name = safe_cache_filename(key)
        target_data = self._data_path(safe_name)
        target_manifest = self._manifest_path(safe_name)
        removed = False

        for target in (target_data, target_manifest):
            try:
                if target.is_file():
                    target.unlink()
                    removed = True
            except OSError:
                pass
        return removed
