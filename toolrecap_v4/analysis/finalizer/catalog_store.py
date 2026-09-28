"""Atomic Catalog checkpoint, cache validation, and capacity preflight."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer import catalog as catalog_module
from toolrecap_v4.analysis.finalizer import packing as packing_module
from toolrecap_v4.analysis.finalizer.catalog import (
    CapacityPreflight,
    CatalogBuilder,
    SeasonEvidenceCatalog,
    canonical_catalog_bytes,
    validate_catalog,
    verify_catalog_against_evidence,
)
from toolrecap_v4.analysis.finalizer.packing import pack_catalog, packed_catalog_bytes, unpack_catalog
from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CatalogCapacityError, CatalogPackingError, CatalogStoreError, CatalogValidationError
from toolrecap_v4.persistence import atomic_write_json
from toolrecap_v4.validator import validate_windows_name


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def compute_catalog_dependencies(
    *,
    project_id: str,
    evidence_revision: str,
    ordered_episodes: Sequence[PreparedEpisode],
    revision_manifest: dict[str, Any],
) -> dict[str, Any]:
    records = {record["episode_id"]: record for record in revision_manifest.get("episodes", [])}
    episodes = sorted(ordered_episodes, key=lambda item: item.episode_id)
    return {
        "project_id": project_id,
        "evidence_revision": evidence_revision,
        "ordered_episodes": [{
            "episode_id": episode.episode_id,
            "source_id": episode.source_id,
            "source_basename": episode.source_basename or episode.source_path.name,
            "duration_ms": episode.duration_ms,
            "evidence_manifest_hash": records.get(episode.episode_id, {}).get("data_hash"),
        } for episode in episodes],
        "catalog_version": catalog_module.CATALOG_VERSION,
        "catalog_projection_version": catalog_module.CATALOG_PROJECTION_VERSION,
        "packing_version": packing_module.PACKING_VERSION,
    }


@dataclass(frozen=True)
class CatalogBuildResult:
    catalog: SeasonEvidenceCatalog
    packed: dict[str, Any]
    capacity: CapacityPreflight
    dependency_digest: str
    manifest_path: Path
    reused: bool


class CatalogStore:
    """Hash-verified Catalog cache under one managed project directory."""

    def __init__(self, storage_root: Path | str, project_id: str) -> None:
        validate_windows_name(project_id, "project_id")
        self.project_id = project_id
        self.base_dir = Path(storage_root).resolve() / "projects" / project_id / "catalog"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_path(self) -> Path:
        return self.base_dir / "active.json"

    def _artifact_dir(self, catalog_hash: str, packing_version: str) -> Path:
        validate_windows_name(catalog_hash, "catalog_hash")
        validate_windows_name(packing_version, "packing_version")
        # Keep managed paths safely below legacy Windows MAX_PATH. Full identities
        # remain in manifests and are verified on every load.
        catalog_dir = catalog_hash[:32]
        packing_dir = f"pack-{hashlib.sha256(packing_version.encode('utf-8')).hexdigest()[:12]}"
        return self.base_dir / catalog_dir / packing_dir

    def _read_object(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CatalogStoreError(f"Catalog artifact is unreadable: {path}") from exc
        if not isinstance(value, dict):
            raise CatalogStoreError(f"Catalog artifact root is not an object: {path}")
        return value

    def load_matching(self, dependency_digest: str) -> CatalogBuildResult | None:
        if not self.active_path.is_file():
            return None
        try:
            active = self._read_object(self.active_path)
            if active.get("status") != "COMPLETE" or active.get("dependency_digest") != dependency_digest:
                return None
            directory = self._artifact_dir(active["catalog_hash"], active["packing_version"])
            manifest_path = directory / "manifest.json"
            catalog_path = directory / "catalog.json"
            packed_path = directory / "packed.json"
            if not all(path.is_file() for path in (manifest_path, catalog_path, packed_path)):
                return None
            manifest = self._read_object(manifest_path)
            if (
                manifest.get("status") != "COMPLETE"
                or manifest.get("dependency_digest") != dependency_digest
                or manifest.get("catalog_hash") != active["catalog_hash"]
                or manifest.get("packing_version") != active["packing_version"]
            ):
                return None
            catalog_bytes = catalog_path.read_bytes()
            packed_bytes = packed_path.read_bytes()
            if hashlib.sha256(catalog_bytes).hexdigest() != manifest.get("catalog_file_hash"):
                return None
            if hashlib.sha256(packed_bytes).hexdigest() != manifest.get("packed_file_hash"):
                return None
            catalog_raw = json.loads(catalog_bytes.decode("utf-8"))
            packed = json.loads(packed_bytes.decode("utf-8"))
            catalog = SeasonEvidenceCatalog.from_dict(catalog_raw)
            validate_catalog(catalog)
            if unpack_catalog(packed) != catalog:
                return None
            capacity = CapacityPreflight(**manifest["capacity"])
            return CatalogBuildResult(catalog, packed, capacity, dependency_digest, manifest_path, True)
        except (
            CatalogStoreError, CatalogValidationError, CatalogPackingError,
            KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError, OSError,
        ):
            return None

    def save(
        self,
        *,
        catalog: SeasonEvidenceCatalog,
        packed: dict[str, Any],
        capacity: CapacityPreflight,
        dependencies: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> Path:
        validate_catalog(catalog)
        if unpack_catalog(packed) != catalog:
            raise CatalogStoreError("Packed Catalog does not round-trip to the canonical Catalog.")
        if cancellation_token:
            cancellation_token.check_cancelled()
        dependency_digest = _digest(dependencies)
        directory = self._artifact_dir(catalog.catalog_hash, packing_module.PACKING_VERSION)
        directory.mkdir(parents=True, exist_ok=True)
        catalog_path = directory / "catalog.json"
        packed_path = directory / "packed.json"
        manifest_path = directory / "manifest.json"

        # Invalidate any prior manifest before replacing data. Only the final
        # COMPLETE manifest can become a cache hit.
        atomic_write_json(manifest_path, {
            "status": "BUILDING",
            "project_id": self.project_id,
            "catalog_hash": catalog.catalog_hash,
            "dependency_digest": dependency_digest,
        })
        # Data artifacts are atomically replaced first. Only the final COMPLETE manifest is a cache hit.
        atomic_write_json(catalog_path, catalog.to_dict())
        if cancellation_token:
            cancellation_token.check_cancelled()
        atomic_write_json(packed_path, packed)
        if cancellation_token:
            cancellation_token.check_cancelled()
        catalog_file_hash = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        packed_file_hash = hashlib.sha256(packed_path.read_bytes()).hexdigest()
        manifest = {
            "status": "COMPLETE",
            "project_id": self.project_id,
            "catalog_version": catalog.catalog_version,
            "catalog_projection_version": catalog_module.CATALOG_PROJECTION_VERSION,
            "packing_version": packing_module.PACKING_VERSION,
            "catalog_hash": catalog.catalog_hash,
            "evidence_revision": catalog.evidence_revision,
            "dependency_digest": dependency_digest,
            "dependencies": dependencies,
            "catalog_file_hash": catalog_file_hash,
            "packed_file_hash": packed_file_hash,
            "capacity": capacity.to_dict(),
        }
        atomic_write_json(manifest_path, manifest)
        if cancellation_token:
            cancellation_token.check_cancelled()
        atomic_write_json(self.active_path, {
            "status": "COMPLETE",
            "dependency_digest": dependency_digest,
            "catalog_hash": catalog.catalog_hash,
            "packing_version": packing_module.PACKING_VERSION,
        })
        return manifest_path


class CatalogService:
    """Local zero-AI Catalog build, capacity measurement, persistence, and reuse."""

    def __init__(self, storage_root: Path | str, *, configured_capacity_bytes: int | None = None) -> None:
        if configured_capacity_bytes is not None and (
            type(configured_capacity_bytes) is not int or configured_capacity_bytes <= 0
        ):
            raise CatalogCapacityError("Configured Catalog capacity must be a positive integer or null.")
        self.storage_root = Path(storage_root).resolve()
        self.configured_capacity_bytes = configured_capacity_bytes

    def _capacity(self, catalog: SeasonEvidenceCatalog, packed: dict[str, Any]) -> CapacityPreflight:
        canonical_size = len(canonical_catalog_bytes(catalog))
        packed_size = len(packed_catalog_bytes(packed))
        if self.configured_capacity_bytes is None:
            status = "UNKNOWN"
        elif packed_size <= self.configured_capacity_bytes:
            status = "FIT"
        else:
            status = "EXCEEDS_CONFIGURED_LIMIT"
        return CapacityPreflight(
            status=status,
            configured_limit_bytes=self.configured_capacity_bytes,
            canonical_bytes=canonical_size,
            packed_bytes=packed_size,
            episode_count=len(catalog.ordered_episodes),
            item_count=len(catalog.items),
            unique_string_count=len(packed["strings"]),
            unique_entity_count=len(packed["entities"]),
            compression_ratio=(packed_size / canonical_size) if canonical_size else None,
        )

    def build_or_load(
        self,
        *,
        project_id: str,
        evidence_revision: str,
        ordered_episodes: Sequence[PreparedEpisode],
        cancellation_token: CancellationToken | None = None,
    ) -> CatalogBuildResult:
        if cancellation_token:
            cancellation_token.check_cancelled()
        evidence_store = EvidenceStore(self.storage_root, project_id)
        revision_manifest = evidence_store.verify_revision(evidence_revision)
        dependencies = compute_catalog_dependencies(
            project_id=project_id,
            evidence_revision=evidence_revision,
            ordered_episodes=ordered_episodes,
            revision_manifest=revision_manifest,
        )
        dependency_digest = _digest(dependencies)
        store = CatalogStore(self.storage_root, project_id)
        cached = store.load_matching(dependency_digest)
        if cached is not None:
            verify_catalog_against_evidence(cached.catalog, evidence_store)
            current_capacity = self._capacity(cached.catalog, cached.packed)
            return CatalogBuildResult(
                cached.catalog, cached.packed, current_capacity,
                cached.dependency_digest, cached.manifest_path, True,
            )
        catalog = CatalogBuilder().build(
            project_id=project_id,
            evidence_revision=evidence_revision,
            ordered_episodes=ordered_episodes,
            evidence_store=evidence_store,
        )
        packed = pack_catalog(catalog)
        if unpack_catalog(packed) != catalog:
            raise CatalogStoreError("Catalog pack/unpack equality failed.")
        capacity = self._capacity(catalog, packed)
        if cancellation_token:
            cancellation_token.check_cancelled()
        manifest_path = store.save(
            catalog=catalog,
            packed=packed,
            capacity=capacity,
            dependencies=dependencies,
            cancellation_token=cancellation_token,
        )
        return CatalogBuildResult(catalog, packed, capacity, dependency_digest, manifest_path, False)
