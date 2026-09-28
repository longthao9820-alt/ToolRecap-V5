"""Immutable, hash-verified Full Episode Evidence Store for Phase 4."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from toolrecap_v4.analysis.models import Evidence
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import EvidenceRevisionError, EvidenceStoreError
from toolrecap_v4.persistence import atomic_write_json
from toolrecap_v4.validator import validate_windows_name

EVIDENCE_STORE_VERSION = "evidence-store-v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class EvidenceQueryResult:
    items: tuple[Evidence, ...]
    total_count: int
    returned_count: int
    complete: bool
    revision: str
    artifact_hash: str


@dataclass(frozen=True)
class EvidenceCompleteness:
    revision: str
    complete: bool
    expected_episodes: int
    completed_episodes: int
    evidence_count: int


class EvidenceStore:
    """Local immutable JSON artifacts with atomic episode and revision manifests."""

    def __init__(self, storage_root: Path | str, project_id: str) -> None:
        validate_windows_name(project_id, "project_id")
        self.storage_root = Path(storage_root).resolve()
        self.project_id = project_id
        self.base_dir = self.storage_root / "projects" / project_id / "evidence"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _revision_dir(self, revision: str) -> Path:
        validate_windows_name(revision, "evidence_revision")
        return self.base_dir / revision

    def _episode_paths(self, revision: str, episode_id: str) -> tuple[Path, Path]:
        validate_windows_name(episode_id, "episode_id")
        directory = self._revision_dir(revision) / "episodes"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{episode_id}.json", directory / f"{episode_id}.manifest.json"

    def _manifest_path(self, revision: str) -> Path:
        return self._revision_dir(revision) / "manifest.json"

    def save_episode(
        self,
        revision: str,
        episode_id: str,
        evidence: Sequence[Evidence],
        *,
        dependency_digest: str,
        cancellation_token: CancellationToken | None = None,
    ) -> str:
        if cancellation_token:
            cancellation_token.check_cancelled()
        ordered = tuple(sorted(evidence, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id)))
        ids = [item.evidence_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise EvidenceStoreError(f"Duplicate evidence ID in {episode_id}.")
        if any(item.episode_id != episode_id for item in ordered):
            raise EvidenceStoreError(f"Evidence episode mismatch while storing {episode_id}.")
        if any(not item.evidence_id.startswith(f"{episode_id}-EV-") for item in ordered):
            raise EvidenceStoreError(f"Evidence ID prefix mismatch while storing {episode_id}.")
        data = {
            "store_version": EVIDENCE_STORE_VERSION,
            "project_id": self.project_id,
            "revision": revision,
            "episode_id": episode_id,
            "evidence": [item.to_dict() for item in ordered],
        }
        data_hash = _sha256(data)
        data_path, manifest_path = self._episode_paths(revision, episode_id)

        if data_path.exists() or manifest_path.exists():
            existing = self._load_episode_payload(revision, episode_id, require_revision_complete=False)
            if _sha256(existing) != data_hash:
                raise EvidenceRevisionError(
                    f"Revision {revision} already contains different evidence for {episode_id}."
                )
            existing_manifest = self._read_json(manifest_path)
            if existing_manifest.get("dependency_digest") != dependency_digest:
                raise EvidenceRevisionError(f"Revision {revision} dependency mismatch for {episode_id}.")
            return data_hash

        data_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(data_path, data)
        if cancellation_token:
            cancellation_token.check_cancelled()
        manifest = {
            "status": "COMPLETE",
            "store_version": EVIDENCE_STORE_VERSION,
            "project_id": self.project_id,
            "revision": revision,
            "episode_id": episode_id,
            "dependency_digest": dependency_digest,
            "data_hash": data_hash,
            "evidence_count": len(ordered),
        }
        atomic_write_json(manifest_path, manifest)
        return data_hash

    def commit_revision(
        self,
        revision: str,
        expected_episode_ids: Sequence[str],
        *,
        dependency_signature: dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> Path:
        if cancellation_token:
            cancellation_token.check_cancelled()
        if len(expected_episode_ids) != len(set(expected_episode_ids)):
            raise EvidenceStoreError("Expected episode IDs must be unique.")
        episode_records: list[dict[str, Any]] = []
        total = 0
        for episode_id in expected_episode_ids:
            payload = self._load_episode_payload(revision, episode_id, require_revision_complete=False)
            _, ep_manifest_path = self._episode_paths(revision, episode_id)
            ep_manifest = self._read_json(ep_manifest_path)
            episode_records.append({
                "episode_id": episode_id,
                "data_hash": ep_manifest["data_hash"],
                "evidence_count": len(payload["evidence"]),
            })
            total += len(payload["evidence"])
        manifest = {
            "status": "COMPLETE",
            "store_version": EVIDENCE_STORE_VERSION,
            "project_id": self.project_id,
            "revision": revision,
            "dependency_signature": dependency_signature,
            "dependency_digest": _sha256(dependency_signature),
            "episodes": episode_records,
            "expected_episode_count": len(expected_episode_ids),
            "completed_episode_count": len(episode_records),
            "evidence_count": total,
        }
        manifest["revision_artifact_hash"] = _sha256({k: v for k, v in manifest.items() if k != "revision_artifact_hash"})
        target = self._manifest_path(revision)
        if target.exists():
            existing = self._read_json(target)
            if existing != manifest:
                raise EvidenceRevisionError(f"Revision {revision} is immutable and already has different content.")
            return target
        if cancellation_token:
            cancellation_token.check_cancelled()
        atomic_write_json(target, manifest)
        return target

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EvidenceStoreError(f"Evidence artifact is unreadable: {path}") from exc
        if not isinstance(value, dict):
            raise EvidenceStoreError(f"Evidence artifact root is not an object: {path}")
        return value

    def verify_revision(self, revision: str) -> dict[str, Any]:
        path = self._manifest_path(revision)
        if not path.is_file():
            raise EvidenceRevisionError(f"Evidence revision does not exist: {revision}")
        manifest = self._read_json(path)
        if manifest.get("status") != "COMPLETE" or manifest.get("revision") != revision or manifest.get("project_id") != self.project_id:
            raise EvidenceRevisionError(f"Evidence revision is incomplete or mismatched: {revision}")
        expected_hash = manifest.get("revision_artifact_hash")
        actual_hash = _sha256({k: v for k, v in manifest.items() if k != "revision_artifact_hash"})
        if expected_hash != actual_hash:
            raise EvidenceStoreError(f"Evidence revision manifest hash mismatch: {revision}")
        for record in manifest.get("episodes", []):
            payload = self._load_episode_payload(revision, record["episode_id"], require_revision_complete=False)
            if len(payload["evidence"]) != record.get("evidence_count"):
                raise EvidenceStoreError(f"Evidence count mismatch for {record['episode_id']}.")
            _, ep_manifest_path = self._episode_paths(revision, record["episode_id"])
            if self._read_json(ep_manifest_path).get("data_hash") != record.get("data_hash"):
                raise EvidenceStoreError(f"Revision artifact hash mismatch for {record['episode_id']}.")
        return manifest

    def _load_episode_payload(self, revision: str, episode_id: str, *, require_revision_complete: bool = True) -> dict[str, Any]:
        if require_revision_complete:
            self.verify_revision(revision)
        data_path, manifest_path = self._episode_paths(revision, episode_id)
        if not data_path.is_file() or not manifest_path.is_file():
            raise EvidenceStoreError(f"Evidence episode artifact is missing: {episode_id}")
        manifest = self._read_json(manifest_path)
        if manifest.get("status") != "COMPLETE" or manifest.get("revision") != revision or manifest.get("episode_id") != episode_id:
            raise EvidenceStoreError(f"Evidence episode artifact is incomplete: {episode_id}")
        payload = self._read_json(data_path)
        if _sha256(payload) != manifest.get("data_hash"):
            raise EvidenceStoreError(f"Evidence artifact hash mismatch for {episode_id}.")
        if payload.get("revision") != revision or payload.get("project_id") != self.project_id or payload.get("episode_id") != episode_id:
            raise EvidenceStoreError(f"Evidence artifact identity mismatch for {episode_id}.")
        return payload

    def get_episode(self, episode_id: str, revision: str) -> EvidenceQueryResult:
        manifest = self.verify_revision(revision)
        record = next((item for item in manifest["episodes"] if item["episode_id"] == episode_id), None)
        if record is None:
            return EvidenceQueryResult((), 0, 0, True, revision, manifest["revision_artifact_hash"])
        payload = self._load_episode_payload(revision, episode_id, require_revision_complete=False)
        items = tuple(Evidence.from_dict(item) for item in payload["evidence"])
        return EvidenceQueryResult(items, len(items), len(items), True, revision, record["data_hash"])

    def get(self, evidence_id: str, revision: str) -> Evidence | None:
        manifest = self.verify_revision(revision)
        episode_id = evidence_id.split("-EV-", 1)[0]
        if not any(record["episode_id"] == episode_id for record in manifest["episodes"]):
            return None
        for item in self.get_episode(episode_id, revision).items:
            if item.evidence_id == evidence_id:
                return item
        return None

    def get_many(self, evidence_ids: Iterable[str], revision: str) -> EvidenceQueryResult:
        manifest = self.verify_revision(revision)
        requested = list(evidence_ids)
        found = tuple(item for evidence_id in requested if (item := self.get(evidence_id, revision)) is not None)
        return EvidenceQueryResult(
            found, len(requested), len(found), len(found) == len(requested), revision, manifest["revision_artifact_hash"]
        )

    def query_range(self, episode_id: str, start_ms: int, end_ms: int, revision: str) -> EvidenceQueryResult:
        if type(start_ms) is not int or type(end_ms) is not int or start_ms < 0 or end_ms <= start_ms:
            raise ValueError("Range requires integer 0 <= start_ms < end_ms")
        episode = self.get_episode(episode_id, revision)
        items = tuple(item for item in episode.items if item.start_ms < end_ms and item.end_ms > start_ms)
        return EvidenceQueryResult(items, len(items), len(items), True, revision, episode.artifact_hash)

    def completeness(self, revision: str) -> EvidenceCompleteness:
        try:
            manifest = self.verify_revision(revision)
        except EvidenceRevisionError:
            return EvidenceCompleteness(revision, False, 0, 0, 0)
        return EvidenceCompleteness(
            revision=revision,
            complete=True,
            expected_episodes=int(manifest["expected_episode_count"]),
            completed_episodes=int(manifest["completed_episode_count"]),
            evidence_count=int(manifest["evidence_count"]),
        )
