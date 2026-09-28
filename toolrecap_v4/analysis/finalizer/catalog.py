"""Complete deterministic Season Evidence Catalog schema, builder, and validation."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import re
from typing import Any, Sequence

from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.models import Evidence, PreparedEpisode
from toolrecap_v4.errors import CatalogValidationError

CATALOG_VERSION = "season-catalog-v1"
CATALOG_PROJECTION_VERSION = "season-catalog-projection-v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _episode_key(episode_id: str) -> tuple[Any, ...]:
    match = re.fullmatch(r"E(\d+)", episode_id)
    return (0, int(match.group(1)), episode_id) if match else (1, episode_id)


def _evidence_key(evidence_id: str) -> tuple[Any, ...]:
    match = re.fullmatch(r"(.+)-EV-(\d+)", evidence_id)
    return (int(match.group(2)), evidence_id) if match else (2**63, evidence_id)


def compute_ids_digest(evidence_ids: Sequence[str]) -> str:
    """Digest the complete ordered Evidence ID set with unambiguous JSON framing."""
    if any(not isinstance(item, str) or not item for item in evidence_ids):
        raise CatalogValidationError("Evidence IDs must be non-empty strings.", issue_codes=("ids_type",))
    return hashlib.sha256(_canonical_bytes(list(evidence_ids))).hexdigest()


def catalog_detail_hash(evidence: Evidence) -> str:
    """Immutable link from a compact Catalog item to its complete local Evidence object."""
    return _hash(evidence.to_dict())


@dataclass(frozen=True)
class CatalogEpisode:
    episode_id: str
    source_id: str
    source_basename: str
    duration_ms: int
    evidence_count: int
    evidence_manifest_hash: str
    transcript_status: str
    evidence_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_id": self.source_id,
            "source_basename": self.source_basename,
            "duration_ms": self.duration_ms,
            "evidence_count": self.evidence_count,
            "evidence_manifest_hash": self.evidence_manifest_hash,
            "transcript_status": self.transcript_status,
            "evidence_status": self.evidence_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogEpisode:
        if set(data) != set(cls.__dataclass_fields__):
            raise CatalogValidationError("Catalog episode fields are malformed.", issue_codes=("episode_fields",))
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CatalogItem:
    evidence_id: str
    episode_id: str
    source_id: str
    start_ms: int
    end_ms: int
    category: str
    entities: tuple[str, ...]
    factual_observation: str
    modality: str
    confidence: float | None
    uncertainty: tuple[str, ...]
    detail_hash: str
    visual_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "episode_id": self.episode_id,
            "source_id": self.source_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "category": self.category,
            "entities": list(self.entities),
            "factual_observation": self.factual_observation,
            "modality": self.modality,
            "confidence": self.confidence,
            "uncertainty": list(self.uncertainty),
            "detail_hash": self.detail_hash,
            "visual_refs": list(self.visual_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogItem:
        if set(data) != set(cls.__dataclass_fields__):
            raise CatalogValidationError("Catalog item fields are malformed.", issue_codes=("item_fields",))
        return cls(
            evidence_id=data["evidence_id"], episode_id=data["episode_id"], source_id=data["source_id"],
            start_ms=data["start_ms"], end_ms=data["end_ms"], category=data["category"],
            entities=tuple(data["entities"]), factual_observation=data["factual_observation"],
            modality=data["modality"], confidence=data["confidence"],
            uncertainty=tuple(data["uncertainty"]), detail_hash=data["detail_hash"],
            visual_refs=tuple(data["visual_refs"]),
        )


@dataclass(frozen=True)
class CatalogCompleteness:
    expected_episode_count: int
    actual_episode_count: int
    expected_evidence_count: int
    actual_evidence_count: int
    ids_digest: str
    missing_ids: tuple[str, ...] = ()
    unexpected_ids: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_episode_count": self.expected_episode_count,
            "actual_episode_count": self.actual_episode_count,
            "expected_evidence_count": self.expected_evidence_count,
            "actual_evidence_count": self.actual_evidence_count,
            "ids_digest": self.ids_digest,
            "missing_ids": list(self.missing_ids),
            "unexpected_ids": list(self.unexpected_ids),
            "duplicate_ids": list(self.duplicate_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogCompleteness:
        if set(data) != set(cls.__dataclass_fields__):
            raise CatalogValidationError("Catalog completeness fields are malformed.", issue_codes=("completeness_fields",))
        return cls(
            expected_episode_count=data["expected_episode_count"],
            actual_episode_count=data["actual_episode_count"],
            expected_evidence_count=data["expected_evidence_count"],
            actual_evidence_count=data["actual_evidence_count"],
            ids_digest=data["ids_digest"],
            missing_ids=tuple(data["missing_ids"]),
            unexpected_ids=tuple(data["unexpected_ids"]),
            duplicate_ids=tuple(data["duplicate_ids"]),
        )


@dataclass(frozen=True)
class SeasonEvidenceCatalog:
    catalog_version: str
    project_id: str
    evidence_revision: str
    catalog_hash: str
    ordered_episodes: tuple[CatalogEpisode, ...]
    items: tuple[CatalogItem, ...]
    completeness: CatalogCompleteness

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "project_id": self.project_id,
            "evidence_revision": self.evidence_revision,
            "ordered_episodes": [episode.to_dict() for episode in self.ordered_episodes],
            "items": [item.to_dict() for item in self.items],
            "completeness": self.completeness.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.semantic_dict()
        result["catalog_hash"] = self.catalog_hash
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeasonEvidenceCatalog:
        required = {"catalog_version", "project_id", "evidence_revision", "catalog_hash", "ordered_episodes", "items", "completeness"}
        if set(data) != required:
            raise CatalogValidationError("Catalog root fields are missing or unexpected.", issue_codes=("root_fields",))
        if not isinstance(data["ordered_episodes"], list) or not isinstance(data["items"], list) or not isinstance(data["completeness"], dict):
            raise CatalogValidationError("Catalog collections have invalid types.", issue_codes=("root_types",))
        try:
            return cls(
                catalog_version=data["catalog_version"], project_id=data["project_id"],
                evidence_revision=data["evidence_revision"], catalog_hash=data["catalog_hash"],
                ordered_episodes=tuple(CatalogEpisode.from_dict(item) for item in data["ordered_episodes"]),
                items=tuple(CatalogItem.from_dict(item) for item in data["items"]),
                completeness=CatalogCompleteness.from_dict(data["completeness"]),
            )
        except (KeyError, TypeError) as exc:
            raise CatalogValidationError("Catalog object is malformed.", issue_codes=("object_shape",)) from exc


@dataclass(frozen=True)
class CapacityPreflight:
    status: str
    configured_limit_bytes: int | None
    canonical_bytes: int
    packed_bytes: int
    episode_count: int
    item_count: int
    unique_string_count: int
    unique_entity_count: int
    compression_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "configured_limit_bytes": self.configured_limit_bytes,
            "canonical_bytes": self.canonical_bytes,
            "packed_bytes": self.packed_bytes,
            "episode_count": self.episode_count,
            "item_count": self.item_count,
            "unique_string_count": self.unique_string_count,
            "unique_entity_count": self.unique_entity_count,
            "compression_ratio": self.compression_ratio,
        }


def compute_catalog_hash(catalog: SeasonEvidenceCatalog) -> str:
    return _hash(catalog.semantic_dict())


def canonical_catalog_bytes(catalog: SeasonEvidenceCatalog) -> bytes:
    return _canonical_bytes(catalog.to_dict())


def validate_catalog(catalog: SeasonEvidenceCatalog) -> None:
    issues: list[str] = []
    if catalog.catalog_version != CATALOG_VERSION:
        issues.append("catalog_version")
    for field_name in ("project_id", "evidence_revision", "catalog_hash"):
        value = getattr(catalog, field_name)
        if not isinstance(value, str) or not value:
            issues.append(field_name)

    episode_ids = [episode.episode_id for episode in catalog.ordered_episodes]
    if len(episode_ids) != len(set(episode_ids)):
        issues.append("duplicate_episode")
    if episode_ids != sorted(episode_ids, key=_episode_key):
        issues.append("episode_order")
    episode_map = {episode.episode_id: episode for episode in catalog.ordered_episodes}
    for episode in catalog.ordered_episodes:
        if any(not isinstance(value, str) or not value for value in (
            episode.episode_id, episode.source_id, episode.source_basename,
            episode.evidence_manifest_hash, episode.transcript_status, episode.evidence_status,
        )):
            issues.append("episode_strings")
        if type(episode.duration_ms) is not int or episode.duration_ms <= 0:
            issues.append("episode_duration")
        if type(episode.evidence_count) is not int or episode.evidence_count < 0:
            issues.append("episode_evidence_count")
        expected_status = "complete_zero" if episode.evidence_count == 0 else "complete"
        if episode.evidence_status != expected_status:
            issues.append("episode_evidence_status")

    ids = [item.evidence_id for item in catalog.items]
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        issues.append("duplicate_evidence_id")
    episode_positions = {episode_id: index for index, episode_id in enumerate(episode_ids)}
    expected_item_order = sorted(
        catalog.items,
        key=lambda item: (episode_positions.get(item.episode_id, 2**63), _evidence_key(item.evidence_id)),
    )
    if list(catalog.items) != expected_item_order:
        issues.append("item_order")
    per_episode_counts = {episode_id: 0 for episode_id in episode_ids}
    for item in catalog.items:
        episode = episode_map.get(item.episode_id)
        if episode is None:
            issues.append("item_episode_reference")
            continue
        per_episode_counts[item.episode_id] += 1
        if item.source_id != episode.source_id:
            issues.append("item_source_reference")
        if any(not isinstance(value, str) or not value for value in (
            item.evidence_id, item.source_id, item.category, item.factual_observation,
            item.modality, item.detail_hash,
        )):
            issues.append("item_strings")
        if type(item.start_ms) is not int or type(item.end_ms) is not int:
            issues.append("item_timestamp_type")
        elif item.start_ms < 0 or item.end_ms <= item.start_ms or item.end_ms > episode.duration_ms:
            issues.append("item_timestamp_range")
        if not all(isinstance(value, str) and value for value in (*item.entities, *item.uncertainty, *item.visual_refs)):
            issues.append("item_list_strings")
        if item.confidence is not None and (
            isinstance(item.confidence, bool) or not isinstance(item.confidence, (int, float))
            or not 0.0 <= float(item.confidence) <= 1.0
        ):
            issues.append("item_confidence")
    if any(per_episode_counts.get(ep.episode_id) != ep.evidence_count for ep in catalog.ordered_episodes):
        issues.append("episode_count_mismatch")

    completeness = catalog.completeness
    integer_fields = (
        completeness.expected_episode_count, completeness.actual_episode_count,
        completeness.expected_evidence_count, completeness.actual_evidence_count,
    )
    if any(type(value) is not int or value < 0 for value in integer_fields):
        issues.append("completeness_types")
    if completeness.expected_episode_count != len(catalog.ordered_episodes) or completeness.actual_episode_count != len(catalog.ordered_episodes):
        issues.append("episode_completeness")
    if completeness.expected_evidence_count != len(catalog.items) or completeness.actual_evidence_count != len(catalog.items):
        issues.append("evidence_completeness")
    if completeness.missing_ids or completeness.unexpected_ids or completeness.duplicate_ids:
        issues.append("membership_issues")
    if completeness.ids_digest != compute_ids_digest(ids):
        issues.append("ids_digest")
    if catalog.catalog_hash != compute_catalog_hash(catalog):
        issues.append("catalog_hash")
    if issues:
        raise CatalogValidationError(
            f"Season Evidence Catalog failed validation: {', '.join(sorted(set(issues)))}",
            issue_codes=tuple(sorted(set(issues))),
        )


def verify_catalog_against_evidence(catalog: SeasonEvidenceCatalog, evidence_store: EvidenceStore) -> None:
    """Verify membership, episode manifests, and every detail link against Full Evidence."""
    validate_catalog(catalog)
    revision = evidence_store.verify_revision(catalog.evidence_revision)
    records = {record["episode_id"]: record for record in revision.get("episodes", [])}
    if set(records) != {episode.episode_id for episode in catalog.ordered_episodes}:
        raise CatalogValidationError("Catalog episodes do not match the Evidence revision.", issue_codes=("evidence_episode_coverage",))
    expected_ids: list[str] = []
    full_by_id: dict[str, Evidence] = {}
    for episode in catalog.ordered_episodes:
        record = records[episode.episode_id]
        if episode.evidence_manifest_hash != record.get("data_hash"):
            raise CatalogValidationError("Catalog episode manifest hash is stale.", issue_codes=("evidence_manifest_hash",))
        result = evidence_store.get_episode(episode.episode_id, catalog.evidence_revision)
        for full in sorted(result.items, key=lambda item: _evidence_key(item.evidence_id)):
            expected_ids.append(full.evidence_id)
            full_by_id[full.evidence_id] = full
    actual_ids = [item.evidence_id for item in catalog.items]
    if actual_ids != expected_ids:
        raise CatalogValidationError("Catalog Evidence membership differs from Full Evidence.", issue_codes=("evidence_membership",))
    for item in catalog.items:
        full = full_by_id.get(item.evidence_id)
        if full is None or item.detail_hash != catalog_detail_hash(full):
            raise CatalogValidationError("Catalog detail hash does not match Full Evidence.", issue_codes=("detail_hash",))


def _transcript_status(episode: PreparedEpisode) -> str:
    transcript = episode.transcript
    if transcript.source_type == "empty":
        return "complete_empty_no_speech" if not transcript.has_speech else "complete_empty"
    return f"complete_{transcript.source_type}"


class CatalogBuilder:
    """Build one complete factual Catalog projection from one active Evidence revision."""

    def build(
        self,
        *,
        project_id: str,
        evidence_revision: str,
        ordered_episodes: Sequence[PreparedEpisode],
        evidence_store: EvidenceStore,
    ) -> SeasonEvidenceCatalog:
        revision_manifest = evidence_store.verify_revision(evidence_revision)
        episodes = tuple(sorted(ordered_episodes, key=lambda item: _episode_key(item.episode_id)))
        if not episodes:
            raise CatalogValidationError("Catalog requires at least one ordered episode.", issue_codes=("no_episodes",))
        if len({episode.episode_id for episode in episodes}) != len(episodes):
            raise CatalogValidationError("Prepared episode IDs are duplicated.", issue_codes=("duplicate_episode",))
        if any(episode.status == "failed" for episode in episodes):
            raise CatalogValidationError("A failed episode cannot enter a complete Catalog.", issue_codes=("episode_failed",))

        manifest_records = {record["episode_id"]: record for record in revision_manifest.get("episodes", [])}
        prepared_ids = [episode.episode_id for episode in episodes]
        if set(manifest_records) != set(prepared_ids):
            raise CatalogValidationError(
                "Complete Evidence revision does not exactly cover the ordered project episodes.",
                issue_codes=("episode_coverage",),
            )

        catalog_episodes: list[CatalogEpisode] = []
        catalog_items: list[CatalogItem] = []
        expected_ids: list[str] = []
        for episode in episodes:
            record = manifest_records[episode.episode_id]
            evidence_result = evidence_store.get_episode(episode.episode_id, evidence_revision)
            evidence_items = tuple(sorted(evidence_result.items, key=lambda item: _evidence_key(item.evidence_id)))
            if len(evidence_items) != record.get("evidence_count"):
                raise CatalogValidationError("Episode Evidence count is incomplete.", issue_codes=("evidence_count",))
            catalog_episodes.append(CatalogEpisode(
                episode_id=episode.episode_id,
                source_id=episode.source_id,
                source_basename=episode.source_basename or episode.source_path.name,
                duration_ms=episode.duration_ms,
                evidence_count=len(evidence_items),
                evidence_manifest_hash=record["data_hash"],
                transcript_status=_transcript_status(episode),
                evidence_status="complete_zero" if not evidence_items else "complete",
            ))
            for evidence in evidence_items:
                expected_ids.append(evidence.evidence_id)
                catalog_items.append(CatalogItem(
                    evidence_id=evidence.evidence_id,
                    episode_id=evidence.episode_id,
                    source_id=evidence.source_id,
                    start_ms=evidence.start_ms,
                    end_ms=evidence.end_ms,
                    category=evidence.category,
                    entities=evidence.entities,
                    factual_observation=evidence.observation,
                    modality=evidence.modality,
                    confidence=evidence.confidence,
                    uncertainty=evidence.uncertainty,
                    detail_hash=catalog_detail_hash(evidence),
                    visual_refs=evidence.visual_refs,
                ))

        expected_count = revision_manifest.get("evidence_count")
        if type(expected_count) is not int or expected_count != len(expected_ids):
            raise CatalogValidationError("Revision Evidence count does not match episode artifacts.", issue_codes=("revision_count",))
        completeness = CatalogCompleteness(
            expected_episode_count=len(episodes), actual_episode_count=len(catalog_episodes),
            expected_evidence_count=expected_count, actual_evidence_count=len(catalog_items),
            ids_digest=compute_ids_digest(expected_ids),
        )
        unhashed = SeasonEvidenceCatalog(
            catalog_version=CATALOG_VERSION, project_id=project_id,
            evidence_revision=evidence_revision, catalog_hash="pending",
            ordered_episodes=tuple(catalog_episodes), items=tuple(catalog_items),
            completeness=completeness,
        )
        catalog = SeasonEvidenceCatalog(
            **{**unhashed.__dict__, "catalog_hash": compute_catalog_hash(unhashed)}
        )
        validate_catalog(catalog)
        verify_catalog_against_evidence(catalog, evidence_store)
        return catalog
