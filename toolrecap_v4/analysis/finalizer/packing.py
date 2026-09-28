"""Lossless structural packing for the complete Season Evidence Catalog."""
from __future__ import annotations

import json
from typing import Any

from toolrecap_v4.analysis.finalizer.catalog import (
    CatalogCompleteness,
    CatalogEpisode,
    CatalogItem,
    SeasonEvidenceCatalog,
    validate_catalog,
)
from toolrecap_v4.errors import CatalogPackingError, CatalogValidationError

PACKING_VERSION = "season-catalog-packed-v1"


def packed_catalog_bytes(packed: dict[str, Any]) -> bytes:
    return json.dumps(packed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pack_catalog(catalog: SeasonEvidenceCatalog) -> dict[str, Any]:
    """Eliminate repeated structure and strings without changing any semantic value."""
    validate_catalog(catalog)
    strings: list[str] = []
    string_positions: dict[str, int] = {}

    def intern(value: str) -> int:
        if value not in string_positions:
            string_positions[value] = len(strings)
            strings.append(value)
        return string_positions[value]

    sources: list[list[int]] = []
    source_positions: dict[tuple[str, str], int] = {}
    episode_source_indices: list[int] = []
    for episode in catalog.ordered_episodes:
        source_key = (episode.source_id, episode.source_basename)
        if source_key not in source_positions:
            source_positions[source_key] = len(sources)
            sources.append([intern(episode.source_id), intern(episode.source_basename)])
        episode_source_indices.append(source_positions[source_key])

    episodes: list[list[Any]] = []
    episode_positions: dict[str, int] = {}
    for index, episode in enumerate(catalog.ordered_episodes):
        episode_positions[episode.episode_id] = index
        episodes.append([
            intern(episode.episode_id), episode_source_indices[index], episode.duration_ms,
            episode.evidence_count, intern(episode.evidence_manifest_hash),
            intern(episode.transcript_status), intern(episode.evidence_status),
        ])

    entities: list[int] = []
    entity_positions: dict[str, int] = {}

    def intern_entity(value: str) -> int:
        if value not in entity_positions:
            entity_positions[value] = len(entities)
            entities.append(intern(value))
        return entity_positions[value]

    items: list[list[Any]] = []
    for item in catalog.items:
        episode_index = episode_positions[item.episode_id]
        source_index = episode_source_indices[episode_index]
        items.append([
            intern(item.evidence_id), episode_index, source_index,
            item.start_ms, item.end_ms, intern(item.category), intern(item.factual_observation),
            [intern_entity(value) for value in item.entities], intern(item.modality), item.confidence,
            [intern(value) for value in item.uncertainty], intern(item.detail_hash),
            [intern(value) for value in item.visual_refs],
        ])

    completeness = catalog.completeness
    return {
        "packing_version": PACKING_VERSION,
        "catalog_version": catalog.catalog_version,
        "project_id": catalog.project_id,
        "evidence_revision": catalog.evidence_revision,
        "catalog_hash": catalog.catalog_hash,
        "strings": strings,
        "sources": sources,
        "episodes": episodes,
        "entities": entities,
        "items": items,
        "episode_count": len(episodes),
        "item_count": len(items),
        "completeness": [
            completeness.expected_episode_count, completeness.actual_episode_count,
            completeness.expected_evidence_count, completeness.actual_evidence_count,
            intern(completeness.ids_digest),
            [intern(value) for value in completeness.missing_ids],
            [intern(value) for value in completeness.unexpected_ids],
            [intern(value) for value in completeness.duplicate_ids],
        ],
    }


def unpack_catalog(packed: dict[str, Any]) -> SeasonEvidenceCatalog:
    """Strictly validate and invert the packed representation."""
    if not isinstance(packed, dict):
        raise CatalogPackingError("Packed Catalog root must be an object.")
    required = {
        "packing_version", "catalog_version", "project_id", "evidence_revision", "catalog_hash",
        "strings", "sources", "episodes", "entities", "items", "episode_count", "item_count", "completeness",
    }
    if set(packed) != required:
        raise CatalogPackingError("Packed Catalog root fields are missing or unexpected.")
    if packed["packing_version"] != PACKING_VERSION:
        raise CatalogPackingError("Unsupported packed Catalog version.")
    strings = packed["strings"]
    sources = packed["sources"]
    episode_rows = packed["episodes"]
    entity_table = packed["entities"]
    item_rows = packed["items"]
    if not all(isinstance(value, list) for value in (strings, sources, episode_rows, entity_table, item_rows)):
        raise CatalogPackingError("Packed Catalog tables must be arrays.")
    if not all(isinstance(value, str) for value in strings) or len(strings) != len(set(strings)):
        raise CatalogPackingError("Packed string table is invalid or duplicated.")

    def string_at(index: Any) -> str:
        if type(index) is not int or index < 0 or index >= len(strings):
            raise CatalogPackingError("Packed Catalog contains an invalid string-table index.")
        return strings[index]

    source_values: list[tuple[str, str]] = []
    for row in sources:
        if not isinstance(row, list) or len(row) != 2:
            raise CatalogPackingError("Packed source row has invalid length.")
        source_values.append((string_at(row[0]), string_at(row[1])))
    if len(source_values) != len(set(source_values)):
        raise CatalogPackingError("Packed source table contains duplicates.")

    entity_values = [string_at(index) for index in entity_table]
    if len(entity_values) != len(set(entity_values)):
        raise CatalogPackingError("Packed entity table contains duplicates.")

    episodes: list[CatalogEpisode] = []
    for row in episode_rows:
        if not isinstance(row, list) or len(row) != 7:
            raise CatalogPackingError("Packed episode row has invalid length.")
        source_index = row[1]
        if type(source_index) is not int or source_index < 0 or source_index >= len(source_values):
            raise CatalogPackingError("Packed episode has an invalid source reference.")
        if type(row[2]) is not int or type(row[3]) is not int:
            raise CatalogPackingError("Packed episode numeric fields must be integers.")
        source_id, basename = source_values[source_index]
        episodes.append(CatalogEpisode(
            episode_id=string_at(row[0]), source_id=source_id, source_basename=basename,
            duration_ms=row[2], evidence_count=row[3], evidence_manifest_hash=string_at(row[4]),
            transcript_status=string_at(row[5]), evidence_status=string_at(row[6]),
        ))
    if type(packed["episode_count"]) is not int or packed["episode_count"] != len(episodes):
        raise CatalogPackingError("Packed episode count does not match the episode table.")

    items: list[CatalogItem] = []
    ids: set[str] = set()
    for row in item_rows:
        if not isinstance(row, list) or len(row) != 13:
            raise CatalogPackingError("Packed item row has invalid length.")
        episode_index, source_index = row[1], row[2]
        if type(episode_index) is not int or episode_index < 0 or episode_index >= len(episodes):
            raise CatalogPackingError("Packed item has an invalid episode reference.")
        if type(source_index) is not int or source_index < 0 or source_index >= len(source_values):
            raise CatalogPackingError("Packed item has an invalid source reference.")
        if type(row[3]) is not int or type(row[4]) is not int:
            raise CatalogPackingError("Packed item timestamps must be integers.")
        for position in (7, 10, 12):
            if not isinstance(row[position], list):
                raise CatalogPackingError("Packed item repeated fields must be arrays.")
        entity_indices = row[7]
        entity_list: list[str] = []
        for index in entity_indices:
            if type(index) is not int or index < 0 or index >= len(entity_values):
                raise CatalogPackingError("Packed item has an invalid entity reference.")
            entity_list.append(entity_values[index])
        confidence = row[9]
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))):
            raise CatalogPackingError("Packed item confidence is malformed.")
        evidence_id = string_at(row[0])
        if evidence_id in ids:
            raise CatalogPackingError("Packed Catalog contains a duplicate Evidence ID.")
        ids.add(evidence_id)
        source_id, _ = source_values[source_index]
        items.append(CatalogItem(
            evidence_id=evidence_id, episode_id=episodes[episode_index].episode_id,
            source_id=source_id, start_ms=row[3], end_ms=row[4], category=string_at(row[5]),
            factual_observation=string_at(row[6]), entities=tuple(entity_list),
            modality=string_at(row[8]), confidence=confidence,
            uncertainty=tuple(string_at(index) for index in row[10]),
            detail_hash=string_at(row[11]), visual_refs=tuple(string_at(index) for index in row[12]),
        ))
    if type(packed["item_count"]) is not int or packed["item_count"] != len(items):
        raise CatalogPackingError("Packed item count does not match the item table.")

    row = packed["completeness"]
    if not isinstance(row, list) or len(row) != 8:
        raise CatalogPackingError("Packed completeness row has invalid length.")
    if any(type(row[index]) is not int for index in range(4)):
        raise CatalogPackingError("Packed completeness counts must be integers.")
    if not all(isinstance(row[index], list) for index in (5, 6, 7)):
        raise CatalogPackingError("Packed completeness membership fields must be arrays.")
    completeness = CatalogCompleteness(
        expected_episode_count=row[0], actual_episode_count=row[1],
        expected_evidence_count=row[2], actual_evidence_count=row[3], ids_digest=string_at(row[4]),
        missing_ids=tuple(string_at(index) for index in row[5]),
        unexpected_ids=tuple(string_at(index) for index in row[6]),
        duplicate_ids=tuple(string_at(index) for index in row[7]),
    )
    catalog = SeasonEvidenceCatalog(
        catalog_version=packed["catalog_version"], project_id=packed["project_id"],
        evidence_revision=packed["evidence_revision"], catalog_hash=packed["catalog_hash"],
        ordered_episodes=tuple(episodes), items=tuple(items), completeness=completeness,
    )
    try:
        validate_catalog(catalog)
    except CatalogValidationError as exc:
        raise CatalogPackingError(f"Unpacked Catalog failed validation: {exc}") from exc
    return catalog
