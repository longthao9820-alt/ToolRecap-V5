from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from catalog_test_helpers import populated_store, prepared
from toolrecap_v4.analysis.finalizer.catalog import (
    CatalogBuilder,
    canonical_catalog_bytes,
    catalog_detail_hash,
    compute_catalog_hash,
    compute_ids_digest,
    validate_catalog,
    verify_catalog_against_evidence,
)
from toolrecap_v4.errors import CatalogValidationError


def test_complete_catalog_represents_every_episode_and_evidence(tmp_path):
    store, revision, episodes, evidence = populated_store(tmp_path)
    catalog = CatalogBuilder().build(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=reversed(episodes), evidence_store=store,
    )
    assert [episode.episode_id for episode in catalog.ordered_episodes] == ["E01", "E02"]
    assert [item.evidence_id for item in catalog.items] == [item.evidence_id for item in evidence]
    assert len(catalog.items) == len(evidence)
    assert catalog.ordered_episodes[1].evidence_count == 0
    assert catalog.ordered_episodes[1].evidence_status == "complete_zero"
    assert catalog.ordered_episodes[1].transcript_status == "complete_empty_no_speech"
    assert catalog.completeness.expected_episode_count == catalog.completeness.actual_episode_count == 2
    assert catalog.completeness.expected_evidence_count == catalog.completeness.actual_evidence_count == 4
    assert catalog.completeness.ids_digest == compute_ids_digest([item.evidence_id for item in evidence])
    assert catalog.completeness.missing_ids == catalog.completeness.unexpected_ids == ()
    assert all(item.detail_hash == catalog_detail_hash(full) for item, full in zip(catalog.items, evidence))


def test_catalog_preserves_secondary_short_repeated_unicode_and_long_text(tmp_path):
    store, revision, episodes, evidence = populated_store(tmp_path)
    catalog = CatalogBuilder().build(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=episodes, evidence_store=store,
    )
    assert catalog.items[0].factual_observation == evidence[0].observation
    assert catalog.items[0].entities == ("Beth", "Jamie")
    assert "Minor Character" in catalog.items[1].entities
    assert catalog.items[1].factual_observation == "A short event."
    assert catalog.items[2].factual_observation == catalog.items[3].factual_observation
    assert catalog.items[2].evidence_id != catalog.items[3].evidence_id
    assert len(catalog.items[2].factual_observation) > 500


def test_catalog_order_hash_and_bytes_are_deterministic(tmp_path):
    store, revision, episodes, _ = populated_store(tmp_path)
    builder = CatalogBuilder()
    first = builder.build(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=episodes, evidence_store=store,
    )
    second = builder.build(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=tuple(reversed(episodes)), evidence_store=store,
    )
    assert first == second
    assert first.catalog_hash == second.catalog_hash == compute_catalog_hash(first)
    assert canonical_catalog_bytes(first) == canonical_catalog_bytes(second)


def test_failed_or_incomplete_episode_blocks_catalog(tmp_path):
    store, revision, episodes, _ = populated_store(tmp_path)
    failed = replace(episodes[1], status="failed")
    with pytest.raises(CatalogValidationError, match="failed episode"):
        CatalogBuilder().build(
            project_id="project-1", evidence_revision=revision,
            ordered_episodes=(episodes[0], failed), evidence_store=store,
        )
    with pytest.raises(CatalogValidationError, match="exactly cover"):
        CatalogBuilder().build(
            project_id="project-1", evidence_revision=revision,
            ordered_episodes=(episodes[0], prepared("E03", "src_003")), evidence_store=store,
        )


@pytest.mark.parametrize("mutation,issue", [
    (lambda catalog: replace(catalog, catalog_hash="bad"), "catalog_hash"),
    (lambda catalog: replace(catalog, items=catalog.items[:-1]), "evidence_completeness"),
    (lambda catalog: replace(catalog, items=(catalog.items[0], *catalog.items)), "duplicate_evidence_id"),
    (lambda catalog: replace(catalog, items=(replace(catalog.items[0], episode_id="E99"), *catalog.items[1:])), "item_episode_reference"),
])
def test_catalog_validation_rejects_corruption(tmp_path, mutation, issue):
    store, revision, episodes, _ = populated_store(tmp_path)
    catalog = CatalogBuilder().build(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=episodes, evidence_store=store,
    )
    with pytest.raises(CatalogValidationError) as raised:
        validate_catalog(mutation(catalog))
    assert issue in raised.value.issue_codes


def test_detail_hash_is_verified_against_full_evidence(tmp_path):
    store, revision, episodes, _ = populated_store(tmp_path)
    catalog = CatalogBuilder().build(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=episodes, evidence_store=store,
    )
    corrupt_item = replace(catalog.items[0], detail_hash="wrong-detail-hash")
    unhashed = replace(catalog, items=(corrupt_item, *catalog.items[1:]))
    internally_valid = replace(unhashed, catalog_hash=compute_catalog_hash(unhashed))
    validate_catalog(internally_valid)
    with pytest.raises(CatalogValidationError) as raised:
        verify_catalog_against_evidence(internally_valid, store)
    assert "detail_hash" in raised.value.issue_codes


def test_catalog_builder_contains_no_editorial_filter_or_ranking_policy():
    source = inspect.getsource(CatalogBuilder).lower()
    for forbidden in (
        "top_k", "top-k", "editorialpolicy", "candidate", "season connection",
        "story ranking", "importance score", "recap-worthy",
    ):
        assert forbidden not in source
