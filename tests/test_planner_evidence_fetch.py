from __future__ import annotations

from dataclasses import replace

import pytest

from catalog_test_helpers import populated_store
from toolrecap_v4.analysis.finalizer.catalog import CatalogBuilder
from toolrecap_v4.analysis.finalizer.evidence_fetch import EvidenceFetcher, EvidenceRequest
from toolrecap_v4.errors import (
    PlannerEvidenceIntegrityError,
    PlannerEvidenceNotFoundError,
    PlannerEvidenceRequestError,
    PlannerEvidenceRevisionError,
)


@pytest.fixture
def env(tmp_path):
    store, revision, episodes, evidence = populated_store(tmp_path)
    catalog = CatalogBuilder().build(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=episodes, evidence_store=store,
    )
    fetcher = EvidenceFetcher(
        project_id="project-1", evidence_revision=revision,
        catalog=catalog, evidence_store=store,
    )
    return fetcher, catalog, store, revision, evidence


def test_exact_single_and_many_id_fetch_are_authoritative_and_ordered(env):
    fetcher, _, _, revision, _ = env
    result = fetcher.fetch(round_id="round-001", requests=(
        EvidenceRequest("req-1", "evidence_ids", ("E01-EV-003", "E01-EV-001")),
    ))
    assert result.evidence_revision == revision
    assert [item.evidence_id for item in result.items] == ["E01-EV-001", "E01-EV-003"]
    assert result.request_results[0].complete
    assert result.request_results[0].total_count == result.request_results[0].returned_count == 2
    assert result.completeness["complete"] is True


def test_duplicate_unknown_and_malformed_ids_are_rejected(env):
    fetcher = env[0]
    with pytest.raises(PlannerEvidenceRequestError, match="Duplicate"):
        fetcher.fetch(round_id="round-001", requests=(
            EvidenceRequest("req", "evidence_ids", ("E01-EV-001", "E01-EV-001")),
        ))
    with pytest.raises(PlannerEvidenceNotFoundError, match="Unknown"):
        fetcher.fetch(round_id="round-001", requests=(
            EvidenceRequest("req", "evidence_ids", ("E01-EV-999",)),
        ))
    with pytest.raises(PlannerEvidenceNotFoundError, match="Malformed"):
        fetcher.fetch(round_id="round-001", requests=(
            EvidenceRequest("req", "evidence_ids", ("best Beth scene",)),
        ))


def test_range_fetch_returns_all_overlaps_and_explicit_empty(env):
    fetcher = env[0]
    result = fetcher.fetch(round_id="round-001", requests=(
        EvidenceRequest("range-1", "episode_range", episode_id="E01", start_ms=2999, end_ms=4001),
        EvidenceRequest("range-2", "episode_range", episode_id="E02", start_ms=0, end_ms=1000),
    ))
    assert [item.evidence_id for item in result.items] == ["E01-EV-001", "E01-EV-002"]
    assert result.request_results[0].evidence_ids == ("E01-EV-001", "E01-EV-002")
    assert result.request_results[1].evidence_ids == ()
    assert result.request_results[1].total_count == 0
    assert result.request_results[1].complete is True


@pytest.mark.parametrize("range_request", [
    EvidenceRequest("r", "episode_range", episode_id="E99", start_ms=0, end_ms=1),
    EvidenceRequest("r", "episode_range", episode_id="E01", start_ms=-1, end_ms=1),
    EvidenceRequest("r", "episode_range", episode_id="E01", start_ms=1, end_ms=1),
    EvidenceRequest("r", "episode_range", episode_id="E01", start_ms=0, end_ms=100_001),
    EvidenceRequest("r", "episode_range", episode_id="E01", start_ms=True, end_ms=2),
    EvidenceRequest("r", "episode_range", episode_id="E01", start_ms=0.0, end_ms=2),
])
def test_invalid_ranges_are_rejected_without_clamping(env, range_request):
    with pytest.raises(PlannerEvidenceRequestError):
        env[0].fetch(round_id="round-001", requests=(range_request,))


def test_wrong_project_revision_and_detail_hash_are_rejected(env):
    fetcher, catalog, store, revision, _ = env
    with pytest.raises(PlannerEvidenceRevisionError):
        EvidenceFetcher(
            project_id="other-project", evidence_revision=revision,
            catalog=catalog, evidence_store=store,
        )
    with pytest.raises(PlannerEvidenceRevisionError):
        EvidenceFetcher(
            project_id="project-1", evidence_revision="evr-stale",
            catalog=catalog, evidence_store=store,
        )
    corrupt_item = replace(catalog.items[0], detail_hash="wrong")
    corrupt_catalog = replace(catalog, items=(corrupt_item, *catalog.items[1:]))
    corrupt_fetcher = EvidenceFetcher(
        project_id="project-1", evidence_revision=revision,
        catalog=corrupt_catalog, evidence_store=store,
    )
    with pytest.raises(PlannerEvidenceIntegrityError):
        corrupt_fetcher.fetch(round_id="round-001", requests=(
            EvidenceRequest("req", "evidence_ids", ("E01-EV-001",)),
        ))
