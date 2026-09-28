from __future__ import annotations

import json

import pytest

from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.models import DialogueReference, Evidence
from toolrecap_v4.errors import EvidenceRevisionError, EvidenceStoreError


def _evidence(eid: str, start: int, end: int, text: str, entity: str = "Beth") -> Evidence:
    return Evidence(
        evidence_id=eid, episode_id=eid.split("-EV-")[0], source_id="src_003",
        start_ms=start, end_ms=end, category="dialogue", observation=text,
        dialogue=(DialogueReference("cue-1", 1, 1, text),), entities=(entity,),
        modality="subtitle", confidence=None, uncertainty=(), visual_refs=(),
        provenance={"chunk_id": "E03-CH-001", "artifact_hash": "abc"},
    )


@pytest.fixture
def populated(tmp_path):
    store = EvidenceStore(tmp_path, "project-1")
    revision = "evr-1234567890abcdef"
    items = (
        _evidence("E03-EV-001", 1000, 2000, "Beth enters."),
        _evidence("E03-EV-002", 1900, 2500, "A short repeated event.", "Minor"),
        _evidence("E03-EV-003", 4000, 5000, "A short repeated event.", "Minor"),
    )
    store.save_episode(revision, "E03", items, dependency_digest="dep")
    store.commit_revision(revision, ["E03"], dependency_signature={"scanner_model": "model"})
    return store, revision, items


def test_get_get_many_episode_and_completeness(populated):
    store, revision, items = populated
    assert store.get("E03-EV-002", revision) == items[1]
    assert store.get("E03-EV-999", revision) is None
    many = store.get_many(["E03-EV-003", "E03-EV-001"], revision)
    assert [item.evidence_id for item in many.items] == ["E03-EV-003", "E03-EV-001"]
    assert many.total_count == many.returned_count == 2 and many.complete
    episode = store.get_episode("E03", revision)
    assert [item.evidence_id for item in episode.items] == [item.evidence_id for item in items]
    complete = store.completeness(revision)
    assert complete.complete and complete.evidence_count == 3


def test_range_query_returns_all_overlaps_in_deterministic_order(populated):
    store, revision, _ = populated
    result = store.query_range("E03", 1950, 4100, revision)
    assert [item.evidence_id for item in result.items] == ["E03-EV-001", "E03-EV-002", "E03-EV-003"]
    assert result.total_count == 3 and result.returned_count == 3 and result.complete


def test_duplicate_looking_and_secondary_entity_observations_are_preserved(populated):
    store, revision, _ = populated
    items = store.get_episode("E03", revision).items
    repeated = [item for item in items if item.observation == "A short repeated event."]
    assert len(repeated) == 2
    assert all("Minor" in item.entities for item in repeated)


def test_wrong_revision_and_partial_revision_are_not_cache_hits(tmp_path):
    store = EvidenceStore(tmp_path, "project-1")
    with pytest.raises(EvidenceRevisionError):
        store.get("E01-EV-001", "evr-missing")
    store.save_episode("evr-partial", "E01", (_evidence("E01-EV-001", 1, 2, "x"),), dependency_digest="d")
    with pytest.raises(EvidenceRevisionError):
        store.get_episode("E01", "evr-partial")


def test_corruption_and_hash_mismatch_are_rejected(populated):
    store, revision, _ = populated
    data_path = store.base_dir / revision / "episodes" / "E03.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    payload["evidence"][0]["observation"] = "tampered"
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceStoreError, match="hash mismatch"):
        store.get_episode("E03", revision)


def test_same_revision_cannot_be_mutated_and_old_revision_survives(tmp_path):
    store = EvidenceStore(tmp_path, "project-1")
    old = "evr-old"
    original = (_evidence("E03-EV-001", 1, 2, "original"),)
    store.save_episode(old, "E03", original, dependency_digest="d1")
    store.commit_revision(old, ["E03"], dependency_signature={"v": 1})
    changed = (_evidence("E03-EV-001", 1, 2, "different"),)
    with pytest.raises(EvidenceRevisionError):
        store.save_episode(old, "E03", changed, dependency_digest="d1")
    new = "evr-new"
    store.save_episode(new, "E03", changed, dependency_digest="d2")
    store.commit_revision(new, ["E03"], dependency_signature={"v": 2})
    assert store.get("E03-EV-001", old).observation == "original"
    assert store.get("E03-EV-001", new).observation == "different"
