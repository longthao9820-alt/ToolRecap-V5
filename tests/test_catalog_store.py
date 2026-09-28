from __future__ import annotations

import json

import pytest

from catalog_test_helpers import evidence, populated_store, prepared
from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer.catalog import CatalogBuilder, canonical_catalog_bytes
from toolrecap_v4.analysis.finalizer.catalog_store import CatalogService, compute_catalog_dependencies
from toolrecap_v4.analysis.finalizer.packing import packed_catalog_bytes
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError


def test_catalog_persistence_metrics_and_cache_reuse(tmp_path, monkeypatch):
    _, revision, episodes, _ = populated_store(tmp_path)
    service = CatalogService(tmp_path)
    first = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert first.reused is False
    assert first.capacity.status == "UNKNOWN"
    assert first.capacity.configured_limit_bytes is None
    assert first.capacity.canonical_bytes == len(canonical_catalog_bytes(first.catalog))
    assert first.capacity.packed_bytes == len(packed_catalog_bytes(first.packed))
    assert first.capacity.episode_count == 2
    assert first.capacity.item_count == 4
    assert first.capacity.unique_entity_count == 3
    assert first.capacity.compression_ratio == pytest.approx(
        first.capacity.packed_bytes / first.capacity.canonical_bytes
    )
    assert first.manifest_path.is_file()
    monkeypatch.setattr(
        CatalogBuilder, "build",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Catalog should be reused")),
    )
    second = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert second.reused is True
    assert second.catalog == first.catalog
    assert second.packed == first.packed


@pytest.mark.parametrize("limit,status", [(10_000_000, "FIT"), (1, "EXCEEDS_CONFIGURED_LIMIT")])
def test_explicit_capacity_is_truthful_and_never_removes_evidence(tmp_path, limit, status):
    _, revision, episodes, evidence_items = populated_store(tmp_path)
    result = CatalogService(tmp_path, configured_capacity_bytes=limit).build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert result.capacity.status == status
    assert len(result.catalog.items) == len(evidence_items)
    assert result.catalog.completeness.actual_evidence_count == len(evidence_items)


def test_capacity_profile_change_reuses_catalog_and_refreshes_preflight(tmp_path):
    _, revision, episodes, _ = populated_store(tmp_path)
    unknown = CatalogService(tmp_path).build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert unknown.capacity.status == "UNKNOWN"
    limited = CatalogService(tmp_path, configured_capacity_bytes=1).build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert limited.reused is True
    assert limited.capacity.status == "EXCEEDS_CONFIGURED_LIMIT"
    assert limited.catalog == unknown.catalog


def test_corrupt_or_partial_catalog_is_not_a_cache_hit(tmp_path):
    _, revision, episodes, _ = populated_store(tmp_path)
    service = CatalogService(tmp_path)
    first = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    catalog_path = first.manifest_path.parent / "catalog.json"
    catalog_path.write_text("{corrupt", encoding="utf-8")
    rebuilt = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert rebuilt.reused is False
    assert rebuilt.catalog == first.catalog
    rebuilt.manifest_path.unlink()
    rebuilt_again = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert rebuilt_again.reused is False
    assert rebuilt_again.manifest_path.is_file()


def test_corrupt_packed_checkpoint_is_rebuilt_from_full_evidence(tmp_path):
    _, revision, episodes, _ = populated_store(tmp_path)
    service = CatalogService(tmp_path)
    first = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    packed_path = first.manifest_path.parent / "packed.json"
    packed = json.loads(packed_path.read_text(encoding="utf-8"))
    packed["items"][0][6] = 999999
    packed_path.write_text(json.dumps(packed), encoding="utf-8")
    rebuilt = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert rebuilt.reused is False
    assert rebuilt.catalog == first.catalog


def test_evidence_revision_change_builds_new_catalog_without_mutating_old(tmp_path):
    _, old_revision, episodes, _ = populated_store(tmp_path)
    service = CatalogService(tmp_path)
    old = service.build_or_load(
        project_id="project-1", evidence_revision=old_revision, ordered_episodes=episodes,
    )
    new_revision = "evr-catalog-new"
    store = EvidenceStore(tmp_path, "project-1")
    store.save_episode(new_revision, "E01", (
        evidence("E01-EV-001", "src_shared", 1000, 3000, "Changed factual evidence."),
    ), dependency_digest="new-e01")
    store.save_episode(new_revision, "E02", (), dependency_digest="new-e02")
    store.commit_revision(new_revision, ["E01", "E02"], dependency_signature={"scanner": "new"})
    new = service.build_or_load(
        project_id="project-1", evidence_revision=new_revision, ordered_episodes=episodes,
    )
    assert new.reused is False
    assert new.catalog.catalog_hash != old.catalog.catalog_hash
    assert old.manifest_path.is_file()


def test_dependency_signature_tracks_only_catalog_inputs(tmp_path):
    store, revision, episodes, _ = populated_store(tmp_path)
    manifest = store.verify_revision(revision)
    first = compute_catalog_dependencies(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=episodes, revision_manifest=manifest,
    )
    changed_manifest = json.loads(json.dumps(manifest))
    changed_manifest["episodes"][0]["data_hash"] = "different"
    second = compute_catalog_dependencies(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=episodes, revision_manifest=changed_manifest,
    )
    assert first != second
    serialized = json.dumps(first)
    for irrelevant in ("prompt", "finalizer", "voice", "render", "output_dir", "parallelism"):
        assert irrelevant not in serialized.lower()


def test_schema_and_packing_versions_invalidate_cache(tmp_path, monkeypatch):
    import toolrecap_v4.analysis.finalizer.catalog as catalog_module
    import toolrecap_v4.analysis.finalizer.packing as packing_module

    _, revision, episodes, _ = populated_store(tmp_path)
    service = CatalogService(tmp_path)
    first = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    monkeypatch.setattr(packing_module, "PACKING_VERSION", "season-catalog-packed-test-v2")
    repacked = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert repacked.reused is False
    assert repacked.catalog.catalog_hash == first.catalog.catalog_hash
    monkeypatch.setattr(catalog_module, "CATALOG_VERSION", "season-catalog-test-v2")
    rebuilt = service.build_or_load(
        project_id="project-1", evidence_revision=revision, ordered_episodes=episodes,
    )
    assert rebuilt.reused is False
    assert rebuilt.catalog.catalog_hash != first.catalog.catalog_hash


def test_cancellation_cannot_publish_complete_catalog(tmp_path):
    _, revision, episodes, _ = populated_store(tmp_path)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        CatalogService(tmp_path).build_or_load(
            project_id="project-1", evidence_revision=revision,
            ordered_episodes=episodes, cancellation_token=token,
        )
    assert not (tmp_path / "projects" / "project-1" / "catalog").exists()


def test_cancellation_during_persistence_leaves_no_complete_checkpoint(tmp_path, monkeypatch):
    import toolrecap_v4.analysis.finalizer.catalog_store as store_module

    _, revision, episodes, _ = populated_store(tmp_path)
    token = CancellationToken()
    original_write = store_module.atomic_write_json

    def cancel_after_catalog(path, data, *args, **kwargs):
        result = original_write(path, data, *args, **kwargs)
        if path.name == "catalog.json":
            token.cancel()
        return result

    monkeypatch.setattr(store_module, "atomic_write_json", cancel_after_catalog)
    with pytest.raises(CancelledError):
        CatalogService(tmp_path).build_or_load(
            project_id="project-1", evidence_revision=revision,
            ordered_episodes=episodes, cancellation_token=token,
        )
    manifests = list((tmp_path / "projects" / "project-1" / "catalog").rglob("manifest.json"))
    assert manifests
    assert all(json.loads(path.read_text())["status"] != "COMPLETE" for path in manifests)
    assert not (tmp_path / "projects" / "project-1" / "catalog" / "active.json").exists()
