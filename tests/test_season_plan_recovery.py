"""Crash-safe, zero-AI Season Plan LOCKED artifact recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_season_plan_closure import setup
from toolrecap_v4.analysis.finalizer.season_plan import SeasonPlanService
from toolrecap_v4.analysis.finalizer.writer import verify_locked_plan_artifact
from toolrecap_v4.errors import SeasonPlanLockError
from toolrecap_v4.persistence import atomic_write_json
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.progress import WorkflowStage, reconstruct_project_progress


def _create(tmp_path: Path):
    catalog, draft, visual, gateway = setup(tmp_path, False)
    events: list[dict] = []
    service = SeasonPlanService(gateway, tmp_path, "planner", progress_callback=events.append)
    result = service.run(
        project_id="project-1", raw_recap_prompt="Unicode — Việt Nam 🙂",
        catalog=catalog, draft=draft, visual=visual,
    )
    return catalog, draft, visual, gateway, service, result, events


def _rerun(service, catalog, draft, visual):
    return service.run(
        project_id="project-1", raw_recap_prompt="Unicode — Việt Nam 🙂",
        catalog=catalog, draft=draft, visual=visual,
    )


def test_valid_utf8_plan_and_matching_locked_artifact_resume(tmp_path: Path) -> None:
    catalog, draft, visual, gateway, service, result, _ = _create(tmp_path)
    verify_locked_plan_artifact(tmp_path, result.plan)
    again = _rerun(service, catalog, draft, visual)
    assert again.reused and again.plan == result.plan
    assert len(gateway.text) == 1


def test_missing_lock_recovers_locally_without_ai_or_plan_rewrite(tmp_path: Path) -> None:
    catalog, draft, visual, gateway, service, result, events = _create(tmp_path)
    before = result.path.read_bytes()
    manifest = result.path.parent / "manifest.json"
    manifest.unlink()
    events.clear()

    recovered = _rerun(service, catalog, draft, visual)

    assert recovered.reused and recovered.plan.plan_hash == result.plan.plan_hash
    assert result.path.read_bytes() == before
    assert len(gateway.text) == 1
    lock = json.loads(manifest.read_text(encoding="utf-8"))
    assert lock["status"] == "LOCKED"
    assert lock["plan_hash"] == result.plan.plan_hash
    assert lock["project_id"] == "project-1"
    assert any("Recovered Season Plan checkpoint locally" in event.get("activity_text", "") for event in events)
    verify_locked_plan_artifact(tmp_path, recovered.plan)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("artifact_hash", "bad", "hash/identity mismatch"),
    ("project_id", "different-project", "different project"),
    ("revision", "plan-different", "different plan revision"),
    ("dependency_digest", "bad", "different revision"),
])
def test_mismatching_locked_identity_is_rejected(tmp_path: Path, field: str, value: str, message: str) -> None:
    catalog, draft, visual, gateway, service, result, _ = _create(tmp_path)
    manifest_path = result.path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(SeasonPlanLockError, match=message):
        _rerun(service, catalog, draft, visual)
    assert len(gateway.text) == 1


@pytest.mark.parametrize("payload", [b"{", b'{"plan_version":"season-plan-v1"}'])
def test_corrupt_or_truncated_plan_is_not_recovered(tmp_path: Path, payload: bytes) -> None:
    catalog, draft, visual, gateway, service, result, _ = _create(tmp_path)
    result.path.write_bytes(payload)
    with pytest.raises(SeasonPlanLockError, match="truncated, or corrupt"):
        _rerun(service, catalog, draft, visual)
    assert len(gateway.text) == 1


def test_old_locked_format_migrates_locally_without_ai(tmp_path: Path) -> None:
    catalog, draft, visual, gateway, service, result, _ = _create(tmp_path)
    manifest_path = result.path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in ("dependency_digest", "project_id", "revision"):
        manifest.pop(field)
    atomic_write_json(manifest_path, manifest)

    migrated = _rerun(service, catalog, draft, visual)

    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert migrated.reused and len(gateway.text) == 1
    assert current["project_id"] == "project-1"
    assert current["revision"] == result.path.parent.name
    assert current["dependency_digest"]


def test_matching_writer_checkpoint_is_preserved_during_recovery(tmp_path: Path) -> None:
    catalog, draft, visual, gateway, service, result, _ = _create(tmp_path)
    writer_manifest = tmp_path / "projects" / "project-1" / "writers" / result.plan.plan_hash / "manifest.json"
    atomic_write_json(writer_manifest, {
        "status": "INCOMPLETE", "season_plan_hash": result.plan.plan_hash,
        "expected_output_count": len(result.plan.outputs), "completed_response_count": 1,
    })
    before = writer_manifest.read_bytes()
    (result.path.parent / "manifest.json").unlink()

    recovered = _rerun(service, catalog, draft, visual)

    assert recovered.reused and writer_manifest.read_bytes() == before
    assert len(gateway.text) == 1


def test_mismatched_writer_checkpoint_blocks_recovery(tmp_path: Path) -> None:
    catalog, draft, visual, gateway, service, result, _ = _create(tmp_path)
    writer_manifest = tmp_path / "projects" / "project-1" / "writers" / result.plan.plan_hash / "manifest.json"
    atomic_write_json(writer_manifest, {
        "status": "INCOMPLETE", "season_plan_hash": "different-plan",
        "expected_output_count": len(result.plan.outputs), "completed_response_count": 1,
    })
    (result.path.parent / "manifest.json").unlink()
    with pytest.raises(SeasonPlanLockError, match="Writer checkpoint belongs"):
        _rerun(service, catalog, draft, visual)
    assert len(gateway.text) == 1


def test_crash_after_plan_write_recovers_but_missing_plan_does_not(tmp_path: Path) -> None:
    catalog, draft, visual, gateway, service, result, events = _create(tmp_path)
    manifest_path = result.path.parent / "manifest.json"
    atomic_write_json(manifest_path, {
        "status": "BUILDING",
        "dependency_digest": json.loads((result.path.parent / "attempt-00.raw.manifest.json").read_text(encoding="utf-8"))["dependency_digest"],
    })
    events.clear()
    recovered = _rerun(service, catalog, draft, visual)
    assert recovered.reused and len(gateway.text) == 1
    assert any("Recovered Season Plan" in event.get("activity_text", "") for event in events)

    result.path.unlink()
    with pytest.raises(SeasonPlanLockError, match="season_plan.json is missing"):
        _rerun(service, catalog, draft, visual)
    assert len(gateway.text) == 1


def test_lock_failure_emits_no_locked_activity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog, draft, visual, gateway = setup(tmp_path, False)
    events: list[dict] = []
    service = SeasonPlanService(gateway, tmp_path, "planner", progress_callback=events.append)
    import toolrecap_v4.analysis.finalizer.season_plan as module
    real_write = module.atomic_write_json

    def interrupted(path, data, *args, **kwargs):
        if path.name == "manifest.json" and data.get("status") == "LOCKED":
            raise OSError("simulated lock commit interruption")
        return real_write(path, data, *args, **kwargs)

    monkeypatch.setattr(module, "atomic_write_json", interrupted)
    with pytest.raises(OSError, match="interruption"):
        service.run(
            project_id="project-1", raw_recap_prompt="Unicode — Việt Nam 🙂",
            catalog=catalog, draft=draft, visual=visual,
        )
    assert not any("Season Plan locked" in event.get("activity_text", "") for event in events)
    assert list(tmp_path.rglob("season_plan.json"))


def _save_resume_state(tmp_path: Path, result) -> ProjectPersistence:
    persistence = ProjectPersistence(tmp_path)
    persistence.save_project({
        "project_id": "project-1", "project_name": "Project", "status": "season_plan_ready",
        "season_plan": {"season_plan_path": str(result.path), "plan_hash": result.plan.plan_hash},
        "sources": [], "source_fingerprints": {}, "prepared_episodes": {}, "outputs": {},
        "timestamps": {}, "error": None,
    })
    persistence.save_checkpoint("project-1", "season_plan", {
        "status": "completed", "season_plan_path": str(result.path),
        "plan_hash": result.plan.plan_hash,
    })
    return persistence


def test_resume_reconstruction_requires_authoritative_lock(tmp_path: Path) -> None:
    _, _, _, _, _, result, _ = _create(tmp_path)
    persistence = _save_resume_state(tmp_path, result)
    valid = reconstruct_project_progress(persistence, "project-1")
    assert valid["pipeline"][WorkflowStage.FINAL_PLAN.value] == "complete"

    manifest = result.path.parent / "manifest.json"
    manifest.unlink()
    recoverable = reconstruct_project_progress(persistence, "project-1")
    assert recoverable["pipeline"][WorkflowStage.FINAL_PLAN.value] == "retry"
    assert "safe local recovery" in recoverable["activity_text"]

    atomic_write_json(manifest, {
        "status": "LOCKED", "plan_hash": result.plan.plan_hash,
        "artifact_hash": "wrong",
    })
    invalid = reconstruct_project_progress(persistence, "project-1")
    assert invalid["pipeline"][WorkflowStage.FINAL_PLAN.value] == "failed"
    assert invalid["state"] == "FAILED"


def test_resume_reconstruction_ignores_writers_from_old_plan_hash(tmp_path: Path) -> None:
    _, _, _, _, _, result, _ = _create(tmp_path)
    persistence = _save_resume_state(tmp_path, result)
    old = persistence.projects_dir / "project-1" / "writers" / "old-plan" / "manifest.json"
    atomic_write_json(old, {
        "status": "INCOMPLETE", "season_plan_hash": "old-plan",
        "expected_output_count": 8, "completed_response_count": 7,
    })
    snapshot = reconstruct_project_progress(persistence, "project-1")
    assert snapshot["stage"] == WorkflowStage.WRITERS.value
    assert snapshot.get("completed") != 7
    assert snapshot.get("total") != 8
