"""Cross-phase semantic invalidation through persisted workflow checkpoints."""

from __future__ import annotations

from dataclasses import replace
import os

import pytest

from test_phase14_workflow_e2e import (
    RAW_PROMPT, SyntheticGateway, SyntheticVoice, _settings, synthetic_environment,
)
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.workflow import ProjectStatus, ProjectWorkflow, WorkflowCallbacks


def _stop_at(target: ProjectStatus) -> WorkflowCallbacks:
    def stop(status: str) -> None:
        if status == target.value:
            raise RuntimeError(f"saved {target.value}")
    return WorkflowCallbacks(on_status_change=stop)


def test_sidecar_scanner_model_reasoning_and_parallelism_invalidation(synthetic_environment):
    source_dir, persistence = synthetic_environment
    project_id = "invalidation-scan"
    gateway = SyntheticGateway(season=False)
    voice = SyntheticVoice(persistence, project_id)
    initial = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=voice)
    initial.create_project(project_id, "Invalidation Project", source_dir / "Episode 01.mp4", prompt=RAW_PROMPT, settings=_settings())
    with pytest.raises(RuntimeError, match="saved evidence_ready"):
        initial.start_project(project_id, callbacks=_stop_at(ProjectStatus.EVIDENCE_READY))
    first_revision = persistence.load_project(project_id)["evidence"]["evidence_revision"]
    scanner_calls = gateway.counts()["scanner"]

    restarted = ProjectWorkflow(
        persistence=ProjectPersistence(storage_root=persistence.root), gateway_client=gateway,
        voice_adapter=voice,
    )
    parallel_only = replace(_settings(), scanner_parallelism=1)
    with pytest.raises(RuntimeError, match="saved evidence_ready"):
        restarted.resume_project(project_id, settings=parallel_only, callbacks=_stop_at(ProjectStatus.EVIDENCE_READY))
    assert persistence.load_project(project_id)["evidence"]["evidence_revision"] == first_revision
    assert gateway.counts()["scanner"] == scanner_calls

    subtitle = source_dir / "Episode 01.en.srt"
    prior_stat = subtitle.stat()
    subtitle.write_text(subtitle.read_text(encoding="utf-8").replace("Primary", "Another"), encoding="utf-8")
    os.utime(subtitle, ns=(prior_stat.st_atime_ns, prior_stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="saved evidence_ready"):
        restarted.resume_project(project_id, settings=parallel_only, callbacks=_stop_at(ProjectStatus.EVIDENCE_READY))
    second_revision = persistence.load_project(project_id)["evidence"]["evidence_revision"]
    assert second_revision != first_revision
    assert gateway.counts()["scanner"] > scanner_calls

    new_model = replace(parallel_only, scanner_model="different-scanner")
    with pytest.raises(RuntimeError, match="saved evidence_ready"):
        restarted.resume_project(project_id, settings=new_model, callbacks=_stop_at(ProjectStatus.EVIDENCE_READY))
    third_revision = persistence.load_project(project_id)["evidence"]["evidence_revision"]
    assert third_revision != second_revision

    new_reasoning = replace(new_model, scanner_reasoning="high")
    with pytest.raises(RuntimeError, match="saved evidence_ready"):
        restarted.resume_project(project_id, settings=new_reasoning, callbacks=_stop_at(ProjectStatus.EVIDENCE_READY))
    assert persistence.load_project(project_id)["evidence"]["evidence_revision"] != third_revision


def test_vision_and_finalizer_changes_reuse_earlier_factual_work(synthetic_environment):
    source_dir, persistence = synthetic_environment
    project_id = "invalidation-visual"
    gateway = SyntheticGateway(season=False)
    voice = SyntheticVoice(persistence, project_id)
    workflow = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=voice)
    workflow.create_project(project_id, "Visual Project", source_dir / "Episode 01.mp4", prompt=RAW_PROMPT, settings=_settings())
    with pytest.raises(RuntimeError, match="saved season_plan_ready"):
        workflow.start_project(project_id, callbacks=_stop_at(ProjectStatus.SEASON_PLAN_READY))
    original_plan = persistence.load_project(project_id)["season_plan"]["plan_hash"]
    original_counts = gateway.counts().copy()
    frame_root = persistence.root / "projects" / project_id / "visual-frames"
    frame_count = len(list(frame_root.rglob("*.jpg")))
    assert frame_count == 2

    restarted = ProjectWorkflow(
        persistence=ProjectPersistence(storage_root=persistence.root), gateway_client=gateway,
        voice_adapter=voice,
    )
    changed_vision = replace(_settings(), vision_model="different-vision")
    with pytest.raises(RuntimeError, match="saved season_plan_ready"):
        restarted.resume_project(project_id, settings=changed_vision, callbacks=_stop_at(ProjectStatus.SEASON_PLAN_READY))
    vision_counts = gateway.counts().copy()
    assert vision_counts["scanner"] == original_counts["scanner"]
    assert vision_counts["season_planner"] == original_counts["season_planner"]
    assert vision_counts["visual_evidence"] == original_counts["visual_evidence"] + 1
    assert len(list(frame_root.rglob("*.jpg"))) == frame_count
    assert persistence.load_project(project_id)["season_plan"]["plan_hash"] != original_plan

    changed_finalizer = replace(changed_vision, planner_model="different-planner", writer_model="different-writer")
    with pytest.raises(RuntimeError, match="saved season_plan_ready"):
        restarted.resume_project(project_id, settings=changed_finalizer, callbacks=_stop_at(ProjectStatus.SEASON_PLAN_READY))
    assert gateway.counts()["scanner"] == vision_counts["scanner"]
    assert gateway.counts()["season_planner"] == vision_counts["season_planner"] + 2
