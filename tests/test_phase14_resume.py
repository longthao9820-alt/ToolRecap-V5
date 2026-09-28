"""Disk-backed Phase 14 restart, reuse, and downstream invalidation acceptance."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import shutil

import pytest

from test_phase14_workflow_e2e import (
    RAW_PROMPT, SyntheticGateway, SyntheticVoice, _run, _settings, synthetic_environment,
)
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import AnalysisPipelineUnavailableError, CancelledError, SourceChangedError
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.workflow import ProjectStatus, ProjectWorkflow, WorkflowCallbacks


@pytest.mark.parametrize("checkpoint", [
    ProjectStatus.EVIDENCE_READY,
    ProjectStatus.CATALOG_READY,
    ProjectStatus.PLANNER_DRAFT_READY,
    ProjectStatus.SEASON_PLAN_READY,
    ProjectStatus.WRITER_DRAFTS_READY,
    ProjectStatus.FINAL_JSON_READY,
])
def test_restart_from_durable_ai_checkpoint(synthetic_environment, checkpoint):
    source_dir, persistence = synthetic_environment
    project_id = f"r{list(ProjectStatus).index(checkpoint)}"
    gateway = SyntheticGateway(season=False)
    voice = SyntheticVoice(persistence, project_id)
    first = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=voice)
    first.create_project(project_id, "Restart Project", source_dir / "Episode 01.mp4", prompt=RAW_PROMPT, settings=_settings())

    def stop_at_saved_state(status: str) -> None:
        if status == checkpoint.value:
            raise RuntimeError("synthetic process interruption")

    with pytest.raises(RuntimeError, match="synthetic process interruption"):
        first.start_project(project_id, callbacks=WorkflowCallbacks(on_status_change=stop_at_saved_state))
    assert persistence.load_project(project_id)["status"] == checkpoint.value
    before = gateway.counts().copy()

    restarted = ProjectWorkflow(
        persistence=ProjectPersistence(storage_root=persistence.root), gateway_client=gateway,
        voice_adapter=voice,
    )
    completed = restarted.resume_project(project_id)
    assert completed["status"] == ProjectStatus.COMPLETED.value
    if checkpoint != ProjectStatus.EVIDENCE_READY:
        assert gateway.counts()["scanner"] == before["scanner"]
    if checkpoint in (
        ProjectStatus.PLANNER_DRAFT_READY, ProjectStatus.SEASON_PLAN_READY,
        ProjectStatus.WRITER_DRAFTS_READY, ProjectStatus.FINAL_JSON_READY,
    ):
        assert gateway.counts()["season_planner"] == before["season_planner"]
    if checkpoint in (ProjectStatus.SEASON_PLAN_READY, ProjectStatus.WRITER_DRAFTS_READY, ProjectStatus.FINAL_JSON_READY):
        assert gateway.counts()["visual_evidence"] == before["visual_evidence"]
        assert gateway.counts()["final_planner_refinement"] == before["final_planner_refinement"]
    if checkpoint in (ProjectStatus.WRITER_DRAFTS_READY, ProjectStatus.FINAL_JSON_READY):
        assert gateway.counts()["output_writer"] == before["output_writer"]


def test_restart_from_prepared_checkpoint(synthetic_environment):
    source_dir, persistence = synthetic_environment
    project_id = "restart-prepared"
    gateway = SyntheticGateway(season=False)
    voice = SyntheticVoice(persistence, project_id)
    first = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=voice)
    first.create_project(
        project_id, "Prepared Project", source_dir / "Episode 01.mp4",
        prompt=RAW_PROMPT, settings=replace(_settings(), scanner_model=""),
    )
    with pytest.raises(AnalysisPipelineUnavailableError, match="PREPARED"):
        first.start_project(project_id)
    assert persistence.load_project(project_id)["status"] == ProjectStatus.PREPARED.value
    assert persistence.has_prepared_manifest(project_id)
    assert not gateway.calls
    restarted = ProjectWorkflow(
        persistence=ProjectPersistence(storage_root=persistence.root), gateway_client=gateway,
        voice_adapter=voice,
    )
    assert restarted.resume_project(project_id, settings=_settings())["status"] == ProjectStatus.COMPLETED.value
    assert gateway.counts()["scanner"] == 1


def test_zero_visual_route_finishes_with_no_image_request(synthetic_environment):
    state, _workflow, gateway, _voice, persistence = _run(synthetic_environment, season=False, visual=False)
    assert state["status"] == ProjectStatus.COMPLETED.value
    assert gateway.counts()["visual_evidence"] == 0
    visual_artifacts = list((persistence.root / "projects" / "synthetic-episode" / "visual").glob("*/visual_evidence.json"))
    assert len(visual_artifacts) == 1
    import json
    payload = json.loads(visual_artifacts[0].read_text(encoding="utf-8"))
    assert payload["completeness"]["complete"] is True
    assert payload["completeness"]["expected"] == 0
    assert payload["evidence"] == []


def test_final_json_restart_and_settings_matrix_are_zero_ai(synthetic_environment):
    state, _workflow, gateway, voice, persistence = _run(synthetic_environment, season=False)
    project_id = "synthetic-episode"
    final_path = persistence._final_path(project_id)
    final_before = hashlib.sha256(final_path.read_bytes()).hexdigest()
    original_counts = gateway.counts().copy()
    initial_voice_calls = len(voice.calls)
    restarted = ProjectWorkflow(
        persistence=ProjectPersistence(storage_root=persistence.root), gateway_client=gateway,
        voice_adapter=voice,
    )

    ai_changed = replace(_settings(), scanner_model="new-scanner", vision_model="new-vision",
                         finalizer_model="new-finalizer", planner_model="new-planner", writer_model="new-writer")
    assert restarted.retry_project(project_id, settings=ai_changed)["status"] == ProjectStatus.COMPLETED.value
    assert gateway.counts() == original_counts
    assert len(voice.calls) == initial_voice_calls

    voice_changed = replace(ai_changed, voice_style="different-style")
    restarted.retry_project(project_id, settings=voice_changed)
    assert len(voice.calls) > initial_voice_calls
    voice_after_change = len(voice.calls)
    assert gateway.counts() == original_counts

    mix_changed = replace(voice_changed, original_audio_db=-6.0)
    restarted.retry_project(project_id, settings=mix_changed)
    assert len(voice.calls) == voice_after_change
    assert gateway.counts() == original_counts

    render_changed = replace(mix_changed, quality="high")
    restarted.retry_project(project_id, settings=render_changed)
    assert len(voice.calls) == voice_after_change
    assert gateway.counts() == original_counts

    manual = synthetic_environment[0].parent / "Manual Published"
    moved = restarted.retry_project(project_id, settings=replace(render_changed, output_dir=str(manual)))
    assert Path(moved["output_dir"]) == manual
    assert (manual / "Primary Story.mp4").is_file()
    assert len(voice.calls) == voice_after_change
    assert gateway.counts() == original_counts
    assert hashlib.sha256(final_path.read_bytes()).hexdigest() == final_before


def test_partial_season_resume_and_source_integrity(synthetic_environment):
    state, _workflow, gateway, voice, persistence = _run(synthetic_environment, season=True)
    publication = Path(state["output_dir"])
    (publication / "Cross Episode Story.mp4").unlink()
    before = gateway.counts().copy()
    voice_before = len(voice.calls)
    skipped: list[str] = []
    restarted = ProjectWorkflow(
        persistence=ProjectPersistence(storage_root=persistence.root), gateway_client=gateway,
        voice_adapter=voice,
    )
    completed = restarted.resume_project(
        "synthetic-season", callbacks=WorkflowCallbacks(on_output_skipped=lambda output_id, _checkpoint: skipped.append(output_id)),
    )
    assert completed["status"] == ProjectStatus.COMPLETED.value
    assert skipped == ["out_001", "out_003"]
    assert completed["outputs"]["out_002"]["status"] == "completed"
    assert gateway.counts() == before and len(voice.calls) == voice_before

    source = synthetic_environment[0] / "Episode 01.mp4"
    with source.open("ab") as handle:
        handle.write(b"mutated")
    with pytest.raises(SourceChangedError):
        restarted.retry_project("synthetic-season")
    assert gateway.counts() == before


def test_two_projects_keep_evidence_voice_and_publication_separate(synthetic_environment):
    source_dir, persistence = synthetic_environment
    first, _workflow, _gateway, _voice, persistence = _run(synthetic_environment, season=False)
    other_dir = source_dir.parent.parent / "Other Show" / "S02"
    other_dir.mkdir(parents=True)
    other_video = other_dir / "Episode 01.mp4"
    shutil.copy2(source_dir / "Episode 01.mp4", other_video)
    (other_dir / "Episode 01.en.srt").write_text(
        "1\n00:00:00,200 --> 00:00:01,400\nOther project event.\n", encoding="utf-8"
    )
    second_id = "project-b"
    gateway = SyntheticGateway(season=False)
    voice = SyntheticVoice(persistence, second_id)
    workflow = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=voice)
    workflow.create_project(second_id, "Other Project", other_video, prompt=RAW_PROMPT, settings=_settings())
    second = workflow.start_project(second_id)
    assert first["status"] == second["status"] == ProjectStatus.COMPLETED.value
    assert Path(first["output_dir"]) != Path(second["output_dir"])
    assert (Path(first["output_dir"]) / "Primary Story.mp4").is_file()
    assert (Path(second["output_dir"]) / "Primary Story.mp4").is_file()
    assert first["evidence"]["evidence_revision"] != second["evidence"]["evidence_revision"]
    assert (persistence.root / "projects" / "synthetic-episode" / "downstream" / "voice").is_dir()
    assert (persistence.root / "projects" / second_id / "downstream" / "voice").is_dir()
    from toolrecap_v4.analysis.evidence_store import EvidenceStore
    from toolrecap_v4.errors import EvidenceRevisionError
    with pytest.raises(EvidenceRevisionError):
        EvidenceStore(persistence.root, second_id).get("E01-EV-001", first["evidence"]["evidence_revision"])


def test_cancel_after_final_json_resumes_without_ai(synthetic_environment):
    source_dir, persistence = synthetic_environment
    project_id = "cancel-final"
    gateway = SyntheticGateway(season=False)
    voice = SyntheticVoice(persistence, project_id)
    workflow = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=voice)
    workflow.create_project(project_id, "Cancel Project", source_dir / "Episode 01.mp4", prompt=RAW_PROMPT, settings=_settings())
    token = CancellationToken()

    def cancel_at_final(status: str) -> None:
        if status == ProjectStatus.FINAL_JSON_READY.value:
            token.cancel()

    with pytest.raises(CancelledError):
        workflow.start_project(project_id, cancellation_token=token,
                               callbacks=WorkflowCallbacks(on_status_change=cancel_at_final))
    assert persistence.load_project(project_id)["status"] == ProjectStatus.CANCELLED.value
    assert persistence.has_final_json(project_id)
    before = gateway.counts().copy()
    restarted = ProjectWorkflow(
        persistence=ProjectPersistence(storage_root=persistence.root), gateway_client=gateway,
        voice_adapter=voice,
    )
    assert restarted.resume_project(project_id)["status"] == ProjectStatus.COMPLETED.value
    assert gateway.counts() == before


def test_corrupt_render_and_voice_cache_are_not_reused(synthetic_environment):
    state, _workflow, gateway, voice, persistence = _run(synthetic_environment, season=False)
    project_id = "synthetic-episode"
    output = Path(state["output_dir"]) / "Primary Story.mp4"
    output.write_bytes(b"corrupt rendered file")
    before = gateway.counts().copy()
    voice_before = len(voice.calls)
    restarted = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=voice)
    result = restarted.retry_project(project_id)
    assert result["status"] == ProjectStatus.COMPLETED.value
    assert output.stat().st_size > len(b"corrupt rendered file")
    assert len(voice.calls) == voice_before
    assert gateway.counts() == before

    wavs = list((persistence.root / "projects" / project_id / "downstream" / "voice").rglob("narration.wav"))
    assert len(wavs) == 1
    wavs[0].write_bytes(b"corrupt WAV")
    restarted.retry_project(project_id, settings=replace(_settings(), quality="high"))
    assert len(voice.calls) == voice_before + 1
    assert gateway.counts() == before
