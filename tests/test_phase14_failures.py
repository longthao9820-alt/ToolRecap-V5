"""Phase 14 synthetic failure, import bypass, and artifact preservation acceptance."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from test_phase14_workflow_e2e import (
    RAW_PROMPT, SyntheticGateway, SyntheticVoice, _run, _settings, synthetic_environment,
)
from toolrecap_v4.errors import SourceChangedError, WriterRepairExhaustedError, ZeroOutputError
from toolrecap_v4.gateway import GatewayResult
from toolrecap_v4.output_paths import OutputDirectoryError
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.renderer import RenderError, SourceCollisionError
from toolrecap_v4.workflow import ProjectStatus, ProjectWorkflow


class NoGateway:
    def submit_text_chat(self, **_kwargs):
        raise AssertionError("Final JSON downstream must not call AI")

    def submit_image_chat(self, **_kwargs):
        raise AssertionError("Final JSON downstream must not call Vision")


def test_imported_final_json_uses_real_render_and_zero_ai(synthetic_environment):
    source_dir, persistence = synthetic_environment
    generated, _workflow, _gateway, _voice, persistence = _run(synthetic_environment, season=False)
    imported_json = persistence.load_final_json("synthetic-episode")
    imported_json["project_id"] = "imported-episode"
    imported_json["project_name"] = "Imported Project"
    source = source_dir / "Episode 01.mp4"
    manual = source_dir.parent / "Imported Outputs"
    voice = SyntheticVoice(persistence, "imported-episode")
    workflow = ProjectWorkflow(persistence=persistence, gateway_client=NoGateway(), voice_adapter=voice)
    workflow.import_project("imported-episode", "Imported Project", source, imported_json, output_dir=manual)
    completed = workflow.start_project("imported-episode")
    assert completed["status"] == ProjectStatus.COMPLETED.value
    assert Path(completed["output_dir"]) == manual
    assert (manual / "Primary Story.mp4").is_file()
    assert completed["prepared_episodes"] == {}
    assert voice.calls


def test_writer_repair_exhaustion_preserves_valid_siblings(synthetic_environment):
    source_dir, persistence = synthetic_environment
    project_id = "repair-fail"

    class UnrepairableGateway(SyntheticGateway):
        def submit_text_chat(self, **kwargs):
            if kwargs["phase"] == "writer_repair":
                self._record(kwargs)
                return GatewayResult(raw_response="{still invalid", bytes_sent=1, metadata={"status_code": 200})
            return super().submit_text_chat(**kwargs)

    gateway = UnrepairableGateway(season=True, invalid_writer=True)
    workflow = ProjectWorkflow(
        persistence=persistence, gateway_client=gateway,
        voice_adapter=SyntheticVoice(persistence, project_id),
    )
    workflow.create_project(project_id, "Repair Project", source_dir, prompt=RAW_PROMPT, settings=_settings())
    with pytest.raises(WriterRepairExhaustedError):
        workflow.start_project(project_id)
    assert not persistence.has_final_json(project_id)
    assert gateway.counts()["writer_repair"] == _settings().writer_repair_attempts
    assert list((persistence.root / "projects" / project_id / "writers").glob("*/out_001/manifest.json"))
    assert list((persistence.root / "projects" / project_id / "writers").glob("*/out_003/manifest.json"))


def test_render_failure_retry_preserves_final_and_voice(synthetic_environment, monkeypatch):
    _state, _workflow, gateway, voice, persistence = _run(synthetic_environment, season=False)
    project_id = "synthetic-episode"
    final_path = persistence._final_path(project_id)
    original_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
    before = gateway.counts().copy()
    voice_count = len(voice.calls)
    workflow = ProjectWorkflow(
        persistence=ProjectPersistence(storage_root=persistence.root), gateway_client=gateway,
        voice_adapter=voice,
    )
    import toolrecap_v4.workflow as workflow_module
    with monkeypatch.context() as patcher:
        patcher.setattr(workflow_module, "render_output", lambda **_kwargs: (_ for _ in ()).throw(RenderError("injected FFmpeg failure")))
        with pytest.raises(RenderError, match="injected FFmpeg failure"):
            workflow.retry_project(project_id, settings=replace(_settings(), quality="high"))
    completed = workflow.retry_project(project_id, settings=replace(_settings(), quality="high"))
    assert completed["status"] == ProjectStatus.COMPLETED.value
    assert gateway.counts() == before
    assert len(voice.calls) == voice_count
    assert hashlib.sha256(final_path.read_bytes()).hexdigest() == original_hash


def test_voice_failure_retry_and_publication_failures_are_zero_ai(synthetic_environment):
    state, _workflow, gateway, voice, persistence = _run(synthetic_environment, season=False)
    project_id = "synthetic-episode"
    final_path = persistence._final_path(project_id)
    final_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
    before = gateway.counts().copy()

    class FailingVoice:
        def synthesize(self, *_args, **_kwargs):
            raise RuntimeError("temporary VoiceStudio failure")

    workflow = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=FailingVoice())
    changed = replace(_settings(), voice_style="new-voice-style")
    with pytest.raises(RuntimeError, match="temporary VoiceStudio failure"):
        workflow.retry_project(project_id, settings=changed)
    assert persistence.has_final_json(project_id)
    workflow.voice_adapter = voice
    assert workflow.retry_project(project_id, settings=changed)["status"] == ProjectStatus.COMPLETED.value

    blocked_root = synthetic_environment[0].parent / "blocked-destination"
    blocked_root.write_bytes(b"unrelated file")
    with pytest.raises(OutputDirectoryError):
        workflow.retry_project(project_id, settings=replace(changed, output_dir=str(blocked_root)))
    assert blocked_root.read_bytes() == b"unrelated file"

    collision_root = synthetic_environment[0].parent / "collision-destination"
    collision_root.mkdir()
    unrelated = collision_root / "Primary Story.mp4"
    unrelated.write_bytes(b"unrelated video")
    with pytest.raises(SourceCollisionError):
        workflow.retry_project(project_id, settings=replace(changed, output_dir=str(collision_root)))
    assert unrelated.read_bytes() == b"unrelated video"
    assert gateway.counts() == before
    assert hashlib.sha256(final_path.read_bytes()).hexdigest() == final_hash


def test_zero_output_plan_does_not_create_final_or_render(synthetic_environment):
    source_dir, persistence = synthetic_environment
    project_id = "zero-output"
    gateway = SyntheticGateway(season=False, visual=False, zero_outputs=True)
    voice = SyntheticVoice(persistence, project_id)
    workflow = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=voice)
    workflow.create_project(project_id, "Zero Project", source_dir / "Episode 01.mp4", prompt=RAW_PROMPT, settings=_settings())
    with pytest.raises(ZeroOutputError):
        workflow.start_project(project_id)
    assert not persistence.has_final_json(project_id)
    assert not voice.calls
    assert gateway.counts()["output_writer"] == 0
    assert gateway.counts()["visual_evidence"] == 0
