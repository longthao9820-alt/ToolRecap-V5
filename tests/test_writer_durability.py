"""Writer transaction, UTF-8, identity, and durable-progress regressions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from test_writer_phase8 import Gateway, output, setup
from toolrecap_v4.analysis.finalizer.writer_service import WriterConfig, WriterService
from toolrecap_v4.errors import WriterPersistenceError, WriterTransportError, WriterValidationError


def test_complete_writer_manifest_proves_artifact_identity_and_utf8(tmp_path: Path) -> None:
    store, episodes, plan, visual = setup(tmp_path, outputs=[output("out_001", eids=["E01-EV-001"])])
    result = WriterService(Gateway(), tmp_path, WriterConfig("model")).run(
        project_id="project-1", raw_prompt="Việt Nam — déjà vu 🙂", language="en-US",
        plan=plan, episodes=episodes, visual=visual,
    )
    directory = tmp_path / "projects" / "project-1" / "writers" / plan.plan_hash / "out_001"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads((directory / "writer_draft.json").read_text(encoding="utf-8"))
    assert result.requested_count == 1
    assert manifest["status"] == "COMPLETE"
    assert manifest["artifact_version"] == "writer-artifact-v2"
    assert manifest["project_id"] == "project-1"
    assert manifest["plan_hash"] == plan.plan_hash
    assert manifest["output_id"] == "out_001"
    assert artifact["project_id"] == "project-1"
    assert artifact["plan_hash"] == plan.plan_hash


def test_corrupt_or_truncated_artifact_is_not_complete(tmp_path: Path) -> None:
    store, episodes, plan, visual = setup(tmp_path, outputs=[output("out_001", eids=["E01-EV-001"])])
    service = WriterService(Gateway(), tmp_path, WriterConfig("model"))
    service.run(project_id="project-1", raw_prompt="p", language="en", plan=plan, episodes=episodes, visual=visual)
    artifact = tmp_path / "projects" / "project-1" / "writers" / plan.plan_hash / "out_001" / "writer_draft.json"
    artifact.write_text("{", encoding="utf-8")
    with pytest.raises(WriterValidationError, match="out_001"):
        from toolrecap_v4.analysis.finalizer.finalization_service import FinalizationConfig, FinalizationService
        FinalizationService(Gateway(), tmp_path, FinalizationConfig("model")).run(
            project_id="project-1", project_name="Project", raw_prompt="p", language="en",
            plan=plan, episodes=episodes, visual=visual,
        )


def test_fault_after_artifact_write_never_emits_complete(tmp_path: Path) -> None:
    store, episodes, plan, visual = setup(tmp_path, outputs=[output("out_001", eids=["E01-EV-001"])])
    import toolrecap_v4.analysis.finalizer.writer_store as module
    real_write = module.atomic_write_json

    def fail_complete(path, data, *args, **kwargs):
        if path.name == "manifest.json" and data.get("status") == "COMPLETE":
            raise OSError("manifest interruption")
        return real_write(path, data, *args, **kwargs)

    events: list[dict] = []
    with patch.object(module, "atomic_write_json", side_effect=fail_complete):
        with pytest.raises(WriterTransportError):
            WriterService(Gateway(), tmp_path, WriterConfig("model"), progress_callback=events.append).run(
                project_id="project-1", raw_prompt="p", language="en", plan=plan, episodes=episodes, visual=visual,
            )
    directory = tmp_path / "projects" / "project-1" / "writers" / plan.plan_hash / "out_001"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] != "COMPLETE"
    assert not any(event.get("activity_text") == "All Writer outputs are complete." for event in events)


def test_progress_only_reports_durable_writer_completion(tmp_path: Path) -> None:
    outputs = [output("out_001", eids=["E01-EV-001"]), output("out_002", eids=["E01-EV-001"])]
    store, episodes, plan, visual = setup(tmp_path, outputs=outputs)
    events: list[dict] = []
    from test_finalization_phase9 import valid_writer
    from toolrecap_v4.gateway import GatewayResult
    class OneValidGateway(Gateway):
        def submit_text_chat(self,**kwargs):
            prompt=json.loads(kwargs["prompt"]);output_id=prompt["output_id"]
            if output_id=="out_002":raise WriterTransportError("failed")
            value=valid_writer(output_id);value["season_plan_hash"]=prompt["season_plan_hash"]
            return GatewayResult(raw_response=json.dumps(value),bytes_sent=1,metadata={"status_code":200})
    service = WriterService(OneValidGateway(), tmp_path, WriterConfig("model"), progress_callback=events.append)
    with pytest.raises(WriterTransportError):
        service.run(project_id="project-1", raw_prompt="p", language="en", plan=plan, episodes=episodes, visual=visual)
    completion_events = [event for event in events if event.get("item_event") == "complete"]
    assert len(completion_events) == 1
    assert completion_events[0]["completed"] == 1
    assert completion_events[0]["total"] == 2
    assert not any(event.get("activity_text") == "All Writer outputs are complete." for event in events)
