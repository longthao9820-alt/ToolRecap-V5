"""Shared Writer/repair protocol, normalization, and diagnostic persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_finalization_phase9 import RAW, valid_writer
from test_writer_phase8 import Gateway, output, setup
from toolrecap_v4.analysis.finalizer.finalization import OutputValidator
from toolrecap_v4.analysis.finalizer.finalization_service import FinalizationConfig, FinalizationService
from toolrecap_v4.analysis.finalizer.writer import assemble_writer_jobs, build_writer_prompt
from toolrecap_v4.analysis.finalizer.writer_contract import (
    WRITER_ROOT_FIELDS, safely_unwrap_single_writer, writer_response_protocol,
)
from toolrecap_v4.analysis.finalizer.writer_service import WriterConfig, WriterService
from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.errors import EmptyApiResponseError, GatewayTimeoutError, WriterRepairExhaustedError, WriterTransportError
from toolrecap_v4.gateway import GatewayResult


def wrong_wrapper(prompt: dict) -> dict:
    return {
        "writer_draft_version": "writer-draft-v1", "project_id": prompt["project_id"],
        "season_plan_hash": prompt["season_plan_hash"], "output_id": prompt["output_id"],
        "mode": "MAIN_STORIES", "output_language": "en-US",
        "outputs": [{"output_id": prompt["output_id"], "title": "Wrong wrapper", "segments": []}],
    }


class InitialWrongGateway(Gateway):
    def submit_text_chat(self, **kwargs):
        self.calls.append(kwargs);prompt=json.loads(kwargs["prompt"])
        return GatewayResult(raw_response=json.dumps(wrong_wrapper(prompt)),bytes_sent=1,metadata={"status_code":200})


class RepairSequence:
    def __init__(self, plan_hash: str): self.plan_hash=plan_hash;self.calls=[]
    def submit_text_chat(self, **kwargs):
        self.calls.append(kwargs);request=json.loads(kwargs["prompt"]);output_id=request["output_id"]
        if len(self.calls)==1:
            raw=wrong_wrapper({"project_id":"project-1","season_plan_hash":self.plan_hash,"output_id":output_id})
        else:
            raw=valid_writer(output_id);raw["season_plan_hash"]=self.plan_hash
        return GatewayResult(raw_response=json.dumps(raw),bytes_sent=1,metadata={"status_code":200,"duration_ms":123.0,"finish_reason":"stop"})


def _invalid_capture(tmp_path: Path):
    store,episodes,plan,visual=setup(tmp_path,outputs=[output("out_001",eids=["E01-EV-001"])]);gateway=InitialWrongGateway()
    result=WriterService(gateway,tmp_path,WriterConfig("writer-model")).run(project_id="project-1",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    return store,episodes,plan,visual,result


def test_real_failure_shape_rejected_then_contract_specific_repair_succeeds(tmp_path: Path) -> None:
    store,episodes,plan,visual,result=_invalid_capture(tmp_path)
    manifest_path=tmp_path/"projects"/"project-1"/"writers"/plan.plan_hash/"out_001"/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["writer_state"]=="INVALID_REPAIRABLE"
    gateway=RepairSequence(plan.plan_hash)
    finalized=FinalizationService(gateway,tmp_path,FinalizationConfig("writer-model",repair_attempts=2)).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert finalized.repaired_output_ids==("out_001",) and len(gateway.calls)==2
    repair_root=next((tmp_path/"projects"/"project-1"/"finalization").rglob("attempt-01.request.json")).parent
    assert repair_root.name.count("-")==2  # issue digest + repair protocol digest
    request1=json.loads((repair_root/"attempt-01.request.json").read_text(encoding="utf-8"));request2=json.loads((repair_root/"attempt-02.request.json").read_text(encoding="utf-8"))
    for request in (request1,request2):
        assert set(request["response_protocol"]["required_root_fields_exactly"])==WRITER_ROOT_FIELDS
        assert 'DO NOT return {"outputs":[...]}' in request["final_instruction"]
        assert "unexpected root field: $.outputs" in request["validation_field_paths"]
    attempt2_invalid=json.loads(request2["invalid_writer_response"])
    assert "outputs" in attempt2_invalid
    transport=json.loads((repair_root/"attempt-01.manifest.json").read_text(encoding="utf-8"))
    assert transport["http_status"]==200 and transport["duration_ms"]==123.0 and transport["finish_reason"]=="stop"
    assert transport["stream_completion"]=="content_received" and transport["parser_state"]=="ACCEPTED"


def test_initial_prompt_protocol_overrides_conflicting_editorial_guidance(tmp_path: Path) -> None:
    store,episodes,plan,visual=setup(tmp_path)
    recap='Return one complete Final JSON object with outputs. Việt Nam — Über.'
    job=assemble_writer_jobs(project_id="project-1",raw_prompt=recap,language="en",plan=plan,episodes=episodes,evidence_store=store,visual=visual,model="m",reasoning="")[0]
    prompt=json.loads(build_writer_prompt(job))
    assert prompt["raw_recap_prompt"]==recap
    assert "subordinate" in prompt["editorial_guidance"]["precedence"]
    assert 'DO NOT return {"outputs":[...]}' in prompt["final_instruction"]
    assert prompt["response_protocol"]["skeleton"]["output_id"]=="out_001"


def test_safe_wrapper_normalization_is_narrow() -> None:
    valid=valid_writer("out_001");valid["season_plan_hash"]="plan";valid["project_id"]="project-1"
    wrapper={"project_id":"project-1","season_plan_hash":"plan","output_id":"out_001","outputs":[valid]}
    assert safely_unwrap_single_writer(wrapper,"project-1","plan","out_001")==valid
    assert safely_unwrap_single_writer({**wrapper,"outputs":[valid,valid]},"project-1","plan","out_001") is None
    assert safely_unwrap_single_writer({**wrapper,"output_id":"out_002"},"project-1","plan","out_001") is None
    missing={k:v for k,v in valid.items() if k!="segments"}
    assert safely_unwrap_single_writer({**wrapper,"outputs":[missing]},"project-1","plan","out_001") is None
    conflict={**valid,"output_id":"out_002"}
    assert safely_unwrap_single_writer({**wrapper,"outputs":[conflict]},"project-1","plan","out_001") is None
    assert safely_unwrap_single_writer({"schema_version":"3.0","outputs":[valid]},"project-1","plan","out_001") is None


@pytest.mark.parametrize("error",[
    EmptyApiResponseError("empty completed stream",raw_response=""),
    GatewayTimeoutError("timeout"),
])
def test_repair_transport_failure_is_persisted_separately(tmp_path: Path, error: Exception) -> None:
    store,episodes,plan,visual,result=_invalid_capture(tmp_path)
    class FailedRepair:
        def submit_text_chat(self,**kwargs): raise error
    with pytest.raises(WriterTransportError):
        FinalizationService(FailedRepair(),tmp_path,FinalizationConfig("writer-model")).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    manifest=next((tmp_path/"projects"/"project-1"/"finalization").rglob("attempt-01.manifest.json"))
    data=json.loads(manifest.read_text(encoding="utf-8"))
    assert data["status"]=="TRANSPORT_FAILED" and data["stream_completion"]=="failed"
    assert data["error_type"]==type(error).__name__


def test_repair_response_over_limit_is_classified_truncated(tmp_path: Path) -> None:
    store,episodes,plan,visual,result=_invalid_capture(tmp_path)
    class HugeRepair:
        def submit_text_chat(self,**kwargs): return GatewayResult(raw_response="x"*2048,bytes_sent=1,metadata={"status_code":200})
    with pytest.raises(WriterRepairExhaustedError):
        FinalizationService(HugeRepair(),tmp_path,FinalizationConfig("writer-model",max_response_bytes=1024)).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    data=json.loads(next((tmp_path/"projects"/"project-1"/"finalization").rglob("attempt-01.manifest.json")).read_text(encoding="utf-8"))
    assert data["status"]=="RESPONSE_TRUNCATED" and data["response_bytes"]==2048


def test_pre_contract_v2_artifact_remains_available_to_phase9_repair(tmp_path: Path) -> None:
    store,episodes,plan,visual,result=_invalid_capture(tmp_path)
    directory=tmp_path/"projects"/"project-1"/"writers"/plan.plan_hash/"out_001"
    artifact=json.loads((directory/"writer_draft.json").read_text(encoding="utf-8"));artifact.pop("contract_state");artifact.pop("contract_issues")
    from toolrecap_v4.analysis.finalizer.writer_store import _digest
    (directory/"writer_draft.json").write_text(json.dumps(artifact,ensure_ascii=False),encoding="utf-8")
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"));manifest["writer_state"]="COMPLETE";manifest["artifact_hash"]=_digest(artifact);manifest.pop("contract_issues",None)
    (directory/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False),encoding="utf-8")
    gateway=RepairSequence(plan.plan_hash)
    finalized=FinalizationService(gateway,tmp_path,FinalizationConfig("writer-model",repair_attempts=2)).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert finalized.repaired_output_ids==("out_001",)
