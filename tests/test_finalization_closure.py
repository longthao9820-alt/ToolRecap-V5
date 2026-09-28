from __future__ import annotations
import json
import pytest
from test_finalization_phase9 import RAW,capture,make_repair_gateway,output
from toolrecap_v4.analysis.finalizer.finalization_service import FinalizationConfig,FinalizationService
from toolrecap_v4.errors import WriterValidationError
from toolrecap_v4.schemas.schema import get_schema_validator
from toolrecap_v4.validator import validate_project

def test_corrupt_writer_artifact_is_technical_and_never_sent_to_repair(tmp_path):
    episodes,plan,visual,_=capture(tmp_path,[output("out_001",eids=["E01-EV-001"])]);manifest=tmp_path/"projects"/"project-1"/"writers"/plan.plan_hash/"out_001"/"manifest.json";d=json.loads(manifest.read_text());d["response_hash"]="corrupt";manifest.write_text(json.dumps(d));gateway=make_repair_gateway(plan)
    with pytest.raises(WriterValidationError) as raised:FinalizationService(gateway,tmp_path,FinalizationConfig("writer-model")).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert raised.value.issue_codes==("WRITER_ARTIFACT_MISSING",) and not gateway.calls

def test_final_revision_manifest_includes_ordered_validated_hashes(tmp_path):
    episodes,plan,visual,_=capture(tmp_path,[output("out_001",eids=["E01-EV-001"])]);result=FinalizationService(make_repair_gateway(plan),tmp_path,FinalizationConfig("writer-model")).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual);manifest=json.loads((tmp_path/"projects"/"project-1"/"finalization"/result.revision/"manifest.json").read_text())
    assert len(manifest["semantic_dependencies"]["ordered_validated_output_hashes"])==1
    assert manifest["revision"]==result.revision and manifest["status"]=="COMPLETE"

def test_downstream_settings_do_not_enter_finalization_dependencies(tmp_path):
    episodes,plan,visual,_=capture(tmp_path,[output("out_001",eids=["E01-EV-001"])]);result=FinalizationService(make_repair_gateway(plan),tmp_path,FinalizationConfig("writer-model")).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual);manifest=json.loads((tmp_path/"projects"/"project-1"/"finalization"/result.revision/"manifest.json").read_text());text=json.dumps(manifest["semantic_dependencies"]).lower()
    for field in ("voice","audio_mix","renderer","gpu","output_dir","publication"):assert field not in text

def test_zero_outputs_json_schema_allows_but_canonical_validator_rejects():
    value={"schema_version":"3.0","project_id":"project-1","project_name":"Project","sources":[],"outputs":[]}
    get_schema_validator().validate(value)
    with pytest.raises(Exception,match="at least one output"):validate_project(value,{})
