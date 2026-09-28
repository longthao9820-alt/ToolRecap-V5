from __future__ import annotations
import json
import pytest
from test_finalization_phase9 import RAW,capture,valid_writer
from test_writer_phase8 import output
from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer.finalization import OutputValidator,map_final_json
from toolrecap_v4.analysis.finalizer.writer import assemble_writer_jobs

def context(tmp_path):
    episodes,plan,visual,_=capture(tmp_path,[output("out_001",eids=["E01-EV-001"])]);job=assemble_writer_jobs(project_id="project-1",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,evidence_store=EvidenceStore(tmp_path,"project-1"),visual=visual,model="writer-model",reasoning="")[0];validator=OutputValidator(project_id="project-1",plan=plan,episodes=episodes,visual=visual,evidence_store=EvidenceStore(tmp_path,"project-1"));d=valid_writer("out_001");d["season_plan_hash"]=plan.plan_hash;return episodes,plan,job,validator,d

@pytest.mark.parametrize("mutate,code",[
    (lambda d:d.update(output_id="out_002"),"OUTPUT_ID_MISMATCH"),
    (lambda d:d["segments"][0].update(evidence_ids=["E01-EV-999"]),"UNKNOWN_EVIDENCE_ID"),
    (lambda d:d["segments"][0]["source_clips"][0].update(episode_id="E99"),"INVALID_EPISODE_ID"),
    (lambda d:d["segments"][0]["source_clips"][0].update(source_id="wrong"),"INVALID_SOURCE_ID"),
    (lambda d:d["segments"][0]["source_clips"][0].update(start_ms=True),"INVALID_TIMESTAMP_TYPE"),
    (lambda d:d["segments"][0]["source_clips"][0].update(end_ms=100001),"RANGE_OUT_OF_BOUNDS"),
    (lambda d:d["segments"].append(dict(d["segments"][0])),"DUPLICATE_SEGMENT_ID"),
    (lambda d:d["narration"].update(text=""),"MISSING_NARRATION"),
])
def test_deterministic_validation_issue_codes(tmp_path,mutate,code):
    episodes,plan,job,validator,d=context(tmp_path);mutate(d);result=validator.validate(job,json.dumps(d));assert result.state=="INVALID_REPAIRABLE" and code in result.issues and result.issues==tuple(sorted(result.issues))

def test_ungrounded_in_bounds_range_rejected(tmp_path):
    episodes,plan,job,validator,d=context(tmp_path);d["segments"][0]["source_clips"][0].update(start_ms=50000,end_ms=51000);result=validator.validate(job,json.dumps(d));assert "UNGROUNDED_SOURCE_RANGE" in result.issues

def test_deterministic_merge_ignores_validation_completion_order(tmp_path):
    episodes,plan,job,validator,d=context(tmp_path);valid=validator.validate(job,json.dumps(d));a=map_final_json(project_id="project-1",project_name="Project",episodes=episodes,plan=plan,validated=[valid]);b=map_final_json(project_id="project-1",project_name="Project",episodes=episodes,plan=plan,validated=list(reversed([valid])));assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
