from __future__ import annotations
import json
from pathlib import Path
import pytest
from test_writer_phase8 import Gateway as BaseGateway,output,setup
from toolrecap_v4.analysis.finalizer.finalization import route_one_shot
from toolrecap_v4.analysis.finalizer.finalization_service import FinalizationConfig,FinalizationService
from toolrecap_v4.analysis.finalizer.writer_service import WriterConfig,WriterService
from toolrecap_v4.errors import WriterRepairExhaustedError,ZeroOutputError
from toolrecap_v4.gateway import GatewayResult
from toolrecap_v4.validator import validate_project
RAW="Raw  prompt\n🙂"

def valid_writer(output_id,title=None,evidence="E01-EV-001"):
    return {"writer_draft_version":"writer-draft-v1","project_id":"project-1","season_plan_hash":None,"output_id":output_id,"title":title or f"Title_{output_id}","narration":{"text":"Grounded original narration."},"segments":[{"segment_id":f"{output_id}-seg-001","narration_text":"Grounded original narration.","episode_ids":["E01"],"evidence_ids":[evidence],"visual_evidence_ids":[],"source_clips":[{"episode_id":"E01","source_id":"src_shared","start_ms":1000,"end_ms":2000}],"editorial_intent":"Explain the planned arc.","uncertainty":[]}],"writer_notes":{}}
class WriterGateway:
    def __init__(self,invalid=()):self.calls=[];self.invalid=set(invalid)
    def submit_text_chat(self,**kw):
        self.calls.append(kw);p=json.loads(kw["prompt"]);oid=p["output_id"]
        if oid in self.invalid:raw="{bad"
        else:d=valid_writer(oid);d["season_plan_hash"]=p["season_plan_hash"];raw=json.dumps(d)
        return GatewayResult(raw_response=raw,bytes_sent=len(kw["prompt"]),metadata={"status_code":200})
class RepairGateway:
    def __init__(self,always_invalid=False):self.calls=[];self.always_invalid=always_invalid
    def submit_text_chat(self,**kw):
        self.calls.append(kw);p=json.loads(kw["prompt"]);oid=p["output_id"]
        if self.always_invalid:return GatewayResult(raw_response="{bad",bytes_sent=1)
        d=valid_writer(oid);d["season_plan_hash"]=p["season_plan_hash"]
        return GatewayResult(raw_response=json.dumps(d),bytes_sent=len(kw["prompt"]),metadata={"status_code":200})

def capture(tmp_path,outputs,invalid=()):
    store,episodes,plan,visual=setup(tmp_path,outputs=outputs);gw=WriterGateway(invalid)
    WriterService(gw,tmp_path,WriterConfig("writer-model")).run(project_id="project-1",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    return episodes,plan,visual,gw

def make_repair_gateway(plan,always_invalid=False):
    class G(RepairGateway):
        def submit_text_chat(self,**kw):
            self.calls.append(kw);p=json.loads(kw["prompt"]);d=valid_writer(p["output_id"]);d["season_plan_hash"]=plan.plan_hash
            return GatewayResult(raw_response="{bad" if self.always_invalid else json.dumps(d),bytes_sent=len(kw["prompt"]),metadata={"status_code":200})
    return G(always_invalid)

def test_valid_outputs_merge_schema3_without_repair(tmp_path):
    outs=[output("out_001",eids=["E01-EV-001"]),output("out_002",eids=["E01-EV-001"])];episodes,plan,visual,_=capture(tmp_path,outs);repair=make_repair_gateway(plan)
    result=FinalizationService(repair,tmp_path,FinalizationConfig("writer-model")).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert not repair.calls and not result.repaired_output_ids
    assert result.final_json["schema_version"]=="3.0" and [o["render_id"] for o in result.final_json["outputs"]]==["out_001","out_002"]
    validate_project(result.final_json,{e.source_basename:e.duration_ms for e in episodes})
    assert all("evidence_ids" not in s for o in result.final_json["outputs"] for s in o["segments"])

def test_only_invalid_output_is_repaired_and_original_preserved(tmp_path):
    outs=[output("out_001",eids=["E01-EV-001"]),output("out_002",eids=["E01-EV-001"]),output("out_003",eids=["E01-EV-001"])];episodes,plan,visual,_=capture(tmp_path,outs,invalid={"out_002"});repair=make_repair_gateway(plan)
    original=(tmp_path/"projects"/"project-1"/"writers"/plan.plan_hash/"out_002"/"raw_response.txt").read_bytes()
    result=FinalizationService(repair,tmp_path,FinalizationConfig("writer-model")).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert result.repaired_output_ids==("out_002",) and len(repair.calls)==1
    assert json.loads(repair.calls[0]["prompt"])["validation_issue_codes"]==["MALFORMED_JSON"]
    assert (tmp_path/"projects"/"project-1"/"writers"/plan.plan_hash/"out_002"/"raw_response.txt").read_bytes()==original

def test_repair_exhaustion_blocks_final_and_preserves_valid_sibling(tmp_path):
    outs=[output("out_001",eids=["E01-EV-001"]),output("out_002",eids=["E01-EV-001"])];episodes,plan,visual,_=capture(tmp_path,outs,invalid={"out_002"});repair=make_repair_gateway(plan,True)
    with pytest.raises(WriterRepairExhaustedError):FinalizationService(repair,tmp_path,FinalizationConfig("writer-model",repair_attempts=2)).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert len(repair.calls)==2 and not list(tmp_path.rglob("finalization/*/final.json"))
    assert list(tmp_path.rglob("out_001/validated_output.json"))

def test_final_json_reused_without_ai(tmp_path):
    episodes,plan,visual,_=capture(tmp_path,[output("out_001",eids=["E01-EV-001"])]);gw=make_repair_gateway(plan);service=FinalizationService(gw,tmp_path,FinalizationConfig("writer-model"));a=service.run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual);b=service.run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert b.reused and a.final_json==b.final_json and not gw.calls

def test_saved_repair_response_recovers_after_crash_without_repeat_ai(tmp_path,monkeypatch):
    outs=[output("out_001",eids=["E01-EV-001"]),output("out_002",eids=["E01-EV-001"])];episodes,plan,visual,_=capture(tmp_path,outs,invalid={"out_002"});gateway=make_repair_gateway(plan);service=FinalizationService(gateway,tmp_path,FinalizationConfig("writer-model"))
    from toolrecap_v4.analysis.finalizer.finalization import OutputValidator
    original=OutputValidator.validate;calls={"n":0}
    def crash_after_repair(self,job,raw):
        result=original(self,job,raw);calls["n"]+=1
        if job.output_id=="out_002" and result.state=="VALID" and calls["n"]>2:raise RuntimeError("crash after repair raw checkpoint")
        return result
    monkeypatch.setattr(OutputValidator,"validate",crash_after_repair)
    with pytest.raises(RuntimeError):service.run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    paid=len(gateway.calls);monkeypatch.setattr(OutputValidator,"validate",original)
    result=service.run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert result.repaired_output_ids==("out_002",) and len(gateway.calls)==paid

def test_corrupt_final_json_is_not_cache_hit_and_rebuilds_locally(tmp_path):
    episodes,plan,visual,_=capture(tmp_path,[output("out_001",eids=["E01-EV-001"])]);gw=make_repair_gateway(plan);service=FinalizationService(gw,tmp_path,FinalizationConfig("writer-model"));a=service.run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    path=tmp_path/"projects"/"project-1"/"finalization"/a.revision/"final.json";path.write_text("{corrupt")
    b=service.run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert b.reused is False and not gw.calls and b.final_json["schema_version"]=="3.0"

def test_zero_output_is_truthful_terminal_limitation(tmp_path):
    episodes,plan,visual,_=capture(tmp_path,[])
    with pytest.raises(ZeroOutputError):FinalizationService(make_repair_gateway(plan),tmp_path,FinalizationConfig("writer-model")).run(project_id="project-1",project_name="Project",raw_prompt=RAW,language="en-US",plan=plan,episodes=episodes,visual=visual)
    assert not list(tmp_path.rglob("final.json"))

@pytest.mark.parametrize("count,capacity,size,expected",[(1,1000,999,"ONE_SHOT_ELIGIBLE"),(2,1000,10,"STAGED"),(1,None,10,"STAGED"),(1,100,101,"STAGED")])
def test_one_shot_routing_foundation_only(count,capacity,size,expected):assert route_one_shot(output_count=count,capacity_bytes=capacity,request_bytes=size)==expected
