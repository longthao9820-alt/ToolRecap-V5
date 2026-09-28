from __future__ import annotations
import json,threading,time
from pathlib import Path
import pytest
from catalog_test_helpers import populated_store
from toolrecap_v4.analysis.finalizer.season_plan import SeasonPlan
from toolrecap_v4.analysis.finalizer.writer import assemble_writer_jobs,best_effort_extract
from toolrecap_v4.analysis.finalizer.writer_service import WriterConfig,WriterService
from toolrecap_v4.analysis.vision.models import VisualEvidence,VisualRunResult
from toolrecap_v4.errors import WriterCapacityError,WriterContextError,WriterEvidenceError,WriterTransportError,WriterVisualEvidenceError
from toolrecap_v4.gateway import GatewayResult

def make_plan(outputs):
    p=SeasonPlan("season-plan-v1","pending","project-1","prompt-hash","catalog-hash","evr-catalog-test","vis-1","draft-hash","response-hash",tuple(outputs))
    import hashlib
    h=hashlib.sha256(json.dumps(p.semantic_dict(),ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return SeasonPlan(**{**p.__dict__,"plan_hash":h})
def output(oid="out_001",eids=None,vids=None,eps=None):return {"output_id":oid,"planner_ref":"draft_output_001","title_concept":"Secondary arc","editorial_thesis":"Small stories matter","story_arc":"Cross episode","episode_ids":eps or ["E01","E02"],"evidence_ids":eids if eids is not None else ["E01-EV-001","E01-EV-002"],"visual_evidence_ids":vids or [],"source_ranges":[{"episode_id":"E01","start_ms":1000,"end_ms":3000}],"uncertainty":["Identity uncertain"],"writer_brief":{"tone":"thoughtful"}}
def setup(tmp_path,outputs=None,visuals=()):
    store,rev,episodes,_=populated_store(tmp_path);plan=make_plan(outputs if outputs is not None else [output()]);visual=VisualRunResult("vis-1",(),tuple(visuals),{"complete":True},0,0)
    import hashlib
    plan_dir=tmp_path/"projects"/"project-1"/"plans"/"plan-test";plan_dir.mkdir(parents=True,exist_ok=True);plan_data=plan.to_dict();(plan_dir/"season_plan.json").write_text(json.dumps(plan_data),encoding="utf-8");(plan_dir/"manifest.json").write_text(json.dumps({"status":"LOCKED","plan_hash":plan.plan_hash,"artifact_hash":hashlib.sha256(json.dumps(plan_data,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()}),encoding="utf-8")
    visual_dir=tmp_path/"projects"/"project-1"/"visual"/"vis-1";visual_dir.mkdir(parents=True,exist_ok=True);visual_data={"visual_revision":"vis-1","requests":[],"evidence":[v.to_dict() for v in visuals],"completeness":visual.completeness};(visual_dir/"visual_evidence.json").write_text(json.dumps(visual_data),encoding="utf-8");(visual_dir/"manifest.json").write_text(json.dumps({"status":"COMPLETE","data_hash":hashlib.sha256(json.dumps(visual_data,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()}),encoding="utf-8")
    return store,episodes,plan,visual
class Gateway:
    def __init__(self,delay=False,fail=None):self.calls=[];self.active=0;self.max_active=0;self.lock=threading.Lock();self.delay=delay;self.fail=fail
    def submit_text_chat(self,**kw):
        with self.lock:self.active+=1;self.max_active=max(self.max_active,self.active);self.calls.append(kw)
        try:
            if self.delay:time.sleep(.02)
            if self.fail and json.loads(kw["prompt"])["output_id"]==self.fail:raise WriterTransportError("failed")
            p=json.loads(kw["prompt"]);return GatewayResult(raw_response=json.dumps({"writer_draft_version":"writer-draft-v1","project_id":p["project_id"],"season_plan_hash":p["season_plan_hash"],"output_id":p["output_id"],"title":"Draft","narration":{"text":"Original grounded narration."},"segments":[],"writer_notes":{}}),bytes_sent=len(kw["prompt"].encode()),metadata={"status_code":200,"duration_ms":2})
        finally:
            with self.lock:self.active-=1

def test_writer_context_is_complete_verbatim_and_path_safe(tmp_path):
    store,episodes,plan,visual=setup(tmp_path);raw='Keep  spacing\nUnicode Việt 🙂 "quote"';jobs=assemble_writer_jobs(project_id="project-1",raw_prompt=raw,language="en-US",plan=plan,episodes=episodes,evidence_store=store,visual=visual,model="route",reasoning="high")
    c=jobs[0].context;assert c["raw_recap_prompt"]==raw and c["output_id"]=="out_001" and c["output_ordinal"]==1
    assert [e["evidence_id"] for e in c["authoritative_full_evidence"]]==["E01-EV-001","E01-EV-002"]
    assert [e["episode_id"] for e in c["source_mapping"]]==["E01","E02"] and "C:/media" not in json.dumps(c)
    assert c["output_language"]=="en-US"

def test_visual_context_has_observations_not_frame_bytes(tmp_path):
    v=VisualEvidence("E01-VIS-001","project-1","E01","src_shared","VR-E01-001",1000,2000,("E01-FR-0001",),(1000,),"A person enters.",(),("door",),(),("identity uncertain",),("E01-EV-001",),{"frame_hashes":["abc"]})
    store,episodes,plan,visual=setup(tmp_path,outputs=[output(vids=["E01-VIS-001"])],visuals=(v,))
    c=assemble_writer_jobs(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,evidence_store=store,visual=visual,model="m",reasoning="")[0].context
    assert c["authoritative_visual_evidence"][0]["observation"]=="A person enters."
    assert "image" not in json.dumps(c).lower() and "frame bytes" not in json.dumps(c).lower()

@pytest.mark.parametrize("change,error",[(lambda p:p["source_ranges"][0].update(start_ms=True),WriterContextError),(lambda p:p.update(evidence_ids=["E01-EV-999"]),WriterEvidenceError),(lambda p:p.update(visual_evidence_ids=["E01-VIS-999"]),WriterVisualEvidenceError)])
def test_context_integrity_failures_are_explicit(tmp_path,change,error):
    store,episodes,_,visual=setup(tmp_path);o=output();change(o);plan=make_plan([o])
    with pytest.raises(error):assemble_writer_jobs(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,evidence_store=store,visual=visual,model="m",reasoning="")

def test_independent_parallel_writers_and_order_association(tmp_path):
    outs=[output(f"out_{i:03d}",eids=["E01-EV-001"]) for i in range(1,4)]
    store,episodes,plan,visual=setup(tmp_path,outs);gw=Gateway(delay=True);result=WriterService(gw,tmp_path,WriterConfig("route",parallelism=2)).run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)
    assert gw.max_active<=2 and gw.max_active==2
    assert [a.output_id for a in result.artifacts]==["out_001","out_002","out_003"]
    prompts=[json.loads(c["prompt"]) for c in gw.calls];assert all("narration" not in p for p in prompts)

def test_per_output_resume_calls_only_missing(tmp_path):
    outs=[output(f"out_{i:03d}",eids=["E01-EV-001"]) for i in range(1,4)];store,episodes,plan,visual=setup(tmp_path,outs);gw=Gateway();service=WriterService(gw,tmp_path,WriterConfig("route"));first=service.run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual);assert len(gw.calls)==3
    d=tmp_path/"projects"/"project-1"/"writers"/plan.plan_hash/"out_002";(d/"manifest.json").unlink()
    second=service.run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual);assert len(gw.calls)==4 and second.requested_count==1 and second.reused_count==2

def test_raw_response_recovery_avoids_repeat_ai_call(tmp_path,monkeypatch):
    import toolrecap_v4.analysis.finalizer.writer_service as module
    store,episodes,plan,visual=setup(tmp_path);gw=Gateway();service=WriterService(gw,tmp_path,WriterConfig("route"));original=module.best_effort_extract;crashed={"v":False}
    def crash(raw):
        if not crashed["v"]:crashed["v"]=True;raise RuntimeError("crash after raw checkpoint")
        return original(raw)
    monkeypatch.setattr(module,"best_effort_extract",crash)
    with pytest.raises(WriterTransportError):service.run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)
    result=service.run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)
    assert len(gw.calls)==1 and result.reused_count==1

def test_partial_failure_preserves_successful_outputs(tmp_path):
    outs=[output(f"out_{i:03d}",eids=["E01-EV-001"]) for i in range(1,4)];store,episodes,plan,visual=setup(tmp_path,outs);gw=Gateway(fail="out_002");service=WriterService(gw,tmp_path,WriterConfig("route",parallelism=2))
    with pytest.raises(WriterTransportError):service.run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)
    base=tmp_path/"projects"/"project-1"/"writers"/plan.plan_hash
    assert (base/"out_001"/"manifest.json").is_file() and (base/"out_003"/"manifest.json").is_file()
    manifest=json.loads((base/"manifest.json").read_text());assert manifest["status"]=="INCOMPLETE" and manifest["failed_count"]==1

def test_writer_model_and_prompt_change_invalidate_but_downstream_settings_are_absent(tmp_path):
    store,episodes,plan,visual=setup(tmp_path);a=assemble_writer_jobs(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,evidence_store=store,visual=visual,model="a",reasoning="low")[0];b=assemble_writer_jobs(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,evidence_store=store,visual=visual,model="b",reasoning="low")[0]
    assert a.dependency_digest!=b.dependency_digest
    text=json.dumps(a.context).lower()
    for irrelevant in ("voicestudio","audio_mix","renderer_gpu","output_directory","publication"):assert irrelevant not in text

def test_zero_outputs_and_capacity(tmp_path):
    store,episodes,plan,visual=setup(tmp_path,outputs=[]);gw=Gateway();r=WriterService(gw,tmp_path,WriterConfig("model-name-does-not-imply-capacity")).run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual);assert not gw.calls and r.manifest["status"]=="COMPLETE"
    other=tmp_path/"capacity";store,episodes,plan,visual=setup(other,outputs=[output(eids=["E01-EV-001"])]);with_service=WriterService(gw,other,WriterConfig("m",max_request_bytes=1024))
    with pytest.raises(WriterTransportError):with_service.run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)
    assert len(gw.calls)==0

@pytest.mark.parametrize("raw,state",[("{bad","UNPARSED"),(json.dumps({"missing":"fields"}),"PARSED")])
def test_response_capture_is_not_phase9_validation(raw,state):
    assert best_effort_extract(raw)[0]==state

def test_truncated_diagnostic_is_not_complete_response(tmp_path):
    store,episodes,plan,visual=setup(tmp_path)
    class Huge(Gateway):
        def submit_text_chat(self,**kw):
            self.calls.append(kw);return GatewayResult(raw_response="x"*2048,bytes_sent=10,metadata={})
    gw=Huge();service=WriterService(gw,tmp_path,WriterConfig("m",max_response_bytes=1024))
    with pytest.raises(WriterTransportError):service.run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)
    m=json.loads((tmp_path/"projects"/"project-1"/"writers"/plan.plan_hash/"out_001"/"manifest.json").read_text())
    assert m["status"]=="DIAGNOSTIC_RESPONSE_TRUNCATED" and m["full_response"] is False

def test_phase8_creates_no_final_json_voice_or_render_artifacts(tmp_path):
    store,episodes,plan,visual=setup(tmp_path);WriterService(Gateway(),tmp_path,WriterConfig("m")).run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)
    assert not list(tmp_path.rglob("final.json"))
    assert not list(tmp_path.rglob("*.mp4"))
    assert not list(tmp_path.rglob("*.wav"))
