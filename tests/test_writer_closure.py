from __future__ import annotations
import json
from dataclasses import replace
import pytest
from test_writer_phase8 import Gateway,output,setup
from toolrecap_v4.analysis.finalizer.writer import assemble_writer_jobs
from toolrecap_v4.analysis.finalizer.writer_service import WriterConfig,WriterService
from toolrecap_v4.analysis.vision.models import VisualEvidence
from toolrecap_v4.errors import WriterRevisionError,WriterTransportError,WriterVisualEvidenceError

def test_exactly_one_job_per_locked_output_and_no_content_created_jobs(tmp_path):
    outs=[output(f"out_{i:03d}",eids=["E01-EV-001"]) for i in range(1,4)];store,episodes,plan,visual=setup(tmp_path,outputs=outs)
    jobs=assemble_writer_jobs(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,evidence_store=store,visual=visual,model="m",reasoning="")
    assert [j.output_id for j in jobs]==["out_001","out_002","out_003"] and len(jobs)==len(plan.outputs)

def test_unlocked_or_corrupt_plan_artifact_rejected(tmp_path):
    store,episodes,plan,visual=setup(tmp_path);manifest=next(tmp_path.rglob("plans/*/manifest.json"));d=json.loads(manifest.read_text());d["status"]="BUILDING";manifest.write_text(json.dumps(d))
    with pytest.raises(WriterRevisionError):WriterService(Gateway(),tmp_path,WriterConfig("m")).run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)

def test_corrupt_visual_artifact_rejected(tmp_path):
    v=VisualEvidence("E01-VIS-001","project-1","E01","src_shared","VR-E01-001",1000,2000,(),(),"fact",(),(),(),(),(),{})
    store,episodes,plan,visual=setup(tmp_path,outputs=[output(vids=["E01-VIS-001"])],visuals=(v,));path=tmp_path/"projects"/"project-1"/"visual"/"vis-1"/"visual_evidence.json";d=json.loads(path.read_text());d["evidence"][0]["observation"]="tampered";path.write_text(json.dumps(d))
    with pytest.raises(WriterVisualEvidenceError):WriterService(Gateway(),tmp_path,WriterConfig("m")).run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)

def test_text_only_transport_and_complete_context(tmp_path):
    store,episodes,plan,visual=setup(tmp_path);gw=Gateway();WriterService(gw,tmp_path,WriterConfig("arbitrary-route")).run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)
    call=gw.calls[0];assert "images" not in call and call["phase"]=="output_writer"
    body=json.loads(call["prompt"]);assert len(body["authoritative_full_evidence"])==2 and "video" not in json.dumps(body).lower() and "image_url" not in json.dumps(body)

def test_writer_response_cannot_change_job_count_or_ids(tmp_path):
    store,episodes,plan,visual=setup(tmp_path);gw=Gateway()
    def response(**kw):
        gw.calls.append(kw);return __import__("toolrecap_v4.gateway",fromlist=["GatewayResult"]).GatewayResult(raw_response=json.dumps({"outputs":["out_999","out_004"]}),bytes_sent=1)
    gw.submit_text_chat=response;r=WriterService(gw,tmp_path,WriterConfig("m")).run(project_id="project-1",raw_prompt="p",language="en",plan=plan,episodes=episodes,visual=visual)
    assert [a.output_id for a in r.artifacts]==["out_001"] and r.artifacts[0].extraction_state=="PARSED"
