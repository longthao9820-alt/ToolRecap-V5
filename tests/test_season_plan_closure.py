from __future__ import annotations
import json
import pytest
from test_visual_phase7 import Gateway,FakeExtractor,env
from toolrecap_v4.analysis.vision import VisionConfig,VisualEvidenceService
from toolrecap_v4.analysis.finalizer.season_plan import SeasonPlanService
from toolrecap_v4.errors import FinalPlannerValidationError
from toolrecap_v4.gateway import GatewayResult

def setup(tmp_path,visual=True):
    catalog,episodes,draft,rev=env(tmp_path,visual);gw=Gateway();vis=VisualEvidenceService(gw,tmp_path,VisionConfig("v" if visual else ""),FakeExtractor(tmp_path)).run(project_id="project-1",draft=draft,planner_draft_hash="h",episodes=episodes,evidence_revision=rev);return catalog,draft,vis,gw

@pytest.mark.parametrize("count",[0,1,4])
def test_ai_output_count_and_order_preserved_with_app_owned_ids(tmp_path,count):
    catalog,draft,vis,gw=setup(tmp_path,False)
    def respond(**kw):
        p=json.loads(kw["prompt"]);outs=[]
        for i in range(count):outs.append({"planner_ref":f"draft_output_{i+1:03d}","title_concept":f"Title {i}","editorial_thesis":"thesis","story_arc":"arc","episode_ids":["E02","E01"],"evidence_ids":["E01-EV-001"],"visual_evidence_ids":[],"source_ranges":[],"uncertainty":[],"writer_brief":{}})
        d={"protocol_version":"final-season-plan-draft-v1","action":"FINAL_SEASON_PLAN_DRAFT","project_id":"project-1","catalog_hash":p["catalog_hash"],"evidence_revision":p["evidence_revision"],"visual_revision":p["visual_revision"],"outputs":outs};return GatewayResult(raw_response=json.dumps(d),bytes_sent=1)
    gw.submit_text_chat=respond;result=SeasonPlanService(gw,tmp_path,"p").run(project_id="project-1",raw_recap_prompt="raw",catalog=catalog,draft=draft,visual=vis)
    assert [o["output_id"] for o in result.plan.outputs]==[f"out_{i:03d}" for i in range(1,count+1)]
    if count:assert result.plan.outputs[0]["episode_ids"]==["E02","E01"]

@pytest.mark.parametrize("field,value",[("catalog_hash","bad"),("evidence_revision","bad"),("visual_revision","bad")])
def test_final_identity_mismatch_rejected_before_lock(tmp_path,field,value):
    catalog,draft,vis,gw=setup(tmp_path,False);original=gw.submit_text_chat
    def bad(**kw):r=original(**kw);d=json.loads(r.raw_response);d[field]=value;return GatewayResult(raw_response=json.dumps(d),bytes_sent=1)
    gw.submit_text_chat=bad
    with pytest.raises(FinalPlannerValidationError):SeasonPlanService(gw,tmp_path,"p").run(project_id="project-1",raw_recap_prompt=field,catalog=catalog,draft=draft,visual=vis)
    assert not list(tmp_path.rglob("season_plan.json"))

def test_prompt_change_creates_new_plan_revision_but_voice_is_not_dependency(tmp_path):
    catalog,draft,vis,gw=setup(tmp_path,False);service=SeasonPlanService(gw,tmp_path,"p")
    a=service.run(project_id="project-1",raw_recap_prompt="A",catalog=catalog,draft=draft,visual=vis);b=service.run(project_id="project-1",raw_recap_prompt="B",catalog=catalog,draft=draft,visual=vis)
    assert a.path!=b.path and len(gw.text)==2
