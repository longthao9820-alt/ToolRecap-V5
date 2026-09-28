from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
import pytest
from catalog_test_helpers import populated_store
from toolrecap_v4.analysis.finalizer.catalog import CatalogBuilder
from toolrecap_v4.analysis.finalizer.planner import PlannerDraft,DraftOutput,VisualRangeRequest
from toolrecap_v4.analysis.finalizer.season_plan import SeasonPlanService
from toolrecap_v4.analysis.vision import FrameArtifact,FrameExtractionPolicy,FrameExtractor,VisionConfig,VisualEvidenceService
from toolrecap_v4.errors import FrameBudgetError,FinalPlannerValidationError,VisualRequestError
from toolrecap_v4.gateway import GatewayResult

class FakeExtractor:
    def __init__(self,root):self.root=Path(root);self.calls=[]
    def extract(self,req,offset=0,count=None,cancellation_token=None):
        self.calls.append(req); d=self.root/"frames";d.mkdir(exist_ok=True); result=[]
        for i,ts in enumerate((req.start_ms,req.end_ms-1),1):
            p=d/f"{req.visual_request_id}-{i}.jpg";Image.new("RGB",(16,16),(i*30,0,0)).save(p);b=p.read_bytes()
            import hashlib
            result.append(FrameArtifact(f"{req.episode_id}-FR-{offset+i:04d}",req.episode_id,req.visual_request_id,ts,p,hashlib.sha256(b).hexdigest(),req.source_fingerprint,"visual-frame-selection-v1"))
        return tuple(result)

class Gateway:
    def __init__(self):self.images=[];self.text=[]
    def submit_image_chat(self,**kw):
        self.images.append(kw); prompt=json.loads(kw["prompt"]); frames=prompt["frames"]
        raw={"protocol_version":"visual-evidence-v1","visual_request_id":prompt["visual_request"]["visual_request_id"],"episode_id":prompt["visual_request"]["episode_id"],"source_id":prompt["visual_request"]["source_id"],"frames":[{"frame_id":f["frame_id"],"timestamp_ms":f["timestamp_ms"],"observations":["A person enters."]} for f in frames],"range_observation":"A person enters the room.","entities":[],"objects":["door"],"on_screen_text":[],"uncertainty":["Identity is not grounded"]}
        return GatewayResult(raw_response=json.dumps(raw),bytes_sent=100,metadata={"status_code":200})
    def submit_text_chat(self,**kw):
        self.text.append(kw);p=json.loads(kw["prompt"]); vis=p["visual_evidence"]
        raw={"protocol_version":"final-season-plan-draft-v1","action":"FINAL_SEASON_PLAN_DRAFT","project_id":"project-1","catalog_hash":p["catalog_hash"],"evidence_revision":p["evidence_revision"],"visual_revision":p["visual_revision"],"outputs":[{"planner_ref":"draft_output_001","title_concept":"Side story","editorial_thesis":"A smaller arc matters.","story_arc":"Across episodes","episode_ids":["E01","E02"],"evidence_ids":["E01-EV-001"],"visual_evidence_ids":[vis[0]["visual_evidence_id"]] if vis else [],"source_ranges":[{"episode_id":"E01","start_ms":1000,"end_ms":3000}],"uncertainty":[],"writer_brief":{"angle":"Preserve evidence"}}]}
        return GatewayResult(raw_response=json.dumps(raw),bytes_sent=100,metadata={"status_code":200})

def env(tmp_path,visual=True):
    store,rev,episodes,_=populated_store(tmp_path);catalog=CatalogBuilder().build(project_id="project-1",evidence_revision=rev,ordered_episodes=episodes,evidence_store=store)
    vr=(VisualRangeRequest("E01",1000,3000,"verify entrant"),) if visual else ()
    out=DraftOutput("draft_output_001","Work","Thesis","Arc",("E01","E02"),("E01-EV-001",),(),(),vr,())
    return catalog,episodes,PlannerDraft("planner-draft-v1","Rationale",1,(out,),()),rev

def test_selective_visual_and_locked_plan(tmp_path):
    catalog,episodes,draft,rev=env(tmp_path);gw=Gateway();extractor=FakeExtractor(tmp_path)
    visual=VisualEvidenceService(gw,tmp_path,VisionConfig("vision-route"),extractor).run(project_id="project-1",draft=draft,planner_draft_hash="draft-hash",episodes=episodes,evidence_revision=rev)
    assert len(extractor.calls)==1 and len(gw.images)==1
    assert visual.evidence[0].visual_evidence_id=="E01-VIS-001"
    assert visual.evidence[0].uncertainty==("Identity is not grounded",)
    assert all(1000<=t<3000 for t in visual.evidence[0].frame_timestamps_ms)
    plan=SeasonPlanService(gw,tmp_path,"planner-route").run(project_id="project-1",raw_recap_prompt="Raw  prompt\n🙂",catalog=catalog,draft=draft,visual=visual)
    assert plan.plan.outputs[0]["output_id"]=="out_001"
    assert plan.path.name=="season_plan.json" and plan.path.is_file()
    assert json.loads(gw.text[0]["prompt"])["raw_recap_prompt"]=="Raw  prompt\n🙂"
    again=SeasonPlanService(gw,tmp_path,"planner-route").run(project_id="project-1",raw_recap_prompt="Raw  prompt\n🙂",catalog=catalog,draft=draft,visual=visual)
    assert again.reused and len(gw.text)==1

def test_zero_visual_path_makes_no_image_call(tmp_path):
    catalog,episodes,draft,rev=env(tmp_path,False);gw=Gateway();extractor=FakeExtractor(tmp_path)
    visual=VisualEvidenceService(gw,tmp_path,VisionConfig(""),extractor).run(project_id="project-1",draft=draft,planner_draft_hash="d",episodes=episodes,evidence_revision=rev)
    assert visual.completeness["complete"] and visual.completeness["expected"]==0
    assert not gw.images and not extractor.calls
    plan=SeasonPlanService(gw,tmp_path,"planner").run(project_id="project-1",raw_recap_prompt="p",catalog=catalog,draft=draft,visual=visual)
    assert len(plan.plan.outputs)==1

def test_visual_range_validation_and_frame_policy(tmp_path):
    _,episodes,draft,_=env(tmp_path);bad=PlannerDraft(draft.draft_version,draft.editorial_rationale,1,(DraftOutput(**{**draft.proposed_outputs[0].__dict__,"visual_requests":(VisualRangeRequest("E99",0,1,"x"),)}),),())
    from toolrecap_v4.analysis.vision.frames import build_visual_requests
    with pytest.raises(VisualRequestError):build_visual_requests("project-1",bad,episodes)
    req=build_visual_requests("project-1",draft,episodes)[0];ex=FrameExtractor(tmp_path,FrameExtractionPolicy(frames_per_range=3,hard_frame_cap=2),runner=lambda *a,**k:None)
    assert ex.timestamps(req)==(1000,1999,2999)
    with pytest.raises(FrameBudgetError):ex.extract(req)

def test_final_planner_cannot_supply_canonical_ids(tmp_path):
    catalog,episodes,draft,rev=env(tmp_path,False);gw=Gateway();visual=VisualEvidenceService(gw,tmp_path,VisionConfig(""),FakeExtractor(tmp_path)).run(project_id="project-1",draft=draft,planner_draft_hash="d",episodes=episodes,evidence_revision=rev)
    original=gw.submit_text_chat
    def bad(**kw):
        r=original(**kw);d=json.loads(r.raw_response);d["outputs"][0]["output_id"]="out_999";return GatewayResult(raw_response=json.dumps(d),bytes_sent=1)
    gw.submit_text_chat=bad
    with pytest.raises(FinalPlannerValidationError):SeasonPlanService(gw,tmp_path,"planner").run(project_id="project-1",raw_recap_prompt="other",catalog=catalog,draft=draft,visual=visual)
