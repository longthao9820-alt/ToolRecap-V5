from __future__ import annotations
import json
from dataclasses import replace
import pytest
from catalog_test_helpers import populated_store
from test_visual_phase7 import FakeExtractor,Gateway,env
from toolrecap_v4.analysis.finalizer.planner import PlannerDraft,DraftOutput,VisualRangeRequest
from toolrecap_v4.analysis.vision.frames import build_visual_requests
from toolrecap_v4.analysis.vision import VisionConfig,VisualEvidenceService
from toolrecap_v4.errors import VisionValidationError,VisualRequestError
from toolrecap_v4.gateway import GatewayResult

@pytest.mark.parametrize("vr",[
    VisualRangeRequest("E99",0,1,"x"),
    VisualRangeRequest("E01",-1,1,"x"),
    VisualRangeRequest("E01",1,1,"x"),
    VisualRangeRequest("E01",0,100001,"x"),
    VisualRangeRequest("E01",True,2,"x"),
    VisualRangeRequest("E01",0.0,2,"x"),
])
def test_visual_request_ranges_rejected_without_clamping(tmp_path,vr):
    _,episodes,draft,_=env(tmp_path);out=replace(draft.proposed_outputs[0],visual_requests=(vr,));bad=replace(draft,proposed_outputs=(out,))
    with pytest.raises(VisualRequestError):build_visual_requests("project-1",bad,episodes)

def test_visual_request_boundary_and_ids_are_deterministic(tmp_path):
    _,episodes,draft,_=env(tmp_path);vr=VisualRangeRequest("E01",0,100000,"boundary");d=replace(draft,proposed_outputs=(replace(draft.proposed_outputs[0],visual_requests=(vr,vr)),))
    a=build_visual_requests("project-1",d,episodes);b=build_visual_requests("project-1",d,tuple(reversed(episodes)))
    assert [x.visual_request_id for x in a]==["VR-E01-001","VR-E01-002"]==[x.visual_request_id for x in b]
    assert all(x.source_id=="src_shared" and x.source_path.name=="E01.mkv" for x in a)

@pytest.mark.parametrize("mutation",[
    lambda d:d.update(protocol_version="bad"),lambda d:d.update(visual_request_id="invented"),
    lambda d:d.update(episode_id="E99"),lambda d:d.update(source_id="wrong"),
    lambda d:d["frames"][0].update(frame_id="invented"),lambda d:d["frames"][0].update(timestamp_ms=9999),
    lambda d:d.update(entities="Beth"),
])
def test_strict_vision_validation_rejects_identity_and_types(tmp_path,mutation):
    catalog,episodes,draft,rev=env(tmp_path);gw=Gateway();ex=FakeExtractor(tmp_path);service=VisualEvidenceService(gw,tmp_path,VisionConfig("v"),ex)
    req=build_visual_requests("project-1",draft,episodes)[0];frames=ex.extract(req)
    d={"protocol_version":"visual-evidence-v1","visual_request_id":req.visual_request_id,"episode_id":req.episode_id,"source_id":req.source_id,"frames":[{"frame_id":frames[0].frame_id,"timestamp_ms":frames[0].timestamp_ms,"observations":["fact"]}],"range_observation":"fact","entities":[],"objects":[],"on_screen_text":[],"uncertainty":[]}
    mutation(d)
    with pytest.raises(VisionValidationError):service._validate(d,req,frames)

def test_raw_vision_recovery_avoids_repeat_paid_call(tmp_path):
    _,episodes,draft,rev=env(tmp_path);gw=Gateway();service=VisualEvidenceService(gw,tmp_path,VisionConfig("v"),FakeExtractor(tmp_path))
    first=service.run(project_id="project-1",draft=draft,planner_draft_hash="h",episodes=episodes,evidence_revision=rev)
    base=tmp_path/"projects"/"project-1"/"visual"/first.visual_revision
    (base/"manifest.json").unlink();(base/"visual_evidence.json").unlink()
    second=service.run(project_id="project-1",draft=draft,planner_draft_hash="h",episodes=episodes,evidence_revision=rev)
    assert len(gw.images)==1 and second.evidence[0].visual_evidence_id==first.evidence[0].visual_evidence_id

def test_phase4_evidence_artifacts_are_not_mutated(tmp_path):
    store,rev,episodes,_=populated_store(tmp_path);before=(store.base_dir/rev/"episodes"/"E01.json").read_bytes();_,_,draft,_=env(tmp_path)
    VisualEvidenceService(Gateway(),tmp_path,VisionConfig("v"),FakeExtractor(tmp_path)).run(project_id="project-1",draft=draft,planner_draft_hash="h",episodes=episodes,evidence_revision=rev)
    assert (store.base_dir/rev/"episodes"/"E01.json").read_bytes()==before

def test_vision_prompt_is_factual_and_bounded(tmp_path):
    _,episodes,draft,rev=env(tmp_path);gw=Gateway();VisualEvidenceService(gw,tmp_path,VisionConfig("v"),FakeExtractor(tmp_path)).run(project_id="project-1",draft=draft,planner_draft_hash="h",episodes=episodes,evidence_revision=rev)
    prompt=json.loads(gw.images[0]["prompt"]);assert prompt["purpose_question"]=="verify entrant" and "complete_packed_catalog" not in prompt
    contract=prompt["response_contract"];template=contract["template"]
    assert contract["additional_properties"] is False
    assert template["visual_request_id"]=="VR-E01-001" and template["episode_id"]=="E01"
    assert [row["frame_id"] for row in template["frames"]]==["E01-FR-0001","E01-FR-0002"]
    system=gw.images[0]["system_prompt"].lower();assert "factual" in system and "no ranking" in system and "narration" in system

def test_reported_observations_wrapper_is_repaired_with_exact_contract(tmp_path):
    class ContractRepairGateway:
        def __init__(self):self.calls=[]
        def submit_image_chat(self,**kwargs):
            self.calls.append(kwargs);payload=json.loads(kwargs["prompt"])
            if len(self.calls)==1:
                return GatewayResult(raw_response=json.dumps({
                    "protocol_version":"visual-evidence-v1",
                    "visual_request_id":"VR-E01-001",
                    "observations":[
                        {"frame_id":"E01-FR-0001","timestamp_ms":1000,"observation":"Two people stand in a hallway."},
                        {"frame_id":"E01-FR-0002","timestamp_ms":2999,"observation":"The same people face each other."},
                    ],
                }))
            assert payload["task"].startswith("Correct only technical JSON contract defects")
            assert "required_response_contract" in payload
            repaired=payload["required_response_contract"]["template"]
            repaired["range_observation"]="Two people interact in the selected interior range."
            repaired["entities"]=[];repaired["objects"]=[];repaired["on_screen_text"]=[];repaired["uncertainty"]=[]
            return GatewayResult(raw_response=json.dumps(repaired))

    _,episodes,draft,rev=env(tmp_path);gateway=ContractRepairGateway()
    result=VisualEvidenceService(gateway,tmp_path,VisionConfig("vision",repair_attempts=1),FakeExtractor(tmp_path)).run(project_id="project-1",draft=draft,planner_draft_hash="h",episodes=episodes,evidence_revision=rev)
    assert len(gateway.calls)==2 and result.evidence[0].observation.startswith("Two people")
    base=tmp_path/"projects"/"project-1"/"visual"/result.visual_revision
    raw_manifest=json.loads((base/"VR-E01-001.raw.manifest.json").read_text())
    assert raw_manifest["attempt"]==1 and raw_manifest["validation_status"]=="VALID"
