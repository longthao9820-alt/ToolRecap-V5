from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
from typing import Any,Callable,Sequence
from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer.catalog import catalog_detail_hash
from toolrecap_v4.analysis.finalizer.planner import PlannerDraft
from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.analysis.vision.frames import FRAME_POLICY_VERSION,FrameExtractionPolicy,FrameExtractor,build_visual_requests
from toolrecap_v4.analysis.vision.models import VisualEvidence,VisualRunResult
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import VisionRepairExhaustedError,VisionValidationError,VisualStoreError
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.persistence import atomic_write_json
from toolrecap_v4.progress import ActivityState,WorkflowStage,safe_emit

VISION_PROTOCOL_VERSION="visual-evidence-v1"; VISION_PROMPT_VERSION="factual-scene-vision-v2"; VISION_VALIDATION_VERSION="visual-validation-v2"
VISION_RESPONSE_KEYS=("protocol_version","visual_request_id","episode_id","source_id","frames","range_observation","entities","objects","on_screen_text","uncertainty")
@dataclass(frozen=True)
class VisionConfig:
    model:str; reasoning:str=""; repair_attempts:int=1; frame_policy:FrameExtractionPolicy=FrameExtractionPolicy()

def _digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _response_contract(req,frames):
    return {
        "instruction":"Return only this JSON object. Do not use Markdown or code fences. Preserve every supplied identity and timestamp exactly.",
        "additional_properties":False,
        "template":{
            "protocol_version":VISION_PROTOCOL_VERSION,
            "visual_request_id":req.visual_request_id,
            "episode_id":req.episode_id,
            "source_id":req.source_id,
            "frames":[{"frame_id":f.frame_id,"timestamp_ms":f.timestamp_ms,"observations":["factual visible observation"]} for f in frames],
            "range_observation":"non-empty factual observation covering the selected range",
            "entities":["visible entity; use a name only when grounded by nearby evidence"],
            "objects":["visible object"],
            "on_screen_text":["legible on-screen text"],
            "uncertainty":["factual uncertainty"],
        },
        "rules":[
            "Include every supplied frame exactly once and in the supplied order.",
            "Each frame object must contain exactly frame_id, timestamp_ms, and observations.",
            "observations, entities, objects, on_screen_text, and uncertainty must be arrays of strings; use an empty array when none apply.",
            "range_observation must be a non-empty string.",
            "Do not add identities, motives, emotions, or off-screen facts that are not grounded.",
        ],
    }

class VisualEvidenceService:
    def __init__(self,gateway:GatewayClient,root:Path|str,config:VisionConfig,extractor:FrameExtractor|None=None,progress_callback:Callable[[dict[str,Any]],None]|None=None):
        self.gateway=gateway; self.root=Path(root); self.config=config; self.extractor=extractor or FrameExtractor(root,config.frame_policy);self.progress_callback=progress_callback
    def _emit(self,**payload):safe_emit(self.progress_callback,stage=WorkflowStage.VISION.value,**payload)
    def run(self,*,project_id:str,draft:PlannerDraft,planner_draft_hash:str,episodes:Sequence[PreparedEpisode],evidence_revision:str,cancellation_token:CancellationToken|None=None)->VisualRunResult:
        requests=build_visual_requests(project_id,draft,episodes)
        self._emit(state=ActivityState.LOCAL_PROCESSING.value,activity_text=f"Visual request plan contains {len(requests)} ranges.",completed=0,total=len(requests),unit="requests")
        deps={"project_id":project_id,"planner_draft_hash":planner_draft_hash,"requests":[r.to_dict() for r in requests],"policy":FRAME_POLICY_VERSION,"vision_model":self.config.model,"vision_reasoning":self.config.reasoning,"prompt":VISION_PROMPT_VERSION,"schema":VISION_PROTOCOL_VERSION,"validation":VISION_VALIDATION_VERSION,"evidence_revision":evidence_revision}
        revision=f"vis-{_digest(deps)[:24]}"; base=self.root/"projects"/project_id/"visual"/revision; manifest=base/"manifest.json"
        if manifest.is_file():
            try:
                m=json.loads(manifest.read_text()); data=json.loads((base/"visual_evidence.json").read_text())
                if m.get("status")=="COMPLETE" and m.get("dependency_digest")==_digest(deps) and _digest(data)==m.get("data_hash"):
                    ev=tuple(self._from_dict(x) for x in data["evidence"]);self._emit(state=ActivityState.RUNNING.value,activity_text=f"Reused {len(requests)} Visual Evidence requests.",completed=len(requests),total=len(requests),unit="requests",reused=len(requests),stage_status="complete"); return VisualRunResult(revision,requests,ev,data["completeness"],len(requests),0)
            except Exception:pass
        base.mkdir(parents=True,exist_ok=True)
        if not requests:
            data={"visual_revision":revision,"requests":[],"evidence":[],"completeness":{"expected":0,"completed":0,"failed":0,"canceled":0,"selected_frame_count":0,"visual_evidence_count":0,"complete":True}}
            atomic_write_json(base/"visual_evidence.json",data); atomic_write_json(manifest,{"status":"COMPLETE","dependency_digest":_digest(deps),"data_hash":_digest(data)})
            self._emit(state=ActivityState.LOCAL_PROCESSING.value,activity_text="Visual Analysis: Not required.",completed=0,total=0,unit="requests",stage_status="skipped")
            return VisualRunResult(revision,(),(),data["completeness"],0,0)
        store=EvidenceStore(self.root,project_id); all_ev=[]; offsets={}; req_counts={}; states=[]
        for r in requests:req_counts[r.episode_id]=req_counts.get(r.episode_id,0)+1
        for req_index,req in enumerate(requests,1):
            if cancellation_token:cancellation_token.check_cancelled()
            per=max(1,min(self.config.frame_policy.frames_per_range,self.config.frame_policy.frames_per_episode//req_counts[req.episode_id]))
            offset=offsets.get(req.episode_id,0)
            self._emit(state=ActivityState.LOCAL_PROCESSING.value,activity_text=f"Extracting selected frames for {req.visual_request_id}...",episode_id=req.episode_id,current_item=req.visual_request_id,completed=req_index-1,total=len(requests),unit="requests",item_event="start",item_key=req.visual_request_id)
            try: frames=self.extractor.extract(req,offset,per,cancellation_token)
            except Exception as exc:
                states.append({"visual_request_id":req.visual_request_id,"state":"EXTRACTION_FAILED","error":str(exc)[:500]});atomic_write_json(manifest,{"status":"INCOMPLETE","dependency_digest":_digest(deps),"request_states":states});raise
            offsets[req.episode_id]=offset+len(frames)
            self._emit(state=ActivityState.LOCAL_PROCESSING.value,activity_text=f"Extracted {len(frames)} frames for {req.visual_request_id}.",episode_id=req.episode_id,current_item=req.visual_request_id,completed=req_index-1,total=len(requests),unit="requests")
            if not frames:
                states.append({"visual_request_id":req.visual_request_id,"state":"NO_USABLE_FRAME"});atomic_write_json(manifest,{"status":"INCOMPLETE","dependency_digest":_digest(deps),"request_states":states});raise VisualStoreError(f"No usable frame for {req.visual_request_id}")
            nearby=store.query_range(req.episode_id,req.start_ms,req.end_ms,evidence_revision).items
            context=[{"evidence_id":e.evidence_id,"observation":e.observation,"detail_hash":catalog_detail_hash(e)} for e in nearby]
            request_payload={"protocol_version":VISION_PROTOCOL_VERSION,"task":"Report only factual visible observations from these selected still frames.","visual_request":req.to_dict(),"frames":[{"frame_id":f.frame_id,"timestamp_ms":f.timestamp_ms} for f in frames],"nearby_factual_evidence":context,"identity_rule":"Name a person only when nearby factual evidence grounds identity; otherwise preserve ambiguity.","purpose_question":req.purpose,"response_contract":_response_contract(req,frames)}
            prompt=json.dumps(request_payload,ensure_ascii=False,separators=(",",":"))
            raw=""; parsed=None; errors=[]; valid=False
            raw_path=base/f"{req.visual_request_id}.raw.txt";raw_manifest=base/f"{req.visual_request_id}.raw.manifest.json"
            if raw_path.is_file() and raw_manifest.is_file():
                try:
                    rm=json.loads(raw_manifest.read_text());rb=raw_path.read_bytes()
                    if rm.get("status")!="RECEIVED" or rm.get("dependency_digest")!=_digest(deps) or rm.get("truncated") or hashlib.sha256(rb).hexdigest()!=rm.get("response_hash"):raise ValueError("raw integrity")
                    raw=rb.decode("utf-8");parsed=json.loads(raw);self._validate(parsed,req,frames);valid=True
                except Exception: parsed=None
            for attempt in range(self.config.repair_attempts+1) if not valid else ():
                self._emit(state=ActivityState.REPAIRING.value if attempt else ActivityState.WAITING_FOR_AI.value,activity_text=(f"Repairing Vision response — attempt {attempt} / {self.config.repair_attempts}" if attempt else f"Waiting for Vision response for {req.visual_request_id}..."),episode_id=req.episode_id,current_item=req.visual_request_id,waiting_for="AI",retry_attempt=attempt if attempt else None,retry_limit=self.config.repair_attempts if attempt else None)
                if attempt:
                    call_payload={"task":"Correct only technical JSON contract defects. Return the corrected JSON object only.","validation_errors":list(errors),"required_response_contract":_response_contract(req,frames),"original_factual_request":request_payload,"invalid_response":raw[:65536]}
                    call_prompt=json.dumps(call_payload,ensure_ascii=False,separators=(",",":"))
                else:call_prompt=prompt
                result=self.gateway.submit_image_chat(prompt=call_prompt,images=[f.path for f in frames],model=self.config.model,system_prompt="Extract factual visual observations only. Return exactly one JSON object matching the supplied response_contract. No Markdown. No ranking, narration, motives, or off-screen invention.",reasoning_effort=self.config.reasoning or None,expect_json=False,cancellation_token=cancellation_token,phase="visual_evidence")
                raw=result.raw_response; encoded=raw.encode("utf-8");stored=encoded[:2*1024*1024];raw_path.write_bytes(stored);raw_meta={"status":"RECEIVED","dependency_digest":_digest(deps),"response_hash":hashlib.sha256(encoded).hexdigest(),"response_bytes":len(encoded),"stored_bytes":len(stored),"truncated":len(stored)!=len(encoded),"visual_request_id":req.visual_request_id,"attempt":attempt};atomic_write_json(raw_manifest,raw_meta)
                self._emit(state=ActivityState.LOCAL_PROCESSING.value,activity_text=f"Vision response received; validating {req.visual_request_id}...",episode_id=req.episode_id,current_item=req.visual_request_id)
                try:
                    parsed=json.loads(raw); self._validate(parsed,req,frames); valid=True;raw_meta["validation_status"]="VALID";atomic_write_json(raw_manifest,raw_meta);break
                except Exception as exc:
                    errors.append(str(exc));raw_meta["validation_status"]="INVALID";raw_meta["validation_error"]=str(exc)[:1000];atomic_write_json(raw_manifest,raw_meta)
            if not valid:
                states.append({"visual_request_id":req.visual_request_id,"state":"VISION_FAILED"});atomic_write_json(manifest,{"status":"INCOMPLETE","dependency_digest":_digest(deps),"request_states":states});raise VisionRepairExhaustedError(f"Vision repair exhausted for {req.visual_request_id}")
            all_ev.append(VisualEvidence(f"{req.episode_id}-VIS-{sum(1 for e in all_ev if e.episode_id==req.episode_id)+1:03d}",project_id,req.episode_id,req.source_id,req.visual_request_id,req.start_ms,req.end_ms,tuple(f.frame_id for f in frames),tuple(f.timestamp_ms for f in frames),parsed["range_observation"],tuple(parsed.get("entities",[])),tuple(parsed.get("objects",[])),tuple(parsed.get("on_screen_text",[])),tuple(parsed.get("uncertainty",[])),req.related_evidence_ids,{"vision_model":self.config.model,"protocol":VISION_PROTOCOL_VERSION,"frame_hashes":[f.content_hash for f in frames],"response_hash":hashlib.sha256(raw.encode()).hexdigest()}))
            states.append({"visual_request_id":req.visual_request_id,"state":"COMPLETE"})
            self._emit(state=ActivityState.RUNNING.value,activity_text=f"Visual Evidence {req.visual_request_id} completed.",episode_id=req.episode_id,current_item=req.visual_request_id,completed=req_index,total=len(requests),unit="requests",item_event="complete",item_key=req.visual_request_id)
        comp={"expected":len(requests),"completed":len(all_ev),"failed":0,"canceled":0,"selected_frame_count":sum(len(e.frame_ids) for e in all_ev),"visual_evidence_count":len(all_ev),"request_ids":[r.visual_request_id for r in requests],"visual_evidence_ids":[e.visual_evidence_id for e in all_ev],"request_states":states,"complete":True}
        data={"visual_revision":revision,"requests":[r.to_dict() for r in requests],"evidence":[e.to_dict() for e in all_ev],"completeness":comp}; atomic_write_json(base/"visual_evidence.json",data); atomic_write_json(manifest,{"status":"COMPLETE","dependency_digest":_digest(deps),"data_hash":_digest(data)})
        self._emit(state=ActivityState.LOCAL_PROCESSING.value,activity_text=f"Visual Evidence complete: {len(all_ev)} objects.",completed=len(requests),total=len(requests),unit="requests",stage_status="complete")
        return VisualRunResult(revision,requests,tuple(all_ev),comp,0,len(requests))
    def _validate(self,d,req,frames):
        if not isinstance(d,dict) or set(d)!=set(VISION_RESPONSE_KEYS):raise VisionValidationError("Vision response must contain exactly the required top-level fields")
        if d.get("protocol_version")!=VISION_PROTOCOL_VERSION or d.get("visual_request_id")!=req.visual_request_id or d.get("episode_id")!=req.episode_id or d.get("source_id")!=req.source_id:raise VisionValidationError("Vision identity/protocol mismatch")
        known={f.frame_id:f.timestamp_ms for f in frames}; rows=d.get("frames")
        if not isinstance(rows,list) or not isinstance(d.get("range_observation"),str) or not d["range_observation"].strip():raise VisionValidationError("Malformed Vision response")
        if [row.get("frame_id") if isinstance(row,dict) else None for row in rows]!=[f.frame_id for f in frames]:raise VisionValidationError("Vision frames must include every supplied frame exactly once and in order")
        for row in rows:
            if not isinstance(row,dict) or set(row)!={"frame_id","timestamp_ms","observations"} or row.get("frame_id") not in known or type(row.get("timestamp_ms")) is not int or row["timestamp_ms"]!=known[row["frame_id"]]:raise VisionValidationError("Unknown frame, timestamp, or frame fields")
            if not isinstance(row.get("observations"),list) or not all(isinstance(x,str) for x in row["observations"]):raise VisionValidationError("Frame observations must be an array of strings")
        for key in ("entities","objects","on_screen_text","uncertainty"):
            if not isinstance(d.get(key,[]),list) or not all(isinstance(x,str) for x in d.get(key,[])):raise VisionValidationError("Invalid Vision arrays")
    def _from_dict(self,d):return VisualEvidence(d["visual_evidence_id"],d["project_id"],d["episode_id"],d["source_id"],d["visual_request_id"],d["start_ms"],d["end_ms"],tuple(d["frame_ids"]),tuple(d["frame_timestamps_ms"]),d["observation"],tuple(d["entities"]),tuple(d["objects"]),tuple(d["on_screen_text"]),tuple(d["uncertainty"]),tuple(d["related_evidence_ids"]),d["provenance"])
