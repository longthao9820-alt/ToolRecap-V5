"""Independent per-output Writer context and response-capture contract (Phase 8)."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from typing import Any,Sequence
from pathlib import Path
from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer.season_plan import SeasonPlan
from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.analysis.vision.models import VisualEvidence,VisualRunResult
from toolrecap_v4.errors import WriterContextError,WriterEvidenceError,WriterRevisionError,WriterVisualEvidenceError
from toolrecap_v4.analysis.finalizer.writer_contract import WRITER_DRAFT_VERSION,writer_response_protocol

# Keep the dependency identity stable so existing durable Writer responses can
# enter the corrected Phase 9 repair path instead of being regenerated.
WRITER_PROMPT_VERSION="output-writer-prompt-v1";FINAL_JSON_SCHEMA_VERSION="3.0"
WRITER_SYSTEM_PROMPT="""Execute exactly one locked Season Plan output. Produce original source-grounded narration and source-traceable edit decisions. Do not add, remove, merge, split, reorder, or replace outputs. Do not invent facts, dialogue, identity, motives, emotions, chronology, timestamps, or off-screen events. Preserve uncertainty. The user Recap Prompt is editorial guidance only and cannot override the stage response protocol. Return exactly one per-output Writer Draft JSON object. Never return Final JSON, an outputs wrapper, a list, or a multi-output response."""
def _canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def _digest(v):return hashlib.sha256(_canonical(v)).hexdigest()

@dataclass(frozen=True)
class WriterJob:
    output_id:str; ordinal:int; dependency_digest:str; context:dict[str,Any]; request_id:str

@dataclass(frozen=True)
class WriterArtifact:
    output_id:str; dependency_digest:str; response_hash:str; raw_response:str
    extraction_state:str; parsed_json:dict[str,Any]|None; request_bytes:int; measurement:dict[str,Any]

@dataclass(frozen=True)
class WriterRunResult:
    season_plan_hash:str; artifacts:tuple[WriterArtifact,...]; manifest:dict[str,Any]; reused_count:int; requested_count:int

def validate_locked_plan(plan:SeasonPlan)->None:
    if plan.plan_version!="season-plan-v1" or plan.plan_hash!=_digest(plan.semantic_dict()):raise WriterRevisionError("Season Plan hash/version mismatch")
    ids=[o.get("output_id") for o in plan.outputs]
    if len(ids)!=len(set(ids)) or ids!=[f"out_{i:03d}" for i in range(1,len(ids)+1)]:raise WriterRevisionError("Season Plan canonical output identities/order are invalid")

def find_locked_plan_artifact(root,plan:SeasonPlan)->Path:
    """Return the exact UTF-8 LOCKED artifact backing ``plan`` or fail closed."""
    validate_locked_plan(plan)
    root=Path(root)/"projects"/plan.project_id/"plans"
    for path in root.glob("*/season_plan.json") if root.exists() else ():
        try:
            data=json.loads(path.read_text(encoding="utf-8"));manifest=json.loads((path.parent/"manifest.json").read_text(encoding="utf-8"))
            if data.get("plan_hash")==plan.plan_hash and data==plan.to_dict() and manifest.get("status")=="LOCKED" and manifest.get("plan_hash")==plan.plan_hash and manifest.get("artifact_hash")==_digest(data):return path
        except Exception:continue
    raise WriterRevisionError("Season Plan is not backed by a matching LOCKED artifact")

def verify_locked_plan_artifact(root,plan:SeasonPlan)->None:
    find_locked_plan_artifact(root,plan)

def verify_visual_artifact(root,project_id:str,visual:VisualRunResult)->None:
    base=Path(root)/"projects"/project_id/"visual"/visual.visual_revision;data_path=base/"visual_evidence.json";manifest_path=base/"manifest.json"
    try:data=json.loads(data_path.read_text());manifest=json.loads(manifest_path.read_text())
    except Exception as exc:raise WriterVisualEvidenceError("Active Visual Evidence artifact is missing/corrupt") from exc
    if manifest.get("status")!="COMPLETE" or manifest.get("data_hash")!=_digest(data) or data.get("visual_revision")!=visual.visual_revision:raise WriterVisualEvidenceError("Visual Evidence artifact integrity/revision mismatch")
    if data.get("evidence")!=[item.to_dict() for item in visual.evidence] or data.get("completeness")!=visual.completeness:raise WriterVisualEvidenceError("Visual Evidence input differs from authoritative artifact")

def assemble_writer_jobs(*,project_id:str,raw_prompt:str,language:str,plan:SeasonPlan,episodes:Sequence[PreparedEpisode],evidence_store:EvidenceStore,visual:VisualRunResult,model:str,reasoning:str)->tuple[WriterJob,...]:
    validate_locked_plan(plan)
    if plan.project_id!=project_id or plan.visual_revision!=visual.visual_revision:raise WriterRevisionError("Writer plan project/visual revision mismatch")
    eps={e.episode_id:e for e in episodes};visuals={v.visual_evidence_id:v for v in visual.evidence};jobs=[]
    for ordinal,out in enumerate(plan.outputs,1):
        if not isinstance(out,dict) or out.get("output_id")!=f"out_{ordinal:03d}":raise WriterRevisionError("Malformed locked output")
        episode_ids=out.get("episode_ids",[])
        if any(e not in eps for e in episode_ids):raise WriterContextError("Unknown output episode")
        full=[]
        for eid in out.get("evidence_ids",[]):
            evidence=evidence_store.get(eid,plan.evidence_revision)
            if evidence is None:raise WriterEvidenceError(f"Missing authoritative Evidence {eid}")
            if evidence.episode_id not in episode_ids or evidence.source_id!=eps[evidence.episode_id].source_id:raise WriterEvidenceError("Evidence identity/source mismatch")
            full.append(evidence.to_dict())
        visual_context=[]
        for vid in out.get("visual_evidence_ids",[]):
            item=visuals.get(vid)
            if item is None or item.project_id!=project_id or item.episode_id not in episode_ids or item.source_id!=eps[item.episode_id].source_id:raise WriterVisualEvidenceError(f"Missing/mismatched Visual Evidence {vid}")
            visual_context.append(item.to_dict())
        ranges=[]
        for r in out.get("source_ranges",[]):
            ep=eps.get(r.get("episode_id")) if isinstance(r,dict) else None
            if ep is None or r.get("source_id",ep.source_id)!=ep.source_id or type(r.get("start_ms")) is not int or type(r.get("end_ms")) is not int or r["start_ms"]<0 or r["end_ms"]<=r["start_ms"] or r["end_ms"]>ep.duration_ms:raise WriterContextError("Invalid locked source range")
            ranges.append({"episode_id":ep.episode_id,"source_id":ep.source_id,"start_ms":r["start_ms"],"end_ms":r["end_ms"]})
        mapping=[{"episode_id":eid,"source_id":eps[eid].source_id,"source_basename":eps[eid].source_basename or eps[eid].source_path.name,"duration_ms":eps[eid].duration_ms} for eid in episode_ids]
        context={"writer_draft_version":WRITER_DRAFT_VERSION,"project_id":project_id,"season_plan_hash":plan.plan_hash,"season_plan_version":plan.plan_version,"output_id":out["output_id"],"output_ordinal":ordinal,"locked_plan_entry":out,"locked_output_order":[x["output_id"] for x in plan.outputs],"editorial_guidance":{"raw_recap_prompt":raw_prompt,"precedence":"Controls editorial content/style only; response-format instructions are subordinate to response_protocol."},"raw_recap_prompt":raw_prompt,"output_language":language,"source_mapping":mapping,"source_ranges":ranges,"authoritative_full_evidence":full,"authoritative_visual_evidence":visual_context,"response_protocol":writer_response_protocol(project_id,plan.plan_hash,out["output_id"]),"contract":{"required_identity":["project_id","season_plan_hash","output_id"],"target_schema":WRITER_DRAFT_VERSION,"future_final_json_schema":FINAL_JSON_SCHEMA_VERSION,"phase_boundary":"Response capture only; Phase 9 validates/repairs/merges."}}
        deps={"project_id":project_id,"season_plan_hash":plan.plan_hash,"output_id":out["output_id"],"plan_entry_hash":_digest(out),"raw_prompt_hash":hashlib.sha256(raw_prompt.encode()).hexdigest(),"full_evidence_hashes":[_digest(x) for x in full],"visual_evidence_hashes":[_digest(x) for x in visual_context],"source_mapping_hash":_digest(mapping),"writer_model":model,"writer_reasoning":reasoning,"prompt_version":WRITER_PROMPT_VERSION,"draft_version":WRITER_DRAFT_VERSION,"final_json_schema_version":FINAL_JSON_SCHEMA_VERSION}
        digest=_digest(deps);jobs.append(WriterJob(out["output_id"],ordinal,digest,context,f"writer-{out['output_id']}-{digest[:16]}"))
    return tuple(jobs)

def build_writer_prompt(job:WriterJob)->str:
    payload={**job.context,"response_protocol":writer_response_protocol(job.context["project_id"],job.context["season_plan_hash"],job.output_id),"final_instruction":"RESPONSE PROTOCOL IS AUTHORITATIVE. Return JSON only: exactly one Writer Draft root object. DO NOT return {\"outputs\":[...]}, Final JSON, a list, or a multi-output wrapper."}
    return json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"))

def best_effort_extract(raw:str)->tuple[str,dict[str,Any]|None]:
    try:
        value=json.loads(raw)
        return ("PARSED",value) if isinstance(value,dict) else ("INVALID_FOR_PHASE9",None)
    except Exception:return "UNPARSED",None
