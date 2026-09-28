"""Phase 9 deterministic Writer validation, targeted repair, and schema 3.0 merge."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
from typing import Any,Sequence
from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer.season_plan import SeasonPlan
from toolrecap_v4.analysis.finalizer.writer import WriterArtifact,WriterJob,assemble_writer_jobs,verify_locked_plan_artifact,verify_visual_artifact
from toolrecap_v4.analysis.finalizer.writer_store import WriterStore
from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.analysis.scanner.prompts import measure_text_request_bytes
from toolrecap_v4.analysis.vision.models import VisualRunResult
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError,FinalJsonMappingError,FinalJsonValidationError,PayloadContextError,WriterCapacityError,WriterRepairExhaustedError,WriterValidationError,ZeroOutputError
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.persistence import ProjectPersistence,atomic_write_json
from toolrecap_v4.validator import unique_publication_title,validate_project
from toolrecap_v4.original_dialogue import source_dialogue_overlaps
from toolrecap_v4.analysis.finalizer.writer_contract import (
    SOURCE_CLIP_FIELDS,WRITER_DRAFT_VERSION,WRITER_ROOT_FIELDS,WRITER_SEGMENT_FIELDS,
    parse_json_object,safely_unwrap_single_writer,validation_diagnostics,
)

VALIDATOR_VERSION="writer-validator-v5";MAPPING_VERSION="writer-to-final-json-v3";REPAIR_PROTOCOL_VERSION="writer-repair-v3"
def _canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def _digest(v):return hashlib.sha256(_canon(v)).hexdigest()

@dataclass(frozen=True)
class ValidationResult:
    output_id:str;state:str;issues:tuple[str,...];normalized:dict[str,Any]|None;artifact_hash:str;diagnostics:tuple[str,...]=()
    dialogue_matches:tuple[dict[str,Any],...]=()

@dataclass(frozen=True)
class FinalizationResult:
    final_json:dict[str,Any];revision:str;artifact_hash:str;validated_outputs:tuple[ValidationResult,...];repaired_output_ids:tuple[str,...];reused:bool

def parse_writer_response(raw:str,*,project_id:str|None=None,plan_hash:str|None=None,output_id:str|None=None)->dict[str,Any]|None:
    value=parse_json_object(raw)
    if value is None:return None
    if project_id and plan_hash and output_id:
        return safely_unwrap_single_writer(value,project_id,plan_hash,output_id) or value
    return value

class OutputValidator:
    def __init__(self,*,project_id:str,plan:SeasonPlan,episodes:Sequence[PreparedEpisode],visual:VisualRunResult,evidence_store:EvidenceStore):self.project_id=project_id;self.plan=plan;self.eps={e.episode_id:e for e in episodes};self.visual={v.visual_evidence_id:v for v in visual.evidence};self.store=evidence_store
    def validate(self,job:WriterJob,raw:str)->ValidationResult:
        artifact_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest();parsed=parse_json_object(raw);d=parse_writer_response(raw,project_id=self.project_id,plan_hash=self.plan.plan_hash,output_id=job.output_id);issues=[]
        if d is None:return ValidationResult(job.output_id,"INVALID_REPAIRABLE",("MALFORMED_JSON",),None,artifact_hash,validation_diagnostics(("MALFORMED_JSON",),parsed))
        if set(d)!=WRITER_ROOT_FIELDS:issues.append("SCHEMA_FIELD_MISSING_OR_EXTRA")
        if d.get("writer_draft_version")!=WRITER_DRAFT_VERSION:issues.append("SCHEMA_VERSION_INVALID")
        if d.get("project_id")!=self.project_id:issues.append("PROJECT_ID_MISMATCH")
        if d.get("season_plan_hash")!=self.plan.plan_hash:issues.append("SEASON_PLAN_MISMATCH")
        if d.get("output_id")!=job.output_id:issues.append("OUTPUT_ID_MISMATCH")
        if not isinstance(d.get("title"),str) or not d.get("title","").strip():issues.append("MISSING_TITLE")
        narr=d.get("narration")
        if not isinstance(narr,dict) or set(narr)!={"text"} or not isinstance(narr.get("text"),str) or not narr.get("text","").strip() or narr.get("text","").strip() in set(job.context.get("locked_output_order",[])):issues.append("MISSING_NARRATION")
        segments=d.get("segments")
        if not isinstance(segments,list) or not segments:issues.append("SEGMENTS_REQUIRED");segments=[]
        seen=set();normalized=[];allowed_e={x["evidence_id"] for x in job.context["authoritative_full_evidence"]};allowed_v={x["visual_evidence_id"] for x in job.context["authoritative_visual_evidence"]};parents=job.context["source_ranges"]
        fields=WRITER_SEGMENT_FIELDS
        dialogue_matches=[]
        for idx,s in enumerate(segments):
            prefix=f"SEGMENT_{idx}"
            if not isinstance(s,dict) or set(s)!=fields:issues.append(f"{prefix}_SCHEMA_INVALID");continue
            sid=s.get("segment_id")
            if not isinstance(sid,str) or not sid.strip():issues.append("EMPTY_SEGMENT_ID")
            elif sid in seen:issues.append("DUPLICATE_SEGMENT_ID")
            seen.add(sid)
            if not isinstance(s.get("narration_text"),str) or not s["narration_text"].strip():issues.append("MISSING_SEGMENT_NARRATION")
            for k in ("episode_ids","evidence_ids","visual_evidence_ids","source_clips","uncertainty"):
                if not isinstance(s.get(k),list):issues.append(f"{prefix}_{k.upper()}_TYPE")
            if not isinstance(s.get("editorial_intent"),str) or not s["editorial_intent"].strip():issues.append("MISSING_EDITORIAL_INTENT")
            eids=s.get("evidence_ids",[]) if isinstance(s.get("evidence_ids"),list) else []
            vids=s.get("visual_evidence_ids",[]) if isinstance(s.get("visual_evidence_ids"),list) else []
            if any(e not in allowed_e or self.store.get(e,self.plan.evidence_revision) is None for e in eids):issues.append("UNKNOWN_EVIDENCE_ID")
            if any(v not in allowed_v or v not in self.visual for v in vids):issues.append("UNKNOWN_VISUAL_EVIDENCE_ID")
            clips=s.get("source_clips",[]) if isinstance(s.get("source_clips"),list) else []
            if not clips:issues.append("SOURCE_CLIP_REQUIRED")
            clean=[]
            source_dialogue_texts=[]
            for c in clips:
                ep=self.eps.get(c.get("episode_id")) if isinstance(c,dict) else None
                if not isinstance(c,dict) or set(c)!=SOURCE_CLIP_FIELDS:issues.append(f"{prefix}_SOURCE_CLIP_SCHEMA_INVALID")
                if ep is None:issues.append("INVALID_EPISODE_ID");continue
                if c.get("source_id")!=ep.source_id:issues.append("INVALID_SOURCE_ID")
                start,end=c.get("start_ms"),c.get("end_ms")
                if type(start) is not int or type(end) is not int:issues.append("INVALID_TIMESTAMP_TYPE");continue
                if start<0 or end<=start or end>ep.duration_ms:issues.append("RANGE_OUT_OF_BOUNDS");continue
                contained=any(p["episode_id"]==ep.episode_id and p["source_id"]==ep.source_id and p["start_ms"]<=start and end<=p["end_ms"] for p in parents)
                evidence_ground=any((ev:=self.store.get(e,self.plan.evidence_revision)) is not None and ev.episode_id==ep.episode_id and ev.start_ms<=start and end<=ev.end_ms for e in eids)
                visual_ground=any((vv:=self.visual.get(v)) is not None and vv.episode_id==ep.episode_id and vv.start_ms<=start and end<=vv.end_ms for v in vids)
                if not (contained or evidence_ground or visual_ground):issues.append("UNGROUNDED_SOURCE_RANGE")
                source_dialogue_texts.extend(
                    cue.text for cue in ep.transcript.cues
                    if cue.start_ms < end and cue.end_ms > start and cue.text.strip()
                )
                clean.append({"episode_id":ep.episode_id,"source_id":ep.source_id,"start_ms":start,"end_ms":end})
            for evidence_id in eids:
                evidence=self.store.get(evidence_id,self.plan.evidence_revision)
                if evidence is not None:
                    source_dialogue_texts.extend(item.text for item in evidence.dialogue if item.text.strip())
            matches=source_dialogue_overlaps(str(s.get("narration_text", "")),source_dialogue_texts)
            dialogue_matches.extend({**match,"field_path":f"$.segments[{idx}].narration_text","segment_id":sid} for match in matches)
            if any(match["blocking"] for match in matches):
                issues.append("SOURCE_DIALOGUE_ROUTED_TO_NARRATION")
            normalized.append({**s,"source_clips":clean})
        issues=tuple(sorted(set(issues)))
        diagnostics=validation_diagnostics(issues,parsed) if issues else ()
        diagnostics+=tuple(f"{m['field_path']}: copied source speech {m['source_text']!r}; rule={m['rule']}" for m in dialogue_matches if m["blocking"])
        if issues:return ValidationResult(job.output_id,"INVALID_REPAIRABLE",issues,None,artifact_hash,diagnostics,tuple(dialogue_matches))
        return ValidationResult(job.output_id,"VALID",(),{**d,"segments":normalized},artifact_hash,diagnostics,tuple(dialogue_matches))

def map_final_json(*,project_id:str,project_name:str,episodes:Sequence[PreparedEpisode],plan:SeasonPlan,validated:Sequence[ValidationResult])->dict[str,Any]:
    by={r.output_id:r for r in validated};outputs=[];used_titles=set()
    for output_ordinal,out in enumerate(plan.outputs,1):
        r=by.get(out["output_id"])
        if r is None or r.state!="VALID" or r.normalized is None:raise FinalJsonMappingError("Cannot merge before every canonical output is VALID")
        segments=[]
        for s in r.normalized["segments"]:
            for clip_index,c in enumerate(s["source_clips"],1):
                ep=next(e for e in episodes if e.episode_id==c["episode_id"])
                segments.append({"segment_id":s["segment_id"] if len(s["source_clips"])==1 else f"{s['segment_id']}-{clip_index:02d}","source_file":ep.source_basename or ep.source_path.name,"start_ms":c["start_ms"],"end_ms":c["end_ms"],"type":"narration","narration":s["narration_text"] if clip_index == 1 else "","source_audio":True,"audio_intent":"NARRATION_WITH_ORIGINAL_AUDIO","subtitles":[], **({"narration_group": s["segment_id"]} if len(s["source_clips"]) > 1 else {})})
        outputs.append({"render_id":out["output_id"],"title":unique_publication_title(r.normalized["title"],used_titles,output_ordinal),"segments":segments})
    return {"schema_version":"3.0","project_id":project_id,"project_name":project_name,"sources":[{"source_file":e.source_basename or e.source_path.name,"duration_ms":e.duration_ms} for e in episodes],"outputs":outputs}

def route_one_shot(*,output_count:int,capacity_bytes:int|None,request_bytes:int)->str:
    return "ONE_SHOT_ELIGIBLE" if output_count==1 and capacity_bytes is not None and request_bytes<=capacity_bytes else "STAGED"
