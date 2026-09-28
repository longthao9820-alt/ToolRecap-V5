"""Deterministic visual timing for fixed-speed narration."""
from __future__ import annotations
from dataclasses import dataclass

FIT_PENDING="FIT_PENDING";FIT_ANALYZING="FIT_ANALYZING";FIT_ADJUSTED="FIT_ADJUSTED";FIT_READY="FIT_READY";FIT_FAILED="FIT_FAILED"

@dataclass(frozen=True)
class NarrationFitPlan:
    segment_id:str;state:str;start_ms:int;source_end_ms:int;render_duration_ms:int
    narration_duration_ms:int;planned_visual_ms:int;hold_ms:int;strategy:tuple[str,...]

def resolve_narration_fit(*,segment_id:str,start_ms:int,end_ms:int,source_duration_ms:int,narration_duration_ms:int)->NarrationFitPlan:
    if start_ms<0 or end_ms<=start_ms or source_duration_ms<=start_ms:raise ValueError("invalid source bounds for narration fit")
    planned=end_ms-start_ms
    if narration_duration_ms<=0:return NarrationFitPlan(segment_id,FIT_READY,start_ms,end_ms,planned,0,planned,0,("PLANNED_VISUAL",))
    target=narration_duration_ms
    if target<=planned:
        return NarrationFitPlan(segment_id,FIT_READY,start_ms,start_ms+target,target,target,planned,0,("TRIM_TO_NARRATION",))
    available=max(0,source_duration_ms-start_ms);source_visual=min(target,available);hold=max(0,target-source_visual);strategies=[]
    if source_visual>planned:strategies.append("EXTEND_SAME_SOURCE")
    if hold:strategies.append("SOURCE_FRAME_HOLD")
    if not strategies:strategies.append("PLANNED_VISUAL")
    return NarrationFitPlan(segment_id,FIT_READY,start_ms,start_ms+source_visual,target,target,planned,hold,tuple(strategies))
