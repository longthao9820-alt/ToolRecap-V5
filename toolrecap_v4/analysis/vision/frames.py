from __future__ import annotations
from dataclasses import dataclass
import hashlib, subprocess
from pathlib import Path
from typing import Callable, Sequence
from toolrecap_v4.analysis.finalizer.planner import PlannerDraft
from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.analysis.vision.models import FrameArtifact, VisualRequest
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import FrameBudgetError, FrameExtractionError, VisualRequestError
from toolrecap_v4.gateway import validate_and_reencode_image
from toolrecap_v4.media import find_binary, run_command

FRAME_POLICY_VERSION="visual-frame-selection-v1"
@dataclass(frozen=True)
class FrameExtractionPolicy:
    frames_per_range:int=6; frames_per_episode:int=24; hard_frame_cap:int=256; max_dimension:int=1920; max_bytes:int=4*1024*1024

def build_visual_requests(project_id:str,draft:PlannerDraft,episodes:Sequence[PreparedEpisode])->tuple[VisualRequest,...]:
    eps={e.episode_id:e for e in episodes}; counts={k:0 for k in eps}; result=[]
    for output in draft.proposed_outputs:
        for vr in output.visual_requests:
            ep=eps.get(vr.episode_id)
            if ep is None or type(vr.start_ms) is not int or type(vr.end_ms) is not int or vr.start_ms<0 or vr.end_ms<=vr.start_ms or vr.end_ms>ep.duration_ms:
                raise VisualRequestError("Invalid Planner visual range; no clamping or substitution is allowed.")
            counts[ep.episode_id]+=1
            result.append(VisualRequest(f"VR-{ep.episode_id}-{counts[ep.episode_id]:03d}",project_id,ep.episode_id,ep.source_id,ep.source_path,ep.source_fingerprint,vr.start_ms,vr.end_ms,vr.purpose,output.draft_ref,tuple(dict.fromkeys((*output.evidence_ids,*output.supporting_evidence_ids)))))
    return tuple(result)

class FrameExtractor:
    def __init__(self,root:Path|str,policy:FrameExtractionPolicy=FrameExtractionPolicy(),runner:Callable|None=None): self.root=Path(root); self.policy=policy; self.runner=runner
    def timestamps(self,request:VisualRequest,count:int|None=None)->tuple[int,...]:
        n=max(1,count or self.policy.frames_per_range); duration=request.end_ms-request.start_ms
        if duration<=1:return (request.start_ms,)
        n=min(n,duration)
        return tuple(request.start_ms+((duration-1)*i//max(1,n-1)) for i in range(n))
    def extract(self,request:VisualRequest,episode_frame_offset:int=0,count:int|None=None,cancellation_token:CancellationToken|None=None)->tuple[FrameArtifact,...]:
        times=self.timestamps(request,count)
        if episode_frame_offset+len(times)>self.policy.hard_frame_cap: raise FrameBudgetError("Visual hard frame safety cap exceeded.")
        key=hashlib.sha256(f"{request.source_fingerprint}|{request.start_ms}|{request.end_ms}|{FRAME_POLICY_VERSION}|{times}".encode()).hexdigest()[:20]
        directory=self.root/"projects"/request.project_id/"visual-frames"/request.visual_request_id/key; directory.mkdir(parents=True,exist_ok=True)
        ffmpeg=find_binary("ffmpeg"); frames=[]
        for i,ts in enumerate(times,1):
            if cancellation_token:cancellation_token.check_cancelled()
            out=directory/f"frame-{i:04d}.jpg"
            if out.is_file():
                clean,_=validate_and_reencode_image(out,max_bytes=self.policy.max_bytes,max_dimension=self.policy.max_dimension);out.write_bytes(clean);digest=hashlib.sha256(clean).hexdigest();frames.append(FrameArtifact(f"{request.episode_id}-FR-{episode_frame_offset+i:04d}",request.episode_id,request.visual_request_id,ts,out,digest,request.source_fingerprint,FRAME_POLICY_VERSION));continue
            cmd=[str(ffmpeg),"-y","-ss",f"{ts/1000:.3f}","-i",str(request.source_path),"-frames:v","1","-vf",f"scale='min({self.policy.max_dimension},iw)':-2",str(out)]
            if self.runner:
                res=self.runner(cmd,capture_output=True,text=True)
                exit_code=getattr(res,"returncode",1)
            else:
                res=run_command(cmd,timeout=60.0,cancellation_token=cancellation_token)
                exit_code=res.exit_code
            if exit_code!=0 or not out.is_file(): raise FrameExtractionError(f"Frame extraction failed for {request.visual_request_id} at {ts}ms")
            clean,_=validate_and_reencode_image(out,max_bytes=self.policy.max_bytes,max_dimension=self.policy.max_dimension)
            out.write_bytes(clean); digest=hashlib.sha256(clean).hexdigest()
            frames.append(FrameArtifact(f"{request.episode_id}-FR-{episode_frame_offset+i:04d}",request.episode_id,request.visual_request_id,ts,out,digest,request.source_fingerprint,FRAME_POLICY_VERSION))
        return tuple(frames)
