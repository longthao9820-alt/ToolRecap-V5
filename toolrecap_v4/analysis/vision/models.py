from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class VisualRequest:
    visual_request_id: str; project_id: str; episode_id: str; source_id: str
    source_path: Path; source_fingerprint: str; start_ms: int; end_ms: int
    purpose: str; planner_ref: str; related_evidence_ids: tuple[str, ...]
    def to_dict(self):
        return {"visual_request_id":self.visual_request_id,"project_id":self.project_id,"episode_id":self.episode_id,"source_id":self.source_id,"source_fingerprint":self.source_fingerprint,"start_ms":self.start_ms,"end_ms":self.end_ms,"purpose":self.purpose,"planner_ref":self.planner_ref,"related_evidence_ids":list(self.related_evidence_ids)}

@dataclass(frozen=True)
class FrameArtifact:
    frame_id: str; episode_id: str; visual_request_id: str; timestamp_ms: int
    path: Path; content_hash: str; source_fingerprint: str; policy_version: str
    def to_dict(self):
        return {"frame_id":self.frame_id,"episode_id":self.episode_id,"visual_request_id":self.visual_request_id,"timestamp_ms":self.timestamp_ms,"content_hash":self.content_hash,"source_fingerprint":self.source_fingerprint,"policy_version":self.policy_version}

@dataclass(frozen=True)
class VisualEvidence:
    visual_evidence_id: str; project_id: str; episode_id: str; source_id: str
    visual_request_id: str; start_ms: int; end_ms: int; frame_ids: tuple[str,...]
    frame_timestamps_ms: tuple[int,...]; observation: str; entities: tuple[str,...]
    objects: tuple[str,...]; on_screen_text: tuple[str,...]; uncertainty: tuple[str,...]
    related_evidence_ids: tuple[str,...]; provenance: dict[str,Any]
    def to_dict(self):
        return {"visual_evidence_id":self.visual_evidence_id,"project_id":self.project_id,"episode_id":self.episode_id,"source_id":self.source_id,"visual_request_id":self.visual_request_id,"start_ms":self.start_ms,"end_ms":self.end_ms,"frame_ids":list(self.frame_ids),"frame_timestamps_ms":list(self.frame_timestamps_ms),"observation":self.observation,"entities":list(self.entities),"objects":list(self.objects),"on_screen_text":list(self.on_screen_text),"uncertainty":list(self.uncertainty),"related_evidence_ids":list(self.related_evidence_ids),"provenance":self.provenance}

@dataclass(frozen=True)
class VisualRunResult:
    visual_revision: str; requests: tuple[VisualRequest,...]; evidence: tuple[VisualEvidence,...]
    completeness: dict[str,Any]; reused_count: int; requested_count: int
