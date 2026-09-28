from __future__ import annotations

import json
import io
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from toolrecap_v4.analysis.models import PreparedEpisode, Transcript, TranscriptCue
from toolrecap_v4.analysis.scanner import ScannerChunkPolicy, ScannerConfig, ScannerService
from toolrecap_v4.analysis.finalizer import PlannerDraft, PlannerRunResult
from toolrecap_v4.discovery import compute_file_fingerprint
from toolrecap_v4.errors import AnalysisPipelineUnavailableError
from toolrecap_v4.gateway import GatewayResult
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.workflow import ProjectStatus, ProjectWorkflow


class Pipeline:
    def __init__(self, episode: PreparedEpisode):
        self.episode = episode
        self.calls = 0

    def prepare_episode(self, **kwargs):
        self.calls += 1
        return self.episode


class Gateway:
    def __init__(self):
        self.calls = []

    def submit_text_chat(self, **kwargs):
        self.calls.append(kwargs)
        request = json.loads(kwargs["prompt"])
        part = request["transcript_parts"][0]
        raw = json.dumps({
            "episode_id": request["episode_id"], "source_id": request["source_id"], "chunk_id": request["chunk_id"],
            "observations": [{
                "start_ms": part["start_ms"], "end_ms": part["end_ms"], "category": "event",
                "observation": "The speaker greets someone.",
                "cue_refs": [{"cue_id": part["cue_id"], "part_index": part["part_index"]}],
                "entities": [], "modality": "subtitle", "confidence": None, "uncertainty": [],
            }],
        })
        return GatewayResult(raw_response=raw, bytes_sent=1234, metadata={"status_code": 200, "duration_ms": 2})


def test_workflow_reaches_catalog_ready_and_stops_before_phase6(tmp_path, monkeypatch):
    source = tmp_path / "episode01.mkv"
    source.write_bytes(b"synthetic source")
    fingerprint = compute_file_fingerprint(source)
    prepared = PreparedEpisode(
        episode_id="E01", source_id="src_001", source_path=source, source_basename=source.name,
        source_fingerprint=fingerprint.sha256, duration_ms=10_000, canvas_width=1920, canvas_height=1080,
        transcript=Transcript(
            episode_id="E01", source_type="sidecar", source_format="srt",
            cues=(TranscriptCue("E01-CUE-001", 1000, 3000, "Hello there."),),
            provenance_hash="transcript-hash",
        ), artifact_hash="prepared-hash",
    )
    persistence = ProjectPersistence(storage_root=tmp_path / "storage")
    state = {
        "schema_version": "3.0", "project_id": "project-1", "project_name": "Project",
        "status": ProjectStatus.CREATED.value, "output_dir": str(tmp_path / "out"),
        "sources": [{"source_file": source.name, "fingerprint": fingerprint.to_dict(), "duration_ms": 10_000}],
        "source_fingerprints": {source.name: fingerprint.to_dict()},
        "prompt": "THIS CREATIVE PROMPT MUST NEVER ENTER SCANNER", "prompt_hash": "x",
        "gateway_identifier": "local", "settings_snapshot": AppSettings().to_dict(),
        "voice_endpoint": "", "timestamps": {}, "final_json": None,
        "prepared_episodes": {}, "outputs": {}, "error": None,
    }
    persistence.save_project(state)
    gateway = Gateway()
    scanner = ScannerService(
        gateway, persistence.root,
        ScannerConfig("model-id", chunk_policy=ScannerChunkPolicy(30_000, 5000)),
    )
    workflow = ProjectWorkflow(
        persistence=persistence, gateway_client=gateway,
        source_preparation_pipeline=Pipeline(prepared), scanner_service=scanner,
    )
    with pytest.raises(AnalysisPipelineUnavailableError, match="Phase 6"):
        workflow.start_project("project-1")
    saved = persistence.load_project("project-1")
    assert saved["status"] == ProjectStatus.CATALOG_READY.value
    assert saved["evidence"]["total_evidence_count"] == 1
    assert saved["catalog"]["evidence_count"] == 1
    assert saved["catalog"]["episode_count"] == 1
    assert saved["catalog"]["capacity"]["status"] == "UNKNOWN"
    assert persistence.load_checkpoint("project-1", "evidence")["status"] == "completed"
    assert persistence.load_checkpoint("project-1", "catalog")["status"] == "completed"
    assert all("THIS CREATIVE PROMPT" not in call["prompt"] for call in gateway.calls)
    assert all(call["phase"] == "scanner" and "images" not in call for call in gateway.calls)
    gateway_call_count = len(gateway.calls)
    with pytest.raises(AnalysisPipelineUnavailableError, match="Phase 6"):
        workflow.resume_project("project-1")
    resumed = persistence.load_project("project-1")
    assert resumed["status"] == ProjectStatus.CATALOG_READY.value
    assert resumed["catalog"]["reused"] is True
    assert len(gateway.calls) == gateway_call_count

    class PlannerStub:
        def __init__(self):
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            draft_path = tmp_path / "storage" / "projects" / "project-1" / "planning" / "stub" / "planner_draft.json"
            return PlannerRunResult(
                project_id="project-1", session_id="planner-stub", dependency_digest="dep",
                draft=PlannerDraft("planner-draft-v1", "Rationale", 0, (), ()),
                round_count=1, request_measurements=(), reused=False, draft_path=draft_path,
            )

    planner = PlannerStub()
    workflow.planner_service = planner
    workflow.visual_service = SimpleNamespace(run=lambda **kwargs: SimpleNamespace(visual_revision="vis-stub", evidence=(), completeness={"complete": True, "failed": 0, "canceled": 0}))
    workflow.season_plan_service = SimpleNamespace(run=lambda **kwargs: SimpleNamespace(plan=SimpleNamespace(plan_hash="plan-hash", outputs=({"output_id":"out_001"},)), path=tmp_path / "season_plan.json", reused=False))
    with pytest.raises(AnalysisPipelineUnavailableError, match="Phase 8"):
        workflow.resume_project("project-1")
    planned = persistence.load_project("project-1")
    assert planned["status"] == ProjectStatus.SEASON_PLAN_READY.value
    assert planned["planner_draft"]["proposed_output_count"] == 0
    assert planned["season_plan"]["plan_hash"] == "plan-hash"
    assert planner.calls[0]["raw_recap_prompt"] == "THIS CREATIVE PROMPT MUST NEVER ENTER SCANNER"
    assert planner.calls[0]["catalog"].catalog_hash == planned["catalog"]["catalog_hash"]
    workflow.writer_service = SimpleNamespace(run=lambda **kwargs: SimpleNamespace(artifacts=(object(),), reused_count=0, requested_count=1))
    final_json={"schema_version":"3.0","project_id":"project-1","project_name":"Project","sources":[{"source_file":"episode01.mkv","duration_ms":10000}],"outputs":[{"render_id":"out_001","title":"Generated_Output","segments":[{"segment_id":"seg-001","source_file":"episode01.mkv","start_ms":1000,"end_ms":3000,"type":"narration","narration":"Spoken narration.","source_audio":False,"subtitles":[]}]}]}
    def finalize(**kwargs):
        persistence.save_final_json("project-1",final_json)
        return SimpleNamespace(revision="final-stub",artifact_hash="final-hash",final_json=final_json,repaired_output_ids=(),reused=False)
    workflow.finalization_service = SimpleNamespace(run=finalize)
    def wav_bytes():
        buf=io.BytesIO()
        with wave.open(buf,"wb") as wav:wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(8000);wav.writeframes(b"\x00\x00"*800)
        return buf.getvalue()
    class Voice:
        calls=0
        def synthesize(self,*args,**kwargs):
            assert persistence.has_final_json("project-1")
            self.calls+=1;return wav_bytes()
    voice=Voice();workflow.voice_adapter=voice
    render_calls={"count":0}
    def fake_render(**kwargs):
        render_calls["count"]+=1
        if render_calls["count"]==1:
            raise RuntimeError("simulated downstream render failure")
        destination=Path(kwargs["output_dir"]);out=destination/"Generated_Output.mp4";out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b"rendered")
        narration=destination/"Generated_Output.narration.srt";original=destination/"Generated_Output.original.srt"
        narration.write_text("1\n00:00:00,000 --> 00:00:01,000\nSpoken narration.\n",encoding="utf-8")
        original.write_text("",encoding="utf-8")
        return SimpleNamespace(output_path=out,narration_srt_path=narration,original_srt_path=original,duration=2.0,video_codec="h264",audio_codec="aac",width=640,height=480,fps=25.0,original_subtitle_state="NO_ORIGINAL_DIALOGUE",original_subtitle_cue_count=0)
    monkeypatch.setattr("toolrecap_v4.workflow.render_output",fake_render)
    gateway_calls=len(gateway.calls)
    with pytest.raises(RuntimeError,match="simulated downstream render failure"):
        workflow.resume_project("project-1")
    failed=persistence.load_project("project-1")
    assert failed["status"]==ProjectStatus.FAILED.value and persistence.has_final_json("project-1") and voice.calls==1
    retried=workflow.retry_project("project-1")
    assert retried["status"]==ProjectStatus.COMPLETED.value and voice.calls==1 and len(gateway.calls)==gateway_calls
    skipped=workflow.retry_project("project-1")
    assert skipped["status"]==ProjectStatus.COMPLETED.value and render_calls["count"]==2 and voice.calls==1
    manual_destination=tmp_path/"manual-publication"
    moved=workflow.retry_project("project-1",settings=AppSettings(output_dir=str(manual_destination)))
    assert moved["status"]==ProjectStatus.COMPLETED.value and render_calls["count"]==3 and voice.calls==1
    assert Path(moved["output_dir"])==manual_destination and (manual_destination/"Generated_Output.mp4").is_file()
    assert len(gateway.calls)==gateway_calls
