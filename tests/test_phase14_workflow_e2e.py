"""Phase 14 synthetic full-workflow acceptance through production orchestration."""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import subprocess
import threading
import wave

import pytest

from toolrecap_v4.analysis.finalizer.packing import unpack_catalog
from toolrecap_v4.analysis.cache import AnalysisCacheManager
from toolrecap_v4.analysis.source_prep.pipeline import SourcePreparationPipeline
from toolrecap_v4.analysis.source_prep.stt import SttResult, SttStatus
from toolrecap_v4.analysis.source_prep.subtitles.models import SubtitleCue
from toolrecap_v4.gateway import GatewayResult
from toolrecap_v4.media import find_binary, probe_media
from toolrecap_v4.renderer import compute_file_sha256
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.validator import validate_project
from toolrecap_v4.workflow import ProjectStatus, ProjectWorkflow


RAW_PROMPT = 'Keep  unusual spaces\n\nViệt Nam 🙂 — "secondary story"!'


def _video(path: Path, *, color: str, audio: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(find_binary("ffmpeg")), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=25:d=2"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                    "-c:a", "aac", "-shortest"]
    else:
        command += ["-an"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", str(path)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    return path


def _wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * 4800)
    return buffer.getvalue()


class SyntheticVoice:
    def __init__(self, persistence: ProjectPersistence, project_id: str):
        self.persistence = persistence
        self.project_id = project_id
        self.calls: list[str] = []
        self.final_hash_at_first_call: str | None = None

    def synthesize(self, text: str, **_kwargs) -> bytes:
        assert self.persistence.has_final_json(self.project_id)
        if self.final_hash_at_first_call is None:
            self.final_hash_at_first_call = hashlib.sha256(
                self.persistence._final_path(self.project_id).read_bytes()
            ).hexdigest()
        self.calls.append(text)
        return _wav()


class SyntheticGateway:
    """External model boundary only; every response passes production validation."""

    def __init__(self, *, season: bool, invalid_writer: bool = False, visual: bool = True, zero_outputs: bool = False):
        self.season = season
        self.invalid_writer = invalid_writer
        self.visual = visual
        self.zero_outputs = zero_outputs
        self.calls: list[tuple[str, dict]] = []
        self.lock = threading.Lock()
        self.project_id = ""

    def _record(self, kwargs: dict) -> dict:
        assert "images" not in kwargs
        assert "data:video" not in kwargs["prompt"]
        with self.lock:
            self.calls.append((kwargs["phase"], kwargs))
        return json.loads(kwargs["prompt"])

    @staticmethod
    def _result(payload: dict | str, request: dict) -> GatewayResult:
        raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return GatewayResult(raw_response=raw, bytes_sent=len(request["prompt"].encode("utf-8")), metadata={"status_code": 200})

    def submit_text_chat(self, **kwargs) -> GatewayResult:
        request = self._record(kwargs)
        phase = kwargs["phase"]
        if phase == "scanner":
            assert RAW_PROMPT not in kwargs["prompt"]
            observations = []
            for part in (() if request["episode_id"] == "E02" else request["transcript_parts"]):
                observations.append({
                    "start_ms": part["start_ms"], "end_ms": part["end_ms"],
                    "category": "event", "observation": f"{request['episode_id']} factual event: {part['text']}",
                    "cue_refs": [{"cue_id": part["cue_id"], "part_index": part["part_index"]}],
                    "entities": [], "modality": "subtitle", "confidence": None, "uncertainty": [],
                })
            return self._result({
                "episode_id": request["episode_id"], "source_id": request["source_id"],
                "chunk_id": request["chunk_id"], "observations": observations,
            }, kwargs)
        if phase == "season_planner":
            self.project_id = request["project_id"]
            assert request["raw_recap_prompt"] == RAW_PROMPT
            catalog = unpack_catalog(request["catalog_transport"]["complete_packed_catalog"])
            assert len(catalog.items) == request["catalog_identity"]["catalog_item_count"]
            ids = {item.episode_id: item.evidence_id for item in catalog.items}
            common = {
                "protocol_version": "season-planner-protocol-v1", "project_id": request["project_id"],
                "catalog_hash": catalog.catalog_hash, "evidence_revision": catalog.evidence_revision,
                "round_id": request["round_id"],
            }
            if request["round_id"] == "round-001":
                return self._result({**common, "action": "REQUEST_EVIDENCE", "requests": [{
                    "request_id": "fetch-primary", "type": "evidence_ids", "evidence_ids": [ids["E01"]],
                }]}, kwargs)
            assert request["exact_full_evidence_fetch"]["completeness"]["complete"] is True
            specs = [(["E01"], [ids["E01"]], "Primary Story")]
            if self.zero_outputs:
                specs = []
            if self.season:
                specs += [(["E01", "E03"], [ids["E01"], ids["E03"]], "Cross Episode Story"),
                          (["E03"], [ids["E03"]], "Secondary Story")]
            outputs = []
            for index, (episodes, evidence_ids, title) in enumerate(specs, 1):
                requests = ([{"episode_id": episodes[-1], "start_ms": 200, "end_ms": 1400,
                              "purpose": "Verify the selected event"}] if self.visual and index == len(specs) else [])
                outputs.append({
                    "draft_ref": f"draft_output_{index:03d}", "working_title": title,
                    "editorial_thesis": "Preserve factual events.", "story_arc": "Events across selected episodes.",
                    "episode_ids": episodes, "evidence_ids": evidence_ids,
                    "supporting_evidence_ids": [], "connections": [],
                    "visual_requests": requests, "uncertainty": [],
                })
            return self._result({**common, "action": "PLANNER_DRAFT", "draft": {
                "draft_version": "planner-draft-v1", "editorial_rationale": "Complete Catalog considered.",
                "proposed_output_count": len(outputs), "proposed_outputs": outputs, "uncertainty": [],
            }}, kwargs)
        if phase == "final_planner_refinement":
            assert request["raw_recap_prompt"] == RAW_PROMPT
            visual_by_episode = {item["episode_id"]: item["visual_evidence_id"] for item in request["visual_evidence"]}
            outputs = []
            for draft in request["planner_draft"]["proposed_outputs"]:
                episodes = draft["episode_ids"]
                outputs.append({
                    "planner_ref": draft["draft_ref"], "title_concept": draft["working_title"],
                    "editorial_thesis": draft["editorial_thesis"], "story_arc": draft["story_arc"],
                    "episode_ids": episodes, "evidence_ids": draft["evidence_ids"],
                    "visual_evidence_ids": [visual_by_episode[episodes[-1]]] if episodes[-1] in visual_by_episode else [],
                    "source_ranges": [{"episode_id": episode, "start_ms": 200, "end_ms": 1400} for episode in episodes],
                    "uncertainty": [], "writer_brief": {"angle": "Factual recap"},
                })
            return self._result({
                "protocol_version": "final-season-plan-draft-v1", "action": "FINAL_SEASON_PLAN_DRAFT",
                "project_id": request["project_id"], "catalog_hash": request["catalog_hash"],
                "evidence_revision": request["evidence_revision"],
                "visual_revision": request["visual_revision"], "outputs": outputs,
            }, kwargs)
        if phase in ("output_writer", "writer_repair"):
            assert request["raw_recap_prompt"] == RAW_PROMPT
            output_id = request["output_id"]
            if phase == "output_writer" and self.invalid_writer and output_id == "out_002":
                return self._result("{bad", kwargs)
            entry = request["locked_plan_entry"]
            mapping = {item["episode_id"]: item["source_id"] for item in request["source_mapping"]}
            narration = "A factual synthetic recap."
            clips = [{"episode_id": row["episode_id"], "source_id": mapping[row["episode_id"]],
                      "start_ms": 300, "end_ms": 900} for row in entry["source_ranges"]]
            return self._result({
                "writer_draft_version": "writer-draft-v1", "project_id": self.project_id,
                "season_plan_hash": request["season_plan_hash"], "output_id": output_id,
                "title": entry["title_concept"], "narration": {"text": narration},
                "segments": [{"segment_id": f"{output_id}-seg-001", "narration_text": narration,
                              "episode_ids": entry["episode_ids"], "evidence_ids": entry["evidence_ids"],
                              "visual_evidence_ids": entry["visual_evidence_ids"], "source_clips": clips,
                              "editorial_intent": "Describe the factual selected events.", "uncertainty": []}],
                "writer_notes": {},
            }, kwargs)
        raise AssertionError(f"Unexpected Gateway phase: {phase}")

    def submit_image_chat(self, **kwargs) -> GatewayResult:
        assert kwargs["phase"] == "visual_evidence"
        request = json.loads(kwargs["prompt"])
        assert all(Path(path).suffix.lower() in (".jpg", ".jpeg", ".png") for path in kwargs["images"])
        assert all(Path(path).stat().st_size < 4 * 1024 * 1024 for path in kwargs["images"])
        with self.lock:
            self.calls.append(("visual_evidence", kwargs))
        visual_request = request["visual_request"]
        return self._result({
            "protocol_version": "visual-evidence-v1",
            "visual_request_id": visual_request["visual_request_id"],
            "episode_id": visual_request["episode_id"], "source_id": visual_request["source_id"],
            "frames": [{"frame_id": frame["frame_id"], "timestamp_ms": frame["timestamp_ms"],
                        "observations": ["A colored scene is visible."]} for frame in request["frames"]],
            "range_observation": "A colored scene is visible.", "entities": [],
            "objects": [], "on_screen_text": [], "uncertainty": [],
        }, kwargs)

    def counts(self) -> Counter:
        return Counter(phase for phase, _call in self.calls)


def _settings() -> AppSettings:
    return AppSettings(
        scanner_model="scanner-test", planner_model="planner-test", writer_model="writer-test",
        vision_model="vision-test", scanner_parallelism=2, writer_parallelism=2,
        vision_frames_per_range=2, use_gpu=False, quality="standard", output_dir="",
    )


@pytest.fixture
def synthetic_environment(tmp_path: Path):
    source_dir = tmp_path / "Synthetic Season" / "S03"
    _video(source_dir / "Episode 01.mp4", color="red", audio=True)
    _video(source_dir / "Episode 02.mp4", color="green", audio=False)
    _video(source_dir / "Episode 03.mp4", color="blue", audio=True)
    (source_dir / "Episode 01.en.srt").write_text(
        "1\n00:00:00,200 --> 00:00:01,400\nPrimary event and another person.\n", encoding="utf-8"
    )
    (source_dir / "Episode 03.en.srt").write_text(
        "1\n00:00:00,200 --> 00:00:01,400\nSecondary character returns.\n", encoding="utf-8"
    )
    (source_dir / "Episode 02.en.srt").write_text(
        "1\n00:00:00,200 --> 00:00:01,400\nAn inconclusive sound is heard.\n", encoding="utf-8"
    )
    return source_dir, ProjectPersistence(storage_root=tmp_path / "managed")


def _run(synthetic_environment, *, season: bool, visual: bool = True, invalid_writer: bool = False, output_dir: Path | None = None):
    source_dir, persistence = synthetic_environment
    project_id = "synthetic-season" if season else "synthetic-episode"
    gateway = SyntheticGateway(season=season, visual=visual, invalid_writer=invalid_writer)
    voice = SyntheticVoice(persistence, project_id)
    workflow = ProjectWorkflow(persistence=persistence, gateway_client=gateway, voice_adapter=voice)
    source_input = source_dir if season else source_dir / "Episode 01.mp4"
    workflow.create_project(project_id, "Synthetic Project", source_input, prompt=RAW_PROMPT,
                            settings=_settings(), output_dir=output_dir)
    state = workflow.start_project(project_id)
    return state, workflow, gateway, voice, persistence


def test_single_episode_full_synthetic_workflow(synthetic_environment):
    state, workflow, gateway, voice, persistence = _run(synthetic_environment, season=False)
    assert state["status"] == ProjectStatus.COMPLETED.value
    final_path = persistence._final_path("synthetic-episode")
    assert final_path.is_file()
    final = persistence.load_final_json("synthetic-episode")
    validate_project(final, {row["source_file"]: row["duration_ms"] for row in state["sources"]})
    final_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
    published = Path(state["output_dir"])
    assert published == synthetic_environment[0].parent / "Outputs_S03"
    assert {path.name for path in published.iterdir()} == {
        "Primary Story.mp4", "Primary Story.narration.srt", "Primary Story.original.srt",
    }
    output = published / "Primary Story.mp4"
    media = probe_media(output)
    assert media.duration > 0 and media.has_video and media.has_audio
    assert state["outputs"]["out_001"]["output_file_hash"] == compute_file_sha256(output)
    assert final_hash == hashlib.sha256(final_path.read_bytes()).hexdigest()
    assert voice.final_hash_at_first_call == final_hash
    counts = gateway.counts()
    assert counts["scanner"] == 1 and counts["season_planner"] == 2
    assert counts["visual_evidence"] == 1 and counts["final_planner_refinement"] == 1
    assert counts["output_writer"] == 1 and counts["writer_repair"] == 0
    assert voice.calls


def test_three_episode_synthetic_season_with_targeted_repair(synthetic_environment):
    state, workflow, gateway, voice, persistence = _run(
        synthetic_environment, season=True, invalid_writer=True,
    )
    assert state["status"] == ProjectStatus.COMPLETED.value
    final = persistence.load_final_json("synthetic-season")
    assert [item["render_id"] for item in final["outputs"]] == ["out_001", "out_002", "out_003"]
    assert [item["title"] for item in final["outputs"]] == [
        "Primary Story", "Cross Episode Story", "Secondary Story",
    ]
    assert {segment["source_file"] for segment in final["outputs"][1]["segments"]} == {
        "Episode 01.mp4", "Episode 03.mp4",
    }
    assert state["evidence"]["episode_counts"]["E02"] == 0
    assert state["catalog"]["episode_count"] == 3
    assert state["catalog"]["evidence_count"] == state["evidence"]["total_evidence_count"]
    published = Path(state["output_dir"])
    assert len(list(published.glob("*.mp4"))) == 3
    assert all(path.suffix.lower() in (".mp4", ".srt") for path in published.iterdir())
    for output in final["outputs"]:
        assert probe_media(published / f"{output['title']}.mp4").has_video
    # Fixed-speed narration is authoritative; short synthetic narration may yield
    # a shorter visual timeline while remaining a valid rendered output.
    assert probe_media(published / "Cross Episode Story.mp4").duration > 0.0
    counts = gateway.counts()
    assert counts["scanner"] == 3 and counts["season_planner"] == 2
    assert counts["visual_evidence"] == 1 and counts["final_planner_refinement"] == 1
    assert counts["output_writer"] == 3 and counts["writer_repair"] == 1
    assert state["finalization"]["repaired_output_ids"] == ["out_002"]
    writer_calls = [json.loads(call["prompt"]) for phase, call in gateway.calls if phase == "output_writer"]
    assert {call["output_id"] for call in writer_calls} == {"out_001", "out_002", "out_003"}
    for call in writer_calls:
        assert call["raw_recap_prompt"] == RAW_PROMPT
        assert [row["evidence_id"] for row in call["authoritative_full_evidence"]] == call["locked_plan_entry"]["evidence_ids"]
    vision_calls = [call for phase, call in gateway.calls if phase == "visual_evidence"]
    assert len(vision_calls) == 1 and len(vision_calls[0]["images"]) == 2
    assert json.loads(vision_calls[0]["prompt"])["visual_request"]["episode_id"] == "E03"
    assert voice.final_hash_at_first_call == hashlib.sha256(persistence._final_path("synthetic-season").read_bytes()).hexdigest()


def test_synthetic_media_variants_use_real_probe_and_source_preparation(tmp_path):
    variant_root = tmp_path / "Variant Clips"
    audio_only_dialogue = _video(variant_root / "No Subtitle Audio.mp4", color="yellow", audio=True)
    no_audio = _video(variant_root / "No Subtitle No Audio.mp4", color="purple", audio=False)
    stt_calls = []

    def fake_stt(**kwargs):
        stt_calls.append(kwargs)
        return SttResult(
            status=SttStatus.SUCCESS,
            cues=(SubtitleCue(start_ms=200, end_ms=1400, text="Synthetic spoken event", source_type="stt", source_format="whisper"),),
            language="eng", model_name="fake-model-boundary",
        )

    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(cache_dir=tmp_path / "cache"),
        transcribe_fn=fake_stt,
        allow_stt=True,
    )
    prepared_audio = pipeline.prepare_episode(audio_only_dialogue, episode_id="E01", source_id="E01")
    assert prepared_audio.transcript_method == "stt"
    assert prepared_audio.transcript.cue_count == 1
    assert len(stt_calls) == 1
    assert stt_calls[0]["selection"].has_audio
    prepared_silent = pipeline.prepare_episode(no_audio, episode_id="E02", source_id="E02")
    assert prepared_silent.status == "no_audio"
    assert prepared_silent.transcript.cue_count == 0
    assert len(stt_calls) == 1


def test_final_json_is_deterministic_across_clean_managed_stores(synthetic_environment, tmp_path):
    source_dir, first_store = synthetic_environment
    first, _workflow, _gateway, _voice, _persistence = _run(synthetic_environment, season=False)
    second_store = ProjectPersistence(storage_root=tmp_path / "m2")
    second, _workflow2, _gateway2, _voice2, _persistence2 = _run(
        (source_dir, second_store), season=False, output_dir=tmp_path / "Second Publication",
    )
    assert first["status"] == second["status"] == ProjectStatus.COMPLETED.value
    assert first_store._final_path("synthetic-episode").read_bytes() == second_store._final_path("synthetic-episode").read_bytes()
