import json
import os
from pathlib import Path
import wave
from types import SimpleNamespace

import pytest

from toolrecap_v4.analysis.finalizer.finalization import map_final_json, ValidationResult
from toolrecap_v4.narration_groups import slice_group_audio
from toolrecap_v4.subtitles import extract_subtitles_for_output
from toolrecap_v4.errors import NarrationFitError
from toolrecap_v4.highlight.service import HighlightPlanner
from toolrecap_v4.highlight.contracts import validate_candidates
from toolrecap_v4.gateway import GatewayResult
from toolrecap_v4.analysis.vision.models import FrameArtifact
from toolrecap_v4.highlight.visual import HighlightVisualSampler
from test_highlight_mode import SOURCES, EVIDENCE, ai_candidate
from test_phase14_workflow_e2e import SyntheticGateway, SyntheticVoice, _settings, _video, _run, RAW_PROMPT, synthetic_environment
from toolrecap_v4.workflow import ProjectWorkflow, ProjectStatus
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.review import plan_digest, parse_timecode, review_text, save_reviewed_plan
from toolrecap_v4.publication import digest, publish, recover, transaction_dir
from toolrecap_v4.persistence import atomic_write_json
from toolrecap_v4.highlight.models import build_highlight_project
from toolrecap_v4.highlight.renderer import render_highlight
from toolrecap_v4.media import probe_media
from toolrecap_v4.preview import boundary_preview


def grouped_output():
    episodes = [SimpleNamespace(episode_id="E01", source_basename="episode.mp4", duration_ms=5000)]
    value = {"title": "A coherent story", "segments": [{"segment_id": "s1", "narration_text": "One speech across two scenes.",
        "source_clips": [{"episode_id": "E01", "start_ms": 0, "end_ms": 1000},
                         {"episode_id": "E01", "start_ms": 2000, "end_ms": 3000}]}]}
    return map_final_json(project_id="p", project_name="Project", episodes=episodes,
        plan=SimpleNamespace(outputs=[{"output_id": "out_001"}]),
        validated=[ValidationResult("out_001", "VALID", (), value, "hash")])["outputs"][0]


def test_narration_is_once_and_subtitle_spans_both_clips():
    output = grouped_output()
    assert [s["narration"] for s in output["segments"]] == ["One speech across two scenes.", ""]
    cues, _ = extract_subtitles_for_output(output)
    assert len(cues) == 1 and cues[0].end_ms == 2000


def test_group_audio_is_losslessly_partitioned_and_never_extended(tmp_path):
    source = tmp_path / "speech.wav"
    samples = bytes(range(256)) * 125  # 2 seconds of mono 16-bit at 8kHz
    with wave.open(str(source), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000); f.writeframes(samples)
    members = grouped_output()["segments"]
    parts, duration = slice_group_audio(source, members, tmp_path)
    reconstructed = b""
    for member in members:
        with wave.open(str(parts[member["segment_id"]]), "rb") as f:
            reconstructed += f.readframes(f.getnframes())
    assert reconstructed == samples and duration == 2000
    members[-1]["end_ms"] = 2500
    with pytest.raises(NarrationFitError):
        slice_group_audio(source, members, tmp_path)


def test_highlight_one_episode_can_return_multiple_related_clips_and_resume(tmp_path):
    calls = []
    candidate = ai_candidate(clips=[
        {"episode_id":"E01","source_id":"src_001","source_file":"e01.mp4","start_ms":1000,"end_ms":2000,"evidence_ids":["E01-EV-001"]},
        {"episode_id":"E01","source_id":"src_001","source_file":"e01.mp4","start_ms":3000,"end_ms":4000,"evidence_ids":["E01-EV-001"]}])
    class Gateway:
        def submit_text_chat(self, **kwargs):
            calls.append(kwargs)
            return GatewayResult(json.dumps({"candidates": [candidate]}))
    for _ in range(2):
        result = HighlightPlanner(Gateway(), "model", checkpoint_dir=tmp_path).plan(
            editorial_prompt="one story", sources=SOURCES[:1], evidence_by_episode=EVIDENCE)
        assert len(result[0]["clips"]) == 2
    assert len(calls) == 1


def test_large_season_uses_rolling_context_without_dropping_episode_evidence():
    evidence = {key: [{**row, "observation": "Observed dialogue. " * 65} for row in rows]
                for key, rows in EVIDENCE.items()}
    calls = []
    class Gateway:
        def submit_text_chat(self, **kwargs):
            calls.append(kwargs)
            payload = json.loads(kwargs["prompt"])
            if kwargs["phase"] == "highlight_episode":
                source = payload["source"]
                return GatewayResult(json.dumps({"candidates": [ai_candidate(
                    title=f"Story in {source['episode_id']}",
                    episode_id=source["episode_id"], source_id=source["source_id"],
                    source_file=source["source_file"], start_ms=1000, end_ms=2000,
                    evidence_ids=[source["episode_id"] + "-EV-001"])]}))
            assert kwargs["phase"] == "highlight_season_rolling"
            return GatewayResult(json.dumps({"candidates": payload["current_storylines"] + payload["episode_candidates"]}))
    outputs = HighlightPlanner(Gateway(), "model", max_request_bytes=5000).plan(
        editorial_prompt="Find stories", sources=SOURCES, evidence_by_episode=evidence)
    assert len(outputs) == 2
    assert [call["phase"] for call in calls] == [
        "highlight_episode", "highlight_episode", "highlight_season_rolling", "highlight_season_rolling",
    ]
    rolling = [json.loads(call["prompt"]) for call in calls if call["phase"] == "highlight_season_rolling"]
    assert len(rolling[0]["factual_evidence"]) == len(evidence["E01"])
    assert len(rolling[1]["factual_evidence"]) == len(evidence["E02"])


def test_unknown_or_cross_episode_evidence_is_rejected():
    candidate = ai_candidate(evidence_ids=["E02-EV-001"])
    with pytest.raises(ValueError, match="another episode"):
        validate_candidates([candidate], {"sources": SOURCES, "season_evidence": EVIDENCE})


def test_sparse_visual_observations_are_source_bound_and_cached(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-frame")
    calls = []
    class Extractor:
        def extract(self, request, offset, count, token):
            return (FrameArtifact(f"E01-FR-{offset+1:04d}", "E01", request.visual_request_id,
                                  request.start_ms, frame, "framehash", "sourcehash", "test"),)
    class Gateway:
        def submit_image_chat(self, **kwargs):
            calls.append(kwargs)
            row = json.loads(kwargs["prompt"])["frames"][0]
            return GatewayResult(json.dumps({"observations": [{**row, "observation": "A visible action"}]}))
    episode = SimpleNamespace(episode_id="E01", source_id="src_001", source_path=tmp_path/"episode.mp4",
                              source_fingerprint="sourcehash", duration_ms=40_000)
    sampler = HighlightVisualSampler(Gateway(), tmp_path, "vision-model", extractor=Extractor())
    first = sampler.run("project", [episode])
    second = sampler.run("project", [episode])
    assert first == second and len(first["E01"]) == 4 and len(calls) == 4
    assert all(item["provenance"]["coverage"].startswith("sparse sample") for item in first["E01"])


def test_highlight_visual_accepts_json_code_fence_and_caches_raw(tmp_path):
    frame = tmp_path / "frame.jpg"; frame.write_bytes(b"frame")
    calls = []
    class Extractor:
        def extract(self, request, offset, count, token):
            return (FrameArtifact(f"E01-FR-{offset+1:04d}", "E01", request.visual_request_id,
                                  request.start_ms, frame, "hash", "source", "test"),)
    class Gateway:
        def submit_image_chat(self, **kwargs):
            calls.append(kwargs)
            item = json.loads(kwargs["prompt"])["frames"][0]
            return GatewayResult("```json\n" + json.dumps({"observations": [{**item, "observation": "Visible fact"}]}) + "\n```")
    episode = SimpleNamespace(episode_id="E01", source_id="src", source_path=tmp_path/"e.mkv",
                              source_fingerprint="source", duration_ms=4000)
    sampler = HighlightVisualSampler(Gateway(), tmp_path, "vision", extractor=Extractor())
    assert len(sampler.run("p", [episode])["E01"]) == 4
    assert len(calls) == 4
    assert list((tmp_path/"projects/p/highlight-visual").rglob("attempt-00.raw.txt"))


def test_highlight_visual_repairs_malformed_response_and_resume_spends_no_ai(tmp_path):
    frame = tmp_path / "frame.jpg"; frame.write_bytes(b"frame")
    calls = []
    class Extractor:
        def extract(self, request, offset, count, token):
            return (FrameArtifact(f"E01-FR-{offset+1:04d}", "E01", request.visual_request_id,
                                  request.start_ms, frame, "hash", "source", "test"),)
    class Gateway:
        def submit_image_chat(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return GatewayResult("I cannot return the requested JSON.")
            request = json.loads(kwargs["prompt"])
            frames = request.get("frames") or request["original_request"]["frames"]
            item = frames[0]
            return GatewayResult(json.dumps({"observations": [{**item, "observation": "Visible fact"}]}))
    episode = SimpleNamespace(episode_id="E01", source_id="src", source_path=tmp_path/"e.mkv",
                              source_fingerprint="source", duration_ms=4000)
    gateway = Gateway()
    sampler = HighlightVisualSampler(gateway, tmp_path, "vision", extractor=Extractor(), repair_attempts=1)
    first = sampler.run("p", [episode])
    paid = len(calls)
    second = sampler.run("p", [episode])
    assert first == second and paid == 5 and len(calls) == paid
    result = next((tmp_path/"projects/p/highlight-visual").rglob("attempt-00.result.json"))
    assert json.loads(result.read_text(encoding="utf-8"))["state"] == "INVALID_REPAIRABLE"
    assert next((tmp_path/"projects/p/highlight-visual").rglob("attempt-01.result.json"))


def test_recap_review_pauses_before_render_and_approved_resume_uses_no_ai(synthetic_environment):
    source_dir, persistence = synthetic_environment
    project_id = "review-episode"
    settings = _settings()
    settings.review_before_render = True
    gateway = SyntheticGateway(season=False)
    workflow = ProjectWorkflow(persistence=persistence, gateway_client=gateway,
                               voice_adapter=SyntheticVoice(persistence, project_id))
    workflow.create_project(project_id, "Review Project", source_dir/"Episode 01.mp4",
                            prompt=RAW_PROMPT, settings=settings)
    pending = workflow.start_project(project_id)
    assert pending["status"] == ProjectStatus.REVIEW_REQUIRED.value
    assert "Selected footage" in review_text(pending)
    assert not list(Path(pending["output_dir"]).glob("*.mp4")) if Path(pending["output_dir"]).exists() else True
    calls = len(gateway.calls)
    pending["review_approved_hash"] = plan_digest(pending)
    persistence.save_project(pending)
    finished = workflow.resume_project(project_id, settings=settings)
    assert finished["status"] == ProjectStatus.COMPLETED.value
    assert len(gateway.calls) == calls


def test_creator_can_edit_recap_title_before_approval(synthetic_environment):
    source_dir, persistence = synthetic_environment
    project_id = "review-title"
    settings = _settings(); settings.review_before_render = True
    gateway = SyntheticGateway(season=False)
    workflow = ProjectWorkflow(persistence=persistence, gateway_client=gateway,
                               voice_adapter=SyntheticVoice(persistence, project_id))
    workflow.create_project(project_id, "Review", source_dir/"Episode 01.mp4",
                            prompt=RAW_PROMPT, settings=settings)
    pending = workflow.start_project(project_id)
    edited = json.loads(json.dumps(pending["final_json"]))
    edited["outputs"][0]["title"] = "A New Publication Title"
    saved = save_reviewed_plan(persistence, project_id, edited)
    assert saved["review_plan_hash"] == plan_digest(saved)
    assert persistence.load_final_json(project_id)["outputs"][0]["title"] == "A New Publication Title"
    assert parse_timecode("00:14:53.950") == 893950


def test_publication_crash_recovery_finishes_verified_pair_without_clobbering_user_file(tmp_path, monkeypatch):
    published = tmp_path / "published"; published.mkdir()
    journal_root = tmp_path / "state"
    targets = [published / "story.mp4", published / "story.srt"]
    sources = [tmp_path / "new.mp4", tmp_path / "new.srt"]
    for source, data in zip(sources, (b"new video", b"new subtitles")):
        source.write_bytes(data)
    owner = "project:story:revision"
    directory = transaction_dir(targets, owner, journal_root)
    directory.mkdir(parents=True)
    for index, source in enumerate(sources):
        (directory / f"new-{index}").write_bytes(source.read_bytes())
    (directory / "commit-0").write_bytes(sources[0].read_bytes())  # <= 1.0.11 cross-drive staging
    atomic_write_json(directory / "manifest.json", {
        "owner": owner, "status": "PREPARED", "targets": [str(p.resolve()) for p in targets],
        "files": [{"old_hash": None, "new_hash": digest(source)} for source in sources],
    })
    targets[0].write_bytes(sources[0].read_bytes())  # crash after first replacement
    real_replace = os.replace
    def require_destination_volume(source, destination):
        source_path, destination_path = Path(source), Path(destination)
        if destination_path in targets:
            assert source_path.parent == destination_path.parent
        return real_replace(source, destination)
    monkeypatch.setattr("toolrecap_v4.publication.os.replace", require_destination_volume)
    assert recover(targets, owner, journal_root)
    assert [p.read_bytes() for p in targets] == [b"new video", b"new subtitles"]
    assert not (directory / "commit-0").exists()
    assert recover(targets, owner, journal_root)

    another_owner = "another:project:revision"
    with pytest.raises(FileExistsError):
        publish(list(zip(sources, targets)), another_owner, journal_root=journal_root)
    assert [p.read_bytes() for p in targets] == [b"new video", b"new subtitles"]


def test_two_episode_highlight_renders_one_original_audio_video_with_remapped_srt(tmp_path):
    first = _video(tmp_path/"E01.mp4", color="red", audio=True)
    second = _video(tmp_path/"E02.mp4", color="blue", audio=True)
    sources = [
        {"episode_id": "E01", "source_id": "src_001", "source_file": first.name, "duration_ms": 2000},
        {"episode_id": "E02", "source_id": "src_002", "source_file": second.name, "duration_ms": 2000},
    ]
    clips = [
        {"episode_id": episode, "source_id": src, "source_file": filename,
         "start_ms": 0, "end_ms": 1000, "evidence_ids": [],
         "subtitle_cues": [{"start_ms": 200, "end_ms": 500, "text": text, "verified": True}]}
        for episode, src, filename, text in
        (("E01", "src_001", first.name, "Before."), ("E02", "src_002", second.name, "After."))
    ]
    project = build_highlight_project(project_id="p", prompt="story", dependency_revision="revision",
                                      candidates=[{"title": "The decision and its result", "storyline": "One cause and payoff",
                                                   "clips": clips}], sources=sources)
    indices = {path.name: probe_media(path).audio_streams[0].index for path in (first, second)}
    output = render_highlight(project.outputs[0], sources={p.name: p for p in (first, second)},
                              publication_dir=tmp_path/"out", audio_stream_indices=indices,
                              settings=_settings())
    media = probe_media(output.video_path)
    subtitles = output.subtitle_path.read_text(encoding="utf-8")
    assert media.has_audio and media.has_video and 1.8 < media.duration < 2.2
    assert "00:00:00,200" in subtitles and "00:00:01,200" in subtitles
    assert "Before." in subtitles and "After." in subtitles


def test_source_boundary_preview_is_short_and_reused(tmp_path):
    source = _video(tmp_path/"episode.mp4", color="blue", audio=True)
    first = boundary_preview(source, 0, 1800, at_end=False, storage_root=tmp_path/"state")
    last = boundary_preview(source, 0, 1800, at_end=True, storage_root=tmp_path/"state")
    assert first.is_file() and last.is_file()
    assert 1.6 < probe_media(first).duration < 2.0
    assert boundary_preview(source, 0, 1800, at_end=True, storage_root=tmp_path/"state") == last


def test_resume_recovers_published_video_after_checkpoint_interruption(synthetic_environment, monkeypatch):
    state, workflow, gateway, voice, persistence = _run(synthetic_environment, season=False)
    project_id = state["project_id"]
    output_id = next(iter(state["outputs"]))
    checkpoint = persistence.load_checkpoint(project_id, output_id)
    checkpoint["status"] = "rendering"
    persistence.save_checkpoint(project_id, output_id, checkpoint)
    state["outputs"][output_id] = checkpoint
    state["status"] = "rendering"
    persistence.save_project(state)
    ai_calls, voice_calls = len(gateway.calls), len(voice.calls)
    monkeypatch.setattr("toolrecap_v4.workflow.render_output", lambda **_kwargs:
                        pytest.fail("Completed publication should be recovered without rendering"))
    resumed = workflow.resume_project(project_id)
    assert resumed["status"] == "completed"
    assert len(gateway.calls) == ai_calls and len(voice.calls) == voice_calls
