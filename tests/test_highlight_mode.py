from pathlib import Path
import json

import pytest

from toolrecap_v4.gateway import GatewayResult
from toolrecap_v4.highlight.models import build_highlight_project
from toolrecap_v4.highlight.renderer import render_highlight
from toolrecap_v4.highlight.service import HighlightPlanner
from toolrecap_v4.highlight.subtitles import highlight_srt
from toolrecap_v4.media import (
    AudioStreamInfo, CommandResult, find_binary, probe_media, run_command,
    select_audio_for_source,
)
from toolrecap_v4.media import EncoderStatus
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.analysis.models import AudioSelection, PreparedEpisode, Transcript, TranscriptCue
from toolrecap_v4.discovery import compute_file_fingerprint
from toolrecap_v4.highlight.renderer import HighlightRenderResult
from toolrecap_v4.renderer import RenderResult
from toolrecap_v4.highlight.service import HighlightWorkflow
from toolrecap_v4.persistence import ProjectPersistence


SOURCES = [
    {"episode_id": "E01", "source_id": "src_001", "source_file": "e01.mp4", "duration_ms": 60_000},
    {"episode_id": "E02", "source_id": "src_002", "source_file": "e02.mp4", "duration_ms": 70_000},
]
EVIDENCE = {s["episode_id"]: [{"evidence_id": s["episode_id"] + "-EV-001",
    "episode_id": s["episode_id"], "source_id": s["source_id"], "start_ms": 0,
    "end_ms": s["duration_ms"], "observation": "A supported conflict"}] for s in SOURCES}


def candidate(**changes):
    value = {
        "title": "Beth challenges the board", "episode_id": "E01", "source_id": "src_001",
        "source_file": "e01.mp4", "start_ms": 10_000, "end_ms": 20_000,
        "evidence_ids": ["E01-EV-001"], "subtitle_cues": [
            {"start_ms": 12_250, "end_ms": 13_500, "text": "You heard me.", "verified": True}
        ],
    }
    value.update(changes)
    return value


def ai_candidate(**changes):
    scene = candidate(**{key: value for key, value in changes.items() if key not in {"storyline", "clips"}})
    clip = {
        key: scene[key]
        for key in ("episode_id", "source_id", "source_file", "start_ms", "end_ms", "evidence_ids")
    }
    value = {
        "title": scene["title"],
        "storyline": changes.get("storyline", scene["title"]),
        "clips": changes.get("clips", [clip]),
    }
    return value


def test_contract_app_ids_relative_srt_visual_only_and_no_recap_fields():
    project = build_highlight_project(
        project_id="highlight-season", prompt="exhaustive", dependency_revision="rev",
        candidates=[candidate(), candidate(title="A quiet decision", start_ms=30_000, end_ms=35_000, subtitle_cues=[])],
        sources=SOURCES,
    )
    assert project.project_mode == "HIGHLIGHT"
    assert [item.output_id for item in project.outputs] == ["hl_001", "hl_002"]
    assert "00:00:02,250 --> 00:00:03,500" in highlight_srt(project.outputs[0])
    assert highlight_srt(project.outputs[1]) == ""
    encoded = json.dumps(project.to_dict()).lower()
    assert all(word not in encoded for word in ("narration", "voicestudio", "commentary_reading_speed", "narration_fit"))


@pytest.mark.parametrize("changes", [
    {"start_ms": -1}, {"end_ms": 10_000}, {"end_ms": 60_001},
    {"source_id": "wrong"}, {"episode_id": "E99"},
])
def test_strict_timestamp_and_identity_validation(changes):
    with pytest.raises(ValueError):
        build_highlight_project(
            project_id="p", prompt="x", dependency_revision="r",
            candidates=[candidate(**changes)], sources=SOURCES,
        )


def test_no_cross_episode_montage_and_no_generic_title():
    with pytest.raises(ValueError):
        build_highlight_project(project_id="p", prompt="x", dependency_revision="r", candidates=[candidate(title="Highlight 1")], sources=SOURCES)
    with pytest.raises(ValueError):
        build_highlight_project(
            project_id="p", prompt="x", dependency_revision="r",
            candidates=[candidate(episode_id="E02", source_id="src_001")], sources=SOURCES,
        )


def test_duplicate_publication_titles_are_rejected_case_insensitively():
    with pytest.raises(ValueError, match="publication title"):
        build_highlight_project(
            project_id="p", prompt="x", dependency_revision="r",
            candidates=[
                candidate(title="A Secret Decision", start_ms=1_000, end_ms=2_000, subtitle_cues=[]),
                candidate(title="a secret decision", start_ms=3_000, end_ms=4_000, subtitle_cues=[]),
            ],
            sources=SOURCES,
        )


class FakeGateway:
    def __init__(self): self.calls = []
    def submit_text_chat(self, **kwargs):
        self.calls.append(kwargs)
        phase = kwargs["phase"]
        if phase == "highlight_episode":
            payload = json.loads(kwargs["prompt"]); source = payload["source"]
            rows = [{
                "title": f"A worthwhile {source['episode_id']} guest moment",
                "storyline": f"A self-contained guest story in {source['episode_id']}",
                "clips": [{
                "episode_id": source["episode_id"], "source_id": source["source_id"],
                "source_file": source["source_file"], "start_ms": 1000, "end_ms": 2000,
                "evidence_ids": [source["episode_id"] + "-EV-001"],
                }],
            }]
        else:
            payload = json.loads(kwargs["prompt"]); rows = payload["episode_candidates"]
        return GatewayResult(json.dumps({"candidates": rows}))


def test_episode_then_season_coverage_has_no_quota_and_secondary_characters():
    gateway = FakeGateway(); planner = HighlightPlanner(gateway, "model")
    rows = planner.plan(editorial_prompt="include secondary and guest characters", sources=SOURCES, evidence_by_episode=EVIDENCE)
    assert len(rows) == 2
    assert [call["phase"] for call in gateway.calls] == ["highlight_episode", "highlight_episode", "highlight_season_coverage"]
    assert all("guest moment" in row["title"] for row in rows)
    season_payload = json.loads(gateway.calls[-1]["prompt"])
    assert season_payload["pass"] == "SEASON_STORYLINE_SYNTHESIS"
    assert "Never combine all season highlights" in season_payload["instruction"]


def test_season_context_groups_only_linked_scenes_into_one_story_output():
    class CrossEpisodeGateway:
        def submit_text_chat(self, **kwargs):
            phase = kwargs["phase"]
            payload = json.loads(kwargs["prompt"])
            if phase == "highlight_episode":
                return GatewayResult(json.dumps({"candidates": []}))
            assert payload["pass"] == "SEASON_STORYLINE_SYNTHESIS"
            return GatewayResult(json.dumps({"candidates": [
                {
                    "title": "The warning and its consequences",
                    "storyline": "A warning in E01 causes the confrontation in E02.",
                    "clips": [
                        {"episode_id": "E01", "source_id": "src_001", "source_file": "e01.mp4",
                         "start_ms": 50_000, "end_ms": 55_000, "evidence_ids": ["E01-EV-001"]},
                        {"episode_id": "E02", "source_id": "src_002", "source_file": "e02.mp4",
                         "start_ms": 2_000, "end_ms": 8_000, "evidence_ids": ["E02-EV-001"]},
                    ],
                },
            ]}))

    rows = HighlightPlanner(CrossEpisodeGateway(), "model").plan(
        editorial_prompt="Find connected cause and consequence scenes",
        sources=SOURCES,
        evidence_by_episode=EVIDENCE,
    )
    assert len(rows) == 1
    assert [(clip["episode_id"], clip["source_file"]) for clip in rows[0]["clips"]] == [
        ("E01", "e01.mp4"), ("E02", "e02.mp4"),
    ]


def test_highlight_contract_mismatch_gets_one_bounded_repair():
    class RepairGateway:
        def __init__(self): self.calls = []
        def submit_text_chat(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["phase"] == "highlight_episode":
                return GatewayResult(json.dumps({"outputs": []}))
            return GatewayResult(json.dumps({"candidates": [{
                "title": "A repaired guest moment", "storyline": "A repaired scene",
                "clips": [{"episode_id": "E01", "source_id": "src_001",
                    "source_file": "e01.mp4", "start_ms": 1000, "end_ms": 2000,
                    "evidence_ids": ["E01-EV-001"]}],
            }]}))

    gateway = RepairGateway()
    rows = HighlightPlanner(gateway, "model").plan(
        editorial_prompt="find scenes", sources=[SOURCES[0]], evidence_by_episode=EVIDENCE,
    )
    assert len(rows) == 1
    assert [call["phase"] for call in gateway.calls] == ["highlight_episode", "highlight_episode_repair"]


def test_direct_renderer_maps_only_selected_english_audio_and_publishes_mp4_srt(tmp_path: Path):
    source = tmp_path / "e01.mp4"; source.write_bytes(b"source")
    project = build_highlight_project(project_id="p", prompt="x", dependency_revision="r", candidates=[candidate()], sources=SOURCES)
    captured = {}
    def shared_renderer(**kwargs):
        captured.update(kwargs)
        folder = Path(kwargs["output_dir"]); folder.mkdir(parents=True, exist_ok=True)
        output_def = kwargs["output_def"]; title = output_def["title"]
        video = folder / f"{title}.mp4"; narration = folder / f"{title}.narration.srt"; original = folder / f"{title}.original.srt"
        video.write_bytes(b"rendered-original-audio"); narration.write_text(""); original.write_text("verified dialogue")
        return RenderResult(video, narration, original, 10.0, title, output_def["render_id"], "h264", "aac", 1920, 1080, 30.0, "ORIGINAL_DIALOGUE_MAPPED", 1)
    result = render_highlight(
        project.outputs[0], sources={"e01.mp4": source},
        publication_dir=tmp_path / "publication", settings=AppSettings(use_gpu=False),
        render_output_fn=shared_renderer,
    )
    assert result.video_path.is_file() and result.subtitle_path.is_file()
    assert captured["voice_adapter"] is None
    assert captured["output_def"]["segments"][0]["type"] == "original_dialogue"
    assert captured["output_def"]["segments"][0]["source_audio"] is True
    assert sorted(p.suffix for p in result.video_path.parent.iterdir()) == [".mp4", ".srt"]


def test_highlight_renderer_refuses_unowned_existing_publication(tmp_path: Path):
    source = tmp_path / "e01.mp4"; source.write_bytes(b"source")
    ffmpeg = tmp_path / "ffmpeg.exe"; ffmpeg.write_bytes(b"binary")
    publication = tmp_path / "publication"; publication.mkdir()
    existing = publication / "Beth challenges the board.mp4"
    existing.write_bytes(b"user-owned")
    project = build_highlight_project(
        project_id="p", prompt="x", dependency_revision="r", candidates=[candidate()], sources=SOURCES,
    )

    with pytest.raises(FileExistsError, match="matching checkpoint"):
        render_highlight(
            project.outputs[0], sources={"e01.mp4": source}, publication_dir=publication,
            ffmpeg_path=ffmpeg, settings=AppSettings(use_gpu=False),
            command_runner=lambda *_args, **_kwargs: CommandResult(0, "", ""),
        )
    assert existing.read_bytes() == b"user-owned"


def test_real_highlight_output_contains_only_english_from_multilingual_source(tmp_path: Path):
    ffmpeg = find_binary("ffmpeg")
    source = tmp_path / "multilingual.mp4"
    create = [
        str(ffmpeg), "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=25:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-metadata:s:a:0", "language=ukr", "-metadata:s:a:0", "title=Ukrainian Dub",
        "-metadata:s:a:1", "language=eng", "-metadata:s:a:1", "title=English Original",
        "-disposition:a:0", "default", "-disposition:a:1", "0", str(source),
    ]
    assert run_command(create, timeout=30).exit_code == 0
    source_probe = probe_media(source)
    selection = select_audio_for_source(source_probe)
    assert selection.selected_index == 2

    project = build_highlight_project(
        project_id="p", prompt="x", dependency_revision="r",
        candidates=[candidate(start_ms=0, end_ms=800, subtitle_cues=[])], sources=SOURCES,
    )
    result = render_highlight(
        project.outputs[0], sources={"e01.mp4": source}, publication_dir=tmp_path / "out",
        settings=AppSettings(use_gpu=False), source_audio_index=selection.selected_index,
        source_audio_language=selection.selected_stream.language,
    )

    output_probe = probe_media(result.video_path)
    assert len(output_probe.audio_streams) == 1
    assert output_probe.audio_streams[0].language == "eng"


def test_highlight_prompt_is_separate_and_speed_is_irrelevant():
    settings = AppSettings(prompt="recap", highlight_prompt="highlight", commentary_reading_speed=1.30)
    assert settings.prompt == "recap" and settings.highlight_prompt == "highlight"
    project = build_highlight_project(project_id="p", prompt=settings.highlight_prompt, dependency_revision="r", candidates=[candidate()], sources=SOURCES)
    assert "commentary_reading_speed" not in json.dumps(project.to_dict())


def test_multi_episode_highlight_delegates_chronological_clips_to_normalized_renderer(tmp_path: Path):
    source1 = tmp_path / "e01.mp4"; source1.write_bytes(b"source-1")
    source2 = tmp_path / "e02.mp4"; source2.write_bytes(b"source-2")
    project = build_highlight_project(
        project_id="p", prompt="x", dependency_revision="r", candidates=[ai_candidate(
            title="The warning and its consequences", storyline="One cause-and-consequence story",
            clips=[
                {"episode_id":"E01","source_id":"src_001","source_file":"e01.mp4","start_ms":50_000,"end_ms":55_000,"evidence_ids":[]},
                {"episode_id":"E02","source_id":"src_002","source_file":"e02.mp4","start_ms":2_000,"end_ms":8_000,"evidence_ids":[]},
            ],
        )], sources=SOURCES,
    )
    captured = {}
    def shared_renderer(**kwargs):
        captured.update(kwargs); folder=Path(kwargs["output_dir"]); folder.mkdir(parents=True,exist_ok=True)
        title=kwargs["output_def"]["title"]; video=folder/f"{title}.mp4"; narr=folder/f"{title}.narration.srt"; orig=folder/f"{title}.original.srt"
        video.write_bytes(b"video");narr.write_text("");orig.write_text("")
        return RenderResult(video,narr,orig,11.0,title,"hl_001","h264","aac",1920,1080,30.0)
    render_highlight(
        project.outputs[0], sources={"e01.mp4":source1,"e02.mp4":source2}, publication_dir=tmp_path/"out",
        settings=AppSettings(use_gpu=False), render_output_fn=shared_renderer,
    )
    segments=captured["output_def"]["segments"]
    assert [(s["source_file"],s["start_ms"],s["end_ms"]) for s in segments] == [
        ("e01.mp4",50_000,55_000),("e02.mp4",2_000,8_000),
    ]
    assert all(s["type"]=="original_dialogue" and s["source_audio"] for s in segments)


def test_highlight_resume_reuses_plan_and_completed_publication(tmp_path: Path, monkeypatch):
    source = tmp_path / "e01.mp4"; source.write_bytes(b"source-video")
    fingerprint = compute_file_fingerprint(source)
    persistence = ProjectPersistence(tmp_path)
    settings = AppSettings(scanner_model="scanner", finalizer_model="finalizer", highlight_prompt="find scenes")
    state = {
        "project_id": "highlight-project", "project_name": "Season", "project_mode": "HIGHLIGHT",
        "status": "created", "prompt": "find scenes", "output_dir": str(tmp_path / "publication"),
        "sources": [{"episode_id": "E01", "source_id": "src_001", "source_file": source.name,
                     "duration_ms": 60_000, "fingerprint": fingerprint.to_dict()}],
        "source_fingerprints": {source.name: fingerprint.to_dict()}, "settings_snapshot": settings.to_dict(),
        "outputs": {}, "timestamps": {},
    }
    persistence.save_project(state)
    english_audio = AudioStreamInfo(
        2, "aac", 2, 48000, language="eng", title="English Original",
        disposition={"default": 0},
    )
    prepared = PreparedEpisode(
        "E01", "src_001", source, 60_000, 1920, 1080, source_fingerprint=fingerprint.sha256,
        audio_streams=(
            AudioStreamInfo(1, "aac", 2, 48000, language="ukr", title="Ukrainian Dub", disposition={"default": 1}),
            english_audio,
        ),
        audio_selection=AudioSelection(english_audio, global_index=2, audio_ordinal=1),
        selected_audio=english_audio, audio_ordinal=1,
        transcript=Transcript(episode_id="E01", source_type="sidecar", source_format="srt",
                              cues=(TranscriptCue("c1", 12_250, 13_500, "You heard me."),)),
    )
    persistence.save_prepared_episode("highlight-project", "E01", prepared.to_dict())

    class Scanner:
        def __init__(self, *args, **kwargs): pass
        def scan_project(self, *args, **kwargs): return type("Scan", (), {"evidence_revision": "ev-rev"})()
    class Store:
        def __init__(self, *args, **kwargs): pass
        def get_episode(self, *args, **kwargs): return type("Result", (), {"items": ()})()
    class Planner:
        def __init__(self): self.calls = 0
        def plan(self, **kwargs):
            self.calls += 1
            return [candidate(subtitle_cues=[])]
    renders = []
    def fake_render(output, **kwargs):
        folder = Path(kwargs["publication_dir"]); folder.mkdir(parents=True, exist_ok=True)
        video, subtitle = folder / f"{output.title}.mp4", folder / f"{output.title}.srt"
        video.write_bytes(b"video"); subtitle.write_bytes(b"srt")
        import hashlib
        return HighlightRenderResult(output.output_id, video, subtitle, hashlib.sha256(b"video").hexdigest(), hashlib.sha256(b"srt").hexdigest())
    monkeypatch.setattr("toolrecap_v4.highlight.service.ScannerService", Scanner)
    monkeypatch.setattr("toolrecap_v4.highlight.service.EvidenceStore", Store)
    monkeypatch.setattr("toolrecap_v4.highlight.service.render_highlight", lambda *a, **k: (renders.append(k) or fake_render(*a, **k)))
    planner = Planner()
    from types import SimpleNamespace
    workflow = HighlightWorkflow(persistence=persistence, gateway_client=FakeGateway(), planner=planner,
                                 source_pipeline=SimpleNamespace(prepare_episode=lambda *a, **k: prepared))
    first = workflow.run("highlight-project", settings=settings)
    second = workflow.run("highlight-project", settings=settings)
    assert first["project_mode"] == second["project_mode"] == "HIGHLIGHT"
    assert planner.calls == 1 and len(renders) == 1
    assert second["outputs"]["hl_001"]["audio_selection_version"] == "highlight-audio-selection-v1"
    assert second["outputs"]["hl_001"]["source_dependencies"]["e01.mp4"]["audio_index"] == 2
    assert second["outputs"]["hl_001"]["source_dependencies"]["e01.mp4"]["audio_language"] == "eng"
