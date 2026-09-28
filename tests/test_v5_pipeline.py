from pathlib import Path
from types import SimpleNamespace

import pytest

from toolrecap_v4.analysis.models import PreparedEpisode, Transcript, TranscriptCue
from toolrecap_v4.gateway import GatewayResult
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.output_paths import resolve_v5_publication_root
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.v5.analyzer import GatewayRenderJsonAnalyzer, select_analysis_model
from toolrecap_v4.v5.workflow import V5Workflow, validate_mode_render_contract


def render_json(source_file: str, *, highlight: bool = False) -> dict:
    return {
        "schema_version": "3.0",
        "project_id": "project-1",
        "project_name": "Project One",
        "sources": [{"source_file": source_file, "duration_ms": 10_000}],
        "outputs": [{
            "render_id": "out_001",
            "title": "Output One",
            "segments": [{
                "segment_id": "seg_001",
                "source_file": source_file,
                "start_ms": 1_000,
                "end_ms": 5_000,
                "type": "original_dialogue" if highlight else "narration",
                "narration": "" if highlight else "A concise factual recap.",
                "source_audio": True,
                "audio_intent": "ORIGINAL_AUDIO" if highlight else "NARRATION_WITH_ORIGINAL_AUDIO",
                "subtitles": [],
            }],
        }],
    }


class Gateway:
    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def submit_text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return GatewayResult(
            raw_response="gateway-json",
            parsed_json=self.response,
            model=kwargs["model"],
        )


class SourcePipeline:
    def prepare_episode(self, source_path, episode_id, source_id, **_kwargs):
        return PreparedEpisode(
            episode_id=episode_id,
            source_id=source_id,
            source_path=Path(source_path),
            source_basename=Path(source_path).name,
            source_fingerprint="source-sha",
            duration_ms=10_000,
            canvas_width=1920,
            canvas_height=1080,
            transcript=Transcript(
                episode_id=episode_id,
                source_type="sidecar",
                source_format="srt",
                language="en",
                cues=(TranscriptCue("cue-1", 1_000, 3_000, "Verified dialogue."),),
            ),
            transcript_method="sidecar",
            artifact_hash="prepared-hash",
        )


def test_analysis_model_reuses_existing_gateway_configuration():
    settings = AppSettings(scanner_model="scanner", finalizer_model="creative", finalizer_reasoning="high")
    assert select_analysis_model(settings) == ("creative", "high")
    assert select_analysis_model(AppSettings(scanner_model="scanner", scanner_reasoning="medium")) == (
        "scanner", "medium",
    )


def test_gateway_analyzer_makes_one_editorial_request(tmp_path):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video")
    episode = SourcePipeline().prepare_episode(video, "E01", "src_001")
    gateway = Gateway(render_json(video.name))
    analyzer = GatewayRenderJsonAnalyzer(
        gateway,
        AppSettings(finalizer_model="creative-model", finalizer_reasoning="high"),
    )
    result = analyzer.analyze(
        mode="RECAP",
        project_id="project-1",
        project_name="Project One",
        editorial_prompt="Create a factual recap.",
        episodes=[episode],
    )
    assert result.final_json["outputs"][0]["render_id"] == "out_001"
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["phase"] == "v5_render_json"
    assert "render_json_schema" in gateway.calls[0]["prompt"]


def test_v5_workflow_is_prepare_one_gateway_call_validate_then_render(tmp_path, monkeypatch):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"synthetic-video")
    persistence = ProjectPersistence(storage_root=tmp_path / "state")
    gateway = Gateway(render_json(video.name))
    settings = AppSettings(finalizer_model="creative-model")
    workflow = V5Workflow(
        persistence=persistence,
        gateway_client=gateway,
        voice_adapter=None,
        source_pipeline=SourcePipeline(),
    )
    monkeypatch.setattr(
        "toolrecap_v4.v5.workflow.probe_media",
        lambda *_args, **_kwargs: SimpleNamespace(duration=10.0),
    )

    rendered = []

    def fake_render(self, project_id, **_kwargs):
        state = self.persistence.load_project(project_id)
        assert state["final_json"]["outputs"]
        state["status"] = "completed"
        self.persistence.save_project(state)
        rendered.append(project_id)
        return state

    monkeypatch.setattr("toolrecap_v4.v5.workflow.ProjectWorkflow.start_project", fake_render)
    workflow.create_project(
        project_id="project-1",
        project_name="Project One",
        source_input=video,
        prompt="Create a factual recap.",
        mode="RECAP",
        settings=settings,
    )
    state = workflow.run("project-1", settings=settings)
    assert state["pipeline_version"] == "5.0"
    assert state["status"] == "completed"
    assert len(gateway.calls) == 1
    assert rendered == ["project-1"]
    assert persistence.load_final_json("project-1")["schema_version"] == "3.0"


def test_highlight_contract_rejects_narration():
    invalid = render_json("episode.mp4", highlight=True)
    invalid["outputs"][0]["segments"][0]["narration"] = "This must not be narrated."
    with pytest.raises(ValueError, match="cannot contain generated narration"):
        validate_mode_render_contract("HIGHLIGHT", invalid)


def test_v5_default_output_is_inside_source_folder_and_uses_folder_name(tmp_path):
    working = tmp_path / "Season 05"
    working.mkdir()
    assert resolve_v5_publication_root(
        manual_output_dir="",
        working_folder=working,
        last_source_file="Game.of.Thrones.S05E10.mkv",
        mode="HIGHLIGHT",
    ) == working / "Output_Season 05_Highlight"
    assert resolve_v5_publication_root(
        manual_output_dir=None,
        working_folder=working,
        last_source_file="Game.of.Thrones.S05E10.mkv",
        mode="RECAP",
    ) == working / "Output_Season 05_Recap"


def test_v5_create_project_keeps_automatic_mode_specific_output(tmp_path, monkeypatch):
    season = tmp_path / "S04"
    season.mkdir()
    first = season / "Episode01.mp4"
    last = season / "Episode10.mp4"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    workflow = V5Workflow(
        persistence=ProjectPersistence(storage_root=tmp_path / "state"),
        gateway_client=Gateway(render_json(first.name)),
        voice_adapter=None,
        source_pipeline=SourcePipeline(),
    )
    monkeypatch.setattr(
        "toolrecap_v4.v5.workflow.probe_media",
        lambda *_args, **_kwargs: SimpleNamespace(duration=10.0),
    )
    state = workflow.create_project(
        project_id="project-1",
        project_name="Project One",
        source_input=[last, first],
        prompt="Find highlights.",
        mode="HIGHLIGHT",
        settings=AppSettings(finalizer_model="creative-model"),
    )
    assert Path(state["output_dir"]) == season / "Output_S04_Highlight"


def test_v5_requested_folder_examples(tmp_path):
    marshals = tmp_path / "1.TVShows" / "Marshals"
    s04 = tmp_path / "Yellowstone (2018)" / "S04"
    marshals.mkdir(parents=True)
    s04.mkdir(parents=True)

    assert resolve_v5_publication_root(
        manual_output_dir="", working_folder=marshals,
        last_source_file="Marshals.S01E08.mkv", mode="HIGHLIGHT",
    ) == marshals / "Output_Marshals_Highlight"
    assert resolve_v5_publication_root(
        manual_output_dir="", working_folder=marshals,
        last_source_file="Marshals.S01E08.mkv", mode="RECAP",
    ) == marshals / "Output_Marshals_Recap"
    assert resolve_v5_publication_root(
        manual_output_dir="", working_folder=s04,
        last_source_file="Yellowstone.S04E10.mkv", mode="HIGHLIGHT",
    ) == s04 / "Output_S04_Highlight"
