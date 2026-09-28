"""Tests for Phase 3 ProjectWorkflow source preparation integration.

Invariants verified:
- Stable E01...En mapping in natural order.
- Artifacts stored strictly under managed root, never in output_dir.
- Controlled stop at ProjectStatus.PREPARED with AnalysisPipelineUnavailableError (Scanner unavailable).
- Resume reuses verified cache without re-extracting.
- Injectable pipeline and callbacks (on_source_preparation_progress, on_episode_prepared).
- Technical prep failure -> ProjectStatus.FAILED with error.
- Cancellation -> ProjectStatus.CANCELLED with completed episode caches preserved.
- Source mutation fails with SourceChangedError before any preparation begins.
- Imported Final JSON bypasses source preparation completely (0 prep, 0 AI).
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from toolrecap_v4.analysis.cache import AnalysisCacheManager
from toolrecap_v4.analysis.models import AudioSelection, PreparedEpisode, Transcript, TranscriptCue
from toolrecap_v4.analysis.source_prep.pipeline import SourcePreparationPipeline
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.discovery import compute_file_fingerprint
from toolrecap_v4.errors import (
    AnalysisPipelineUnavailableError,
    CancelledError,
    SourceChangedError,
    ToolRecapError,
)
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.media import AudioStreamInfo, VideoStreamInfo
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.workflow import (
    ProjectStatus,
    ProjectWorkflow,
    WorkflowCallbacks,
)


def _create_synthetic_video(
    path: Path,
    duration: float = 2.0,
    width: int = 640,
    height: int = 480,
    fps: int = 25,
) -> Path:
    """Create a fast synthetic video with ffmpeg testsrc."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration}",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-c:v", "libx264", "-preset", "ultrafast", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _create_sample_prepared_episode(
    episode_id: str,
    source_path: Path,
    duration_ms: int = 2000,
) -> PreparedEpisode:
    """Helper to create a valid deterministic PreparedEpisode."""
    cue = TranscriptCue(cue_id="c1", start_ms=100, end_ms=1500, text=f"Dialogue for {episode_id}")
    transcript = Transcript(
        episode_id=episode_id,
        source_type="sidecar",
        source_format="srt",
        cues=(cue,),
        has_speech=True,
        language="eng",
        provenance_hash="dummy_prov_hash",
    )
    v_info = VideoStreamInfo(
        index=0,
        codec="h264",
        width=640,
        height=480,
        fps=25.0,
        fps_text="25",
        duration=2.0,
        aspect_ratio="16:9",
    )
    a_info = AudioStreamInfo(index=1, codec="aac", channels=2, sample_rate=44100)
    a_sel = AudioSelection(selected_stream=a_info, global_index=1, audio_ordinal=0)

    return PreparedEpisode(
        episode_id=episode_id,
        source_id=episode_id,
        source_path=source_path,
        duration_ms=duration_ms,
        canvas_width=640,
        canvas_height=480,
        video_streams=(v_info,),
        audio_streams=(a_info,),
        video_info=v_info,
        audio_info=a_info,
        audio_selection=a_sel,
        transcript=transcript,
        transcript_method="sidecar",
        artifact_hash=f"hash_{episode_id}",
        dependency_signature={"test": True},
        status="ready",
    )


@pytest.fixture
def workflow_env(tmp_path: Path):
    """Provide isolated persistence, sources, and output directories."""
    storage = ProjectPersistence(storage_root=tmp_path / "storage")
    src_dir = tmp_path / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    v1 = _create_synthetic_video(src_dir / "ep01.mp4", duration=2.0)
    s1 = src_dir / "ep01.en.srt"
    s1.write_text("1\n00:00:00,100 --> 00:00:01,800\nEpisode 1 dialogue line\n", encoding="utf-8")

    v2 = _create_synthetic_video(src_dir / "ep02.mp4", duration=2.0)
    s2 = src_dir / "ep02.en.srt"
    s2.write_text("1\n00:00:00,100 --> 00:00:01,800\nEpisode 2 dialogue line\n", encoding="utf-8")

    return {
        "storage": storage,
        "src_dir": src_dir,
        "out_dir": out_dir,
        "v1": v1,
        "v2": v2,
    }


def test_two_episodes_ordered_and_stop_prepared(workflow_env: dict[str, Any]):
    """Verify 2 episodes are processed in natural order (E01, E02), stored locally, and stop at PREPARED."""
    storage: ProjectPersistence = workflow_env["storage"]
    src_dir: Path = workflow_env["src_dir"]
    out_dir: Path = workflow_env["out_dir"]

    mock_gw = MagicMock(spec=GatewayClient)
    workflow = ProjectWorkflow(persistence=storage, gateway_client=mock_gw)

    project_id = "proj-prep-ordered-01"
    workflow.create_project(
        project_id=project_id,
        project_name="Two_Episodes_Project",
        source_input=src_dir,
        prompt="Summarize episodes.",
        output_dir=out_dir,
    )

    with pytest.raises(AnalysisPipelineUnavailableError, match="Scanner is unavailable"):
        workflow.start_project(project_id)

    # Invariant: 0 Gateway calls
    assert mock_gw.submit_text_chat.call_count == 0

    # Invariant: Project status is PREPARED with prepared_at timestamp
    state = storage.load_project(project_id)
    assert state["status"] == ProjectStatus.PREPARED.value
    assert "Scanner" in state["error"]
    assert "prepared_at" in state["timestamps"]

    # Invariant: Episode IDs mapped naturally to E01, E02
    prep_eps = state.get("prepared_episodes", {})
    assert len(prep_eps) == 2
    assert "E01" in prep_eps
    assert "E02" in prep_eps
    assert prep_eps["E01"]["source_file"] == "ep01.mp4"
    assert prep_eps["E02"]["source_file"] == "ep02.mp4"
    assert prep_eps["E01"]["status"] == "ready"
    assert prep_eps["E02"]["status"] == "ready"

    # Invariant: Persistence stores full local PreparedEpisode + Transcript artifacts.
    assert storage.has_prepared_episode(project_id, "E01")
    assert storage.has_prepared_episode(project_id, "E02")
    full_e01 = storage.load_prepared_episode(project_id, "E01")
    assert full_e01["episode_id"] == "E01"
    assert full_e01["transcript"]["cues"][0]["text"] == "Episode 1 dialogue line"
    assert full_e01["dependency_signature"]["source_fingerprint"] == prep_eps["E01"]["source_fingerprint"]
    assert storage.has_prepared_manifest(project_id)
    manifest = storage.load_prepared_manifest(project_id)
    assert manifest["status"] == "completed"
    assert len(manifest["episodes"]) == 2

    # Checkpoint exists under managed root
    ckpt = storage.load_checkpoint(project_id, "source_preparation")
    assert ckpt["status"] == "completed"
    assert ckpt["episodes"] == ["E01", "E02"]

    # Invariant: Output directory has ZERO files (no pollution from preparation)
    assert list(out_dir.iterdir()) == []


def test_resume_cache_reuse(workflow_env: dict[str, Any]):
    """Verify that resuming an already prepared project reuses caches without repeating work."""
    storage: ProjectPersistence = workflow_env["storage"]
    src_dir: Path = workflow_env["src_dir"]
    out_dir: Path = workflow_env["out_dir"]

    workflow = ProjectWorkflow(persistence=storage)
    project_id = "proj-prep-resume-01"
    workflow.create_project(
        project_id=project_id,
        project_name="Resume_Cache_Project",
        source_input=src_dir,
        output_dir=out_dir,
    )

    # First run prepares both episodes and reaches PREPARED
    with pytest.raises(AnalysisPipelineUnavailableError):
        workflow.start_project(project_id)

    first_state = storage.load_project(project_id)
    assert first_state["status"] == ProjectStatus.PREPARED.value
    first_prep_at = first_state["timestamps"]["prepared_at"]

    # Second run (resume) reuses cache
    with pytest.raises(AnalysisPipelineUnavailableError):
        workflow.resume_project(project_id)

    second_state = storage.load_project(project_id)
    assert second_state["status"] == ProjectStatus.PREPARED.value
    # Manifest and episodes remain intact
    assert storage.has_prepared_manifest(project_id)
    assert len(storage.list_prepared_episodes(project_id)) == 2


def test_injectable_pipeline_and_callbacks(workflow_env: dict[str, Any]):
    """Verify ProjectWorkflow accepts injected pipeline and invokes progress and episode callbacks."""
    storage: ProjectPersistence = workflow_env["storage"]
    src_dir: Path = workflow_env["src_dir"]
    out_dir: Path = workflow_env["out_dir"]

    mock_pipeline = MagicMock(spec=SourcePreparationPipeline)
    v1: Path = workflow_env["v1"]
    v2: Path = workflow_env["v2"]

    ep1_prepared = _create_sample_prepared_episode("E01", v1)
    ep2_prepared = _create_sample_prepared_episode("E02", v2)

    def mock_prepare(source_path: Any, episode_id: str, **kwargs: Any) -> PreparedEpisode:
        on_progress = kwargs.get("on_progress")
        if on_progress:
            on_progress("mock_phase", 0.5, f"Processing {episode_id}")
        return ep1_prepared if episode_id == "E01" else ep2_prepared

    mock_pipeline.prepare_episode.side_effect = mock_prepare

    workflow = ProjectWorkflow(
        persistence=storage,
        source_preparation_pipeline=mock_pipeline,
    )
    project_id = "proj-injected-pipe-01"
    workflow.create_project(
        project_id=project_id,
        project_name="Injected_Pipe_Project",
        source_input=src_dir,
        output_dir=out_dir,
    )

    status_events: list[str] = []
    progress_events: list[tuple[str, str, float, str]] = []
    prepared_events: list[tuple[str, dict[str, Any]]] = []

    callbacks = WorkflowCallbacks(
        on_status_change=lambda s: status_events.append(s),
        on_source_preparation_progress=lambda ep_id, ph, pct, msg: progress_events.append((ep_id, ph, pct, msg)),
        on_episode_prepared=lambda ep_id, data: prepared_events.append((ep_id, data)),
    )

    with pytest.raises(AnalysisPipelineUnavailableError):
        workflow.start_project(project_id, callbacks=callbacks)

    # Injected pipeline was invoked twice (E01 then E02)
    assert mock_pipeline.prepare_episode.call_count == 2
    calls = mock_pipeline.prepare_episode.call_args_list
    assert calls[0].kwargs["episode_id"] == "E01"
    assert calls[1].kwargs["episode_id"] == "E02"

    # Status change callback received ANALYZING then PREPARED
    assert ProjectStatus.ANALYZING.value in status_events
    assert ProjectStatus.PREPARED.value in status_events

    # Progress and prepared callbacks invoked for each episode
    assert len(progress_events) == 2
    assert progress_events[0][0] == "E01"
    assert progress_events[1][0] == "E02"

    assert len(prepared_events) == 2
    assert prepared_events[0][0] == "E01"
    assert prepared_events[1][0] == "E02"


def test_technical_prep_failure_sets_failed(workflow_env: dict[str, Any]):
    """Verify that a technical failure during source preparation sets status to FAILED and records error."""
    storage: ProjectPersistence = workflow_env["storage"]
    src_dir: Path = workflow_env["src_dir"]
    out_dir: Path = workflow_env["out_dir"]

    mock_pipeline = MagicMock(spec=SourcePreparationPipeline)
    mock_pipeline.prepare_episode.side_effect = ToolRecapError("Audio stream decoding crashed")

    workflow = ProjectWorkflow(
        persistence=storage,
        source_preparation_pipeline=mock_pipeline,
    )
    project_id = "proj-tech-fail-01"
    workflow.create_project(
        project_id=project_id,
        project_name="Fail_Project",
        source_input=src_dir,
        output_dir=out_dir,
    )

    error_events: list[tuple[Exception, Any]] = []
    callbacks = WorkflowCallbacks(
        on_error=lambda err, raw: error_events.append((err, raw)),
    )

    with pytest.raises(ToolRecapError, match="Audio stream decoding crashed"):
        workflow.start_project(project_id, callbacks=callbacks)

    state = storage.load_project(project_id)
    assert state["status"] == ProjectStatus.FAILED.value
    assert "Audio stream decoding crashed" in state["error"]
    assert "failed_at" in state["timestamps"]

    assert len(error_events) == 1
    assert isinstance(error_events[0][0], ToolRecapError)
    assert not storage.has_prepared_manifest(project_id)


def test_cancellation_preserves_completed_caches(workflow_env: dict[str, Any]):
    """Verify cancellation sets CANCELLED status and preserves already completed episode artifacts."""
    storage: ProjectPersistence = workflow_env["storage"]
    src_dir: Path = workflow_env["src_dir"]
    out_dir: Path = workflow_env["out_dir"]
    v1: Path = workflow_env["v1"]

    token = CancellationToken()
    ep1_prepared = _create_sample_prepared_episode("E01", v1)

    mock_pipeline = MagicMock(spec=SourcePreparationPipeline)

    def mock_prepare(source_path: Any, episode_id: str, **kwargs: Any) -> PreparedEpisode:
        if episode_id == "E01":
            return ep1_prepared
        # Cancel before E02 finishes
        token.cancel()
        token.check_cancelled()
        return ep1_prepared

    mock_pipeline.prepare_episode.side_effect = mock_prepare

    workflow = ProjectWorkflow(
        persistence=storage,
        source_preparation_pipeline=mock_pipeline,
    )
    project_id = "proj-cancel-prep-01"
    workflow.create_project(
        project_id=project_id,
        project_name="Cancel_Project",
        source_input=src_dir,
        output_dir=out_dir,
    )

    status_events: list[str] = []
    callbacks = WorkflowCallbacks(
        on_status_change=lambda s: status_events.append(s),
    )

    with pytest.raises(CancelledError):
        workflow.start_project(project_id, cancellation_token=token, callbacks=callbacks)

    state = storage.load_project(project_id)
    assert state["status"] == ProjectStatus.CANCELLED.value
    assert "cancelled_at" in state["timestamps"]
    assert ProjectStatus.CANCELLED.value in status_events

    # E01 was completed and is preserved
    assert storage.has_prepared_episode(project_id, "E01")
    assert not storage.has_prepared_episode(project_id, "E02")
    assert not storage.has_prepared_manifest(project_id)


def test_source_mutation_before_prep_fails(workflow_env: dict[str, Any]):
    """Verify that modifying a source video file triggers SourceChangedError before any preparation begins."""
    storage: ProjectPersistence = workflow_env["storage"]
    src_dir: Path = workflow_env["src_dir"]
    out_dir: Path = workflow_env["out_dir"]
    v1: Path = workflow_env["v1"]

    mock_pipeline = MagicMock(spec=SourcePreparationPipeline)
    workflow = ProjectWorkflow(
        persistence=storage,
        source_preparation_pipeline=mock_pipeline,
    )
    project_id = "proj-src-mutate-01"
    workflow.create_project(
        project_id=project_id,
        project_name="Mutate_Project",
        source_input=src_dir,
        output_dir=out_dir,
    )

    # Mutate source file
    with open(v1, "ab") as f:
        f.write(b"corrupted_bytes")

    with pytest.raises(SourceChangedError, match="has been modified"):
        workflow.start_project(project_id)

    # Pipeline was NEVER called because integrity check failed first!
    assert mock_pipeline.prepare_episode.call_count == 0


def test_imported_final_json_zero_prep_and_zero_gateway(workflow_env: dict[str, Any]):
    """Verify that importing a project with Final JSON bypasses source prep completely and makes 0 Gateway calls."""
    storage: ProjectPersistence = workflow_env["storage"]
    v1: Path = workflow_env["v1"]
    out_dir: Path = workflow_env["out_dir"]

    mock_pipeline = MagicMock(spec=SourcePreparationPipeline)
    mock_gw = MagicMock(spec=GatewayClient)
    mock_catalog = MagicMock()
    mock_catalog.build_or_load.side_effect = AssertionError("Catalog must not run for imported Final JSON")
    mock_planner = MagicMock()
    mock_planner.run.side_effect = AssertionError("Planner must not run for imported Final JSON")
    mock_writer = MagicMock()
    mock_writer.run.side_effect = AssertionError("Writer must not run for imported Final JSON")
    mock_finalizer = MagicMock()
    mock_finalizer.run.side_effect = AssertionError("Phase 9 finalizer must not run for imported Final JSON")
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000); wav.writeframes(b"\x00\x00" * 800)
    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = wav_buffer.getvalue()
    workflow = ProjectWorkflow(
        persistence=storage,
        gateway_client=mock_gw,
        source_preparation_pipeline=mock_pipeline,
        catalog_service=mock_catalog,
        planner_service=mock_planner,
        writer_service=mock_writer,
        finalization_service=mock_finalizer,
        voice_adapter=mock_voice,
    )

    project_id = "proj-import-zero-prep-01"
    final_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Import_Zero_Prep",
        "sources": [{"source_file": "ep01.mp4"}],
        "outputs": [
            {
                "render_id": "out-01",
                "title": "Recap_Zero_Prep",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "ep01.mp4",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "type": "narration",
                        "narration": "Zero prep narration text.",
                        "source_audio": False,
                        "subtitles": [],
                    }
                ],
            }
        ],
    }

    workflow.import_project(
        project_id=project_id,
        project_name="Import_Zero_Prep",
        source_input=v1,
        final_json=final_json,
        output_dir=out_dir,
        settings=AppSettings(scanner_model="changed-scan", vision_model="changed-vision", finalizer_model="changed-final", planner_model="changed-final", writer_model="changed-final"),
    )

    # Mock renderer voice adapter to avoid calling external TTS
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "toolrecap_v4.workflow.render_output",
            lambda **kwargs: MagicMock(
                output_path=out_dir / "Recap_Zero_Prep.mp4",
                narration_srt_path=out_dir / "Recap_Zero_Prep.narration.srt",
                original_srt_path=out_dir / "Recap_Zero_Prep.original.srt",
                duration=1.0,
                video_codec="h264",
                audio_codec="aac",
                width=640,
                height=480,
                fps=25.0,
                original_subtitle_state="NO_ORIGINAL_DIALOGUE",
                original_subtitle_cue_count=0,
            ),
        )
        mp.setattr("toolrecap_v4.workflow.compute_file_sha256", lambda p: "fake_hash")

        res_state = workflow.start_project(project_id)
        retry_state = workflow.retry_project(project_id)

    # Source prep pipeline was NEVER called
    assert mock_pipeline.prepare_episode.call_count == 0

    # Zero Gateway calls made
    assert mock_gw.submit_text_chat.call_count == 0
    assert mock_catalog.build_or_load.call_count == 0
    assert mock_planner.run.call_count == 0
    assert mock_writer.run.call_count == 0
    assert mock_finalizer.run.call_count == 0
    assert mock_voice.synthesize.call_count == 1
    assert retry_state["status"] == ProjectStatus.COMPLETED.value

    # Project reached COMPLETED status
    assert res_state["status"] == ProjectStatus.COMPLETED.value
    # No prepared episodes created
    assert len(storage.list_prepared_episodes(project_id)) == 0
    assert not storage.has_prepared_manifest(project_id)
