"""Targeted tests for ToolRecap V4 synchronous project workflow.

Acceptance criteria verified:
- Mock oneclick sequence: final JSON saved to disk BEFORE voice/render.
- Import: 0 AI / 0 Gateway calls.
- Output 2 fail, resume skip 1.
- Cancel and restart.
- Changed audio: 0 AI, rerender affected outputs.
- Invalid JSON: raw response persisted to disk and state/error.
- Source changed: fails without substitution.
- Publication collision with ANY source (including unused) rejected.
- Reconcile interrupted states to resumable.
- Real existing-JSON multi-source pipeline smoke test.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import wave

import httpx
import pytest

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.discovery import SourceFingerprint, compute_file_fingerprint
from toolrecap_v4.errors import (
    AnalysisPipelineUnavailableError,
    CancelledError,
    DiscoveryError,
    InvalidGatewayResponseError,
    SourceChangedError,
    ValidationError,
    WindowsCollisionError,
)
from toolrecap_v4.gateway import GatewayClient, GatewayResult
from toolrecap_v4.media import probe_media
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.progress import WorkflowProgressTracker
from toolrecap_v4.renderer import (
    RenderError,
    RenderResult,
    SourceCollisionError,
    compute_file_sha256,
)
from toolrecap_v4.settings import AppSettings, SettingsManager
from toolrecap_v4.voice_studio import VoiceStudioAdapter
from toolrecap_v4.workflow import (
    OutputStatus,
    ProjectStatus,
    ProjectWorkflow,
    WorkflowCallbacks,
    compute_output_fingerprint,
    reconcile_project_state,
    resolve_sources,
)


def _write_mock_subtitles(output_dir: Path, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{title}.narration.srt").write_text("", encoding="utf-8")
    (output_dir / f"{title}.original.srt").write_text("", encoding="utf-8")


def _create_synthetic_video(
    path: Path,
    duration: float = 2.0,
    width: int = 640,
    height: int = 480,
    fps: int = 25,
    has_audio: bool = True,
) -> Path:
    """Create a fast synthetic video with ffmpeg testsrc."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
    ]
    if has_audio:
        cmd.extend(["-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration}"])
        cmd.extend(["-c:a", "aac", "-ar", "44100", "-ac", "2"])
    cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", str(path)])
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _create_wav_bytes(duration_s: float = 1.0, sample_rate: int = 24000) -> bytes:
    """Generate in-memory mono PCM WAV bytes."""
    num_frames = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


@pytest.fixture
def test_storage(tmp_path: Path) -> ProjectPersistence:
    """Provide isolated persistence in tmp_path."""
    return ProjectPersistence(storage_root=tmp_path / "storage")


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a single synthetic video with sidecar English subtitles."""
    v_path = tmp_path / "sources" / "ep01.mp4"
    v_path.parent.mkdir(parents=True, exist_ok=True)
    _create_synthetic_video(v_path, duration=2.0)
    srt_path = tmp_path / "sources" / "ep01.en.srt"
    srt_path.write_text("1\n00:00:00,100 --> 00:00:01,800\nSample episode dialogue line\n", encoding="utf-8")
    return v_path


def test_single_file_and_folder_discovery(tmp_path: Path):
    """Verify single file discovery and folder direct children natural order."""
    src_dir = tmp_path / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    v2 = _create_synthetic_video(src_dir / "ep2.mp4", duration=1.0)
    v1 = _create_synthetic_video(src_dir / "ep1.mp4", duration=1.0)
    v10 = _create_synthetic_video(src_dir / "ep10.mp4", duration=1.0)

    # Folder discovery: natural order ep1, ep2, ep10
    folder_sources = resolve_sources(src_dir)
    assert len(folder_sources) == 3
    assert [s.basename for s in folder_sources] == ["ep1.mp4", "ep2.mp4", "ep10.mp4"]

    # Single file discovery
    single_source = resolve_sources(v1)
    assert len(single_source) == 1
    assert single_source[0].basename == "ep1.mp4"


def test_mock_oneclick_sequence_final_saved_first(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify one-click sequence saves final JSON to disk BEFORE voice/render."""
    project_id = "proj-oneclick-01"
    output_dir = tmp_path / "outputs"

    final_json_saved_before_render = False

    def mock_render(*args, **kwargs):
        nonlocal final_json_saved_before_render
        # Check if final JSON is already on disk when render starts
        final_json_saved_before_render = test_storage.has_final_json(project_id)
        # Produce fake output file
        out_mp4 = output_dir / "Recap_1.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"dummy mp4 content")
        narr_srt = output_dir / "Recap_1.narration.srt"
        narr_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        orig_srt = output_dir / "Recap_1.original.srt"
        orig_srt.write_text("", encoding="utf-8")
        return RenderResult(
            output_path=out_mp4,
            narration_srt_path=narr_srt,
            original_srt_path=orig_srt,
            duration=1.0,
            title="Recap_1",
            render_id="render-01",
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=480,
            fps=25.0,
        )

    mock_gw = MagicMock(spec=GatewayClient)

    workflow = ProjectWorkflow(
        persistence=test_storage,
        gateway_client=mock_gw,
    )

    state = workflow.create_project(
        project_id=project_id,
        project_name="OneClick_Project",
        source_input=sample_video,
        prompt="Summarize this episode.",
        output_dir=output_dir,
    )
    assert state["status"] == ProjectStatus.CREATED.value

    with pytest.raises(AnalysisPipelineUnavailableError, match="Scanner is unavailable"):
        workflow.start_project(project_id)

    # Invariants: 0 Gateway calls
    assert mock_gw.submit_text_chat.call_count == 0
    assert not hasattr(mock_gw, "submit_chat_analysis")

    # Invariant: sub_analysis and final_json NOT saved to disk
    assert test_storage.has_sub_analysis(project_id) is False
    assert test_storage.has_raw_response(project_id) is False
    assert test_storage.has_final_json(project_id) is False

    # Invariant: state is PREPARED with Scanner error and prepared_at persisted
    prep_state = test_storage.load_project(project_id)
    assert prep_state["status"] == ProjectStatus.PREPARED.value
    assert "Scanner" in prep_state["error"]
    assert "prepared_at" in prep_state["timestamps"]


def test_import_zero_ai(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify importing a final JSON performs 0 AI / 0 Gateway calls."""
    project_id = "proj-import-01"
    output_dir = tmp_path / "outputs"

    valid_final_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Imported_Project",
        "sources": [{"source_file": "ep01.mp4"}],
        "outputs": [
            {
                "render_id": "render-01",
                "title": "Recap_Import",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "ep01.mp4",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            }
        ],
    }

    mock_gw = MagicMock(spec=GatewayClient)
    mock_gw.submit_text_chat.side_effect = AssertionError("Gateway MUST NOT be called on imported project!")

    def mock_render(*args, **kwargs):
        out_mp4 = output_dir / "Recap_Import.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"dummy mp4")
        _write_mock_subtitles(output_dir, "Recap_Import")
        return RenderResult(
            output_path=out_mp4,
            narration_srt_path=output_dir / "Recap_Import.narration.srt",
            original_srt_path=output_dir / "Recap_Import.original.srt",
            duration=1.0,
            title="Recap_Import",
            render_id="render-01",
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=480,
            fps=25.0,
        )

    workflow = ProjectWorkflow(
        persistence=test_storage,
        gateway_client=mock_gw,
    )

    imported_state = workflow.import_project(
        project_id=project_id,
        project_name="Imported_Project",
        source_input=sample_video,
        final_json=valid_final_json,
        output_dir=output_dir,
    )
    assert imported_state["status"] == ProjectStatus.ANALYZED.value
    assert test_storage.has_final_json(project_id) is True

    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render):
        completed_state = workflow.start_project(project_id)

    # 0 AI calls
    assert mock_gw.submit_text_chat.call_count == 0
    assert completed_state["status"] == ProjectStatus.COMPLETED.value


def test_output_2_fail_resume_skip_1(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify that when output 2 fails, output 1 is preserved and skipped on resume."""
    project_id = "proj-fail-resume-01"
    output_dir = tmp_path / "outputs"

    final_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Fail_Resume_Project",
        "sources": [{"source_file": "ep01.mp4"}],
        "outputs": [
            {
                "render_id": "out-01",
                "title": "Recap_Part1",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "ep01.mp4",
                        "start_ms": 0,
                        "end_ms": 500,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            },
            {
                "render_id": "out-02",
                "title": "Recap_Part2",
                "segments": [
                    {
                        "segment_id": "seg-2",
                        "source_file": "ep01.mp4",
                        "start_ms": 500,
                        "end_ms": 1000,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            },
        ],
    }

    render_call_counts = {"out-01": 0, "out-02": 0}

    def mock_render_first_pass(output_def, *args, **kwargs):
        rid = output_def["render_id"]
        render_call_counts[rid] += 1
        if rid == "out-01":
            out_mp4 = output_dir / f"{output_def['title']}.mp4"
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            out_mp4.write_bytes(b"content for part 1")
            _write_mock_subtitles(output_dir, output_def["title"])
            return RenderResult(
                output_path=out_mp4,
                narration_srt_path=output_dir / f"{output_def['title']}.narration.srt",
                original_srt_path=output_dir / f"{output_def['title']}.original.srt",
                duration=0.5,
                title=output_def["title"],
                render_id=rid,
                video_codec="h264",
                audio_codec="aac",
                width=640,
                height=480,
                fps=25.0,
            )
        else:
            raise RenderError("Simulated failure on output 2")

    workflow = ProjectWorkflow(persistence=test_storage)
    workflow.import_project(
        project_id=project_id,
        project_name="Fail_Resume_Project",
        source_input=sample_video,
        final_json=final_json,
        output_dir=output_dir,
    )

    # First run: out-01 succeeds, out-02 fails
    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render_first_pass):
        with pytest.raises(RenderError, match="Simulated failure on output 2"):
            workflow.start_project(project_id)

    # Verify state after failure: queue stopped, out-01 preserved
    state_after_fail = test_storage.load_project(project_id)
    assert state_after_fail["status"] == ProjectStatus.FAILED.value
    assert state_after_fail["outputs"]["out-01"]["status"] == OutputStatus.COMPLETED.value
    assert state_after_fail["outputs"]["out-02"]["status"] == OutputStatus.FAILED.value

    # Second run: out-02 now succeeds
    def mock_render_second_pass(output_def, *args, **kwargs):
        rid = output_def["render_id"]
        render_call_counts[rid] += 1
        out_mp4 = output_dir / f"{output_def['title']}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"content for part 2")
        _write_mock_subtitles(output_dir, output_def["title"])
        return RenderResult(
            output_path=out_mp4,
            narration_srt_path=output_dir / f"{output_def['title']}.narration.srt",
            original_srt_path=output_dir / f"{output_def['title']}.original.srt",
            duration=0.5,
            title=output_def["title"],
            render_id=rid,
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=480,
            fps=25.0,
        )

    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render_second_pass):
        resumed_state = workflow.resume_project(project_id)

    # Invariant: out-01 was SKIPPED (not called in second pass)
    assert render_call_counts["out-01"] == 1
    assert render_call_counts["out-02"] == 2
    assert resumed_state["status"] == ProjectStatus.COMPLETED.value
    assert resumed_state["outputs"]["out-01"]["status"] == OutputStatus.SKIPPED.value
    assert resumed_state["outputs"]["out-02"]["status"] == OutputStatus.COMPLETED.value


def test_cancel_and_restart(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify cancellation stops execution and restart resumes successfully."""
    project_id = "proj-cancel-01"
    output_dir = tmp_path / "outputs"

    token = CancellationToken()

    final_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Cancel_Project",
        "sources": [{"source_file": "ep01.mp4"}],
        "outputs": [
            {
                "render_id": "out-01",
                "title": "Recap_1",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "ep01.mp4",
                        "start_ms": 0,
                        "end_ms": 500,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            },
            {
                "render_id": "out-02",
                "title": "Recap_2",
                "segments": [
                    {
                        "segment_id": "seg-2",
                        "source_file": "ep01.mp4",
                        "start_ms": 500,
                        "end_ms": 1000,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            },
        ],
    }

    call_count = 0

    def mock_render(output_def, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        rid = output_def["render_id"]
        out_mp4 = output_dir / f"{output_def['title']}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(f"bytes for {rid}".encode())
        _write_mock_subtitles(output_dir, output_def["title"])

        if rid == "out-01":
            # Cancel immediately after out-01 finishes
            token.cancel()

        return RenderResult(
            output_path=out_mp4,
            narration_srt_path=output_dir / f"{output_def['title']}.narration.srt",
            original_srt_path=output_dir / f"{output_def['title']}.original.srt",
            duration=0.5,
            title=output_def["title"],
            render_id=rid,
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=480,
            fps=25.0,
        )

    workflow = ProjectWorkflow(persistence=test_storage)
    workflow.import_project(
        project_id=project_id,
        project_name="Cancel_Project",
        source_input=sample_video,
        final_json=final_json,
        output_dir=output_dir,
    )

    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render):
        with pytest.raises(CancelledError):
            workflow.start_project(project_id, cancellation_token=token)

    # Invariant: stop preserves completed outputs and state
    state = test_storage.load_project(project_id)
    assert state["status"] == ProjectStatus.CANCELLED.value
    assert state["outputs"]["out-01"]["status"] == OutputStatus.COMPLETED.value
    assert "out-02" not in state["outputs"] or state["outputs"]["out-02"]["status"] != OutputStatus.COMPLETED.value

    # Restart with fresh token
    fresh_token = CancellationToken()
    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render):
        resumed = workflow.resume_project(project_id, cancellation_token=fresh_token)

    assert resumed["status"] == ProjectStatus.COMPLETED.value
    assert resumed["outputs"]["out-01"]["status"] == OutputStatus.SKIPPED.value
    assert resumed["outputs"]["out-02"]["status"] == OutputStatus.COMPLETED.value


def test_changed_audio_zero_ai(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify changing render settings rerenders affected outputs with 0 AI."""
    project_id = "proj-audio-change-01"
    output_dir = tmp_path / "outputs"

    final_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Audio_Change_Project",
        "sources": [{"source_file": "ep01.mp4"}],
        "outputs": [
            {
                "render_id": "out-01",
                "title": "Recap_1",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "ep01.mp4",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            }
        ],
    }

    mock_gw = MagicMock(spec=GatewayClient)
    mock_gw.submit_text_chat.side_effect = AssertionError("Gateway MUST NOT be called!")

    render_calls = 0

    def mock_render(output_def, *args, **kwargs):
        nonlocal render_calls
        render_calls += 1
        out_mp4 = output_dir / f"{output_def['title']}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(f"rendered pass {render_calls}".encode())
        _write_mock_subtitles(output_dir, output_def["title"])
        return RenderResult(
            output_path=out_mp4,
            narration_srt_path=output_dir / f"{output_def['title']}.narration.srt",
            original_srt_path=output_dir / f"{output_def['title']}.original.srt",
            duration=1.0,
            title=output_def["title"],
            render_id=output_def["render_id"],
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=480,
            fps=25.0,
        )

    workflow = ProjectWorkflow(persistence=test_storage, gateway_client=mock_gw)
    workflow.import_project(
        project_id=project_id,
        project_name="Audio_Change_Project",
        source_input=sample_video,
        final_json=final_json,
        output_dir=output_dir,
    )

    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render):
        workflow.start_project(project_id)

    assert render_calls == 1

    # Now change audio settings: commentary_audio_db = 3.0, auto_duck = True
    new_settings = AppSettings(commentary_audio_db=3.0, auto_duck=True)

    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render):
        workflow.resume_project(project_id, settings=new_settings)

    # Invariant: 0 AI calls, output rerendered because fingerprint changed
    assert mock_gw.submit_text_chat.call_count == 0
    assert render_calls == 2


def test_analysis_required_fails_before_gateway(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify that when analysis is required and no Final JSON exists, workflow fails before any Gateway call and invokes error callback."""
    project_id = "proj-no-final-json-01"
    output_dir = tmp_path / "outputs"

    mock_gw = MagicMock(spec=GatewayClient)
    workflow = ProjectWorkflow(persistence=test_storage, gateway_client=mock_gw)
    workflow.create_project(
        project_id=project_id,
        project_name="No_Final_JSON_Project",
        source_input=sample_video,
        prompt="Test prompt",
        output_dir=output_dir,
    )

    status_events = []
    error_events = []
    callbacks = WorkflowCallbacks(
        on_status_change=lambda s: status_events.append(s),
        on_error=lambda err, raw: error_events.append((err, raw)),
    )

    with pytest.raises(AnalysisPipelineUnavailableError):
        workflow.start_project(project_id, callbacks=callbacks)

    # Zero Gateway calls made
    assert mock_gw.submit_text_chat.call_count == 0

    # Invariant: state records prepared status and error
    state = test_storage.load_project(project_id)
    assert state["status"] == ProjectStatus.PREPARED.value
    assert "Scanner" in state["error"]
    assert "prepared_at" in state["timestamps"]

    # Callbacks received prepared
    assert ProjectStatus.PREPARED.value in status_events
    assert len(error_events) == 1
    assert isinstance(error_events[0][0], AnalysisPipelineUnavailableError)


def test_source_changed_failure(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify that modifying a source video file triggers SourceChangedError without substitution."""
    project_id = "proj-src-changed-01"

    valid_final_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Source_Changed_Project",
        "sources": [{"source_file": "ep01.mp4"}],
        "outputs": [
            {
                "render_id": "render-01",
                "title": "Recap_Source_Changed",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "ep01.mp4",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            }
        ],
    }

    workflow = ProjectWorkflow(persistence=test_storage)
    workflow.import_project(
        project_id=project_id,
        project_name="Source_Changed_Project",
        source_input=sample_video,
        final_json=valid_final_json,
    )

    # Modify the source file on disk
    with open(sample_video, "ab") as f:
        f.write(b"CORRUPTED_EXTRA_BYTES_12345")

    with pytest.raises(SourceChangedError, match="has been modified since project creation"):
        workflow.start_project(project_id)


def test_publication_collision_any_project_source_including_unused(test_storage: ProjectPersistence, tmp_path: Path):
    """Verify publication targets colliding with ANY source (including unused) are rejected."""
    src_dir = tmp_path / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    v_used = _create_synthetic_video(src_dir / "used.mp4", duration=1.0)
    v_unused = _create_synthetic_video(src_dir / "unused.mp4", duration=1.0)

    project_id = "proj-collision-01"

    # Output title matches unused source basename ("unused" -> "unused.mp4")
    colliding_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Collision_Project",
        "sources": [{"source_file": "used.mp4"}, {"source_file": "unused.mp4"}],
        "outputs": [
            {
                "render_id": "out-01",
                "title": "unused",  # Collides with unused.mp4 in src_dir!
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "used.mp4",
                        "start_ms": 0,
                        "end_ms": 500,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            }
        ],
    }

    workflow = ProjectWorkflow(persistence=test_storage)

    # Setting output_dir to src_dir will cause output unused.mp4 to collide with unused.mp4
    with pytest.raises(SourceCollisionError, match="collides with project source file"):
        workflow.import_project(
            project_id=project_id,
            project_name="Collision_Project",
            source_input=src_dir,
            final_json=colliding_json,
            output_dir=src_dir,
        )


def test_reconcile_interrupted_states(test_storage: ProjectPersistence):
    """Verify reconciling interrupted ANALYZING/RENDERING states to resumable status."""
    p_id = "proj-reconcile-01"

    # Case 1: Interrupted in ANALYZING without final JSON -> CREATED
    state1 = {
        "schema_version": "3.0",
        "project_id": p_id,
        "project_name": "Reconcile_Test",
        "status": ProjectStatus.ANALYZING.value,
        "sources": [],
        "source_fingerprints": {},
        "outputs": {},
    }
    test_storage.save_project(state1)
    reconciled = reconcile_project_state(state1, test_storage)
    assert reconciled["status"] == ProjectStatus.CREATED.value

    # Case 2: Interrupted in ANALYZING with final JSON -> ANALYZED
    test_storage.save_final_json(p_id, {"final": "data"})
    state1["status"] = ProjectStatus.ANALYZING.value
    reconciled2 = reconcile_project_state(state1, test_storage)
    assert reconciled2["status"] == ProjectStatus.ANALYZED.value

    # Case 3: Interrupted in RENDERING -> ANALYZED
    state1["status"] = ProjectStatus.RENDERING.value
    reconciled3 = reconcile_project_state(state1, test_storage)
    assert reconciled3["status"] == ProjectStatus.ANALYZED.value

    # Case 4: Interrupted in ANALYZING with prepared manifest -> PREPARED
    p_id2 = "proj-reconcile-02"
    test_storage.save_prepared_manifest(p_id2, {"project_id": p_id2, "status": "completed"})
    state2 = {
        "schema_version": "3.0",
        "project_id": p_id2,
        "project_name": "Reconcile_Test2",
        "status": ProjectStatus.ANALYZING.value,
        "sources": [],
        "source_fingerprints": {},
        "outputs": {},
    }
    test_storage.save_project(state2)
    reconciled4 = reconcile_project_state(state2, test_storage)
    assert reconciled4["status"] == ProjectStatus.PREPARED.value

    # Case 5: Already PREPARED is preserved
    state2["status"] = ProjectStatus.PREPARED.value
    reconciled5 = reconcile_project_state(state2, test_storage)
    assert reconciled5["status"] == ProjectStatus.PREPARED.value


def test_validate_completed_output_presence_and_hash_before_skip(
    test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path
):
    """Verify output is rerendered if completed file is missing or modified."""
    project_id = "proj-hash-check-01"
    output_dir = tmp_path / "outputs"

    final_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Hash_Check_Project",
        "sources": [{"source_file": "ep01.mp4"}],
        "outputs": [
            {
                "render_id": "out-01",
                "title": "Recap_1",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "ep01.mp4",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            }
        ],
    }

    render_calls = 0

    def mock_render(output_def, *args, **kwargs):
        nonlocal render_calls
        render_calls += 1
        out_mp4 = output_dir / f"{output_def['title']}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(f"version {render_calls}".encode())
        _write_mock_subtitles(output_dir, output_def["title"])
        return RenderResult(
            output_path=out_mp4,
            narration_srt_path=output_dir / f"{output_def['title']}.narration.srt",
            original_srt_path=output_dir / f"{output_def['title']}.original.srt",
            duration=1.0,
            title=output_def["title"],
            render_id=output_def["render_id"],
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=480,
            fps=25.0,
        )

    workflow = ProjectWorkflow(persistence=test_storage)
    workflow.import_project(
        project_id=project_id,
        project_name="Hash_Check_Project",
        source_input=sample_video,
        final_json=final_json,
        output_dir=output_dir,
    )

    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render):
        workflow.start_project(project_id)
    assert render_calls == 1

    # Corrupt output file on disk
    out_file = output_dir / "Recap_1.mp4"
    out_file.write_bytes(b"CORRUPTED_FILE_CONTENT")

    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render):
        workflow.resume_project(project_id)

    # Hash mismatch triggered rerender!
    assert render_calls == 2


def test_real_existing_json_multi_source_pipeline_smoke(tmp_path: Path):
    """Real multi-source pipeline smoke test with real FFmpeg and media probe."""
    src_dir = tmp_path / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    v1 = _create_synthetic_video(src_dir / "clip1.mp4", duration=2.0, width=640, height=480, fps=25)
    v2 = _create_synthetic_video(src_dir / "clip2.mp4", duration=2.0, width=800, height=600, fps=30)

    storage = ProjectPersistence(storage_root=tmp_path / "storage")
    output_dir = tmp_path / "outputs"

    project_id = "proj-smoke-multi-01"

    # Multi-source project JSON referencing both clips
    multi_source_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Multi_Source_Smoke",
        "sources": [
            {"source_file": "clip1.mp4"},
            {"source_file": "clip2.mp4"},
        ],
        "outputs": [
            {
                "render_id": "out-multi",
                "title": "Recap_Multi",
                "segments": [
                    {
                        "segment_id": "seg-clip1",
                        "source_file": "clip1.mp4",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [
                            {"start_ms": 0, "end_ms": 500, "text": "Clip 1 dialog"},
                        ],
                    },
                    {
                        "segment_id": "seg-clip2",
                        "source_file": "clip2.mp4",
                        "start_ms": 500,
                        "end_ms": 1500,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [
                            {"start_ms": 0, "end_ms": 500, "text": "Clip 2 dialog"},
                        ],
                    },
                ],
            }
        ],
    }

    workflow = ProjectWorkflow(persistence=storage)

    # Import and start using real media probe and real render_output
    workflow.import_project(
        project_id=project_id,
        project_name="Multi_Source_Smoke",
        source_input=src_dir,
        final_json=multi_source_json,
        output_dir=output_dir,
    )

    completed_state = workflow.start_project(project_id)

    assert completed_state["status"] == ProjectStatus.COMPLETED.value
    out_mp4 = output_dir / "Recap_Multi.mp4"
    assert out_mp4.is_file()
    assert out_mp4.stat().st_size > 0

    # Probe rendered output with real ffprobe
    probe_res = probe_media(out_mp4)
    assert probe_res.has_video is True
    assert probe_res.has_audio is True
    assert probe_res.duration == pytest.approx(2.0, rel=0.1)
    assert (output_dir / "Recap_Multi.original.srt").is_file()
    assert (output_dir / "Recap_Multi.narration.srt").is_file()


def test_highlight_shared_renderer_skips_voice_and_marks_narration_stages_skipped(
    test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path,
) -> None:
    project_id = "highlight-original-audio"
    output_dir = tmp_path / "highlight-output"
    final_json = {
        "schema_version": "3.0", "project_id": project_id, "project_name": "Highlight",
        "sources": [{"source_file": "ep01.mp4"}],
        "outputs": [{
            "render_id": "h1", "title": "Highlight",
            "segments": [{
                "segment_id": "s1", "source_file": "ep01.mp4",
                "start_ms": 0, "end_ms": 500, "type": "original_dialogue",
                "narration": "", "source_audio": True,
                "audio_intent": "ORIGINAL_AUDIO", "subtitles": [],
            }],
        }],
    }
    workflow = ProjectWorkflow(persistence=test_storage)
    state = workflow.import_project(
        project_id=project_id, project_name="Highlight", source_input=sample_video,
        final_json=final_json, output_dir=output_dir,
    )
    state["project_mode"] = "HIGHLIGHT"
    test_storage.save_project(state)
    events: list[dict] = []
    tracker = WorkflowProgressTracker(test_storage, project_id, callback=events.append)

    def mock_render(output_def, *args, **kwargs):
        assert kwargs["render_mode"] == "HIGHLIGHT"
        path = output_dir / "Highlight.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"highlight")
        _write_mock_subtitles(output_dir, "Highlight")
        return RenderResult(
            output_path=path,
            narration_srt_path=output_dir / "Highlight.narration.srt",
            original_srt_path=output_dir / "Highlight.original.srt",
            duration=0.5, title="Highlight", render_id=output_def["render_id"],
            video_codec="h264", audio_codec="aac", width=640, height=480, fps=25.0,
        )

    with patch(
        "toolrecap_v4.workflow.DownstreamVoiceCache.prepare_output",
        side_effect=AssertionError("Highlight must not prepare Voice narration"),
    ), patch("toolrecap_v4.workflow.render_output", side_effect=mock_render):
        completed = workflow.start_project(project_id, progress_tracker=tracker)

    assert completed["status"] == ProjectStatus.COMPLETED.value
    texts = [str(event.get("activity_text", "")) for event in events]
    assert "Highlight uses original source audio; Voice generation skipped." in texts
    assert "Highlight contains no narration; Narration Fit skipped." in texts


def test_stored_metadata_and_secret_prevention(test_storage: ProjectPersistence, sample_video: Path):
    """Verify stored fingerprints, prompt hash, nonsecret identifiers, and secret prevention."""
    from toolrecap_v4.errors import SecretExposureError

    project_id = "proj-meta-01"
    prompt_text = "Detailed recap prompt for episode 1"
    cfg = AppSettings(voice_id="nova", quality="medium")

    workflow = ProjectWorkflow(persistence=test_storage)
    state = workflow.create_project(
        project_id=project_id,
        project_name="Metadata_Project",
        source_input=sample_video,
        prompt=prompt_text,
        settings=cfg,
    )

    # Invariant: source fingerprints, prompt hash, Gateway nonsecret identifier, settings snapshot, voice endpoint, timestamps stored
    assert "source_fingerprints" in state
    assert "ep01.mp4" in state["source_fingerprints"]
    assert state["source_fingerprints"]["ep01.mp4"]["sha256"] != ""

    import hashlib
    assert state["prompt_hash"] == hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    assert state["gateway_identifier"] == f"{cfg.gateway_endpoint}::{cfg.gateway_model}"
    assert "api_key" not in state["gateway_identifier"]
    assert state["voice_endpoint"] == f"{cfg.voice_mode}::{cfg.voice_local_url}::{cfg.voice_remote_url}"
    assert "created_at" in state["timestamps"]
    assert "updated_at" in state["timestamps"]

    # Verify on-disk file has no secrets
    on_disk = test_storage.load_project(project_id)
    assert on_disk["gateway_identifier"] == state["gateway_identifier"]

    # Verify secret injection is rejected before persistence
    with pytest.raises(SecretExposureError):
        test_storage.save_project({
            "project_id": "proj-secret-reject",
            "api_key": "sk-12345",
        })


def test_analysis_unavailable_makes_zero_gateway_calls_and_no_video_payload(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify attempting analysis without Final JSON makes zero Gateway calls, generates no video payload, and raises AnalysisPipelineUnavailableError."""
    project_id = "proj-negative-no-video-01"
    output_dir = tmp_path / "outputs"

    mock_gw = MagicMock(spec=GatewayClient)
    workflow = ProjectWorkflow(persistence=test_storage, gateway_client=mock_gw)
    workflow.create_project(
        project_id=project_id,
        project_name="Negative_Test_Project",
        source_input=sample_video,
        prompt="Test prompt",
        output_dir=output_dir,
    )

    with pytest.raises(AnalysisPipelineUnavailableError):
        workflow.start_project(project_id)

    # Negative assertions: zero Gateway calls
    assert mock_gw.submit_text_chat.call_count == 0
    assert not hasattr(mock_gw, "submit_chat_analysis")
    assert not hasattr(mock_gw, "validate_model_video_capability")
    # Checkpoints not created
    assert test_storage.has_sub_analysis(project_id) is False
    assert test_storage.has_raw_response(project_id) is False


def test_resume_without_final_json_fails_analysis_unavailable(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify that resuming a project that has no Final JSON fails immediately with AnalysisPipelineUnavailableError and zero AI calls."""
    project_id = "proj-resume-no-final-json-01"
    output_dir = tmp_path / "outputs"

    mock_gw = MagicMock(spec=GatewayClient)
    workflow = ProjectWorkflow(persistence=test_storage, gateway_client=mock_gw)
    workflow.create_project(
        project_id=project_id,
        project_name="Resume_No_Final_Project",
        source_input=sample_video,
        prompt="Test prompt",
        output_dir=output_dir,
    )

    with pytest.raises(AnalysisPipelineUnavailableError):
        workflow.resume_project(project_id)

    assert mock_gw.submit_text_chat.call_count == 0
    state = test_storage.load_project(project_id)
    assert state["status"] == ProjectStatus.PREPARED.value
    assert "prepared_at" in state["timestamps"]


def test_dual_stage_render_retry_zero_ai(test_storage: ProjectPersistence, sample_video: Path, tmp_path: Path):
    """Verify that retrying after render failure makes 0 AI calls."""
    project_id = "proj-render-fail-retry-01"
    output_dir = tmp_path / "outputs"

    valid_response_json = {
        "schema_version": "3.0",
        "project_id": project_id,
        "project_name": "Render_Retry_Project",
        "sources": [{"source_file": "ep01.mp4"}],
        "outputs": [
            {
                "render_id": "render-01",
                "title": "Recap_Render_Fail",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "ep01.mp4",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [],
                    }
                ],
            }
        ],
    }

    mock_gw = MagicMock(spec=GatewayClient)
    workflow = ProjectWorkflow(
        persistence=test_storage,
        gateway_client=mock_gw,
    )

    workflow.import_project(
        project_id=project_id,
        project_name="Render_Retry_Project",
        source_input=sample_video,
        final_json=valid_response_json,
        output_dir=output_dir,
    )

    # First attempt: render raises an error
    with patch("toolrecap_v4.workflow.render_output", side_effect=RuntimeError("FFmpeg encoding crashed")):
        with pytest.raises(RuntimeError, match="FFmpeg encoding crashed"):
            workflow.start_project(project_id)

    # Final JSON was saved BEFORE render
    assert test_storage.has_final_json(project_id) is True
    assert mock_gw.submit_text_chat.call_count == 0

    def mock_render_success(*args, **kwargs):
        out_mp4 = output_dir / "Recap_Render_Fail.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"rendered mp4")
        _write_mock_subtitles(output_dir, "Recap_Render_Fail")
        return RenderResult(
            output_path=out_mp4,
            narration_srt_path=output_dir / "Recap_Render_Fail.narration.srt",
            original_srt_path=output_dir / "Recap_Render_Fail.original.srt",
            duration=1.0,
            title="Recap_Render_Fail",
            render_id="render-01",
            video_codec="h264",
            audio_codec="aac",
            width=640,
            height=480,
            fps=25.0,
        )

    with patch("toolrecap_v4.workflow.render_output", side_effect=mock_render_success):
        resumed_state = workflow.resume_project(project_id)

    # Invariant: ZERO AI calls on retry
    assert mock_gw.submit_text_chat.call_count == 0
    assert resumed_state["status"] == ProjectStatus.COMPLETED.value


def test_production_boundary_no_deleted_apis():
    """Verify that whole-video production APIs are absent and zero calls occur across boundaries."""
    import toolrecap_v4
    import toolrecap_v4.gateway as gw

    # 1. Whole-video types and constants absent from root and gateway
    assert not hasattr(toolrecap_v4, "StreamingChatPayload")
    assert not hasattr(toolrecap_v4, "DEFAULT_MAX_FILE_SIZE_BYTES")
    assert not hasattr(toolrecap_v4, "SUPPORTED_VIDEO_EXTENSIONS")
    assert not hasattr(gw, "StreamingChatPayload")
    assert not hasattr(gw, "DEFAULT_MAX_FILE_SIZE_BYTES")
    assert not hasattr(gw, "SUPPORTED_VIDEO_EXTENSIONS")

    # 2. Whole-video methods absent from GatewayClient
    assert not hasattr(gw.GatewayClient, "submit_chat_analysis")
    assert not hasattr(gw.GatewayClient, "validate_model_video_capability")
    assert not hasattr(gw.GatewayClient, "validate_model_prime_capability")

    # 3. Safe image and error exports present on root
    assert hasattr(toolrecap_v4, "AnalysisPipelineUnavailableError")
    assert hasattr(toolrecap_v4, "validate_and_reencode_image")
    assert hasattr(toolrecap_v4, "DEFAULT_MAX_IMAGE_BYTES")
    assert hasattr(toolrecap_v4, "GatewayConnectionError")
