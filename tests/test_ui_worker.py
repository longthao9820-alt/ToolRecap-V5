"""Tests for background worker, thread safety, duplicate start prevention, and error sanitization."""

from pathlib import Path
import queue
import time
from unittest.mock import MagicMock, patch
import pytest

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, GatewayError
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.secrets import DPAPISecretStore
from toolrecap_v4.settings import AppSettings, SettingsManager
from toolrecap_v4.ui.worker import WorkerMessage, WorkflowWorker, format_clean_error
from toolrecap_v4.workflow import OutputStatus, ProjectStatus, ProjectWorkflow


def test_format_clean_error_redacts_secrets(tmp_path: Path):
    """Verify secrets stored in DPAPI are cleanly redacted from error strings."""
    store = DPAPISecretStore(storage_root=tmp_path)
    store.set_secret("gateway_api_key", "sk-topsecret-999888777")

    raw_err = GatewayError("Failed request with key sk-topsecret-999888777 at route /v1/chat")
    cleaned = format_clean_error(raw_err, secrets=["sk-topsecret-999888777"])

    assert "sk-topsecret-999888777" not in cleaned
    assert "[REDACTED]" in cleaned


def test_duplicate_start_prevention(tmp_path: Path):
    """Verify worker refuses duplicate start when already running."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    worker = WorkflowWorker(persistence=persistence)

    worker._is_running = True
    # Simulate an active thread
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    worker._thread = mock_thread

    with pytest.raises(RuntimeError, match="Một tác vụ đang được thực hiện"):
        worker.start_new_project(
            project_id="proj-duplicate",
            project_name="Duplicate Test",
            source_input=[],
        )

    with pytest.raises(RuntimeError, match="Một tác vụ đang được thực hiện"):
        worker.resume_project(project_id="proj-duplicate")

    with pytest.raises(RuntimeError, match="Một tác vụ đang được thực hiện"):
        worker.render_existing_json(
            project_id="proj-duplicate",
            project_name="Duplicate Test",
            source_input=[],
            final_json={},
        )


def test_worker_cancellation(tmp_path: Path):
    """Verify worker cancellation sets cancellation token."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    worker = WorkflowWorker(persistence=persistence)

    worker._cancellation_token = CancellationToken()
    assert not worker._cancellation_token.is_cancelled

    worker.cancel()
    assert worker._cancellation_token.is_cancelled

    # Check that cancel log message was queued
    msg = worker.queue.get_nowait()
    assert msg.kind == "log"
    assert "dừng" in msg.data.lower()


def test_callbacks_queue_dispatch(tmp_path: Path):
    """Verify WorkflowCallbacks put appropriate WorkerMessage items into the queue."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    q: queue.Queue[WorkerMessage] = queue.Queue()
    worker = WorkflowWorker(msg_queue=q, persistence=persistence)

    callbacks = worker._build_callbacks()

    # Test status change
    callbacks.on_status_change(ProjectStatus.ANALYZING.value)
    m1 = q.get_nowait()
    assert m1.kind == "status_change"
    assert m1.data == ProjectStatus.ANALYZING.value
    m1_log = q.get_nowait()
    assert m1_log.kind == "log"
    assert "phân tích" in m1_log.data.lower()

    # Test output completed
    callbacks.on_output_completed("render-01", {"output_path": "test.mp4", "duration": 12.5})
    m2 = q.get_nowait()
    assert m2.kind == "output_completed"
    assert m2.data["render_id"] == "render-01"
    m2_log = q.get_nowait()
    assert m2_log.kind == "log"
    assert "render-01" in m2_log.data

    # Test output skipped
    callbacks.on_output_skipped("render-02", {"output_path": "skip.mp4"})
    m3 = q.get_nowait()
    assert m3.kind == "output_skipped"
    assert m3.data["render_id"] == "render-02"
    m3_log = q.get_nowait()
    assert m3_log.kind == "log"
    assert "render-02" in m3_log.data

    # Test output failed
    callbacks.on_output_failed("render-03", "FFmpeg exited with code 1")
    m4 = q.get_nowait()
    assert m4.kind == "output_failed"
    assert m4.data["render_id"] == "render-03"
    m4_log = q.get_nowait()
    assert m4_log.kind == "log"
    assert "render-03" in m4_log.data


def test_render_existing_json_0_ai(tmp_path: Path):
    """Verify render_existing_json executes workflow with 0 AI Gateway requests."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    worker = WorkflowWorker(persistence=persistence)

    # Mock ProjectWorkflow to verify no gateway calls
    with patch.object(worker, "_create_workflow") as mock_create_wf:
        mock_wf = MagicMock()
        mock_create_wf.return_value = mock_wf

        mock_wf.import_project.return_value = {
            "project_id": "test_import",
            "status": ProjectStatus.ANALYZED.value,
        }
        mock_wf.start_project.return_value = {
            "project_id": "test_import",
            "status": ProjectStatus.COMPLETED.value,
        }

        final_json_data = {
            "schema_version": "3.0",
            "project_id": "test_import",
            "outputs": [
                {
                    "render_id": "out-1",
                    "title": "Part1",
                    "segments": [
                        {
                            "segment_id": "s1",
                            "source_file": "vid.mp4",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "type": "narration",
                            "narration": "Hello",
                            "source_audio": False,
                            "subtitles": [],
                        }
                    ],
                }
            ],
        }

        worker.render_existing_json(
            project_id="test_import",
            project_name="Part1",
            source_input=["dummy.mp4"],
            final_json=final_json_data,
        )

        # Wait for thread to finish
        worker._thread.join(timeout=3.0)

        # Ensure import_project was called (which validates and saves final JSON directly)
        mock_wf.import_project.assert_called_once()
        mock_wf.start_project.assert_called_once()

        # Check messages in queue
        msgs = []
        while not worker.queue.empty():
            msgs.append(worker.queue.get_nowait())

        finished_msg = next((m for m in msgs if m.kind == "finished"), None)
        assert finished_msg is not None
        assert finished_msg.data["status"] == ProjectStatus.COMPLETED.value
