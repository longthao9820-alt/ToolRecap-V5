from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from toolrecap_v4.discovery import SourceFingerprint
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.highlight.service import HighlightWorkflow
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.progress import ActivityState, WorkflowProgressTracker
from toolrecap_v4.ui.main_window import MainWindow


class Clock:
    def __init__(self) -> None: self.mono = 100.0; self.wall = 1_800_000_000.0
    def monotonic(self) -> float: return self.mono
    def wall_clock(self) -> float: return self.wall
    def advance(self, seconds: float) -> None: self.mono += seconds; self.wall += seconds


def _state(root: Path, project_id: str = "highlight-test") -> ProjectPersistence:
    persistence = ProjectPersistence(root)
    persistence.save_project({
        "project_id": project_id, "project_name": "Highlight", "project_mode": "HIGHLIGHT",
        "status": "created", "sources": [], "outputs": {}, "timestamps": {},
    })
    return persistence


def test_highlight_uses_shared_tracker_for_stage_session_project_last_activity_and_ai_wait(tmp_path: Path) -> None:
    persistence = _state(tmp_path)
    clock = Clock(); snapshots = []
    tracker = WorkflowProgressTracker(
        persistence, "highlight-test", callback=snapshots.append,
        monotonic=clock.monotonic, wall_clock=clock.wall_clock,
    )
    workflow = HighlightWorkflow(
        persistence=persistence, gateway_client=MagicMock(spec=GatewayClient),
        tracker=tracker, progress_callback=snapshots.append,
    )
    workflow.start_tracking("highlight-test", resuming=False)
    clock.advance(65)
    workflow._emit("Preparing Sources", "Source preparation completed.", 1, 1, current_item="E01")
    prep = tracker.tick()
    assert prep["stage_label"] == "Preparing Sources"
    assert prep["stage_elapsed_seconds"] == 65
    assert prep["session_elapsed_seconds"] == prep["project_elapsed_seconds"] == 65
    assert prep["recent_activity"] and prep["last_activity_at"]

    workflow._emit("Scanning Evidence", "Waiting for Scanner response...", state=ActivityState.WAITING_FOR_AI.value, waiting_for="AI")
    assert tracker.tick()["stage_elapsed_seconds"] == 0
    waiting_timestamp = tracker.tick()["last_activity_at"]
    clock.advance(20)
    assert tracker.tick()["last_activity_at"] == waiting_timestamp  # timer refresh is not activity
    workflow._emit("Scanning Evidence", "Scanner response received.", state=ActivityState.LOCAL_PROCESSING.value)
    assert tracker.tick()["state"] == ActivityState.LOCAL_PROCESSING.value

    clock.advance(10)
    tracker.terminate(ActivityState.CANCELLED, activity_text="Highlight project cancelled.")
    accumulated = tracker.tick()["project_elapsed_seconds"]
    resumed = WorkflowProgressTracker(
        persistence, "highlight-test", monotonic=clock.monotonic, wall_clock=clock.wall_clock,
    )
    resumed.begin(resuming=True); clock.advance(15)
    assert resumed.tick()["project_elapsed_seconds"] == accumulated + 15


def _source(path: Path) -> SourceFingerprint:
    path.write_bytes(b"synthetic")
    stat = path.stat()
    return SourceFingerprint(path.name, str(path), stat.st_size, stat.st_mtime_ns, "hash", path.suffix)


def test_recap_and_highlight_start_feedback_is_immediate_before_worker_work(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path / "state")
    source = _source(tmp_path / "episode.mp4")
    app = MainWindow(persistence=persistence)
    try:
        app.discovered_sources = [source]
        app.recap_prompt_text.insert("1.0", "Recap prompt")
        with patch.object(app.worker, "start_new_project") as start:
            app._on_start_project()
            start.assert_called_once()
            assert app.activity_state_var.get() == ActivityState.RUNNING.value
            assert app.activity_stage_var.get() == "Preparing Sources"
            assert "Preparing" in app.activity_text_var.get()
            assert "Project started" in app.activity_log.get("1.0", "end")
            assert str(app.btn_start.cget("state")) == "disabled"

        app._set_running_state(False)
        app.highlight_sources = [source]
        app.highlight_prompt_text.insert("1.0", "Highlight prompt")
        with patch.object(app.worker, "start_highlight_project") as start:
            app._on_start_highlight()
            start.assert_called_once()
            assert app.activity_state_var.get() == ActivityState.RUNNING.value
            assert app.activity_stage_var.get() == "Preparing Sources"
            assert "Highlight" in app.activity_text_var.get()
            assert "Highlight project started" in app.activity_log.get("1.0", "end")
            assert str(app.btn_start_highlight.cget("state")) == "disabled"
        app.update_idletasks()  # UI remains responsive without waiting for preparation
    finally:
        app.destroy()


def test_recap_and_highlight_resume_feedback_is_immediate(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path)
    for project_id, mode in (("recap", "RECAP"), ("highlight", "HIGHLIGHT")):
        persistence.save_project({
            "project_id": project_id, "project_name": project_id, "project_mode": mode,
            "status": "failed", "sources": [], "outputs": {}, "timestamps": {},
        })
    app = MainWindow(persistence=persistence)
    try:
        with patch.object(app.worker, "resume_project") as resume:
            app._on_resume_current("RECAP")
            resume.assert_called_once()
            assert app.activity_state_var.get() == ActivityState.RESUMING.value
            assert "Resume requested" in app.activity_log.get("1.0", "end")
        app._set_running_state(False)
        with patch.object(app.worker, "resume_project") as resume:
            app._on_resume_current("HIGHLIGHT")
            resume.assert_called_once()
            assert app.activity_state_var.get() == ActivityState.RESUMING.value
            assert "Resume requested" in app.activity_log.get("1.0", "end")
    finally:
        app.destroy()
