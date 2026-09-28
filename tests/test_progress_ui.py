"""Tk status panel rendering tests; all workflow work remains mocked."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.progress import ActivityState, PIPELINE_STAGES
from toolrecap_v4.ui.main_window import MainWindow
from toolrecap_v4.ui.worker import WorkerMessage


def _snapshot(**overrides):
    value = {
        "project_id": "p1",
        "state": ActivityState.RUNNING.value,
        "stage": "scanner",
        "stage_label": "Scanning Episode Evidence",
        "activity_text": "Scanner chunk completed.",
        "active": True,
        "current_item": "E09-CH-005",
        "completed": 83,
        "total": 96,
        "unit": "chunks",
        "reused": 83,
        "session_elapsed_seconds": 18 * 60,
        "project_elapsed_seconds": 72 * 60,
        "stage_elapsed_seconds": 18 * 60,
        "estimated_remaining_seconds": 18 * 60,
        "last_activity_at": datetime.now(timezone.utc).isoformat(),
        "recent_activity": [
            {"timestamp": datetime.now(timezone.utc).isoformat(), "text": "E09-CH-004 completed"},
            {"timestamp": datetime.now(timezone.utc).isoformat(), "text": "E09-CH-005 started"},
        ],
        "pipeline": {stage: ("complete" if stage == "preparation" else "active" if stage == "scanner" else "pending") for stage in PIPELINE_STAGES},
    }
    value.update(overrides)
    return value


def test_status_panel_is_visible_and_renders_real_scanner_progress(tmp_path: Path) -> None:
    app = MainWindow(persistence=ProjectPersistence(tmp_path))
    try:
        app.current_project_id = "p1"
        app._handle_worker_message(WorkerMessage("activity", _snapshot()))
        app.update_idletasks()
        app.update()
        assert app.activity_panel.winfo_ismapped()
        assert app.activity_state_var.get() == ActivityState.RUNNING.value
        assert app.activity_stage_var.get() == "Scanning Episode Evidence"
        assert app.activity_item_var.get() == "E09-CH-005"
        assert app.activity_progress_var.get() == "83 / 96 chunks — 86% (reused: 83)"
        assert str(app.progressbar.cget("mode")) == "determinate"
        assert app.activity_stage_time_var.get() == "18 minutes"
        assert app.activity_project_time_var.get() == "1 hour 12 minutes"
        assert app.activity_eta_var.get() == "18 minutes"
        assert "E09-CH-004 completed" in app.activity_log.get("1.0", "end")
        assert "● Scanner" in app.pipeline_var.get()
    finally:
        app.destroy()


def test_indeterminate_wait_repair_failure_and_completion_summary(tmp_path: Path) -> None:
    app = MainWindow(persistence=ProjectPersistence(tmp_path))
    try:
        app.current_project_id = "p1"
        waiting = _snapshot(
            state=ActivityState.WAITING_FOR_AI.value,
            stage="planner", stage_label="Planning Episode and Season Stories",
            current_item="round-001", completed=None, total=None,
            estimated_remaining_seconds=None,
            activity_text="Waiting for Planner response...",
        )
        app._handle_worker_message(WorkerMessage("activity", waiting))
        app.update_idletasks()
        app.update()
        assert not app.progressbar.winfo_ismapped()
        assert app.activity_dots_label.winfo_ismapped()
        assert "total not yet known" in app.activity_progress_var.get()
        assert app.activity_eta_var.get() == "Unknown"

        app._handle_worker_message(WorkerMessage("activity_patch", {
            "state": ActivityState.REPAIRING.value,
            "activity_text": "Repairing out_003 — attempt 1 / 2",
            "retry_attempt": 1, "retry_limit": 2,
        }))
        assert app.activity_state_var.get() == ActivityState.REPAIRING.value
        assert "out_003" in app.activity_text_var.get()

        app._handle_worker_message(WorkerMessage("activity_patch", {
            "state": ActivityState.FAILED.value, "active": False,
            "error": "Gateway empty stream for E09-CH-005",
        }))
        assert app.activity_state_var.get() == ActivityState.FAILED.value
        assert "Gateway empty stream" in app.activity_summary_var.get()
        assert app.activity_item_var.get() == "round-001"  # failure preserves prior progress/item

        app._handle_worker_message(WorkerMessage("activity", _snapshot(
            state=ActivityState.COMPLETE.value, active=False,
            stage="publish", stage_label="Publishing Output",
            current_item=None, completed=6, total=6, reused=0, unit="outputs",
            analysis_seconds=84 * 60, downstream_seconds=38 * 60,
            completion_seconds=122 * 60, published_count=6,
            output_folder="D:/Outputs_S02",
        )))
        summary = app.activity_summary_var.get()
        assert app.activity_state_var.get() == ActivityState.COMPLETE.value
        assert "Analysis Time: 1 hour 24 minutes" in summary
        assert "Downstream Time: 38 minutes" in summary
        assert "Completion Time: 2 hours 02 minutes" in summary
        assert "Videos Published: 6" in summary
        assert "D:/Outputs_S02" in summary
    finally:
        app.destroy()


def test_zero_visual_and_voice_cache_reuse_labels_are_truthful(tmp_path: Path) -> None:
    app = MainWindow(persistence=ProjectPersistence(tmp_path))
    try:
        app.current_project_id = "p1"
        app._handle_worker_message(WorkerMessage("activity", _snapshot(
            state=ActivityState.LOCAL_PROCESSING.value,
            stage="vision", stage_label="Analysing Selected Visuals",
            activity_text="Visual Analysis: Not required.",
            current_item=None, completed=0, total=0, unit="requests", reused=0,
        )))
        assert app.activity_progress_var.get() == "0 / 0 requests — Not required"
        assert "Not required" in app.activity_text_var.get()

        app._handle_worker_message(WorkerMessage("activity", _snapshot(
            stage="voice", stage_label="Generating Voice",
            activity_text="Narration seg_007 reused from cache.",
            current_item="seg_007", completed=7, total=12, unit="segments", reused=7,
        )))
        assert "reused: 7" in app.activity_progress_var.get()
        assert "reused from cache" in app.activity_text_var.get()
    finally:
        app.destroy()


def test_unknown_progress_uses_three_dot_text_and_stops_without_heartbeat_activity(tmp_path: Path) -> None:
    app = MainWindow(persistence=ProjectPersistence(tmp_path))
    try:
        app.current_project_id = "p1"
        snapshot = _snapshot(
            state=ActivityState.WAITING_FOR_AI.value, stage="planner",
            stage_label="Planning Episode and Season Stories", completed=None, total=None,
            activity_text="Waiting for Planner response",
        )
        app._handle_worker_message(WorkerMessage("activity", snapshot))
        first = app.activity_dots_var.get()
        app._animate_activity_dots()
        second = app.activity_dots_var.get()
        app._animate_activity_dots()
        third = app.activity_dots_var.get()
        assert first.endswith(".")
        assert second.endswith("..")
        assert third.endswith("...")
        last_activity = app.activity_last_var.get()
        app._handle_worker_message(WorkerMessage("activity_patch", {
            "state": ActivityState.COMPLETE.value, "active": False,
            "activity_text": "Planner complete", "completed": 3, "total": 3,
        }))
        assert app.activity_dots_var.get() == ""
        assert app.activity_last_var.get() == last_activity
    finally:
        app.destroy()
