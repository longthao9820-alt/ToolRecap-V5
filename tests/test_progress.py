"""Deterministic tests for real-time workflow status, timing, ETA and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolrecap_v4.persistence import ProjectPersistence, atomic_write_json
from toolrecap_v4.progress import (
    ActivityState,
    ProgressEvent,
    WorkflowProgressTracker,
    WorkflowStage,
    format_duration,
    reconstruct_project_progress,
)


class FakeClock:
    def __init__(self) -> None:
        self.monotonic_value = 100.0
        self.wall_value = 1_800_000_000.0

    def monotonic(self) -> float:
        return self.monotonic_value

    def wall(self) -> float:
        return self.wall_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.wall_value += seconds


@pytest.mark.parametrize(("minutes", "expected"), [
    (0, "0 minutes"),
    (8, "8 minutes"),
    (59, "59 minutes"),
    (60, "1 hour 00 minutes"),
    (61, "1 hour 01 minutes"),
    (126, "2 hours 06 minutes"),
])
def test_duration_format(minutes: int, expected: str) -> None:
    assert format_duration(minutes * 60) == expected


def _project(persistence: ProjectPersistence, project_id: str = "p1", status: str = "created") -> None:
    persistence.save_project({
        "project_id": project_id,
        "project_name": project_id,
        "status": status,
        "sources": [],
        "source_fingerprints": {},
        "prepared_episodes": {},
        "outputs": {},
        "timestamps": {},
    })


def test_states_stage_timing_analysis_downstream_and_completion(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path)
    _project(persistence)
    clock = FakeClock()
    emitted: list[dict] = []
    tracker = WorkflowProgressTracker(
        persistence, "p1", callback=emitted.append,
        monotonic=clock.monotonic, wall_clock=clock.wall,
    )
    assert tracker.begin(resuming=False)["state"] == ActivityState.STARTING.value
    tracker.emit(ProgressEvent(
        state=ActivityState.RUNNING.value, stage=WorkflowStage.PREPARATION.value,
        activity_text="Preparing", completed=0, total=2, unit="episodes",
    ))
    clock.advance(59 * 60)
    assert tracker.tick()["session_elapsed_seconds"] == 59 * 60
    tracker.emit(ProgressEvent(
        state=ActivityState.WAITING_FOR_AI.value, stage=WorkflowStage.SCANNER.value,
        activity_text="Waiting", chunk_id="E01-CH-001", completed=0, total=2, unit="chunks",
        item_event="start", item_key="E01-CH-001",
    ))
    clock.advance(60)
    first = tracker.emit(ProgressEvent(
        state=ActivityState.RUNNING.value, stage=WorkflowStage.SCANNER.value,
        activity_text="Complete", chunk_id="E01-CH-001", completed=1, total=2, unit="chunks",
        item_event="complete", item_key="E01-CH-001",
    ))
    assert first["estimated_remaining_seconds"] == pytest.approx(60)
    tracker.emit(ProgressEvent(
        state=ActivityState.RETRYING.value, stage=WorkflowStage.SCANNER.value,
        activity_text="Retry", retry_attempt=1, retry_limit=2,
    ))
    tracker.emit(ProgressEvent(
        state=ActivityState.REPAIRING.value, stage=WorkflowStage.SCANNER.value,
        activity_text="Repair", retry_attempt=1, retry_limit=2,
    ))
    clock.advance(60)
    tracker.emit(ProgressEvent(
        state=ActivityState.LOCAL_PROCESSING.value, stage=WorkflowStage.FINAL_JSON.value,
        activity_text="Final JSON Ready", completed=2, total=2, unit="outputs",
        analysis_complete=True, stage_status="complete",
    ))
    analysis = tracker.tick()["analysis_seconds"]
    clock.advance(5 * 60)
    final = tracker.terminate(
        ActivityState.COMPLETE, activity_text="Done", published_count=2,
        output_folder="D:/Outputs",
    )
    assert analysis == 61 * 60
    assert final["completion_seconds"] == 66 * 60
    assert final["downstream_seconds"] == 5 * 60
    assert final["published_count"] == 2
    assert final["state"] == ActivityState.COMPLETE.value
    assert len(final["recent_activity"]) <= 12
    assert persistence.has_operational_status("p1")
    assert emitted[-1]["state"] == ActivityState.COMPLETE.value


def test_restart_excludes_closed_time_and_accumulates_sessions(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path)
    _project(persistence)
    clock = FakeClock()
    first = WorkflowProgressTracker(
        persistence, "p1", monotonic=clock.monotonic, wall_clock=clock.wall,
    )
    first.begin(resuming=False)
    first.emit(ProgressEvent(
        state=ActivityState.RUNNING.value, stage=WorkflowStage.SCANNER.value,
        activity_text="Run",
    ))
    clock.advance(30 * 60)
    failed = first.terminate(ActivityState.FAILED, activity_text="Gateway failed", error="empty stream")
    assert failed["project_elapsed_seconds"] == 30 * 60

    clock.advance(10 * 60 * 60)  # application closed; this must never count
    second = WorkflowProgressTracker(
        ProjectPersistence(tmp_path), "p1", monotonic=clock.monotonic, wall_clock=clock.wall,
    )
    resumed = second.begin(resuming=True)
    assert resumed["state"] == ActivityState.RESUMING.value
    assert resumed["project_elapsed_seconds"] == 30 * 60
    assert resumed["session_elapsed_seconds"] == 0
    clock.advance(31 * 60)
    cancelled = second.terminate(ActivityState.CANCELLED, activity_text="Cancelled")
    assert cancelled["project_elapsed_seconds"] == 61 * 60
    assert cancelled["session_elapsed_seconds"] == 31 * 60


def test_eta_unknown_smoothing_and_cached_work_not_sampled(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path)
    _project(persistence)
    clock = FakeClock()
    tracker = WorkflowProgressTracker(
        persistence, "p1", monotonic=clock.monotonic, wall_clock=clock.wall,
    )
    tracker.begin(resuming=False)
    unknown = tracker.emit(ProgressEvent(
        state=ActivityState.WAITING_FOR_AI.value, stage=WorkflowStage.PLANNER.value,
        activity_text="Planner", completed=None, total=None,
    ))
    assert unknown["estimated_remaining_seconds"] is None
    tracker.emit(ProgressEvent(
        state=ActivityState.RUNNING.value, stage=WorkflowStage.WRITERS.value,
        activity_text="cached", current_item="out_001", completed=1, total=4,
        item_event="complete", item_key="out_001", item_reused=True,
    ))
    assert tracker.tick()["estimated_remaining_seconds"] is None
    tracker.emit(ProgressEvent(
        state=ActivityState.WAITING_FOR_AI.value, stage=WorkflowStage.WRITERS.value,
        activity_text="start", current_item="out_002", completed=1, total=4,
        item_event="start", item_key="out_002",
    ))
    clock.advance(100)
    first = tracker.emit(ProgressEvent(
        state=ActivityState.RUNNING.value, stage=WorkflowStage.WRITERS.value,
        activity_text="done", current_item="out_002", completed=2, total=4,
        item_event="complete", item_key="out_002",
    ))
    assert first["estimated_remaining_seconds"] == pytest.approx(200)
    tracker.emit(ProgressEvent(
        state=ActivityState.WAITING_FOR_AI.value, stage=WorkflowStage.WRITERS.value,
        activity_text="start", current_item="out_003", completed=2, total=4,
        item_event="start", item_key="out_003",
    ))
    clock.advance(300)
    smoothed = tracker.emit(ProgressEvent(
        state=ActivityState.RUNNING.value, stage=WorkflowStage.WRITERS.value,
        activity_text="done", current_item="out_003", completed=3, total=4,
        item_event="complete", item_key="out_003",
    ))
    assert smoothed["estimated_remaining_seconds"] == pytest.approx(200)


def test_failure_retains_progress_and_gateway_identity(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path)
    _project(persistence)
    clock = FakeClock()
    tracker = WorkflowProgressTracker(
        persistence, "p1", monotonic=clock.monotonic, wall_clock=clock.wall,
    )
    tracker.begin(resuming=False)
    tracker.emit(ProgressEvent(
        state=ActivityState.WAITING_FOR_AI.value, stage=WorkflowStage.SCANNER.value,
        activity_text="Waiting for Scanner response...", chunk_id="E09-CH-005",
        completed=83, total=96, unit="chunks",
    ))
    clock.advance(24 * 60)
    failed = tracker.terminate(
        ActivityState.FAILED,
        activity_text="Gateway connection failed for E09-CH-005.",
        error="Gateway streaming response completed with empty content.",
    )
    assert failed["stage"] == WorkflowStage.SCANNER.value
    assert failed["current_item"] == "E09-CH-005"
    assert failed["completed"] == 83 and failed["total"] == 96
    assert failed["state"] == ActivityState.FAILED.value
    assert "Gateway" in failed["error"]
    assert "repair exhausted" not in failed["error"].lower()


def test_old_project_without_timing_is_compatible(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path)
    _project(persistence, status="catalog_ready")
    snapshot = reconstruct_project_progress(persistence, "p1")
    assert snapshot["stage"] == WorkflowStage.PLANNER.value
    assert snapshot["project_elapsed_seconds"] is None
    assert snapshot["previous_timing_available"] is False


@pytest.mark.parametrize("state", [
    ActivityState.IDLE, ActivityState.STARTING, ActivityState.RUNNING,
    ActivityState.RESUMING, ActivityState.WAITING_FOR_AI,
    ActivityState.RETRYING, ActivityState.REPAIRING,
    ActivityState.FAILED, ActivityState.COMPLETE,
])
def test_required_top_level_states_round_trip(tmp_path: Path, state: ActivityState) -> None:
    persistence = ProjectPersistence(tmp_path / state.name.lower())
    _project(persistence)
    tracker = WorkflowProgressTracker(persistence, "p1")
    snapshot = tracker.emit(ProgressEvent(
        state=state.value, stage=WorkflowStage.PLANNER.value,
        activity_text=state.value,
    ))
    assert snapshot["state"] == state.value
    assert persistence.load_operational_status("p1")["state"] == state.value


def test_reconstructs_catalog_plan_writer_and_final_json_checkpoints(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path)
    persistence.save_project({
        "project_id": "p1", "project_name": "p1", "status": "cancelled",
        "sources": [], "source_fingerprints": {}, "prepared_episodes": {}, "outputs": {},
        "final_json": {"outputs": [{"render_id": "render-1"}, {"render_id": "render-2"}]},
        "timestamps": {}, "error": "Writer interrupted",
    })
    for checkpoint in ("catalog", "planner_draft", "season_plan", "final_json"):
        persistence.save_checkpoint("p1", checkpoint, {"status": "completed"})
    writer = persistence.projects_dir / "p1" / "writers" / "plan"
    writer.mkdir(parents=True)
    atomic_write_json(writer / "manifest.json", {
        "status": "INCOMPLETE", "expected_output_count": 6,
        "completed_response_count": 2, "failures": {"out_003": "transport"},
    })
    snapshot = reconstruct_project_progress(persistence, "p1")
    assert snapshot["stage"] == WorkflowStage.WRITERS.value
    assert (snapshot["completed"], snapshot["total"], snapshot["reused"]) == (2, 6, 2)
    assert snapshot["current_item"] == "out_003"
    assert snapshot["pipeline"][WorkflowStage.CATALOG.value] == "complete"
    assert snapshot["pipeline"][WorkflowStage.PLANNER.value] == "complete"
    assert snapshot["pipeline"][WorkflowStage.FINAL_PLAN.value] == "failed"
    assert snapshot["pipeline"][WorkflowStage.FINAL_JSON.value] == "complete"
