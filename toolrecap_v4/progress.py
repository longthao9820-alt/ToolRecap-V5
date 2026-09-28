"""Thread-safe, checkpoint-backed workflow observability for ToolRecap V4.

Operational status is intentionally separate from editorial artifacts.  The tracker
persists only on meaningful workflow events; one-second UI timer refreshes are
derived in memory and never cause checkpoint writes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import enum
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping

from toolrecap_v4.persistence import ProjectPersistence

logger = logging.getLogger(__name__)

PROGRESS_SCHEMA_VERSION = "workflow-progress-v1"
ACTIVITY_HISTORY_LIMIT = 12


class ActivityState(str, enum.Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RESUMING = "RESUMING"
    RUNNING = "RUNNING"
    WAITING_FOR_AI = "WAITING FOR AI"
    LOCAL_PROCESSING = "LOCAL PROCESSING"
    RETRYING = "RETRYING"
    REPAIRING = "REPAIRING"
    CANCELLING = "CANCELLING"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"


class WorkflowStage(str, enum.Enum):
    PREPARATION = "preparation"
    AI_GATEWAY = "ai_gateway"
    VALIDATE_JSON = "validate_json"
    REVIEW = "review"
    SCANNER = "scanner"
    CATALOG = "catalog"
    PLANNER = "planner"
    EVIDENCE = "evidence"
    VISION = "vision"
    FINAL_PLAN = "final_plan"
    WRITERS = "writers"
    FINAL_JSON = "final_json"
    VOICE = "voice"
    NARRATION_FIT = "narration_fit"
    AUDIO_MIX = "audio_mix"
    SUBTITLES = "subtitles"
    RENDER = "render"
    PUBLISH = "publish"


STAGE_LABELS: dict[str, str] = {
    WorkflowStage.PREPARATION.value: "Preparing Sources",
    WorkflowStage.AI_GATEWAY.value: "AI Gateway Analysis",
    WorkflowStage.VALIDATE_JSON.value: "Validating Render JSON",
    WorkflowStage.REVIEW.value: "Review",
    WorkflowStage.SCANNER.value: "Scanning Episode Evidence",
    WorkflowStage.CATALOG.value: "Building Season Catalog",
    WorkflowStage.PLANNER.value: "Planning Episode and Season Stories",
    WorkflowStage.EVIDENCE.value: "Fetching Detailed Evidence",
    WorkflowStage.VISION.value: "Analysing Selected Visuals",
    WorkflowStage.FINAL_PLAN.value: "Finalising Season Plan",
    WorkflowStage.WRITERS.value: "Writing Recap Outputs",
    WorkflowStage.FINAL_JSON.value: "Building Final JSON",
    WorkflowStage.VOICE.value: "Generating Voice",
    WorkflowStage.NARRATION_FIT.value: "Fitting Video to Narration",
    WorkflowStage.AUDIO_MIX.value: "Mixing Audio",
    WorkflowStage.SUBTITLES.value: "Building Subtitles",
    WorkflowStage.RENDER.value: "Rendering Video",
    WorkflowStage.PUBLISH.value: "Publishing Output",
    "highlight_coverage": "Building Highlight Coverage",
    "highlight_planner": "Planning Highlight Scenes",
    "highlight_validation": "Validating Highlights",
    "highlight_json": "Building Highlight JSON",
    "highlight_render": "Rendering Highlights",
    "highlight_srt": "Building Original Dialogue SRT",
}

PIPELINE_STAGES: tuple[str, ...] = (
    WorkflowStage.PREPARATION.value,
    WorkflowStage.SCANNER.value,
    WorkflowStage.CATALOG.value,
    WorkflowStage.PLANNER.value,
    WorkflowStage.EVIDENCE.value,
    WorkflowStage.VISION.value,
    WorkflowStage.FINAL_PLAN.value,
    WorkflowStage.WRITERS.value,
    WorkflowStage.FINAL_JSON.value,
    WorkflowStage.VOICE.value,
    WorkflowStage.NARRATION_FIT.value,
    WorkflowStage.AUDIO_MIX.value,
    WorkflowStage.SUBTITLES.value,
    WorkflowStage.RENDER.value,
    WorkflowStage.PUBLISH.value,
)
V5_PIPELINE_STAGES: tuple[str, ...] = (
    WorkflowStage.PREPARATION.value,
    WorkflowStage.AI_GATEWAY.value,
    WorkflowStage.VALIDATE_JSON.value,
    WorkflowStage.REVIEW.value,
    WorkflowStage.VOICE.value,
    WorkflowStage.NARRATION_FIT.value,
    WorkflowStage.AUDIO_MIX.value,
    WorkflowStage.SUBTITLES.value,
    WorkflowStage.RENDER.value,
    WorkflowStage.PUBLISH.value,
)
ACTIVE_STATES = {
    ActivityState.STARTING.value, ActivityState.RESUMING.value,
    ActivityState.RUNNING.value, ActivityState.WAITING_FOR_AI.value,
    ActivityState.LOCAL_PROCESSING.value, ActivityState.RETRYING.value,
    ActivityState.REPAIRING.value, ActivityState.CANCELLING.value,
}


def format_duration(seconds: float | int | None) -> str:
    """Format actual/estimated duration without seconds in the main summary."""
    if seconds is None:
        return "Unavailable"
    minutes = max(0, int(float(seconds) // 60))
    if minutes < 60:
        return f"{minutes} minutes"
    hours, remaining = divmod(minutes, 60)
    unit = "hour" if hours == 1 else "hours"
    return f"{hours} {unit} {remaining:02d} minutes"


def format_wait_duration(seconds: float | int | None) -> str:
    """Format a live AI wait timer; seconds are useful only for this field."""
    if seconds is None:
        return "0 seconds"
    value = max(0, int(seconds))
    if value < 60:
        return f"{value} second" if value == 1 else f"{value} seconds"
    return format_duration(value)


def format_last_activity(seconds: float | int | None) -> str:
    if seconds is None:
        return "Unavailable"
    value = max(0, int(seconds))
    if value < 60:
        return "just now" if value < 2 else f"{value} seconds ago"
    minutes = value // 60
    return f"{minutes} minute ago" if minutes == 1 else f"{minutes} minutes ago"


def _utc_iso(wall_seconds: float) -> str:
    return datetime.fromtimestamp(wall_seconds, tz=timezone.utc).isoformat()


def _pipeline_default() -> dict[str, str]:
    return {stage: "pending" for stage in PIPELINE_STAGES}


def safe_emit(callback: Callable[[dict[str, Any]], None] | None, **payload: Any) -> None:
    """Emit an observability event without allowing UI diagnostics to break work."""
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        logger.exception("Workflow progress callback failed")


@dataclass(frozen=True)
class ProgressEvent:
    """Generic event accepted by :class:`WorkflowProgressTracker`."""

    state: str
    stage: str | None = None
    activity_text: str = ""
    episode_id: str | None = None
    chunk_id: str | None = None
    output_id: str | None = None
    current_item: str | None = None
    completed: int | None = None
    total: int | None = None
    unit: str = "items"
    reused: int | None = None
    retry_attempt: int | None = None
    retry_limit: int | None = None
    waiting_for: str | None = None
    error: str | None = None
    stage_status: str | None = None
    item_event: str | None = None
    item_key: str | None = None
    item_reused: bool = False
    analysis_complete: bool = False
    published_count: int | None = None
    output_folder: str | None = None
    pipeline_mode: str | None = None


class WorkflowProgressTracker:
    """Monotonic active-time tracker with event-based atomic persistence."""

    def __init__(
        self,
        persistence: ProjectPersistence,
        project_id: str,
        *,
        callback: Callable[[dict[str, Any]], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.persistence = persistence
        self.project_id = project_id
        self.callback = callback
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self._lock = threading.RLock()
        self._session_started: float | None = None
        self._stage_started: float | None = None
        self._stage_base = 0.0
        self._item_starts: dict[tuple[str, str], float] = {}
        self._snapshot = self._load_or_default()
        self._history = deque(self._snapshot.get("recent_activity", []), maxlen=ACTIVITY_HISTORY_LIMIT)
        self._base_accumulated = float(self._snapshot.get("project_elapsed_seconds") or 0.0)
        self._rates: dict[str, list[float]] = {
            str(stage): [float(value) for value in values][-8:]
            for stage, values in (self._snapshot.get("rate_samples") or {}).items()
            if isinstance(values, list)
        }

    def _load_or_default(self) -> dict[str, Any]:
        if self.persistence.has_operational_status(self.project_id):
            try:
                data = self.persistence.load_operational_status(self.project_id)
                if data.get("schema_version") == PROGRESS_SCHEMA_VERSION:
                    data["active"] = False
                    data["session_elapsed_seconds"] = 0.0
                    return data
            except Exception:
                logger.warning("Ignoring corrupt operational status for %s", self.project_id)
        return {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "project_id": self.project_id,
            "state": ActivityState.IDLE.value,
            "stage": None,
            "stage_label": "Not started",
            "activity_text": "Ready",
            "active": False,
            "completed": None,
            "total": None,
            "percent": None,
            "unit": "items",
            "reused": None,
            "session_elapsed_seconds": 0.0,
            "project_elapsed_seconds": None,
            "stage_elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "analysis_seconds": None,
            "downstream_seconds": None,
            "completion_seconds": None,
            "stage_durations_seconds": {},
            "pipeline": _pipeline_default(),
            "recent_activity": [],
            "last_activity_at": None,
            "event_timestamp": None,
            "previous_timing_available": False,
        }

    def begin(self, *, resuming: bool) -> dict[str, Any]:
        with self._lock:
            now = self.monotonic()
            self._base_accumulated = float(self._snapshot.get("project_elapsed_seconds") or 0.0)
            self._session_started = now
            current_stage = self._snapshot.get("stage")
            durations = self._snapshot.setdefault("stage_durations_seconds", {})
            self._stage_base = float(durations.get(current_stage, 0.0)) if current_stage else 0.0
            self._stage_started = now if current_stage else None
            state = ActivityState.RESUMING.value if resuming else ActivityState.STARTING.value
            return self.emit(ProgressEvent(
                state=state,
                stage=current_stage,
                activity_text="Reconstructing saved checkpoints..." if resuming else "Starting project...",
            ))

    def emit_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        fields = ProgressEvent.__dataclass_fields__
        return self.emit(ProgressEvent(**{key: value for key, value in payload.items() if key in fields}))

    def emit(self, event: ProgressEvent) -> dict[str, Any]:
        with self._lock:
            now_mono = self.monotonic()
            now_wall = self.wall_clock()
            old_stage = self._snapshot.get("stage")
            new_stage = event.stage or old_stage
            if new_stage != old_stage:
                self._finalize_stage(now_mono)
                self._stage_base = float(
                    self._snapshot.setdefault("stage_durations_seconds", {}).get(new_stage, 0.0)
                ) if new_stage else 0.0
                self._stage_started = now_mono if new_stage else None

            self._snapshot.update({
                "schema_version": PROGRESS_SCHEMA_VERSION,
                "project_id": self.project_id,
                "state": event.state,
                "stage": new_stage,
                "stage_label": STAGE_LABELS.get(new_stage or "", "Not started"),
                "activity_text": event.activity_text,
                "episode_id": event.episode_id,
                "chunk_id": event.chunk_id,
                "output_id": event.output_id,
                "current_item": event.current_item or event.chunk_id or event.output_id or event.episode_id,
                "completed": event.completed,
                "total": event.total,
                "unit": event.unit,
                "reused": event.reused,
                "retry_attempt": event.retry_attempt,
                "retry_limit": event.retry_limit,
                "waiting_for": event.waiting_for,
                "error": event.error,
                "active": event.state in ACTIVE_STATES,
                "last_activity_at": _utc_iso(now_wall),
                "event_timestamp": _utc_iso(now_wall),
                "previous_timing_available": True,
            })
            if event.pipeline_mode is not None:
                self._snapshot["pipeline_mode"] = event.pipeline_mode
            if event.published_count is not None:
                self._snapshot["published_count"] = event.published_count
            if event.output_folder is not None:
                self._snapshot["output_folder"] = event.output_folder
            if event.total is not None and event.total > 0 and event.completed is not None:
                self._snapshot["percent"] = min(100.0, max(0.0, event.completed * 100.0 / event.total))
            else:
                self._snapshot["percent"] = None

            pipeline = self._snapshot.setdefault("pipeline", _pipeline_default())
            if new_stage:
                stage_order = V5_PIPELINE_STAGES if (
                    event.pipeline_mode == "V5" or self._snapshot.get("pipeline_mode") == "V5"
                ) else PIPELINE_STAGES
                for stage in stage_order:
                    if stage == new_stage:
                        pipeline[stage] = event.stage_status or (
                            "failed" if event.state == ActivityState.FAILED.value else
                            "retry" if event.state in (ActivityState.RETRYING.value, ActivityState.REPAIRING.value) else
                            "active"
                        )
                if event.stage_status in {"complete", "skipped", "failed"}:
                    pipeline[new_stage] = event.stage_status

            item_key = event.item_key or event.current_item or event.chunk_id or event.output_id
            if new_stage and item_key:
                key = (new_stage, item_key)
                if event.item_event == "start":
                    self._item_starts[key] = now_mono
                elif event.item_event == "complete":
                    started = self._item_starts.pop(key, None)
                    if started is not None and not event.item_reused:
                        self._rates.setdefault(new_stage, []).append(max(0.0, now_mono - started))
                        self._rates[new_stage] = self._rates[new_stage][-8:]

            self._refresh_times(now_mono)
            self._update_eta(new_stage, event.completed, event.total)
            if event.analysis_complete and self._snapshot.get("analysis_seconds") is None:
                self._snapshot["analysis_seconds"] = self._snapshot["project_elapsed_seconds"]
            self._append_activity(now_wall, event)
            self._snapshot["rate_samples"] = self._rates
            return self._persist_and_publish()

    def _append_activity(self, wall: float, event: ProgressEvent) -> None:
        text = event.activity_text.strip()
        if not text:
            return
        item = event.current_item or event.chunk_id or event.output_id or event.episode_id
        if item and item not in text:
            text = f"{item}: {text}"
        self._history.append({"timestamp": _utc_iso(wall), "text": text[:300]})
        self._snapshot["recent_activity"] = list(self._history)

    def _refresh_times(self, now_mono: float) -> None:
        session = 0.0 if self._session_started is None else max(0.0, now_mono - self._session_started)
        project = self._base_accumulated + session
        stage = self._stage_base
        if self._stage_started is not None:
            stage += max(0.0, now_mono - self._stage_started)
        self._snapshot["session_elapsed_seconds"] = session
        self._snapshot["project_elapsed_seconds"] = project
        self._snapshot["stage_elapsed_seconds"] = stage

    def _update_eta(self, stage: str | None, completed: int | None, total: int | None) -> None:
        if not stage or completed is None or total is None or total <= completed:
            self._snapshot["estimated_remaining_seconds"] = 0.0 if total is not None and completed is not None and total <= completed else None
            return
        samples = self._rates.get(stage, [])
        if not samples:
            self._snapshot["estimated_remaining_seconds"] = None
            return
        raw = (sum(samples[-8:]) / len(samples[-8:])) * (total - completed)
        previous = self._snapshot.get("estimated_remaining_seconds")
        self._snapshot["estimated_remaining_seconds"] = raw if previous is None else previous * 0.7 + raw * 0.3

    def _finalize_stage(self, now_mono: float) -> None:
        stage = self._snapshot.get("stage")
        if not stage or self._stage_started is None:
            return
        duration = self._stage_base + max(0.0, now_mono - self._stage_started)
        self._snapshot.setdefault("stage_durations_seconds", {})[stage] = duration
        self._stage_started = None

    def terminate(
        self,
        state: ActivityState,
        *,
        activity_text: str,
        error: str | None = None,
        output_folder: str | None = None,
        published_count: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            now_mono = self.monotonic()
            now_wall = self.wall_clock()
            self._refresh_times(now_mono)
            self._finalize_stage(now_mono)
            total = float(self._snapshot.get("project_elapsed_seconds") or 0.0)
            self._base_accumulated = total
            self._session_started = None
            self._stage_started = None
            self._snapshot.update({
                "state": state.value,
                "active": False,
                "activity_text": activity_text,
                "error": error,
                "last_activity_at": _utc_iso(now_wall),
                "event_timestamp": _utc_iso(now_wall),
                "session_elapsed_seconds": self._snapshot.get("session_elapsed_seconds", 0.0),
                "project_elapsed_seconds": total,
                "completion_seconds": total if state == ActivityState.COMPLETE else None,
            })
            analysis = self._snapshot.get("analysis_seconds")
            if analysis is not None:
                self._snapshot["downstream_seconds"] = max(0.0, total - float(analysis))
            if output_folder is not None:
                self._snapshot["output_folder"] = output_folder
            if published_count is not None:
                self._snapshot["published_count"] = published_count
            stage = self._snapshot.get("stage")
            if stage:
                self._snapshot.setdefault("pipeline", _pipeline_default())[stage] = (
                    "complete" if state == ActivityState.COMPLETE else
                    "failed" if state == ActivityState.FAILED else "pending"
                )
            self._history.append({"timestamp": _utc_iso(now_wall), "text": activity_text[:300]})
            self._snapshot["recent_activity"] = list(self._history)
            return self._persist_and_publish()

    def tick(self) -> dict[str, Any]:
        """Return a live in-memory snapshot without writing a timer checkpoint."""
        with self._lock:
            if self._snapshot.get("active"):
                self._refresh_times(self.monotonic())
            return dict(self._snapshot)

    def _persist_and_publish(self) -> dict[str, Any]:
        result = dict(self._snapshot)
        self.persistence.save_operational_status(self.project_id, result)
        if self.callback is not None:
            try:
                self.callback(result)
            except Exception:
                logger.exception("Workflow activity UI callback failed")
        return result


def _safe_checkpoint(persistence: ProjectPersistence, project_id: str, checkpoint_id: str) -> dict[str, Any] | None:
    try:
        data = persistence.load_checkpoint(project_id, checkpoint_id)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _stage_from_project(state: Mapping[str, Any]) -> str:
    status = str(state.get("status", ""))
    error = str(state.get("error") or "").lower()
    if state.get("pipeline_version") == "5.0":
        return {
            "created": WorkflowStage.PREPARATION.value,
            "new": WorkflowStage.PREPARATION.value,
            "analyzing": WorkflowStage.AI_GATEWAY.value,
            "final_json_ready": WorkflowStage.VALIDATE_JSON.value,
            "review_required": WorkflowStage.REVIEW.value,
            "rendering": WorkflowStage.RENDER.value,
            "completed": WorkflowStage.PUBLISH.value,
        }.get(status, WorkflowStage.PREPARATION.value)
    if "scanner" in error:
        return WorkflowStage.SCANNER.value
    if "catalog" in error:
        return WorkflowStage.CATALOG.value
    if "planner" in error:
        return WorkflowStage.PLANNER.value
    if "writer" in error:
        return WorkflowStage.WRITERS.value
    if "vision" in error or "visual" in error:
        return WorkflowStage.VISION.value
    mapping = {
        "created": WorkflowStage.PREPARATION.value,
        "new": WorkflowStage.PREPARATION.value,
        "analyzing": WorkflowStage.PREPARATION.value,
        "prepared": WorkflowStage.SCANNER.value,
        "evidence_ready": WorkflowStage.CATALOG.value,
        "catalog_ready": WorkflowStage.PLANNER.value,
        "planner_draft_ready": WorkflowStage.VISION.value,
        "season_plan_ready": WorkflowStage.WRITERS.value,
        "writer_drafts_ready": WorkflowStage.FINAL_JSON.value,
        "final_json_ready": WorkflowStage.VOICE.value,
        "analyzed": WorkflowStage.VOICE.value,
        "json_ready": WorkflowStage.VOICE.value,
        "rendering": WorkflowStage.RENDER.value,
        "completed": WorkflowStage.PUBLISH.value,
    }
    return mapping.get(status, WorkflowStage.PREPARATION.value)


def _latest_writer_manifest(
    persistence: ProjectPersistence, project_id: str,
    expected_plan_hash: str | None = None,
) -> dict[str, Any] | None:
    root = persistence.projects_dir / project_id / "writers"
    if not root.is_dir():
        return None
    if expected_plan_hash:
        exact = root / expected_plan_hash / "manifest.json"
        candidates = [exact] if exact.is_file() else []
    else:
        candidates = sorted(
            (path for path in root.glob("*/manifest.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    for path in candidates:
        try:
            import json
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("expected_output_count"), int):
                return value
        except (OSError, ValueError, TypeError):
            continue
    return None


def _scanner_item_from_error(error: str) -> str | None:
    match = re.search(r"\bE\d{2}-CH-\d{3}\b", error)
    return match.group(0) if match else None


def reconstruct_project_progress(
    persistence: ProjectPersistence,
    project_id: str,
) -> dict[str, Any]:
    """Build a truthful restart snapshot from operational and authoritative artifacts.

    This runs only when a saved project is selected, never on the one-second UI tick.
    """
    state = persistence.load_project(project_id)
    if state.get("project_mode", "RECAP") == "HIGHLIGHT" and state.get("pipeline_version") != "5.0":
        try:
            highlight_final = persistence.load_checkpoint(project_id, "highlight_final")
        except Exception:
            highlight_final = None
        prepared_count = len(persistence.list_prepared_episodes(project_id))
        source_count = len(state.get("sources", []))
        status = str(state.get("status", ""))
        if status == "completed":
            stage_label, activity = "Publishing", "Saved Highlight project is complete."
            activity_state = ActivityState.COMPLETE.value
        elif highlight_final and highlight_final.get("status") == "COMPLETE":
            stage_label, activity = "Rendering Highlights", "Validated Highlight plan is ready to resume publication."
            activity_state = ActivityState.IDLE.value
        elif prepared_count and prepared_count >= source_count:
            stage_label, activity = "Scanning Evidence", "Prepared Highlight sources will be reused on resume."
            activity_state = ActivityState.IDLE.value
        else:
            stage_label, activity = "Preparing Sources", "Saved Highlight project is ready to continue."
            activity_state = ActivityState.IDLE.value
        if status == "failed":
            activity_state, activity = ActivityState.FAILED.value, "Saved Highlight failure; valid checkpoints are preserved."
        elif status == "cancelled":
            activity_state = ActivityState.CANCELLED.value
        return {
            "schema_version": PROGRESS_SCHEMA_VERSION, "project_id": project_id,
            "pipeline_mode": "HIGHLIGHT", "state": activity_state, "stage": stage_label,
            "stage_label": stage_label, "activity_text": activity, "active": False,
            "completed": prepared_count if stage_label == "Preparing Sources" else None,
            "total": source_count if stage_label == "Preparing Sources" else None,
            "percent": (prepared_count * 100.0 / source_count if source_count and stage_label == "Preparing Sources" else None),
            "unit": "episodes", "session_elapsed_seconds": 0.0, "project_elapsed_seconds": None,
            "stage_elapsed_seconds": 0.0, "estimated_remaining_seconds": None,
            "recent_activity": [], "last_activity_at": None, "reconstructed": True,
            "error": state.get("error") if status == "failed" else None,
        }
    if persistence.has_operational_status(project_id):
        try:
            snapshot = persistence.load_operational_status(project_id)
        except Exception:
            snapshot = {}
    else:
        snapshot = {}
    snapshot = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "project_id": project_id,
        "pipeline_mode": "V5" if state.get("pipeline_version") == "5.0" else "RECAP",
        "state": ActivityState.IDLE.value,
        "stage": _stage_from_project(state),
        "stage_label": STAGE_LABELS[_stage_from_project(state)],
        "activity_text": "Saved project is ready to continue.",
        "active": False,
        "completed": None,
        "total": None,
        "percent": None,
        "unit": "items",
        "reused": None,
        "session_elapsed_seconds": 0.0,
        "project_elapsed_seconds": None,
        "stage_elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "analysis_seconds": None,
        "downstream_seconds": None,
        "completion_seconds": None,
        "stage_durations_seconds": {},
        "pipeline": _pipeline_default(),
        "recent_activity": [],
        "last_activity_at": None,
        "event_timestamp": None,
        "previous_timing_available": False,
        **snapshot,
    }
    snapshot["reconstructed"] = True
    snapshot["active"] = False
    snapshot["session_elapsed_seconds"] = 0.0
    status = str(state.get("status", ""))
    if status == "failed":
        snapshot["state"] = ActivityState.FAILED.value
        snapshot["error"] = state.get("error")
        snapshot["activity_text"] = "Saved failure; checkpoints are available for retry."
    elif status == "cancelled":
        snapshot["state"] = ActivityState.CANCELLED.value
    elif status == "completed":
        snapshot["state"] = ActivityState.COMPLETE.value
    elif snapshot.get("state") not in {ActivityState.FAILED.value,ActivityState.CANCELLED.value}:
        snapshot["state"] = ActivityState.IDLE.value

    if state.get("pipeline_version") == "5.0":
        pipeline = snapshot.setdefault("pipeline", _pipeline_default())
        prepared_count = len(persistence.list_prepared_episodes(project_id))
        if prepared_count >= len(state.get("sources", [])):
            pipeline[WorkflowStage.PREPARATION.value] = "complete"
        if isinstance(state.get("final_json"), dict) or persistence.has_final_json(project_id):
            pipeline[WorkflowStage.AI_GATEWAY.value] = "complete"
            pipeline[WorkflowStage.VALIDATE_JSON.value] = "complete"
        if status == "review_required":
            pipeline[WorkflowStage.REVIEW.value] = "active"
        elif isinstance(state.get("final_json"), dict):
            pipeline[WorkflowStage.REVIEW.value] = "complete"
        if status == "completed":
            for stage in V5_PIPELINE_STAGES:
                pipeline[stage] = "complete"
        current_stage = _stage_from_project(state)
        if status == "failed" and snapshot.get("stage") in V5_PIPELINE_STAGES:
            current_stage = str(snapshot["stage"])
        snapshot["pipeline_mode"] = "V5"
        snapshot["stage"] = current_stage
        snapshot["stage_label"] = STAGE_LABELS[current_stage]
        pipeline[current_stage] = "failed" if status == "failed" else pipeline.get(current_stage, "active")
        return snapshot

    pipeline = snapshot.setdefault("pipeline", _pipeline_default())
    prepared = None
    try:
        prepared = persistence.load_prepared_manifest(project_id)
    except Exception:
        pass
    if prepared and prepared.get("status") == "completed":
        pipeline[WorkflowStage.PREPARATION.value] = "complete"

    checkpoints = {
        WorkflowStage.CATALOG.value: "catalog",
        WorkflowStage.PLANNER.value: "planner_draft",
        WorkflowStage.FINAL_PLAN.value: "season_plan",
        WorkflowStage.WRITERS.value: "writer_drafts",
        WorkflowStage.FINAL_JSON.value: "final_json",
    }
    checkpoint_data: dict[str, dict[str, Any]] = {}
    for stage, checkpoint_id in checkpoints.items():
        data = _safe_checkpoint(persistence, project_id, checkpoint_id)
        if data and data.get("status") == "completed":
            if stage == WorkflowStage.FINAL_PLAN.value:
                path_value = None
                try:
                    import json
                    from toolrecap_v4.analysis.finalizer.season_plan import SeasonPlanService
                    from toolrecap_v4.analysis.finalizer.writer import find_locked_plan_artifact

                    path_value = data.get("season_plan_path") or (state.get("season_plan") or {}).get("season_plan_path")
                    plan_path = Path(path_value)
                    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
                    plan = SeasonPlanService._from_dict(plan_data)
                    if data.get("plan_hash") != plan.plan_hash:
                        raise ValueError("Season Plan checkpoint hash differs from plan content")
                    find_locked_plan_artifact(persistence.root, plan)
                except Exception as exc:
                    pipeline[stage] = "retry" if path_value and Path(path_value).is_file() and not (Path(path_value).parent / "manifest.json").is_file() else "failed"
                    snapshot["stage"] = stage
                    snapshot["stage_label"] = STAGE_LABELS[stage]
                    snapshot["state"] = ActivityState.FAILED.value if pipeline[stage] == "failed" else ActivityState.IDLE.value
                    snapshot["activity_text"] = (
                        "Season Plan checkpoint requires safe local recovery; press Continue."
                        if pipeline[stage] == "retry" else "Season Plan checkpoint integrity error."
                    )
                    snapshot["error"] = str(exc)
                    continue
            pipeline[stage] = "complete"
            checkpoint_data[stage] = data
    evidence = _safe_checkpoint(persistence, project_id, "evidence")
    if evidence and evidence.get("status") == "completed":
        pipeline[WorkflowStage.SCANNER.value] = "complete"
        pipeline[WorkflowStage.EVIDENCE.value] = "complete"

    current_stage = snapshot.get("stage") if pipeline.get(WorkflowStage.FINAL_PLAN.value) in {"retry", "failed"} else _stage_from_project(state)
    active_plan_hash = (state.get("season_plan") or {}).get("plan_hash") if isinstance(state.get("season_plan"), dict) else None
    writer_manifest = _latest_writer_manifest(persistence, project_id, active_plan_hash)
    if writer_manifest and active_plan_hash:
        from toolrecap_v4.analysis.finalizer.writer_store import WriterStore

        store = WriterStore(persistence.root, project_id, active_plan_hash)
        output_ids = writer_manifest.get("ordered_output_ids") if isinstance(writer_manifest.get("ordered_output_ids"), list) else []
        durable_ids = [output_id for output_id in output_ids if store.inspect_output(output_id)[0]]
        writer_manifest = {**writer_manifest, "completed_response_count": len(durable_ids), "durable_output_ids": durable_ids}
    if status in {"failed", "cancelled"} and writer_manifest:
        expected = int(writer_manifest.get("expected_output_count", 0))
        completed_writer = int(writer_manifest.get("completed_response_count", 0))
        if expected > completed_writer:
            current_stage = WorkflowStage.WRITERS.value
    snapshot["stage"] = current_stage
    snapshot["stage_label"] = STAGE_LABELS[current_stage]
    pipeline[current_stage] = "failed" if status == "failed" else pipeline.get(current_stage, "active")

    if current_stage == WorkflowStage.SCANNER.value:
        try:
            from toolrecap_v4.analysis.models import PreparedEpisode
            from toolrecap_v4.analysis.scanner import ScannerChunkPolicy, ScannerConfig, ScannerService
            from toolrecap_v4.gateway import GatewayClient
            from toolrecap_v4.settings import AppSettings

            cfg = AppSettings.from_dict(state.get("settings_snapshot", {}))
            episodes = [
                PreparedEpisode.from_dict(item)
                for item in persistence.list_prepared_episodes(project_id)
            ]
            if episodes and cfg.scanner_model.strip():
                scanner = ScannerService(
                    GatewayClient(), persistence.root,
                    ScannerConfig(
                        model=cfg.scanner_model,
                        reasoning=cfg.scanner_reasoning,
                        parallelism=cfg.scanner_parallelism,
                        chunk_policy=ScannerChunkPolicy(
                            max_duration_ms=cfg.scanner_chunk_duration_ms,
                            max_request_bytes=cfg.scanner_max_request_bytes,
                        ),
                        repair_attempts=cfg.scanner_repair_attempts,
                    ),
                )
                scanner_progress = scanner.inspect_project_progress(project_id, episodes)
                snapshot.update({
                    "completed": scanner_progress["completed"],
                    "total": scanner_progress["total"],
                    "percent": scanner_progress["percent"],
                    "unit": "chunks",
                    "reused": scanner_progress["completed"],
                    "current_item": _scanner_item_from_error(str(state.get("error") or "")) or scanner_progress["next_chunk"],
                    "episode_id": scanner_progress.get("next_episode"),
                    "chunk_id": _scanner_item_from_error(str(state.get("error") or "")) or scanner_progress["next_chunk"],
                    "activity_text": f"Saved Scanner progress: {scanner_progress['completed']} / {scanner_progress['total']} chunks.",
                })
        except Exception:
            logger.exception("Could not reconstruct Scanner progress for %s", project_id)

    writer = checkpoint_data.get(WorkflowStage.WRITERS.value)
    if current_stage in (WorkflowStage.WRITERS.value, WorkflowStage.FINAL_JSON.value) and writer:
        snapshot.update({
            "completed": int(writer.get("completed_response_count", 0)),
            "total": int(writer.get("expected_output_count", 0)),
            "unit": "outputs",
            "reused": int(writer.get("reused_count", 0)),
        })
    elif current_stage == WorkflowStage.WRITERS.value and writer_manifest:
        expected = int(writer_manifest.get("expected_output_count", 0))
        completed_writer = int(writer_manifest.get("completed_response_count", 0))
        failures = writer_manifest.get("failures") if isinstance(writer_manifest.get("failures"), dict) else {}
        next_output = next(iter(failures), None)
        snapshot.update({
            "completed": completed_writer,
            "total": expected,
            "percent": completed_writer * 100.0 / expected if expected else None,
            "unit": "outputs",
            "reused": completed_writer,
            "current_item": next_output,
            "output_id": next_output,
            "activity_text": f"Saved Writer progress: {completed_writer} / {expected} outputs; {len(failures)} require retry.",
        })
    final_json = state.get("final_json")
    outputs = final_json.get("outputs", []) if isinstance(final_json, dict) else []
    if final_json:
        pipeline[WorkflowStage.FINAL_JSON.value] = "complete"
        if snapshot.get("analysis_seconds") is None:
            snapshot["analysis_time_unavailable"] = True
    completed_outputs = 0
    for output in outputs:
        render_id = output.get("render_id")
        if not render_id:
            continue
        data = _safe_checkpoint(persistence, project_id, render_id)
        if data and data.get("status") in {"completed", "skipped"}:
            path = data.get("output_path")
            if path and Path(path).is_file():
                completed_outputs += 1
    if current_stage in {WorkflowStage.VOICE.value, WorkflowStage.RENDER.value, WorkflowStage.PUBLISH.value}:
        snapshot.update({"completed": completed_outputs, "total": len(outputs), "unit": "outputs", "reused": completed_outputs})
    if status == "completed":
        snapshot["published_count"] = completed_outputs
        snapshot["output_folder"] = state.get("output_dir")
    return snapshot
