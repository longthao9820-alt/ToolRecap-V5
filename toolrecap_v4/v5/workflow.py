"""Simplified ToolRecap V5 workflow.

Pipeline: prepare sources -> one Gateway analysis -> technical validation ->
optional review -> existing VoiceStudio/FFmpeg renderer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from toolrecap_v4.analysis.cache import AnalysisCacheManager
from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.analysis.source_prep.pipeline import SourcePreparationPipeline
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.discovery import natural_sort_key
from toolrecap_v4.errors import CancelledError
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.media import probe_media
from toolrecap_v4.output_paths import derive_working_folder, resolve_v5_publication_root
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.progress import (
    ActivityState,
    ProgressEvent,
    WorkflowProgressTracker,
    WorkflowStage,
)
from toolrecap_v4.settings import AppSettings, SettingsManager
from toolrecap_v4.v5.analyzer import GatewayRenderJsonAnalyzer
from toolrecap_v4.validator import (
    check_for_secrets,
    validate_project,
    validate_windows_name,
)
from toolrecap_v4.voice_studio import VoiceStudioAdapter
from toolrecap_v4.workflow import (
    ProjectStatus,
    ProjectWorkflow,
    WorkflowCallbacks,
    resolve_sources,
    verify_source_integrity,
)

V5_PIPELINE_VERSION = "5.0"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def validate_mode_render_contract(mode: str, final_json: dict[str, Any]) -> None:
    """Enforce renderer-facing mode rules without making editorial decisions."""
    if mode != "HIGHLIGHT":
        return
    for output in final_json.get("outputs", []):
        for segment in output.get("segments", []):
            segment_id = str(segment.get("segment_id", ""))
            if str(segment.get("narration", "")).strip():
                raise ValueError(f"Highlight segment '{segment_id}' cannot contain generated narration")
            if segment.get("type") != "original_dialogue":
                raise ValueError(f"Highlight segment '{segment_id}' must use type=original_dialogue")
            if segment.get("source_audio") is not True:
                raise ValueError(f"Highlight segment '{segment_id}' must preserve source_audio")
            if segment.get("audio_intent") != "ORIGINAL_AUDIO":
                raise ValueError(f"Highlight segment '{segment_id}' must use audio_intent=ORIGINAL_AUDIO")


class V5Workflow:
    """One editorial AI boundary followed by the stable render engine."""

    def __init__(
        self,
        *,
        persistence: ProjectPersistence,
        gateway_client: GatewayClient,
        voice_adapter: VoiceStudioAdapter | None,
        settings_manager: SettingsManager | None = None,
        source_pipeline: SourcePreparationPipeline | None = None,
        analyzer: GatewayRenderJsonAnalyzer | None = None,
    ) -> None:
        self.persistence = persistence
        self.gateway = gateway_client
        self.voice_adapter = voice_adapter
        self.settings_manager = settings_manager or SettingsManager(persistence=persistence)
        self.source_pipeline = source_pipeline or SourcePreparationPipeline(
            cache_manager=AnalysisCacheManager(persistence.root / "cache" / "analysis"),
            allow_stt=False,
            allow_bitmap_ocr=False,
        )
        self.analyzer = analyzer

    def create_project(
        self,
        *,
        project_id: str,
        project_name: str,
        source_input: str | Path | Sequence[str | Path],
        prompt: str,
        mode: str,
        settings: AppSettings,
        output_dir: str | Path | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        validate_windows_name(project_id, "project_id")
        validate_windows_name(project_name, "project_name")
        normalized_mode = mode.upper()
        if normalized_mode not in {"RECAP", "HIGHLIGHT"}:
            raise ValueError("V5 mode must be RECAP or HIGHLIGHT")
        if not prompt.strip():
            raise ValueError(f"{normalized_mode.title()} prompt cannot be empty")
        fingerprints = resolve_sources(source_input, cancellation_token)
        working_folder = derive_working_folder(source_input)
        manual_output = str(output_dir or settings.output_dir or "").strip()
        publication = resolve_v5_publication_root(
            manual_output_dir=manual_output,
            working_folder=working_folder,
            last_source_file=fingerprints[-1].basename,
            mode=normalized_mode,
        )
        sources: list[dict[str, Any]] = []
        for index, fingerprint in enumerate(fingerprints, 1):
            if cancellation_token:
                cancellation_token.check_cancelled()
            duration_ms = round(probe_media(
                fingerprint.path, cancellation_token=cancellation_token,
            ).duration * 1000.0)
            sources.append({
                "episode_id": f"E{index:02d}",
                "source_id": f"src_{index:03d}",
                "source_file": fingerprint.basename,
                "duration_ms": duration_ms,
                "fingerprint": fingerprint.to_dict(),
            })
        timestamp = _now()
        state: dict[str, Any] = {
            "schema_version": "5.0",
            "pipeline_version": V5_PIPELINE_VERSION,
            "project_mode": normalized_mode,
            "project_id": project_id,
            "project_name": project_name,
            "status": ProjectStatus.CREATED.value,
            "prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_dir": str(publication),
            "manual_output_dir": manual_output,
            "working_folder": str(working_folder),
            "sources": sources,
            "source_fingerprints": {item["source_file"]: item["fingerprint"] for item in sources},
            "settings_snapshot": settings.to_dict(),
            "gateway_identifier": f"{settings.gateway_endpoint}::v5-single-analysis",
            "voice_endpoint": f"{settings.voice_mode}::{settings.voice_local_url}::{settings.voice_remote_url}",
            "prepared_episodes": {},
            "final_json": None,
            "outputs": {},
            "error": None,
            "timestamps": {"created_at": timestamp, "updated_at": timestamp},
        }
        check_for_secrets(state)
        self.persistence.save_project(state)
        return state

    def _emit(
        self,
        tracker: WorkflowProgressTracker,
        stage: WorkflowStage,
        text: str,
        *,
        state: ActivityState = ActivityState.RUNNING,
        completed: int | None = None,
        total: int | None = None,
        current_item: str | None = None,
        item_event: str | None = None,
        stage_status: str | None = None,
    ) -> None:
        tracker.emit(ProgressEvent(
            state=state.value,
            stage=stage.value,
            activity_text=text,
            completed=completed,
            total=total,
            unit="episodes" if stage == WorkflowStage.PREPARATION else "items",
            current_item=current_item,
            item_event=item_event,
            item_key=current_item,
            stage_status=stage_status,
            pipeline_mode="V5",
        ))

    def _prepare(
        self,
        state: dict[str, Any],
        tracker: WorkflowProgressTracker,
        cancellation_token: CancellationToken | None,
    ) -> list[PreparedEpisode]:
        prepared: list[PreparedEpisode] = []
        ordered = sorted(state["sources"], key=lambda item: natural_sort_key(item["source_file"]))
        total = len(ordered)
        for index, source in enumerate(ordered, 1):
            episode_id = source["episode_id"]
            self._emit(
                tracker, WorkflowStage.PREPARATION, "Preparing source media and transcript...",
                state=ActivityState.LOCAL_PROCESSING,
                completed=index - 1, total=total, current_item=episode_id, item_event="start",
            )
            episode = self.source_pipeline.prepare_episode(
                source["fingerprint"]["path"],
                episode_id,
                source["source_id"],
                source_fingerprint=source["fingerprint"].get("sha256"),
                on_progress=lambda _phase, _pct, message, eid=episode_id, done=index - 1: self._emit(
                    tracker, WorkflowStage.PREPARATION, message,
                    state=ActivityState.LOCAL_PROCESSING,
                    completed=done, total=total, current_item=eid,
                ),
                cancellation_token=cancellation_token,
            )
            if episode.transcript_method == "ocr":
                raise ValueError(
                    f"{source['source_file']}: bitmap subtitles require OCR, which is disabled in V5."
                )
            if not episode.transcript.cues:
                raise ValueError(
                    f"{source['source_file']}: no usable English text subtitles were extracted."
                )
            self.persistence.save_prepared_episode(state["project_id"], episode_id, episode.to_dict())
            state["prepared_episodes"][episode_id] = {
                "episode_id": episode_id,
                "source_file": source["source_file"],
                "duration_ms": episode.duration_ms,
                "transcript_method": episode.transcript_method,
                "cue_count": episode.transcript.cue_count,
                "artifact_hash": episode.artifact_hash,
            }
            state["timestamps"]["updated_at"] = _now()
            self.persistence.save_project(state)
            prepared.append(episode)
            self._emit(
                tracker, WorkflowStage.PREPARATION, "Source preparation completed.",
                completed=index, total=total, current_item=episode_id, item_event="complete",
                stage_status="complete" if index == total else None,
            )
        return prepared

    def run(
        self,
        project_id: str,
        *,
        output_dir: str | Path | None = None,
        settings: AppSettings | None = None,
        cancellation_token: CancellationToken | None = None,
        callbacks: WorkflowCallbacks | None = None,
        progress_tracker: WorkflowProgressTracker | None = None,
        resuming: bool = False,
    ) -> dict[str, Any]:
        state = self.persistence.load_project(project_id)
        if state.get("pipeline_version") != V5_PIPELINE_VERSION:
            raise ValueError("Project is not a ToolRecap V5 project")
        cfg = settings or AppSettings.from_dict(state["settings_snapshot"])
        tracker = progress_tracker or WorkflowProgressTracker(
            self.persistence,
            project_id,
            callback=callbacks.on_activity if callbacks else None,
        )
        if not tracker.tick().get("active"):
            tracker.begin(resuming=resuming)
        try:
            final_json = state.get("final_json")
            if final_json is None and self.persistence.has_final_json(project_id):
                final_json = self.persistence.load_final_json(project_id)
                state["final_json"] = final_json
            if final_json is None:
                verify_source_integrity(state["source_fingerprints"], cancellation_token)
                state["status"] = ProjectStatus.ANALYZING.value
                state["timestamps"]["analyzing_started_at"] = _now()
                state["timestamps"]["updated_at"] = _now()
                self.persistence.save_project(state)
                if callbacks and callbacks.on_status_change:
                    callbacks.on_status_change(ProjectStatus.ANALYZING.value)
                episodes = self._prepare(state, tracker, cancellation_token)
                analyzer = self.analyzer or GatewayRenderJsonAnalyzer(
                    self.gateway,
                    cfg,
                    progress_callback=tracker.emit_payload,
                )
                self._emit(
                    tracker, WorkflowStage.AI_GATEWAY,
                    "AI Gateway is analysing the prompt and composing Render JSON...",
                    state=ActivityState.WAITING_FOR_AI,
                )
                result = analyzer.analyze(
                    mode=state["project_mode"],
                    project_id=project_id,
                    project_name=state["project_name"],
                    editorial_prompt=state["prompt"],
                    episodes=episodes,
                    cancellation_token=cancellation_token,
                )
                state["raw_response"] = result.raw_response
                state["analysis_request_digest"] = result.request_digest
                state["analysis_model"] = result.model
                if callbacks and callbacks.on_raw_response:
                    callbacks.on_raw_response(result.raw_response)
                self._emit(
                    tracker, WorkflowStage.AI_GATEWAY,
                    "Render JSON received from AI Gateway.",
                    state=ActivityState.LOCAL_PROCESSING, stage_status="complete",
                )
                self._emit(
                    tracker, WorkflowStage.VALIDATE_JSON,
                    "Validating Render JSON against source files and durations...",
                    state=ActivityState.LOCAL_PROCESSING,
                )
                durations = {item["source_file"]: int(item["duration_ms"]) for item in state["sources"]}
                validate_project(result.final_json, source_durations=durations, cancellation_token=cancellation_token)
                validate_mode_render_contract(state["project_mode"], result.final_json)
                if result.final_json.get("project_id") != project_id:
                    raise ValueError("Render JSON project_id does not match the active project")
                if result.final_json.get("project_name") != state["project_name"]:
                    raise ValueError("Render JSON project_name does not match the active project")
                final_json = result.final_json
                self.persistence.save_final_json(project_id, final_json)
                state["final_json"] = final_json
                state["status"] = ProjectStatus.FINAL_JSON_READY.value
                state["timestamps"]["final_json_ready_at"] = _now()
                state["timestamps"]["updated_at"] = _now()
                self.persistence.save_project(state)
                if callbacks and callbacks.on_final_json:
                    callbacks.on_final_json(final_json)
                if callbacks and callbacks.on_final_json_ready:
                    callbacks.on_final_json_ready({"status": "completed", "output_count": len(final_json["outputs"])})
                if callbacks and callbacks.on_status_change:
                    callbacks.on_status_change(ProjectStatus.FINAL_JSON_READY.value)
                self._emit(
                    tracker, WorkflowStage.VALIDATE_JSON,
                    "Render JSON is valid and saved.",
                    state=ActivityState.LOCAL_PROCESSING, stage_status="complete",
                )

            self._emit(
                tracker, WorkflowStage.REVIEW,
                "Checking review approval before rendering...",
                state=ActivityState.LOCAL_PROCESSING,
            )
            renderer_workflow = ProjectWorkflow(
                persistence=self.persistence,
                gateway_client=self.gateway,
                voice_adapter=self.voice_adapter,
                settings_manager=self.settings_manager,
                storage_root=self.persistence.root,
            )
            rendered = renderer_workflow.start_project(
                project_id=project_id,
                output_dir=output_dir or state["output_dir"],
                settings=cfg,
                cancellation_token=cancellation_token,
                callbacks=callbacks,
                progress_tracker=tracker,
            )
            rendered["pipeline_version"] = V5_PIPELINE_VERSION
            self.persistence.save_project(rendered)
            return rendered
        except CancelledError:
            latest = self.persistence.load_project(project_id)
            latest["status"] = ProjectStatus.CANCELLED.value
            latest["timestamps"]["cancelled_at"] = _now()
            latest["timestamps"]["updated_at"] = _now()
            self.persistence.save_project(latest)
            if tracker.tick().get("active"):
                tracker.terminate(ActivityState.CANCELLED, activity_text="V5 workflow cancelled.")
            raise
        except Exception as exc:
            # ProjectWorkflow owns cancellation/failure persistence after the
            # render boundary. V5 owns failures before that boundary.
            latest = self.persistence.load_project(project_id)
            if latest.get("status") not in {
                ProjectStatus.CANCELLED.value,
                ProjectStatus.FAILED.value,
                ProjectStatus.REVIEW_REQUIRED.value,
            }:
                latest["status"] = ProjectStatus.FAILED.value
                latest["error"] = str(exc)
                latest["timestamps"]["failed_at"] = _now()
                latest["timestamps"]["updated_at"] = _now()
                self.persistence.save_project(latest)
            if tracker.tick().get("active"):
                tracker.terminate(
                    ActivityState.FAILED,
                    activity_text="V5 workflow failed.",
                    error=str(exc),
                )
            raise
