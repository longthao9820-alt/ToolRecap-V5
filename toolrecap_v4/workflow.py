"""Project workflow controller and execution engine for ToolRecap V4.

Invariants:
- Single file supported, folder direct children natural order.
- One project one Gateway submission.
- Raw response persisted including invalid JSON via callback/error field.
- Final validate actual probe durations then save BEFORE voice/render.
- Resume/import NEVER Gateway once Final JSON exists.
- Stop keeps completed outputs final/state.
- Reconcile interrupted ANALYZING/RENDERING to resumable.
- Render-settings fingerprint rerender affected outputs only no AI; source changes fail no substitution.
- Store source fingerprints, prompt hash, Gateway nonsecret identifier, settings snapshot, voice endpoint, timestamps.
- Validate completed output presence/hash before skip.
- Output failure stops queue preserves complete previous outputs.
- No secrets state/log.
- Prevent publication paths colliding ANY project sources including unused.
- Minimal controller no scheduler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from toolrecap_v4.analysis.cache import AnalysisCacheManager
from toolrecap_v4.analysis.finalizer import CatalogService, FinalizationConfig, FinalizationService, PlannerConfig, PlannerService, SeasonPlanService, WriterConfig, WriterService
from toolrecap_v4.analysis.vision import FrameExtractionPolicy, VisionConfig, VisualEvidenceService
from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.analysis.scanner import ScannerChunkPolicy, ScannerConfig, ScannerService
from toolrecap_v4.analysis.source_prep.pipeline import SourcePreparationPipeline
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.discovery import (
    SUPPORTED_EXTENSIONS,
    SourceFingerprint,
    compute_file_fingerprint,
    discover_sources,
    is_windows_reserved_stem,
    natural_sort_key,
)
from toolrecap_v4.downstream import DownstreamVoiceCache
from toolrecap_v4.errors import (
    AnalysisPipelineUnavailableError,
    CancelledError,
    DiscoveryError,
    InvalidGatewayResponseError,
    PersistenceError,
    SourceChangedError,
    SourceNotFoundError,
    ToolRecapError,
    ValidationError,
    WindowsCollisionError,
    WindowsNameError,
    WindowsReservedNameError,
)
from toolrecap_v4.gateway import GatewayClient, GatewayResult
from toolrecap_v4.media import probe_media
from toolrecap_v4.persistence import ProjectPersistence, get_storage_root
from toolrecap_v4.output_paths import derive_working_folder, ensure_publication_root, resolve_publication_root
from toolrecap_v4.renderer import (
    RenderError,
    RenderResult,
    SourceCollisionError,
    compute_file_sha256,
    render_output,
)
from toolrecap_v4.settings import AppSettings, SettingsManager
from toolrecap_v4.validator import check_for_secrets, validate_project, validate_windows_name
from toolrecap_v4.voice_studio import VoiceStudioAdapter
from toolrecap_v4.progress import (
    ActivityState,
    ProgressEvent,
    WorkflowProgressTracker,
    WorkflowStage,
)
from toolrecap_v4.original_dialogue import source_dialogue_map_from_episodes


class ProjectStatus(str, enum.Enum):
    CREATED = "created"
    NEW = "new"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    JSON_READY = "json_ready"
    PREPARED = "prepared"
    EVIDENCE_READY = "evidence_ready"
    CATALOG_READY = "catalog_ready"
    PLANNER_DRAFT_READY = "planner_draft_ready"
    SEASON_PLAN_READY = "season_plan_ready"
    WRITER_DRAFTS_READY = "writer_drafts_ready"
    FINAL_JSON_READY = "final_json_ready"
    REVIEW_REQUIRED = "review_required"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def normalize(cls, status: Union[str, ProjectStatus]) -> str:
        """Migrate and normalize status between user brief and internal enum names."""
        val = status.value if isinstance(status, ProjectStatus) else str(status)
        if val == cls.NEW.value:
            return cls.CREATED.value
        if val == cls.JSON_READY.value:
            return cls.ANALYZED.value
        return val


class OutputStatus(str, enum.Enum):
    PENDING = "pending"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowCallbacks:
    """Optional callbacks for tracking workflow progress."""

    on_status_change: Optional[Callable[[str], None]] = None
    on_sub_analysis: Optional[Callable[[str], None]] = None
    on_raw_response: Optional[Callable[[str], None]] = None
    on_final_json: Optional[Callable[[Dict[str, Any]], None]] = None
    on_output_started: Optional[Callable[[str], None]] = None
    on_output_completed: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_output_failed: Optional[Callable[[str, str], None]] = None
    on_output_skipped: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_error: Optional[Callable[[Exception, Optional[str]], None]] = None
    on_source_preparation_progress: Optional[Callable[..., None]] = None
    on_episode_prepared: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_evidence_ready: Optional[Callable[[str, Dict[str, int]], None]] = None
    on_catalog_ready: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_planner_draft_ready: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_season_plan_ready: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_writer_drafts_ready: Optional[Callable[[Dict[str, Any]], None]] = None
    on_final_json_ready: Optional[Callable[[Dict[str, Any]], None]] = None
    on_activity: Optional[Callable[[Dict[str, Any]], None]] = None


def resolve_sources(
    source_input: str | Path | Sequence[str | Path],
    cancellation_token: Optional[CancellationToken] = None,
) -> List[SourceFingerprint]:
    """Resolve single file, folder, or list of files into naturally sorted SourceFingerprint list.
    
    Rules:
    - Single file: validated, fingerprinted, returned as 1-element list.
    - Folder: direct children only (non-recursive), natural order, Windows collision check.
    - List/sequence: validated, fingerprinted, naturally sorted, Windows collision check.
    """
    if cancellation_token:
        cancellation_token.check_cancelled()

    if isinstance(source_input, (str, Path)):
        p = Path(source_input).resolve()
        if not p.exists():
            raise DiscoveryError(f"Source path does not exist: {p}")
        if p.is_dir():
            return discover_sources(p, cancellation_token=cancellation_token)
        elif p.is_file():
            ext = p.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise DiscoveryError(f"Unsupported media extension '{ext}' for file {p.name}")
            if is_windows_reserved_stem(p.stem):
                raise WindowsReservedNameError(f"Source file uses Windows reserved name: {p.name}")
            validate_windows_name(p.name, "source_file")
            fp = compute_file_fingerprint(p, cancellation_token=cancellation_token)
            return [fp]
        else:
            raise DiscoveryError(f"Source path is neither file nor directory: {p}")

    elif isinstance(source_input, (list, tuple)):
        if not source_input:
            raise DiscoveryError("Source list cannot be empty.")
        fps: List[SourceFingerprint] = []
        casefold_seen: Dict[str, str] = {}
        for item in source_input:
            if cancellation_token:
                cancellation_token.check_cancelled()
            p = Path(item).resolve()
            if not p.is_file():
                raise DiscoveryError(f"Source file does not exist or is not a file: {p}")
            ext = p.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise DiscoveryError(f"Unsupported media extension '{ext}' for file {p.name}")
            if is_windows_reserved_stem(p.stem):
                raise WindowsReservedNameError(f"Source file uses Windows reserved name: {p.name}")
            validate_windows_name(p.name, "source_file")
            cf = p.name.casefold()
            if cf in casefold_seen:
                raise WindowsCollisionError(
                    f"Duplicate source basename under casefold: '{p.name}' conflicts with '{casefold_seen[cf]}'"
                )
            casefold_seen[cf] = p.name
            fps.append(compute_file_fingerprint(p, cancellation_token=cancellation_token))

        fps.sort(key=lambda x: natural_sort_key(x.basename))
        return fps
    else:
        raise TypeError(f"Invalid source_input type: {type(source_input).__name__}")


def compute_output_fingerprint(
    output_def: Dict[str, Any],
    source_fingerprints: Dict[str, Any],
    settings: AppSettings,
    source_dialogue_digest: str = "",
) -> str:
    """Compute deterministic SHA-256 fingerprint for an output, its sources, and render settings."""
    out_copy = {
        "render_id": output_def.get("render_id"),
        "title": output_def.get("title"),
        "segments": output_def.get("segments", []),
    }

    used_sources: Dict[str, Any] = {}
    for seg in output_def.get("segments", []):
        src_name = seg.get("source_file")
        if src_name and src_name in source_fingerprints:
            s_fp = source_fingerprints[src_name]
            if isinstance(s_fp, dict):
                used_sources[src_name] = {
                    "sha256": s_fp.get("sha256"),
                    "size_bytes": s_fp.get("size_bytes"),
                }
            else:
                used_sources[src_name] = {
                    "sha256": s_fp.sha256,
                    "size_bytes": s_fp.size_bytes,
                }

    render_settings = {
        "original_dialogue_mapper_version": "original-dialogue-mapper-v1",
        "original_audio_db": settings.original_audio_db,
        "commentary_audio_db": settings.commentary_audio_db,
        "auto_duck": settings.auto_duck,
        "ducking_amount_db": settings.ducking_amount_db,
        "target_loudness_lufs": settings.target_loudness_lufs,
        "true_peak_db": settings.true_peak_db,
        "highlight_normalize_loudness": settings.highlight_normalize_loudness,
        "use_gpu": settings.use_gpu,
        "quality": settings.quality,
        "video_codec": settings.video_codec,
        "canvas_width": settings.canvas_width,
        "canvas_height": settings.canvas_height,
        "canvas_fps": settings.canvas_fps,
        "canvas_auto": getattr(settings, "canvas_auto", True),
        "burn_subtitles": settings.burn_subtitles,
        "output_format": settings.output_format,
        "voice_id": settings.voice_id,
        "voice_language": settings.voice_language,
        "voice_style": settings.voice_style,
        "voice_model": settings.voice_model,
        "commentary_reading_speed": settings.commentary_reading_speed,
        "voice_mode": settings.voice_mode,
        "voice_local_url": settings.voice_local_url,
        "voice_remote_url": settings.voice_remote_url,
    }

    payload = {
        "output": out_copy,
        "sources": used_sources,
        "settings": render_settings,
        "source_dialogue_digest": source_dialogue_digest,
    }

    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_publication_collision(
    outputs: Sequence[Dict[str, Any]],
    output_dir: Path | str,
    all_source_paths: Sequence[Path | str],
) -> None:
    """Ensure publication paths do not collide with ANY project sources (including unused)."""
    out_dir = Path(output_dir).resolve()
    all_sources = {Path(p).resolve() for p in all_source_paths}

    for out in outputs:
        title = out.get("title", "")
        if not title:
            continue
        targets = [
            out_dir / f"{title}.mp4",
            out_dir / f"{title}.narration.srt",
            out_dir / f"{title}.original.srt",
        ]
        for t in targets:
            if t.resolve() in all_sources:
                raise SourceCollisionError(
                    f"Publication target '{t.name}' at '{t.resolve()}' collides with project source file."
                )


def verify_source_integrity(
    source_fingerprints: Dict[str, Any],
    cancellation_token: Optional[CancellationToken] = None,
) -> None:
    """Verify source files exist and have not changed since project creation."""
    for basename, fp_data in source_fingerprints.items():
        if cancellation_token:
            cancellation_token.check_cancelled()
        path_str = fp_data.get("path") if isinstance(fp_data, dict) else fp_data.path
        expected_sha = fp_data.get("sha256") if isinstance(fp_data, dict) else fp_data.sha256
        expected_size = fp_data.get("size_bytes") if isinstance(fp_data, dict) else fp_data.size_bytes

        p = Path(path_str).resolve()
        if not p.is_file():
            raise SourceNotFoundError(
                f"Source file '{basename}' not found at '{p}'. Substitution is strictly prohibited."
            )

        current_fp = compute_file_fingerprint(p, cancellation_token=cancellation_token)
        if current_fp.sha256 != expected_sha or current_fp.size_bytes != expected_size:
            raise SourceChangedError(
                f"Source file '{basename}' has been modified since project creation (expected sha256={expected_sha[:8]}..., "
                f"current sha256={current_fp.sha256[:8]}...). Substitution is strictly prohibited."
            )


def reconcile_project_state(
    project_state: Dict[str, Any],
    persistence: ProjectPersistence,
) -> Dict[str, Any]:
    """Reconcile interrupted ANALYZING/RENDERING states to resumable status."""
    project_id = project_state["project_id"]
    raw_status = project_state.get("status")
    status = ProjectStatus.normalize(raw_status) if raw_status else None

    if status == ProjectStatus.ANALYZING.value:
        if persistence.has_final_json(project_id) or project_state.get("final_json"):
            project_state["status"] = ProjectStatus.ANALYZED.value
        else:
            try:
                evidence_checkpoint = persistence.load_checkpoint(project_id, "evidence")
            except PersistenceError:
                evidence_checkpoint = None
            try:
                catalog_checkpoint = persistence.load_checkpoint(project_id, "catalog")
            except PersistenceError:
                catalog_checkpoint = None
            if catalog_checkpoint and catalog_checkpoint.get("status") == "completed":
                try:
                    planner_checkpoint = persistence.load_checkpoint(project_id, "planner_draft")
                except PersistenceError:
                    planner_checkpoint = None
                if planner_checkpoint and planner_checkpoint.get("status") == "completed":
                    try:
                        plan_checkpoint = persistence.load_checkpoint(project_id, "season_plan")
                    except PersistenceError:
                        plan_checkpoint = None
                    try:
                        writer_checkpoint = persistence.load_checkpoint(project_id, "writer_drafts")
                    except PersistenceError:
                        writer_checkpoint = None
                    if writer_checkpoint and writer_checkpoint.get("status") == "completed":
                        project_state["status"] = ProjectStatus.WRITER_DRAFTS_READY.value
                    else:
                        project_state["status"] = ProjectStatus.SEASON_PLAN_READY.value if plan_checkpoint and plan_checkpoint.get("status") == "completed" else ProjectStatus.PLANNER_DRAFT_READY.value
                else:
                    project_state["status"] = ProjectStatus.CATALOG_READY.value
            elif evidence_checkpoint and evidence_checkpoint.get("status") == "completed":
                project_state["status"] = ProjectStatus.EVIDENCE_READY.value
            elif persistence.has_prepared_manifest(project_id):
                project_state["status"] = ProjectStatus.PREPARED.value
            else:
                project_state["status"] = ProjectStatus.CREATED.value
        persistence.save_project(project_state)
    elif status == ProjectStatus.RENDERING.value:
        project_state["status"] = ProjectStatus.ANALYZED.value
        persistence.save_project(project_state)

    return project_state


class ProjectWorkflow:
    """Synchronous worker-friendly controller for ToolRecap V4 projects."""

    def __init__(
        self,
        persistence: Optional[ProjectPersistence] = None,
        gateway_client: Optional[GatewayClient] = None,
        voice_adapter: Optional[VoiceStudioAdapter] = None,
        settings_manager: Optional[SettingsManager] = None,
        storage_root: Optional[Path | str] = None,
        source_preparation_pipeline: Optional[SourcePreparationPipeline] = None,
        scanner_service: Optional[ScannerService] = None,
        catalog_service: Optional[CatalogService] = None,
        planner_service: Optional[PlannerService] = None,
        visual_service: Optional[VisualEvidenceService] = None,
        season_plan_service: Optional[SeasonPlanService] = None,
        writer_service: Optional[WriterService] = None,
        finalization_service: Optional[FinalizationService] = None,
        progress_monotonic: Optional[Callable[[], float]] = None,
        progress_wall_clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.persistence = persistence or ProjectPersistence(storage_root=storage_root)
        self.settings_manager = settings_manager or SettingsManager(persistence=self.persistence)
        self.gateway_client = gateway_client or GatewayClient()
        self.voice_adapter = voice_adapter
        self.source_preparation_pipeline = source_preparation_pipeline
        self.scanner_service = scanner_service
        self.catalog_service = catalog_service
        self.planner_service = planner_service
        self.visual_service = visual_service
        self.season_plan_service = season_plan_service
        self.writer_service = writer_service
        self.finalization_service = finalization_service
        self.progress_monotonic = progress_monotonic
        self.progress_wall_clock = progress_wall_clock

    def create_project(
        self,
        project_id: str,
        project_name: str,
        source_input: str | Path | Sequence[str | Path],
        prompt: str = "",
        output_dir: Optional[Path | str] = None,
        settings: Optional[AppSettings] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """Create a new project from sources and persist initial state."""
        validate_windows_name(project_id, "project_id")
        if not re.match(r"^[A-Za-z0-9._-]+$", project_id):
            raise ValidationError(f"project_id '{project_id}' must match pattern ^[A-Za-z0-9._-]+$")
        validate_windows_name(project_name, "project_name")

        cfg = settings or self.settings_manager.load()
        working_folder = derive_working_folder(source_input)
        fps = resolve_sources(source_input, cancellation_token=cancellation_token)
        if not fps:
            raise DiscoveryError("No supported source files discovered.")

        manual_output_dir = str(output_dir or cfg.output_dir or "").strip()
        resolved_output_dir = resolve_publication_root(
            manual_output_dir=manual_output_dir, working_folder=working_folder,
        )

        source_fps_map = {fp.basename: fp.to_dict() for fp in fps}
        effective_prompt = prompt or cfg.prompt
        prompt_hash = hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest() if effective_prompt else ""

        gw_identifier = f"{cfg.gateway_endpoint}::{cfg.gateway_model}"
        voice_endpoint = f"{cfg.voice_mode}::{cfg.voice_local_url}::{cfg.voice_remote_url}"

        now_iso = datetime.now(timezone.utc).isoformat()
        state: Dict[str, Any] = {
            "schema_version": "3.0",
            "project_mode": "RECAP",
            "project_id": project_id,
            "project_name": project_name,
            "status": ProjectStatus.CREATED.value,
            "output_dir": str(resolved_output_dir),
            "manual_output_dir": manual_output_dir,
            "working_folder": str(working_folder),
            "sources": [
                {
                    "source_file": fp.basename,
                    "fingerprint": fp.to_dict(),
                    "duration_ms": int(round(probe_media(fp.path, cancellation_token=cancellation_token).duration * 1000.0)),
                }
                for fp in fps
            ],
            "source_fingerprints": source_fps_map,
            "prompt": effective_prompt,
            "prompt_hash": prompt_hash,
            "gateway_identifier": gw_identifier,
            "settings_snapshot": cfg.to_dict(),
            "voice_endpoint": voice_endpoint,
            "timestamps": {
                "created_at": now_iso,
                "updated_at": now_iso,
            },
            "sub_analysis": None,
            "raw_response": None,
            "final_json": None,
            "prepared_episodes": {},
            "outputs": {},
            "error": None,
        }

        check_for_secrets(state)
        self.persistence.save_project(state)
        return state

    def import_project(
        self,
        project_id: str,
        project_name: str,
        source_input: str | Path | Sequence[str | Path],
        final_json: Dict[str, Any],
        output_dir: Optional[Path | str] = None,
        settings: Optional[AppSettings] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """Import an existing final project JSON with sources and validate before save."""
        validate_windows_name(project_id, "project_id")
        if not re.match(r"^[A-Za-z0-9._-]+$", project_id):
            raise ValidationError(f"project_id '{project_id}' must match pattern ^[A-Za-z0-9._-]+$")
        validate_windows_name(project_name, "project_name")

        cfg = settings or self.settings_manager.load()
        working_folder = derive_working_folder(source_input)
        fps = resolve_sources(source_input, cancellation_token=cancellation_token)
        if not fps:
            raise DiscoveryError("No supported source files discovered.")

        manual_output_dir = str(output_dir or cfg.output_dir or "").strip()
        resolved_output_dir = resolve_publication_root(
            manual_output_dir=manual_output_dir, working_folder=working_folder,
        )

        source_fps_map = {fp.basename: fp.to_dict() for fp in fps}

        # Probe actual durations for all discovered sources
        source_durations = {
            fp.basename: int(round(probe_media(fp.path, cancellation_token=cancellation_token).duration * 1000.0))
            for fp in fps
        }

        # Validate final_json against actual probed durations
        validate_project(final_json, source_durations=source_durations, cancellation_token=cancellation_token)

        # Check publication collision against ALL project sources including unused
        all_source_paths = [fp.path for fp in fps]
        check_publication_collision(final_json.get("outputs", []), resolved_output_dir, all_source_paths)

        # Save final JSON atomically to disk BEFORE voice/render
        self.persistence.save_final_json(project_id, final_json)

        gw_identifier = f"{cfg.gateway_endpoint}::{cfg.gateway_model}"
        voice_endpoint = f"{cfg.voice_mode}::{cfg.voice_local_url}::{cfg.voice_remote_url}"

        now_iso = datetime.now(timezone.utc).isoformat()
        state: Dict[str, Any] = {
            "schema_version": "3.0",
            "project_mode": "RECAP",
            "project_id": project_id,
            "project_name": project_name,
            "status": ProjectStatus.ANALYZED.value,
            "output_dir": str(resolved_output_dir),
            "manual_output_dir": manual_output_dir,
            "working_folder": str(working_folder),
            "sources": [
                {
                    "source_file": fp.basename,
                    "fingerprint": fp.to_dict(),
                    "duration_ms": source_durations[fp.basename],
                }
                for fp in fps
            ],
            "source_fingerprints": source_fps_map,
            "prompt": "",
            "prompt_hash": "",
            "gateway_identifier": gw_identifier,
            "settings_snapshot": cfg.to_dict(),
            "voice_endpoint": voice_endpoint,
            "timestamps": {
                "created_at": now_iso,
                "updated_at": now_iso,
                "analyzed_at": now_iso,
            },
            "sub_analysis": None,
            "raw_response": None,
            "final_json": final_json,
            "prepared_episodes": {},
            "outputs": {},
            "error": None,
        }

        check_for_secrets(state)
        self.persistence.save_project(state)
        return state

    def start_project(
        self,
        project_id: str,
        output_dir: Optional[Path | str] = None,
        settings: Optional[AppSettings] = None,
        cancellation_token: Optional[CancellationToken] = None,
        callbacks: Optional[WorkflowCallbacks] = None,
        source_preparation_pipeline: Optional[SourcePreparationPipeline] = None,
        progress_tracker: WorkflowProgressTracker | None = None,
    ) -> Dict[str, Any]:
        """Start or run project workflow synchronously."""
        return self._execute_workflow(
            project_id=project_id,
            output_dir=output_dir,
            settings=settings,
            cancellation_token=cancellation_token,
            callbacks=callbacks,
            source_preparation_pipeline=source_preparation_pipeline,
            progress_tracker=progress_tracker,
            run_mode="start",
        )

    def resume_project(
        self,
        project_id: str,
        output_dir: Optional[Path | str] = None,
        settings: Optional[AppSettings] = None,
        cancellation_token: Optional[CancellationToken] = None,
        callbacks: Optional[WorkflowCallbacks] = None,
        source_preparation_pipeline: Optional[SourcePreparationPipeline] = None,
        progress_tracker: WorkflowProgressTracker | None = None,
    ) -> Dict[str, Any]:
        """Resume project workflow, skipping completed outputs whose fingerprints match."""
        return self._execute_workflow(
            project_id=project_id,
            output_dir=output_dir,
            settings=settings,
            cancellation_token=cancellation_token,
            callbacks=callbacks,
            source_preparation_pipeline=source_preparation_pipeline,
            progress_tracker=progress_tracker,
            run_mode="resume",
        )

    def retry_project(
        self,
        project_id: str,
        output_dir: Optional[Path | str] = None,
        settings: Optional[AppSettings] = None,
        cancellation_token: Optional[CancellationToken] = None,
        callbacks: Optional[WorkflowCallbacks] = None,
        source_preparation_pipeline: Optional[SourcePreparationPipeline] = None,
        progress_tracker: WorkflowProgressTracker | None = None,
    ) -> Dict[str, Any]:
        """Retry failed outputs or rerender invalidated outputs."""
        return self._execute_workflow(
            project_id=project_id,
            output_dir=output_dir,
            settings=settings,
            cancellation_token=cancellation_token,
            callbacks=callbacks,
            source_preparation_pipeline=source_preparation_pipeline,
            progress_tracker=progress_tracker,
            run_mode="retry",
        )

    def get_project(self, project_id: str) -> Dict[str, Any]:
        """Load and return current project state."""
        validate_windows_name(project_id, "project_id")
        return self.persistence.load_project(project_id)

    def reconcile_state(self, project_id: str) -> Dict[str, Any]:
        """Reconcile interrupted states to resumable status."""
        validate_windows_name(project_id, "project_id")
        state = self.persistence.load_project(project_id)
        return reconcile_project_state(state, self.persistence)

    def _execute_workflow(
        self,
        project_id: str,
        output_dir: Optional[Path | str] = None,
        settings: Optional[AppSettings] = None,
        cancellation_token: Optional[CancellationToken] = None,
        callbacks: Optional[WorkflowCallbacks] = None,
        source_preparation_pipeline: Optional[SourcePreparationPipeline] = None,
        run_mode: str = "start",
        progress_tracker: WorkflowProgressTracker | None = None,
    ) -> Dict[str, Any]:
        """Synchronous execution engine for create/start/resume/retry."""
        validate_windows_name(project_id, "project_id")

        # Step 1: Load state and reconcile interrupted states
        state = self.persistence.load_project(project_id)
        state = reconcile_project_state(state, self.persistence)
        tracker_kwargs: Dict[str, Any] = {}
        if self.progress_monotonic is not None:
            tracker_kwargs["monotonic"] = self.progress_monotonic
        if self.progress_wall_clock is not None:
            tracker_kwargs["wall_clock"] = self.progress_wall_clock
        tracker = progress_tracker or WorkflowProgressTracker(
            self.persistence, project_id,
            callback=callbacks.on_activity if callbacks else None,
            **tracker_kwargs,
        )
        if not tracker.tick().get("active"):
            tracker.begin(resuming=run_mode in {"resume", "retry"})

        # Check if matching final_json exists (Resume/import NEVER Gateway once Final JSON exists)
        final_json = state.get("final_json")
        if final_json is None and self.persistence.has_final_json(project_id):
            final_json = self.persistence.load_final_json(project_id)
            state["final_json"] = final_json

        try:
            # If no Final JSON: verify sources, run sequential source prep, controlled stop before Scanner
            if final_json is None:
                cfg = settings or AppSettings.from_dict(state.get("settings_snapshot", {}))
                if cancellation_token:
                    cancellation_token.check_cancelled()

                # Source integrity check BEFORE any preparation (source mutation fails before prep)
                verify_source_integrity(state["source_fingerprints"], cancellation_token=cancellation_token)

                state["status"] = ProjectStatus.ANALYZING.value
                now_iso = datetime.now(timezone.utc).isoformat()
                state.setdefault("timestamps", {})["analyzing_started_at"] = now_iso
                state["timestamps"]["updated_at"] = now_iso
                self.persistence.save_project(state)
                if callbacks and callbacks.on_status_change:
                    callbacks.on_status_change(ProjectStatus.ANALYZING.value)

                # Resolve ordered sources with deterministic E01...En mapping
                if state.get("sources"):
                    ordered_sources = state["sources"]
                else:
                    fps_list = list(state.get("source_fingerprints", {}).values())
                    fps_list.sort(key=lambda x: natural_sort_key(x.get("basename", "")))
                    ordered_sources = [{"source_file": f.get("basename"), "fingerprint": f} for f in fps_list]

                tracker.emit(ProgressEvent(
                    state=ActivityState.LOCAL_PROCESSING.value,
                    stage=WorkflowStage.PREPARATION.value,
                    activity_text="Preparing source episodes...",
                    completed=0, total=len(ordered_sources), unit="episodes",
                ))

                prep_pipeline = (
                    source_preparation_pipeline
                    or self.source_preparation_pipeline
                    or SourcePreparationPipeline(cache_manager=AnalysisCacheManager(cache_dir=self.persistence.root / "cache" / "analysis"), allow_stt=False)
                )

                prep_episodes: Dict[str, Any] = state.setdefault("prepared_episodes", {})

                for i, src_entry in enumerate(ordered_sources, start=1):
                    if cancellation_token:
                        cancellation_token.check_cancelled()

                    episode_id = f"E{i:02d}"
                    tracker.emit(ProgressEvent(
                        state=ActivityState.LOCAL_PROCESSING.value,
                        stage=WorkflowStage.PREPARATION.value,
                        activity_text="Preparing episode media and transcript...",
                        episode_id=episode_id, current_item=episode_id,
                        completed=i - 1, total=len(ordered_sources), unit="episodes",
                        item_event="start", item_key=episode_id,
                    ))
                    src_fp = src_entry.get("fingerprint", {})
                    src_file = src_entry.get("source_file") or (src_fp.get("basename") if isinstance(src_fp, dict) else "")
                    src_map_entry = state["source_fingerprints"].get(src_file, {})
                    src_path = (
                        src_fp.get("path")
                        if isinstance(src_fp, dict) and src_fp.get("path")
                        else src_map_entry.get("path")
                    )
                    src_sha = (
                        src_fp.get("sha256")
                        if isinstance(src_fp, dict) and src_fp.get("sha256")
                        else src_map_entry.get("sha256")
                    )

                    def _progress_cb(phase: str, pct: float, msg: str) -> None:
                        if callbacks and callbacks.on_source_preparation_progress:
                            callbacks.on_source_preparation_progress(episode_id, phase, pct, msg)
                        tracker.emit(ProgressEvent(
                            state=ActivityState.LOCAL_PROCESSING.value,
                            stage=WorkflowStage.PREPARATION.value,
                            activity_text=msg or f"Preparing {episode_id}: {phase}",
                            episode_id=episode_id, current_item=episode_id,
                            completed=i - 1, total=len(ordered_sources), unit="episodes",
                        ))

                    try:
                        prepared_ep = prep_pipeline.prepare_episode(
                            source_path=src_path,
                            episode_id=episode_id,
                            source_id=episode_id,
                            source_fingerprint=src_sha,
                            force_refresh=False,
                            on_progress=_progress_cb,
                            cancellation_token=cancellation_token,
                        )
                    except CancelledError:
                        raise
                    except Exception as e:
                        state["status"] = ProjectStatus.FAILED.value
                        state["error"] = f"Source preparation failed for {episode_id}: {e}"
                        now_err = datetime.now(timezone.utc).isoformat()
                        state.setdefault("timestamps", {})["failed_at"] = now_err
                        state["timestamps"]["updated_at"] = now_err
                        self.persistence.save_project(state)
                        if callbacks and callbacks.on_status_change:
                            callbacks.on_status_change(ProjectStatus.FAILED.value)
                        if callbacks and callbacks.on_error:
                            callbacks.on_error(e, None)
                        raise

                    if prepared_ep.status == "failed":
                        err = ToolRecapError(f"Episode {episode_id} source preparation produced failed status")
                        state["status"] = ProjectStatus.FAILED.value
                        state["error"] = str(err)
                        now_err = datetime.now(timezone.utc).isoformat()
                        state.setdefault("timestamps", {})["failed_at"] = now_err
                        state["timestamps"]["updated_at"] = now_err
                        self.persistence.save_project(state)
                        if callbacks and callbacks.on_status_change:
                            callbacks.on_status_change(ProjectStatus.FAILED.value)
                        if callbacks and callbacks.on_error:
                            callbacks.on_error(err, None)
                        raise err

                    has_aud = False
                    if prepared_ep.audio_selection:
                        has_aud = prepared_ep.audio_selection.has_audio
                    elif prepared_ep.audio_info:
                        has_aud = True

                    ep_summary = {
                        "episode_id": episode_id,
                        "source_file": src_file or Path(src_path).name,
                        "source_fingerprint": src_sha,
                        "duration_ms": prepared_ep.duration_ms,
                        "status": prepared_ep.status,
                        "transcript_method": prepared_ep.transcript_method,
                        "artifact_hash": prepared_ep.artifact_hash,
                        "cues_count": prepared_ep.transcript.cue_count,
                        "canvas_width": prepared_ep.canvas_width,
                        "canvas_height": prepared_ep.canvas_height,
                        "has_audio": has_aud,
                        "source_timeline": dict(prepared_ep.source_timeline),
                        "prepared_at": datetime.now(timezone.utc).isoformat(),
                    }
                    from toolrecap_v4.editorial import transcript_structure
                    for source_entry in state["sources"]:
                        if source_entry.get("source_file") == ep_summary["source_file"]:
                            source_entry["structure_signals"] = transcript_structure(prepared_ep.transcript.cues)
                    # Persist full local PreparedEpisode/Transcript; project state keeps a lightweight summary.
                    self.persistence.save_prepared_episode(project_id, episode_id, prepared_ep.to_dict())
                    prep_episodes[episode_id] = ep_summary
                    state["timestamps"]["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self.persistence.save_project(state)

                    if callbacks and callbacks.on_episode_prepared:
                        callbacks.on_episode_prepared(episode_id, ep_summary)
                    tracker.emit(ProgressEvent(
                        state=ActivityState.RUNNING.value,
                        stage=WorkflowStage.PREPARATION.value,
                        activity_text="Episode preparation checkpoint saved.",
                        episode_id=episode_id, current_item=episode_id,
                        completed=i, total=len(ordered_sources), unit="episodes",
                        item_event="complete", item_key=episode_id,
                    ))

                # All episodes successfully prepared -> persist manifest and checkpoint
                manifest_data = {
                    "project_id": project_id,
                    "status": "completed",
                    "episodes": list(prep_episodes.values()),
                    "total_episodes": len(ordered_sources),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                self.persistence.save_prepared_manifest(project_id, manifest_data)
                self.persistence.save_checkpoint(project_id, "source_preparation", {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "episodes": [ep["episode_id"] for ep in prep_episodes.values()],
                })
                tracker.emit(ProgressEvent(
                    state=ActivityState.LOCAL_PROCESSING.value,
                    stage=WorkflowStage.PREPARATION.value,
                    activity_text="All episode preparation checkpoints are complete.",
                    completed=len(ordered_sources), total=len(ordered_sources),
                    unit="episodes", stage_status="complete",
                ))

                # A fresh install requires an explicit provider-neutral Scanner model.
                scanner = self.scanner_service
                if scanner is None and cfg.scanner_model.strip():
                    scanner = ScannerService(
                        gateway_client=self.gateway_client,
                        storage_root=self.persistence.root,
                        config=ScannerConfig(
                            model=cfg.scanner_model,
                            reasoning=cfg.scanner_reasoning,
                            parallelism=cfg.scanner_parallelism,
                            chunk_policy=ScannerChunkPolicy(
                                max_duration_ms=cfg.scanner_chunk_duration_ms,
                                max_request_bytes=cfg.scanner_max_request_bytes,
                            ),
                            repair_attempts=cfg.scanner_repair_attempts,
                        ),
                        progress_callback=tracker.emit_payload,
                    )

                if scanner is not None and hasattr(scanner, "progress_callback"):
                    scanner.progress_callback = tracker.emit_payload

                if scanner is None:
                    scanner_msg = "Analysis pipeline stopped: source preparation complete (PREPARED); Scanner is unavailable until a Scanner model is configured."
                    now_prep = datetime.now(timezone.utc).isoformat()
                    state["status"] = ProjectStatus.PREPARED.value
                    state["error"] = scanner_msg
                    state.setdefault("timestamps", {})["prepared_at"] = now_prep
                    state["timestamps"]["updated_at"] = now_prep
                    self.persistence.save_project(state)
                    if callbacks and callbacks.on_status_change:
                        callbacks.on_status_change(ProjectStatus.PREPARED.value)
                    scanner_err = AnalysisPipelineUnavailableError(scanner_msg)
                    if callbacks and callbacks.on_error:
                        callbacks.on_error(scanner_err, None)
                    raise scanner_err

                prepared_models = [
                    PreparedEpisode.from_dict(self.persistence.load_prepared_episode(project_id, episode_id))
                    for episode_id in sorted(prep_episodes)
                ]
                try:
                    evidence_result = scanner.scan_project(
                        project_id,
                        prepared_models,
                        cancellation_token=cancellation_token,
                    )
                except CancelledError:
                    raise
                except Exception as exc:
                    state["status"] = ProjectStatus.FAILED.value
                    state["error"] = f"Scanner failed: {exc}"
                    now_err = datetime.now(timezone.utc).isoformat()
                    state.setdefault("timestamps", {})["failed_at"] = now_err
                    state["timestamps"]["updated_at"] = now_err
                    self.persistence.save_project(state)
                    if callbacks and callbacks.on_status_change:
                        callbacks.on_status_change(ProjectStatus.FAILED.value)
                    if callbacks and callbacks.on_error:
                        callbacks.on_error(exc, None)
                    raise

                evidence_summary = {
                    "status": "completed",
                    "evidence_revision": evidence_result.evidence_revision,
                    "episode_counts": evidence_result.episode_counts,
                    "total_evidence_count": evidence_result.total_evidence_count,
                    "reused_chunk_count": evidence_result.reused_chunk_count,
                    "requested_chunk_count": evidence_result.requested_chunk_count,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                self.persistence.save_checkpoint(project_id, "evidence", evidence_summary)
                now_ready = datetime.now(timezone.utc).isoformat()
                phase5_msg = "Full Episode Evidence is complete (EVIDENCE_READY); Season Evidence Catalog is pending."
                state["status"] = ProjectStatus.EVIDENCE_READY.value
                state["evidence"] = evidence_summary
                state["error"] = phase5_msg
                state.setdefault("timestamps", {})["evidence_ready_at"] = now_ready
                state["timestamps"]["updated_at"] = now_ready
                self.persistence.save_project(state)
                if callbacks and callbacks.on_evidence_ready:
                    callbacks.on_evidence_ready(evidence_result.evidence_revision, evidence_result.episode_counts)
                if callbacks and callbacks.on_status_change:
                    callbacks.on_status_change(ProjectStatus.EVIDENCE_READY.value)

                catalog_service = self.catalog_service or CatalogService(self.persistence.root)
                tracker.emit(ProgressEvent(
                    state=ActivityState.LOCAL_PROCESSING.value,
                    stage=WorkflowStage.CATALOG.value,
                    activity_text="Building the complete Season Evidence Catalog...",
                ))
                try:
                    catalog_result = catalog_service.build_or_load(
                        project_id=project_id,
                        evidence_revision=evidence_result.evidence_revision,
                        ordered_episodes=prepared_models,
                        cancellation_token=cancellation_token,
                    )
                except CancelledError:
                    raise
                except Exception as exc:
                    state["status"] = ProjectStatus.FAILED.value
                    state["error"] = f"Catalog failed: {exc}"
                    now_err = datetime.now(timezone.utc).isoformat()
                    state.setdefault("timestamps", {})["failed_at"] = now_err
                    state["timestamps"]["updated_at"] = now_err
                    self.persistence.save_project(state)
                    if callbacks and callbacks.on_status_change:
                        callbacks.on_status_change(ProjectStatus.FAILED.value)
                    if callbacks and callbacks.on_error:
                        callbacks.on_error(exc, None)
                    raise

                catalog_summary = {
                    "status": "completed",
                    "catalog_version": catalog_result.catalog.catalog_version,
                    "catalog_hash": catalog_result.catalog.catalog_hash,
                    "evidence_revision": catalog_result.catalog.evidence_revision,
                    "episode_count": len(catalog_result.catalog.ordered_episodes),
                    "evidence_count": len(catalog_result.catalog.items),
                    "ids_digest": catalog_result.catalog.completeness.ids_digest,
                    "capacity": catalog_result.capacity.to_dict(),
                    "reused": catalog_result.reused,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                self.persistence.save_checkpoint(project_id, "catalog", catalog_summary)
                now_catalog = datetime.now(timezone.utc).isoformat()
                phase6_msg = "Analysis pipeline stopped: Complete Season Evidence Catalog is ready (CATALOG_READY); Phase 6 Season Planner is not implemented."
                state["status"] = ProjectStatus.CATALOG_READY.value
                state["catalog"] = catalog_summary
                state["error"] = phase6_msg
                state.setdefault("timestamps", {})["catalog_ready_at"] = now_catalog
                state["timestamps"]["updated_at"] = now_catalog
                self.persistence.save_project(state)
                if callbacks and callbacks.on_catalog_ready:
                    callbacks.on_catalog_ready(catalog_result.catalog.catalog_hash, catalog_summary)
                if callbacks and callbacks.on_status_change:
                    callbacks.on_status_change(ProjectStatus.CATALOG_READY.value)
                tracker.emit(ProgressEvent(
                    state=ActivityState.LOCAL_PROCESSING.value,
                    stage=WorkflowStage.CATALOG.value,
                    activity_text=("Reused Season Catalog checkpoint." if catalog_result.reused else "Season Catalog checkpoint completed."),
                    completed=len(catalog_result.catalog.items), total=len(catalog_result.catalog.items),
                    unit="evidence items", reused=len(catalog_result.catalog.items) if catalog_result.reused else 0,
                    stage_status="complete",
                ))

                planner_service = self.planner_service
                if planner_service is None and cfg.planner_model.strip():
                    planner_service = PlannerService(
                        gateway_client=self.gateway_client,
                        storage_root=self.persistence.root,
                        config=PlannerConfig(
                            model=cfg.planner_model,
                            reasoning=cfg.planner_reasoning,
                            max_rounds=cfg.planner_max_rounds,
                            repair_attempts=cfg.planner_repair_attempts,
                            max_request_bytes=cfg.planner_max_request_bytes,
                        ),
                        progress_callback=tracker.emit_payload,
                    )
                if planner_service is not None and hasattr(planner_service, "progress_callback"):
                    planner_service.progress_callback = tracker.emit_payload
                if planner_service is None:
                    phase6_err = AnalysisPipelineUnavailableError(phase6_msg)
                    if callbacks and callbacks.on_error:
                        callbacks.on_error(phase6_err, None)
                    raise phase6_err

                try:
                    planner_result = planner_service.run(
                        project_id=project_id,
                        raw_recap_prompt=state.get("prompt", ""),
                        catalog=catalog_result.catalog,
                        cancellation_token=cancellation_token,
                    )
                except CancelledError:
                    raise
                except Exception as exc:
                    state["status"] = ProjectStatus.FAILED.value
                    state["error"] = f"Season Planner failed: {exc}"
                    now_err = datetime.now(timezone.utc).isoformat()
                    state.setdefault("timestamps", {})["failed_at"] = now_err
                    state["timestamps"]["updated_at"] = now_err
                    self.persistence.save_project(state)
                    if callbacks and callbacks.on_status_change:
                        callbacks.on_status_change(ProjectStatus.FAILED.value)
                    if callbacks and callbacks.on_error:
                        callbacks.on_error(exc, None)
                    raise

                draft_summary = {
                    "status": "completed",
                    "session_id": planner_result.session_id,
                    "dependency_digest": planner_result.dependency_digest,
                    "round_count": planner_result.round_count,
                    "proposed_output_count": planner_result.draft.proposed_output_count,
                    "draft_path": str(planner_result.draft_path),
                    "reused": planner_result.reused,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                self.persistence.save_checkpoint(project_id, "planner_draft", draft_summary)
                now_draft = datetime.now(timezone.utc).isoformat()
                phase7_msg = "Analysis pipeline stopped: Planner Draft is ready (PLANNER_DRAFT_READY); Phase 7 selective visual evidence is not implemented."
                state["status"] = ProjectStatus.PLANNER_DRAFT_READY.value
                state["planner_draft"] = draft_summary
                state["error"] = phase7_msg
                state.setdefault("timestamps", {})["planner_draft_ready_at"] = now_draft
                state["timestamps"]["updated_at"] = now_draft
                self.persistence.save_project(state)
                if callbacks and callbacks.on_planner_draft_ready:
                    callbacks.on_planner_draft_ready(planner_result.session_id, draft_summary)
                if callbacks and callbacks.on_status_change:
                    callbacks.on_status_change(ProjectStatus.PLANNER_DRAFT_READY.value)
                has_visual = any(o.visual_requests for o in planner_result.draft.proposed_outputs)
                visual_service = self.visual_service
                if visual_service is None and (cfg.vision_model.strip() or not has_visual):
                    visual_service = VisualEvidenceService(
                        self.gateway_client, self.persistence.root,
                        VisionConfig(cfg.vision_model, cfg.vision_reasoning, cfg.vision_repair_attempts,
                            FrameExtractionPolicy(cfg.vision_frames_per_range,cfg.vision_frames_per_episode,cfg.vision_hard_frame_cap)),
                        progress_callback=tracker.emit_payload,
                    )
                if visual_service is not None and hasattr(visual_service, "progress_callback"):
                    visual_service.progress_callback = tracker.emit_payload
                if visual_service is None:
                    phase7_err = AnalysisPipelineUnavailableError(phase7_msg + " Vision model is not configured.")
                    if callbacks and callbacks.on_error: callbacks.on_error(phase7_err,None)
                    raise phase7_err
                planner_draft_hash = hashlib.sha256(json.dumps(planner_result.draft.to_dict(),ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
                visual_result = visual_service.run(project_id=project_id,draft=planner_result.draft,planner_draft_hash=planner_draft_hash,episodes=prepared_models,evidence_revision=evidence_result.evidence_revision,cancellation_token=cancellation_token)
                plan_service = self.season_plan_service or SeasonPlanService(
                    self.gateway_client, self.persistence.root, cfg.planner_model,
                    cfg.planner_reasoning, repair_attempts=cfg.planner_repair_attempts,
                    max_request_bytes=cfg.planner_max_request_bytes,
                    progress_callback=tracker.emit_payload,
                )
                if hasattr(plan_service, "progress_callback"):
                    plan_service.progress_callback = tracker.emit_payload
                plan_result = plan_service.run(project_id=project_id,raw_recap_prompt=state.get("prompt",""),catalog=catalog_result.catalog,draft=planner_result.draft,visual=visual_result,cancellation_token=cancellation_token)
                plan_summary={"status":"completed","plan_hash":plan_result.plan.plan_hash,"visual_revision":visual_result.visual_revision,"output_count":len(plan_result.plan.outputs),"season_plan_path":str(plan_result.path),"reused":plan_result.reused,"completed_at":datetime.now(timezone.utc).isoformat()}
                self.persistence.save_checkpoint(project_id,"season_plan",plan_summary)
                now_plan=datetime.now(timezone.utc).isoformat();phase8_msg="Analysis pipeline stopped: locked Season Plan is ready (SEASON_PLAN_READY); Phase 8 Output Writers are not implemented."
                state["status"]=ProjectStatus.SEASON_PLAN_READY.value;state["season_plan"]=plan_summary;state["error"]=phase8_msg;state.setdefault("timestamps",{})["season_plan_ready_at"]=now_plan;state["timestamps"]["updated_at"]=now_plan;self.persistence.save_project(state)
                if callbacks and callbacks.on_season_plan_ready:callbacks.on_season_plan_ready(plan_result.plan.plan_hash,plan_summary)
                if callbacks and callbacks.on_status_change:callbacks.on_status_change(ProjectStatus.SEASON_PLAN_READY.value)
                writer_service=self.writer_service
                if writer_service is None and cfg.writer_model.strip():writer_service=WriterService(self.gateway_client,self.persistence.root,WriterConfig(cfg.writer_model,cfg.writer_reasoning,cfg.writer_parallelism,cfg.writer_max_request_bytes,cfg.writer_max_response_bytes),progress_callback=tracker.emit_payload)
                if writer_service is not None and hasattr(writer_service,"progress_callback"):writer_service.progress_callback=tracker.emit_payload
                if writer_service is None:
                    phase8_err=AnalysisPipelineUnavailableError(phase8_msg)
                    if callbacks and callbacks.on_error:callbacks.on_error(phase8_err,None)
                    raise phase8_err
                writer_result=writer_service.run(project_id=project_id,raw_prompt=state.get("prompt",""),language=cfg.recap_language,plan=plan_result.plan,episodes=prepared_models,visual=visual_result,cancellation_token=cancellation_token)
                writer_manifest=getattr(writer_result,"manifest",{"status":"COMPLETE","completed_response_count":len(writer_result.artifacts),"durable_response_count":len(writer_result.artifacts)})
                writer_summary={"status":"completed" if writer_manifest.get("status")=="COMPLETE" else "repair_required","season_plan_hash":plan_result.plan.plan_hash,"expected_output_count":len(plan_result.plan.outputs),"completed_response_count":writer_manifest.get("completed_response_count",0),"durable_response_count":writer_manifest.get("durable_response_count",len(writer_result.artifacts)),"repair_required_count":len(plan_result.plan.outputs)-writer_manifest.get("completed_response_count",0),"reused_count":writer_result.reused_count,"requested_count":writer_result.requested_count,"completed_at":datetime.now(timezone.utc).isoformat()}
                self.persistence.save_checkpoint(project_id,"writer_drafts",writer_summary);now_w=datetime.now(timezone.utc).isoformat();phase9_msg="Analysis pipeline stopped: per-output Writer responses are ready (WRITER_DRAFTS_READY); Phase 9 validation/repair/merge is not implemented."
                state["status"]=ProjectStatus.WRITER_DRAFTS_READY.value;state["writer_drafts"]=writer_summary;state["error"]=phase9_msg;state.setdefault("timestamps",{})["writer_drafts_ready_at"]=now_w;state["timestamps"]["updated_at"]=now_w;self.persistence.save_project(state)
                if callbacks and callbacks.on_writer_drafts_ready:callbacks.on_writer_drafts_ready(writer_summary)
                if callbacks and callbacks.on_status_change:callbacks.on_status_change(ProjectStatus.WRITER_DRAFTS_READY.value)
                finalizer=self.finalization_service or FinalizationService(self.gateway_client,self.persistence.root,FinalizationConfig(cfg.writer_model,cfg.writer_reasoning,cfg.writer_repair_attempts,cfg.writer_max_request_bytes,cfg.writer_max_response_bytes),progress_callback=tracker.emit_payload)
                if hasattr(finalizer,"progress_callback"):finalizer.progress_callback=tracker.emit_payload
                finalized=finalizer.run(project_id=project_id,project_name=state["project_name"],raw_prompt=state.get("prompt",""),language=cfg.recap_language,plan=plan_result.plan,episodes=prepared_models,visual=visual_result,cancellation_token=cancellation_token)
                final_summary={"status":"completed","revision":finalized.revision,"artifact_hash":finalized.artifact_hash,"output_count":len(finalized.final_json["outputs"]),"repaired_output_ids":list(finalized.repaired_output_ids),"reused":finalized.reused,"completed_at":datetime.now(timezone.utc).isoformat()}
                self.persistence.save_checkpoint(project_id,"final_json",final_summary);now_f=datetime.now(timezone.utc).isoformat();phase10_msg="Analysis pipeline stopped: canonical schema 3.0 Final JSON is ready (FINAL_JSON_READY); Phase 10 settings migration is not implemented."
                state["status"]=ProjectStatus.FINAL_JSON_READY.value;state["final_json"]=finalized.final_json;state["finalization"]=final_summary;state["error"]=phase10_msg;state.setdefault("timestamps",{})["final_json_ready_at"]=now_f;state["timestamps"]["updated_at"]=now_f;self.persistence.save_project(state)
                if callbacks and callbacks.on_final_json_ready:callbacks.on_final_json_ready(final_summary)
                if callbacks and callbacks.on_status_change:callbacks.on_status_change(ProjectStatus.FINAL_JSON_READY.value)
                tracker.emit(ProgressEvent(
                    state=ActivityState.LOCAL_PROCESSING.value,
                    stage=WorkflowStage.FINAL_JSON.value,
                    activity_text="Final JSON Ready — analysis complete.",
                    completed=len(finalized.final_json["outputs"]),
                    total=len(finalized.final_json["outputs"]), unit="outputs",
                    reused=len(finalized.final_json["outputs"]) if finalized.reused else 0,
                    stage_status="complete", analysis_complete=True,
                ))
                # Phase 11 boundary: canonical Final JSON is now the sole
                # downstream authority and enters the same path as imports.
                final_json = finalized.final_json

            # Step 2: Source integrity check (source changes fail no substitution)
            from toolrecap_v4.review import plan_digest
            review_hash = plan_digest(state)
            review_settings = settings or AppSettings.from_dict(state.get("settings_snapshot", {}))
            if review_settings.review_before_render and state.get("review_approved_hash") != review_hash:
                state["status"] = ProjectStatus.REVIEW_REQUIRED.value
                state["review_plan_hash"] = review_hash
                state["timestamps"]["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.persistence.save_project(state)
                tracker.terminate(ActivityState.PAUSED, activity_text="Plan ready for review before rendering.")
                return state
            verify_source_integrity(state["source_fingerprints"], cancellation_token=cancellation_token)

            # Step 3: Resolve settings and output directory
            cfg = settings or AppSettings.from_dict(state.get("settings_snapshot", {}))
            source_durations = {
                item["source_file"]: int(item.get("duration_ms", 0))
                for item in state.get("sources", [])
            }
            validate_project(final_json, source_durations=source_durations, cancellation_token=cancellation_token)
            final_json_hash = hashlib.sha256(
                json.dumps(final_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if output_dir is not None:
                manual_output_dir = str(output_dir).strip()
            elif settings is not None:
                manual_output_dir = str(settings.output_dir or "").strip()
            else:
                manual_output_dir = str(state.get("manual_output_dir", "")).strip()
            working_folder = state.get("working_folder")
            if not working_folder:
                working_folder = derive_working_folder(
                    [fp["path"] for fp in state["source_fingerprints"].values()]
                )
            resolved_output_dir = ensure_publication_root(resolve_publication_root(
                manual_output_dir=manual_output_dir, working_folder=working_folder,
            ))
            state["output_dir"] = str(resolved_output_dir)
            state["manual_output_dir"] = manual_output_dir
            state["working_folder"] = str(working_folder)

            # Step 4: Check publication collision against ALL project sources (including unused)
            all_source_paths = [fp["path"] for fp in state["source_fingerprints"].values()]
            check_publication_collision(final_json.get("outputs", []), resolved_output_dir, all_source_paths)

            # Step 5: Render outputs sequentially
            if cancellation_token:
                cancellation_token.check_cancelled()

            state["status"] = ProjectStatus.RENDERING.value
            now_iso = datetime.now(timezone.utc).isoformat()
            state["timestamps"]["rendering_started_at"] = now_iso
            state["timestamps"]["updated_at"] = now_iso
            self.persistence.save_project(state)
            if callbacks and callbacks.on_status_change:
                callbacks.on_status_change(ProjectStatus.RENDERING.value)

            source_paths_mapping = {
                fp["basename"]: fp["path"]
                for fp in state["source_fingerprints"].values()
            }
            prepared_for_dialogue = [
                PreparedEpisode.from_dict(item)
                for item in self.persistence.list_prepared_episodes(project_id)
            ]
            source_dialogue_map = source_dialogue_map_from_episodes(prepared_for_dialogue)
            source_dialogue_input = source_dialogue_map if prepared_for_dialogue else None
            source_dialogue_digest = hashlib.sha256(json.dumps(
                {source: [cue.to_dict() for cue in cues] for source, cues in source_dialogue_map.items()},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()

            outputs_def = final_json.get("outputs", [])
            outputs_state = state.setdefault("outputs", {})
            project_mode = str(state.get("project_mode", "RECAP")).upper()
            is_highlight = project_mode == "HIGHLIGHT"
            if is_highlight:
                tracker.emit(ProgressEvent(
                    state=ActivityState.LOCAL_PROCESSING.value,
                    stage=WorkflowStage.VOICE.value,
                    activity_text="Highlight uses original source audio; Voice generation skipped.",
                    completed=0, total=0, unit="segments", stage_status="skipped",
                ))
                tracker.emit(ProgressEvent(
                    state=ActivityState.LOCAL_PROCESSING.value,
                    stage=WorkflowStage.NARRATION_FIT.value,
                    activity_text="Highlight contains no narration; Narration Fit skipped.",
                    completed=0, total=0, unit="segments", stage_status="skipped",
                ))
            else:
                tracker.emit(ProgressEvent(
                    state=ActivityState.RUNNING.value,
                    stage=WorkflowStage.VOICE.value,
                    activity_text=("No downstream outputs are required." if not outputs_def else "Preparing downstream output queue..."),
                    completed=0, total=len(outputs_def), unit="outputs",
                    stage_status="skipped" if not outputs_def else None,
                ))

            published_count = 0
            for output_index, out_def in enumerate(outputs_def, start=1):
                if cancellation_token:
                    cancellation_token.check_cancelled()

                render_id = out_def["render_id"]
                title = out_def["title"]

                # Compute current render fingerprint
                cur_fp = compute_output_fingerprint(
                    out_def, state["source_fingerprints"], cfg, source_dialogue_digest,
                )

                # Check existing checkpoint
                ckpt = None
                try:
                    ckpt = self.persistence.load_checkpoint(project_id, render_id)
                except Exception:
                    ckpt = outputs_state.get(render_id)

                intended_video = (resolved_output_dir / f"{title}.mp4").resolve()
                from toolrecap_v4.publication import recover
                publication_owner = f"{project_id}:{render_id}:{cur_fp}"
                recovered_publication = recover([
                    intended_video, (resolved_output_dir / f"{title}.narration.srt").resolve(),
                    (resolved_output_dir / f"{title}.original.srt").resolve(),
                ], publication_owner, self.persistence.root)
                if recovered_publication and (
                    not ckpt or ckpt.get("status") not in (OutputStatus.COMPLETED.value, OutputStatus.SKIPPED.value)
                ):
                    narration_path = (resolved_output_dir / f"{title}.narration.srt").resolve()
                    original_path = (resolved_output_dir / f"{title}.original.srt").resolve()
                    verified_probe = probe_media(intended_video, cancellation_token=cancellation_token)
                    original_text = original_path.read_text(encoding="utf-8")
                    original_count = len([block for block in original_text.strip().split("\n\n") if block.strip()])
                    ckpt = {
                        "render_id": render_id, "title": title,
                        "status": OutputStatus.COMPLETED.value,
                        "output_path": str(intended_video),
                        "narration_srt_path": str(narration_path),
                        "original_srt_path": str(original_path),
                        "narration_srt_hash": compute_file_sha256(narration_path),
                        "original_srt_hash": compute_file_sha256(original_path),
                        "original_subtitle_state": (
                            "ORIGINAL_DIALOGUE_MAPPED" if original_count else "NO_ORIGINAL_DIALOGUE"
                        ),
                        "original_subtitle_cue_count": original_count,
                        "duration": verified_probe.duration, "fingerprint": cur_fp,
                        "output_file_hash": compute_file_sha256(intended_video),
                        "video_codec": verified_probe.video_streams[0].codec,
                        "audio_codec": verified_probe.audio_streams[0].codec,
                        "width": verified_probe.width, "height": verified_probe.height,
                        "fps": verified_probe.fps,
                    }
                    self.persistence.save_checkpoint(project_id, render_id, ckpt)
                    outputs_state[render_id] = ckpt
                    self.persistence.save_project(state)
                checkpoint_owns_destination = bool(
                    ckpt and ckpt.get("output_path")
                    and Path(ckpt["output_path"]).resolve() == intended_video
                )
                checkpoint_owns_destination = checkpoint_owns_destination or recovered_publication
                if not checkpoint_owns_destination:
                    unowned = next((path for path in (
                        resolved_output_dir / f"{title}.mp4",
                        resolved_output_dir / f"{title}.narration.srt",
                        resolved_output_dir / f"{title}.original.srt",
                    ) if path.exists()), None)
                    if unowned is not None:
                        raise SourceCollisionError(
                            f"Publication target '{unowned}' already exists without a matching project checkpoint."
                        )

                # Validate completed output presence and hash before skip
                should_skip = False
                if ckpt and ckpt.get("status") in (OutputStatus.COMPLETED.value, OutputStatus.SKIPPED.value):
                    ckpt_fp = ckpt.get("fingerprint")
                    out_path_str = ckpt.get("output_path")
                    expected_hash = ckpt.get("output_file_hash")

                    if ckpt_fp == cur_fp and out_path_str and expected_hash:
                        out_path = Path(out_path_str)
                        expected_path = (resolved_output_dir / f"{title}.mp4").resolve()
                        narration_srt = Path(ckpt.get("narration_srt_path", ""))
                        original_srt = Path(ckpt.get("original_srt_path", ""))
                        subtitle_state = ckpt.get("original_subtitle_state")
                        subtitle_hashes_valid = (
                            narration_srt.is_file() and original_srt.is_file()
                            and ckpt.get("narration_srt_hash") == compute_file_sha256(narration_srt)
                            and ckpt.get("original_srt_hash") == compute_file_sha256(original_srt)
                            and subtitle_state in {"ORIGINAL_DIALOGUE_MAPPED", "NO_ORIGINAL_DIALOGUE"}
                            and (
                                subtitle_state != "ORIGINAL_DIALOGUE_MAPPED"
                                or (int(ckpt.get("original_subtitle_cue_count", 0)) > 0 and original_srt.stat().st_size > 0)
                            )
                        )
                        if out_path.resolve() == expected_path and out_path.is_file() and subtitle_hashes_valid:
                            actual_hash = compute_file_sha256(out_path)
                            if actual_hash == expected_hash:
                                should_skip = True

                if should_skip:
                    skip_ckpt = {
                        **ckpt,
                        "status": OutputStatus.SKIPPED.value,
                    }
                    self.persistence.save_checkpoint(project_id, render_id, skip_ckpt)
                    outputs_state[render_id] = skip_ckpt
                    state["timestamps"]["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self.persistence.save_project(state)
                    if callbacks and callbacks.on_output_skipped:
                        callbacks.on_output_skipped(render_id, skip_ckpt)
                    published_count += 1
                    tracker.emit(ProgressEvent(
                        state=ActivityState.RUNNING.value,
                        stage=WorkflowStage.PUBLISH.value,
                        activity_text="Reused completed render and publication checkpoint.",
                        output_id=render_id, current_item=render_id,
                        completed=published_count, total=len(outputs_def), unit="outputs",
                        reused=published_count, item_reused=True,
                    ))
                    continue

                # Render output
                if callbacks and callbacks.on_output_started:
                    callbacks.on_output_started(render_id)

                in_progress_ckpt = {
                    "render_id": render_id,
                    "title": title,
                    "status": OutputStatus.RENDERING.value,
                    "fingerprint": cur_fp,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    # Record ownership before rendering so a crash after atomic
                    # publication but before COMPLETE can be resumed safely.
                    "output_path": str(intended_video),
                }
                self.persistence.save_checkpoint(project_id, render_id, in_progress_ckpt)
                outputs_state[render_id] = in_progress_ckpt
                self.persistence.save_project(state)

                try:
                    narration_audio_map: dict[str, Path] = {}
                    if not is_highlight:
                        voice_cache = DownstreamVoiceCache(
                            self.persistence.root, project_id,
                            progress_callback=tracker.emit_payload,
                        )
                        voice_result = voice_cache.prepare_output(
                            output_def=out_def,
                            final_json_hash=final_json_hash,
                            settings=cfg,
                            voice_adapter=self.voice_adapter,
                            source_dialogue_map=source_dialogue_input,
                            cancellation_token=cancellation_token,
                        )
                        narration_audio_map = voice_result.narration_audio_map
                    render_res = render_output(
                        output_def=out_def,
                        source_paths=source_paths_mapping,
                        output_dir=resolved_output_dir,
                        settings=cfg,
                        voice_adapter=self.voice_adapter,
                        narration_audio_map=narration_audio_map,
                        publication_owner=publication_owner,
                        publication_journal_root=self.persistence.root,
                        allow_publication_overwrite=checkpoint_owns_destination,
                        source_dialogue_map=source_dialogue_input,
                        cancellation_token=cancellation_token,
                        progress_callback=tracker.emit_payload,
                        render_mode=project_mode,
                    )
                except CancelledError:
                    raise
                except Exception as e:
                    # Output failure stops queue preserves complete previous outputs!
                    fail_ckpt = {
                        "render_id": render_id,
                        "title": title,
                        "status": OutputStatus.FAILED.value,
                        "error": str(e),
                        "fingerprint": cur_fp,
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "output_path": str(intended_video),
                    }
                    self.persistence.save_checkpoint(project_id, render_id, fail_ckpt)
                    outputs_state[render_id] = fail_ckpt
                    state["status"] = ProjectStatus.FAILED.value
                    state["error"] = f"Output '{render_id}' failed: {e}"
                    state["timestamps"]["failed_at"] = datetime.now(timezone.utc).isoformat()
                    self.persistence.save_project(state)
                    if callbacks and callbacks.on_output_failed:
                        callbacks.on_output_failed(render_id, str(e))
                    raise

                # Render succeeded
                out_file_hash = compute_file_sha256(render_res.output_path)
                now_iso = datetime.now(timezone.utc).isoformat()
                completed_ckpt = {
                    "render_id": render_id,
                    "title": title,
                    "status": OutputStatus.COMPLETED.value,
                    "output_path": str(render_res.output_path),
                    "narration_srt_path": str(render_res.narration_srt_path),
                    "original_srt_path": str(render_res.original_srt_path),
                    "narration_srt_hash": compute_file_sha256(render_res.narration_srt_path),
                    "original_srt_hash": compute_file_sha256(render_res.original_srt_path),
                    "original_subtitle_state": render_res.original_subtitle_state,
                    "original_subtitle_cue_count": render_res.original_subtitle_cue_count,
                    "duration": render_res.duration,
                    "fingerprint": cur_fp,
                    "output_file_hash": out_file_hash,
                    "completed_at": now_iso,
                    "video_codec": render_res.video_codec,
                    "audio_codec": render_res.audio_codec,
                    "width": render_res.width,
                    "height": render_res.height,
                    "fps": render_res.fps,
                }
                self.persistence.save_checkpoint(project_id, render_id, completed_ckpt)
                outputs_state[render_id] = completed_ckpt
                state["timestamps"]["updated_at"] = now_iso
                self.persistence.save_project(state)
                if callbacks and callbacks.on_output_completed:
                    callbacks.on_output_completed(render_id, completed_ckpt)
                published_count += 1
                tracker.emit(ProgressEvent(
                    state=ActivityState.RUNNING.value,
                    stage=WorkflowStage.PUBLISH.value,
                    activity_text=f"Published output {output_index} / {len(outputs_def)}.",
                    output_id=render_id, current_item=render_id,
                    completed=published_count, total=len(outputs_def), unit="outputs",
                    item_event="complete", item_key=render_id,
                ))

            # All outputs complete!
            state["status"] = ProjectStatus.COMPLETED.value
            now_iso = datetime.now(timezone.utc).isoformat()
            state["timestamps"]["completed_at"] = now_iso
            state["timestamps"]["updated_at"] = now_iso
            self.persistence.save_project(state)
            if callbacks and callbacks.on_status_change:
                callbacks.on_status_change(ProjectStatus.COMPLETED.value)

            tracker.terminate(
                ActivityState.COMPLETE,
                activity_text="Project complete. Publication checkpoints are ready.",
                output_folder=str(resolved_output_dir),
                published_count=published_count,
            )

            return state

        except CancelledError:
            state["status"] = ProjectStatus.CANCELLED.value
            now_iso = datetime.now(timezone.utc).isoformat()
            state.setdefault("timestamps", {})["cancelled_at"] = now_iso
            state.setdefault("timestamps", {})["updated_at"] = now_iso
            self.persistence.save_project(state)
            if callbacks and callbacks.on_status_change:
                callbacks.on_status_change(ProjectStatus.CANCELLED.value)
            tracker.terminate(
                ActivityState.CANCELLED,
                activity_text="Project cancelled; completed checkpoints were preserved.",
            )
            raise
        except Exception as exc:
            tracker.terminate(
                ActivityState.FAILED,
                activity_text="Workflow failed; prior checkpoints and progress were preserved.",
                error=str(exc)[:1000],
            )
            raise
