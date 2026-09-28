"""Background worker thread and thread-safe queue dispatcher for ToolRecap V4.

Invariants:
- Background execution in dedicated daemon thread.
- Zero Tkinter or GUI calls from worker thread.
- Thread-safe communication exclusively via queue.Queue.
- Prevents duplicate start while workflow is running.
- Cancellation stops safely at current checkpoint.
- Sanitizes errors: readable messages, no tracebacks, no secrets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import queue
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.discovery import (
    SUPPORTED_EXTENSIONS,
    SourceFingerprint,
    compute_file_fingerprint,
    discover_sources,
    is_windows_reserved_stem,
)
from toolrecap_v4.errors import (
    CancelledError,
    DiscoveryError,
    ToolRecapError,
    WindowsCollisionError,
    WindowsReservedNameError,
)
from toolrecap_v4.gateway import GatewayClient, sanitize_message as gw_sanitize
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.secrets import DPAPISecretStore
from toolrecap_v4.settings import AppSettings, SettingsManager
from toolrecap_v4.voice_studio import VoiceStudioAdapter, sanitize_message as vs_sanitize
from toolrecap_v4.workflow import (
    ProjectStatus,
    ProjectWorkflow,
    WorkflowCallbacks,
)
from toolrecap_v4.progress import ActivityState, ProgressEvent, WorkflowProgressTracker, WorkflowStage

logger = logging.getLogger(__name__)


@dataclass
class WorkerMessage:
    """Message sent from background worker thread to Tkinter UI via queue."""
    kind: str
    data: Any = None


def format_clean_error(exc: Exception, secrets: Optional[List[str]] = None) -> str:
    """Extract a user-friendly error string without stack trace or secrets."""
    msg = str(exc)
    if not msg:
        msg = type(exc).__name__

    # Redact secrets
    all_secrets = secrets or []
    try:
        store = DPAPISecretStore()
        for k in store.list_keys():
            val = store.get_secret(k)
            if val:
                all_secrets.append(val)
    except Exception:
        pass

    msg = gw_sanitize(msg, all_secrets)
    msg = vs_sanitize(msg, all_secrets)

    # Translate common technical patterns to friendly Vietnamese
    if "Connection refused" in msg or "ConnectError" in msg:
        return f"Không thể kết nối đến máy chủ: {msg}"
    if "Timeout" in msg or "timed out" in msg:
        return f"Hết thời gian chờ phản hồi (Timeout): {msg}"
    if "videoInput" in msg:
        return f"Mô hình AI không hỗ trợ phân tích video (videoInput): {msg}"
    if "DPAPI" in msg:
        return f"Lỗi xác thực bảo mật Windows DPAPI: {msg}"

    return msg


class SourceDiscoveryWorker:
    """Manages background discovery of source files and folders with thread-safe queue dispatching.

    Invariants:
    - Dedicated daemon thread per discovery operation.
    - Zero Tkinter or GUI calls from worker thread.
    - Thread-safe communication exclusively via queue.Queue.
    - Token-based generation guard against stale / out-of-order results.
    - Cancellation stops discovery without hanging.
    - Fast stat-only probing without computing full movie sha256.
    """

    def __init__(self, msg_queue: Optional[queue.Queue[WorkerMessage]] = None) -> None:
        self.queue: queue.Queue[WorkerMessage] = msg_queue or queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._cancellation_token: Optional[CancellationToken] = None
        self._active_token_id: int = 0
        self._lock = threading.Lock()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if a discovery thread is currently running."""
        with self._lock:
            return self._is_running and self._thread is not None and self._thread.is_alive()

    @property
    def active_token_id(self) -> int:
        """Return the current active generation token ID."""
        with self._lock:
            return self._active_token_id

    def cancel(self) -> None:
        """Request cooperative cancellation of current discovery task."""
        with self._lock:
            if self._cancellation_token:
                self._cancellation_token.cancel()

    def wait_until_idle(self, timeout: float = 2.0) -> bool:
        """Wait for active discovery thread to finish (useful in tests)."""
        t = None
        with self._lock:
            t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)
            return not t.is_alive()
        return True

    def start_discovery(self, path: Union[str, Path], is_folder: bool) -> int:
        """Launch discovery for file or folder in background thread.

        Returns generation token ID. Cancels any previously active discovery.
        """
        with self._lock:
            if self._cancellation_token:
                self._cancellation_token.cancel()
            self._active_token_id += 1
            token_id = self._active_token_id
            token = CancellationToken()
            self._cancellation_token = token
            self._is_running = True

        p = Path(path).resolve()

        def _worker_target() -> None:
            try:
                if token.is_cancelled:
                    raise CancelledError("Discovery cancelled before start.")

                if is_folder:
                    def _progress(
                        cur: int,
                        tot: int,
                        source: Optional[SourceFingerprint] = None,
                    ) -> None:
                        if not token.is_cancelled:
                            payload = {
                                "token": token_id,
                                "current": cur,
                                "total": tot,
                                "path": str(p),
                                "name": p.name,
                            }
                            if source is not None:
                                payload["source"] = source
                            self.queue.put(
                                WorkerMessage(
                                    kind="discovery_progress",
                                    data=payload,
                                )
                            )

                    # discover_sources preserves its public two-argument
                    # callback contract unless the UI worker explicitly opts
                    # into incremental SourceFingerprint payloads.
                    _progress._toolrecap_include_source = True  # type: ignore[attr-defined]

                    fps = discover_sources(
                        p,
                        cancellation_token=token,
                        compute_hash=False,
                        progress_callback=_progress,
                    )
                    if not token.is_cancelled:
                        self.queue.put(
                            WorkerMessage(
                                kind="discovery_completed",
                                data={
                                    "token": token_id,
                                    "kind": "folder",
                                    "path": str(p),
                                    "name": p.name,
                                    "sources": fps,
                                },
                            )
                        )
                else:
                    if not p.exists():
                        raise DiscoveryError(f"File video không tồn tại: {p.name}")
                    if not p.is_file():
                        raise DiscoveryError(f"Đường dẫn không phải tệp: {p.name}")

                    ext = p.suffix.lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        raise DiscoveryError(f"Định dạng video không được hỗ trợ: '{ext}' ({p.name})")
                    if is_windows_reserved_stem(p.stem):
                        raise WindowsReservedNameError(f"Tên tệp sử dụng tên bảo lưu của Windows: {p.name}")

                    fp = compute_file_fingerprint(p, cancellation_token=token, compute_hash=False)
                    if not token.is_cancelled:
                        self.queue.put(
                            WorkerMessage(
                                kind="discovery_completed",
                                data={
                                    "token": token_id,
                                    "kind": "file",
                                    "path": str(p),
                                    "name": p.name,
                                    "sources": [fp],
                                },
                            )
                        )
            except CancelledError:
                self.queue.put(
                    WorkerMessage(kind="discovery_cancelled", data={"token": token_id})
                )
            except Exception as exc:
                if not token.is_cancelled:
                    clean_err = format_clean_error(exc)
                    self.queue.put(
                        WorkerMessage(
                            kind="discovery_failed",
                            data={"token": token_id, "error": clean_err, "path": str(p), "name": p.name},
                        )
                    )
            finally:
                with self._lock:
                    if self._active_token_id == token_id:
                        self._is_running = False

        self._thread = threading.Thread(target=_worker_target, daemon=True, name="DiscoveryWorkerThread")
        self._thread.start()
        return token_id


class WorkflowWorker:
    """Manages background execution of ProjectWorkflow with thread-safe queue dispatching."""

    def __init__(
        self,
        msg_queue: Optional[queue.Queue[WorkerMessage]] = None,
        persistence: Optional[ProjectPersistence] = None,
        settings_manager: Optional[SettingsManager] = None,
    ) -> None:
        self.queue: queue.Queue[WorkerMessage] = msg_queue or queue.Queue()
        self.persistence = persistence or ProjectPersistence()
        self.settings_manager = settings_manager or SettingsManager(persistence=self.persistence)
        self._thread: Optional[threading.Thread] = None
        self._cancellation_token: Optional[CancellationToken] = None
        self._lock = threading.Lock()
        self._is_running = False
        self._active_project_id: Optional[str] = None

    @property
    def is_running(self) -> bool:
        """Check if a worker thread is currently running."""
        with self._lock:
            return self._is_running and self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        """Request cancellation of current workflow."""
        with self._lock:
            if self._cancellation_token:
                self._cancellation_token.cancel()
                self._put_message("log", "Đang gửi yêu cầu dừng tới tiến trình...")
                self._put_message("activity_patch", {
                    "state": ActivityState.CANCELLING.value,
                    "activity_text": "Cancelling safely at the next checkpoint...",
                    "active": True,
                })

    def _put_message(self, kind: str, data: Any = None) -> None:
        """Thread-safely put a message into the queue."""
        self.queue.put(WorkerMessage(kind=kind, data=data))

    def _create_workflow(self, settings: AppSettings) -> ProjectWorkflow:
        """Construct the legacy/import renderer workflow with existing service settings."""
        store = DPAPISecretStore(storage_root=self.persistence.root)
        gw_key = store.get_secret("gateway_api_key")
        vs_key = store.get_secret("voice_remote_api_key")

        gw_client = GatewayClient(
            base_url=settings.gateway_endpoint,
            api_key=gw_key,
        )

        vs_adapter = VoiceStudioAdapter(
            mode=settings.voice_mode,
            local_url=settings.voice_local_url,
            remote_url=settings.voice_remote_url,
            remote_api_key=vs_key,
        )

        return ProjectWorkflow(
            persistence=self.persistence,
            gateway_client=gw_client,
            voice_adapter=vs_adapter,
            settings_manager=self.settings_manager,
            storage_root=self.persistence.root,
        )

    def _create_v5_workflow(self, settings: AppSettings):
        """Construct the simplified V5 Gateway-to-render workflow."""
        from toolrecap_v4.v5 import V5Workflow

        store = DPAPISecretStore(storage_root=self.persistence.root)
        gateway = GatewayClient(
            base_url=settings.gateway_endpoint,
            api_key=store.get_secret("gateway_api_key"),
        )
        voice = VoiceStudioAdapter(
            mode=settings.voice_mode,
            local_url=settings.voice_local_url,
            remote_url=settings.voice_remote_url,
            remote_api_key=store.get_secret("voice_remote_api_key"),
        )
        return V5Workflow(
            persistence=self.persistence,
            gateway_client=gateway,
            voice_adapter=voice,
            settings_manager=self.settings_manager,
        )

    def start_new_project(
        self,
        project_id: str,
        project_name: str,
        source_input: Union[str, Path, Sequence[Union[str, Path]]],
        prompt: str = "",
        output_dir: Optional[Union[str, Path]] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        """Launch start of a new project in background thread."""
        with self._lock:
            if self._is_running:
                raise RuntimeError("Một tác vụ đang được thực hiện. Vui lòng chờ hoặc bấm Dừng trước khi bắt đầu tác vụ mới.")
            self._is_running = True
            self._active_project_id = project_id
            token = CancellationToken()
            self._cancellation_token = token

        cfg = settings or self.settings_manager.load()

        def _worker_target() -> None:
            pre_tracker: WorkflowProgressTracker | None = None
            try:
                pre_tracker = WorkflowProgressTracker(
                    self.persistence, project_id,
                    callback=lambda snapshot: self._put_message("activity", snapshot),
                )
                pre_tracker.begin(resuming=False)
                pre_tracker.emit(ProgressEvent(
                    state=ActivityState.RUNNING.value, stage=WorkflowStage.PREPARATION.value,
                    activity_text="Project started. Preparing source media...",
                ))
                self._put_message("log", f"Bắt đầu khởi tạo dự án '{project_name}' ({project_id})...")
                wf = self._create_v5_workflow(cfg)

                # Step 1: Create project and probe sources
                self._put_message("status_change", ProjectStatus.CREATED.value)
                state = wf.create_project(
                    project_id=project_id,
                    project_name=project_name,
                    source_input=source_input,
                    prompt=prompt or cfg.prompt,
                    mode="RECAP",
                    output_dir=output_dir,
                    settings=cfg,
                    cancellation_token=token,
                )
                pre_tracker.emit(ProgressEvent(
                    state=ActivityState.LOCAL_PROCESSING.value, stage=WorkflowStage.PREPARATION.value,
                    activity_text="Source inventory verified; beginning episode preparation.",
                    completed=0, total=len(state.get("sources", [])), unit="episodes",
                ))
                self._put_message("log", f"Đã quét và kiểm tra {len(state.get('sources', []))} tệp video nguồn.")

                # Step 2: Build callbacks
                callbacks = self._build_callbacks()

                # Step 3: Start workflow execution
                self._put_message("log", "Gửi yêu cầu phân tích tới AI Gateway...")
                final_state = wf.run(
                    project_id,
                    output_dir=output_dir,
                    settings=cfg,
                    cancellation_token=token,
                    callbacks=callbacks,
                    progress_tracker=pre_tracker,
                )

                actual_status = final_state.get("status", ProjectStatus.COMPLETED.value)
                self._put_message("finished", {"status": actual_status, "project": final_state})
                self._put_message("log", f"Dự án '{project_name}' đã sẵn sàng duyệt kế hoạch." if actual_status == ProjectStatus.REVIEW_REQUIRED.value else f"Dự án '{project_name}' đã hoàn thành toàn bộ quy trình!")

            except CancelledError:
                if pre_tracker is not None and pre_tracker.tick().get("active"):
                    pre_tracker.terminate(ActivityState.CANCELLED, activity_text="Project cancelled during source preparation.")
                self._put_message("log", "Tiến trình đã được dừng an toàn tại điểm checkpoint.")
                self._put_message("status_change", ProjectStatus.CANCELLED.value)
                self._put_message("finished", {"status": ProjectStatus.CANCELLED.value, "project_id": project_id})
            except Exception as e:
                clean_err = format_clean_error(e)
                if pre_tracker is not None and pre_tracker.tick().get("active"):
                    pre_tracker.terminate(ActivityState.FAILED, activity_text="Source preparation failed.", error=clean_err)
                self._put_message("log", f"Lỗi tiến trình: {clean_err}")
                self._put_message("error", {"message": clean_err})
                self._put_message("finished", {"status": ProjectStatus.FAILED.value, "project_id": project_id, "error": clean_err})
            finally:
                with self._lock:
                    self._is_running = False
                    self._active_project_id = None

        self._thread = threading.Thread(target=_worker_target, daemon=True, name="WorkflowWorkerThread")
        self._thread.start()

    def start_highlight_project(
        self,
        project_id: str,
        project_name: str,
        source_input: Union[str, Path, Sequence[Union[str, Path]]],
        prompt: str,
        output_dir: Optional[Union[str, Path]] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        """Run Highlight through the same simplified V5 analysis boundary."""
        with self._lock:
            if self._is_running:
                raise RuntimeError("A workflow is already active.")
            self._is_running = True
            self._active_project_id = project_id
            token = CancellationToken()
            self._cancellation_token = token
        cfg = settings or self.settings_manager.load()

        def _target() -> None:
            workflow = None
            try:
                tracker = WorkflowProgressTracker(
                    self.persistence, project_id,
                    callback=lambda snapshot: self._put_message("activity", snapshot),
                )
                tracker.begin(resuming=False)
                workflow = self._create_v5_workflow(cfg)
                workflow.create_project(
                    project_id=project_id, project_name=project_name, source_input=source_input,
                    prompt=prompt, mode="HIGHLIGHT", settings=cfg,
                    output_dir=output_dir, cancellation_token=token,
                )
                final_state = workflow.run(
                    project_id, output_dir=output_dir, settings=cfg,
                    cancellation_token=token, callbacks=self._build_callbacks(),
                    progress_tracker=tracker,
                )
                self._put_message("finished", {"status": final_state.get("status", ProjectStatus.COMPLETED.value), "project": final_state, "mode": "HIGHLIGHT"})
            except CancelledError:
                self._put_message("finished", {"status": ProjectStatus.CANCELLED.value, "project_id": project_id, "mode": "HIGHLIGHT"})
            except Exception as exc:
                clean = format_clean_error(exc)
                self._put_message("error", {"message": clean})
                self._put_message("finished", {"status": ProjectStatus.FAILED.value, "project_id": project_id, "error": clean, "mode": "HIGHLIGHT"})
            finally:
                with self._lock:
                    self._is_running = False
                    self._active_project_id = None

        self._thread = threading.Thread(target=_target, daemon=True, name="HighlightWorkflowWorker")
        self._thread.start()

    def resume_project(
        self,
        project_id: str,
        output_dir: Optional[Union[str, Path]] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        """Resume an existing project from checkpoints in background thread."""
        with self._lock:
            if self._is_running:
                raise RuntimeError("Một tác vụ đang được thực hiện. Vui lòng chờ hoặc bấm Dừng.")
            self._is_running = True
            self._active_project_id = project_id
            token = CancellationToken()
            self._cancellation_token = token

        cfg = settings or self.settings_manager.load()

        def _worker_target() -> None:
            pre_tracker: WorkflowProgressTracker | None = None
            try:
                pre_tracker = WorkflowProgressTracker(
                    self.persistence, project_id,
                    callback=lambda snapshot: self._put_message("activity", snapshot),
                )
                pre_tracker.begin(resuming=True)
                pre_tracker.emit(ProgressEvent(
                    state=ActivityState.RESUMING.value,
                    stage=pre_tracker.tick().get("stage") or WorkflowStage.PREPARATION.value,
                    activity_text="Resume requested. Loading saved project...",
                ))
                self._put_message("log", f"Tiếp tục thực hiện dự án '{project_id}'...")
                saved_state = self.persistence.load_project(project_id)
                resume_mode = saved_state.get("project_mode", "RECAP")
                if saved_state.get("pipeline_version") == "5.0":
                    workflow = self._create_v5_workflow(cfg)
                    final_state = workflow.run(
                        project_id, output_dir=output_dir, settings=cfg,
                        cancellation_token=token, callbacks=self._build_callbacks(),
                        progress_tracker=pre_tracker, resuming=True,
                    )
                elif saved_state.get("project_mode", "RECAP") == "HIGHLIGHT":
                    from toolrecap_v4.highlight import HighlightWorkflow
                    store = DPAPISecretStore(storage_root=self.persistence.root)
                    gateway = GatewayClient(base_url=cfg.gateway_endpoint, api_key=store.get_secret("gateway_api_key"))
                    workflow = HighlightWorkflow(
                        persistence=self.persistence, gateway_client=gateway, settings_manager=self.settings_manager,
                        progress_callback=lambda payload: self._put_message("activity_patch", {"project_id": project_id, **payload}),
                        tracker=pre_tracker, tracker_started=True,
                    )
                    final_state = workflow.run(project_id, settings=cfg, cancellation_token=token)
                else:
                    wf = self._create_workflow(cfg)
                    callbacks = self._build_callbacks()
                    final_state = wf.resume_project(
                        project_id=project_id,
                        output_dir=output_dir,
                        settings=cfg,
                        cancellation_token=token,
                        callbacks=callbacks,
                        progress_tracker=pre_tracker,
                    )

                actual_status = final_state.get("status", ProjectStatus.COMPLETED.value)
                self._put_message("finished", {"status": actual_status, "project": final_state, "mode": resume_mode})
                self._put_message("log", f"Dự án '{project_id}' chờ duyệt kế hoạch." if actual_status == ProjectStatus.REVIEW_REQUIRED.value else f"Dự án '{project_id}' đã hoàn tất thành công!")

            except CancelledError:
                if pre_tracker is not None and pre_tracker.tick().get("active"):
                    pre_tracker.terminate(ActivityState.CANCELLED, activity_text="Resume cancelled.")
                self._put_message("log", "Tiến trình tiếp tục đã dừng an toàn.")
                self._put_message("status_change", ProjectStatus.CANCELLED.value)
                self._put_message("finished", {"status": ProjectStatus.CANCELLED.value, "project_id": project_id})
            except Exception as e:
                clean_err = format_clean_error(e)
                if pre_tracker is not None and pre_tracker.tick().get("active"):
                    pre_tracker.terminate(ActivityState.FAILED, activity_text="Resume failed.", error=clean_err)
                self._put_message("log", f"Lỗi khi tiếp tục dự án: {clean_err}")
                self._put_message("error", {"message": clean_err})
                self._put_message("finished", {"status": ProjectStatus.FAILED.value, "project_id": project_id, "error": clean_err})
            finally:
                with self._lock:
                    self._is_running = False
                    self._active_project_id = None

        self._thread = threading.Thread(target=_worker_target, daemon=True, name="WorkflowWorkerThread")
        self._thread.start()

    def render_existing_json(
        self,
        project_id: str,
        project_name: str,
        source_input: Union[str, Path, Sequence[Union[str, Path]]],
        final_json: Dict[str, Any],
        output_dir: Optional[Union[str, Path]] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        """Render from existing final JSON (0 AI requests)."""
        with self._lock:
            if self._is_running:
                raise RuntimeError("Một tác vụ đang được thực hiện. Vui lòng chờ hoặc bấm Dừng.")
            self._is_running = True
            self._active_project_id = project_id
            token = CancellationToken()
            self._cancellation_token = token

        cfg = settings or self.settings_manager.load()

        def _worker_target() -> None:
            try:
                self._put_message("activity", {
                    "project_id": project_id, "state": ActivityState.STARTING.value,
                    "stage": WorkflowStage.FINAL_JSON.value,
                    "stage_label": "Building Final JSON", "activity_text": "Validating imported Final JSON (zero AI)...",
                    "active": True, "completed": None, "total": None, "percent": None,
                    "session_elapsed_seconds": 0.0, "project_elapsed_seconds": 0.0,
                    "stage_elapsed_seconds": 0.0, "estimated_remaining_seconds": None,
                    "recent_activity": [], "last_activity_at": None,
                })
                self._put_message("log", f"Nhập kịch bản JSON có sẵn cho dự án '{project_name}' (0 yêu cầu AI)...")
                wf = self._create_workflow(cfg)

                # Step 1: Import existing validated JSON
                state = wf.import_project(
                    project_id=project_id,
                    project_name=project_name,
                    source_input=source_input,
                    final_json=final_json,
                    output_dir=output_dir,
                    settings=cfg,
                    cancellation_token=token,
                )
                self._put_message("status_change", ProjectStatus.ANALYZED.value)
                self._put_message("final_json", final_json)
                self._put_message("log", f"Đã nạp {len(final_json.get('outputs', []))} video mục tiêu từ JSON. Bắt đầu dựng...")

                # Step 2: Render
                callbacks = self._build_callbacks()
                final_state = wf.start_project(
                    project_id=project_id,
                    output_dir=output_dir,
                    settings=cfg,
                    cancellation_token=token,
                    callbacks=callbacks,
                )

                self._put_message("finished", {"status": ProjectStatus.COMPLETED.value, "project": final_state})
                self._put_message("log", f"Đã dựng hoàn tất toàn bộ video từ JSON có sẵn!")

            except CancelledError:
                self._put_message("log", "Tiến trình dựng từ JSON đã dừng an toàn.")
                self._put_message("status_change", ProjectStatus.CANCELLED.value)
                self._put_message("finished", {"status": ProjectStatus.CANCELLED.value, "project_id": project_id})
            except Exception as e:
                clean_err = format_clean_error(e)
                self._put_message("log", f"Lỗi dựng từ JSON: {clean_err}")
                self._put_message("error", {"message": clean_err})
                self._put_message("finished", {"status": ProjectStatus.FAILED.value, "project_id": project_id, "error": clean_err})
            finally:
                with self._lock:
                    self._is_running = False
                    self._active_project_id = None

        self._thread = threading.Thread(target=_worker_target, daemon=True, name="WorkflowWorkerThread")
        self._thread.start()

    def _build_callbacks(self) -> WorkflowCallbacks:
        """Construct WorkflowCallbacks dispatching all events to the thread-safe queue."""
        def on_status_change(status: str) -> None:
            self._put_message("status_change", status)
            if status == ProjectStatus.ANALYZING.value:
                self._put_message("log", "Đang phân tích nội dung video qua AI Gateway...")
            elif status == ProjectStatus.ANALYZED.value:
                self._put_message("log", "Phân tích AI hoàn tất. Kịch bản Final JSON đã được kiểm tra và lưu an toàn.")
            elif status == ProjectStatus.RENDERING.value:
                self._put_message("log", "Bắt đầu giai đoạn dựng video (Voice narration + FFmpeg mix)...")
            elif status == ProjectStatus.COMPLETED.value:
                self._put_message("log", "Trạng thái dự án: Hoàn thành.")
            elif status == ProjectStatus.CANCELLED.value:
                self._put_message("log", "Trạng thái dự án: Đã dừng.")

        def on_sub_analysis(sub_text: str) -> None:
            self._put_message("sub_analysis", sub_text)
            self._put_message("log", "Phân tích Sub video hoàn tất, đã lưu checkpoint nội dung.")

        def on_raw_response(raw: str) -> None:
            self._put_message("raw_response", raw)

        def on_final_json(final_data: Dict[str, Any]) -> None:
            self._put_message("final_json", final_data)

        def on_output_started(render_id: str) -> None:
            self._put_message("output_started", render_id)
            self._put_message("log", f"Bắt đầu dựng video: {render_id}...")

        def on_output_completed(render_id: str, data: Dict[str, Any]) -> None:
            self._put_message("output_completed", {"render_id": render_id, "data": data})
            self._put_message("log", f"Dựng xong video: {render_id} (thời lượng: {data.get('duration', 0):.1f}s)")

        def on_output_failed(render_id: str, error_msg: str) -> None:
            clean_err = format_clean_error(Exception(error_msg))
            self._put_message("output_failed", {"render_id": render_id, "error": clean_err})
            self._put_message("log", f"Lỗi khi dựng video {render_id}: {clean_err}")

        def on_output_skipped(render_id: str, data: Dict[str, Any]) -> None:
            self._put_message("output_skipped", {"render_id": render_id, "data": data})
            self._put_message("log", f"Bỏ qua video {render_id}: đã hoàn tất trước đó và khớp mã kiểm tra.")

        def on_error(e: Exception, raw: Optional[str]) -> None:
            clean = format_clean_error(e)
            self._put_message("error", {"message": clean, "raw_response": raw})

        def on_activity(snapshot: Dict[str, Any]) -> None:
            self._put_message("activity", snapshot)

        return WorkflowCallbacks(
            on_status_change=on_status_change,
            on_sub_analysis=on_sub_analysis,
            on_raw_response=on_raw_response,
            on_final_json=on_final_json,
            on_output_started=on_output_started,
            on_output_completed=on_output_completed,
            on_output_failed=on_output_failed,
            on_output_skipped=on_output_skipped,
            on_error=on_error,
            on_activity=on_activity,
        )
