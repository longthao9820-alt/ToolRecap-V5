"""Main desktop application window for ToolRecap V5.

Invariants:
- Visual styling, layout, fonts, and dimensions aligned with ToolRecap V2 ui.py (1120x760, min 920x680, vista/clam theme).
- Header with Title, Subtitle, Update check button, and Settings button.
- In-app NotificationBanner with visible [X] close button.
- Movie Source section with File and Folder selection buttons and clean status label.
- Unified 5-column Source Episodes & Output Queue table (Episode, Source Video, Stage, Progress, Status).
- Progress bar and bottom action controls: Start, Stop, Open Output Folder, Resume, Render from JSON (0 AI), Clear.
- Status bar showing current status on left and hardware GPU encoder info on right.
- Thread-safe background worker communication exclusively via queue (no Tk calls from worker).
- Safe window close with cancellation and checkpoint preservation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import copy
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# Ensure Tcl/Tk paths on Windows for robust portable execution
if sys.platform == "win32":
    tcl_path = os.path.join(sys.base_prefix, "tcl", "tcl8.6")
    tk_path = os.path.join(sys.base_prefix, "tcl", "tk8.6")
    if os.path.isdir(tcl_path) and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = tcl_path
    if os.path.isdir(tk_path) and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = tk_path

    if getattr(sys, "frozen", False):
        base_mei = getattr(sys, "_MEIPASS", None)
        if base_mei:
            for candidate in [
                os.path.join(base_mei, "_tcl_data"),
                os.path.join(base_mei, "tcl", "tcl8.6"),
                os.path.join(base_mei, "tcl8.6"),
                os.path.join(base_mei, "tcl"),
            ]:
                if os.path.isdir(candidate):
                    os.environ["TCL_LIBRARY"] = candidate.replace("\\", "/")
                    break
            for candidate in [
                os.path.join(base_mei, "_tk_data"),
                os.path.join(base_mei, "tcl", "tk8.6"),
                os.path.join(base_mei, "tk8.6"),
                os.path.join(base_mei, "tk"),
            ]:
                if os.path.isdir(candidate):
                    os.environ["TK_LIBRARY"] = candidate.replace("\\", "/")
                    break

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from toolrecap_v4.__version__ import __version__
from toolrecap_v4.discovery import (
    SUPPORTED_EXTENSIONS,
    SourceFingerprint,
    compute_file_fingerprint,
    discover_sources,
    natural_sort_key,
)
from toolrecap_v4.media import detect_gpu_encoder, probe_media
from toolrecap_v4.persistence import ProjectPersistence, read_json
from toolrecap_v4.settings import AppSettings, SettingsManager
from toolrecap_v4.ui.notifications import WindowsNotificationService
from toolrecap_v4.ui.settings_dialog import SettingsDialog
from toolrecap_v4.ui.workspace import WorkspaceLabels, WorkspacePage
from toolrecap_v4.ui.worker import SourceDiscoveryWorker, WorkerMessage, WorkflowWorker
from toolrecap_v4.workflow import OutputStatus, ProjectStatus
from toolrecap_v4.progress import (
    ACTIVE_STATES,
    ActivityState,
    WorkflowStage,
    PIPELINE_STAGES,
    V5_PIPELINE_STAGES,
    STAGE_LABELS,
    format_duration as format_workflow_duration,
    format_last_activity,
    format_wait_duration,
    reconstruct_project_progress,
)

logger = logging.getLogger(__name__)

SEASON_STAGE_TRANSLATIONS: dict[str, str] = {
    "Finalizer": "Finalizer",
    "finalizer": "Finalizer",
    "Technical JSON Validation": "Kiểm tra kỹ thuật Final JSON",
    "json_validation": "Kiểm tra kỹ thuật Final JSON",
    "Final JSON Repair": "Sửa Final JSON",
    "json_repair": "Sửa Final JSON",
    "CANDIDATE_DISCOVERY": "Khám phá ứng viên",
    "candidate_discovery": "Khám phá ứng viên",
    "Candidate Discovery": "Khám phá ứng viên",
    "CANDIDATE_CONSOLIDATION": "Hợp nhất ứng viên",
    "candidate_consolidation": "Hợp nhất ứng viên",
    "Candidate Consolidation": "Hợp nhất ứng viên",
    "CANDIDATE_VERIFYING": "Xác minh kết quả 0 output",
    "candidate_verifying": "Xác minh kết quả 0 output",
    "Candidate Verifying": "Xác minh kết quả 0 output",
    "ZERO_OUTPUT_VERIFICATION": "Xác minh kết quả 0 output",
    "zero_output_verification": "Xác minh kết quả 0 output",
    "Zero Output Verification": "Xác minh kết quả 0 output",
    "Season Mining": "Khai thác cốt truyện",
    "season_mining": "Khai thác cốt truyện",
    "Season Batch": "Phân tích nhóm mùa",
    "season_batch": "Phân tích nhóm mùa",
    "Episode Summarizing": "Tóm tắt tập",
    "episode_summarizing": "Tóm tắt tập",
    "Season Merging": "Hợp nhất cốt truyện",
    "season_merging": "Hợp nhất cốt truyện",
}


def translate_season_stage(stage: str) -> str:
    if not stage:
        return "Sẵn sàng"
    return SEASON_STAGE_TRANSLATIONS.get(stage, stage)


def safe_after(widget: tk.Misc | None, ms: int, func: Callable, *args: Any) -> str | None:
    """Safely schedule a callback on a tkinter widget, catching post-destroy / mainloop exceptions."""
    if widget is None:
        return None

    def _wrapped() -> None:
        try:
            if hasattr(widget, "_is_closed") and getattr(widget, "_is_closed", False):
                return
            if hasattr(widget, "winfo_exists") and not widget.winfo_exists():
                return
            func(*args)
        except (tk.TclError, RuntimeError):
            pass
        except Exception:
            pass

    try:
        if hasattr(widget, "_is_closed") and getattr(widget, "_is_closed", False):
            return None
        if hasattr(widget, "winfo_exists") and not widget.winfo_exists():
            return None
        return widget.after(ms, _wrapped)
    except (tk.TclError, RuntimeError):
        return None
    except Exception:
        return None


def format_bytes(size_bytes: int) -> str:
    """Format bytes into readable human unit (MB, GB)."""
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    secs = int(round(seconds))
    mins, s = divmod(secs, 60)
    hrs, m = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class NotificationBanner(ttk.Frame):
    """Dismissible in-app banner with a visible 'X' button, using consistent grid geometry."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_dismiss: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, padding=(10, 6))
        self.on_dismiss = on_dismiss
        self._visible = False
        self._auto_dismiss_job: str | None = None

        self.columnconfigure(1, weight=1)

        self.icon_label = ttk.Label(self, text="ℹ", font=("Segoe UI", 11, "bold"))
        self.icon_label.grid(row=0, column=0, padx=(4, 8), sticky="w")

        self.message_label = ttk.Label(self, text="", font=("Segoe UI", 9), wraplength=700)
        self.message_label.grid(row=0, column=1, sticky="ew")

        self.close_button = ttk.Button(
            self,
            text="✕",
            width=3,
            command=self.dismiss,
            style="NotificationClose.TButton",
        )
        self.close_button.grid(row=0, column=2, padx=(8, 4), sticky="e")

        self._configure_styles()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.configure("NotificationClose.TButton", font=("Segoe UI", 9, "bold"), padding=2)

    @property
    def is_visible(self) -> bool:
        return self._visible

    def show(
        self,
        message: str,
        level: str = "info",
        *,
        auto_dismiss_ms: int = 0,
        row: int = 1,
        column: int = 0,
    ) -> None:
        """Display notification message using grid geometry consistently with parent container."""
        if self._auto_dismiss_job:
            try:
                self.after_cancel(self._auto_dismiss_job)
            except Exception:
                pass
            self._auto_dismiss_job = None

        icons = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✕",
        }
        self.icon_label.config(text=icons.get(level, "ℹ"))
        self.message_label.config(text=message)

        self.grid(row=row, column=column, sticky="ew", pady=(0, 8))
        self._visible = True

        if auto_dismiss_ms > 0:
            self._auto_dismiss_job = self.after(auto_dismiss_ms, self.dismiss)

    def dismiss(self) -> None:
        """Hide notification via grid_remove and invoke dismiss callback."""
        if self._auto_dismiss_job:
            try:
                self.after_cancel(self._auto_dismiss_job)
            except Exception:
                pass
            self._auto_dismiss_job = None

        if self._visible:
            try:
                self.grid_remove()
            except Exception:
                pass
            self._visible = False

        if self.on_dismiss:
            try:
                self.on_dismiss()
            except Exception:
                pass

    def destroy(self) -> None:
        if self._auto_dismiss_job:
            try:
                self.after_cancel(self._auto_dismiss_job)
            except Exception:
                pass
            self._auto_dismiss_job = None
        super().destroy()


class MainWindow(tk.Tk):
    """Main desktop application window for ToolRecap V5."""

    def __init__(self, persistence: Optional[ProjectPersistence] = None) -> None:
        for attempt in range(3):
            try:
                super().__init__()
                break
            except tk.TclError:
                if attempt == 2:
                    raise
                time.sleep(0.05)

        self.title(f"ToolRecap V5 — Gateway to Render (v{__version__})")
        self.geometry("1120x900")
        self.minsize(920, 760)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Map>", self._on_window_activated, add="+")
        self.bind("<FocusIn>", self._on_window_activated, add="+")

        # Storage & State
        self.persistence = persistence or ProjectPersistence()
        self.settings_manager = SettingsManager(persistence=self.persistence)
        self.notification_service = WindowsNotificationService()

        self.msg_queue: queue.Queue[WorkerMessage] = queue.Queue()
        self.worker = WorkflowWorker(
            msg_queue=self.msg_queue,
            persistence=self.persistence,
            settings_manager=self.settings_manager,
        )
        self.discovery_worker = SourceDiscoveryWorker(
            msg_queue=self.msg_queue,
        )
        self._active_discovery_token: int = 0
        self._is_closed: bool = False
        self._poll_job: Optional[str] = None
        self._gpu_detect_job: Optional[str] = None
        self._close_job: Optional[str] = None
        self._activity_timer_job: Optional[str] = None
        self._dots_job: Optional[str] = None
        self._dots_phase = 1
        # Compatibility alias: Settings is now an embedded Frame, never a modal.
        self.active_modal: Optional[SettingsDialog] = None
        self.settings_page: Optional[SettingsDialog] = None
        self.current_page = "recap"
        self._discovery_mode = "recap"
        self._shutting_down = False

        self.settings = self.settings_manager.load()
        self.discovered_sources: List[SourceFingerprint] = []
        self.highlight_sources: List[SourceFingerprint] = []
        self.current_project_id: Optional[str] = None
        self.current_project_name: Optional[str] = None
        self._resume_project_ids: Dict[str, str] = {}
        self._highlight_resume_project_ids: Dict[str, str] = {}
        self._current_project_by_mode: Dict[str, Optional[str]] = {"RECAP": None, "HIGHLIGHT": None}
        self._output_paths: Dict[str, str] = {}

        # Tkinter variables matching V2
        self.source_var = tk.StringVar(value="Chưa chọn video hoặc thư mục.")
        self.highlight_source_var = tk.StringVar(value="No highlight source selected.")
        self.status_var = tk.StringVar(value="Sẵn sàng.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label_var = tk.StringVar(value="0%")
        self.gpu_status_var = tk.StringVar(value="Đang kiểm tra phần cứng...")
        self.activity_state_var = tk.StringVar(value=ActivityState.IDLE.value)
        self.activity_stage_var = tk.StringVar(value="Not started")
        self.activity_item_var = tk.StringVar(value="—")
        self.activity_progress_var = tk.StringVar(value="Progress not yet measurable")
        self.activity_text_var = tk.StringVar(value="Ready")
        self.activity_dots_var = tk.StringVar(value="")
        self.activity_stage_time_var = tk.StringVar(value="0 minutes")
        self.activity_session_time_var = tk.StringVar(value="0 minutes")
        self.activity_project_time_var = tk.StringVar(value="Unavailable")
        self.activity_eta_var = tk.StringVar(value="Unknown")
        self.activity_last_var = tk.StringVar(value="Unavailable")
        self.activity_wait_var = tk.StringVar(value="")
        self.activity_summary_var = tk.StringVar(value="")
        self.pipeline_var = tk.StringVar(value="○ Preparation → ○ Scanner → ○ Catalog → ○ Planner")
        self._activity_snapshot: Dict[str, Any] = {
            "state": ActivityState.IDLE.value, "active": False,
            "session_elapsed_seconds": 0.0, "project_elapsed_seconds": None,
            "stage_elapsed_seconds": 0.0, "estimated_remaining_seconds": None,
            "last_activity_at": None, "recent_activity": [],
        }
        self._activity_received_monotonic = time.monotonic()
        self._activity_context = "idle"
        self._discovery_started_monotonic: float | None = None

        self._build_style()
        self._build_ui()
        if self.settings_manager.last_load_warning:
            self.banner.show(self.settings_manager.last_load_warning, level="warning")
        self._refresh_queue_table()

        # Background tasks
        self._gpu_detect_job = safe_after(self, 200, self._detect_gpu_background)

        # Check restartable projects
        self._check_restart_projects()

        # Start periodic queue polling
        self._poll_job = safe_after(self, 100, self._poll_queue)
        self._activity_timer_job = safe_after(self, 1000, self._update_activity_timer)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        theme = "vista" if "vista" in style.theme_names() else "clam"
        style.theme_use(theme)

        style.configure("Title.TLabel", font=("Segoe UI Semibold", 16))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#555555")
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=(16, 8))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 5))
        style.configure("Danger.TButton", font=("Segoe UI Semibold", 10), padding=(14, 8))
        style.configure("Treeview", rowheight=32, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9), padding=6)
        style.configure("Horizontal.TProgressbar", thickness=14)
        self.style = style

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)
        self.main_container = container

        # 1. Top Header
        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="ToolRecap V5", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Tự động sản xuất video recap hoàn chỉnh chỉ với một cú nhấp chuột",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        nav_box = ttk.Frame(header)
        nav_box.grid(row=0, column=2, rowspan=2, sticky="e")
        self.page_buttons: Dict[str, ttk.Button] = {}
        for key, label in (("recap", "Recap"), ("highlight", "Highlight"), ("settings", "Settings")):
            button = ttk.Button(nav_box, text=label, command=lambda page=key: self._show_page(page))
            button.pack(side="left", padx=3)
            self.page_buttons[key] = button
        btn_box = ttk.Frame(header)
        btn_box.grid(row=0, column=3, rowspan=2, sticky="e", padx=(8, 0))
        self.update_btn = ttk.Button(btn_box, text="🔄 Kiểm tra cập nhật", command=self._on_check_update)
        self.update_btn.pack(side="left", padx=4)
        self.settings_btn = self.page_buttons["settings"]

        # 2. Notification Banner (with visible [X])
        self.banner = NotificationBanner(container, on_dismiss=self._on_banner_dismissed)
        self.banner.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.banner.grid_remove()

        # 3. Single-window page host. All workspaces are persistent Frames.
        self.workspace_host = ttk.Frame(container)
        self.workspace_host.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        self.workspace_host.columnconfigure(0, weight=1)
        self.workspace_host.rowconfigure(0, weight=1)
        self.recap_page = WorkspacePage(
            self.workspace_host,
            labels=WorkspaceLabels(
                "Source", "📁 Select File", "📂 Select Folder", "Unfinished Project:",
                "⏯ Resume Saved Project", "Recap Prompt", "Episodes & Output Queue",
                "▶ Start Creating Recap Videos", "Clear",
            ),
            source_variable=self.source_var, prompt=self.settings.prompt,
            on_select_file=lambda: self._choose_file_for_mode("recap"),
            on_select_folder=lambda: self._choose_folder_for_mode("recap"),
            on_saved_selected=self._on_select_resume_project,
            on_resume_saved=lambda: self._on_resume_detected_project("RECAP"),
            on_delete_saved=lambda: self._on_delete_saved_project("RECAP"),
            on_start=self._on_start_project, on_stop=self._on_stop_project,
            on_open_output=self._open_output_folder, on_resume=self._on_resume_current,
            on_clear=self._clear_queue, on_use_json=self._on_render_existing_json,
            on_review=lambda: self._on_review_plan("RECAP"),
            review_before_render=self.settings.review_before_render,
        )
        self.recap_page.grid(row=0, column=0, sticky="nsew")
        self.highlight_page = WorkspacePage(
            self.workspace_host,
            labels=WorkspaceLabels(
                "Highlight Source", "📁 Select Episode", "📂 Select Season Folder",
                "Unfinished Highlight Project:", "⏯ Resume Saved Project", "Highlight Prompt",
                "Episodes & Output Queue", "▶ Create Highlights", "Clear Highlight Source",
            ),
            source_variable=self.highlight_source_var, prompt=self.settings.highlight_prompt,
            on_select_file=lambda: self._choose_file_for_mode("highlight"),
            on_select_folder=lambda: self._choose_folder_for_mode("highlight"),
            on_saved_selected=self._on_select_highlight_resume_project,
            on_resume_saved=lambda: self._on_resume_detected_project("HIGHLIGHT"),
            on_delete_saved=lambda: self._on_delete_saved_project("HIGHLIGHT"),
            on_start=self._on_start_highlight, on_stop=self._on_stop_project,
            on_open_output=self._open_output_folder, on_resume=lambda: self._on_resume_current("HIGHLIGHT"),
            on_clear=self._clear_highlight,
            on_review=lambda: self._on_review_plan("HIGHLIGHT"),
            review_before_render=self.settings.review_before_render,
        )
        self.highlight_page.grid(row=0, column=0, sticky="nsew")

        # Compatibility aliases for established tests and handlers.
        self.tree = self.recap_page.tree
        self.highlight_tree = self.highlight_page.tree
        self.btn_select_file = self.recap_page.select_file_button
        self.btn_select_folder = self.recap_page.select_folder_button
        self.lbl_source = self.recap_page.source_label
        self.resume_banner = self.recap_page.saved_panel
        self.cmb_resume_projects = self.recap_page.saved_combo
        self.btn_resume_saved = self.recap_page.resume_saved_button
        self.recap_prompt_text = self.recap_page.prompt_text
        self.highlight_prompt_text = self.highlight_page.prompt_text

        # 5. Dedicated real-time workflow status/activity panel
        self.activity_panel = ttk.LabelFrame(container, text="Real-Time Status / Activity", padding=10)
        self.activity_panel.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.activity_panel.columnconfigure(0, weight=3)
        self.activity_panel.columnconfigure(1, weight=2)

        primary = ttk.Frame(self.activity_panel)
        primary.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        primary.columnconfigure(1, weight=1)

        ttk.Label(primary, text="STATUS", font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w")
        self.activity_state_label = ttk.Label(
            primary, textvariable=self.activity_state_var,
            font=("Segoe UI Semibold", 12), foreground="#0969da",
        )
        self.activity_state_label.grid(row=0, column=1, sticky="w", padx=(8, 0))

        detail_rows = (
            ("Current Stage:", self.activity_stage_var),
            ("Current Item:", self.activity_item_var),
            ("Progress:", self.activity_progress_var),
            ("Activity:", self.activity_text_var),
        )
        for row, (label, variable) in enumerate(detail_rows, start=1):
            ttk.Label(primary, text=label, font=("Segoe UI Semibold", 9)).grid(
                row=row, column=0, sticky="nw", pady=(3, 0),
            )
            ttk.Label(primary, textvariable=variable, wraplength=560).grid(
                row=row, column=1, sticky="w", padx=(8, 0), pady=(3, 0),
            )

        self.progressbar = ttk.Progressbar(primary, variable=self.progress_var, maximum=100)
        self.progressbar.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(7, 4))
        self._progress_indeterminate = False
        self.activity_dots_label = ttk.Label(primary, textvariable=self.activity_dots_var, foreground="#555")
        self.activity_dots_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(7, 4))
        self.activity_dots_label.grid_remove()

        ttk.Label(primary, textvariable=self.pipeline_var, wraplength=630, foreground="#555").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(2, 5),
        )

        timing = ttk.Frame(primary)
        timing.grid(row=7, column=0, columnspan=2, sticky="ew")
        for column in range(3):
            timing.columnconfigure(column, weight=1)
        timing_values = (
            ("Stage Time", self.activity_stage_time_var),
            ("Current Session", self.activity_session_time_var),
            ("Project Processing", self.activity_project_time_var),
            ("Estimated Remaining", self.activity_eta_var),
            ("Last Activity", self.activity_last_var),
            ("AI Wait", self.activity_wait_var),
        )
        for index, (label, variable) in enumerate(timing_values):
            row, column = divmod(index, 3)
            cell = ttk.Frame(timing)
            cell.grid(row=row, column=column, sticky="w", padx=(0, 12), pady=(2, 0))
            ttk.Label(cell, text=f"{label}:", font=("Segoe UI Semibold", 8)).pack(anchor="w")
            ttk.Label(cell, textvariable=variable).pack(anchor="w")

        ttk.Label(primary, textvariable=self.activity_summary_var, wraplength=630, foreground="#0969da").grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(6, 0),
        )

        recent = ttk.Frame(self.activity_panel)
        recent.grid(row=0, column=1, sticky="nsew")
        recent.columnconfigure(0, weight=1)
        recent.rowconfigure(1, weight=1)
        ttk.Label(recent, text="Recent Activity", font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w")
        self.activity_log = tk.Text(
            recent, height=10, width=43, wrap="word", state=tk.DISABLED,
            font=("Consolas", 8), relief="solid", borderwidth=1,
        )
        self.activity_log.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        self.btn_start = self.recap_page.start_button
        self.btn_stop = self.recap_page.stop_button
        self.btn_open_out = self.recap_page.open_output_button
        self.btn_resume = self.recap_page.resume_button
        self.btn_render_json = self.recap_page.use_json_button
        self.btn_clear = self.recap_page.clear_button
        self.btn_start_highlight = self.highlight_page.start_button
        self.btn_stop_highlight = self.highlight_page.stop_button
        self.btn_open_highlight_out = self.highlight_page.open_output_button
        self.btn_resume_highlight = self.highlight_page.resume_button
        self.btn_highlight_file = self.highlight_page.select_file_button
        self.btn_highlight_folder = self.highlight_page.select_folder_button

        # 7. Status Bar Frame
        status_bar = ttk.Frame(container, relief="sunken", padding=(8, 6))
        status_bar.grid(row=4, column=0, sticky="ew")
        status_bar.columnconfigure(1, weight=1)

        ttk.Label(status_bar, text="Trạng thái:", font=("Segoe UI Semibold", 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Label(status_bar, textvariable=self.status_var, font=("Segoe UI", 9)).grid(row=0, column=1, sticky="w")
        ttk.Label(status_bar, textvariable=self.gpu_status_var, font=("Segoe UI", 8), foreground="#555").grid(row=0, column=2, sticky="e")

        # Compatibility & Internal Aliases
        self.btn_cancel = self.btn_stop
        self.tree_sources = self.tree
        self.tree_outputs = self.tree
        self.lbl_src_summary = self.lbl_source
        self.progress_bar = self.progressbar

        # Offscreen logging & stage label for internal message tracking / testing
        self.lbl_stage = ttk.Label(self, textvariable=self.status_var)
        self.txt_log = tk.Text(self, height=1)

        self.settings_page = SettingsDialog(
            parent=self.workspace_host,
            settings_manager=self.settings_manager,
            persistence=self.persistence,
            on_saved=self._on_settings_saved,
        )
        self.settings_page.grid(row=0, column=0, sticky="nsew")
        self.active_modal = None
        self._show_page("recap")

    def _build_highlight_page(self) -> None:
        """Compatibility hook; both modes are now built by :class:`WorkspacePage`."""
        return None

    def _show_page(self, page: str) -> None:
        pages = {"recap": self.recap_page, "highlight": self.highlight_page, "settings": self.settings_page}
        target = pages.get(page)
        if target is None:
            return
        self.current_page = page
        target.tkraise()
        for name, button in self.page_buttons.items():
            button.configure(state="disabled" if name == page else "normal")

    def _on_settings_saved(self, new_settings: AppSettings) -> None:
        current_recap = self.recap_prompt_text.get("1.0", "end-1c")
        current_highlight = self.highlight_prompt_text.get("1.0", "end-1c") if hasattr(self, "highlight_prompt_text") else ""
        if not current_recap.strip() or current_recap == getattr(self.settings, "prompt", ""):
            self.recap_prompt_text.delete("1.0", tk.END)
            self.recap_prompt_text.insert("1.0", new_settings.prompt)
        if not current_highlight.strip() or current_highlight == getattr(self.settings, "highlight_prompt", ""):
            self.highlight_prompt_text.delete("1.0", tk.END)
            self.highlight_prompt_text.insert("1.0", new_settings.highlight_prompt)
        self.settings = new_settings
        self.banner.show("Đã lưu cài đặt thành công!", level="success")
        self._log("Cài đặt hệ thống đã được cập nhật.")

    def _refresh_highlight_table(self) -> None:
        for item in self.highlight_tree.get_children():
            self.highlight_tree.delete(item)
        for index, source in enumerate(self.highlight_sources, 1):
            self.highlight_tree.insert("", "end", iid=f"hl_ep_{index}", values=(index, source.basename, "Ready", "0%", "Ready"))
        if self.highlight_sources:
            label = "Episode" if len(self.highlight_sources) == 1 else "Season"
            self.highlight_source_var.set(f"{label}: {len(self.highlight_sources)} source file(s)")

    def _clear_highlight(self) -> None:
        if self.worker.is_running or self.discovery_worker.is_running:
            return
        self.highlight_sources.clear()
        self._refresh_highlight_table()
        self.highlight_source_var.set("No highlight source selected.")

    def _on_start_highlight(self) -> None:
        """Start the distinct Highlight workflow; never enters Recap narration."""
        if self.worker.is_running or self.discovery_worker.is_running:
            self.banner.show("A workflow is already active.", level="warning")
            return
        if not self.highlight_sources:
            self.banner.show("Select an episode or season folder for Highlight Mode.", level="warning")
            return
        prompt = self.highlight_prompt_text.get("1.0", "end-1c").strip()
        if not prompt:
            self.banner.show("Highlight Prompt cannot be empty.", level="warning")
            return
        selected_paths = {str(Path(source.path).resolve()).casefold() for source in self.highlight_sources}
        for project in self.persistence.list_projects():
            if project.get("project_mode", "RECAP") != "HIGHLIGHT" or project.get("status") == ProjectStatus.COMPLETED.value:
                continue
            saved_paths = {
                str(Path(item.get("fingerprint", {}).get("path", "")).resolve()).casefold()
                for item in project.get("sources", []) if item.get("fingerprint", {}).get("path")
            }
            if saved_paths == selected_paths:
                self._check_restart_projects(select_project_id=project["project_id"])
                self.banner.show("This Highlight source already has an unfinished project. Use Resume.", level="warning")
                return
        self.settings = self.settings_manager.load()
        self.settings.highlight_prompt = prompt
        self.settings.review_before_render = bool(self.highlight_page.review_var.get())
        self.settings_manager.save(self.settings)
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        first = Path(self.highlight_sources[0].path)
        base = first.stem if len(self.highlight_sources) == 1 else (first.parent.name or "season")
        clean = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)[:24]
        project_id = f"highlight_{clean}_{now_str}"
        self.current_project_id = project_id
        self.current_project_name = base
        self._current_project_by_mode["HIGHLIGHT"] = project_id
        self._set_running_state(True)
        self._activate_operation_ui("HIGHLIGHT", resuming=False)
        self.worker.start_highlight_project(
            project_id=project_id,
            project_name=base,
            source_input=[source.path for source in self.highlight_sources],
            prompt=prompt,
            settings=self.settings,
        )

    # -------------------------------------------------------------------------
    # GPU and Updater Background Checks
    # -------------------------------------------------------------------------
    def _detect_gpu_background(self) -> None:
        msg_queue = self.msg_queue

        def _check() -> None:
            try:
                status = detect_gpu_encoder()
                if status.available:
                    label = f"Mã hóa: {status.label} ({status.gpu_name})" if status.gpu_name else f"Mã hóa: {status.label}"
                else:
                    label = "Mã hóa: CPU (libx264)"
            except Exception:
                label = "Phần cứng: CPU"
            msg_queue.put(WorkerMessage(kind="gpu_status", data=label))

        threading.Thread(target=_check, daemon=True, name="GpuCheckThread").start()

    def _set_discovery_busy(self, is_busy: bool, msg: str = "") -> None:
        page = self.highlight_page if self._discovery_mode == "highlight" else self.recap_page
        if is_busy:
            page.start_button.config(state="disabled")
            page.stop_button.config(state="normal")
            page.resume_button.config(state="disabled")
            if page.use_json_button is not None:
                page.use_json_button.config(state="disabled")
            if msg:
                self.status_var.set(msg)
                (self.highlight_source_var if self._discovery_mode == "highlight" else self.source_var).set(msg)
        else:
            if not self.worker.is_running:
                page.start_button.config(state="normal")
                page.stop_button.config(state="disabled")
                mode = "HIGHLIGHT" if self._discovery_mode == "highlight" else "RECAP"
                page.resume_button.config(state="normal" if self._current_project_by_mode[mode] else "disabled")
                if page.use_json_button is not None:
                    page.use_json_button.config(state="normal")
                page.select_file_button.config(state="normal")
                page.select_folder_button.config(state="normal")
                page.clear_button.config(state="normal")

    def _begin_discovery_activity(self, path: Path, *, is_folder: bool) -> None:
        """Reset stale workflow status and show source discovery immediately."""
        mode = "HIGHLIGHT" if self._discovery_mode == "highlight" else "RECAP"
        mode_name = "Highlight" if mode == "HIGHLIGHT" else "Recap"
        target = "folder" if is_folder else "file"
        now = datetime.now(timezone.utc).isoformat()
        self._activity_context = "discovery"
        self._discovery_started_monotonic = time.monotonic()
        self._set_activity_snapshot({
            "state": ActivityState.LOCAL_PROCESSING.value,
            "stage": WorkflowStage.PREPARATION.value,
            "stage_label": STAGE_LABELS[WorkflowStage.PREPARATION.value],
            "activity_text": f"Scanning {target}: {path.name}",
            "active": True,
            "current_item": path.name,
            "completed": 0,
            # Zero means the directory enumeration is still establishing the
            # total.  Discovery updates it as soon as candidates are known.
            "total": 0,
            "unit": "videos",
            "session_elapsed_seconds": 0.0,
            "project_elapsed_seconds": None,
            "stage_elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "recent_activity": [{
                "timestamp": now,
                "text": f"{mode_name} source scan started: {path.name}",
            }],
            "last_activity_at": now,
            "event_timestamp": now,
            "waiting_for": None,
            "source_discovery": True,
            "pipeline_mode": mode,
            "pipeline": {
                stage: ("active" if stage == WorkflowStage.PREPARATION.value else "pending")
                for stage in PIPELINE_STAGES
            },
        })

    def _update_discovery_activity(
        self,
        *,
        current: int,
        total: int,
        name: str,
        source: SourceFingerprint | None = None,
        finished: bool = False,
        failed: str | None = None,
        cancelled: bool = False,
    ) -> None:
        """Render source scanning in the shared real-time activity panel."""
        now = datetime.now(timezone.utc).isoformat()
        started = self._discovery_started_monotonic
        elapsed = max(0.0, time.monotonic() - started) if started is not None else 0.0
        rows = list(self._activity_snapshot.get("recent_activity") or [])
        if failed:
            state = ActivityState.FAILED.value
            activity = f"Source scan failed: {failed}"
            event = activity
        elif cancelled:
            state = ActivityState.CANCELLED.value
            activity = "Source scan cancelled."
            event = activity
        elif finished:
            state = ActivityState.COMPLETE.value
            activity = f"Source scan complete: {total} video(s) ready."
            event = activity
        else:
            state = ActivityState.LOCAL_PROCESSING.value
            activity = f"Scanning sources ({current}/{total})..." if total else "Scanning sources..."
            event = f"Found {source.basename} ({current}/{total})." if source else activity
        if not rows or rows[-1].get("text") != event:
            rows.append({"timestamp": now, "text": event})
        patch = {
            "state": state,
            "activity_text": activity,
            "active": not (finished or failed or cancelled),
            "current_item": source.basename if source is not None else name,
            "completed": current,
            "total": total,
            "unit": "videos",
            "session_elapsed_seconds": elapsed,
            "stage_elapsed_seconds": elapsed,
            "last_activity_at": now,
            "event_timestamp": now,
            "recent_activity": rows[-12:],
        }
        if failed:
            patch["error"] = failed
        pipeline = dict(self._activity_snapshot.get("pipeline") or {})
        pipeline[WorkflowStage.PREPARATION.value] = "complete" if finished else (
            "failed" if failed else "active"
        )
        patch["pipeline"] = pipeline
        self._set_activity_snapshot(patch, merge=True)

    # -------------------------------------------------------------------------
    # Source Loading (Single File -> Single Episode, Folder -> Season)
    # -------------------------------------------------------------------------
    def _choose_file(self) -> None:
        self._choose_file_for_mode("highlight" if self.current_page == "highlight" else "recap")

    def _choose_file_for_mode(self, mode: str) -> None:
        if self.worker.is_running:
            return
        self._discovery_mode = mode
        filetypes = [
            ("Video files", "*.mp4;*.mkv;*.mov;*.avi;*.webm;*.m4v;*.ts;*.m2ts"),
            ("All files", "*.*"),
        ]
        chosen = filedialog.askopenfilename(
            parent=self,
            title="Chọn file video",
            filetypes=filetypes,
        )
        if chosen:
            self._load_file(Path(chosen))

    def _choose_folder(self) -> None:
        self._choose_folder_for_mode("highlight" if self.current_page == "highlight" else "recap")

    def _choose_folder_for_mode(self, mode: str) -> None:
        if self.worker.is_running:
            return
        self._discovery_mode = mode
        chosen = filedialog.askdirectory(
            parent=self,
            title="Chọn thư mục chứa video",
        )
        if chosen:
            self._load_folder(Path(chosen))

    def _load_file(self, path: Path) -> None:
        p = Path(path).resolve()
        if self._discovery_mode == "highlight":
            self.highlight_sources.clear()
            self._refresh_highlight_table()
        else:
            self.discovered_sources.clear()
            self._refresh_queue_table()
        self._set_discovery_busy(True, f"Đang kiểm tra file: {p.name}")
        self._begin_discovery_activity(p, is_folder=False)
        self._active_discovery_token = self.discovery_worker.start_discovery(p, is_folder=False)

    def _load_folder(self, path: Path) -> None:
        p = Path(path).resolve()
        if self._discovery_mode == "highlight":
            self.highlight_sources.clear()
            self._refresh_highlight_table()
        else:
            self.discovered_sources.clear()
            self._refresh_queue_table()
        self._set_discovery_busy(True, f"Đang quét thư mục: {p.name}")
        self._begin_discovery_activity(p, is_folder=True)
        self._active_discovery_token = self.discovery_worker.start_discovery(p, is_folder=True)

    def _clear_queue(self) -> None:
        if self.worker.is_running:
            return
        if self.discovery_worker.is_running:
            self.discovery_worker.cancel()
        self._active_discovery_token = 0
        self._set_discovery_busy(False)
        self.discovered_sources.clear()
        self._output_paths.clear()
        self._refresh_queue_table()
        self.source_var.set("Chưa chọn video hoặc thư mục.")
        self.status_var.set("Sẵn sàng.")
        self.progress_var.set(0.0)
        self.banner.dismiss()

    def _on_clear_sources(self) -> None:
        self._clear_queue()

    def _on_select_file(self) -> None:
        self._choose_file()

    def _on_select_folder(self) -> None:
        self._choose_folder()

    def _refresh_queue_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        count = len(self.discovered_sources)
        if count == 0:
            self.source_var.set("Chưa chọn video hoặc thư mục.")
            return

        if count == 1:
            fp = self.discovered_sources[0]
            self.source_var.set(f"File: {fp.basename}")
            self.tree.insert(
                "",
                "end",
                iid="ep_1",
                values=("1", fp.basename, "Sẵn sàng", "0%", "Sẵn sàng"),
            )
        else:
            self.source_var.set(f"Folder: {count} video")
            for idx, fp in enumerate(self.discovered_sources, start=1):
                self.tree.insert(
                    "",
                    "end",
                    iid=f"ep_{idx}",
                    values=(str(idx), fp.basename, "Sẵn sàng", "0%", "Sẵn sàng"),
                )
            if self.current_project_id and "season" in self.current_project_id:
                self.tree.insert(
                    "",
                    "end",
                    iid="season",
                    values=("Season", "Toàn bộ mùa phim", "Sẵn sàng", "0%", "Sẵn sàng"),
                )

    def _refresh_sources_table(self) -> None:
        self._refresh_queue_table()

    # -------------------------------------------------------------------------
    # WORKFLOW CONTROL (START / STOP / RESUME / JSON)
    # -------------------------------------------------------------------------
    def _on_start_project(self) -> None:
        if self.worker.is_running or self.discovery_worker.is_running:
            messagebox.showwarning("Đang thực hiện", "Một tác vụ đang chạy. Vui lòng chờ hoặc bấm Dừng.", parent=self)
            return

        if not self.discovered_sources:
            self.banner.show("Chưa có video nào trong danh sách. Hãy chọn file hoặc thư mục trước.", level="warning")
            return

        selected_paths = {str(Path(source.path).resolve()).casefold() for source in self.discovered_sources}
        for project in self.persistence.list_projects():
            if project.get("project_mode", "RECAP") != "RECAP":
                continue
            if project.get("status") not in {status.value for status in ProjectStatus if status != ProjectStatus.COMPLETED}:
                continue
            saved_paths = {
                str(Path(entry["fingerprint"]["path"]).resolve()).casefold()
                for entry in project.get("sources", [])
                if isinstance(entry, dict) and isinstance(entry.get("fingerprint"), dict)
                and isinstance(entry["fingerprint"].get("path"), str)
            }
            if saved_paths and saved_paths == selected_paths:
                self._check_restart_projects(select_project_id=project["project_id"])
                self.banner.show(
                    f"Nguồn này đã thuộc dự án chưa hoàn thành '{project['project_id']}'. "
                    "Hãy dùng 'Tiếp tục dự án đã lưu'.",
                    level="warning",
                )
                return

        self.settings = self.settings_manager.load()
        recap_prompt = self.recap_prompt_text.get("1.0", "end-1c").strip()
        if not recap_prompt:
            self.banner.show("Recap Prompt cannot be empty.", level="warning")
            return
        self.settings.prompt = recap_prompt
        self.settings.review_before_render = bool(self.recap_page.review_var.get())
        self.settings_manager.save(self.settings)

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        first_path = Path(self.discovered_sources[0].path)
        if len(self.discovered_sources) == 1:
            stem = first_path.stem
            proj_name = stem
            clean_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:24]
            proj_id = f"proj_{clean_stem}_{now_str}"
        else:
            parent_name = first_path.parent.name or "Season"
            proj_name = parent_name
            clean_parent = "".join(c if c.isalnum() or c in "._-" else "_" for c in parent_name)[:24]
            proj_id = f"season_{clean_parent}_{now_str}"

        self.current_project_id = proj_id
        self.current_project_name = proj_name
        self._current_project_by_mode["RECAP"] = proj_id

        self._refresh_queue_table()
        self._set_running_state(True)
        self._activate_operation_ui("RECAP", resuming=False)
        self.status_var.set(f"Bắt đầu dự án: {proj_name}")
        self._log(f"Khởi động dự án mới '{proj_name}' (ID: {proj_id})...")

        source_paths = [fp.path for fp in self.discovered_sources]
        out_dir = self.settings.output_dir if self.settings.output_dir.strip() else None

        self.worker.start_new_project(
            project_id=proj_id,
            project_name=proj_name,
            source_input=source_paths,
            prompt=recap_prompt,
            output_dir=out_dir,
            settings=self.settings,
        )

    def _on_stop_project(self) -> None:
        if self.discovery_worker.is_running:
            self.discovery_worker.cancel()
            self.status_var.set("Đang dừng tác vụ chọn nguồn...")
            self._log("Người dùng bấm Dừng chọn nguồn. Đang gửi tín hiệu hủy...")
            return

        if not self.worker.is_running:
            return

        ans = messagebox.askyesno(
            "Xác nhận dừng",
            "Bạn có chắc chắn muốn dừng tiến trình hiện tại không?\n"
            "Dự án sẽ dừng an toàn và lưu điểm dừng (checkpoint) để có thể tiếp tục sau.",
            parent=self,
        )
        if ans:
            self.status_var.set("Đang dừng các tác vụ và đóng tiến trình...")
            self._log("Người dùng bấm Dừng. Đang gửi tín hiệu hủy...")
            self.worker.cancel()

    def _on_resume_current(self, mode: str | None = None) -> None:
        if self.worker.is_running or self.discovery_worker.is_running:
            return
        expected_mode = mode or ("HIGHLIGHT" if self.current_page == "highlight" else "RECAP")
        project_id = self._current_project_by_mode.get(expected_mode)
        if not project_id:
            self.banner.show("No saved project is selected for this mode.", level="info")
            return
        project = self.persistence.load_project(project_id)
        if project.get("project_mode", "RECAP") != expected_mode:
            self.banner.show("Saved project mode does not match this workspace.", level="error")
            return

        self._set_running_state(True)
        self._activate_operation_ui(expected_mode, resuming=True)
        self.current_project_id = project_id
        self.current_project_name = project.get("project_name") or project_id
        self._log(f"Tiếp tục dự án '{project_id}'...")
        self.status_var.set(f"Tiếp tục dự án '{self.current_project_name}'...")
        self.worker.resume_project(
            project_id=project_id,
            output_dir=self.settings.output_dir or None,
            settings=(AppSettings.from_dict({**self.settings.to_dict(), "review_before_render": True})
                      if project.get("status") == ProjectStatus.REVIEW_REQUIRED.value else self.settings),
        )

    def _on_review_plan(self, mode: str) -> None:
        project_id = self._current_project_by_mode.get(mode)
        if not project_id:
            self.banner.show("Select a saved project to review its plan.", level="info")
            return
        state = self.persistence.load_project(project_id)
        if state.get("project_mode", "RECAP") != mode:
            self.banner.show("This project belongs to another mode.", level="error")
            return
        from toolrecap_v4.review import plan_data, plan_digest, parse_timecode, review_text, save_reviewed_plan
        draft = copy.deepcopy(state)
        shown_digest = plan_digest(state)
        dialog = tk.Toplevel(self)
        dialog.title(f"Review plan — {state.get('project_name', project_id)}")
        dialog.geometry("860x620")
        editor = ScrolledText(dialog, wrap="word", font=("Consolas", 10))
        editor.pack(fill="both", expand=True, padx=12, pady=12)
        editor.insert("1.0", review_text(draft))
        editor.configure(state="disabled")
        clips = []
        for output in (plan_data(draft) or {}).get("outputs", []):
            for clip in output.get("clips", output.get("segments", [])):
                clips.append((output, clip))
        if clips:
            ttk.Label(dialog, text="Select a clip to inspect its cut boundaries:").pack(anchor="w", padx=12)
            choices = tk.Listbox(dialog, height=min(5, len(clips)), exportselection=False)
            choices.pack(fill="x", padx=12, pady=(3, 8))
            for output, clip in clips:
                choices.insert("end", f"{output['title']} — {clip['source_file']} "
                              f"({clip['start_ms']/1000:.3f}s → {clip['end_ms']/1000:.3f}s)")
            choices.selection_set(0)
            preview_buttons = ttk.Frame(dialog)
            preview_buttons.pack(fill="x", padx=12, pady=(0, 8))

            def refresh_review() -> None:
                clips.clear()
                choices.delete(0, "end")
                for output in (plan_data(draft) or {}).get("outputs", []):
                    for clip in output.get("clips", output.get("segments", [])):
                        clips.append((output, clip))
                        choices.insert("end", f"{output['title']} — {clip['source_file']} "
                                      f"({clip['start_ms']/1000:.3f}s → {clip['end_ms']/1000:.3f}s)")
                if clips:
                    choices.selection_set(0)
                editor.configure(state="normal")
                editor.delete("1.0", "end")
                editor.insert("1.0", review_text(draft))
                editor.configure(state="disabled")

            def preview_boundary(at_end: bool) -> None:
                selection = choices.curselection()
                if not selection:
                    return
                _, clip = clips[selection[0]]
                source_item = next((item for item in state.get("sources", [])
                                    if item.get("source_file") == clip.get("source_file")), None)
                if source_item is None:
                    messagebox.showerror("Missing source", "The selected clip's source is unavailable.", parent=dialog)
                    return
                fingerprint = source_item.get("fingerprint", {})

                def generate() -> None:
                    try:
                        from toolrecap_v4.preview import boundary_preview
                        path = boundary_preview(Path(fingerprint["path"]), clip["start_ms"], clip["end_ms"],
                                                at_end=at_end, storage_root=self.persistence.root,
                                                source_hash=fingerprint.get("sha256", ""))
                        safe_after(dialog, 0, lambda: os.startfile(path) if sys.platform == "win32" else None)
                    except Exception as exc:
                        safe_after(dialog, 0, lambda error=str(exc): messagebox.showerror(
                            "Preview failed", error, parent=dialog))

                threading.Thread(target=generate, daemon=True, name="ClipPreviewWorker").start()

            ttk.Button(preview_buttons, text="Preview start (5s)",
                       command=lambda: preview_boundary(False)).pack(side="left", padx=(0, 8))
            ttk.Button(preview_buttons, text="Preview end (5s)",
                       command=lambda: preview_boundary(True)).pack(side="left")
            if state.get("status") == ProjectStatus.REVIEW_REQUIRED.value:
                def selected_pair():
                    selection = choices.curselection()
                    return clips[selection[0]] if selection and selection[0] < len(clips) else None

                def edit_title() -> None:
                    pair = selected_pair()
                    if not pair:
                        return
                    output, _clip = pair
                    value = simpledialog.askstring("Edit title", "Publication title:",
                                                   initialvalue=output["title"], parent=dialog)
                    if value is not None:
                        output["title"] = value.strip()
                        refresh_review()

                def edit_cut() -> None:
                    pair = selected_pair()
                    if not pair:
                        return
                    _output, clip = pair
                    from toolrecap_v4.review import _clock
                    start = simpledialog.askstring("Edit clip start", "Start (HH:MM:SS.mmm):",
                                                   initialvalue=_clock(clip["start_ms"]), parent=dialog)
                    if start is None:
                        return
                    end = simpledialog.askstring("Edit clip end", "End (HH:MM:SS.mmm):",
                                                 initialvalue=_clock(clip["end_ms"]), parent=dialog)
                    if end is None:
                        return
                    try:
                        start_ms, end_ms = parse_timecode(start), parse_timecode(end)
                        if end_ms <= start_ms:
                            raise ValueError("End must be after start")
                    except ValueError as exc:
                        messagebox.showerror("Invalid time", str(exc), parent=dialog)
                        return
                    clip["start_ms"], clip["end_ms"] = start_ms, end_ms
                    if mode == "HIGHLIGHT" and "subtitle_cues" in clip:
                        clip["subtitle_cues"] = [cue for cue in clip.get("subtitle_cues", [])
                                                 if cue["start_ms"] >= start_ms and cue["end_ms"] <= end_ms]
                    refresh_review()

                def omit_story() -> None:
                    pair = selected_pair()
                    if pair:
                        plan_data(draft)["outputs"].remove(pair[0])
                        refresh_review()

                ttk.Button(preview_buttons, text="Edit title", command=edit_title).pack(side="left", padx=(8, 0))
                ttk.Button(preview_buttons, text="Edit cut", command=edit_cut).pack(side="left", padx=(8, 0))
                ttk.Button(preview_buttons, text="Omit story", command=omit_story).pack(side="left", padx=(8, 0))
        buttons = ttk.Frame(dialog)
        buttons.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(buttons, text="Close", command=dialog.destroy).pack(side="right")
        if state.get("status") == ProjectStatus.REVIEW_REQUIRED.value:
            def save_changes() -> bool:
                nonlocal shown_digest
                current = self.persistence.load_project(project_id)
                if plan_digest(current) != shown_digest:
                    messagebox.showerror("Plan changed", "The plan changed. Reopen it before saving.", parent=dialog)
                    return False
                if plan_digest(draft) != shown_digest:
                    try:
                        saved = save_reviewed_plan(self.persistence, project_id, plan_data(draft))
                    except Exception as exc:
                        messagebox.showerror("Invalid plan edit", str(exc), parent=dialog)
                        return False
                    draft.clear()
                    draft.update(copy.deepcopy(saved))
                    shown_digest = plan_digest(saved)
                    if clips:
                        refresh_review()
                return True

            def approve() -> None:
                if not save_changes():
                    return
                current = self.persistence.load_project(project_id)
                digest = plan_digest(current)
                if not digest or digest != current.get("review_plan_hash"):
                    messagebox.showerror("Plan changed", "The plan changed. Reopen it before approval.", parent=dialog)
                    return
                current["review_approved_hash"] = digest
                self.persistence.save_project(current)
                dialog.destroy()
                self._on_resume_current(mode)
            ttk.Button(buttons, text="Save changes", command=save_changes).pack(side="right", padx=(0, 8))
            ttk.Button(buttons, text="Approve & Render", command=approve).pack(side="right", padx=(0, 8))

    def _on_render_existing_json(self) -> None:
        if self.worker.is_running or self.discovery_worker.is_running:
            messagebox.showwarning("Đang thực hiện", "Một tác vụ đang chạy. Vui lòng chờ hoặc bấm Dừng.", parent=self)
            return

        json_path_str = filedialog.askopenfilename(
            parent=self,
            title="Chọn tệp Final JSON có sẵn để dựng (0 AI)",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not json_path_str:
            return

        try:
            final_data = read_json(Path(json_path_str))
        except Exception as e:
            messagebox.showerror("Lỗi đọc JSON", f"Không thể đọc tệp JSON: {e}", parent=self)
            return

        if not self.discovered_sources:
            messagebox.showinfo(
                "Chọn video nguồn",
                "Vui lòng chọn tệp video hoặc thư mục chứa các video nguồn trước khi dùng tính năng dựng từ JSON.",
                parent=self,
            )
            return

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        proj_id = f"import_{now_str}"
        proj_name = Path(json_path_str).stem

        self.current_project_id = proj_id
        self.current_project_name = proj_name

        self._refresh_queue_table()
        self._populate_outputs_from_json(final_data)
        self._set_running_state(True)
        self.status_var.set(f"Dựng từ JSON có sẵn: {proj_name}")
        self._log(f"Dựng từ JSON có sẵn '{proj_name}' (0 yêu cầu AI)...")

        source_paths = [fp.path for fp in self.discovered_sources]
        out_dir = self.settings.output_dir if self.settings.output_dir.strip() else None

        self.worker.render_existing_json(
            project_id=proj_id,
            project_name=proj_name,
            source_input=source_paths,
            final_json=final_data,
            output_dir=out_dir,
            settings=self.settings,
        )

    def _open_output_folder(self) -> None:
        target_dir = None
        selected_mode = "HIGHLIGHT" if self.current_page == "highlight" else "RECAP"
        selected_project = self.current_project_id if self.worker.is_running else self._current_project_by_mode[selected_mode]
        if selected_project:
            try:
                state = self.persistence.load_project(selected_project)
                target_dir = state.get("output_dir")
            except Exception:
                pass

        if not target_dir:
            target_dir = self.settings.output_dir or str(self.persistence.root / "outputs")

        p = Path(target_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(p))
        else:
            subprocess.Popen(["xdg-open", str(p)])

    def _on_open_output_dir(self) -> None:
        self._open_output_folder()

    def _on_open_settings(self, initial_tab: Optional[int | str] = None) -> None:
        if self.settings_page is None:
            return
        if initial_tab is not None:
            self.settings_page._select_tab(initial_tab)
        self._show_page("settings")

    def _on_window_activated(self, _event: Any = None) -> None:
        """Restore the one main application window when Windows activates it."""
        if self._shutting_down or self._is_closed:
            return
        try:
            if self.state() in ("withdrawn", "iconic"):
                self.deiconify()
            self.lift()
        except tk.TclError:
            pass

    def _on_check_update(self) -> None:
        self._on_open_settings(initial_tab="update")

    def _manual_check_update(self) -> None:
        self._on_check_update()

    # -------------------------------------------------------------------------
    # RESTARTS / RESUME DETECTION
    # -------------------------------------------------------------------------
    def _show_persisted_project(self, project: Dict[str, Any]) -> None:
        self._activity_context = "persisted"
        project_id = project["project_id"]
        self.current_project_id = project_id
        self.current_project_name = project.get("project_name") or project_id
        mode = project.get("project_mode", "RECAP")
        self._current_project_by_mode[mode] = project_id
        self._output_paths.clear()
        sources = project.get("sources", [])
        target_tree = self.highlight_tree if mode == "HIGHLIGHT" else self.tree
        for item in target_tree.get_children():
            target_tree.delete(item)
        for index, source in enumerate(sources, start=1):
            source_name = source.get("source_file", "") if isinstance(source, dict) else ""
            target_tree.insert(
                "", "end", iid=f"saved_ep_{index}",
                values=(str(index), source_name, project.get("status", ""), "", "Đã lưu"),
            )
        if mode == "HIGHLIGHT":
            self.highlight_source_var.set(f"Saved Highlight project: {self.current_project_name} ({len(sources)} video)")
            self._show_page("highlight")
        else:
            self.source_var.set(f"Dự án đã lưu: {self.current_project_name} ({len(sources)} video)")
            self._show_page("recap")
        self.status_var.set(f"Có thể tiếp tục dự án '{self.current_project_name}' từ checkpoint đã lưu.")
        (self.highlight_page.resume_button if mode == "HIGHLIGHT" else self.recap_page.resume_button).config(state="normal")
        self._load_persisted_activity_async(project_id)

    def _load_persisted_activity_async(self, project_id: str) -> None:
        """Reconstruct checkpoints off the Tk thread, once per project selection."""
        msg_queue = self.msg_queue
        persistence = self.persistence

        def _load() -> None:
            try:
                snapshot = reconstruct_project_progress(persistence, project_id)
                msg_queue.put(WorkerMessage(kind="activity", data=snapshot))
            except Exception as exc:
                logger.warning("Failed to reconstruct saved workflow progress for %s: %s", project_id, exc)

        threading.Thread(
            target=_load, daemon=True, name=f"ProgressReconstruction-{project_id[:24]}",
        ).start()

    def _on_select_resume_project(self, _event: Any = None) -> None:
        project_id = self._resume_project_ids.get(self.cmb_resume_projects.get())
        if not project_id:
            return
        try:
            self._show_persisted_project(self.persistence.load_project(project_id))
        except Exception as exc:
            self.banner.show(f"Không thể nạp dự án đã lưu: {exc}", level="error")

    def _on_select_highlight_resume_project(self, _event: Any = None) -> None:
        project_id = self._highlight_resume_project_ids.get(self.highlight_page.saved_combo.get())
        if not project_id:
            return
        try:
            project = self.persistence.load_project(project_id)
            if project.get("project_mode") != "HIGHLIGHT":
                raise ValueError("Selected project is not a Highlight project")
            self._show_persisted_project(project)
        except Exception as exc:
            self.banner.show(f"Cannot load saved Highlight project: {exc}", level="error")

    def _check_restart_projects(self, *, select_project_id: str | None = None) -> None:
        """Detect any unfinished projects from past sessions and offer resume."""
        try:
            projects = self.persistence.list_projects()
            recap_projects: Dict[str, str] = {}
            highlight_projects: Dict[str, str] = {}
            statuses = {status.value for status in ProjectStatus if status != ProjectStatus.COMPLETED}
            for p in projects:
                status = p.get("status")
                project_id = p.get("project_id")
                if status in statuses and isinstance(project_id, str) and project_id:
                    name = p.get("project_name") or project_id
                    target = highlight_projects if p.get("project_mode", "RECAP") == "HIGHLIGHT" else recap_projects
                    target[f"{name} — {project_id} ({status})"] = project_id

            previous_recap = select_project_id or self._resume_project_ids.get(self.recap_page.saved_combo.get())
            previous_highlight = select_project_id or self._highlight_resume_project_ids.get(self.highlight_page.saved_combo.get())
            self._resume_project_ids = recap_projects
            self._highlight_resume_project_ids = highlight_projects
            recap_selected = next((label for label, pid in recap_projects.items() if pid == previous_recap), "")
            highlight_selected = next((label for label, pid in highlight_projects.items() if pid == previous_highlight), "")
            if not recap_selected and recap_projects:
                recap_selected = next(iter(recap_projects))
            if not highlight_selected and highlight_projects:
                highlight_selected = next(iter(highlight_projects))
            self.recap_page.show_saved_projects(list(recap_projects), recap_selected)
            self.highlight_page.show_saved_projects(list(highlight_projects), highlight_selected)
            if recap_selected:
                self._current_project_by_mode["RECAP"] = recap_projects[recap_selected]
                self.recap_page.resume_button.config(state="normal")
            else:
                self.recap_page.resume_button.config(state="disabled")
            if highlight_selected:
                self._current_project_by_mode["HIGHLIGHT"] = highlight_projects[highlight_selected]
                self.highlight_page.resume_button.config(state="normal")
            else:
                self.highlight_page.resume_button.config(state="disabled")
            if select_project_id:
                selected_project = next(
                    (project for project in projects if project.get("project_id") == select_project_id), None,
                )
                if selected_project is not None:
                    self._show_persisted_project(selected_project)
        except Exception as e:
            logger.warning("Failed to check restart projects: %s", e)

    def _on_resume_detected_project(self, mode: str = "RECAP") -> None:
        if self.worker.is_running or self.discovery_worker.is_running:
            return
        mapping = self._highlight_resume_project_ids if mode == "HIGHLIGHT" else self._resume_project_ids
        combo = self.highlight_page.saved_combo if mode == "HIGHLIGHT" else self.recap_page.saved_combo
        proj_id = mapping.get(combo.get())
        if not proj_id:
            return
        try:
            proj_data = self.persistence.load_project(proj_id)
            if proj_data.get("project_mode", "RECAP") != mode:
                raise ValueError(f"Cannot resume {proj_data.get('project_mode', 'RECAP')} project from {mode} workspace")
            self._show_persisted_project(proj_data)

            final_json = proj_data.get("final_json")
            if final_json:
                self._populate_outputs_from_json(final_json)

            self._set_running_state(True)
            self._activate_operation_ui(mode, resuming=True)
            self._log(f"Tiếp tục thực hiện dự án: '{proj_id}'...")
            self.settings = self.settings_manager.load()
            self.worker.resume_project(
                project_id=proj_id,
                output_dir=self.settings.output_dir or None,
                settings=self.settings,
            )
        except Exception as e:
            messagebox.showerror("Lỗi nạp dự án", f"Không thể nạp dự án {proj_id}: {e}", parent=self)

    def _on_delete_saved_project(self, mode: str) -> None:
        """Delete a user-selected unfinished project but preserve publications."""
        if self.worker.is_running or self.discovery_worker.is_running:
            self.banner.show("Stop the active task before deleting a saved project.", level="warning")
            return
        mapping = self._highlight_resume_project_ids if mode == "HIGHLIGHT" else self._resume_project_ids
        page = self.highlight_page if mode == "HIGHLIGHT" else self.recap_page
        project_id = mapping.get(page.saved_combo.get())
        if not project_id:
            self.banner.show("Select an unfinished project to delete.", level="info")
            return
        try:
            project = self.persistence.load_project(project_id)
        except Exception as exc:
            self.banner.show(f"Cannot load saved project: {exc}", level="error")
            return
        project_name = project.get("project_name") or project_id
        confirmed = messagebox.askyesno(
            "Delete unfinished project",
            f"Delete '{project_name}' and all of its saved checkpoints?\n\n"
            "Published MP4/SRT files will be kept.",
            parent=self,
        )
        if not confirmed:
            return
        try:
            self.persistence.delete_project(project_id)
            if self.current_project_id == project_id:
                self.current_project_id = None
                self.current_project_name = None
            if self._current_project_by_mode.get(mode) == project_id:
                self._current_project_by_mode.pop(mode, None)
            if mode == "HIGHLIGHT":
                self._clear_highlight()
            else:
                self._clear_queue()
            self._check_restart_projects()
            self.banner.show(f"Deleted unfinished project '{project_name}'.", level="success")
            self._log(f"Đã xóa project và checkpoint: {project_id}. File output đã xuất vẫn được giữ lại.")
        except Exception as exc:
            self.banner.show(f"Cannot delete saved project: {exc}", level="error")

    # -------------------------------------------------------------------------
    # QUEUE POLLING & DISPATCHING (THREAD SAFE)
    # -------------------------------------------------------------------------
    def _poll_queue(self) -> None:
        if getattr(self, "_is_closed", False):
            return
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self._handle_worker_message(msg)
        except queue.Empty:
            pass
        except Exception:
            # A dispatch failure used to be swallowed silently, leaving source
            # discovery apparently stuck with an empty queue.  Preserve UI
            # polling, but make the actual fault diagnosable.
            logger.exception("Failed to dispatch a background worker message")
        finally:
            if not getattr(self, "_is_closed", False):
                self._poll_job = safe_after(self, 100, self._poll_queue)

    def _activate_operation_ui(self, mode: str, *, resuming: bool) -> None:
        """Paint accepted Start/Resume state before background preparation begins."""
        self._activity_context = "workflow"
        now = datetime.now(timezone.utc).isoformat()
        mode_name = "Highlight" if mode == "HIGHLIGHT" else "Recap"
        activity = "Loading saved project..." if resuming else (
            "Preparing Highlight sources..." if mode == "HIGHLIGHT" else "Preparing sources..."
        )
        event = "Resume requested." if resuming else (
            "Highlight project started." if mode == "HIGHLIGHT" else "Project started."
        )
        snapshot = {
            "project_id": self.current_project_id,
            "state": ActivityState.RESUMING.value if resuming else ActivityState.RUNNING.value,
            "stage": WorkflowStage.PREPARATION.value,
            "stage_label": STAGE_LABELS[WorkflowStage.PREPARATION.value],
            "activity_text": activity,
            "active": True,
            "current_item": None,
            "completed": None,
            "total": None,
            "percent": None,
            "unit": "episodes",
            "session_elapsed_seconds": 0.0,
            "project_elapsed_seconds": (
                self._activity_snapshot.get("project_elapsed_seconds") if resuming else 0.0
            ),
            "stage_elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "recent_activity": [
                {"timestamp": now, "text": event},
                {"timestamp": now, "text": activity},
            ],
            "last_activity_at": now,
            "event_timestamp": now,
            "waiting_for": None,
            "pipeline_mode": mode,
            "pipeline": {stage: "pending" for stage in PIPELINE_STAGES},
        }
        snapshot["pipeline"][WorkflowStage.PREPARATION.value] = "active"
        self._set_activity_snapshot(snapshot)
        self.status_var.set("Resuming" if resuming else "Running")
        # Flush layout/paint requests only; no heavy work or arbitrary sleeps.
        self.update_idletasks()

    def _set_activity_snapshot(self, snapshot: Dict[str, Any], *, merge: bool = False) -> None:
        if merge:
            self._activity_snapshot.update(snapshot)
        else:
            self._activity_snapshot = dict(snapshot)
        self._activity_received_monotonic = time.monotonic()
        self._render_activity_snapshot()

    def _activity_live_seconds(self, field: str) -> float | None:
        value = self._activity_snapshot.get(field)
        if value is None:
            return None
        result = float(value)
        if self._activity_snapshot.get("active"):
            result += max(0.0, time.monotonic() - self._activity_received_monotonic)
        return result

    def _last_activity_age(self) -> float | None:
        raw = self._activity_snapshot.get("last_activity_at")
        if not raw:
            return None
        try:
            value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - value.astimezone(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None

    def _update_activity_timer(self) -> None:
        if getattr(self, "_is_closed", False):
            return
        self._render_activity_snapshot(timer_only=True)
        self._activity_timer_job = safe_after(self, 1000, self._update_activity_timer)

    def _start_activity_dots(self) -> None:
        if self._dots_job is None:
            self._dots_phase = 1
            self._animate_activity_dots()

    def _animate_activity_dots(self) -> None:
        if getattr(self, "_is_closed", False) or not self._activity_snapshot.get("active"):
            self._dots_job = None
            return
        base = str(self._activity_snapshot.get("activity_text") or "Working")
        self.activity_dots_var.set(base + "." * self._dots_phase)
        self._dots_phase = 1 if self._dots_phase >= 3 else self._dots_phase + 1
        self._dots_job = safe_after(self, 700, self._animate_activity_dots)

    def _stop_activity_dots(self) -> None:
        if self._dots_job:
            try:
                self.after_cancel(self._dots_job)
            except Exception:
                pass
            self._dots_job = None
        self.activity_dots_var.set("")

    def _render_activity_snapshot(self, *, timer_only: bool = False) -> None:
        snapshot = self._activity_snapshot
        state = str(snapshot.get("state") or ActivityState.IDLE.value)
        self.activity_state_var.set(state)
        colors = {
            ActivityState.FAILED.value: "#b42318",
            ActivityState.COMPLETE.value: "#1a7f37",
            ActivityState.CANCELLED.value: "#9a6700",
            ActivityState.CANCELLING.value: "#9a6700",
            ActivityState.REPAIRING.value: "#9a6700",
            ActivityState.RETRYING.value: "#9a6700",
        }
        self.activity_state_label.config(foreground=colors.get(state, "#0969da"))

        self.activity_stage_time_var.set(format_workflow_duration(self._activity_live_seconds("stage_elapsed_seconds")))
        self.activity_session_time_var.set(format_workflow_duration(self._activity_live_seconds("session_elapsed_seconds")))
        self.activity_project_time_var.set(format_workflow_duration(self._activity_live_seconds("project_elapsed_seconds")))
        eta = snapshot.get("estimated_remaining_seconds")
        self.activity_eta_var.set("Unknown" if eta is None else format_workflow_duration(float(eta)))
        age = self._last_activity_age()
        self.activity_last_var.set(format_last_activity(age))
        if state == ActivityState.WAITING_FOR_AI.value:
            self.activity_wait_var.set(format_wait_duration(age))
        else:
            self.activity_wait_var.set("—")

        if timer_only:
            return

        self.activity_stage_var.set(str(snapshot.get("stage_label") or STAGE_LABELS.get(str(snapshot.get("stage") or ""), "Not started")))
        item = snapshot.get("current_item") or snapshot.get("chunk_id") or snapshot.get("output_id") or snapshot.get("episode_id")
        self.activity_item_var.set(str(item or "—"))
        self.activity_text_var.set(str(snapshot.get("activity_text") or "Ready"))
        completed = snapshot.get("completed")
        total = snapshot.get("total")
        unit = str(snapshot.get("unit") or "items")
        reused = snapshot.get("reused")
        if completed is not None and total is not None:
            if int(total) > 0:
                percent = float(completed) * 100.0 / float(total)
                progress_text = f"{completed} / {total} {unit} — {percent:.0f}%"
                if reused:
                    progress_text += f" (reused: {reused})"
                self.progress_var.set(percent)
                if self._progress_indeterminate:
                    self.progressbar.stop()
                    self._progress_indeterminate = False
                self.progressbar.config(mode="determinate")
                self.progressbar.grid()
                self.activity_dots_label.grid_remove()
                self._stop_activity_dots()
            else:
                progress_text = (
                    f"Scanning {unit} — total not yet known"
                    if snapshot.get("active") and snapshot.get("source_discovery")
                    else f"0 / 0 {unit} — Not required"
                )
                self.progress_var.set(0.0)
                if self._progress_indeterminate:
                    self.progressbar.stop()
                    self._progress_indeterminate = False
                self.progressbar.config(mode="determinate")
                self.progressbar.grid()
                self.activity_dots_label.grid_remove()
                self._stop_activity_dots()
        else:
            progress_text = "In progress — total not yet known" if snapshot.get("active") else "Progress not yet measurable"
            self.progressbar.stop()
            self.progressbar.grid_remove()
            self._progress_indeterminate = False
            if snapshot.get("active"):
                self.activity_dots_label.grid()
                self._start_activity_dots()
            else:
                self._stop_activity_dots()
        self.activity_progress_var.set(progress_text)

        short_labels = {
            "preparation": "Preparation", "scanner": "Scanner", "catalog": "Catalog",
            "planner": "Planner", "evidence": "Evidence", "vision": "Vision",
            "final_plan": "Final Plan", "writers": "Writers", "final_json": "Final JSON",
            "ai_gateway": "AI Gateway", "validate_json": "Validate JSON", "review": "Review",
            "voice": "Voice", "narration_fit": "Narration Fit", "audio_mix": "Audio",
            "subtitles": "Subtitles", "render": "Render", "publish": "Publish",
        }
        symbols = {"complete": "✓", "active": "●", "pending": "○", "retry": "↻", "skipped": "—", "failed": "!"}
        if snapshot.get("pipeline_mode") == "V5":
            pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
            self.pipeline_var.set("  →  ".join(
                f"{symbols.get(str(pipeline.get(stage, 'pending')), '○')} {short_labels[stage]}"
                for stage in V5_PIPELINE_STAGES
            ))
        elif snapshot.get("pipeline_mode") == "HIGHLIGHT":
            highlight_stages = (
                "Preparing Sources", "Scanning Evidence", "Building Highlight Coverage",
                "Planning Highlight Scenes", "Validating Highlights", "Building Highlight JSON",
                "Rendering Highlights", "Building Original Dialogue SRT", "Publishing",
            )
            current = str(snapshot.get("stage_label") or snapshot.get("stage") or "")
            current_index = highlight_stages.index(current) if current in highlight_stages else -1
            self.pipeline_var.set("  →  ".join(
                f"{'✓' if index < current_index or (index == current_index and state == ActivityState.COMPLETE.value) else '●' if index == current_index else '○'} {label}"
                for index, label in enumerate(highlight_stages)
            ))
        else:
            pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
            self.pipeline_var.set("  →  ".join(
                f"{symbols.get(str(pipeline.get(stage, 'pending')), '○')} {short_labels[stage]}"
                for stage in PIPELINE_STAGES
            ))

        summary: list[str] = []
        if snapshot.get("analysis_seconds") is not None:
            summary.append(f"Analysis Time: {format_workflow_duration(snapshot['analysis_seconds'])}")
        elif snapshot.get("analysis_time_unavailable"):
            summary.append("Previous Analysis Time: Unavailable")
        if snapshot.get("downstream_seconds") is not None:
            summary.append(f"Downstream Time: {format_workflow_duration(snapshot['downstream_seconds'])}")
        if snapshot.get("completion_seconds") is not None:
            summary.append(f"Completion Time: {format_workflow_duration(snapshot['completion_seconds'])}")
        if snapshot.get("published_count") is not None:
            summary.append(f"Videos Published: {snapshot['published_count']}")
        if snapshot.get("output_folder"):
            summary.append(f"Output Folder: {snapshot['output_folder']}")
        if snapshot.get("error"):
            summary.append(f"Error: {str(snapshot['error'])[:300]}")
        self.activity_summary_var.set("  |  ".join(summary))

        rows = snapshot.get("recent_activity") if isinstance(snapshot.get("recent_activity"), list) else []
        display_lines: list[str] = []
        for row in rows[-12:]:
            try:
                timestamp = datetime.fromisoformat(str(row.get("timestamp", "")).replace("Z", "+00:00")).astimezone()
                prefix = timestamp.strftime("%H:%M:%S")
            except (TypeError, ValueError):
                prefix = "--:--:--"
            display_lines.append(f"{prefix}  {str(row.get('text', ''))[:240]}")
        self.activity_log.config(state=tk.NORMAL)
        self.activity_log.delete("1.0", tk.END)
        self.activity_log.insert("1.0", "\n".join(display_lines) if display_lines else "No activity yet.")
        self.activity_log.config(state=tk.DISABLED)
        self.activity_log.see(tk.END)

    def _handle_worker_message(self, msg: WorkerMessage) -> None:
        kind = msg.kind
        data = msg.data

        if kind == "log":
            self._log(str(data))

        elif kind == "activity" and isinstance(data, dict):
            event_project = data.get("project_id")
            if event_project and self.current_project_id and event_project != self.current_project_id:
                return
            if data.get("reconstructed") and (
                self.worker.is_running
                or self.discovery_worker.is_running
                or self._activity_context == "discovery"
            ):
                return
            self._activity_context = "persisted" if data.get("reconstructed") else "workflow"
            self._set_activity_snapshot(data)

        elif kind == "activity_patch" and isinstance(data, dict):
            self._set_activity_snapshot(data, merge=True)

        elif kind == "status_change":
            status = str(data)
            status_map = {
                ProjectStatus.CREATED.value: "Sẵn sàng phân tích (NEW)",
                ProjectStatus.NEW.value: "Sẵn sàng phân tích (NEW)",
                ProjectStatus.ANALYZING.value: "Đang phân tích video qua AI Gateway...",
                ProjectStatus.ANALYZED.value: "Đã phân tích xong kịch bản (JSON_READY)",
                ProjectStatus.JSON_READY.value: "Đã có kịch bản JSON (JSON_READY)",
                ProjectStatus.RENDERING.value: "Đang dựng video thành phẩm...",
                ProjectStatus.COMPLETED.value: "Đã hoàn thành toàn bộ dự án!",
                ProjectStatus.CANCELLED.value: "Đã dừng tiến trình tại điểm checkpoint.",
                ProjectStatus.FAILED.value: "Dự án gặp sự cố dừng lại.",
            }
            display_status = status_map.get(status, status)
            self.status_var.set(display_status)
            self.lbl_stage.config(text=f"Trạng thái: {display_status}")

            if status == ProjectStatus.ANALYZING.value:
                for item in self.tree.get_children():
                    if item.startswith("ep_") or item == "season":
                        self.tree.set(item, "stage", "Analyzing")
                        self.tree.set(item, "status", "Đang phân tích")
            elif status in (ProjectStatus.ANALYZED.value, ProjectStatus.JSON_READY.value):
                for item in self.tree.get_children():
                    if item.startswith("ep_") or item == "season":
                        self.tree.set(item, "stage", "Evidence Complete")
                        self.tree.set(item, "progress", "100%")
                        self.tree.set(item, "status", "Hoàn tất")

        elif kind == "final_json":
            if isinstance(data, dict):
                self._populate_outputs_from_json(data)

        elif kind == "output_started":
            render_id = str(data)
            if self.tree.exists(render_id):
                self.tree.set(render_id, "stage", "Rendering")
                self.tree.set(render_id, "status", "Đang dựng...")
            elif self.tree.exists(f"out_{render_id}"):
                self.tree.set(f"out_{render_id}", "stage", "Rendering")
                self.tree.set(f"out_{render_id}", "status", "Đang dựng...")

        elif kind == "output_completed":
            info = data.get("data", {}) if isinstance(data, dict) else {}
            render_id = data.get("render_id", "") if isinstance(data, dict) else str(data)
            out_path = info.get("output_path", "")
            target_id = render_id if self.tree.exists(render_id) else f"out_{render_id}"
            if self.tree.exists(target_id):
                if out_path:
                    self.tree.set(target_id, "source_video", Path(out_path).name)
                    self._output_paths[target_id] = out_path
                    self._output_paths[render_id] = out_path
                self.tree.set(target_id, "stage", "Hoàn thành")
                self.tree.set(target_id, "progress", "100%")
                self.tree.set(target_id, "status", "Hoàn tất")

        elif kind == "output_skipped":
            info = data.get("data", {}) if isinstance(data, dict) else {}
            render_id = data.get("render_id", "") if isinstance(data, dict) else str(data)
            out_path = info.get("output_path", "")
            target_id = render_id if self.tree.exists(render_id) else f"out_{render_id}"
            if self.tree.exists(target_id):
                if out_path:
                    self.tree.set(target_id, "source_video", Path(out_path).name)
                    self._output_paths[target_id] = out_path
                    self._output_paths[render_id] = out_path
                self.tree.set(target_id, "stage", "Bỏ qua")
                self.tree.set(target_id, "progress", "100%")
                self.tree.set(target_id, "status", "Đã có sẵn")

        elif kind == "output_failed":
            render_id = data.get("render_id", "") if isinstance(data, dict) else str(data)
            err = data.get("error", "Lỗi dựng") if isinstance(data, dict) else ""
            target_id = render_id if self.tree.exists(render_id) else f"out_{render_id}"
            if self.tree.exists(target_id):
                self.tree.set(target_id, "stage", "Lỗi")
                self.tree.set(target_id, "status", f"Thất bại ({err[:30]})")

        elif kind == "error":
            clean_err = data.get("message", "Lỗi không xác định") if isinstance(data, dict) else str(data)
            self._log(f"LỖI: {clean_err}")
            self.banner.show(f"Lỗi: {clean_err}", level="error")

        elif kind == "gpu_status":
            self.gpu_status_var.set(str(data))

        elif kind == "discovery_progress":
            token = data.get("token") if isinstance(data, dict) else 0
            if token != self._active_discovery_token:
                return
            cur = int(data.get("current", 0))
            tot = int(data.get("total", 0))
            p_name = str(data.get("name", ""))
            source = data.get("source")
            if isinstance(source, SourceFingerprint):
                target = self.highlight_sources if self._discovery_mode == "highlight" else self.discovered_sources
                source_key = str(Path(source.path)).casefold()
                if not any(str(Path(item.path)).casefold() == source_key for item in target):
                    target.append(source)
                if self._discovery_mode == "highlight":
                    self._refresh_highlight_table()
                    self.highlight_source_var.set(f"Đang quét: {p_name} ({cur}/{tot})")
                else:
                    self._refresh_queue_table()
                    self.source_var.set(f"Đang quét: {p_name} ({cur}/{tot})")
            if tot > 0:
                self.status_var.set(f"Đang kiểm tra danh sách video ({cur}/{tot})...")
            else:
                self.status_var.set(f"Đang kiểm tra danh sách video trong {p_name}...")
            self._update_discovery_activity(
                current=cur, total=tot, name=p_name, source=source if isinstance(source, SourceFingerprint) else None,
            )

        elif kind == "discovery_completed":
            token = data.get("token") if isinstance(data, dict) else 0
            if token != self._active_discovery_token:
                return
            fps = data.get("sources", [])
            p_name = data.get("name", "")
            d_kind = data.get("kind", "")

            if self._discovery_mode == "highlight":
                if not fps:
                    self.highlight_sources.clear()
                    self._refresh_highlight_table()
                    self._set_discovery_busy(False)
                    self.highlight_source_var.set(f"{p_name}: no supported videos")
                    self.status_var.set("Không tìm thấy video nào được hỗ trợ.")
                    self._update_discovery_activity(
                        current=0, total=0, name=p_name,
                        failed="No supported videos were found.",
                    )
                    self.banner.show("No supported highlight source was found.", level="warning")
                    return
                self.highlight_sources = list(fps)
                self._refresh_highlight_table()
                self._set_discovery_busy(False)
                self.status_var.set(f"Highlight source ready: {p_name} ({len(fps)} episode(s)).")
                self._update_discovery_activity(
                    current=len(fps), total=len(fps), name=p_name, finished=True,
                )
                self.banner.show("Highlight source is ready.", level="success")
                return

            if d_kind == "file":
                if not fps:
                    self._set_discovery_busy(False)
                    self._update_discovery_activity(
                        current=0, total=0, name=p_name,
                        failed="The selected video file is unavailable.",
                    )
                    self.banner.show(f"File video không tồn tại: {p_name}", level="warning")
                    return
                self.discovered_sources = fps
                self.source_var.set(f"File: {p_name}")
                self._refresh_queue_table()
                self._set_discovery_busy(False)
                self.status_var.set(f"Đã nạp file {p_name}. Nhấn 'Start Creating Recap Videos' để bắt đầu.")
                self._update_discovery_activity(
                    current=1, total=1, name=p_name, finished=True,
                )
                self.banner.show(f"Đã chọn file: {p_name} (SINGLE_EPISODE)", level="info")
            elif d_kind == "folder":
                if not fps:
                    self.discovered_sources.clear()
                    self._refresh_queue_table()
                    self._set_discovery_busy(False)
                    self.source_var.set(f"Folder: {p_name} (0 video)")
                    self.status_var.set("Không tìm thấy video nào được hỗ trợ.")
                    self._update_discovery_activity(
                        current=0, total=0, name=p_name,
                        failed="No supported videos were found.",
                    )
                    self.banner.show(f"Không tìm thấy video nào được hỗ trợ trực tiếp trong: {p_name}", level="warning")
                    return
                self.discovered_sources = fps
                self.source_var.set(f"Folder: {p_name} ({len(fps)} video)")
                self._refresh_queue_table()
                self._set_discovery_busy(False)
                self.status_var.set(f"Đã nạp mùa phim {p_name} ({len(fps)} tập). Nhấn 'Start Creating Recap Videos' để bắt đầu.")
                self._update_discovery_activity(
                    current=len(fps), total=len(fps), name=p_name, finished=True,
                )
                self.banner.show(f"Đã tìm thấy {len(fps)} tập phim trong {p_name}. Sẵn sàng phân tích mùa phim!", level="success")

        elif kind == "discovery_cancelled":
            token = data.get("token") if isinstance(data, dict) else 0
            if token != self._active_discovery_token:
                return
            self._set_discovery_busy(False)
            self.status_var.set("Đã dừng tác vụ chọn nguồn.")
            self._update_discovery_activity(
                current=int(self._activity_snapshot.get("completed") or 0),
                total=int(self._activity_snapshot.get("total") or 0),
                name=str(self._activity_snapshot.get("current_item") or "source"),
                cancelled=True,
            )
            self._log("Đã dừng tác vụ chọn nguồn.")

        elif kind == "discovery_failed":
            token = data.get("token") if isinstance(data, dict) else 0
            if token != self._active_discovery_token:
                return
            self._set_discovery_busy(False)
            err = data.get("error", "Lỗi nạp nguồn") if isinstance(data, dict) else str(data)
            self.status_var.set(f"Lỗi: {err}")
            self._update_discovery_activity(
                current=int(self._activity_snapshot.get("completed") or 0),
                total=int(self._activity_snapshot.get("total") or 0),
                name=str(data.get("name", "source")) if isinstance(data, dict) else "source",
                failed=err,
            )
            lvl = "warning" if "không tồn tại" in err.lower() else "error"
            self.banner.show(f"{err}", level=lvl)
            self._log(f"Lỗi nạp nguồn: {err}")

        elif kind == "finished":
            status = data.get("status") if isinstance(data, dict) else ""
            finished_mode = data.get("mode", "RECAP") if isinstance(data, dict) else "RECAP"
            self._set_running_state(False)

            hwnd = self.winfo_id() if self.settings.flash_taskbar else None
            proj_name = self.current_project_name or "Dự án"

            if status == ProjectStatus.COMPLETED.value:
                msg = "Highlight publication completed!" if finished_mode == "HIGHLIGHT" else "Đã hoàn thành toàn bộ dự án video recap thành công!"
                self.status_var.set(msg)
                self.banner.show(f"Đã hoàn thành toàn bộ dự án '{proj_name}' thành công!", level="success")
                if finished_mode == "HIGHLIGHT":
                    for item in self.highlight_tree.get_children():
                        self.highlight_tree.set(item, "stage", "Published")
                        self.highlight_tree.set(item, "progress", "100%")
                        self.highlight_tree.set(item, "status", "Complete")
                if self.settings.notify_complete:
                    self.notification_service.notify_project_completed(
                        project_name=proj_name,
                        outputs_count=len(self._output_paths),
                        sound=self.settings.play_completion_sound,
                        hwnd=hwnd,
                        on_open_output=self._on_open_output_dir,
                    )
            elif status == ProjectStatus.CANCELLED.value:
                self.status_var.set("Đã dừng tiến trình tại điểm checkpoint.")
                self.banner.show("Tiến trình đã được dừng an toàn.", level="warning")
            elif status == ProjectStatus.FAILED.value:
                err_text = data.get("error", "Tiến trình bị gián đoạn") if isinstance(data, dict) else "Lỗi"
                if self._activity_snapshot.get("state") != ActivityState.FAILED.value:
                    self._set_activity_snapshot({
                        "state": ActivityState.FAILED.value, "active": False,
                        "activity_text": "Workflow failed; prior progress was preserved.",
                        "error": err_text,
                    }, merge=True)
                self.status_var.set(f"Lỗi: {err_text}")
                self.banner.show(f"Lỗi: {err_text}", level="error")
                if self.settings.notify_error:
                    self.notification_service.notify_project_failed(
                        project_name=proj_name,
                        error_msg=err_text,
                        sound=self.settings.play_completion_sound,
                        hwnd=hwnd,
                    )

            self._check_restart_projects()

    def _populate_outputs_from_json(self, final_json: Dict[str, Any]) -> None:
        outputs = final_json.get("outputs", [])
        for i, out_def in enumerate(outputs, start=1):
            rid = out_def.get("render_id", f"render-{i}")
            title = out_def.get("title", f"Output {i}")
            out_name = f"{title}.mp4" if title else f"output_{i}.mp4"
            if not self.tree.exists(rid):
                self.tree.insert(
                    "",
                    "end",
                    iid=rid,
                    values=(f"Output {i}", out_name, "Chờ dựng", "0%", title),
                )
            else:
                self.tree.set(rid, "stage", "Chờ dựng")
                self.tree.set(rid, "status", title)

    def _on_double_click_tree(self, event: Any) -> None:
        item = self.tree.selection()
        if not item:
            return
        row_id = item[0]
        out_path = self._output_paths.get(row_id)
        if out_path:
            p = Path(out_path)
            if p.is_file():
                if sys.platform == "win32":
                    os.startfile(str(p))
                else:
                    subprocess.Popen(["xdg-open", str(p)])

    def _set_running_state(self, is_running: bool) -> None:
        for mode, page in (("RECAP", self.recap_page), ("HIGHLIGHT", self.highlight_page)):
            page.start_button.config(state="disabled" if is_running else "normal")
            page.stop_button.config(state="normal" if is_running else "disabled")
            page.resume_button.config(
                state="disabled" if is_running else ("normal" if self._current_project_by_mode[mode] else "disabled")
            )
            page.select_file_button.config(state="disabled" if is_running else "normal")
            page.select_folder_button.config(state="disabled" if is_running else "normal")
            page.clear_button.config(state="disabled" if is_running else "normal")
            if page.use_json_button is not None:
                page.use_json_button.config(state="disabled" if is_running else "normal")

    def _log(self, message: str) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        clean_text = f"[{now_str}] {message}\n"
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, clean_text)
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _on_banner_dismissed(self) -> None:
        pass

    def _on_close(self) -> None:
        if self._shutting_down:
            return
        if self.worker.is_running:
            if not messagebox.askyesno(
                "Đang xử lý",
                "Đang có tiến trình hoạt động. Bạn có chắc chắn muốn dừng và thoát?\n"
                "Dự án sẽ được lưu điểm dừng (checkpoint) an toàn để có thể tiếp tục sau.",
                parent=self,
            ):
                return
            self._log("Đang hủy tác vụ an toàn trước khi đóng ứng dụng...")
            self.worker.cancel()
            if self.discovery_worker.is_running:
                self.discovery_worker.cancel()
            self._close_job = safe_after(self, 1500, self.destroy)
        else:
            if self.discovery_worker.is_running:
                self.discovery_worker.cancel()
            self.destroy()

    def cancel_owned_after(self) -> None:
        """Cancel pending after() callbacks owned by this window."""
        for job_attr in ("_poll_job", "_gpu_detect_job", "_close_job", "_activity_timer_job", "_dots_job"):
            job = getattr(self, job_attr, None)
            if job:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                setattr(self, job_attr, None)

        if hasattr(self, "banner") and getattr(self.banner, "_auto_dismiss_job", None):
            try:
                self.banner.after_cancel(self.banner._auto_dismiss_job)
            except Exception:
                pass
            self.banner._auto_dismiss_job = None

    def destroy(self) -> None:
        """Safe window destruction with closed guard, after cancellation, and worker stopping."""
        if getattr(self, "_is_closed", False):
            return
        self._shutting_down = True
        self._is_closed = True

        self.cancel_owned_after()

        if hasattr(self, "discovery_worker") and self.discovery_worker.is_running:
            self.discovery_worker.cancel()
        if hasattr(self, "worker") and self.worker.is_running:
            self.worker.cancel()

        try:
            super().destroy()
        except Exception:
            pass


def run_app() -> None:
    """Single-click application entrypoint."""
    if "--selfcheck" in sys.argv:
        from toolrecap_v4.selfcheck import run_selfcheck

        sys.exit(run_selfcheck())
    if "--update-helper" in sys.argv:
        from toolrecap_v4.updater.helper import main
        sys.argv.remove("--update-helper")
        main()
        return
    if "--update-handshake" in sys.argv:
        print("TOOLRECAP_V5_HANDSHAKE_OK")
        print("TOOLRECAP_V4_HANDSHAKE_OK")
        sys.exit(0)
    app = MainWindow()
    app.mainloop()
