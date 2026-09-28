"""Embedded settings page for ToolRecap V4.

Invariants:
- Exactly 4 sidebar panes: Recap, AI Gateway, Voice, Render and Output.
- Geometry: 920x680, minsize: 840x600.
- DPAPI secret storage: masked credentials, never saved to settings.json.
- AI Gateway: provider-neutral Scanner, Vision, and unified Finalizer controls.
- Voice: external VoiceStudio modes (auto/local/remote) instead of local runtime repair.
- Recap / Prompt: exact user prompt, no editorial policies, reset clears to empty string.
- Voice preview and render use the exact same VoiceStudioAdapter.
- Ducking amount control enabled only when auto-duck is checked.
- All fields effective, persist to settings and DPAPI, and validate cleanly.
- Preserves all V3 settings: notifications, updater, VoiceStudio modes, audio mix.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import replace
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional
import urllib.parse

from toolrecap_v4.__version__ import __version__
from toolrecap_v4.gateway import GatewayClient, sanitize_message
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.secrets import DPAPISecretStore
from toolrecap_v4.settings import AppSettings, COMMENTARY_SPEED_VALUES, GatewayFormValues, GatewaySettingsController, REASONING_VALUES, SettingsManager
from toolrecap_v4.downstream import apply_commentary_speed
from toolrecap_v4.endpoints import normalize_http_endpoint
from toolrecap_v4.ui.notifications import WindowsNotificationService
from toolrecap_v4.updater import UpdateCheckResult, UpdateManager
from toolrecap_v4.voice_studio import (
    DEFAULT_V2_VOICE_ID,
    V2_VOICE_PRESETS,
    VoiceStudioAdapter,
    get_v2_voice_presets,
    get_voice_display_name,
    resolve_voice_id,
)

logger = logging.getLogger(__name__)

MASKED_SECRET_PLACEHOLDER = "••••••••••••"


def validate_url_no_credentials(url: str, field_name: str) -> str:
    """Ensure URL is valid HTTP/HTTPS and contains no embedded credentials."""
    return normalize_http_endpoint(url, field_name, allow_empty=True)


class _NotebookCompat:
    """Compatibility shim exposing notebook index/select for tests and legacy callers."""

    def __init__(self, dialog: SettingsDialog) -> None:
        self._dialog = dialog

    def select(self, tab_id: Any = None) -> Any:
        if tab_id is None:
            cur = self._dialog._current_pane
            if cur == "Recap":
                return "tab_prompt"
            if cur == "Highlight":
                return "tab_highlight"
            if cur == "AI Gateway":
                return "tab_gw"
            if cur == "Voice":
                sub = getattr(self._dialog, "voice_notebook", None)
                sub_idx = sub.index(sub.select()) if sub else 0
                return "tab_voice" if sub_idx == 0 else "tab_audio"
            if cur == "Render and Output":
                sub = getattr(self._dialog, "render_notebook", None)
                sub_idx = sub.index(sub.select()) if sub else 0
                if sub_idx == 1:
                    return "tab_notify"
                if sub_idx == 2:
                    return "tab_update"
                return "tab_render"
            return "tab_prompt"
        self._dialog._select_tab(tab_id)

    def index(self, tab_id: Any) -> int:
        if tab_id == "end":
            return 8
        if tab_id in (0, "tab_gw"):
            return 0
        if tab_id in (1, "tab_prompt", "tab_recap"):
            return 1
        if tab_id in (7, "tab_highlight"):
            return 7
        if tab_id in (2, "tab_voice"):
            return 2
        if tab_id in (3, "tab_audio"):
            return 3
        if tab_id in (4, "tab_render"):
            return 4
        if tab_id in (5, "tab_notify", "tab_notifications"):
            return 5
        if tab_id in (6, "tab_update", "tab_updates"):
            return 6
        if isinstance(tab_id, int):
            return tab_id
        cur_sel = self.select()
        if tab_id == cur_sel:
            mapping = {
                "tab_gw": 0,
                "tab_prompt": 1,
                "tab_voice": 2,
                "tab_audio": 3,
                "tab_render": 4,
                "tab_notify": 5,
                "tab_update": 6,
                "tab_highlight": 7,
            }
            return mapping.get(cur_sel, 0)
        return 0


class SettingsDialog(ttk.Frame):
    """The single authoritative Settings UI, embedded in the main window.

    The historical class name is retained for import compatibility.  This is a
    normal ``ttk.Frame``: it never creates an OS window, modal grab, or taskbar
    entry.
    """

    def __init__(
        self,
        parent: tk.Misc,
        settings_manager: Optional[SettingsManager] = None,
        persistence: Optional[ProjectPersistence] = None,
        on_saved: Optional[Callable[[AppSettings], None]] = None,
        initial_tab: Optional[int | str] = None,
        *,
        on_save: Optional[Callable[[AppSettings], None]] = None,
        settings: Optional[AppSettings] = None,
        store: Any = None,
        on_closed: Optional[Callable[["SettingsDialog"], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.parent = parent
        self.persistence = persistence or ProjectPersistence()
        self.settings_manager = settings_manager or SettingsManager(persistence=self.persistence)
        self.secret_store = DPAPISecretStore(storage_root=self.persistence.root)
        self.gateway_controller = GatewaySettingsController(self.settings_manager, self.secret_store)
        self.on_saved = on_saved or on_save
        self.on_closed = on_closed
        self._closed = False
        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._ui_poll_job: Optional[str] = None
        self.settings = settings or self.settings_manager.load()

        self._panes: Dict[str, ttk.Frame] = {}
        self._nav_buttons: Dict[str, ttk.Button] = {}
        self._current_pane = "Recap"
        self.notebook = _NotebookCompat(self)

        self._init_variables()
        self._build_ui()
        self._load_values()

        if initial_tab is not None:
            self._select_tab(initial_tab)
        else:
            self._show_pane("Recap")

        self._ui_poll_job = self.after(25, self._poll_ui_queue)

    def activate(self) -> bool:
        """Raise this embedded page without changing OS-level window state."""
        if self._closed or not self.winfo_exists():
            return False
        try:
            self.tkraise()
            self.focus_set()
            return True
        except tk.TclError:
            return False

    def _post_ui(self, callback: Callable[[], None]) -> None:
        """Marshal worker results to Tk's thread; synchronous tests remain deterministic."""
        if threading.current_thread() is threading.main_thread():
            callback()
        else:
            self._ui_queue.put(callback)

    def _poll_ui_queue(self) -> None:
        if self._closed:
            return
        try:
            while True:
                self._ui_queue.get_nowait()()
        except queue.Empty:
            pass
        except tk.TclError:
            return
        finally:
            if not self._closed and self.winfo_exists():
                self._ui_poll_job = self.after(25, self._poll_ui_queue)

    # -------------------------------------------------------------------------
    # VARIABLES & STATE
    # -------------------------------------------------------------------------
    def _init_variables(self) -> None:
        val = self.settings

        # Recap variables
        self.recap_language_var = tk.StringVar(value=getattr(val, "recap_language", "en-US") or "en-US")
        self.recap_mode_var = tk.StringVar(value=getattr(val, "recap_mode", "MAIN_STORIES") or "MAIN_STORIES")
        self.content_type_var = tk.StringVar(value=getattr(val, "content_type", "US_TV_SHOW") or "US_TV_SHOW")
        self.rights_var = tk.StringVar(value=getattr(val, "source_rights_status", "UNVERIFIED") or "UNVERIFIED")
        self.var_prompt_status = tk.StringVar()
        self.highlight_prompt_var = tk.StringVar(value=getattr(val, "highlight_prompt", ""))

        # Existing model fields are retained; V5 uses Finalizer as its primary
        # single analysis model and Scanner only as a compatibility fallback.
        self.gateway_enabled_var = tk.BooleanVar(value=True)
        self.var_gw_endpoint = tk.StringVar(value=val.gateway_endpoint)
        self.var_gw_key = tk.StringVar()
        self.var_gw_key_visible = tk.BooleanVar(value=False)

        sub_model_val = getattr(val, "scanner_model", "") or getattr(val, "gateway_sub_model", "") or getattr(val, "gateway_model", "")
        self.var_gw_sub_model = tk.StringVar(value=sub_model_val)
        self.var_gw_sub_reasoning = tk.StringVar(value=getattr(val, "scanner_reasoning", "") or getattr(val, "gateway_sub_reasoning", "") or "")
        self.var_scanner_parallelism = tk.IntVar(value=val.scanner_parallelism)
        self.var_scanner_chunk_ms = tk.IntVar(value=val.scanner_chunk_duration_ms)
        self.var_vision_model = tk.StringVar(value=val.vision_model)
        self.var_vision_reasoning = tk.StringVar(value=val.vision_reasoning)

        prime_model_val, prime_reasoning_val, self._finalizer_values_unified = val.finalizer_ui_values()
        self.var_gw_prime_model = tk.StringVar(value=prime_model_val)
        self.var_gw_prime_reasoning = tk.StringVar(value=prime_reasoning_val)

        self.var_gw_model = self.var_gw_sub_model
        self.var_gw_thinking = tk.BooleanVar(value=bool(val.gateway_thinking or val.gateway_sub_reasoning or val.gateway_prime_reasoning))

        self.sub_status_var = tk.StringVar(value="Model dự phòng chưa được kiểm tra.")
        self.prime_status_var = tk.StringVar(value="Model phân tích V5 chưa được kiểm tra." if self._finalizer_values_unified else "Các cấu hình model cũ đang khác nhau; V5 sẽ ưu tiên model phân tích bên dưới.")
        self._gateway_test_running = {"scanner": False, "finalizer": False}
        self.ai_status_var = self.sub_status_var

        # Dual model compatibility aliases
        self.sub_model_var = self.var_gw_sub_model
        self.sub_reasoning_var = self.var_gw_sub_reasoning
        self.prime_model_var = self.var_gw_prime_model
        self.prime_reasoning_var = self.var_gw_prime_reasoning

        # Voice & VoiceStudio variables
        self.var_voice_mode = tk.StringVar(value=val.voice_mode)
        self.var_voice_local = tk.StringVar(value=val.voice_local_url)
        self.var_voice_remote = tk.StringVar(value=val.voice_remote_url)
        self.var_voice_key = tk.StringVar()
        self.var_voice_key_visible = tk.BooleanVar(value=False)
        raw_voice = val.voice_id or DEFAULT_V2_VOICE_ID
        self.var_voice_id = tk.StringVar(value=get_voice_display_name(raw_voice) if raw_voice else "")
        self.var_voice_lang = tk.StringVar(value=val.voice_language)
        self.var_voice_style = tk.StringVar(value=val.voice_style)
        self.var_commentary_speed = tk.StringVar(value=f"{val.commentary_reading_speed:.2f}x")
        self.voice_status_var = tk.StringVar(value="")

        # Audio Mix variables
        self.var_orig_db = tk.DoubleVar(value=val.original_audio_db)
        self.var_comm_db = tk.DoubleVar(value=val.commentary_audio_db)
        self.var_auto_duck = tk.BooleanVar(value=val.auto_duck)
        self.var_duck_db = tk.DoubleVar(value=val.ducking_amount_db)
        self.var_target_lufs = tk.DoubleVar(value=val.target_loudness_lufs)
        self.var_true_peak = tk.DoubleVar(value=val.true_peak_db)
        self.var_highlight_normalize = tk.BooleanVar(value=val.highlight_normalize_loudness)

        # Render variables
        self.var_out_dir = tk.StringVar(value=val.output_dir)
        self.var_use_gpu = tk.BooleanVar(value=val.use_gpu)
        self.var_quality = tk.StringVar(value=val.quality)
        self.var_codec = tk.StringVar(value=val.video_codec)
        self.var_canvas_w = tk.IntVar(value=val.canvas_width)
        self.var_canvas_h = tk.IntVar(value=val.canvas_height)
        self.var_canvas_fps = tk.DoubleVar(value=val.canvas_fps)
        self.var_canvas_auto = tk.BooleanVar(value=True)
        self.var_burn_subs = tk.BooleanVar(value=val.burn_subtitles)
        self.var_out_format = tk.StringVar(value=val.output_format)

        # Notifications variables
        self.var_notify_complete = tk.BooleanVar(value=val.notify_complete)
        self.var_notify_error = tk.BooleanVar(value=val.notify_error)
        self.var_play_sound = tk.BooleanVar(value=val.play_completion_sound)
        self.var_flash_taskbar = tk.BooleanVar(value=val.flash_taskbar)
        self.var_notify_update = tk.BooleanVar(value=val.notify_update)

        # Updater variables
        self.var_update_repo = tk.StringVar(value=val.update_repo)
        self.update_manager = UpdateManager(
            storage_root=self.persistence.root,
            activity_probe=lambda: bool(getattr(getattr(self.parent, "worker", None), "is_running", False)),
        )
        self._last_check_result: Optional[UpdateCheckResult] = None
        self._staged_update_path: Optional[Path] = None

        # DPAPI tracking
        self._had_gw_key = False
        self._had_vs_key = False

        # V2 compatibility aliases
        self.endpoint_var = self.var_gw_endpoint
        self.key_var = self.var_gw_key
        self.show_key_var = self.var_gw_key_visible
        self.orig_audio_var = self.var_orig_db
        self.commentary_var = self.var_comm_db
        self.auto_duck_var = self.var_auto_duck
        self.ducking_var = self.var_duck_db
        self.target_loudness_var = self.var_target_lufs
        self.true_peak_var = self.var_true_peak
        self.quality_var = self.var_quality
        self.gpu_var = self.var_use_gpu
        self.burn_var = self.var_burn_subs
        self.output_var = self.var_out_dir
        self.voice_var = self.var_voice_id
        self.voice_style_var = self.var_voice_style

    def _load_values(self) -> None:
        """Load values including DPAPI masked status into UI."""
        gw_key = self.secret_store.get_secret("gateway_api_key")
        if gw_key:
            self.var_gw_key.set(MASKED_SECRET_PLACEHOLDER)
            self._had_gw_key = True
        else:
            self.var_gw_key.set("")
            self._had_gw_key = False

        vs_key = self.secret_store.get_secret("voice_remote_api_key")
        if vs_key:
            self.var_voice_key.set(MASKED_SECRET_PLACEHOLDER)
            self._had_vs_key = True
        else:
            self.var_voice_key.set("")
            self._had_vs_key = False

        raw_voice = self.settings.voice_id or DEFAULT_V2_VOICE_ID
        self.var_voice_id.set(get_voice_display_name(raw_voice) if raw_voice else "")

        self.txt_prompt.delete("1.0", tk.END)
        self.txt_prompt.insert("1.0", self.settings.prompt)
        self._update_prompt_status()
        self.highlight_prompt_var.set(getattr(self.settings, "highlight_prompt", ""))
        if hasattr(self, "txt_highlight_prompt"):
            self.txt_highlight_prompt.delete("1.0", tk.END)
            self.txt_highlight_prompt.insert("1.0", self.highlight_prompt_var.get())

    def _reload_non_prompt_values(self) -> None:
        """Restore widgets from the last persisted settings snapshot."""
        val = self.settings
        assignments = (
            (self.recap_language_var, val.recap_language), (self.recap_mode_var, val.recap_mode),
            (self.content_type_var, val.content_type), (self.rights_var, val.source_rights_status),
            (self.var_gw_endpoint, val.gateway_endpoint), (self.var_gw_sub_model, val.scanner_model),
            (self.var_gw_sub_reasoning, val.scanner_reasoning), (self.var_scanner_parallelism, val.scanner_parallelism),
            (self.var_scanner_chunk_ms, val.scanner_chunk_duration_ms), (self.var_vision_model, val.vision_model),
            (self.var_vision_reasoning, val.vision_reasoning), (self.var_voice_mode, val.voice_mode),
            (self.var_voice_local, val.voice_local_url), (self.var_voice_remote, val.voice_remote_url),
            (self.var_voice_lang, val.voice_language), (self.var_voice_style, val.voice_style),
            (self.var_commentary_speed, f"{val.commentary_reading_speed:.2f}x"),
            (self.var_orig_db, val.original_audio_db), (self.var_comm_db, val.commentary_audio_db),
            (self.var_auto_duck, val.auto_duck), (self.var_duck_db, val.ducking_amount_db),
            (self.var_target_lufs, val.target_loudness_lufs), (self.var_true_peak, val.true_peak_db),
            (self.var_highlight_normalize, val.highlight_normalize_loudness),
            (self.var_out_dir, val.output_dir), (self.var_use_gpu, val.use_gpu),
            (self.var_quality, val.quality), (self.var_codec, val.video_codec),
            (self.var_canvas_w, val.canvas_width), (self.var_canvas_h, val.canvas_height),
            (self.var_canvas_fps, val.canvas_fps), (self.var_burn_subs, val.burn_subtitles),
            (self.var_out_format, val.output_format), (self.var_update_repo, val.update_repo),
        )
        for variable, value in assignments:
            variable.set(value)
        prime_model, prime_reasoning, _ = val.finalizer_ui_values()
        self.var_gw_prime_model.set(prime_model)
        self.var_gw_prime_reasoning.set(prime_reasoning)
        self.var_voice_id.set(get_voice_display_name(val.voice_id or DEFAULT_V2_VOICE_ID))

    # -------------------------------------------------------------------------
    # MAIN UI SHELL
    # -------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        root = ttk.Frame(canvas, padding=12)
        window_id = canvas.create_window((0, 0), window=root, anchor="nw")
        root.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        self.settings_canvas = canvas
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # Settings-local navigation remains embedded in the main application.
        sidebar = ttk.Frame(root, padding=(0, 4, 12, 4))
        sidebar.grid(row=0, column=0, sticky="ns")

        for name in ("Recap", "Highlight", "AI Gateway", "Voice", "Render and Output"):
            btn = ttk.Button(
                sidebar,
                text=name,
                width=20,
                command=lambda target=name: self._show_pane(target),
            )
            btn.pack(fill="x", pady=3)
            self._nav_buttons[name] = btn

        host = ttk.Frame(root, padding=(16, 8))
        host.grid(row=0, column=1, sticky="nsew")
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)

        self._panes["Recap"] = self._build_recap(host)
        self._panes["Highlight"] = self._build_highlight(host)
        self._panes["AI Gateway"] = self._build_ai(host)
        self._panes["Voice"] = self._build_voice(host)
        self._panes["Render and Output"] = self._build_render(host)

        for pane in self._panes.values():
            pane.grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(self, padding=(12, 8))
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.lbl_status = ttk.Label(bottom, text="", foreground="#1E40AF", font=("Segoe UI", 9))
        self.lbl_status.pack(side="left", padx=4)

        ttk.Button(bottom, text="Hoàn tác", command=self._on_close).pack(side="right")
        self.btn_save = ttk.Button(
            bottom,
            text="Lưu cài đặt",
            style="Primary.TButton",
            command=self._on_save,
        )
        self.btn_save.pack(side="right", padx=(0, 8))

    def _show_pane(self, name: str) -> None:
        if name in self._panes:
            self._current_pane = name
            self._panes[name].tkraise()
            for key, button in self._nav_buttons.items():
                if key == name:
                    button.state(["disabled"])
                    button.configure(state="disabled")
                else:
                    button.state(["!disabled"])
                    button.configure(state="normal")

    def _select_subtab(self, pane_name: str, subtab: int | str) -> None:
        if pane_name == "Voice" and hasattr(self, "voice_notebook"):
            try:
                self.voice_notebook.select(subtab)
            except Exception:
                pass
        elif pane_name == "Render and Output" and hasattr(self, "render_notebook"):
            try:
                self.render_notebook.select(subtab)
            except Exception:
                pass

    def _select_tab(self, tab: int | str) -> None:
        """Select a pane or subtab by name or index for backward compatibility."""
        if isinstance(tab, int):
            index_map = {
                0: ("AI Gateway", None),
                1: ("Recap", None),
                2: ("Voice", 0),
                3: ("Voice", 1),
                4: ("Render and Output", 0),
                5: ("Render and Output", 1),
                6: ("Render and Output", 2),
            }
            target = index_map.get(tab)
            if target:
                self._show_pane(target[0])
                if target[1] is not None:
                    self._select_subtab(target[0], target[1])
        elif isinstance(tab, str):
            s = tab.strip().lower()
            if s in ("recap", "prompt"):
                self._show_pane("Recap")
            elif s in ("highlight", "highlights", "tab_highlight"):
                self._show_pane("Highlight")
            elif s in ("gateway", "gw", "ai", "ai gateway"):
                self._show_pane("AI Gateway")
            elif s in ("voice", "voicestudio"):
                self._show_pane("Voice")
                self._select_subtab("Voice", 0)
            elif s in ("audio", "audiomix", "audio mix"):
                self._show_pane("Voice")
                self._select_subtab("Voice", 1)
            elif s in ("render", "output"):
                self._show_pane("Render and Output")
                self._select_subtab("Render and Output", 0)
            elif s in ("notify", "notifications"):
                self._show_pane("Render and Output")
                self._select_subtab("Render and Output", 1)
            elif s in ("update", "updates"):
                self._show_pane("Render and Output")
                self._select_subtab("Render and Output", 2)

    # -------------------------------------------------------------------------
    # PANE 1: RECAP & PROMPT
    # -------------------------------------------------------------------------
    def _build_recap(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Cấu hình Kịch bản & Recap", font=("Segoe UI Semibold", 14)).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 14)
        )

        # Recap Prompt (Exact prompt, no editorial policies)
        prompt_box = ttk.LabelFrame(frame, text="Kịch bản biên tập (Recap Prompt)", padding=8)
        prompt_box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(0, 4))
        prompt_box.columnconfigure(0, weight=1)
        prompt_box.rowconfigure(1, weight=1)

        prompt_toolbar = ttk.Frame(prompt_box)
        prompt_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(
            prompt_toolbar,
            text="Hướng dẫn kịch bản gửi tới AI Gateway:",
            font=("Segoe UI", 9),
            foreground="#475569",
        ).pack(side="left")

        self.btn_clear_prompt = ttk.Button(prompt_toolbar, text="Xóa trống", command=self._clear_prompt)
        self.btn_clear_prompt.pack(side="right", padx=(4, 0))

        self.btn_reset_prompt = ttk.Button(
            prompt_toolbar,
            text="Đặt lại mặc định (Trống)",
            command=self._reset_prompt,
        )
        self.btn_reset_prompt.pack(side="right")
        self.reload_prompt_btn = self.btn_reset_prompt

        text_frame = ttk.Frame(prompt_box)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.txt_prompt = tk.Text(text_frame, wrap="word", height=12, font=("Segoe UI", 9))
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.txt_prompt.yview)
        self.txt_prompt.configure(yscrollcommand=scroll.set)
        self.txt_prompt.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.txt_prompt.bind("<KeyRelease>", lambda e: self._update_prompt_status())

        self.lbl_prompt_warning = ttk.Label(
            prompt_box,
            textvariable=self.var_prompt_status,
            font=("Segoe UI", 9, "bold"),
        )
        self.lbl_prompt_warning.grid(row=2, column=0, sticky="w", pady=(4, 0))

        return frame

    def _on_recap_language_changed(self, event: tk.Event | None = None) -> None:
        lang = self.recap_language_var.get()
        self.var_voice_lang.set(lang)

    def _update_prompt_status(self) -> None:
        content = self.txt_prompt.get("1.0", tk.END).strip()
        if not content:
            self.var_prompt_status.set("⚠ Kịch bản đang trống! Vui lòng nhập hướng dẫn kịch bản.")
            self.lbl_prompt_warning.config(foreground="#D97706")
        else:
            self.var_prompt_status.set(f"Độ dài: {len(content)} ký tự")
            self.lbl_prompt_warning.config(foreground="#16A34A")

    def _reset_prompt(self) -> None:
        """Reset prompt to default empty string; strictly no invented editorial prompt."""
        self.txt_prompt.delete("1.0", tk.END)
        self._update_prompt_status()

    def _reset_prompt_template(self) -> None:
        self._reset_prompt()

    def _clear_prompt(self) -> None:
        self.txt_prompt.delete("1.0", tk.END)
        self._update_prompt_status()

    def _build_highlight(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        ttk.Label(frame, text="Highlight", font=("Segoe UI Semibold", 14)).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(
            frame,
            text=("Highlight Prompt controls editorial scene discovery only. "
                  "ToolRecap's versioned schema and validation rules remain authoritative."),
            wraplength=690,
            foreground="#475569",
        ).grid(row=1, column=0, sticky="ew", pady=(0, 8))
        text_frame = ttk.Frame(frame)
        text_frame.grid(row=2, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.txt_highlight_prompt = tk.Text(text_frame, wrap="word", height=18, font=("Segoe UI", 9))
        self.txt_highlight_prompt.insert("1.0", self.highlight_prompt_var.get())
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.txt_highlight_prompt.yview)
        self.txt_highlight_prompt.configure(yscrollcommand=scroll.set)
        self.txt_highlight_prompt.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        return frame

    # -------------------------------------------------------------------------
    # PANE 2: AI GATEWAY (V5 single analysis plus legacy-compatible fields)
    # -------------------------------------------------------------------------
    def _build_ai(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="AI Gateway", font=("Segoe UI Semibold", 14)).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        chk_gw = ttk.Checkbutton(
            frame,
            text="V5 gửi một yêu cầu phân tích theo prompt và nhận Render JSON hoàn chỉnh.",
            variable=self.gateway_enabled_var,
        )
        chk_gw.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))

        # API Endpoint
        ttk.Label(frame, text="API endpoint:").grid(row=2, column=0, sticky="w", pady=4)
        self.endpoint_entry = ttk.Entry(frame, textvariable=self.var_gw_endpoint)
        self.endpoint_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=4)

        # API Key (DPAPI Masked)
        ttk.Label(frame, text="API key:").grid(row=3, column=0, sticky="w", pady=4)
        self.key_entry = ttk.Entry(frame, textvariable=self.var_gw_key, show="●")
        self.key_entry.grid(row=3, column=1, sticky="ew", padx=(10, 6), pady=4)
        self.show_key_check = ttk.Checkbutton(
            frame, text="Hiện", variable=self.var_gw_key_visible, command=self._toggle_gw_key
        )
        self.show_key_check.grid(row=3, column=2, sticky="w")

        ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 6))
        ttk.Label(frame, text="Model dự phòng", font=("Segoe UI Semibold", 10)).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(0, 4)
        )

        ttk.Label(frame, text="Model:").grid(row=6, column=0, sticky="w", pady=3)
        self.sub_model_combo = ttk.Combobox(
            frame,
            textvariable=self.var_gw_sub_model,
            values=(),
        )
        self.sub_model_combo.grid(row=6, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)
        self.sub_model_entry = self.sub_model_combo
        self.model_combo = self.sub_model_combo

        ttk.Label(frame, text="Reasoning:").grid(row=7, column=0, sticky="w", pady=3)
        self.sub_reasoning_combo = ttk.Combobox(
            frame,
            textvariable=self.var_gw_sub_reasoning,
            values=REASONING_VALUES,
            state="readonly",
        )
        self.sub_reasoning_combo.grid(row=7, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)

        ttk.Label(frame, text="Parallelism (1–32):").grid(row=8, column=0, sticky="w", pady=3)
        self.scanner_parallelism_spin = ttk.Spinbox(frame, from_=1, to=32, textvariable=self.var_scanner_parallelism)
        self.scanner_parallelism_spin.grid(row=8, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)
        ttk.Label(frame, text="Chunk duration (ms):").grid(row=9, column=0, sticky="w", pady=3)
        self.scanner_chunk_spin = ttk.Spinbox(frame, from_=1000, to=3600000, increment=1000, textvariable=self.var_scanner_chunk_ms)
        self.scanner_chunk_spin.grid(row=9, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)

        ttk.Label(frame, text="Vision model (tương thích V4):").grid(row=10, column=0, sticky="w", pady=3)
        self.vision_model_entry = ttk.Entry(frame, textvariable=self.var_vision_model)
        self.vision_model_entry.grid(row=10, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)
        ttk.Label(frame, text="Vision reasoning:").grid(row=11, column=0, sticky="w", pady=3)
        self.vision_reasoning_combo = ttk.Combobox(frame, textvariable=self.var_vision_reasoning, values=REASONING_VALUES, state="readonly")
        self.vision_reasoning_combo.grid(row=11, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)

        sub_btn_box = ttk.Frame(frame)
        sub_btn_box.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        self.btn_test_sub = ttk.Button(
            sub_btn_box,
            text="Test model dự phòng",
            command=self._test_sub_connection,
        )
        self.btn_test_sub.pack(side="left")
        self.btn_test_gw = self.btn_test_sub

        self.lbl_sub_test_result = ttk.Label(
            frame,
            textvariable=self.sub_status_var,
            foreground="#075fc9",
            wraplength=600,
            font=("Segoe UI", 9),
        )
        self.lbl_sub_test_result.grid(row=13, column=0, columnspan=3, sticky="w", pady=(2, 6))
        self.lbl_gw_test_result = self.lbl_sub_test_result
        self.ai_status_lbl = self.lbl_sub_test_result

        ttk.Separator(frame, orient="horizontal").grid(row=14, column=0, columnspan=3, sticky="ew", pady=(6, 6))
        ttk.Label(frame, text="Model phân tích V5", font=("Segoe UI Semibold", 10)).grid(
            row=15, column=0, columnspan=3, sticky="w", pady=(0, 4)
        )

        ttk.Label(frame, text="Model:").grid(row=16, column=0, sticky="w", pady=3)
        self.prime_model_combo = ttk.Combobox(
            frame,
            textvariable=self.var_gw_prime_model,
            values=(),
        )
        self.prime_model_combo.grid(row=16, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)
        self.prime_model_entry = self.prime_model_combo

        ttk.Label(frame, text="Reasoning:").grid(row=17, column=0, sticky="w", pady=3)
        self.prime_reasoning_combo = ttk.Combobox(
            frame,
            textvariable=self.var_gw_prime_reasoning,
            values=REASONING_VALUES,
            state="readonly",
        )
        self.prime_reasoning_combo.grid(row=17, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)

        prime_btn_box = ttk.Frame(frame)
        prime_btn_box.grid(row=18, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        self.btn_test_prime = ttk.Button(
            prime_btn_box,
            text="Test model phân tích V5",
            command=self._test_prime_connection,
        )
        self.btn_test_prime.pack(side="left")

        self.lbl_prime_test_result = ttk.Label(
            frame,
            textvariable=self.prime_status_var,
            foreground="#075fc9",
            wraplength=600,
            font=("Segoe UI", 9),
        )
        self.lbl_prime_test_result.grid(row=19, column=0, columnspan=3, sticky="w", pady=(2, 6))

        self.chk_thinking = ttk.Checkbutton(
            frame,
            text="Thinking do Gateway quản lý",
            variable=self.var_gw_thinking,
            state="disabled",
        )

        note = ttk.Label(
            frame,
            text="* Lưu ý: API key được bảo vệ an toàn bằng mã hóa Windows DPAPI, không lưu văn bản thường trong settings.json.",
            font=("Segoe UI", 8),
            foreground="#6b7280",
            wraplength=600,
        )
        note.grid(row=20, column=0, columnspan=3, sticky="w", pady=(8, 0))

        return frame

    def _toggle_gw_key(self) -> None:
        if hasattr(self, "key_entry") and self.key_entry is not None:
            self.key_entry.configure(show="" if self.var_gw_key_visible.get() else "●")

    def _gateway_form_values(self) -> GatewayFormValues:
        return GatewayFormValues(
            endpoint=self.var_gw_endpoint.get(), api_key=self.var_gw_key.get(),
            scanner_model=self.var_gw_sub_model.get(), scanner_reasoning=self.var_gw_sub_reasoning.get(),
            scanner_parallelism=int(self.var_scanner_parallelism.get()), scanner_chunk_duration_ms=int(self.var_scanner_chunk_ms.get()),
            vision_model=self.var_vision_model.get(), vision_reasoning=self.var_vision_reasoning.get(),
            finalizer_model=self.var_gw_prime_model.get(), finalizer_reasoning=self.var_gw_prime_reasoning.get(),
        )

    def _run_gateway_test(self, role: str) -> None:
        if self._gateway_test_running[role]:
            return
        status = self.sub_status_var if role == "scanner" else self.prime_status_var
        label = self.lbl_sub_test_result if role == "scanner" else self.lbl_prime_test_result
        button = self.btn_test_sub if role == "scanner" else self.btn_test_prime
        try:
            form = self._gateway_form_values()
            GatewaySettingsController.validate(form)
        except Exception as exc:
            status.set(f"✗ Cấu hình không hợp lệ: {str(exc)[:160]}")
            label.config(foreground="#DC2626")
            return
        self._gateway_test_running[role] = True
        button.config(state="disabled")
        status.set("Đang kiểm tra kết nối…")
        label.config(foreground="#2563EB")
        results: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)

        def _poll() -> None:
            try:
                result = results.get_nowait()
            except queue.Empty:
                if self.winfo_exists():
                    self.after(25, _poll)
                return
            self._gateway_test_running[role] = False
            button.config(state="normal")
            status.set(("✓ " if result["ok"] else "✗ ") + result["message"])
            label.config(foreground="#16A34A" if result["ok"] else "#DC2626")

        def _work() -> None:
            result = self.gateway_controller.test(form, role, masked_placeholder=MASKED_SECRET_PLACEHOLDER)
            results.put(result)
        self.after(25, _poll)
        threading.Thread(target=_work, daemon=True).start()

    def _test_sub_connection(self) -> None:
        self._run_gateway_test("scanner")

    def _test_prime_connection(self) -> None:
        self._run_gateway_test("finalizer")

    def _test_gateway_connection(self) -> None:
        self._test_sub_connection()
        self._test_prime_connection()

    # -------------------------------------------------------------------------
    # PANE 3: VOICE & AUDIO MIX (External VoiceStudio + Audio Mix)
    # -------------------------------------------------------------------------
    def _build_voice(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Cấu hình Giọng đọc & Âm thanh", font=("Segoe UI Semibold", 14)).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self.voice_notebook = ttk.Notebook(frame)
        self.voice_notebook.grid(row=1, column=0, sticky="nsew")

        tab_vs = ttk.Frame(self.voice_notebook, padding=12)
        tab_audio = ttk.Frame(self.voice_notebook, padding=12)

        self.voice_notebook.add(tab_vs, text="VoiceStudio (Giọng đọc)")
        self.voice_notebook.add(tab_audio, text="Hòa âm (Audio Mix)")

        self._build_voicestudio_subtab(tab_vs)
        self._build_audiomix_subtab(tab_audio)

        return frame

    def _build_voicestudio_subtab(self, p: ttk.Frame) -> None:
        p.columnconfigure(1, weight=1)
        row = 0

        lbl_desc = ttk.Label(
            p,
            text="Cấu hình VoiceStudio bên ngoài để đọc lời bình narration. Thử giọng và dựng dùng cùng một adapter.",
            font=("Segoe UI", 9),
            foreground="#475569",
        )
        lbl_desc.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        # Mode
        ttk.Label(p, text="Chế độ kết nối (Mode):", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", pady=4)
        f_mode = ttk.Frame(p)
        f_mode.grid(row=row, column=1, columnspan=2, sticky="w", padx=10, pady=4)
        for m_val, m_text in [("auto", "Tự động (Auto)"), ("local", "Cục bộ (Local)"), ("remote", "Từ xa (Remote/Tailscale)")]:
            ttk.Radiobutton(f_mode, text=m_text, value=m_val, variable=self.var_voice_mode).pack(side="left", padx=6)
        row += 1

        # Local URL
        ttk.Label(p, text="VoiceStudio Local URL:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(p, textvariable=self.var_voice_local).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=4)
        row += 1

        # Remote URL
        ttk.Label(p, text="VoiceStudio Remote URL:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(p, textvariable=self.var_voice_remote).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=4)
        row += 1

        # Remote Credential/Key (DPAPI Masked)
        ttk.Label(p, text="Mật mã từ xa (Remote Key):").grid(row=row, column=0, sticky="w", pady=4)
        self.ent_voice_key = ttk.Entry(p, textvariable=self.var_voice_key, show="●")
        self.ent_voice_key.grid(row=row, column=1, sticky="ew", padx=(10, 6), pady=4)
        self.chk_show_vs = ttk.Checkbutton(
            p, text="Hiện", variable=self.var_voice_key_visible, command=self._toggle_voice_key
        )
        self.chk_show_vs.grid(row=row, column=2, sticky="w")
        row += 1

        # Voice selection (All 12 V2 presets + backend aliases, displayed with human-readable names)
        ttk.Label(p, text="Tên giọng đọc (Voice ID):", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", pady=4)
        v2_voice_ids = list(V2_VOICE_PRESETS.keys())
        initial_voices = v2_voice_ids + ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        initial_display_voices = [get_voice_display_name(vid) for vid in initial_voices]
        self.cmb_voice_id = ttk.Combobox(
            p,
            textvariable=self.var_voice_id,
            values=initial_display_voices,
        )
        self.cmb_voice_id.grid(row=row, column=1, sticky="ew", padx=(10, 6), pady=4)
        self.cbo_voice = self.cmb_voice_id
        btn_fetch = ttk.Button(p, text="Tải danh sách", command=self._fetch_voices_from_backend)
        btn_fetch.grid(row=row, column=2, sticky="w")
        row += 1

        # Voice language & style
        ttk.Label(p, text="Ngôn ngữ giọng (Language):").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(p, textvariable=self.var_voice_lang).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=4)
        row += 1

        ttk.Label(p, text="Phong cách (Style):").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(p, textvariable=self.var_voice_style).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=4)
        row += 1

        ttk.Label(p, text="Commentary Reading Speed:").grid(row=row, column=0, sticky="w", pady=4)
        self.cmb_commentary_speed = ttk.Combobox(p, textvariable=self.var_commentary_speed, values=[f"{value:.2f}x" for value in COMMENTARY_SPEED_VALUES], state="readonly", width=10)
        self.cmb_commentary_speed.grid(row=row, column=1, sticky="w", padx=(10, 0), pady=4)
        ttk.Label(p, text="Fixed narration speed; ToolRecap adjusts video timing to match.", foreground="#64748B").grid(row=row, column=2, sticky="w", padx=6)
        row += 1

        # Action bar: Test, Preview, Progress, Status
        f_actions = ttk.Frame(p)
        f_actions.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        self.btn_test_vs = ttk.Button(f_actions, text="Kiểm tra kết nối", command=self._test_voice_connection)
        self.btn_test_vs.pack(side="left")

        self.btn_preview_voice = ttk.Button(f_actions, text="🔊 Nghe thử giọng (Preview)", command=self._preview_voice)
        self.btn_preview_voice.pack(side="left", padx=(8, 0))
        self.btn_preview = self.btn_preview_voice

        self.voice_preview_prog = ttk.Progressbar(f_actions, length=100, mode="determinate")
        self.voice_preview_prog.pack(side="left", padx=(8, 0))
        row += 1

        self.lbl_voice_status = ttk.Label(p, textvariable=self.voice_status_var, font=("Segoe UI", 8), foreground="#6b7280", wraplength=600)
        self.lbl_voice_status.grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _build_audiomix_subtab(self, p: ttk.Frame) -> None:
        p.columnconfigure(1, weight=1)
        row = 0

        lbl_desc = ttk.Label(
            p,
            text="Điều chỉnh âm lượng âm thanh gốc, giọng bình luận, và chuẩn hóa âm lượng xuất bản.",
            font=("Segoe UI", 9),
            foreground="#475569",
        )
        lbl_desc.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        # Original Audio
        ttk.Label(p, text="Original audio:").grid(row=row, column=0, sticky="w", pady=4)
        f_orig = ttk.Frame(p)
        f_orig.grid(row=row, column=1, sticky="w", padx=10, pady=4)
        self.orig_audio_spin = ttk.Spinbox(
            f_orig, from_=-60.0, to=24.0, increment=0.5, textvariable=self.var_orig_db, width=8
        )
        self.orig_audio_spin.pack(side="left")
        ttk.Label(f_orig, text="dB (Mặc định: 0.0, dải: -60 đến +24)").pack(side="left", padx=(6, 0))
        row += 1

        # Commentary Voice
        ttk.Label(p, text="Commentary voice:").grid(row=row, column=0, sticky="w", pady=4)
        f_comm = ttk.Frame(p)
        f_comm.grid(row=row, column=1, sticky="w", padx=10, pady=4)
        self.commentary_spin = ttk.Spinbox(
            f_comm, from_=-60.0, to=24.0, increment=0.5, textvariable=self.var_comm_db, width=8
        )
        self.commentary_spin.pack(side="left")
        ttk.Label(f_comm, text="dB (Mặc định: 0.0, dải: -60 đến +24)").pack(side="left", padx=(6, 0))
        row += 1

        # Auto-duck
        self.chk_auto_duck = ttk.Checkbutton(
            p,
            text="Tự động hạ âm lượng video gốc khi có lời bình (Auto-duck)",
            variable=self.var_auto_duck,
            command=self._toggle_duck_controls,
        )
        self.chk_auto_duck.grid(row=row, column=0, columnspan=3, sticky="w", pady=6)
        row += 1

        # Ducking Amount
        ttk.Label(p, text="Ducking amount:").grid(row=row, column=0, sticky="w", pady=4)
        f_duck = ttk.Frame(p)
        f_duck.grid(row=row, column=1, sticky="w", padx=10, pady=4)
        self.ducking_spin = ttk.Spinbox(
            f_duck, from_=-60.0, to=0.0, increment=1.0, textvariable=self.var_duck_db, width=8
        )
        self.ducking_spin.pack(side="left")
        self.ent_duck = self.ducking_spin
        ttk.Label(f_duck, text="dB (Mặc định: -12.0, dải: -60 đến 0)").pack(side="left", padx=(6, 0))
        row += 1

        # Target Loudness
        ttk.Label(p, text="Target loudness:").grid(row=row, column=0, sticky="w", pady=4)
        f_lufs = ttk.Frame(p)
        f_lufs.grid(row=row, column=1, sticky="w", padx=10, pady=4)
        self.target_loudness_spin = ttk.Spinbox(
            f_lufs, from_=-70.0, to=0.0, increment=1.0, textvariable=self.var_target_lufs, width=8
        )
        self.target_loudness_spin.pack(side="left")
        ttk.Label(f_lufs, text="LUFS (Mặc định: -14.0, dải: -70 đến 0)").pack(side="left", padx=(6, 0))
        row += 1

        # True Peak
        ttk.Label(p, text="True peak:").grid(row=row, column=0, sticky="w", pady=4)
        f_tp = ttk.Frame(p)
        f_tp.grid(row=row, column=1, sticky="w", padx=10, pady=4)
        self.true_peak_spin = ttk.Spinbox(
            f_tp, from_=-20.0, to=0.0, increment=0.5, textvariable=self.var_true_peak, width=8
        )
        self.true_peak_spin.pack(side="left")
        ttk.Label(f_tp, text="dBTP (Mặc định: -1.0, dải: -20 đến 0)").pack(side="left", padx=(6, 0))
        row += 1

        self.chk_highlight_normalize = ttk.Checkbutton(
            p,
            text="Chuẩn hóa loudness cho Highlight (tắt = giữ mức âm thanh nguồn)",
            variable=self.var_highlight_normalize,
        )
        self.chk_highlight_normalize.grid(row=row, column=0, columnspan=3, sticky="w", pady=6)
        row += 1

        # Reset Audio Defaults
        btn_reset_audio = ttk.Button(p, text="Đặt lại âm thanh mặc định", command=self._reset_audio_defaults)
        btn_reset_audio.grid(row=row, column=0, sticky="w", pady=12)

        self._toggle_duck_controls()

    def _toggle_duck_controls(self) -> None:
        state = tk.NORMAL if self.var_auto_duck.get() else tk.DISABLED
        if hasattr(self, "ducking_spin"):
            self.ducking_spin.config(state=state)

    def _reset_audio_defaults(self) -> None:
        self.var_orig_db.set(0.0)
        self.var_comm_db.set(0.0)
        self.var_auto_duck.set(False)
        self.var_duck_db.set(-12.0)
        self.var_target_lufs.set(-14.0)
        self.var_true_peak.set(-1.0)
        self.var_highlight_normalize.set(False)
        self._toggle_duck_controls()

    def _toggle_voice_key(self) -> None:
        if hasattr(self, "ent_voice_key") and self.ent_voice_key is not None:
            self.ent_voice_key.configure(show="" if self.var_voice_key_visible.get() else "●")

    def _get_voice_adapter(self) -> VoiceStudioAdapter:
        mode = self.var_voice_mode.get()
        local_url = self.var_voice_local.get().strip()
        remote_url = self.var_voice_remote.get().strip()
        key_input = self.var_voice_key.get()
        remote_key = self.secret_store.get_secret("voice_remote_api_key") if key_input == MASKED_SECRET_PLACEHOLDER else key_input

        return VoiceStudioAdapter(
            mode=mode,
            local_url=local_url,
            remote_url=remote_url,
            remote_api_key=remote_key or None,
            timeout=10.0,
        )

    def _test_voice_connection(self) -> None:
        self.voice_status_var.set("Đang kiểm tra kết nối VoiceStudio...")
        self.lbl_voice_status.config(foreground="#2563EB")

        def _do_test() -> None:
            try:
                adapter = self._get_voice_adapter()
                health = adapter.check_health()
                status_txt = health.get("status", "ok") if isinstance(health, dict) else "ok"
                self._post_ui(lambda: (
                    self.voice_status_var.set(f"✓ Kết nối VoiceStudio thành công ({status_txt})"),
                    self.lbl_voice_status.config(foreground="#16A34A"),
                    self.voice_preview_prog.configure(value=100),
                ))
            except Exception as e:
                self._post_ui(lambda err=str(e): (
                    self.voice_status_var.set(f"✗ Lỗi kết nối: {err[:80]}"),
                    self.lbl_voice_status.config(foreground="#DC2626"),
                    self.voice_preview_prog.configure(value=0),
                ))

        threading.Thread(target=_do_test, daemon=True).start()

    def _fetch_voices_from_backend(self) -> None:
        self.voice_status_var.set("Đang lấy danh sách giọng từ VoiceStudio...")
        self.lbl_voice_status.config(foreground="#2563EB")

        def _do_fetch() -> None:
            try:
                adapter = self._get_voice_adapter()
                voices = adapter.get_voices()
                voice_ids = []
                for v in voices:
                    if isinstance(v, dict):
                        vid = v.get("id") or v.get("name")
                        if vid:
                            voice_ids.append(str(vid))
                    elif isinstance(v, str):
                        voice_ids.append(v)
                if voice_ids:
                    display_names = [get_voice_display_name(vid) for vid in voice_ids]
                    self._post_ui(lambda: (
                        self.cmb_voice_id.config(values=display_names),
                        self.voice_status_var.set(f"✓ Đã tải {len(display_names)} giọng đọc."),
                        self.lbl_voice_status.config(foreground="#16A34A"),
                    ))
                else:
                    self._post_ui(lambda: (
                        self.voice_status_var.set("Không tìm thấy danh sách giọng."),
                        self.lbl_voice_status.config(foreground="#D97706"),
                    ))
            except Exception as e:
                self._post_ui(lambda err=str(e): (
                    self.voice_status_var.set(f"✗ Lỗi tải danh sách: {err[:80]}"),
                    self.lbl_voice_status.config(foreground="#DC2626"),
                ))

        threading.Thread(target=_do_fetch, daemon=True).start()

    def _preview_voice(self) -> None:
        raw_val = self.var_voice_id.get().strip()
        voice = resolve_voice_id(raw_val) or "alloy"
        lang = self.var_voice_lang.get().strip() or "en-US"
        style = self.var_voice_style.get().strip()

        self.voice_status_var.set("Đang tạo mẫu giọng...")
        self.lbl_voice_status.config(foreground="#2563EB")
        self.voice_preview_prog.configure(value=20)

        def _do_preview() -> None:
            try:
                adapter = self._get_voice_adapter()
                sample_text = "Xin chào, đây là giọng đọc thử nghiệm của hệ thống ToolRecap V5."
                if lang.startswith("en"):
                    sample_text = "Hello, this is a voice synthesis preview from ToolRecap V5."

                tmp_wav = self.persistence.root / "cache" / "preview_sample.wav"
                tmp_wav.parent.mkdir(parents=True, exist_ok=True)

                self._post_ui(lambda: self.voice_preview_prog.configure(value=50))
                wav_bytes = adapter.synthesize(
                    input_text=sample_text,
                    voice=voice,
                    model=self.settings.voice_model,
                    language=lang,
                    style=style,
                )
                speed=float(self.var_commentary_speed.get().rstrip("x"))
                wav_bytes=apply_commentary_speed(wav_bytes,speed)
                tmp_wav.write_bytes(wav_bytes)
                self._post_ui(lambda: self.voice_preview_prog.configure(value=100))

                try:
                    import winsound
                    winsound.PlaySound(str(tmp_wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
                except Exception as audio_err:
                    logger.warning("Playback error: %s", audio_err)

                self._post_ui(lambda: (
                    self.voice_status_var.set("✓ Đã tạo và phát âm thanh thử nghiệm!"),
                    self.lbl_voice_status.config(foreground="#16A34A"),
                ))
            except Exception as e:
                self._post_ui(lambda err=str(e): (
                    self.voice_status_var.set(f"✗ Lỗi thử giọng: {err[:80]}"),
                    self.lbl_voice_status.config(foreground="#DC2626"),
                    self.voice_preview_prog.configure(value=0),
                ))

        threading.Thread(target=_do_preview, daemon=True).start()

    # -------------------------------------------------------------------------
    # PANE 4: RENDER, NOTIFICATIONS & UPDATES
    # -------------------------------------------------------------------------
    def _build_render(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Cấu hình Render, Thông báo & Cập nhật", font=("Segoe UI Semibold", 14)).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self.render_notebook = ttk.Notebook(frame)
        self.render_notebook.grid(row=1, column=0, sticky="nsew")

        tab_render = ttk.Frame(self.render_notebook, padding=12)
        tab_notify = ttk.Frame(self.render_notebook, padding=12)
        tab_update = ttk.Frame(self.render_notebook, padding=12)

        self.render_notebook.add(tab_render, text="Xuất video (Render)")
        self.render_notebook.add(tab_notify, text="Thông báo (Notifications)")
        self.render_notebook.add(tab_update, text="Cập nhật (Updates)")

        self._build_render_subtab(tab_render)
        self._build_notify_subtab(tab_notify)
        self._build_update_subtab(tab_update)

        return frame

    def _build_render_subtab(self, p: ttk.Frame) -> None:
        p.columnconfigure(1, weight=1)
        row = 0

        lbl_desc = ttk.Label(
            p,
            text="Cấu hình thông số xuất video thành phẩm, phần cứng GPU và phụ đề.",
            font=("Segoe UI", 9),
            foreground="#475569",
        )
        lbl_desc.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        # Output Dir
        ttk.Label(p, text="Thư mục xuất video:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(p, textvariable=self.var_out_dir).grid(row=row, column=1, sticky="ew", padx=(10, 6), pady=5)
        ttk.Button(p, text="Duyệt...", command=self._browse_output_dir).grid(row=row, column=2, sticky="w", pady=5)
        row += 1
        ttk.Label(
            p,
            text="Để trống = tự động tạo thư mục cùng cấp: Outputs_<tên thư mục nguồn>",
            foreground="#6b7280",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 5))
        row += 1

        # GPU Acceleration
        ttk.Checkbutton(
            p,
            text="Bật tăng tốc phần cứng GPU (NVENC / AMF / QSV)",
            variable=self.var_use_gpu,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        # Video Quality
        ttk.Label(p, text="Chất lượng video:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(
            p,
            textvariable=self.var_quality,
            values=["high", "medium", "low", "standard", "source"],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=5)
        row += 1

        # Video Codec
        ttk.Label(p, text="Bộ mã hóa (Codec):").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(
            p,
            textvariable=self.var_codec,
            values=["h264", "hevc"],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=5)
        row += 1

        # Resolution / Canvas (Auto from source video)
        ttk.Label(p, text="Độ phân giải khung hình:").grid(row=row, column=0, sticky="w", pady=5)
        f_res = ttk.Frame(p)
        f_res.grid(row=row, column=1, columnspan=2, sticky="w", padx=(10, 0), pady=5)
        self.lbl_canvas_auto = ttk.Label(
            f_res,
            text="Tự động theo video nguồn (Khung hình chuẩn chẵn)",
            font=("Segoe UI", 9, "bold"),
            foreground="#1E40AF",
        )
        self.lbl_canvas_auto.pack(side="left")
        self.canvas_auto_label = self.lbl_canvas_auto
        ttk.Label(f_res, text="   @   ").pack(side="left")
        self.ent_canvas_fps = ttk.Entry(f_res, textvariable=self.var_canvas_fps, width=6)
        self.ent_canvas_fps.pack(side="left")
        ttk.Label(f_res, text=" FPS").pack(side="left")
        row += 1

        # Burn Subtitles
        ttk.Checkbutton(
            p,
            text="Nhúng thẳng phụ đề vào video (Burn subtitles)",
            variable=self.var_burn_subs,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        # Output Format
        ttk.Label(p, text="Định dạng xuất:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(p, textvariable=self.var_out_format, width=12).grid(row=row, column=1, sticky="w", padx=(10, 0), pady=5)

    def _browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Chọn thư mục xuất video mặc định")
        if folder:
            self.var_out_dir.set(folder)

    def _build_notify_subtab(self, p: ttk.Frame) -> None:
        row = 0

        lbl_desc = ttk.Label(
            p,
            text="Cấu hình thông báo Windows Desktop Toast, âm thanh và nhấp nháy Taskbar khi dự án kết thúc.",
            font=("Segoe UI", 9),
            foreground="#475569",
        )
        lbl_desc.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 12))
        row += 1

        ttk.Checkbutton(
            p, text="Hiện thông báo Windows khi hoàn tất dự án", variable=self.var_notify_complete
        ).grid(row=row, column=0, sticky="w", pady=5)
        row += 1

        ttk.Checkbutton(
            p, text="Hiện thông báo Windows khi gặp sự cố dừng dự án", variable=self.var_notify_error
        ).grid(row=row, column=0, sticky="w", pady=5)
        row += 1

        ttk.Checkbutton(
            p, text="Phát âm thanh thông báo khi xong", variable=self.var_play_sound
        ).grid(row=row, column=0, sticky="w", pady=5)
        row += 1

        ttk.Checkbutton(
            p, text="Nhấp nháy biểu tượng trên thanh tác vụ (Taskbar) khi ở chế độ nền", variable=self.var_flash_taskbar
        ).grid(row=row, column=0, sticky="w", pady=5)
        row += 1

        ttk.Checkbutton(
            p, text="Thông báo khi có bản cập nhật mới", variable=self.var_notify_update
        ).grid(row=row, column=0, sticky="w", pady=5)
        row += 1

        btn_test_notify = ttk.Button(p, text="Thử thông báo ngay", command=self._test_notification)
        btn_test_notify.grid(row=row, column=0, sticky="w", pady=14)

        self.lbl_notify_test = ttk.Label(p, text="", font=("Segoe UI", 9))
        self.lbl_notify_test.grid(row=row, column=1, sticky="w", padx=8, pady=14)

    def _test_notification(self) -> None:
        service = WindowsNotificationService()
        hwnd = self.winfo_id() if self.var_flash_taskbar.get() else None
        res = service.send_toast(
            title="ToolRecap V5 — Kiểm tra thông báo",
            body="Thông báo kiểm tra hệ thống thành công.",
            sound=self.var_play_sound.get(),
            hwnd=hwnd,
        )
        if res.api_success:
            self.lbl_notify_test.config(text="✓ Đã gửi thông báo tới Windows!", foreground="#16A34A")
        else:
            self.lbl_notify_test.config(text=f"✗ Gửi thất bại: {res.error}", foreground="#DC2626")

    def _build_update_subtab(self, p: ttk.Frame) -> None:
        p.columnconfigure(1, weight=1)
        row = 0

        lbl_desc = ttk.Label(
            p,
            text="Thông tin phiên bản ứng dụng và cấu hình máy chủ cập nhật GitHub.",
            font=("Segoe UI", 9),
            foreground="#475569",
        )
        lbl_desc.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        ttk.Label(p, text="Phiên bản hiện tại:", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Label(p, text=f"ToolRecap V5 — v{__version__}", font=("Segoe UI", 9)).grid(row=row, column=1, columnspan=2, sticky="w", padx=10, pady=5)
        row += 1

        ttk.Label(p, text="Kho lưu trữ GitHub:", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", pady=5)
        ent_repo = ttk.Entry(p, textvariable=self.var_update_repo, width=32)
        ent_repo.grid(row=row, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(p, text="(Ví dụ: owner/repo — để trống nếu chưa cấu hình)", font=("Segoe UI", 8), foreground="#64748B").grid(row=row, column=2, sticky="w", padx=4, pady=5)
        row += 1

        ttk.Label(p, text="Trạng thái cập nhật:", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", pady=5)
        self.lbl_update_status = ttk.Label(
            p,
            text="Chưa cấu hình kho lưu trữ cập nhật (Not configured)" if not self.var_update_repo.get().strip() else "Sẵn sàng kiểm tra bản cập nhật",
            font=("Segoe UI", 9),
            foreground="#D97706" if not self.var_update_repo.get().strip() else "#475569",
        )
        self.lbl_update_status.grid(row=row, column=1, columnspan=2, sticky="w", padx=10, pady=5)
        row += 1

        btn_box = ttk.Frame(p)
        btn_box.grid(row=row, column=0, columnspan=3, sticky="w", pady=14)

        self.btn_check_update = ttk.Button(btn_box, text="Kiểm tra cập nhật", command=self._check_update)
        self.btn_check_update.pack(side="left", padx=(0, 8))

        self.btn_download_update = ttk.Button(btn_box, text="Tải bản cập nhật", command=self._download_update, state="disabled")
        self.btn_download_update.pack(side="left", padx=8)

        self.btn_apply_update = ttk.Button(btn_box, text="Cài đặt & Khởi động lại", command=self._apply_update, state="disabled")
        self.btn_apply_update.pack(side="left", padx=8)
        row += 1

        self.lbl_update_result = ttk.Label(p, text="", font=("Segoe UI", 9))
        self.lbl_update_result.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)

    def _check_update(self) -> None:
        repo = self.var_update_repo.get().strip()
        if not repo:
            msg = "Chưa cấu hình kho lưu trữ cập nhật (Not configured). Vui lòng nhập kho lưu trữ GitHub (owner/repo)."
            self.lbl_update_status.config(text="Chưa cấu hình kho lưu trữ cập nhật (Not configured)", foreground="#D97706")
            self.lbl_update_result.config(text=msg, foreground="#D97706")
            return

        self.lbl_update_status.config(text="Đang kiểm tra bản cập nhật...", foreground="#2563EB")
        self.lbl_update_result.config(text="Đang kết nối tới GitHub...", foreground="#2563EB")
        self.btn_check_update.config(state="disabled")

        def _work() -> None:
            try:
                result = self.update_manager.check_for_updates(repo)
                self._post_ui(lambda: self._finish_update_check(result))
            except Exception as exc:
                self._post_ui(lambda error=exc: self._finish_update_check_error(error))

        threading.Thread(target=_work, daemon=True, name="SettingsUpdateCheck").start()

    def _finish_update_check_error(self, error: Exception) -> None:
        if self._closed:
            return
        self.btn_check_update.config(state="normal")
        self.lbl_update_status.config(text="Lỗi kiểm tra cập nhật", foreground="#DC2626")
        self.lbl_update_result.config(text=f"Lỗi: {error}", foreground="#DC2626")

    def _finish_update_check(self, result: UpdateCheckResult) -> None:
        if self._closed:
            return
        self.btn_check_update.config(state="normal")
        self._last_check_result = result

        if result.status == "not_configured":
            self.lbl_update_status.config(text=result.message, foreground="#D97706")
            self.lbl_update_result.config(text=result.message, foreground="#D97706")
        elif result.status == "up_to_date":
            self.lbl_update_status.config(text="Đã cập nhật mới nhất", foreground="#16A34A")
            self.lbl_update_result.config(text=result.message, foreground="#16A34A")
            self.btn_download_update.config(state="disabled")
            self.btn_apply_update.config(state="disabled")
        elif result.status == "update_available":
            self.lbl_update_status.config(text=f"Có bản cập nhật mới v{result.latest_version}", foreground="#2563EB")
            self.lbl_update_result.config(text=result.message, foreground="#2563EB")
            self.btn_download_update.config(state="normal")
            self.btn_apply_update.config(state="disabled")
        else:
            self.lbl_update_status.config(text="Lỗi kiểm tra cập nhật", foreground="#DC2626")
            self.lbl_update_result.config(text=result.message, foreground="#DC2626")

    def _download_update(self) -> None:
        if not self._last_check_result or self._last_check_result.status != "update_available":
            return

        self.btn_download_update.config(state="disabled")
        self.lbl_update_status.config(text="Đang tải bản cập nhật...", foreground="#2563EB")
        self.lbl_update_result.config(text="Đang tải và xác thực gói cập nhật...", foreground="#2563EB")
        result = self._last_check_result

        def _work() -> None:
            try:
                def on_progress(fraction: float, msg: str) -> None:
                    self._post_ui(lambda: self.lbl_update_result.config(text=f"{int(fraction * 100)}% - {msg}"))

                staged_path = self.update_manager.download_and_stage(result, progress_callback=on_progress)
                self._post_ui(lambda: self._finish_update_download(staged_path))
            except Exception as exc:
                self._post_ui(lambda error=exc: self._finish_update_download_error(error))

        threading.Thread(target=_work, daemon=True, name="SettingsUpdateDownload").start()

    def _finish_update_download(self, staged_path: Path) -> None:
        if self._closed:
            return
        self._staged_update_path = staged_path
        self.lbl_update_status.config(text="Đã tải và xác thực hoàn tất", foreground="#16A34A")
        self.lbl_update_result.config(text="Gói cập nhật đã sẵn sàng để cài đặt.", foreground="#16A34A")
        self.btn_apply_update.config(state="normal")

    def _finish_update_download_error(self, error: Exception) -> None:
        if self._closed:
            return
        self.lbl_update_status.config(text="Tải bản cập nhật thất bại", foreground="#DC2626")
        self.lbl_update_result.config(text=f"Lỗi: {error}", foreground="#DC2626")
        self.btn_download_update.config(state="normal")

    def _apply_update(self) -> None:
        if hasattr(self.parent, "worker") and getattr(self.parent.worker, "is_running", False):
            self.lbl_update_result.config(text="Chờ dự án dừng trước khi cập nhật.", foreground="#D97706")
            return
        if not self._staged_update_path or not self._last_check_result:
            return

        confirm = messagebox.askyesno(
            "Xác nhận cập nhật",
            f"Bạn có chắc chắn muốn cài đặt bản cập nhật v{self._last_check_result.latest_version}?\nỨng dụng sẽ tự động khởi động lại sau khi hoàn tất.",
            parent=self,
        )
        if not confirm:
            return

        try:
            target_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
            self.update_manager.launch_apply_helper(
                staged_path=self._staged_update_path,
                target_dir=target_dir,
            )
            self.destroy()
            if self.parent:
                self.parent.destroy()
        except Exception as e:
            self.lbl_update_result.config(text=f"Lỗi áp dụng cập nhật: {e}", foreground="#DC2626")

    # -------------------------------------------------------------------------
    # SAVE & VALIDATION
    # -------------------------------------------------------------------------
    def _on_save(self) -> None:
        """Validate all inputs and persist settings and DPAPI secrets."""
        original_settings = self.settings
        try:
            # 1. Validate URLs
            gw_url = validate_url_no_credentials(self.var_gw_endpoint.get(), "Gateway Endpoint")
            if not gw_url:
                raise ValueError("Gateway Endpoint không được để trống.")

            vs_local = validate_url_no_credentials(self.var_voice_local.get(), "VoiceStudio Local URL")
            vs_remote = validate_url_no_credentials(self.var_voice_remote.get(), "VoiceStudio Remote URL")
            gateway_form = self._gateway_form_values()
            GatewaySettingsController.validate(gateway_form)

            # 2. Validate Numeric ranges
            orig_db = float(self.var_orig_db.get())
            if not (-60.0 <= orig_db <= 24.0):
                raise ValueError("Âm lượng gốc phải nằm trong khoảng từ -60.0 dB đến +24.0 dB.")

            comm_db = float(self.var_comm_db.get())
            if not (-60.0 <= comm_db <= 24.0):
                raise ValueError("Âm lượng lời bình phải nằm trong khoảng từ -60.0 dB đến +24.0 dB.")

            duck_db = float(self.var_duck_db.get())
            if not (-60.0 <= duck_db <= 0.0):
                raise ValueError("Mức giảm ducking phải nằm trong khoảng từ -60.0 dB đến 0.0 dB.")

            target_lufs = float(self.var_target_lufs.get())
            if not (-70.0 <= target_lufs <= 0.0):
                raise ValueError("Độ lớn mục tiêu LUFS phải nằm trong khoảng từ -70.0 đến 0.0.")

            true_peak = float(self.var_true_peak.get())
            if not (-20.0 <= true_peak <= 0.0):
                raise ValueError("Mức đỉnh thực True peak phải nằm trong khoảng từ -20.0 đến 0.0 dBTP.")

            canvas_fps = float(self.var_canvas_fps.get())
            if not (1.0 <= canvas_fps <= 120.0):
                raise ValueError("Tốc độ khung hình FPS phải nằm trong khoảng từ 1.0 đến 120.0.")

            # 3. Update AppSettings instance
            self.settings = replace(self.settings)
            self.settings.recap_language = self.recap_language_var.get()
            self.settings.recap_mode = self.recap_mode_var.get()
            self.settings.content_type = self.content_type_var.get()
            self.settings.source_rights_status = self.rights_var.get()
            self.settings.prompt = self.txt_prompt.get("1.0", "end-1c")
            self.settings.highlight_prompt = self.txt_highlight_prompt.get("1.0", "end-1c")

            self.settings.voice_mode = self.var_voice_mode.get().strip()
            self.settings.voice_local_url = vs_local
            self.settings.voice_remote_url = vs_remote
            raw_voice_choice = self.var_voice_id.get().strip()
            self.settings.voice_id = resolve_voice_id(raw_voice_choice)
            self.settings.voice_language = self.var_voice_lang.get().strip()
            self.settings.voice_style = self.var_voice_style.get().strip()
            speed=float(self.var_commentary_speed.get().rstrip("x"))
            if not any(abs(speed-value)<1e-9 for value in COMMENTARY_SPEED_VALUES):raise ValueError("Commentary Reading Speed không được hỗ trợ.")
            self.settings.commentary_reading_speed=round(speed,2)

            self.settings.original_audio_db = orig_db
            self.settings.commentary_audio_db = comm_db
            self.settings.auto_duck = bool(self.var_auto_duck.get())
            self.settings.ducking_amount_db = duck_db
            self.settings.target_loudness_lufs = target_lufs
            self.settings.true_peak_db = true_peak
            self.settings.highlight_normalize_loudness = bool(self.var_highlight_normalize.get())

            self.settings.output_dir = self.var_out_dir.get().strip()
            self.settings.use_gpu = bool(self.var_use_gpu.get())
            self.settings.quality = self.var_quality.get().strip()
            self.settings.video_codec = self.var_codec.get().strip()
            self.settings.canvas_auto = True
            self.var_canvas_auto.set(True)
            self.settings.canvas_width = int(self.var_canvas_w.get()) if hasattr(self, "var_canvas_w") and self.var_canvas_w.get() else 1920
            self.settings.canvas_height = int(self.var_canvas_h.get()) if hasattr(self, "var_canvas_h") and self.var_canvas_h.get() else 1080
            self.settings.canvas_fps = canvas_fps
            self.settings.burn_subtitles = bool(self.var_burn_subs.get())
            self.settings.output_format = self.var_out_format.get().strip()

            self.settings.notify_complete = bool(self.var_notify_complete.get())
            self.settings.notify_error = bool(self.var_notify_error.get())
            self.settings.play_completion_sound = bool(self.var_play_sound.get())
            self.settings.flash_taskbar = bool(self.var_flash_taskbar.get())
            self.settings.notify_update = bool(self.var_notify_update.get())

            self.settings.update_repo = self.var_update_repo.get().strip()

            # 4. Atomically persist non-secret Gateway settings and secure key.
            self.settings = self.gateway_controller.save(
                self.settings, gateway_form, masked_placeholder=MASKED_SECRET_PLACEHOLDER
            )

            # 5. Save Voice secret exclusively to DPAPISecretStore.
            vs_key_input = self.var_voice_key.get()
            if vs_key_input and vs_key_input != MASKED_SECRET_PLACEHOLDER:
                self.secret_store.set_secret("voice_remote_api_key", vs_key_input.strip())
            elif not vs_key_input and self._had_vs_key:
                self.secret_store.delete_secret("voice_remote_api_key")

            if self.on_saved:
                self.on_saved(self.settings)

            self.lbl_status.config(text="✓ Đã lưu cài đặt thành công!", foreground="#16A34A")
            if getattr(getattr(self.parent, "worker", None), "is_running", False):
                self.lbl_status.config(
                    text="✓ Đã lưu. Thay đổi sẽ áp dụng cho lần chạy phù hợp tiếp theo.",
                    foreground="#D97706",
                )

        except Exception as e:
            self.settings = original_settings
            safe_error = sanitize_message(
                str(e),
                [self.var_gw_key.get(), self.var_voice_key.get()],
            )
            self.lbl_status.config(text=f"⚠ {safe_error}", foreground="#DC2626")

    def _on_close(self) -> None:
        """Revert unsaved edits without closing the application or this page."""
        self.settings = self.settings_manager.load()
        self._load_values()
        self._reload_non_prompt_values()
        self.lbl_status.config(text="Đã hoàn tác các thay đổi chưa lưu.", foreground="#475569")

    def destroy(self) -> None:
        """Cancel page-owned callbacks during main-window shutdown."""
        if self._closed:
            return
        self._closed = True
        if self._ui_poll_job:
            try:
                self.after_cancel(self._ui_poll_job)
            except Exception:
                pass
            self._ui_poll_job = None
        try:
            super().destroy()
        finally:
            callback = self.on_closed
            self.on_closed = None
            if callback:
                try:
                    callback(self)
                except Exception:
                    logger.exception("Settings dialog close callback failed")
