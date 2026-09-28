"""Tests for MainWindow and SettingsDialog UI components."""

import os
from pathlib import Path
import sys

if sys.platform == "win32":
    tcl_path = os.path.join(sys.base_prefix, "tcl", "tcl8.6")
    tk_path = os.path.join(sys.base_prefix, "tcl", "tk8.6")
    if os.path.isdir(tcl_path) and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = tcl_path
    if os.path.isdir(tk_path) and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = tk_path

import tkinter as tk
import time
from unittest.mock import MagicMock, patch
import pytest

from toolrecap_v4.discovery import SourceFingerprint
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.secrets import DPAPISecretStore
from toolrecap_v4.settings import AppSettings, SettingsManager
from toolrecap_v4.ui.main_window import MainWindow, format_bytes, format_duration
from toolrecap_v4.ui.settings_dialog import (
    MASKED_SECRET_PLACEHOLDER,
    SettingsDialog,
    validate_url_no_credentials,
)
from toolrecap_v4.ui.worker import WorkerMessage
from toolrecap_v4.workflow import ProjectStatus


def test_format_helpers():
    """Verify format_bytes and format_duration format correctly."""
    assert format_bytes(500) == "0.5 KB"
    assert format_bytes(1048576) == "1.0 MB"
    assert format_bytes(1073741824) == "1.00 GB"

    assert format_duration(45) == "00:45"
    assert format_duration(125) == "02:05"
    assert format_duration(3665) == "01:01:05"


def test_validate_url_no_credentials():
    """Verify URL validation ensures HTTP/HTTPS and rejects embedded credentials."""
    assert validate_url_no_credentials("http://localhost:8000", "URL") == "http://localhost:8000"
    assert validate_url_no_credentials("https://api.openai.com/v1", "URL") == "https://api.openai.com/v1"
    assert validate_url_no_credentials("", "URL") == ""

    with pytest.raises(ValueError, match="bắt đầu bằng"):
        validate_url_no_credentials("ftp://example.com", "URL")

    with pytest.raises(ValueError, match="thông tin xác thực"):
        validate_url_no_credentials("https://user:pass@example.com", "URL")


def test_main_window_init(tmp_path: Path):
    """Verify MainWindow initializes with proper defaults and widgets."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()
        assert app.persistence is persistence
        assert app.title().startswith("ToolRecap V5")
        assert str(app.btn_start["state"]) == "normal"
        assert str(app.btn_stop["state"]) == "disabled"
        assert str(app.btn_resume["state"]) == "disabled"
        assert len(app.discovered_sources) == 0
    finally:
        app.destroy()


def test_main_window_source_controls(tmp_path: Path):
    """Verify source list controls, table refresh, async selection wait queue, and clearing."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()

        fp1 = SourceFingerprint(
            basename="Ep01.mp4",
            path=str(tmp_path / "Ep01.mp4"),
            size_bytes=104857600,
            mtime_ns=1000,
            sha256="hash1",
            extension=".mp4",
        )
        fp2 = SourceFingerprint(
            basename="Ep02.mp4",
            path=str(tmp_path / "Ep02.mp4"),
            size_bytes=209715200,
            mtime_ns=2000,
            sha256="hash2",
            extension=".mp4",
        )

        app.discovered_sources = [fp1, fp2]
        app._refresh_sources_table()
        app.update_idletasks()

        items = app.tree_sources.get_children()
        assert len(items) == 2
        assert "2 video" in app.lbl_src_summary["text"]

        app._on_clear_sources()
        app.update_idletasks()
        assert len(app.tree_sources.get_children()) == 0
        assert "Chưa chọn" in app.lbl_src_summary["text"]

        # Adapt existing UI test for async selection with wait queue
        sample_folder = tmp_path / "Season1"
        sample_folder.mkdir()
        (sample_folder / "Episode1.mp4").write_bytes(b"dummy1")
        (sample_folder / "Episode2.mp4").write_bytes(b"dummy2")

        app._load_folder(sample_folder)
        start_t = time.time()
        while time.time() - start_t < 3.0:
            app._poll_queue()
            app.update_idletasks()
            if len(app.discovered_sources) == 2:
                break
            time.sleep(0.02)

        assert len(app.discovered_sources) == 2
        assert len(app.tree_sources.get_children()) == 2
        assert "2 video" in app.lbl_src_summary["text"]

        app._on_clear_sources()
        app.update_idletasks()
        assert len(app.discovered_sources) == 0
        assert len(app.tree_sources.get_children()) == 0
    finally:
        app.destroy()


def test_main_window_outputs_and_worker_messages(tmp_path: Path):
    """Verify output treeview population and worker message handling."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()

        final_json = {
            "outputs": [
                {
                    "render_id": "out-1",
                    "title": "Tập 1",
                    "segments": [
                        {"start_ms": 0, "end_ms": 5000},
                        {"start_ms": 5000, "end_ms": 15000},
                    ],
                },
                {
                    "render_id": "out-2",
                    "title": "Tập 2",
                    "segments": [{"start_ms": 0, "end_ms": 8000}],
                },
            ]
        }

        app._handle_worker_message(WorkerMessage(kind="final_json", data=final_json))
        app.update_idletasks()

        assert len(app.tree_outputs.get_children()) == 2
        row1 = app.tree_outputs.item("out-1")["values"]
        assert row1[0] == "Output 1"
        assert "Tập 1" in str(row1[1])
        assert row1[2] == "Chờ dựng"
        assert row1[3] == "0%"
        assert row1[4] == "Tập 1"

        # Test output_started
        app._handle_worker_message(WorkerMessage(kind="output_started", data="out-1"))
        assert app.tree_outputs.item("out-1")["values"][2] == "Rendering"
        assert app.tree_outputs.item("out-1")["values"][4] == "Đang dựng..."

        # Test output_completed
        app._handle_worker_message(
            WorkerMessage(
                kind="output_completed",
                data={"render_id": "out-1", "data": {"output_path": "out1.mp4", "duration": 15.0}},
            )
        )
        assert app.tree_outputs.item("out-1")["values"][1] == "out1.mp4"
        assert app.tree_outputs.item("out-1")["values"][2] == "Hoàn thành"
        assert app.tree_outputs.item("out-1")["values"][3] == "100%"
        assert "Hoàn tất" in app.tree_outputs.item("out-1")["values"][4]

        # Test output_failed
        app._handle_worker_message(
            WorkerMessage(kind="output_failed", data={"render_id": "out-2", "error": "FFmpeg crash"})
        )
        assert app.tree_outputs.item("out-2")["values"][2] == "Lỗi"
        assert "Thất bại" in app.tree_outputs.item("out-2")["values"][4]

        # Test status_change
        app._handle_worker_message(WorkerMessage(kind="status_change", data=ProjectStatus.RENDERING.value))
        assert "Đang dựng video" in app.lbl_stage["text"]

        # Test log
        app._handle_worker_message(WorkerMessage(kind="log", data="Kiểm tra nhật ký"))
        log_content = app.txt_log.get("1.0", tk.END)
        assert "Kiểm tra nhật ký" in log_content
    finally:
        app.destroy()


def test_main_window_start_naming(tmp_path: Path):
    """Verify safe project naming works for single file and multi-file folders without crash."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    # Save non-empty prompt so start doesn't prompt modal question
    mgr = SettingsManager(persistence=persistence)
    s = mgr.load()
    s.prompt = "Recap prompt"
    mgr.save(s)

    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()

        # Single file
        fp_single = SourceFingerprint(
            basename="Movie.mp4",
            path=str(tmp_path / "Movie.mp4"),
            size_bytes=100,
            mtime_ns=100,
            sha256="h",
            extension=".mp4",
        )
        app.discovered_sources = [fp_single]

        with patch.object(app.worker, "start_new_project") as mock_start:
            app._on_start_project()
            assert app.current_project_name == "Movie"
            assert app.current_project_id.startswith("proj_Movie_")
            mock_start.assert_called_once()

        # Multi file
        season_dir = tmp_path / "MySeason"
        season_dir.mkdir(exist_ok=True)
        fp_m1 = SourceFingerprint(
            basename="Ep1.mp4",
            path=str(season_dir / "Ep1.mp4"),
            size_bytes=100,
            mtime_ns=100,
            sha256="h1",
            extension=".mp4",
        )
        fp_m2 = SourceFingerprint(
            basename="Ep2.mp4",
            path=str(season_dir / "Ep2.mp4"),
            size_bytes=100,
            mtime_ns=100,
            sha256="h2",
            extension=".mp4",
        )
        app.discovered_sources = [fp_m1, fp_m2]

        with patch.object(app.worker, "start_new_project") as mock_start:
            app._on_start_project()
            assert app.current_project_name == "MySeason"
            assert app.current_project_id.startswith("season_MySeason_")
            mock_start.assert_called_once()
    finally:
        app.destroy()


def test_settings_dialog_masked_secrets_and_validation(tmp_path: Path):
    """Verify SettingsDialog masks DPAPI secrets, validates inputs, and saves correctly."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    secret_store = DPAPISecretStore(storage_root=tmp_path)
    secret_store.set_secret("gateway_api_key", "secret-gw-key-1234")

    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()

        dialog = SettingsDialog(app, persistence=persistence)
        dialog.update_idletasks()

        # Gateway key should be masked
        assert dialog.var_gw_key.get() == MASKED_SECRET_PLACEHOLDER
        assert dialog._had_gw_key is True

        # Check tabs count
        assert dialog.notebook.index("end") == 8

        # Invalid URL validation
        dialog.var_gw_endpoint.set("invalid://not-http")
        with patch("toolrecap_v4.ui.settings_dialog.messagebox.showerror") as mock_err:
            dialog._on_save()
            mock_err.assert_not_called()
            assert "bắt đầu bằng" in dialog.lbl_status.cget("text")

        # Fix URL and set invalid FPS
        dialog.var_gw_endpoint.set("http://localhost:8000")
        dialog.var_canvas_fps.set(0)
        with patch("toolrecap_v4.ui.settings_dialog.messagebox.showerror") as mock_err:
            dialog._on_save()
            mock_err.assert_not_called()
            assert "FPS" in dialog.lbl_status.cget("text")

        # Verify canvas auto label is present and width/height entry widgets removed
        assert hasattr(dialog, "lbl_canvas_auto")
        assert "Tự động" in dialog.lbl_canvas_auto.cget("text")

        # Verify voice combobox has human-readable names and resolves to voice ID on save
        assert "Documentarian — Nam — ToolRecap Local" in dialog.cmb_voice_id["values"]
        dialog.var_voice_id.set("Documentarian — Nam — ToolRecap Local")

        # Fix FPS and save valid
        dialog.var_canvas_fps.set(30.0)
        dialog.var_gw_sub_model.set("scanner-route")
        dialog.var_gw_prime_model.set("finalizer-route")
        dialog._on_save()

        # Ensure settings were saved
        mgr = SettingsManager(persistence=persistence)
        loaded = mgr.load()
        assert loaded.gateway_endpoint == "http://localhost:8000"
        assert loaded.canvas_auto is True
        assert loaded.voice_id == "voicestudio.en.documentarian"

        # Ensure secret was preserved in DPAPI and not leaked
        assert secret_store.get_secret("gateway_api_key") == "secret-gw-key-1234"

        dialog.destroy()
    finally:
        app.destroy()


def test_settings_dialog_initial_tab_and_prompt_reset(tmp_path: Path):
    """Verify SettingsDialog respects initial_tab and resets prompt to empty string."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()

        # 1. Open with initial_tab="update"
        dialog = SettingsDialog(app, persistence=persistence, initial_tab="update")
        dialog.update_idletasks()

        # Update tab is index 6
        selected_idx = dialog.notebook.index(dialog.notebook.select())
        assert selected_idx == 6

        # 2. Open with numeric initial_tab=1 (prompt tab)
        dialog.destroy()
        dialog = SettingsDialog(app, persistence=persistence, initial_tab=1)
        dialog.update_idletasks()
        assert dialog.notebook.index(dialog.notebook.select()) == 1

        # 3. Test prompt reset: must be empty, no invented editorial text
        dialog.txt_prompt.insert("1.0", "User custom prompt")
        dialog._update_prompt_status()
        assert "User custom prompt" in dialog.txt_prompt.get("1.0", "end")

        dialog._reset_prompt()
        assert dialog.txt_prompt.get("1.0", "end").strip() == ""
        assert "⚠ Kịch bản đang trống" in dialog.var_prompt_status.get()
        assert "mẫu gợi ý" not in dialog.var_prompt_status.get()

        dialog.destroy()
    finally:
        app.destroy()


def test_main_window_check_update_shows_embedded_settings_update_tab(tmp_path: Path):
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()
        app._on_check_update()
        assert app.current_page == "settings"
        assert app.settings_page.notebook.index(app.settings_page.notebook.select()) == 6
    finally:
        app.destroy()


def test_first_launch_no_automatic_import(tmp_path: Path):
    """Verify first launch creates no unauthenticated imports, empty prompt, and clean secrets."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    mgr = SettingsManager(persistence=persistence)
    settings = mgr.load()

    # Prompt is empty, no invented defaults
    assert settings.prompt == ""

    # Secret store is clean, no auto-imported keys
    secret_store = DPAPISecretStore(storage_root=tmp_path)
    assert secret_store.get_secret("gateway_api_key") is None
    assert secret_store.get_secret("voice_remote_api_key") is None

    # Settings file does not exist until saved
    settings_file = tmp_path / "settings" / "settings.json"
    assert not settings_file.exists()


def test_embedded_settings_sidebar_layout(tmp_path: Path):
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()
        dialog = SettingsDialog(app, persistence=persistence)
        dialog.update_idletasks()

        expected = {"Recap", "Highlight", "AI Gateway", "Voice", "Render and Output"}
        assert set(dialog._panes.keys()) == expected
        assert set(dialog._nav_buttons.keys()) == expected
        assert dialog.winfo_toplevel() is app
        assert hasattr(dialog, "settings_canvas")

        # Active pane starts at Recap
        assert dialog._current_pane == "Recap"
        assert str(dialog._nav_buttons["Recap"]["state"]) == "disabled"
        assert str(dialog._nav_buttons["AI Gateway"]["state"]) != "disabled"

        # Switch to AI Gateway
        dialog._show_pane("AI Gateway")
        assert dialog._current_pane == "AI Gateway"
        assert str(dialog._nav_buttons["AI Gateway"]["state"]) == "disabled"
        assert str(dialog._nav_buttons["Recap"]["state"]) != "disabled"

        # Switch to Voice: has VoiceStudio and Audio Mix subtabs
        dialog._show_pane("Voice")
        assert dialog._current_pane == "Voice"
        assert dialog.voice_notebook.index("end") == 2
        assert dialog.voice_notebook.tab(0, "text") == "VoiceStudio (Giọng đọc)"
        assert dialog.voice_notebook.tab(1, "text") == "Hòa âm (Audio Mix)"

        # Switch to Render and Output: has Render, Notifications, Update subtabs
        dialog._show_pane("Render and Output")
        assert dialog._current_pane == "Render and Output"
        assert dialog.render_notebook.index("end") == 3
        assert dialog.render_notebook.tab(0, "text") == "Xuất video (Render)"
        assert dialog.render_notebook.tab(1, "text") == "Thông báo (Notifications)"
        assert dialog.render_notebook.tab(2, "text") == "Cập nhật (Updates)"

        dialog.destroy()
    finally:
        app.destroy()


def test_settings_dialog_dual_model_and_reasoning_save_and_callbacks(tmp_path: Path):
    """Verify SettingsDialog edits, saves, and invokes callbacks with dual Sub/Prime models and reasoning."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()
        mock_saved_cb = MagicMock()
        dialog = SettingsDialog(app, persistence=persistence, on_saved=mock_saved_cb)
        dialog.update_idletasks()

        # Update dual model and reasoning fields
        dialog.var_gw_sub_model.set("ag/custom-sub-v1")
        dialog.var_gw_sub_reasoning.set("high")
        dialog.var_scanner_parallelism.set(5)
        dialog.var_scanner_chunk_ms.set(180000)
        dialog.var_vision_model.set("vision/custom-v1")
        dialog.var_vision_reasoning.set("medium")
        dialog.var_gw_prime_model.set("ag/custom-prime-v1")
        dialog.var_gw_prime_reasoning.set("low")

        dialog._on_save()

        # Invariant: on_saved callback received updated AppSettings
        mock_saved_cb.assert_called_once()
        saved_settings = mock_saved_cb.call_args[0][0]
        assert isinstance(saved_settings, AppSettings)
        assert saved_settings.gateway_sub_model == "ag/custom-sub-v1"
        assert saved_settings.gateway_sub_reasoning == "high"
        assert saved_settings.gateway_prime_model == "ag/custom-prime-v1"
        assert saved_settings.gateway_prime_reasoning == "low"
        assert saved_settings.scanner_model == "ag/custom-sub-v1"
        assert saved_settings.scanner_parallelism == 5
        assert saved_settings.scanner_chunk_duration_ms == 180000
        assert saved_settings.vision_model == "vision/custom-v1"
        assert saved_settings.finalizer_model == "ag/custom-prime-v1"
        assert saved_settings.planner_model == saved_settings.writer_model == "ag/custom-prime-v1"

        # Invariant: settings persisted to disk
        mgr = SettingsManager(persistence=persistence)
        loaded = mgr.load()
        assert loaded.gateway_sub_model == "ag/custom-sub-v1"
        assert loaded.gateway_sub_reasoning == "high"
        assert loaded.gateway_prime_model == "ag/custom-prime-v1"
        assert loaded.gateway_prime_reasoning == "low"
        assert loaded.vision_model == "vision/custom-v1"
        assert loaded.finalizer_model == "ag/custom-prime-v1"

        dialog.destroy()
    finally:
        app.destroy()
