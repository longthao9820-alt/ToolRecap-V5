from pathlib import Path
import threading

from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.ui.main_window import MainWindow
from toolrecap_v4.ui.settings_dialog import SettingsDialog
from toolrecap_v4.ui.workspace import WorkspacePage


def _owned_toplevels(app: MainWindow) -> list[str]:
    result = []
    for child in app.tk.call("winfo", "children", "."):
        if str(app.tk.call("winfo", "class", child)).lower() == "toplevel":
            result.append(str(child))
    return result


def test_settings_is_embedded_and_navigation_keeps_one_toplevel(tmp_path: Path) -> None:
    app = MainWindow(persistence=ProjectPersistence(tmp_path))
    try:
        assert isinstance(app.settings_page, SettingsDialog)
        assert app.settings_page.winfo_toplevel() is app
        assert _owned_toplevels(app) == []
        for _ in range(20):
            for page in ("recap", "settings", "highlight", "settings", "recap"):
                app._show_page(page)
                app.update_idletasks()
                assert app.current_page == page
                assert _owned_toplevels(app) == []
        assert str(app.tk.call("grab", "current")) in {"", "()"}
    finally:
        app.destroy()


def test_page_state_and_settings_revert_are_preserved(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.highlight_prompt_text.delete("1.0", "end")
        app.highlight_prompt_text.insert("1.0", "find every worthwhile character moment")
        app._show_page("settings")
        app.settings_page.var_gw_endpoint.set("https://unsaved.example")
        app._show_page("highlight")
        assert app.highlight_prompt_text.get("1.0", "end-1c") == "find every worthwhile character moment"
        app._show_page("settings")
        assert app.settings_page.var_gw_endpoint.get() == "https://unsaved.example"
        app.settings_page._on_close()
        assert app.settings_page.var_gw_endpoint.get() == app.settings_manager.load().gateway_endpoint
        assert app.winfo_exists()
    finally:
        app.destroy()


def test_settings_validation_is_inline_and_async_queue_does_not_block_navigation(tmp_path: Path) -> None:
    app = MainWindow(persistence=ProjectPersistence(tmp_path))
    try:
        page = app.settings_page
        page.var_gw_endpoint.set("invalid://endpoint")
        page._on_save()
        assert page.lbl_status.cget("text").startswith("⚠")
        calls: list[str] = []
        worker = threading.Thread(target=lambda: page._post_ui(lambda: calls.append("done")))
        worker.start(); worker.join(timeout=1)
        assert calls == []
        app._show_page("highlight")
        assert app.current_page == "highlight"
        page._poll_ui_queue()
        assert calls == ["done"]
    finally:
        app.destroy()


def test_recap_and_highlight_use_same_workspace_component_and_geometry(tmp_path: Path) -> None:
    app = MainWindow(persistence=ProjectPersistence(tmp_path))
    try:
        recap, highlight = app.recap_page, app.highlight_page
        assert isinstance(recap, WorkspacePage) and isinstance(highlight, WorkspacePage)
        assert type(recap.source_panel) is type(highlight.source_panel)
        assert type(recap.prompt_panel) is type(highlight.prompt_panel)
        assert type(recap.queue_panel) is type(highlight.queue_panel)
        assert type(recap.action_toolbar) is type(highlight.action_toolbar)
        assert recap.tree["columns"] == highlight.tree["columns"]
        assert recap.prompt_text.cget("height") == highlight.prompt_text.cget("height")
        assert recap.select_file_button.cget("width") == highlight.select_file_button.cget("width")
        assert recap.resume_saved_button.cget("width") == highlight.resume_saved_button.cget("width")
        assert recap.stop_button.cget("text") == highlight.stop_button.cget("text") == "⏹ Stop"
        assert recap.open_output_button.cget("text") == highlight.open_output_button.cget("text")
        assert recap.saved_panel.winfo_class() == highlight.saved_panel.winfo_class()
        # The one authoritative status/recent-activity panel lives outside both pages.
        assert app.activity_panel.master is app.main_container
        assert app.activity_log.master.master is app.activity_panel
    finally:
        app.destroy()


def test_mode_specific_saved_project_panels_never_cross_resume(tmp_path: Path) -> None:
    persistence = ProjectPersistence(tmp_path)
    for project_id, mode in (("recap-1", "RECAP"), ("highlight-1", "HIGHLIGHT")):
        persistence.save_project({
            "project_id": project_id, "project_name": project_id, "project_mode": mode,
            "status": "failed", "sources": [], "timestamps": {},
        })
    app = MainWindow(persistence=persistence)
    try:
        assert set(app._resume_project_ids.values()) == {"recap-1"}
        assert set(app._highlight_resume_project_ids.values()) == {"highlight-1"}
        assert "highlight-1" not in app.recap_page.saved_combo.get()
        assert "recap-1" not in app.highlight_page.saved_combo.get()
    finally:
        app.destroy()
