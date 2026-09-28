"""Persisted unfinished projects remain visible and resumable after GUI restart."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from toolrecap_v4.discovery import SourceFingerprint
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.ui.main_window import MainWindow
from toolrecap_v4.workflow import ProjectStatus


def _save_project(persistence: ProjectPersistence, project_id: str, status: ProjectStatus, source: Path) -> None:
    persistence.save_project({
        "project_id": project_id,
        "project_name": "Shared Project Name",
        "status": status.value,
        "sources": [{"source_file": source.name, "fingerprint": {"path": str(source)}}],
        "timestamps": {"updated_at": f"2026-09-23T20:00:{len(project_id):02d}+07:00"},
    })


def test_restart_lists_every_unfinished_stage_and_resumes_selected_project(tmp_path: Path) -> None:
    first = ProjectPersistence(storage_root=tmp_path)
    source = tmp_path / "Season With Spaces" / "E01.mp4"
    source.parent.mkdir()
    source.write_bytes(b"synthetic source")
    unfinished = [status for status in ProjectStatus if status != ProjectStatus.COMPLETED]
    for index, status in enumerate(unfinished, 1):
        _save_project(first, f"saved-{index:02d}", status, source)
    _save_project(first, "completed-project", ProjectStatus.COMPLETED, source)
    before = {path.name: path.read_bytes() for path in first.projects_dir.glob("*.json")}

    restarted = ProjectPersistence(storage_root=tmp_path)
    assert len(restarted.list_projects()) == len(unfinished) + 1
    app = MainWindow(persistence=restarted)
    try:
        app.update_idletasks()
        app.update()
        assert app.resume_banner.winfo_manager() == "pack"
        assert app.resume_banner.master.winfo_manager() == "grid"
        assert app.resume_banner.winfo_ismapped()
        assert len(app._resume_project_ids) == len(unfinished)
        assert set(app._resume_project_ids.values()) == {f"saved-{index:02d}" for index in range(1, len(unfinished) + 1)}
        assert "completed-project" not in app._resume_project_ids.values()
        assert str(app.btn_resume_saved["state"]) == "normal"
        assert str(app.btn_resume["state"]) == "normal"

        selected_id = next(
            project_id for project_id in app._resume_project_ids.values()
            if restarted.load_project(project_id)["status"] == ProjectStatus.EVIDENCE_READY.value
        )
        display = next(label for label, project_id in app._resume_project_ids.items() if project_id == selected_id)
        app.cmb_resume_projects.set(display)
        app._on_select_resume_project()
        assert app.current_project_id == selected_id
        assert app.current_project_name == "Shared Project Name"
        assert app.tree.get_children()
        assert "đã lưu" in app.source_var.get()
        with patch.object(app.worker, "resume_project") as resume, patch.object(app.worker, "start_new_project") as start:
            app.btn_resume_saved.invoke()
            resume.assert_called_once()
            assert resume.call_args.kwargs["project_id"] == selected_id
            start.assert_not_called()
        assert {path.name: path.read_bytes() for path in first.projects_dir.glob("*.json")} == before
    finally:
        app.destroy()


def test_starting_same_source_selects_saved_project_instead_of_creating_duplicate(tmp_path: Path) -> None:
    persistence = ProjectPersistence(storage_root=tmp_path)
    source = tmp_path / "Different Show" / "S03" / "E01.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic source")
    _save_project(persistence, "existing-project", ProjectStatus.FAILED, source)
    app = MainWindow(persistence=ProjectPersistence(storage_root=tmp_path))
    try:
        app.discovered_sources = [SourceFingerprint(
            basename=source.name, path=str(source), size_bytes=source.stat().st_size,
            mtime_ns=source.stat().st_mtime_ns, sha256="synthetic-hash", extension=".mp4",
        )]
        app._refresh_queue_table()
        with patch.object(app.worker, "start_new_project") as start:
            app._on_start_project()
            start.assert_not_called()
        assert app.current_project_id == "existing-project"
        assert app.cmb_resume_projects.get() in app._resume_project_ids
        assert app._resume_project_ids[app.cmb_resume_projects.get()] == "existing-project"
        assert len(list(persistence.projects_dir.glob("*.json"))) == 1
    finally:
        app.destroy()
