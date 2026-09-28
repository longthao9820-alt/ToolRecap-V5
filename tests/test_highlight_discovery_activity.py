"""Regression coverage for incremental Highlight source discovery UI."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import sys

if sys.platform == "win32":
    tcl_path = os.path.join(sys.base_prefix, "tcl", "tcl8.6")
    tk_path = os.path.join(sys.base_prefix, "tcl", "tk8.6")
    if os.path.isdir(tcl_path) and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = tcl_path
    if os.path.isdir(tk_path) and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = tk_path

from toolrecap_v4.discovery import SourceFingerprint
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.progress import ActivityState
from toolrecap_v4.ui.main_window import MainWindow
from toolrecap_v4.ui.worker import SourceDiscoveryWorker, WorkerMessage


def _source(path: Path) -> SourceFingerprint:
    stat = path.stat()
    return SourceFingerprint(
        basename=path.name,
        path=str(path.resolve()),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256="",
        extension=path.suffix.lower(),
    )


def test_discovery_worker_emits_initial_total_and_each_ready_source(tmp_path: Path) -> None:
    season = tmp_path / "Season"
    season.mkdir()
    for name in ("Episode 1.mp4", "Episode 2.mkv"):
        (season / name).write_bytes(b"video")

    messages: queue.Queue[WorkerMessage] = queue.Queue()
    worker = SourceDiscoveryWorker(messages)
    token = worker.start_discovery(season, is_folder=True)
    assert worker.wait_until_idle(timeout=3.0)

    received: list[WorkerMessage] = []
    while not messages.empty():
        received.append(messages.get_nowait())

    progress = [message.data for message in received if message.kind == "discovery_progress"]
    assert progress[0]["token"] == token
    assert (progress[0]["current"], progress[0]["total"]) == (0, 2)
    assert [item["source"].basename for item in progress[1:]] == ["Episode 1.mp4", "Episode 2.mkv"]
    assert received[-1].kind == "discovery_completed"


def test_highlight_scan_populates_rows_and_real_time_activity_incrementally(tmp_path: Path) -> None:
    persistence = ProjectPersistence(storage_root=tmp_path / "state")
    app = MainWindow(persistence=persistence)
    season = tmp_path / "Season"
    season.mkdir()
    first_path = season / "Episode 1.mp4"
    second_path = season / "Episode 2.mp4"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    first = _source(first_path)
    second = _source(second_path)

    try:
        app._discovery_mode = "highlight"
        app._active_discovery_token = 42
        app._set_discovery_busy(True, "Đang quét thư mục: Season")
        app._begin_discovery_activity(season, is_folder=True)

        app._handle_worker_message(WorkerMessage("discovery_progress", {
            "token": 42, "current": 0, "total": 2, "name": "Season",
        }))
        app._handle_worker_message(WorkerMessage("discovery_progress", {
            "token": 42, "current": 1, "total": 2, "name": "Season", "source": first,
        }))

        assert [item.basename for item in app.highlight_sources] == ["Episode 1.mp4"]
        assert len(app.highlight_tree.get_children()) == 1
        assert app.activity_state_var.get() == ActivityState.LOCAL_PROCESSING.value
        assert app.activity_progress_var.get() == "1 / 2 videos — 50%"
        assert "Found Episode 1.mp4" in app.activity_log.get("1.0", "end")
        assert str(app.btn_start_highlight.cget("state")) == "disabled"

        # A late saved-project reconstruction must not replace the active scan.
        app._handle_worker_message(WorkerMessage("activity", {
            "reconstructed": True,
            "state": ActivityState.IDLE.value,
            "activity_text": "Saved Highlight project is ready to continue.",
        }))
        assert app.activity_state_var.get() == ActivityState.LOCAL_PROCESSING.value

        app._handle_worker_message(WorkerMessage("discovery_completed", {
            "token": 42,
            "kind": "folder",
            "name": "Season",
            "path": str(season),
            "sources": [first, second],
        }))

        assert [item.basename for item in app.highlight_sources] == ["Episode 1.mp4", "Episode 2.mp4"]
        assert len(app.highlight_tree.get_children()) == 2
        assert app.activity_state_var.get() == ActivityState.COMPLETE.value
        assert app.activity_progress_var.get() == "2 / 2 videos — 100%"
        assert "2 video(s) ready" in app.activity_text_var.get()
        assert str(app.btn_start_highlight.cget("state")) == "normal"
    finally:
        app.destroy()
