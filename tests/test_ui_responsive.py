"""Responsive UI tests: slow I/O heartbeat, cancellation, stale replacement, and safe window close."""

import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

if sys.platform == "win32":
    tcl_path = os.path.join(sys.base_prefix, "tcl", "tcl8.6")
    tk_path = os.path.join(sys.base_prefix, "tcl", "tk8.6")
    if os.path.isdir(tcl_path) and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = tcl_path
    if os.path.isdir(tk_path) and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = tk_path

import tkinter as tk

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.discovery import SourceFingerprint, compute_file_fingerprint, discover_sources
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.ui.main_window import MainWindow
from toolrecap_v4.ui.worker import SourceDiscoveryWorker, WorkerMessage


def test_slow_io_ui_heartbeat(tmp_path: Path):
    """Demonstrate UI heartbeat ticks continuously while a scan is deliberately slow."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)

    # Prepare dummy files in directory
    scan_dir = tmp_path / "SlowSeason"
    scan_dir.mkdir()
    for i in range(1, 6):
        (scan_dir / f"Episode_{i:02d}.mp4").write_bytes(f"video_content_{i}".encode("utf-8"))

    # Mock discover_sources to simulate slow I/O with deliberate delays per file
    original_discover = discover_sources

    def slow_discover(directory, cancellation_token=None, compute_hash=False, progress_callback=None):
        results = []
        matching = sorted(Path(directory).glob("*.mp4"))
        total = len(matching)
        for idx, p in enumerate(matching, start=1):
            if cancellation_token:
                cancellation_token.check_cancelled()
            time.sleep(0.04)  # Deliberate I/O delay
            fp = compute_file_fingerprint(p, cancellation_token=cancellation_token, compute_hash=False)
            results.append(fp)
            if progress_callback:
                progress_callback(idx, total)
        return results

    try:
        app.update_idletasks()

        # Heartbeat tracker simulating main UI loop event processing
        heartbeat_ticks = 0
        progress_observed = []

        with patch("toolrecap_v4.ui.worker.discover_sources", side_effect=slow_discover):
            app._load_folder(scan_dir)

            # While discovery worker runs in background, main thread processes events & heartbeats
            start_t = time.time()
            while app.discovery_worker.is_running and time.time() - start_t < 4.0:
                heartbeat_ticks += 1
                app._poll_queue()
                # Paint pending geometry/redraw work without draining every
                # unrelated recurring Tk timer.  app.update() can consume the
                # entire deliberately short worker lifetime in one call on a
                # busy Windows CI desktop, making this timing assertion flaky.
                app.update_idletasks()
                cur_status = app.status_var.get()
                if "Đang kiểm tra" in cur_status and cur_status not in progress_observed:
                    progress_observed.append(cur_status)
                time.sleep(0.015)

            # Final queue flush
            app._poll_queue()
            app.update_idletasks()

        # Invariant: UI heartbeat was not blocked by background slow I/O
        assert heartbeat_ticks >= 5, f"Expected at least 5 heartbeat ticks, got {heartbeat_ticks}"

        # Invariant: Progress updates were observed on UI thread
        assert len(progress_observed) >= 1

        # Invariant: Discovery completed successfully and populated UI
        assert len(app.discovered_sources) == 5
        assert len(app.tree_sources.get_children()) == 5
        assert "5 video" in app.lbl_src_summary["text"]
        assert "Sẵn sàng" in app.status_var.get() or "Đã nạp" in app.status_var.get()
    finally:
        try:
            app.destroy()
        except Exception:
            pass


def test_discovery_cancel_responsive(tmp_path: Path):
    """Verify Stop button cooperatively cancels in-flight discovery and restores UI."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)

    scan_dir = tmp_path / "CancelSeason"
    scan_dir.mkdir()
    for i in range(1, 10):
        (scan_dir / f"Ep_{i}.mp4").write_bytes(b"data")

    # Deliberately slow discovery that yields control until cancelled
    def slow_discover_with_cancel(directory, cancellation_token=None, compute_hash=False, progress_callback=None):
        for idx in range(1, 100):
            if cancellation_token:
                cancellation_token.check_cancelled()
            time.sleep(0.05)
            if progress_callback:
                progress_callback(idx, 100)
        return []

    try:
        app.update_idletasks()

        with patch("toolrecap_v4.ui.worker.discover_sources", side_effect=slow_discover_with_cancel):
            app._load_folder(scan_dir)

            # Wait briefly until thread is confirmed running
            for _ in range(20):
                app._poll_queue()
                app.update()
                if app.discovery_worker.is_running:
                    break
                time.sleep(0.01)

            assert app.discovery_worker.is_running
            assert str(app.btn_stop["state"]) == "normal"
            assert str(app.btn_start["state"]) == "disabled"

            # User triggers Stop
            app._on_stop_project()

            # Process UI queue until cancellation completes
            start_t = time.time()
            while app.discovery_worker.is_running and time.time() - start_t < 2.0:
                app._poll_queue()
                app.update()
                time.sleep(0.01)

            app._poll_queue()
            app.update_idletasks()

        # Worker must have stopped
        assert not app.discovery_worker.is_running
        assert "Đã dừng tác vụ chọn nguồn" in app.status_var.get()
        assert str(app.btn_start["state"]) == "normal"
        assert str(app.btn_stop["state"]) == "disabled"
    finally:
        try:
            app.destroy()
        except Exception:
            pass


def test_discovery_stale_replacement(tmp_path: Path):
    """Verify rapid consecutive selections discard stale results from superseded tasks."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)

    folder_a = tmp_path / "SeasonA"
    folder_a.mkdir()
    (folder_a / "EpA1.mp4").write_bytes(b"A1")
    (folder_a / "EpA2.mp4").write_bytes(b"A2")

    folder_b = tmp_path / "SeasonB"
    folder_b.mkdir()
    (folder_b / "EpB1.mp4").write_bytes(b"B1")
    (folder_b / "EpB2.mp4").write_bytes(b"B2")
    (folder_b / "EpB3.mp4").write_bytes(b"B3")

    a_started = threading.Event()
    a_cancelled = threading.Event()
    b_completed = threading.Event()

    def mock_discover(directory, cancellation_token=None, compute_hash=False, progress_callback=None):
        dir_str = str(directory)
        if "SeasonA" in dir_str:
            if cancellation_token:
                cancellation_token.register_callback(a_cancelled.set)
            a_started.set()
            a_cancelled.wait(timeout=5.0)
            if cancellation_token:
                cancellation_token.check_cancelled()
            return [
                SourceFingerprint(basename="EpA1.mp4", path=str(folder_a / "EpA1.mp4"), size_bytes=2, mtime_ns=1, sha256="", extension=".mp4"),
                SourceFingerprint(basename="EpA2.mp4", path=str(folder_a / "EpA2.mp4"), size_bytes=2, mtime_ns=1, sha256="", extension=".mp4"),
            ]
        else:
            results = [
                SourceFingerprint(basename="EpB1.mp4", path=str(folder_b / "EpB1.mp4"), size_bytes=2, mtime_ns=1, sha256="", extension=".mp4"),
                SourceFingerprint(basename="EpB2.mp4", path=str(folder_b / "EpB2.mp4"), size_bytes=2, mtime_ns=1, sha256="", extension=".mp4"),
                SourceFingerprint(basename="EpB3.mp4", path=str(folder_b / "EpB3.mp4"), size_bytes=2, mtime_ns=1, sha256="", extension=".mp4"),
            ]
            b_completed.set()
            return results

    try:
        app.update_idletasks()

        with patch("toolrecap_v4.ui.worker.discover_sources", side_effect=mock_discover):
            # Start selection A
            app._load_folder(folder_a)
            token_a = app._active_discovery_token
            assert token_a > 0

            # Deterministic wait: ensure worker A has started before superseding
            assert a_started.wait(timeout=5.0)

            # Immediately supersede with selection B before A completes
            app._load_folder(folder_b)
            token_b = app._active_discovery_token
            assert token_b > token_a

            # Deterministic wait: ensure worker B has produced results
            assert b_completed.wait(timeout=5.0)

            # Drain queue until results processed
            deadline = time.time() + 5.0
            while len(app.discovered_sources) < 3 and time.time() < deadline:
                app._poll_queue()
                app.update()
                time.sleep(0.005)

            app._poll_queue()
            app.update_idletasks()

        # Invariant: Only results from token B are present
        assert len(app.discovered_sources) == 3
        basenames = [s.basename for s in app.discovered_sources]
        assert basenames == ["EpB1.mp4", "EpB2.mp4", "EpB3.mp4"]
        assert "EpA1.mp4" not in basenames
        assert len(app.tree_sources.get_children()) == 3
    finally:
        try:
            app.discovery_worker.cancel()
            app.discovery_worker.wait_until_idle(timeout=2.0)
        except Exception:
            pass
        try:
            app.destroy()
        except Exception:
            pass


def test_ui_close_safe_cancels_discovery_without_join(tmp_path: Path):
    """Verify _on_close cancels active discovery worker immediately without joining on UI thread."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)

    # Deliberate long running discovery
    scan_dir = tmp_path / "CloseSeason"
    scan_dir.mkdir()
    (scan_dir / "Video.mp4").write_bytes(b"data")

    def blocking_discover(directory, cancellation_token=None, compute_hash=False, progress_callback=None):
        for _ in range(100):
            if cancellation_token and cancellation_token.is_cancelled:
                return []
            time.sleep(0.05)
        return []

    try:
        app.update_idletasks()

        with patch("toolrecap_v4.ui.worker.discover_sources", side_effect=blocking_discover):
            app._load_folder(scan_dir)

            for _ in range(20):
                if app.discovery_worker.is_running:
                    break
                time.sleep(0.01)

            assert app.discovery_worker.is_running

            # Track if join was ever called on the worker thread
            worker_thread = app.discovery_worker._thread
            assert worker_thread is not None

            with patch.object(worker_thread, "join", side_effect=AssertionError("UI thread must NOT call join()!")):
                # Execute _on_close
                app._on_close()

            # Invariant: discovery cancellation was signalled
            token = app.discovery_worker._cancellation_token
            assert token is not None and token.is_cancelled
    finally:
        try:
            app.destroy()
        except Exception:
            pass


def test_ui_close_safe_with_workflow_running(tmp_path: Path):
    """Verify _on_close cancels both workflow and discovery when confirmed by user."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)

    try:
        app.update_idletasks()

        # Simulate running workflow worker and discovery worker with active threads
        mock_wf_thread = MagicMock()
        mock_wf_thread.is_alive.return_value = True
        app.worker._thread = mock_wf_thread
        app.worker._is_running = True
        app.worker._cancellation_token = CancellationToken()

        mock_disc_thread = MagicMock()
        mock_disc_thread.is_alive.return_value = True
        app.discovery_worker._thread = mock_disc_thread
        app.discovery_worker._is_running = True
        app.discovery_worker._cancellation_token = CancellationToken()

        assert app.worker.is_running
        assert app.discovery_worker.is_running

        # Case 1: User rejects confirmation dialog
        with patch("toolrecap_v4.ui.main_window.messagebox.askyesno", return_value=False):
            app._on_close()
            assert not app.worker._cancellation_token.is_cancelled
            assert not app.discovery_worker._cancellation_token.is_cancelled

        # Case 2: User confirms dialog
        with patch("toolrecap_v4.ui.main_window.messagebox.askyesno", return_value=True):
            with patch("toolrecap_v4.ui.main_window.safe_after") as mock_after:
                app._on_close()
                assert app.worker._cancellation_token.is_cancelled
                assert app.discovery_worker._cancellation_token.is_cancelled
                mock_after.assert_called_once_with(app, 1500, app.destroy)
    finally:
        try:
            app.destroy()
        except Exception:
            pass


def test_source_integrity_fast_stat_only(tmp_path: Path):
    """Verify source video files are probed via fast stat without payload mutation or full SHA-256."""
    video_file = tmp_path / "IntegrityTest.mp4"
    payload = b"Original bytes " * 1024
    video_file.write_bytes(payload)
    original_stat = video_file.stat()

    fp = compute_file_fingerprint(video_file, compute_hash=False)

    # Invariant: SHA-256 is deferred (empty string)
    assert fp.sha256 == ""
    assert fp.size_bytes == len(payload)
    assert fp.basename == "IntegrityTest.mp4"

    # Invariant: Source bytes untouched
    current_stat = video_file.stat()
    assert current_stat.st_size == original_stat.st_size
    assert video_file.read_bytes() == payload
