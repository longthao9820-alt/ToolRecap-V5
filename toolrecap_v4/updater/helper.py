"""Separate finite helper for atomic update application and rollback."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import List, Optional

from toolrecap_v4.errors import UpdateApplyError
from toolrecap_v4.persistence import get_storage_root
from toolrecap_v4.updater.validator import validate_package


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def is_process_running(pid: int) -> bool:
    """Check if process with given PID is still running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(
                SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                wintypes.DWORD(pid),
            )
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    STILL_ACTIVE = 259
                    return exit_code.value == STILL_ACTIVE
                return False
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def wait_for_process_exit(pid: int, timeout_seconds: float = 15.0) -> bool:
    """Finite wait for process exit. Returns True if exited, False if timed out."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        if not is_process_running(pid):
            return True
        time.sleep(0.1)
    return not is_process_running(pid)


def apply_staged_update_with_rollback(
    target_dir: Path,
    staged_dir: Path,
    backup_dir: Path,
    parent_pid: Optional[int] = None,
    executable: Optional[Path | str] = None,
    timeout_wait_exit: float = 15.0,
    handshake_timeout: float = 10.0,
    handshake_args: Optional[List[str]] = None,
    output_dirs: Optional[List[Path]] = None,
    storage_root: Optional[Path | str] = None,
) -> bool:
    """Apply staged update to target_dir with finite process wait and rollback.
    
    Invariants enforced:
    - Never touches LOCALAPPDATA storage root or output directories.
    - Waits for parent app process exit up to timeout_wait_exit.
    - Backs up ONLY package files that will be overwritten.
    - Preserves unknown files in target_dir.
    - Replaces files and performs startup handshake.
    - Rolls back to backup if startup handshake fails.
    - No unsafe shell interpolation (safe argument list).
    """
    if handshake_args is None:
        handshake_args = ["--update-handshake"]

    target_dir = Path(target_dir).resolve()
    staged_dir = Path(staged_dir).resolve()
    backup_dir = Path(backup_dir).resolve()

    if not target_dir.is_dir():
        raise UpdateApplyError(f"Target directory does not exist: {target_dir}")
    if not staged_dir.is_dir():
        raise UpdateApplyError(f"Staged directory does not exist: {staged_dir}")
    if _paths_overlap(target_dir, staged_dir) or _paths_overlap(target_dir, backup_dir) or _paths_overlap(staged_dir, backup_dir):
        raise UpdateApplyError("Target, staged, and backup directories must be separate")

    try:
        validate_package(staged_dir)
    except Exception as exc:
        raise UpdateApplyError(f"Staged package validation failed: {exc}") from exc

    # Invariant: Never touch LOCALAPPDATA or user output directories
    roots_to_check = {get_storage_root().resolve()}
    if storage_root:
        roots_to_check.add(Path(storage_root).resolve())

    for s_root in roots_to_check:
        if _paths_overlap(target_dir, s_root):
            raise UpdateApplyError("Target directory must not be in LOCALAPPDATA storage root")

    if output_dirs:
        for out_dir in output_dirs:
            out_resolved = Path(out_dir).resolve()
            if _paths_overlap(target_dir, out_resolved):
                raise UpdateApplyError("Target directory must not be in output publication directory")

    # 1. Finite wait for parent app exit
    if parent_pid is not None and parent_pid > 0:
        exited = wait_for_process_exit(parent_pid, timeout_seconds=timeout_wait_exit)
        if not exited:
            raise UpdateApplyError(
                f"Parent process {parent_pid} did not exit within {timeout_wait_exit} seconds; aborting update"
            )

    # 2. Identify all package files in staged_dir
    staged_files: list[Path] = [p for p in staged_dir.rglob("*") if p.is_file()]
    if not staged_files:
        raise UpdateApplyError(f"No files found in staged directory: {staged_dir}")

    relative_files = [p.relative_to(staged_dir) for p in staged_files]

    # 3. Backup ONLY package files that will be replaced
    shutil.rmtree(backup_dir, ignore_errors=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up_rel_paths: set[Path] = set()

    try:
        for rel in relative_files:
            target_file = target_dir / rel
            if target_file.exists():
                dst_backup = backup_dir / rel
                dst_backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target_file, dst_backup)
                backed_up_rel_paths.add(rel)

        # 4. Replace files in target_dir
        for rel in relative_files:
            src = staged_dir / rel
            dst = target_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # 5. Startup handshake
        if executable is not None:
            exe_path = Path(executable)
            if not exe_path.is_absolute():
                exe_path = target_dir / executable

            if exe_path.is_file():
                cmd = [str(exe_path)] + handshake_args
                try:
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=handshake_timeout,
                        check=False,
                    )
                    if proc.returncode != 0:
                        raise UpdateApplyError(
                            f"Startup handshake failed with exit code {proc.returncode}: {proc.stderr.decode(errors='replace')[:200]}"
                        )
                except subprocess.TimeoutExpired as e:
                    raise UpdateApplyError(
                        f"Startup handshake timed out after {handshake_timeout} seconds"
                    ) from e

        # Handshake succeeded! Clean up backup
        shutil.rmtree(backup_dir, ignore_errors=True)
        shutil.rmtree(staged_dir, ignore_errors=True)
        return True

    except Exception as e:
        # 6. ROLLBACK: Restore original files and delete new files
        for rel in relative_files:
            target_file = target_dir / rel
            if rel in backed_up_rel_paths:
                backup_file = backup_dir / rel
                if backup_file.is_file():
                    shutil.copy2(backup_file, target_file)
            else:
                # Newly added file not present before update; remove it
                if target_file.is_file():
                    target_file.unlink(missing_ok=True)

        shutil.rmtree(backup_dir, ignore_errors=True)
        shutil.rmtree(staged_dir, ignore_errors=True)
        if isinstance(e, UpdateApplyError):
            raise
        raise UpdateApplyError(f"Update application failed and was rolled back: {e}") from e


def main() -> None:
    """CLI entrypoint for standalone update helper process."""
    parser = argparse.ArgumentParser(description="ToolRecap V5 Update Helper")
    parser.add_argument("--update-helper", action="store_true", help="Flag for helper mode")
    parser.add_argument("--target-dir", required=True, help="Application install directory")
    parser.add_argument("--staged-dir", required=True, help="Directory with staged update files")
    parser.add_argument("--backup-dir", required=True, help="Directory to store backup files")
    parser.add_argument("--parent-pid", type=int, default=None, help="PID of application to wait for")
    parser.add_argument("--executable", default=None, help="Executable path for handshake/restart")
    parser.add_argument("--storage-root", default=None, help="Storage root directory to protect")
    parser.add_argument("--timeout", type=float, default=15.0, help="Wait timeout for parent process")
    parser.add_argument("--handshake-timeout", type=float, default=10.0, help="Wait timeout for startup handshake")

    args = parser.parse_args()

    try:
        apply_staged_update_with_rollback(
            target_dir=Path(args.target_dir),
            staged_dir=Path(args.staged_dir),
            backup_dir=Path(args.backup_dir),
            parent_pid=args.parent_pid,
            executable=args.executable,
            timeout_wait_exit=args.timeout,
            handshake_timeout=args.handshake_timeout,
            storage_root=Path(args.storage_root) if args.storage_root else None,
        )
        print("TOOLRECAP_V4_UPDATE_SUCCESS")
        sys.exit(0)
    except Exception as e:
        print(f"TOOLRECAP_V4_UPDATE_FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
