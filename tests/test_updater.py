"""Acceptance and security tests for ToolRecap V4 Updater."""

import json
import os
from pathlib import Path
import sys
import time
import zipfile
from unittest.mock import MagicMock, PropertyMock, patch
import pytest

from toolrecap_v4.__version__ import __version__
from toolrecap_v4.errors import (
    ChecksumMismatchError,
    MaliciousArchiveError,
    PackageValidationError,
    UpdateApplyError,
    UpdateInProgressError,
)
from toolrecap_v4.persistence import ProjectPersistence, get_storage_root
from toolrecap_v4.settings import AppSettings, SettingsManager
from toolrecap_v4.ui.main_window import MainWindow
from toolrecap_v4.ui.settings_dialog import SettingsDialog
from toolrecap_v4.updater import (
    DEFAULT_MAX_FILE_COUNT,
    DEFAULT_MAX_RATIO,
    DEFAULT_MAX_UNCOMPRESSED_BYTES,
    SemVer,
    UpdateCheckResult,
    UpdateManager,
    apply_staged_update_with_rollback,
    calculate_sha256,
    parse_sha256_content,
    safe_extract_zip,
    validate_package,
    verify_checksum,
)


def _write_valid_package(package_dir: Path, version: str = "4.1.0") -> Path:
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "ToolRecapV4.exe").write_bytes(b"MZfakeexe")
    (package_dir / "_internal").mkdir(exist_ok=True)
    (package_dir / "_internal" / "base_library.zip").write_bytes(b"python standard library")
    (package_dir / "_internal" / "python312.dll").write_bytes(b"python runtime")
    (package_dir / "ffmpeg.exe").write_bytes(b"MZfakeffmpeg")
    (package_dir / "ffprobe.exe").write_bytes(b"MZfakeffprobe")
    (package_dir / "FFMPEG_LICENSE.txt").write_text("license", encoding="utf-8")
    schema_dir = package_dir / "schemas"
    schema_dir.mkdir(exist_ok=True)
    schema = Path(__file__).parents[1] / "toolrecap_v4" / "schemas" / "recap_v3_schema.json"
    (schema_dir / schema.name).write_bytes(schema.read_bytes())
    marker = {
        "package_format": "toolrecap-portable-v1",
        "schema_version": "3.0",
        "version": version,
        "required_files": [
            "ToolRecapV4.exe", "ffmpeg.exe", "ffprobe.exe",
            "FFMPEG_LICENSE.txt", "schemas/recap_v3_schema.json",
            "_internal/base_library.zip", "_internal/python312.dll",
        ],
    }
    (package_dir / "package_marker.json").write_text(json.dumps(marker), encoding="utf-8")
    return package_dir


# =============================================================================
# 1. SEMVER TESTS (Single version source & full SemVer 2.0.0 precedence)
# =============================================================================

def test_semver_single_version_source():
    """Verify single source of truth __version__ parses as valid SemVer."""
    cur = SemVer.parse(__version__)
    assert cur == SemVer.parse("5.0.4")
    assert cur.major == 5
    assert cur.minor == 0
    assert cur.patch == 4


def test_semver_comparisons():
    """Verify SemVer 2.0.0 comparison rules."""
    v300 = SemVer.parse("3.0.0")
    v301 = SemVer.parse("3.0.1")
    v310 = SemVer.parse("3.1.0")
    v400 = SemVer.parse("4.0.0")
    v299 = SemVer.parse("2.9.9")

    # Basic ordering
    assert v300 == SemVer.parse("v3.0.0")
    assert v300 < v301
    assert v301 > v300
    assert v300 < v310
    assert v310 < v400
    assert v300 > v299

    # Pre-release precedence
    v300_alpha = SemVer.parse("3.0.0-alpha")
    v300_alpha1 = SemVer.parse("3.0.0-alpha.1")
    v300_beta = SemVer.parse("3.0.0-beta")
    v300_rc1 = SemVer.parse("3.0.0-rc.1")
    v300_rc2 = SemVer.parse("3.0.0-rc.2")

    # Normal version has higher precedence than prerelease
    assert v300 > v300_rc2
    assert v300 > v300_alpha

    # Prerelease comparisons
    assert v300_alpha < v300_alpha1
    assert v300_alpha < v300_beta
    assert v300_beta < v300_rc1
    assert v300_rc1 < v300_rc2

    # Build metadata ignored
    v300_build1 = SemVer.parse("3.0.0+20260921")
    v300_build2 = SemVer.parse("3.0.0+20260922")
    assert v300_build1 == v300_build2
    assert v300_build1 == v300


def test_semver_invalid():
    """Verify invalid version strings are rejected."""
    for invalid in ["not-a-version", "1.0", "1.0.0.0", "", "v"]:
        with pytest.raises(ValueError):
            SemVer.parse(invalid)


# =============================================================================
# 2. CHECKSUM VERIFICATION TESTS
# =============================================================================

def test_checksum_verification_and_mismatch(tmp_path: Path):
    """Verify SHA256 calculation, matching, and mismatch rejection."""
    test_file = tmp_path / "test.zip"
    content = b"ToolRecapV4 test payload"
    test_file.write_bytes(content)

    import hashlib
    expected_hash = hashlib.sha256(content).hexdigest()

    # Valid check
    assert verify_checksum(test_file, expected_hash)
    assert verify_checksum(test_file, expected_hash.upper())  # Case-insensitive

    # Mismatch check
    wrong_hash = "0" * 64
    assert not verify_checksum(test_file, wrong_hash)

    # Parser handles multiple formats
    assert parse_sha256_content(f"{expected_hash}  test.zip\n") == expected_hash
    assert parse_sha256_content(f"{expected_hash} *test.zip\n") == expected_hash
    assert parse_sha256_content(f"{expected_hash}\n") == expected_hash

    with pytest.raises(ValueError):
        parse_sha256_content("invalid hash content")


# =============================================================================
# 3. MALICIOUS ARCHIVE TESTS (Path traversal, Symlinks, Zip bombs)
# =============================================================================

def test_malicious_archive_path_traversal(tmp_path: Path):
    """Verify archive with path traversal (../ or absolute path) is rejected."""
    bad_zip = tmp_path / "traversal.zip"
    extract_target = tmp_path / "extracted"
    extract_target.mkdir()

    with zipfile.ZipFile(bad_zip, "w") as zf:
        # Write traversal entry
        zf.writestr("../../evil.txt", b"malicious content")

    with pytest.raises(MaliciousArchiveError, match="Path traversal"):
        safe_extract_zip(bad_zip, extract_target)

    # Target directory should not contain evil.txt
    assert not (tmp_path / "evil.txt").exists()


def test_malicious_archive_symlink(tmp_path: Path):
    """Verify archive with symlink entry is rejected."""
    symlink_zip = tmp_path / "symlink.zip"
    extract_target = tmp_path / "extracted"
    extract_target.mkdir()

    with zipfile.ZipFile(symlink_zip, "w") as zf:
        zinfo = zipfile.ZipInfo("link_target")
        # Unix symlink attribute: 0o120000 << 16
        zinfo.external_attr = 0o120777 << 16
        zf.writestr(zinfo, b"/etc/passwd")

    with pytest.raises(MaliciousArchiveError, match="Symlink detected"):
        safe_extract_zip(symlink_zip, extract_target)


def test_malicious_archive_zip_bomb_size_and_ratio(tmp_path: Path):
    """Verify archive exceeding max uncompressed size or ratio is rejected."""
    bomb_zip = tmp_path / "bomb.zip"
    extract_target = tmp_path / "extracted"
    extract_target.mkdir()

    with zipfile.ZipFile(bomb_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 2 MB of zeros compresses to a tiny size
        zf.writestr("huge.bin", b"\x00" * (2 * 1024 * 1024))

    # Test max_uncompressed_bytes constraint
    with pytest.raises(MaliciousArchiveError, match="Archive uncompressed size.*exceeds limit"):
        safe_extract_zip(bomb_zip, extract_target, max_uncompressed_bytes=1024 * 1024)

    # Test file count limit
    count_zip = tmp_path / "count.zip"
    with zipfile.ZipFile(count_zip, "w") as zf:
        for i in range(15):
            zf.writestr(f"file_{i}.txt", b"a")

    with pytest.raises(MaliciousArchiveError, match="exceeding limit of 10"):
        safe_extract_zip(count_zip, extract_target, max_file_count=10)


# =============================================================================
# 4. PACKAGE VALIDATION TESTS
# =============================================================================

def test_package_validation(tmp_path: Path):
    """Verify package validator checks exact one-folder runtime resources."""
    pkg_dir = tmp_path / "package"
    pkg_dir.mkdir()

    # Initially empty
    with pytest.raises(PackageValidationError, match="missing main executable"):
        validate_package(pkg_dir)

    _write_valid_package(pkg_dir)
    marker = pkg_dir / "package_marker.json"
    marker_data = json.loads(marker.read_text(encoding="utf-8"))
    marker_data["version"] = "4.0.1"
    marker.write_text(json.dumps(marker_data), encoding="utf-8")
    with pytest.raises(PackageValidationError, match="does not match expected '4.1.0'"):
        validate_package(pkg_dir, expected_version="4.1.0")
    marker_data["version"] = "4.1.0"
    marker.write_text(json.dumps(marker_data), encoding="utf-8")
    validate_package(pkg_dir, expected_version="4.1.0")  # Should pass without error


# =============================================================================
# 5. ACTIVE GUARD TESTS
# =============================================================================

def test_active_guard_prevents_concurrent_runs(tmp_path: Path):
    """Verify active guard prevents simultaneous update operations."""
    mgr = UpdateManager(storage_root=tmp_path)

    with mgr.active_guard():
        with pytest.raises(UpdateInProgressError, match="already in progress"):
            with mgr.active_guard():
                pass


def test_settings_update_manager_observes_active_workflow(tmp_path: Path):
    """The real Settings dialog connects the project worker to the staging guard."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        dialog = SettingsDialog(app, persistence=persistence)
        try:
            assert dialog.update_manager.activity_probe is not None
            with patch.object(type(app.worker), "is_running", new_callable=PropertyMock, return_value=True):
                assert dialog.update_manager.activity_probe() is True
                with pytest.raises(UpdateInProgressError, match="workflow is active"):
                    dialog.update_manager.download_and_stage(UpdateCheckResult(
                        status="update_available", current_version="4.0.0", latest_version="4.1.0",
                        zip_url="https://example.test/package.zip",
                        sha256_url="https://example.test/package.sha256.txt",
                    ))
        finally:
            dialog.destroy()
    finally:
        app.destroy()


# =============================================================================
# 6. ROLLBACK & USERDATA PRESERVATION TESTS
# =============================================================================

def test_helper_rollback_on_failed_startup_handshake(tmp_path: Path):
    """Verify helper rolls back all replaced files and deletes new files if startup fails."""
    target_dir = tmp_path / "app_install"
    target_dir.mkdir()
    staged_dir = tmp_path / "staged"
    _write_valid_package(staged_dir)
    backup_dir = tmp_path / "backup"

    # Original files in app_install
    orig_exe = target_dir / "ToolRecapV4.exe"
    orig_exe.write_text("ORIGINAL_EXE_V4.0.0", encoding="utf-8")
    user_file = target_dir / "user_custom_data.txt"
    user_file.write_text("PRESERVE_MY_USER_DATA", encoding="utf-8")

    # Staged update files (new exe + new file)
    new_exe = staged_dir / "ToolRecapV4.exe"
    new_exe.write_text("NEW_EXE_V4.1.0", encoding="utf-8")
    new_extra = staged_dir / "new_library.dll"
    new_extra.write_text("NEW_LIBRARY_BYTES", encoding="utf-8")

    # Mock executable runner that fails handshake (exit code 1)
    mock_runner = tmp_path / "fail_handshake.bat"
    mock_runner.write_text("@echo off\nexit /b 1\n", encoding="utf-8")

    with pytest.raises(UpdateApplyError, match="Startup handshake failed"):
        apply_staged_update_with_rollback(
            target_dir=target_dir,
            staged_dir=staged_dir,
            backup_dir=backup_dir,
            executable=mock_runner,
            handshake_args=[],
            timeout_wait_exit=1.0,
            handshake_timeout=5.0,
        )

    # Verify ROLLBACK occurred:
    # 1. Original exe restored
    assert orig_exe.read_text(encoding="utf-8") == "ORIGINAL_EXE_V4.0.0"
    # 2. User data preserved
    assert user_file.read_text(encoding="utf-8") == "PRESERVE_MY_USER_DATA"
    # 3. Newly added file removed
    assert not (target_dir / "new_library.dll").exists()
    # 4. Backup directory cleaned up
    assert not backup_dir.exists()


def test_helper_success_preserves_unknown_files(tmp_path: Path):
    """Verify successful update replaces target files while preserving unknown user files."""
    target_dir = tmp_path / "app_install"
    target_dir.mkdir()
    staged_dir = tmp_path / "staged"
    _write_valid_package(staged_dir)
    backup_dir = tmp_path / "backup"

    # Original files
    orig_exe = target_dir / "ToolRecapV4.exe"
    orig_exe.write_text("ORIGINAL_EXE", encoding="utf-8")
    user_plugin = target_dir / "plugins" / "custom_plugin.dll"
    user_plugin.parent.mkdir(parents=True, exist_ok=True)
    user_plugin.write_text("USER_PLUGIN", encoding="utf-8")

    # Staged files
    (staged_dir / "ToolRecapV4.exe").write_text("UPDATED_EXE", encoding="utf-8")
    (staged_dir / "new_file.txt").write_text("NEW_FILE", encoding="utf-8")

    # Mock executable runner that succeeds (exit code 0)
    mock_runner = tmp_path / "success_handshake.bat"
    mock_runner.write_text("@echo off\nexit /b 0\n", encoding="utf-8")

    success = apply_staged_update_with_rollback(
        target_dir=target_dir,
        staged_dir=staged_dir,
        backup_dir=backup_dir,
        executable=mock_runner,
        handshake_args=[],
        timeout_wait_exit=1.0,
        handshake_timeout=5.0,
    )
    assert success is True

    # Check updated files
    assert orig_exe.read_text(encoding="utf-8") == "UPDATED_EXE"
    assert (target_dir / "new_file.txt").read_text(encoding="utf-8") == "NEW_FILE"

    # Invariant: preserve unknown files!
    assert user_plugin.exists()
    assert user_plugin.read_text(encoding="utf-8") == "USER_PLUGIN"


def test_helper_rejects_localappdata_and_outputs(tmp_path: Path):
    """Verify helper refuses to apply updates to LOCALAPPDATA or output folders."""
    storage_root = get_storage_root(tmp_path / "storage")
    staged_dir = tmp_path / "staged"
    _write_valid_package(staged_dir)
    backup_dir = tmp_path / "backup"

    # Target is LOCALAPPDATA
    with pytest.raises(UpdateApplyError, match="must not be in LOCALAPPDATA"):
        apply_staged_update_with_rollback(
            target_dir=storage_root,
            storage_root=storage_root,
            staged_dir=staged_dir,
            backup_dir=backup_dir,
        )

    # Target is outputs directory
    out_dir = tmp_path / "output_recap"
    out_dir.mkdir()
    with pytest.raises(UpdateApplyError, match="must not be in output publication directory"):
        apply_staged_update_with_rollback(
            target_dir=out_dir,
            staged_dir=staged_dir,
            backup_dir=backup_dir,
            output_dirs=[out_dir],
        )


# =============================================================================
# 7. MISSING REPO / NOT CONFIGURED TESTS
# =============================================================================

def test_missing_repo_not_configured(tmp_path: Path):
    """Verify empty/missing repository reports not configured with zero fake updates."""
    mgr = UpdateManager(storage_root=tmp_path)

    # Empty string
    res = mgr.check_for_updates("")
    assert res.status == "not_configured"
    assert "Chưa cấu hình" in res.message

    # Whitespace string
    res2 = mgr.check_for_updates("   ")
    assert res2.status == "not_configured"


# =============================================================================
# 8. UI WIRING & WORKFLOW TESTS
# =============================================================================

def test_settings_dialog_update_tab_wiring(tmp_path: Path):
    """Verify SettingsDialog update tab controls wired to backend."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()
        dialog = SettingsDialog(app, persistence=persistence)
        dialog.update_idletasks()

        # Update tab exists (Tab index 6 is 7th tab)
        assert dialog.var_update_repo.get() == "longthao9820-alt/ToolRecap-V5"
        assert "Sẵn sàng kiểm tra bản cập nhật" in dialog.lbl_update_status.cget("text")

        # Check button with empty repo triggers not configured message
        dialog.var_update_repo.set("")
        with patch("toolrecap_v4.ui.settings_dialog.messagebox.showinfo") as mock_info:
            dialog._check_update()
            mock_info.assert_not_called()
            assert "Not configured" in dialog.lbl_update_status.cget("text")
            assert "Not configured" in dialog.lbl_update_result.cget("text")
            assert "Chưa cấu hình" in dialog.lbl_update_result.cget("text")

        # Configure repository
        dialog.var_update_repo.set("myorg/myrepo")

        # Mock UpdateManager check_for_updates returning update_available
        fake_result = UpdateCheckResult(
            status="update_available",
            current_version="4.0.0",
            latest_version="4.1.0",
            zip_url="https://example.com/ToolRecapV4-v4.1.0-windows-portable.zip",
            sha256_url="https://example.com/ToolRecapV4-v4.1.0-windows-portable.zip.sha256.txt",
            zip_name="ToolRecapV4-v4.1.0-windows-portable.zip",
            sha256_name="ToolRecapV4-v4.1.0-windows-portable.zip.sha256.txt",
            message="Có bản cập nhật mới: v4.1.0!",
        )

        with patch.object(dialog.update_manager, "check_for_updates", return_value=fake_result):
            dialog._check_update()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and dialog._last_check_result is not fake_result:
                app.update()
                time.sleep(0.01)
            assert "Có bản cập nhật mới v4.1.0" in dialog.lbl_update_status.cget("text")
            assert str(dialog.btn_download_update["state"]) == "normal"

        # Mock download_and_stage
        fake_staged_path = tmp_path / "staged_v4.1.0"
        fake_staged_path.mkdir()
        with patch.object(dialog.update_manager, "download_and_stage", return_value=fake_staged_path):
            dialog._download_update()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and dialog._staged_update_path != fake_staged_path:
                app.update()
                time.sleep(0.01)
            assert "Đã tải và xác thực hoàn tất" in dialog.lbl_update_status.cget("text")
            assert str(dialog.btn_apply_update["state"]) == "normal"

        # Test Apply Update: User CANCELS confirmation
        with patch("toolrecap_v4.ui.settings_dialog.messagebox.askyesno", return_value=False):
            with patch.object(dialog.update_manager, "launch_apply_helper") as mock_helper:
                dialog._apply_update()
                mock_helper.assert_not_called()

        # Test Apply Update: User CONFIRMS
        with patch("toolrecap_v4.ui.settings_dialog.messagebox.askyesno", return_value=True):
            with patch.object(dialog.update_manager, "launch_apply_helper") as mock_helper, patch.object(dialog, "destroy"), patch.object(app, "destroy"):
                dialog._apply_update()
                mock_helper.assert_called_once()

        # Test Save persists update_repo
        dialog.var_update_repo.set("savedorg/savedrepo")
        dialog.var_gw_sub_model.set("scanner-route")
        dialog.var_gw_prime_model.set("finalizer-route")
        dialog._on_save()

        mgr = SettingsManager(persistence=persistence)
        saved_settings = mgr.load()
        assert saved_settings.update_repo == "savedorg/savedrepo"

        dialog.destroy()
    finally:
        app.destroy()
