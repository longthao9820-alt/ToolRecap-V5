"""UpdateManager: Coordinates update checking, downloading, staging, and apply."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, Iterator, Optional

import httpx

from toolrecap_v4.__version__ import __version__
from toolrecap_v4.errors import (
    ChecksumMismatchError,
    UpdateApplyError,
    UpdateCheckError,
    UpdateInProgressError,
)
from toolrecap_v4.persistence import get_storage_root
from toolrecap_v4.updater.archive import parse_sha256_content, safe_extract_zip, verify_checksum
from toolrecap_v4.updater.semver import SemVer
from toolrecap_v4.updater.validator import validate_package


@dataclass
class UpdateCheckResult:
    """Outcome of checking for updates."""

    status: str  # "not_configured", "up_to_date", "update_available", "missing_assets", "error"
    current_version: str
    latest_version: Optional[str] = None
    zip_url: Optional[str] = None
    sha256_url: Optional[str] = None
    zip_name: Optional[str] = None
    sha256_name: Optional[str] = None
    release_name: Optional[str] = None
    release_body: Optional[str] = None
    message: str = ""


class UpdateManager:
    """Manages update discovery, safe staging in LOCALAPPDATA, and applying."""

    def __init__(
        self,
        storage_root: Optional[Path | str] = None,
        client: Optional[httpx.Client] = None,
        current_version: Optional[str] = None,
        activity_probe: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.storage_root = get_storage_root(storage_root)
        self.current_version = current_version or __version__
        self.client = client
        self.activity_probe = activity_probe
        self.staging_dir = self.storage_root / "updates" / "staging"
        self.backup_dir = self.storage_root / "updates" / "backup"

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._is_active = False

    @contextmanager
    def active_guard(self) -> Iterator[None]:
        """Guard against concurrent update check/download/apply operations."""
        with self._lock:
            if self._is_active:
                raise UpdateInProgressError("An update operation is already in progress")
            self._is_active = True
        try:
            yield
        finally:
            with self._lock:
                self._is_active = False

    def _get_client(self) -> httpx.Client:
        if self.client is not None:
            return self.client
        return httpx.Client(timeout=30.0)

    def _assert_application_idle(self) -> None:
        if self.activity_probe is not None and self.activity_probe():
            raise UpdateInProgressError("A project workflow is active; stop it before updating")

    def check_for_updates(
        self,
        repo: str,
        api_base_url: str = "https://api.github.com",
    ) -> UpdateCheckResult:
        """Check GitHub release for new version.
        
        If repo is empty, returns 'not_configured' status with no network calls.
        """
        with self.active_guard():
            clean_repo = repo.strip() if repo else ""
            if not clean_repo:
                return UpdateCheckResult(
                    status="not_configured",
                    current_version=self.current_version,
                    message="Chưa cấu hình kho lưu trữ cập nhật (Not configured)",
                )

            # Invariant: Semver single version source
            try:
                cur_semver = SemVer.parse(self.current_version)
            except ValueError as e:
                raise UpdateCheckError(f"Current version '{self.current_version}' is invalid SemVer: {e}") from e

            url = f"{api_base_url.rstrip('/')}/repos/{clean_repo}/releases/latest"
            headers = {
                "User-Agent": "ToolRecapV4-Updater",
                "Accept": "application/vnd.github.v3+json",
            }

            try:
                client = self._get_client()
                resp = client.get(url, headers=headers)
                if resp.status_code == 404:
                    return UpdateCheckResult(
                        status="error",
                        current_version=self.current_version,
                        message=f"Kho lưu trữ '{clean_repo}' không tồn tại hoặc chưa có bản phát hành nào (404).",
                    )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return UpdateCheckResult(
                    status="error",
                    current_version=self.current_version,
                    message=f"Lỗi khi kiểm tra bản cập nhật từ GitHub: {e}",
                )

            tag_name = data.get("tag_name", "")
            try:
                latest_semver = SemVer.parse(tag_name)
            except ValueError:
                return UpdateCheckResult(
                    status="error",
                    current_version=self.current_version,
                    message=f"Thẻ phiên bản mới nhất '{tag_name}' không đúng định dạng SemVer.",
                )

            if latest_semver <= cur_semver:
                return UpdateCheckResult(
                    status="up_to_date",
                    current_version=self.current_version,
                    latest_version=str(latest_semver),
                    message=f"Phiên bản hiện tại v{self.current_version} là mới nhất.",
                )

            # Look for matching Windows portable assets
            assets = data.get("assets", [])
            expected_zip_names = (
                f"ToolRecapV5-v{latest_semver}-windows-portable.zip".lower(),
                f"ToolRecapV4-v{latest_semver}-windows-portable.zip".lower(),
            )
            expected_zip_name = expected_zip_names[0]
            expected_sha_names = {
                *(f"{name}.sha256.txt" for name in expected_zip_names),
                f"toolrecapv5-v{latest_semver}-windows-portable.sha256.txt",
                f"toolrecapv4-v{latest_semver}-windows-portable.sha256.txt",
            }

            zip_asset: Optional[Dict[str, Any]] = None
            sha_asset: Optional[Dict[str, Any]] = None

            for asset in assets:
                name_lower = asset.get("name", "").lower()
                if name_lower in expected_zip_names and zip_asset is None:
                    zip_asset = asset
                elif name_lower in expected_sha_names or (
                    name_lower.startswith((f"toolrecapv5-v{latest_semver}", f"toolrecapv4-v{latest_semver}"))
                    and name_lower.endswith(".sha256.txt")
                ):
                    sha_asset = asset

            if not zip_asset or not sha_asset:
                return UpdateCheckResult(
                    status="missing_assets",
                    current_version=self.current_version,
                    latest_version=str(latest_semver),
                    message=(
                        f"Tìm thấy phiên bản mới v{latest_semver}, nhưng không tìm thấy tệp "
                        f"'{expected_zip_name}' hoặc tệp .sha256.txt tương ứng trên GitHub Release."
                    ),
                )

            return UpdateCheckResult(
                status="update_available",
                current_version=self.current_version,
                latest_version=str(latest_semver),
                zip_url=zip_asset.get("browser_download_url"),
                sha256_url=sha_asset.get("browser_download_url"),
                zip_name=zip_asset.get("name"),
                sha256_name=sha_asset.get("name"),
                release_name=data.get("name"),
                release_body=data.get("body"),
                message=f"Có bản cập nhật mới: v{latest_semver}!",
            )

    def download_and_stage(
        self,
        check_result: UpdateCheckResult,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """Download zip and sha256, verify checksum, safely extract and validate package.
        
        Stages files strictly in LOCALAPPDATA/ToolRecapV4/updates/staging.
        """
        with self.active_guard():
            self._assert_application_idle()
            if check_result.status != "update_available" or not check_result.zip_url or not check_result.sha256_url:
                raise UpdateCheckError("No valid update available to download")

            try:
                latest = SemVer.parse(str(check_result.latest_version or ""))
                current = SemVer.parse(self.current_version)
            except ValueError as exc:
                raise UpdateCheckError(f"Invalid update version: {exc}") from exc
            if latest <= current:
                raise UpdateCheckError("Staged update must be newer than the current version")

            client = self._get_client()

            # 1. Download and parse SHA256
            if progress_callback:
                progress_callback(0.1, "Đang tải tệp mã băm SHA256...")
            resp_sha = client.get(check_result.sha256_url, follow_redirects=True)
            resp_sha.raise_for_status()
            expected_sha256 = parse_sha256_content(resp_sha.text)

            # 2. Download ZIP
            if progress_callback:
                progress_callback(0.3, "Đang tải bản cập nhật...")
            zip_filename = check_result.zip_name or "update.zip"
            if Path(zip_filename).name != zip_filename or Path(zip_filename).is_absolute():
                raise UpdateCheckError("Update archive name is unsafe")
            zip_target = self.staging_dir / zip_filename
            staged_extract_dir = self.staging_dir / f"v{check_result.latest_version}"
            if staged_extract_dir.exists():
                shutil.rmtree(staged_extract_dir, ignore_errors=True)
            try:
                with client.stream("GET", check_result.zip_url, follow_redirects=True) as response:
                    response.raise_for_status()
                    expected_length = response.headers.get("content-length")
                    downloaded = 0
                    with open(zip_target, "wb") as handle:
                        for chunk in response.iter_bytes(chunk_size=65536):
                            downloaded += len(chunk)
                            handle.write(chunk)
                    if expected_length is not None and downloaded != int(expected_length):
                        raise UpdateCheckError("Update download was incomplete")

                if progress_callback:
                    progress_callback(0.7, "Đang kiểm tra mã băm SHA256...")
                if not verify_checksum(zip_target, expected_sha256):
                    raise ChecksumMismatchError(
                        f"Checksum mismatch for update archive: expected {expected_sha256}"
                    )

                if progress_callback:
                    progress_callback(0.85, "Đang giải nén an toàn...")
                staged_extract_dir.mkdir(parents=True, exist_ok=True)
                safe_extract_zip(zip_target, staged_extract_dir)
                if progress_callback:
                    progress_callback(0.95, "Đang xác thực gói cài đặt...")
                validate_package(staged_extract_dir, expected_version=check_result.latest_version)
            except Exception:
                shutil.rmtree(staged_extract_dir, ignore_errors=True)
                raise
            finally:
                zip_target.unlink(missing_ok=True)

            if progress_callback:
                progress_callback(1.0, "Tải và kiểm tra bản cập nhật hoàn tất!")

            return staged_extract_dir

    def launch_apply_helper(
        self,
        staged_path: Path,
        target_dir: Path,
        executable: Optional[Path | str] = None,
    ) -> subprocess.Popen:
        """Launch external update helper process to replace files after app exit."""
        staged_path = Path(staged_path).resolve()
        target_dir = Path(target_dir).resolve()
        self._assert_application_idle()
        validate_package(staged_path)

        if getattr(sys, "frozen", False):
            src_exe = Path(sys.executable).resolve()
            # If src_exe is inside target_dir, copy to staging helper runtime outside target_dir
            # to avoid locked own executable during swap
            if src_exe == target_dir or target_dir in src_exe.parents:
                helper_runtime_dir = self.storage_root / "updates" / "helper_runtime"
                shutil.rmtree(helper_runtime_dir, ignore_errors=True)
                helper_runtime_dir.mkdir(parents=True, exist_ok=True)
                helper_exe = helper_runtime_dir / src_exe.name
                shutil.copy2(src_exe, helper_exe)
                src_internal = src_exe.parent / "_internal"
                if src_internal.is_dir():
                    dst_internal = helper_runtime_dir / "_internal"
                    shutil.copytree(src_internal, dst_internal)
            else:
                helper_exe = src_exe

            cmd = [
                str(helper_exe),
                "--update-helper",
                "--target-dir",
                str(target_dir),
                "--staged-dir",
                str(staged_path),
                "--backup-dir",
                str(self.backup_dir),
                "--parent-pid",
                str(os.getpid()),
                "--storage-root",
                str(self.storage_root),
            ]
            if executable is None:
                executable = target_dir / src_exe.name
        else:
            cmd = [
                sys.executable,
                "-m",
                "toolrecap_v4.update_helper",
                "--target-dir",
                str(target_dir),
                "--staged-dir",
                str(staged_path),
                "--backup-dir",
                str(self.backup_dir),
                "--parent-pid",
                str(os.getpid()),
                "--storage-root",
                str(self.storage_root),
            ]
        if executable:
            cmd.extend(["--executable", str(executable)])

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd,
            creationflags=creationflags,
            close_fds=True,
        )
        return proc
