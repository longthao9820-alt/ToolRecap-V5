"""Validation and content audit for extracted one-folder update packages."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Optional

from toolrecap_v4.errors import PackageValidationError
from toolrecap_v4.updater.semver import SemVer

EXPECTED_SCHEMA_VERSION = "3.0"
EXPECTED_PACKAGE_FORMAT = "toolrecap-portable-v1"
FORBIDDEN_ROOT_COMPONENTS = {
    ".git", ".pytest_cache", "tests", "projects", "prepared", "evidence", "catalog",
    "planning", "visual", "writers", "finalization", "secrets", "outputs",
}
FORBIDDEN_FILENAMES = {"settings.json", "credentials.dpapi", ".env"}


def find_executable(package_dir: Path, names: list[str]) -> Optional[Path]:
    for name in names:
        for candidate in (package_dir / name, package_dir / "bin" / name):
            if candidate.is_file():
                return candidate
    return None


def audit_package_contents(package_dir: Path) -> None:
    """Reject developer metadata and user state from an end-user distribution."""
    root = Path(package_dir).resolve()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        lowered_parts = {part.casefold() for part in relative.parts}
        first_part = relative.parts[0].casefold() if relative.parts else ""
        if first_part in FORBIDDEN_ROOT_COMPONENTS or ".git" in lowered_parts or ".pytest_cache" in lowered_parts:
            raise PackageValidationError(f"Package contains forbidden development/user-state path: {relative}")
        if "pytest" in lowered_parts:
            raise PackageValidationError(f"Package contains pytest runtime: {relative}")
        if path.name.casefold() in FORBIDDEN_FILENAMES:
            raise PackageValidationError(f"Package contains forbidden user-state file: {relative}")


def validate_package(package_dir: Path, expected_version: Optional[str] = None) -> None:
    """Validate the exact Phase 13 one-folder layout and required runtime resources."""
    package_dir = Path(package_dir).resolve()
    if not package_dir.is_dir():
        raise PackageValidationError(f"Package path is not a directory: {package_dir}")
    audit_package_contents(package_dir)

    exe = find_executable(package_dir, [
        "ToolRecapV5.exe", "toolrecapv5.exe", "toolrecap_v5.exe",
        "ToolRecapV4.exe", "toolrecapv4.exe", "toolrecap_v4.exe",
    ])
    if not exe:
        raise PackageValidationError("Package is missing main executable 'ToolRecapV5.exe'")
    runtime_dir = package_dir / "_internal"
    if not runtime_dir.is_dir():
        raise PackageValidationError("Package is missing PyInstaller one-folder runtime '_internal'")
    if not (runtime_dir / "base_library.zip").is_file():
        raise PackageValidationError("Package is missing _internal/base_library.zip")
    if not any(
        path.is_file() and re.fullmatch(r"python\d+\.dll", path.name.casefold())
        for path in runtime_dir.iterdir()
    ):
        raise PackageValidationError("Package is missing Python runtime DLL in _internal")
    if not find_executable(package_dir, ["ffmpeg.exe"]):
        raise PackageValidationError("Package is missing 'ffmpeg.exe'")
    if not find_executable(package_dir, ["ffprobe.exe"]):
        raise PackageValidationError("Package is missing 'ffprobe.exe'")

    marker_path = package_dir / "package_marker.json"
    if not marker_path.is_file():
        raise PackageValidationError("Package is missing package_marker.json")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PackageValidationError(f"Invalid package_marker.json: {exc}") from exc
    if not isinstance(marker, dict):
        raise PackageValidationError("package_marker.json must contain an object")
    if marker.get("package_format") != EXPECTED_PACKAGE_FORMAT:
        raise PackageValidationError(
            f"Package format must be '{EXPECTED_PACKAGE_FORMAT}'"
        )
    if str(marker.get("schema_version", "")) != EXPECTED_SCHEMA_VERSION:
        raise PackageValidationError(
            f"Package schema_version does not match expected '{EXPECTED_SCHEMA_VERSION}'"
        )
    version = str(marker.get("version", ""))
    try:
        SemVer.parse(version)
    except ValueError as exc:
        raise PackageValidationError(f"Package version is not valid SemVer: {version!r}") from exc
    if expected_version and SemVer.parse(version) != SemVer.parse(expected_version):
        raise PackageValidationError(
            f"Package version '{version}' does not match expected '{expected_version}'"
        )

    schema_path = package_dir / "schemas" / "recap_v3_schema.json"
    if not schema_path.is_file():
        raise PackageValidationError("Package is missing schemas/recap_v3_schema.json")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_version = schema.get("properties", {}).get("schema_version", {}).get("const")
    except Exception as exc:
        raise PackageValidationError(f"Invalid packaged schema JSON: {exc}") from exc
    if str(schema_version) != EXPECTED_SCHEMA_VERSION:
        raise PackageValidationError(
            f"Schema definition version '{schema_version}' does not match expected '{EXPECTED_SCHEMA_VERSION}'"
        )

    required_files = marker.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        raise PackageValidationError("package_marker.json must list required_files")
    for raw_relative in required_files:
        relative = Path(str(raw_relative))
        if relative.is_absolute() or ".." in relative.parts:
            raise PackageValidationError(f"Unsafe required_files entry: {raw_relative!r}")
        if not (package_dir / relative).is_file():
            raise PackageValidationError(f"Package is missing required file: {relative}")
