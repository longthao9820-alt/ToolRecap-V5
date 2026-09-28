"""Revalidate, self-check, and archive an existing Phase 13 portable build."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from build_portable import (
    DIST_DIR,
    PROJECT_ROOT,
    RELEASE_ZIP,
    _clean_runtime_environment,
    _create_release_zip,
    run_packaged_selfcheck,
)
from toolrecap_v4.__version__ import __version__
from toolrecap_v4.updater.validator import audit_package_contents, validate_package


def repackage() -> str:
    executable = DIST_DIR / "ToolRecapV4.exe"
    if not executable.is_file():
        raise RuntimeError(f"Existing portable build not found: {executable}")
    readme = PROJECT_ROOT / "README.md"
    if readme.is_file():
        shutil.copy2(readme, DIST_DIR / "README.md")

    validate_package(DIST_DIR, expected_version=__version__)
    audit_package_contents(DIST_DIR)
    with tempfile.TemporaryDirectory(prefix="toolrecap_repackage_") as temporary:
        temp_root = Path(temporary)
        environment = _clean_runtime_environment(temp_root / "LocalAppData")
        run_packaged_selfcheck(
            executable, cwd=temp_root, report_path=temp_root / "selfcheck.json", environment=environment,
        )
    checksum = _create_release_zip()
    print(f"Repackaged: {RELEASE_ZIP}")
    print(f"SHA256: {checksum}")
    return checksum


if __name__ == "__main__":
    repackage()
