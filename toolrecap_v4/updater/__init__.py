"""Updater module for ToolRecap V4."""

from toolrecap_v4.updater.archive import (
    DEFAULT_MAX_FILE_COUNT,
    DEFAULT_MAX_RATIO,
    DEFAULT_MAX_UNCOMPRESSED_BYTES,
    calculate_sha256,
    parse_sha256_content,
    safe_extract_zip,
    verify_checksum,
)
from toolrecap_v4.updater.helper import (
    apply_staged_update_with_rollback,
    is_process_running,
    wait_for_process_exit,
)
from toolrecap_v4.updater.manager import (
    UpdateCheckResult,
    UpdateManager,
)
from toolrecap_v4.updater.semver import (
    SemVer,
)
from toolrecap_v4.updater.validator import (
    audit_package_contents,
    validate_package,
)

__all__ = [
    "SemVer",
    "calculate_sha256",
    "parse_sha256_content",
    "verify_checksum",
    "safe_extract_zip",
    "validate_package",
    "audit_package_contents",
    "apply_staged_update_with_rollback",
    "is_process_running",
    "wait_for_process_exit",
    "UpdateCheckResult",
    "UpdateManager",
    "DEFAULT_MAX_UNCOMPRESSED_BYTES",
    "DEFAULT_MAX_FILE_COUNT",
    "DEFAULT_MAX_RATIO",
]
