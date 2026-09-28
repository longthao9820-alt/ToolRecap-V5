"""Safe archive verification and extraction with security guards."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
from typing import Optional
import zipfile

from toolrecap_v4.errors import ChecksumMismatchError, MaliciousArchiveError

DEFAULT_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
DEFAULT_MAX_FILE_COUNT = 10000
DEFAULT_MAX_RATIO = 100.0


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file in streaming chunks."""
    file_path = Path(file_path).resolve()
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def parse_sha256_content(content: str) -> str:
    """Extract SHA256 hex string from sha256.txt content.
    
    Handles standard formats:
    - '<hash>  <filename>'
    - '<hash> *<filename>'
    - '<hash>'
    """
    for line in content.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts:
            candidate = parts[0].strip().lower()
            if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate):
                return candidate
    raise ValueError(f"No valid 64-character SHA256 hash found in content: {content[:100]!r}")


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify that file's SHA256 matches expected checksum."""
    actual_sha256 = calculate_sha256(file_path)
    clean_expected = expected_sha256.strip().lower()
    return actual_sha256 == clean_expected


def safe_extract_zip(
    zip_path: Path,
    extract_to: Path,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
    max_file_count: int = DEFAULT_MAX_FILE_COUNT,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> None:
    """Safely extract a ZIP archive rejecting traversal, symlinks, and zip bombs.
    
    Raises:
        MaliciousArchiveError: If archive violates security constraints.
    """
    zip_path = Path(zip_path).resolve()
    extract_to = Path(extract_to).resolve()
    extract_to.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(zip_path):
        raise MaliciousArchiveError(f"File is not a valid zip archive: {zip_path}")

    created_paths: list[Path] = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infolist = zf.infolist()

            # 1. Check total file count
            if len(infolist) > max_file_count:
                raise MaliciousArchiveError(
                    f"Archive contains {len(infolist)} files, exceeding limit of {max_file_count}"
                )

            # 2. Check total uncompressed size from metadata
            total_size_meta = sum(info.file_size for info in infolist)
            if total_size_meta > max_uncompressed_bytes:
                raise MaliciousArchiveError(
                    f"Archive uncompressed size ({total_size_meta} bytes) exceeds limit of {max_uncompressed_bytes} bytes"
                )

            # 3. Pre-scan each entry for path traversal, symlinks, and ratio
            for info in infolist:
                # Path traversal check
                raw_name = info.filename
                # Disallow drive letters (e.g. C:), absolute paths, and parent traversal
                normalized = os.path.normpath(raw_name)
                if (
                    normalized.startswith("..")
                    or f"{os.sep}.." in normalized
                    or "/.." in normalized
                    or "\\.." in normalized
                    or os.path.isabs(raw_name)
                    or ":" in raw_name
                    or raw_name.startswith("/")
                    or raw_name.startswith("\\")
                ):
                    raise MaliciousArchiveError(f"Path traversal detected in archive entry: {raw_name}")

                dest = (extract_to / normalized).resolve()
                try:
                    dest.relative_to(extract_to)
                except ValueError:
                    raise MaliciousArchiveError(f"Path traversal outside target directory: {raw_name}")

                # Symlink check (POSIX mode 0o120000)
                unix_mode = (info.external_attr >> 16) & 0o177777
                if (unix_mode & 0o170000) == 0o120000:
                    raise MaliciousArchiveError(f"Symlink detected in archive entry: {raw_name}")

                # Compression ratio check for significant files
                if info.compress_size > 0 and info.file_size > 10 * 1024 * 1024:
                    ratio = info.file_size / info.compress_size
                    if ratio > max_ratio:
                        raise MaliciousArchiveError(
                            f"Suspicious compression ratio {ratio:.1f} in archive entry: {raw_name}"
                        )

            # 4. Stream extraction with cumulative byte counter
            total_extracted = 0
            for info in infolist:
                dest = (extract_to / os.path.normpath(info.filename)).resolve()
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    created_paths.append(dest)
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(dest, "wb") as dst:
                    created_paths.append(dest)
                    while chunk := src.read(65536):
                        total_extracted += len(chunk)
                        if total_extracted > max_uncompressed_bytes:
                            raise MaliciousArchiveError(
                                f"Archive exceeded maximum uncompressed size ({max_uncompressed_bytes} bytes) during extraction"
                            )
                        dst.write(chunk)

    except Exception:
        # Clean up any files created before failure
        for p in reversed(created_paths):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink(missing_ok=True)
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
        raise
