"""Source video discovery and fingerprinting with Windows collision detection."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    DiscoveryError,
    WindowsCollisionError,
    WindowsReservedNameError,
)

# 8 supported video formats (case-insensitive)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4",
    ".mkv",
    ".mov",
    ".m4v",
    ".avi",
    ".ts",
    ".m2ts",
    ".webm",
})

# Windows reserved device names
WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset({
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
})


@dataclass(frozen=True)
class SourceFingerprint:
    """Immutable fingerprint of a discovered source video file."""

    basename: str
    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    extension: str

    def to_dict(self) -> dict:
        return asdict(self)


def natural_sort_key(s: str) -> Tuple[List[Tuple[int, int | str]], str]:
    """Deterministic natural sort key.
    
    Splits string into numeric and non-numeric chunks.
    Numeric chunks sort numerically. Non-numeric chunks sort by casefold.
    Verbatim string is appended as tie-breaker for strict determinism.
    """
    parts = re.split(r"(\d+)", s)
    tokens: List[Tuple[int, int | str]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part.casefold()))
    return (tokens, s)


def compute_file_fingerprint(
    file_path: Path,
    cancellation_token: Optional[CancellationToken] = None,
    compute_hash: bool = True,
) -> SourceFingerprint:
    """Compute sha256, size, and mtime for a file without modifying its bytes."""
    if cancellation_token:
        cancellation_token.check_cancelled()

    path_obj = file_path.resolve()
    stat_result = path_obj.stat()
    size_bytes = stat_result.st_size
    mtime_ns = stat_result.st_mtime_ns

    if not compute_hash:
        return SourceFingerprint(
            basename=path_obj.name,
            path=str(path_obj),
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            sha256="",
            extension=path_obj.suffix.lower(),
        )

    hasher = hashlib.sha256()
    # Read-only binary mode to strictly preserve source bytes
    with open(path_obj, "rb") as f:
        while True:
            if cancellation_token:
                cancellation_token.check_cancelled()
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)

    return SourceFingerprint(
        basename=path_obj.name,
        path=str(path_obj),
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        sha256=hasher.hexdigest(),
        extension=path_obj.suffix.lower(),
    )


def is_windows_reserved_stem(stem: str) -> bool:
    """Check if the stem or name matches a Windows reserved device name."""
    # Split at first dot if present (e.g. CON.tar.gz -> CON)
    base = stem.split(".")[0].strip().upper()
    return base in WINDOWS_RESERVED_NAMES


def discover_sources(
    directory: str | Path,
    cancellation_token: Optional[CancellationToken] = None,
    compute_hash: bool = True,
    progress_callback: Optional[Callable[..., None]] = None,
) -> List[SourceFingerprint]:
    """Discover source video files non-recursively with natural sorting and collision detection.
    
    Rules:
    - Non-recursive: subdirectories are never traversed.
    - Formats: mp4, mkv, mov, m4v, avi, ts, m2ts, webm (case-insensitive).
    - Windows collision detection: rejects duplicate basenames under casefold.
    - Rejects Windows reserved device names in discovered files.
    - Returns deterministically naturally sorted list of SourceFingerprint.
    """
    if cancellation_token:
        cancellation_token.check_cancelled()

    dir_path = Path(directory).resolve()
    if not dir_path.exists():
        raise DiscoveryError(f"Discovery directory does not exist: {dir_path}")
    if not dir_path.is_dir():
        raise DiscoveryError(f"Discovery path is not a directory: {dir_path}")

    # Casefold collision tracker: casefolded_name -> original_names
    casefold_map: Dict[str, List[str]] = {}
    matching_files: List[Path] = []

    try:
        # Non-recursive directory traversal
        for entry in dir_path.iterdir():
            if cancellation_token:
                cancellation_token.check_cancelled()

            # Ignore directories even if their name ends with a supported extension
            # Non-recursive: do not enter subdirectories
            if not entry.is_file():
                continue

            ext = entry.suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                stem = entry.stem
                if is_windows_reserved_stem(stem):
                    raise WindowsReservedNameError(
                        f"Discovered file uses Windows reserved name: {entry.name}"
                    )

                cf = entry.name.casefold()
                casefold_map.setdefault(cf, []).append(entry.name)
                matching_files.append(entry)
    except OSError as e:
        raise DiscoveryError(f"Failed to read directory {dir_path}: {e}") from e

    # Windows collision check: multiple files resolving to the same casefold
    collisions = [names for names in casefold_map.values() if len(names) > 1]
    if collisions:
        raise WindowsCollisionError(
            f"Windows filename collision detected in {dir_path}: {collisions}"
        )

    # Sort deterministically using natural sort key on filename
    matching_files.sort(key=lambda p: natural_sort_key(p.name))

    # Compute fingerprints
    fingerprints: List[SourceFingerprint] = []
    total_matching = len(matching_files)
    # The desktop discovery worker opts in to a richer callback so the UI can
    # paint the queue before the whole folder has been fingerprinted.  Keep the
    # established two-argument callback contract for every other caller.
    include_source = bool(
        progress_callback
        and getattr(progress_callback, "_toolrecap_include_source", False)
    )
    if include_source and progress_callback:
        progress_callback(0, total_matching, None)
    for idx, p in enumerate(matching_files, start=1):
        if cancellation_token:
            cancellation_token.check_cancelled()
        if compute_hash:
            fp = compute_file_fingerprint(p, cancellation_token, compute_hash=True)
        else:
            stat_res = p.stat()
            fp = SourceFingerprint(
                basename=p.name,
                path=str(p.resolve()),
                size_bytes=stat_res.st_size,
                mtime_ns=stat_res.st_mtime_ns,
                sha256="",
                extension=p.suffix.lower(),
            )
        fingerprints.append(fp)
        if progress_callback:
            if include_source:
                progress_callback(idx, total_matching, fp)
            else:
                progress_callback(idx, total_matching)

    return fingerprints


def create_source_map(
    fingerprints: List[SourceFingerprint],
) -> Dict[str, SourceFingerprint]:
    """Create exact basename source mapping with Windows collision detection."""
    source_map: Dict[str, SourceFingerprint] = {}
    casefold_seen: Dict[str, str] = {}

    for fp in fingerprints:
        cf = fp.basename.casefold()
        if cf in casefold_seen:
            raise WindowsCollisionError(
                f"Duplicate source basename under casefold: '{fp.basename}' conflicts with '{casefold_seen[cf]}'"
            )
        casefold_seen[cf] = fp.basename
        source_map[fp.basename] = fp

    return source_map
