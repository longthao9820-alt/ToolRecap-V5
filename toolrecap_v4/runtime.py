"""Relocation-safe runtime and resource paths for source and frozen execution."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable


def is_frozen() -> bool:
    """Return whether the process is running from a frozen application bundle."""
    return bool(getattr(sys, "frozen", False))


def application_root() -> Path:
    """Return the read-only application/install root, never the process CWD."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_runtime_root() -> Path:
    """Return PyInstaller's runtime root or the source project root."""
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return application_root()


def package_mode() -> str:
    return "frozen" if is_frozen() else "source"


def resource_candidates(relative_paths: Iterable[str | Path]) -> tuple[Path, ...]:
    """Build deterministic resource candidates without consulting the CWD."""
    roots = (bundled_runtime_root(), application_root())
    candidates: list[Path] = []
    for relative in relative_paths:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Resource path must be safe and relative: {relative}")
        for root in roots:
            candidate = (root / rel).resolve()
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def find_resource(*relative_paths: str | Path) -> Path:
    """Find a packaged/source resource or raise a deterministic error."""
    candidates = resource_candidates(relative_paths)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Required application resource was not found; searched: {searched}")
