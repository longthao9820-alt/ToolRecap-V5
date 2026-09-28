"""Authoritative publication output-root resolution for ToolRecap V4."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

from toolrecap_v4.errors import ToolRecapError


class OutputDirectoryError(ToolRecapError):
    """Raised when publication resolution/creation cannot proceed safely."""


def derive_working_folder(source_selection: str | Path | Sequence[str | Path]) -> Path:
    """Use selected folder, or a selected file/list's common parent folder."""
    if isinstance(source_selection, (str, Path)):
        source = Path(source_selection).expanduser().absolute()
        if source.is_dir():
            return source
        if source.is_file():
            return source.parent
        raise OutputDirectoryError(f"Source selection does not exist: {source}")
    sources = [Path(value).expanduser().absolute() for value in source_selection]
    if not sources:
        raise OutputDirectoryError("Cannot derive a working folder from an empty source selection.")
    parents = {str(source.parent).casefold(): source.parent for source in sources}
    if len(parents) != 1:
        raise OutputDirectoryError(
            "Sources span multiple working folders; configure a manual output directory."
        )
    return next(iter(parents.values()))


def resolve_publication_root(
    *, manual_output_dir: str | Path | None, working_folder: str | Path,
) -> Path:
    """Manual wins exactly; otherwise return sibling Outputs_<folder-name>."""
    manual_text = str(manual_output_dir or "").strip()
    if manual_text:
        manual = Path(manual_text).expanduser()
        if not manual.is_absolute():
            raise OutputDirectoryError("Manual output directory must be an absolute path.")
        return manual
    working = Path(working_folder).expanduser().absolute()
    if not working.name or working.parent == working:
        raise OutputDirectoryError(
            "Working folder has no usable name; configure a manual output directory."
        )
    return working.parent / f"Outputs_{working.name}"


def resolve_v5_publication_root(
    *,
    manual_output_dir: str | Path | None,
    working_folder: str | Path,
    last_source_file: str | Path,
    mode: str,
) -> Path:
    """Resolve the V5 destination inside the selected source folder.

    A configured manual directory still wins exactly. Otherwise V5 creates a
    child of the source working folder named
    ``Output_<source-folder-name>_<Highlight|Recap>``.

    ``last_source_file`` is retained for API compatibility with existing
    callers and saved workflows; automatic V5 naming no longer depends on it.
    """
    manual_text = str(manual_output_dir or "").strip()
    if manual_text:
        return resolve_publication_root(
            manual_output_dir=manual_text,
            working_folder=working_folder,
        )
    working = Path(working_folder).expanduser().absolute()
    if not working.name or working.parent == working:
        raise OutputDirectoryError(
            "Working folder has no usable name; configure a manual output directory."
        )
    raw_folder_name = working.name.strip() or "Video"
    safe_folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", raw_folder_name)
    safe_folder_name = re.sub(r"\s+", " ", safe_folder_name).strip(" ._") or "Video"
    safe_folder_name = safe_folder_name[:120].rstrip(" ._") or "Video"
    suffix = "Highlight" if str(mode).upper() == "HIGHLIGHT" else "Recap"
    return working / f"Output_{safe_folder_name}_{suffix}"


def ensure_publication_root(path: str | Path) -> Path:
    """Create the resolved destination without any fallback location."""
    target = Path(path)
    if target.exists() and not target.is_dir():
        raise OutputDirectoryError(f"Publication root exists as a file: {target}")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot create publication root '{target}': {exc}") from exc
    if not target.is_dir():
        raise OutputDirectoryError(f"Publication root is unavailable: {target}")
    return target
