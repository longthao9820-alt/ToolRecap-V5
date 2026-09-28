"""Durable atomic JSON persistence in LOCALAPPDATA with secret prevention."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import uuid

from toolrecap_v4.errors import PersistenceError, SecretExposureError
from toolrecap_v4.validator import check_for_secrets, validate_windows_name

DEFAULT_APP_DIR_NAME = "ToolRecapV4"


def native_io_path(path: Path | str) -> str:
    """Return an OS path that remains usable beyond legacy Windows MAX_PATH.

    Python normally opts into long-path support, but portable/frozen builds still
    inherit the host Windows policy.  Prefixing absolute paths keeps checkpoint
    writes reliable on machines with a long user profile or project identifier.
    """
    resolved = str(Path(path).resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def get_storage_root(override_dir: Optional[str | Path] = None) -> Path:
    """Return storage directory strictly in LOCALAPPDATA (never in app dir).
    
    If override_dir is provided (e.g., during tests), uses that path.
    Otherwise resolves %LOCALAPPDATA%/ToolRecapV4.
    """
    if override_dir:
        root = Path(override_dir).resolve()
    else:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            root = Path(local_app_data) / DEFAULT_APP_DIR_NAME
        else:
            root = Path.home() / "AppData" / "Local" / DEFAULT_APP_DIR_NAME

    root.mkdir(parents=True, exist_ok=True)
    return root


def atomic_write_json(
    target_path: Path,
    data: Dict[str, Any],
    indent: int = 2,
) -> None:
    """Durable atomic JSON write using fsync and atomic rename.
    
    Prevents partial writes, corruption on power failure, and rejects secrets.
    """
    # Reject secrets before any disk activity
    check_for_secrets(data)

    target_path = target_path.resolve()
    target_dir = target_path.parent
    os.makedirs(native_io_path(target_dir), exist_ok=True)

    # Unique temporary file in the same directory ensures atomic os.replace on same volume
    tmp_path = target_dir / f"{target_path.name}.tmp.{uuid.uuid4().hex}"

    try:
        with open(native_io_path(tmp_path), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(native_io_path(tmp_path), native_io_path(target_path))
    except Exception as e:
        if os.path.exists(native_io_path(tmp_path)):
            try:
                os.unlink(native_io_path(tmp_path))
            except OSError:
                pass
        if isinstance(e, SecretExposureError):
            raise
        raise PersistenceError(f"Failed to atomically write JSON to {target_path}: {e}") from e


def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file from disk."""
    path = path.resolve()
    if not os.path.isfile(native_io_path(path)):
        raise PersistenceError(f"JSON file does not exist: {path}")
    try:
        with open(native_io_path(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise PersistenceError(f"Failed to read JSON from {path}: {e}") from e


class ProjectPersistence:
    """Manages project state and output checkpoint persistence in LOCALAPPDATA."""

    def __init__(self, storage_root: Optional[str | Path] = None) -> None:
        self.root = get_storage_root(storage_root)
        self.projects_dir = self.root / "projects"
        self.checkpoints_dir = self.root / "checkpoints"
        self.settings_dir = self.root / "settings"
        self.raw_dir = self.root / "raw"
        self.sub_dir = self.root / "sub_analysis"
        self.final_dir = self.root / "final"
        self.prepared_dir = self.root / "prepared"

        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sub_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_dir.mkdir(parents=True, exist_ok=True)

    def _project_path(self, project_id: str) -> Path:
        validate_windows_name(project_id, "project_id")
        return self.projects_dir / f"{project_id}.json"

    def _raw_path(self, project_id: str) -> Path:
        validate_windows_name(project_id, "project_id")
        return self.raw_dir / f"{project_id}.txt"

    def _sub_path(self, project_id: str) -> Path:
        validate_windows_name(project_id, "project_id")
        return self.sub_dir / f"{project_id}.txt"

    def _final_path(self, project_id: str) -> Path:
        validate_windows_name(project_id, "project_id")
        return self.final_dir / f"{project_id}.json"

    def _checkpoint_dir(self, project_id: str) -> Path:
        validate_windows_name(project_id, "project_id")
        d = self.checkpoints_dir / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _checkpoint_path(self, project_id: str, checkpoint_id: str) -> Path:
        validate_windows_name(checkpoint_id, "checkpoint_id")
        return self._checkpoint_dir(project_id) / f"{checkpoint_id}.json"

    def _operational_status_path(self, project_id: str) -> Path:
        """Return the dedicated non-editorial workflow status artifact path."""
        validate_windows_name(project_id, "project_id")
        return self.projects_dir / project_id / "operational_status.json"

    def save_project(self, project_data: Dict[str, Any]) -> Path:
        """Save project state atomically in LOCALAPPDATA."""
        if "project_id" not in project_data:
            raise PersistenceError("project_data missing 'project_id'.")
        project_id = project_data["project_id"]
        target = self._project_path(project_id)
        atomic_write_json(target, project_data)
        return target

    def load_project(self, project_id: str) -> Dict[str, Any]:
        """Load project state from LOCALAPPDATA."""
        target = self._project_path(project_id)
        return read_json(target)

    def save_checkpoint(
        self, project_id: str, checkpoint_id: str, checkpoint_data: Dict[str, Any]
    ) -> Path:
        """Save output checkpoint atomically in LOCALAPPDATA."""
        target = self._checkpoint_path(project_id, checkpoint_id)
        atomic_write_json(target, checkpoint_data)
        return target

    def load_checkpoint(self, project_id: str, checkpoint_id: str) -> Dict[str, Any]:
        """Load output checkpoint from LOCALAPPDATA."""
        target = self._checkpoint_path(project_id, checkpoint_id)
        return read_json(target)

    def save_operational_status(self, project_id: str, status_data: Dict[str, Any]) -> Path:
        """Persist workflow observability/timing separately from editorial Final JSON."""
        target = self._operational_status_path(project_id)
        atomic_write_json(target, status_data)
        return target

    def load_operational_status(self, project_id: str) -> Dict[str, Any]:
        """Load the latest event-based workflow status snapshot."""
        return read_json(self._operational_status_path(project_id))

    def has_operational_status(self, project_id: str) -> bool:
        """Return whether a project has V4 operational timing metadata."""
        return self._operational_status_path(project_id).is_file()

    def save_settings(self, settings_data: Dict[str, Any]) -> Path:
        """Save application settings atomically in LOCALAPPDATA."""
        target = self.settings_dir / "settings.json"
        atomic_write_json(target, settings_data)
        return target

    def load_settings(self) -> Dict[str, Any]:
        """Load application settings from LOCALAPPDATA."""
        target = self.settings_dir / "settings.json"
        return read_json(target)

    def save_raw_response(self, project_id: str, raw_response: str) -> Path:
        """Save raw Gateway response atomically in LOCALAPPDATA."""
        validate_windows_name(project_id, "project_id")
        check_for_secrets({"raw": raw_response})
        target = self._raw_path(project_id)
        tmp_path = target.parent / f"{target.name}.tmp.{uuid.uuid4().hex}"
        try:
            with open(native_io_path(tmp_path), "w", encoding="utf-8") as f:
                f.write(raw_response)
                f.flush()
                os.fsync(f.fileno())
            os.replace(native_io_path(tmp_path), native_io_path(target))
            return target
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if isinstance(e, SecretExposureError):
                raise
            raise PersistenceError(f"Failed to atomically write raw response to {target}: {e}") from e

    def load_raw_response(self, project_id: str) -> str:
        """Load raw Gateway response from LOCALAPPDATA."""
        target = self._raw_path(project_id)
        if not target.exists():
            raise PersistenceError(f"Raw response file does not exist: {target}")
        try:
            with open(native_io_path(target), "r", encoding="utf-8") as handle:
                return handle.read()
        except Exception as e:
            raise PersistenceError(f"Failed to read raw response from {target}: {e}") from e

    def has_raw_response(self, project_id: str) -> bool:
        """Check if raw response exists on disk."""
        return self._raw_path(project_id).exists()

    def save_sub_analysis(self, project_id: str, sub_analysis: str) -> Path:
        """Save raw Sub video analysis text atomically in LOCALAPPDATA."""
        validate_windows_name(project_id, "project_id")
        check_for_secrets({"sub_analysis": sub_analysis})
        target = self._sub_path(project_id)
        tmp_path = target.parent / f"{target.name}.tmp.{uuid.uuid4().hex}"
        try:
            with open(native_io_path(tmp_path), "w", encoding="utf-8") as f:
                f.write(sub_analysis)
                f.flush()
                os.fsync(f.fileno())
            os.replace(native_io_path(tmp_path), native_io_path(target))
            return target
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if isinstance(e, SecretExposureError):
                raise
            raise PersistenceError(f"Failed to atomically write Sub analysis to {target}: {e}") from e

    def load_sub_analysis(self, project_id: str) -> str:
        """Load raw Sub video analysis from LOCALAPPDATA."""
        target = self._sub_path(project_id)
        if not target.exists():
            raise PersistenceError(f"Sub analysis file does not exist: {target}")
        try:
            with open(native_io_path(target), "r", encoding="utf-8") as handle:
                return handle.read()
        except Exception as e:
            raise PersistenceError(f"Failed to read Sub analysis from {target}: {e}") from e

    def has_sub_analysis(self, project_id: str) -> bool:
        """Check if Sub analysis exists on disk."""
        return self._sub_path(project_id).exists()

    def save_final_json(self, project_id: str, final_json: Dict[str, Any]) -> Path:
        """Save final validated project JSON atomically in LOCALAPPDATA."""
        validate_windows_name(project_id, "project_id")
        target = self._final_path(project_id)
        atomic_write_json(target, final_json)
        return target

    def load_final_json(self, project_id: str) -> Dict[str, Any]:
        """Load final validated project JSON from LOCALAPPDATA."""
        target = self._final_path(project_id)
        return read_json(target)

    def has_final_json(self, project_id: str) -> bool:
        """Check if final JSON exists on disk."""
        return self._final_path(project_id).exists()

    def _prepared_dir(self, project_id: str) -> Path:
        validate_windows_name(project_id, "project_id")
        d = self.prepared_dir / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _prepared_episode_path(self, project_id: str, episode_id: str) -> Path:
        validate_windows_name(episode_id, "episode_id")
        return self._prepared_dir(project_id) / f"{episode_id}.json"

    def _prepared_manifest_path(self, project_id: str) -> Path:
        return self._prepared_dir(project_id) / "manifest.json"

    def save_prepared_episode(
        self, project_id: str, episode_id: str, prepared_data: Dict[str, Any]
    ) -> Path:
        """Save prepared episode artifact metadata atomically in LOCALAPPDATA."""
        target = self._prepared_episode_path(project_id, episode_id)
        atomic_write_json(target, prepared_data)
        return target

    def load_prepared_episode(self, project_id: str, episode_id: str) -> Dict[str, Any]:
        """Load prepared episode artifact metadata from LOCALAPPDATA."""
        target = self._prepared_episode_path(project_id, episode_id)
        return read_json(target)

    def has_prepared_episode(self, project_id: str, episode_id: str) -> bool:
        """Check if prepared episode exists on disk."""
        target = self._prepared_episode_path(project_id, episode_id)
        return target.exists()

    def list_prepared_episodes(self, project_id: str) -> List[Dict[str, Any]]:
        """List all prepared episodes for a project."""
        p_dir = self._prepared_dir(project_id)
        if not p_dir.exists():
            return []
        eps: List[Dict[str, Any]] = []
        for p in sorted(p_dir.glob("E*.json")):
            try:
                eps.append(read_json(p))
            except Exception:
                continue
        return eps

    def save_prepared_manifest(
        self, project_id: str, manifest_data: Dict[str, Any]
    ) -> Path:
        """Save project source preparation manifest atomically in LOCALAPPDATA."""
        target = self._prepared_manifest_path(project_id)
        atomic_write_json(target, manifest_data)
        return target

    def load_prepared_manifest(self, project_id: str) -> Dict[str, Any]:
        """Load project source preparation manifest from LOCALAPPDATA."""
        target = self._prepared_manifest_path(project_id)
        return read_json(target)

    def has_prepared_manifest(self, project_id: str) -> bool:
        """Check if project source preparation manifest exists on disk."""
        target = self._prepared_manifest_path(project_id)
        return target.exists()

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all persisted projects, sorted with most recently updated first."""
        if not self.projects_dir.exists():
            return []
        projects: List[Dict[str, Any]] = []
        for p in self.projects_dir.glob("*.json"):
            try:
                data = read_json(p)
                if isinstance(data, dict) and "project_id" in data:
                    projects.append(data)
            except Exception:
                continue

        def _sort_key(item: Dict[str, Any]) -> str:
            ts = item.get("timestamps", {})
            if isinstance(ts, dict):
                return ts.get("updated_at") or ts.get("created_at") or ""
            return ""

        projects.sort(key=_sort_key, reverse=True)
        return projects

    def delete_project(self, project_id: str) -> list[Path]:
        """Delete one saved project and its private checkpoints/caches.

        Published videos and subtitles are intentionally preserved. Settings,
        credentials, and shared source-analysis caches are never touched.
        """
        validate_windows_name(project_id, "project_id")
        removed: list[Path] = []
        files = (
            self._project_path(project_id),
            self._raw_path(project_id),
            self._sub_path(project_id),
            self._final_path(project_id),
        )
        directories = (
            self.projects_dir / project_id,
            self.checkpoints_dir / project_id,
            self.prepared_dir / project_id,
        )
        for path in files:
            if path.is_file():
                path.unlink()
                removed.append(path)
        allowed_parents = {
            self.projects_dir.resolve(),
            self.checkpoints_dir.resolve(),
            self.prepared_dir.resolve(),
        }
        for directory in directories:
            resolved = directory.resolve()
            if resolved.parent not in allowed_parents:
                raise PersistenceError(f"Unsafe project cleanup target: {resolved}")
            if resolved.is_dir():
                shutil.rmtree(resolved)
                removed.append(resolved)

        journal_root = self.root / "publication-journals"
        if journal_root.is_dir():
            for manifest in journal_root.glob("*/manifest.json"):
                try:
                    data = read_json(manifest)
                except PersistenceError:
                    continue
                if str(data.get("owner", "")).startswith(f"{project_id}:"):
                    directory = manifest.parent.resolve()
                    if directory.parent == journal_root.resolve():
                        shutil.rmtree(directory)
                        removed.append(directory)
        return removed
