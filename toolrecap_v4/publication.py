"""Recoverable publication with durable ownership receipts and content hashes."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil

from toolrecap_v4.persistence import atomic_write_json, get_storage_root, native_io_path, read_json


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(native_io_path(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_file(path: Path) -> bool:
    return os.path.isfile(native_io_path(path))


def _unlink(path: Path) -> None:
    try:
        os.unlink(native_io_path(path))
    except FileNotFoundError:
        pass


def _copy_with_fsync(source: Path, destination: Path) -> None:
    """Copy bytes durably; source and destination may be on different drives."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(native_io_path(source), "rb") as input_stream, open(native_io_path(destination), "wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _destination_temporary(target: Path, directory: Path, index: int) -> Path:
    """Return a deterministic staging file on the target's own volume."""
    return target.parent / f".{target.name}.toolrecap-{directory.name}-{index}.tmp"


def transaction_dir(targets: list[Path], owner: str, journal_root: Path | None = None) -> Path:
    key = hashlib.sha256((owner + "\n" + "\n".join(str(p.resolve()) for p in targets)).encode()).hexdigest()[:24]
    root = Path(journal_root) if journal_root is not None else get_storage_root()
    return root / "publication-journals" / key


def recover(targets: list[Path], owner: str, journal_root: Path | None = None) -> bool:
    """Finish an interrupted publish; never overwrite an externally changed file."""
    directory = transaction_dir(targets, owner, journal_root)
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return False
    state = read_json(manifest)
    if state.get("owner") != owner or state.get("targets") != [str(p.resolve()) for p in targets]:
        raise ValueError("Publication ownership mismatch")
    rows = state["files"]
    if len(rows) != len(targets):
        raise ValueError("Incomplete publication manifest")
    if state["status"] == "COMPLETE":
        complete = all(_is_file(p) and digest(p) == row["new_hash"] for p, row in zip(targets, rows))
        if complete:
            for i, target in enumerate(targets):
                _unlink(directory / f"new-{i}")
                _unlink(directory / f"commit-{i}")
                _unlink(_destination_temporary(target, directory, i))
        return complete
    for i, (target, row) in enumerate(zip(targets, rows)):
        current = digest(target) if _is_file(target) else None
        if current not in {row["old_hash"], row["new_hash"]}:
            raise FileExistsError(f"Publication target changed outside this project: {target}")
        staged = directory / f"new-{i}"
        if not _is_file(staged) or digest(staged) != row["new_hash"]:
            raise ValueError(f"Publication staging file missing/corrupt: {target.name}")
    for i, (target, row) in enumerate(zip(targets, rows)):
        temporary = _destination_temporary(target, directory, i)
        if _is_file(target) and digest(target) == row["new_hash"]:
            _unlink(temporary)
            continue
        if _is_file(temporary) and digest(temporary) != row["new_hash"]:
            _unlink(temporary)
        if not _is_file(temporary):
            _copy_with_fsync(directory / f"new-{i}", temporary)
        if digest(temporary) != row["new_hash"]:
            raise ValueError(f"Destination staging hash mismatch: {target.name}")
        # Both paths are now in target.parent, so os.replace is atomic and
        # valid even when LOCALAPPDATA and the publication folder use different drives.
        os.replace(native_io_path(temporary), native_io_path(target))
    state["status"] = "COMPLETE"
    atomic_write_json(manifest, state)
    for i in range(len(targets)):
        _unlink(directory / f"new-{i}")
        _unlink(directory / f"commit-{i}")  # cleanup journals created by <= 1.0.11
        _unlink(_destination_temporary(targets[i], directory, i))
    return True


def publish(files: list[tuple[Path, Path]], owner: str, cancellation_token=None,
            journal_root: Path | None = None, allow_existing: bool = False) -> None:
    targets = [destination.resolve() for _, destination in files]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Publication targets must be nonempty and unique")
    if len({p.parent for p in targets}) != 1:
        raise ValueError("Publication files must share a destination directory")
    directory = transaction_dir(targets, owner, journal_root)
    previously_owned = (directory / "manifest.json").is_file()
    if previously_owned:
        recover(targets, owner, journal_root)
    if not (allow_existing or previously_owned):
        existing = next((target for target in targets if _is_file(target)), None)
        if existing is not None:
            raise FileExistsError(f"Publication target already exists without this project's ownership: {existing}")
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ((source, _), target) in enumerate(zip(files, targets)):
        if cancellation_token:
            cancellation_token.check_cancelled()
        staged = directory / f"new-{i}"
        _copy_with_fsync(source, staged)
        rows.append({"old_hash": digest(target) if _is_file(target) else None, "new_hash": digest(staged)})
    if cancellation_token:
        cancellation_token.check_cancelled()
    atomic_write_json(directory / "manifest.json", {
        "owner": owner, "status": "PREPARED", "targets": [str(p) for p in targets], "files": rows,
    })
    recover(targets, owner, journal_root)
