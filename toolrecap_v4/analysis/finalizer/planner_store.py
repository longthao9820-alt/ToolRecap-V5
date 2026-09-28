"""Atomic Planner session, round, raw-response, fetch, and draft checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import PlannerSessionError
from toolrecap_v4.persistence import atomic_write_json, native_io_path
from toolrecap_v4.validator import validate_windows_name


class PlannerStore:
    def __init__(self, storage_root: Path | str, project_id: str, session_id: str) -> None:
        validate_windows_name(project_id, "project_id")
        validate_windows_name(session_id, "planner_session_id")
        self.project_id = project_id
        self.session_id = session_id
        self.base = Path(storage_root).resolve() / "projects" / project_id / "planning" / session_id
        self.rounds = self.base / "rounds"

    @property
    def session_path(self) -> Path:
        return self.base / "planner_session.json"

    @property
    def draft_path(self) -> Path:
        return self.base / "planner_draft.json"

    def initialize(self, dependency_digest: str, dependencies: dict[str, Any]) -> None:
        if self.session_path.is_file():
            return
        atomic_write_json(self.session_path, {
            "status": "RUNNING", "project_id": self.project_id, "session_id": self.session_id,
            "dependency_digest": dependency_digest, "dependencies": dependencies,
        })

    def load_draft(self, dependency_digest: str) -> dict[str, Any] | None:
        if not self.session_path.is_file() or not self.draft_path.is_file():
            return None
        try:
            session = json.loads(self.session_path.read_text(encoding="utf-8"))
            draft = json.loads(self.draft_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if session.get("status") != "PLANNER_DRAFT_READY" or session.get("dependency_digest") != dependency_digest:
            return None
        payload = draft.get("action")
        if draft.get("status") != "COMPLETE" or not isinstance(payload, dict):
            return None
        expected = draft.get("action_hash")
        actual = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload if expected == actual else None

    def _round_dir(self, round_id: str) -> Path:
        validate_windows_name(round_id, "planner_round_id")
        return self.rounds / round_id

    def save_raw(
        self, round_id: str, attempt: int, raw: str, *, limit: int,
        dependency_digest: str, measurement: dict[str, Any],
    ) -> None:
        directory = self._round_dir(round_id)
        directory.mkdir(parents=True, exist_ok=True)
        encoded = raw.encode("utf-8")
        stored = encoded[:limit]
        path = directory / f"attempt-{attempt:02d}.raw.txt"
        tmp = directory / f"{path.name}.tmp.{uuid.uuid4().hex}"
        try:
            with open(native_io_path(tmp), "wb") as handle:
                handle.write(stored)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(native_io_path(tmp), native_io_path(path))
            atomic_write_json(directory / f"attempt-{attempt:02d}.raw.manifest.json", {
                "status": "RECEIVED", "round_id": round_id,
                "dependency_digest": dependency_digest,
                "response_hash": hashlib.sha256(encoded).hexdigest(),
                "response_bytes": len(encoded), "stored_bytes": len(stored),
                "truncated_for_diagnostics": len(stored) != len(encoded),
                "measurement": measurement,
            })
        except Exception as exc:
            try:
                os.unlink(native_io_path(tmp))
            except FileNotFoundError:
                pass
            raise PlannerSessionError(f"Failed to save Planner raw response: {exc}", round_id=round_id) from exc

    def load_latest_raw(self, round_id: str, dependency_digest: str) -> tuple[int, str] | None:
        directory = self._round_dir(round_id)
        for manifest_path in sorted(directory.glob("attempt-*.raw.manifest.json"), reverse=True):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                attempt = int(manifest_path.name.split("-")[1].split(".")[0])
                raw = (directory / f"attempt-{attempt:02d}.raw.txt").read_bytes()
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                manifest.get("status") != "RECEIVED"
                or manifest.get("dependency_digest") != dependency_digest
                or manifest.get("truncated_for_diagnostics")
                or len(raw) != manifest.get("stored_bytes")
                or hashlib.sha256(raw).hexdigest() != manifest.get("response_hash")
            ):
                continue
            try:
                return attempt, raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
        return None

    def load_round(self, round_id: str, dependency_digest: str) -> dict[str, Any] | None:
        directory = self._round_dir(round_id)
        manifest_path, response_path = directory / "manifest.json", directory / "response.json"
        if not manifest_path.is_file() or not response_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            response_bytes = response_path.read_bytes()
            response = json.loads(response_bytes.decode("utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if (
            manifest.get("status") != "COMPLETE"
            or manifest.get("dependency_digest") != dependency_digest
            or hashlib.sha256(response_bytes).hexdigest() != manifest.get("response_file_hash")
        ):
            return None
        fetch = None
        fetch_path = directory / "evidence_fetch.json"
        if manifest.get("has_fetch"):
            if not fetch_path.is_file():
                return None
            try:
                fetch_bytes = fetch_path.read_bytes()
                if hashlib.sha256(fetch_bytes).hexdigest() != manifest.get("fetch_file_hash"):
                    return None
                fetch = json.loads(fetch_bytes.decode("utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                return None
        return {"action": response["action"], "fetch": fetch}

    def save_round(
        self, round_id: str, *, dependency_digest: str, action: dict[str, Any],
        fetch: dict[str, Any] | None, cancellation_token: CancellationToken | None,
    ) -> Path:
        directory = self._round_dir(round_id)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        atomic_write_json(manifest_path, {"status": "BUILDING", "round_id": round_id})
        response_path = directory / "response.json"
        atomic_write_json(response_path, {"action": action})
        if cancellation_token:
            cancellation_token.check_cancelled()
        fetch_hash = None
        if fetch is not None:
            fetch_path = directory / "evidence_fetch.json"
            atomic_write_json(fetch_path, fetch)
            fetch_hash = hashlib.sha256(fetch_path.read_bytes()).hexdigest()
        if cancellation_token:
            cancellation_token.check_cancelled()
        atomic_write_json(manifest_path, {
            "status": "COMPLETE", "round_id": round_id,
            "dependency_digest": dependency_digest,
            "response_file_hash": hashlib.sha256(response_path.read_bytes()).hexdigest(),
            "has_fetch": fetch is not None, "fetch_file_hash": fetch_hash,
        })
        return manifest_path

    def save_draft(
        self, *, dependency_digest: str, action: dict[str, Any],
        cancellation_token: CancellationToken | None,
    ) -> Path:
        if cancellation_token:
            cancellation_token.check_cancelled()
        action_hash = hashlib.sha256(
            json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        atomic_write_json(self.draft_path, {"status": "COMPLETE", "action_hash": action_hash, "action": action})
        if cancellation_token:
            cancellation_token.check_cancelled()
        atomic_write_json(self.session_path, {
            "status": "PLANNER_DRAFT_READY", "project_id": self.project_id,
            "session_id": self.session_id, "dependency_digest": dependency_digest,
        })
        return self.draft_path
