"""Atomic, read-back-verified Writer artifacts and manifests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any

from toolrecap_v4.analysis.finalizer.writer import WriterArtifact, WriterJob
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import WriterPersistenceError
from toolrecap_v4.persistence import atomic_write_json, native_io_path

WRITER_ARTIFACT_VERSION = "writer-artifact-v2"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with open(native_io_path(path), "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Writer JSON root is not an object: {path.name}")
    return value


def _read_bytes(path: Path) -> bytes:
    with open(native_io_path(path), "rb") as handle:
        return handle.read()


def _read_text(path: Path) -> str:
    return _read_bytes(path).decode("utf-8")


def _is_file(path: Path) -> bool:
    return os.path.isfile(native_io_path(path))


def _atomic_bytes(path: Path, data: bytes) -> None:
    os.makedirs(native_io_path(path.parent), exist_ok=True)
    temporary = path.parent / f"{path.name}.tmp.{uuid.uuid4().hex}"
    try:
        with open(native_io_path(temporary), "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(native_io_path(temporary), native_io_path(path))
    finally:
        try:
            os.unlink(native_io_path(temporary))
        except FileNotFoundError:
            pass


class WriterStore:
    def __init__(self, root: Path | str, project_id: str, plan_hash: str) -> None:
        self.base = Path(root).resolve() / "projects" / project_id / "writers" / plan_hash
        self.project_id = project_id
        self.plan_hash = plan_hash

    def output_dir(self, output_id: str) -> Path:
        return self.base / output_id

    def inspect_output(self, output_id: str, *, allow_verifying: bool = False, require_contract_valid: bool = True) -> tuple[bool, str]:
        """Verify an output from persisted identities only, for restart reconstruction."""
        directory = self.output_dir(output_id)
        try:
            manifest = _read_json(directory / "manifest.json")
            artifact = _read_json(directory / "writer_draft.json")
            context = _read_json(directory / "context_manifest.json")
            raw = _read_bytes(directory / "raw_response.txt")
            if manifest.get("status") != "COMPLETE" and not (allow_verifying and manifest.get("status") == "VERIFYING"): return False, "incomplete checkpoint"
            if require_contract_valid and manifest.get("writer_state") != "VALID": return False, "Writer Draft contract invalid"
            if not require_contract_valid and manifest.get("writer_state") not in {"VALID","INVALID_REPAIRABLE","VERIFYING","COMPLETE"}: return False, "unknown Writer contract state"
            if manifest.get("artifact_version") != WRITER_ARTIFACT_VERSION or artifact.get("artifact_version") != WRITER_ARTIFACT_VERSION: return False, "unsupported artifact version"
            for value in (manifest, artifact, context):
                if value.get("project_id") != self.project_id or value.get("plan_hash") != self.plan_hash or value.get("output_id") != output_id: return False, "wrong project/plan/output identity"
            if manifest.get("dependency_digest") != artifact.get("dependency_digest") or context.get("dependency_digest") != artifact.get("dependency_digest"): return False, "dependency mismatch"
            response_hash = hashlib.sha256(raw).hexdigest()
            if manifest.get("response_hash") != response_hash or artifact.get("response_hash") != response_hash: return False, "response hash mismatch"
            if manifest.get("artifact_hash") != _digest(artifact): return False, "artifact hash mismatch"
            raw.decode("utf-8")
            return True, "complete"
        except UnicodeDecodeError:
            return False, "invalid UTF-8"
        except json.JSONDecodeError:
            return False, "invalid JSON"
        except OSError:
            return False, "artifact missing"
        except (ValueError, KeyError):
            return False, "artifact malformed"

    def _artifact(self, job: WriterJob, raw: str, extraction_state: str, parsed: dict[str, Any] | None, contract_state: str, contract_issues: tuple[str,...]) -> dict[str, Any]:
        return {
            "artifact_version": WRITER_ARTIFACT_VERSION,
            "project_id": self.project_id,
            "plan_hash": self.plan_hash,
            "output_id": job.output_id,
            "request_id": job.request_id,
            "dependency_digest": job.dependency_digest,
            "response_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "extraction_state": extraction_state,
            "parsed_json": parsed,
            "contract_state": contract_state,
            "contract_issues": list(contract_issues),
        }

    def _verify_pair(self, job: WriterJob, directory: Path, *, allow_verifying: bool = False, require_contract_valid: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        complete, reason = self.inspect_output(job.output_id, allow_verifying=allow_verifying, require_contract_valid=require_contract_valid)
        if not complete:
            raise WriterPersistenceError(f"Writer artifact {job.output_id} validation failed: {reason}")
        manifest = _read_json(directory / "manifest.json")
        artifact = _read_json(directory / "writer_draft.json")
        raw = _read_bytes(directory / "raw_response.txt")
        if manifest.get("status") != "COMPLETE" and not (allow_verifying and manifest.get("status") == "VERIFYING"):
            raise WriterPersistenceError(f"Writer artifact {job.output_id} is not COMPLETE")
        if manifest.get("artifact_version") != WRITER_ARTIFACT_VERSION or artifact.get("artifact_version") != WRITER_ARTIFACT_VERSION:
            raise WriterPersistenceError(f"Writer artifact {job.output_id} has unsupported artifact version")
        if artifact.get("project_id") != self.project_id or artifact.get("plan_hash") != self.plan_hash or artifact.get("output_id") != job.output_id:
            raise WriterPersistenceError(f"Writer artifact {job.output_id} identity mismatch")
        if artifact.get("dependency_digest") != job.dependency_digest or manifest.get("dependency_digest") != job.dependency_digest:
            raise WriterPersistenceError(f"Writer artifact {job.output_id} dependency mismatch")
        response_hash = hashlib.sha256(raw).hexdigest()
        if artifact.get("response_hash") != response_hash or manifest.get("response_hash") != response_hash:
            raise WriterPersistenceError(f"Writer artifact {job.output_id} response hash mismatch")
        if manifest.get("artifact_hash") != _digest(artifact):
            raise WriterPersistenceError(f"Writer artifact {job.output_id} artifact hash mismatch")
        if manifest.get("writer_state") in {"VALID","INVALID_REPAIRABLE"} and artifact.get("contract_state") != manifest.get("writer_state"):
            raise WriterPersistenceError(f"Writer artifact {job.output_id} contract state mismatch")
        if manifest.get("output_id") != job.output_id or manifest.get("plan_hash") != self.plan_hash or manifest.get("project_id") != self.project_id:
            raise WriterPersistenceError(f"Writer manifest {job.output_id} identity mismatch")
        context = _read_json(directory / "context_manifest.json")
        if context.get("output_id") != job.output_id or context.get("dependency_digest") != job.dependency_digest:
            raise WriterPersistenceError(f"Writer context {job.output_id} identity mismatch")
        return manifest, artifact

    def load(self, job: WriterJob) -> WriterArtifact | None:
        directory = self.output_dir(job.output_id)
        try:
            manifest, artifact = self._verify_pair(job, directory)
            raw = _read_text(directory / "raw_response.txt")
            return WriterArtifact(job.output_id, job.dependency_digest, manifest["response_hash"], raw, artifact["extraction_state"], artifact.get("parsed_json"), manifest["request_bytes"], manifest["measurement"])
        except (OSError, ValueError, KeyError, UnicodeDecodeError, WriterPersistenceError):
            return None

    def load_for_phase9(self, job: WriterJob) -> WriterArtifact | None:
        directory=self.output_dir(job.output_id)
        try:
            manifest,artifact=self._verify_pair(job,directory,require_contract_valid=False)
            raw=_read_text(directory/"raw_response.txt")
            return WriterArtifact(job.output_id,job.dependency_digest,manifest["response_hash"],raw,artifact["extraction_state"],artifact.get("parsed_json"),manifest["request_bytes"],manifest["measurement"])
        except (OSError,ValueError,KeyError,UnicodeDecodeError,WriterPersistenceError):return None

    def load_received_raw(self, job: WriterJob) -> str | None:
        directory = self.output_dir(job.output_id)
        try:
            manifest = _read_json(directory / "manifest.json")
            raw_path = directory / "raw_response.txt"
            raw_bytes = _read_bytes(raw_path)
            if manifest.get("status") == "RECEIVED" and manifest.get("project_id") == self.project_id and manifest.get("plan_hash") == self.plan_hash and manifest.get("output_id") == job.output_id and manifest.get("dependency_digest") == job.dependency_digest and manifest.get("full_response") is True and hashlib.sha256(raw_bytes).hexdigest() == manifest.get("response_hash"):
                return raw_bytes.decode("utf-8")
        except (OSError, ValueError, KeyError, UnicodeDecodeError):
            pass
        return None

    def save_received_raw(self, job: WriterJob, raw: str, request_bytes: int, measurement: dict[str, Any], max_bytes: int) -> None:
        directory = self.output_dir(job.output_id)
        encoded = raw.encode("utf-8")
        if len(encoded) > max_bytes:
            _atomic_bytes(directory / "diagnostic_response.txt", encoded[:max_bytes])
            atomic_write_json(directory / "manifest.json", {"status": "DIAGNOSTIC_RESPONSE_TRUNCATED", "writer_state": "FAILED", "artifact_version": WRITER_ARTIFACT_VERSION, "project_id": self.project_id, "plan_hash": self.plan_hash, "output_id": job.output_id, "dependency_digest": job.dependency_digest, "response_bytes": len(encoded), "stored_bytes": max_bytes, "full_response": False})
            raise WriterPersistenceError(f"Writer response for {job.output_id} exceeds full-response storage cap")
        _atomic_bytes(directory / "raw_response.txt", encoded)
        atomic_write_json(directory / "manifest.json", {"status": "RECEIVED", "writer_state": "RESPONSE_RECEIVED", "artifact_version": WRITER_ARTIFACT_VERSION, "project_id": self.project_id, "plan_hash": self.plan_hash, "output_id": job.output_id, "dependency_digest": job.dependency_digest, "response_hash": hashlib.sha256(encoded).hexdigest(), "response_bytes": len(encoded), "full_response": True, "request_bytes": request_bytes, "measurement": measurement})

    def save(self, job: WriterJob, raw: str, extraction_state: str, parsed: dict[str, Any] | None, request_bytes: int, measurement: dict[str, Any], max_bytes: int, cancellation_token: CancellationToken | None, contract_state: str="VALID", contract_issues: tuple[str,...]=()) -> WriterArtifact:
        directory = self.output_dir(job.output_id)
        if not _is_file(directory / "raw_response.txt"):
            self.save_received_raw(job, raw, request_bytes, measurement, max_bytes)
        if cancellation_token:
            cancellation_token.check_cancelled()
        artifact = self._artifact(job, raw, extraction_state, parsed, contract_state, contract_issues)
        context = {"artifact_version": WRITER_ARTIFACT_VERSION, "project_id": self.project_id, "plan_hash": self.plan_hash, "output_id": job.output_id, "request_id": job.request_id, "dependency_digest": job.dependency_digest, "context_hash": hashlib.sha256(_canonical(job.context)).hexdigest()}
        atomic_write_json(directory / "context_manifest.json", context)
        atomic_write_json(directory / "writer_draft.json", artifact)
        if cancellation_token:
            cancellation_token.check_cancelled()
        if _read_json(directory / "writer_draft.json") != artifact:
            raise WriterPersistenceError(f"Writer artifact {job.output_id} read-back mismatch")
        manifest = {"status": "VERIFYING", "writer_state": "VERIFYING", "artifact_version": WRITER_ARTIFACT_VERSION, "project_id": self.project_id, "plan_hash": self.plan_hash, "output_id": job.output_id, "dependency_digest": job.dependency_digest, "artifact_hash": _digest(artifact), "response_hash": artifact["response_hash"], "response_bytes": len(raw.encode("utf-8")), "full_response": True, "extraction_state": extraction_state, "request_bytes": request_bytes, "measurement": measurement}
        atomic_write_json(directory / "manifest.json", manifest)
        persisted_manifest, persisted_artifact = self._verify_pair(job, directory, allow_verifying=True, require_contract_valid=False)
        if persisted_artifact != artifact or persisted_manifest != manifest:
            raise WriterPersistenceError(f"Writer artifact {job.output_id} final verification mismatch")
        manifest = {**manifest, "status": "COMPLETE", "writer_state": contract_state, "contract_issues": list(contract_issues)}
        atomic_write_json(directory / "manifest.json", manifest)
        persisted_manifest, persisted_artifact = self._verify_pair(job, directory, require_contract_valid=contract_state=="VALID")
        if persisted_artifact != artifact or persisted_manifest != manifest:
            raise WriterPersistenceError(f"Writer artifact {job.output_id} COMPLETE verification mismatch")
        return WriterArtifact(job.output_id, job.dependency_digest, artifact["response_hash"], raw, extraction_state, parsed, request_bytes, measurement)

    def save_project_manifest(self, plan_hash: str, output_ids: list[str], artifacts: dict[str, WriterArtifact], failures: dict[str, str]) -> dict[str, Any]:
        durable_ids: list[str] = [];valid_ids: list[str] = []
        for ordinal, output_id in enumerate(output_ids, 1):
            artifact = artifacts.get(output_id)
            if artifact is None:
                continue
            probe = WriterJob(output_id, ordinal, artifact.dependency_digest, {}, "")
            if self.load_for_phase9(probe) is not None:
                durable_ids.append(output_id)
                if self.load(probe) is not None:valid_ids.append(output_id)
            else:
                failures.setdefault(output_id, "durable Writer artifact read-back verification failed")
        state="COMPLETE" if len(valid_ids)==len(output_ids) and not failures else "REPAIR_REQUIRED" if len(durable_ids)==len(output_ids) and not failures else "INCOMPLETE"
        data = {"artifact_version": WRITER_ARTIFACT_VERSION, "status": state, "writer_state": state, "project_id": self.project_id, "season_plan_hash": plan_hash, "expected_output_count": len(output_ids), "ordered_output_ids": output_ids, "completed_response_count": len(valid_ids), "durable_response_count":len(durable_ids), "pending_count": len(output_ids) - len(durable_ids) - len(failures), "failed_count": len(failures), "canceled_count": 0, "outputs": {oid: {"dependency_digest": artifacts[oid].dependency_digest, "response_hash": artifacts[oid].response_hash, "extraction_state": artifacts[oid].extraction_state} for oid in durable_ids}, "failures": failures}
        atomic_write_json(self.base / "manifest.json", data)
        return data
