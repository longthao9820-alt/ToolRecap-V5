"""Bounded still-frame observations for Highlight scenes with sparse dialogue."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence
import uuid

from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.analysis.vision.frames import FrameExtractionPolicy, FrameExtractor
from toolrecap_v4.analysis.vision.models import VisualRequest
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.persistence import atomic_write_json, native_io_path, read_json
from toolrecap_v4.analysis.finalizer.writer_contract import parse_json_object
from toolrecap_v4.errors import PersistenceError, VisionRepairExhaustedError, VisionResponseError
from toolrecap_v4.progress import ActivityState, WorkflowStage, safe_emit

VISUAL_DISCOVERY_VERSION = "highlight-still-sample-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _windows(duration_ms: int, count: int = 4) -> list[tuple[int, int]]:
    if duration_ms <= 0:
        return []
    return [(duration_ms * i // count, duration_ms * (i + 1) // count)
            for i in range(count) if duration_ms * (i + 1) // count > duration_ms * i // count]


class HighlightVisualSampler:
    def __init__(self, gateway: GatewayClient, root: Path, model: str, reasoning: str = "",
                 frames_per_episode: int = 24, extractor: FrameExtractor | None = None,
                 repair_attempts: int = 1, max_response_bytes: int = 2 * 1024 * 1024,
                 progress_callback: Callable[[dict[str, Any]], None] | None = None):
        self.gateway, self.root, self.model, self.reasoning = gateway, Path(root), model, reasoning
        self.frames_per_episode = max(4, min(frames_per_episode, 24))
        if repair_attempts < 0 or repair_attempts > 3:
            raise ValueError("Highlight Vision repair attempts must be between 0 and 3")
        self.repair_attempts = repair_attempts
        self.max_response_bytes = max_response_bytes
        self.progress_callback = progress_callback
        self.extractor = extractor or FrameExtractor(root, FrameExtractionPolicy(
            frames_per_range=6, frames_per_episode=24, hard_frame_cap=256, max_dimension=960))

    def _emit(self, **payload: Any) -> None:
        safe_emit(self.progress_callback, stage=WorkflowStage.VISION.value, **payload)

    def run(self, project_id: str, episodes: Sequence[PreparedEpisode],
            token: CancellationToken | None = None) -> dict[str, list[dict[str, Any]]]:
        """Return factual, sparse samples with provenance, not a full video analysis."""
        observations: dict[str, list[dict[str, Any]]] = {}
        for episode in episodes:
            if token:
                token.check_cancelled()
            rows: list[dict[str, Any]] = []
            for ordinal, (start_ms, end_ms) in enumerate(_windows(episode.duration_ms), 1):
                request = VisualRequest(
                    f"HV-{episode.episode_id}-{ordinal:02d}", project_id,
                    episode.episode_id, episode.source_id, episode.source_path,
                    episode.source_fingerprint, start_ms, end_ms,
                    "Identify only what is visible in sampled frames", "highlight-discovery", (),
                )
                frames = self.extractor.extract(request, (ordinal - 1) * 6,
                                                min(6, self.frames_per_episode // 4), token)
                if not frames:
                    continue
                identity = {
                    "version": VISUAL_DISCOVERY_VERSION, "model": self.model,
                    "reasoning": self.reasoning, "request": request.to_dict(),
                    "frames": [{"frame_id": f.frame_id, "time": f.timestamp_ms,
                                "hash": f.content_hash} for f in frames],
                }
                checkpoint = self.root / "projects" / project_id / "highlight-visual" / f"{_digest(identity)}.json"
                parsed = None
                if checkpoint.is_file():
                    try:
                        saved = read_json(checkpoint)
                        if saved.get("identity_hash") == _digest(identity):
                            parsed = saved["response"]
                            self._validate(parsed, frames)
                    except (ValueError, KeyError, TypeError):
                        parsed = None
                if parsed is None:
                    payload = {"task": "Describe only visibly supported events in these sparse sample frames."
                               " Do not infer motion, dialogue, identity, or causality between frames.",
                               "episode_id": episode.episode_id,
                               "frames": [{"frame_id": f.frame_id, "timestamp_ms": f.timestamp_ms}
                                          for f in frames],
                               "response_format": {"observations": [{"frame_id": "exact supplied ID",
                                   "timestamp_ms": "exact supplied integer", "observation": "factual visual description"}]}}
                    parsed = self._request_and_repair(
                        identity, payload, frames, checkpoint.with_suffix(""), token,
                    )
                    atomic_write_json(checkpoint, {"identity_hash": _digest(identity), "response": parsed})
                for frame, row in zip(frames, parsed["observations"]):
                    rows.append({"evidence_id": f"{episode.episode_id}-HV-{ordinal:02d}-{frame.frame_id.rsplit('-', 1)[-1]}",
                                 "episode_id": episode.episode_id, "source_id": episode.source_id,
                                 "start_ms": frame.timestamp_ms,
                                 "end_ms": min(episode.duration_ms, frame.timestamp_ms + 1000),
                                 "observation": row["observation"], "modality": "sampled_still_frame",
                                 "provenance": {"frame_id": frame.frame_id, "frame_hash": frame.content_hash,
                                                "coverage": "sparse sample; other moments were not inspected"}})
            observations[episode.episode_id] = rows
        return observations

    def _request_and_repair(self, identity: dict[str, Any], payload: dict[str, Any],
                            frames: Sequence[Any], directory: Path,
                            token: CancellationToken | None) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        identity_hash = _digest(identity)
        invalid_raw = ""
        errors: list[str] = []
        for attempt in range(self.repair_attempts + 1):
            if token:
                token.check_cancelled()
            raw_path = directory / f"attempt-{attempt:02d}.raw.txt"
            manifest_path = directory / f"attempt-{attempt:02d}.manifest.json"
            raw = self._load_raw(raw_path, manifest_path, identity_hash, attempt)
            phase = "highlight_visual_discovery" if attempt == 0 else "highlight_visual_repair"
            if raw is None:
                if attempt == 0:
                    call_payload = payload
                    system_prompt = (
                        "Return exactly one JSON object with observations for every supplied frame. "
                        "Report visible facts only; do not guess events between sample frames. "
                        "Do not use Markdown or code fences."
                    )
                else:
                    call_payload = {
                        "task": "Repair only the Highlight visual response defects. Return one complete JSON object.",
                        "validation_errors": errors,
                        "original_request": payload,
                        "invalid_response": invalid_raw[:65536],
                        "required_response_format": payload["response_format"],
                        "rules": [
                            "Include every supplied frame exactly once and in the supplied order.",
                            "Copy frame_id and timestamp_ms exactly.",
                            "Observation must be a non-empty factual description of visible content.",
                            "Return JSON only, without Markdown or commentary.",
                        ],
                    }
                    system_prompt = (
                        "Repair one visual response to the supplied JSON format. Preserve factual grounding and "
                        "exact frame identities. Return JSON only."
                    )
                self._emit(
                    state=ActivityState.WAITING_FOR_AI.value if attempt == 0 else ActivityState.REPAIRING.value,
                    activity_text=(
                        f"Waiting for sampled-frame analysis {identity['request']['visual_request_id']}..."
                        if attempt == 0 else
                        f"Repairing sampled-frame response — attempt {attempt} / {self.repair_attempts}."
                    ),
                    current_item=identity["request"]["visual_request_id"], waiting_for="AI",
                    retry_attempt=attempt if attempt else None, retry_limit=self.repair_attempts if attempt else None,
                )
                result = self.gateway.submit_image_chat(
                    prompt=json.dumps(call_payload, ensure_ascii=False),
                    images=[f.path for f in frames], model=self.model,
                    system_prompt=system_prompt, reasoning_effort=self.reasoning or None,
                    stream=False, expect_json=False, cancellation_token=token, phase=phase,
                )
                raw = result.raw_response
                self._save_raw(raw_path, manifest_path, raw, identity_hash, attempt, phase)
            invalid_raw = raw
            parsed = parse_json_object(raw)
            try:
                if parsed is None:
                    preview = raw[:120].replace("\r", " ").replace("\n", " ")
                    raise ValueError(
                        "Response is empty or is not one JSON object"
                        + (f"; response begins with {preview!r}" if preview else "; response was empty")
                    )
                self._validate(parsed, frames)
                atomic_write_json(directory / f"attempt-{attempt:02d}.result.json", {
                    "state": "VALID", "validation_errors": [], "parser_state": "ACCEPTED",
                })
                self._emit(
                    state=ActivityState.LOCAL_PROCESSING.value,
                    activity_text=f"Sampled-frame response validated for {identity['request']['visual_request_id']}.",
                    current_item=identity["request"]["visual_request_id"],
                )
                return parsed
            except (ValueError, TypeError, KeyError) as exc:
                errors = [str(exc)]
                atomic_write_json(directory / f"attempt-{attempt:02d}.result.json", {
                    "state": "INVALID_REPAIRABLE", "validation_errors": errors,
                    "parser_state": "ACCEPTED" if parsed is not None else "REJECTED",
                })
        raise VisionRepairExhaustedError(
            f"Highlight sampled-frame response repair exhausted for "
            f"{identity['request']['visual_request_id']}: {errors[0] if errors else 'unknown defect'}"
        )

    def _save_raw(self, raw_path: Path, manifest_path: Path, raw: str,
                  identity_hash: str, attempt: int, phase: str) -> None:
        encoded = raw.encode("utf-8")
        if len(encoded) > self.max_response_bytes:
            diagnostic = encoded[:self.max_response_bytes]
            self._atomic_bytes(raw_path.with_suffix(".diagnostic.txt"), diagnostic)
            atomic_write_json(manifest_path, {
                "status": "RESPONSE_TRUNCATED", "identity_hash": identity_hash,
                "attempt": attempt, "phase": phase, "response_bytes": len(encoded),
                "stored_bytes": len(diagnostic), "full_response": False,
                "response_hash": hashlib.sha256(encoded).hexdigest(),
            })
            raise VisionResponseError(
                f"Highlight visual response exceeded {self.max_response_bytes} bytes"
            )
        self._atomic_bytes(raw_path, encoded)
        atomic_write_json(manifest_path, {
            "status": "RECEIVED", "identity_hash": identity_hash,
            "attempt": attempt, "phase": phase, "response_bytes": len(encoded),
            "stored_bytes": len(encoded), "full_response": True,
            "response_hash": hashlib.sha256(encoded).hexdigest(),
        })

    @staticmethod
    def _atomic_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f"{path.name}.tmp.{uuid.uuid4().hex}"
        try:
            with open(native_io_path(temporary), "wb") as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            os.replace(native_io_path(temporary), native_io_path(path))
        finally:
            try: os.unlink(native_io_path(temporary))
            except FileNotFoundError: pass

    @staticmethod
    def _load_raw(raw_path: Path, manifest_path: Path, identity_hash: str,
                  attempt: int) -> str | None:
        try:
            manifest = read_json(manifest_path)
            encoded = raw_path.read_bytes()
            if (
                manifest.get("status") == "RECEIVED"
                and manifest.get("identity_hash") == identity_hash
                and manifest.get("attempt") == attempt
                and manifest.get("full_response") is True
                and manifest.get("stored_bytes") == len(encoded)
                and manifest.get("response_hash") == hashlib.sha256(encoded).hexdigest()
            ):
                return encoded.decode("utf-8")
        except (OSError, ValueError, KeyError, UnicodeDecodeError, PersistenceError):
            pass
        return None

    @staticmethod
    def _validate(parsed: Any, frames: Sequence[Any]) -> None:
        if not isinstance(parsed, dict) or set(parsed) != {"observations"}:
            raise ValueError("Highlight visual response must contain observations only")
        rows = parsed["observations"]
        if not isinstance(rows, list) or len(rows) != len(frames):
            raise ValueError("Highlight visual response must cover every supplied frame")
        for row, frame in zip(rows, frames):
            if not isinstance(row, dict) or set(row) != {"frame_id", "timestamp_ms", "observation"}:
                raise ValueError("Highlight visual frame fields mismatch")
            if row["frame_id"] != frame.frame_id or type(row["timestamp_ms"]) is not int or row["timestamp_ms"] != frame.timestamp_ms:
                raise ValueError("Highlight visual frame identity mismatch")
            if not isinstance(row["observation"], str) or not row["observation"].strip():
                raise ValueError("Highlight visual frame observation is empty")
