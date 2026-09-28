"""Bounded factual Scanner orchestration, chunk resume, repair, and stable Evidence IDs."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence
import uuid

from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.models import Evidence, PreparedEpisode
from toolrecap_v4.analysis.scanner.chunking import CHUNK_POLICY_VERSION, ScannerChunk, ScannerChunkPolicy, plan_scanner_chunks
from toolrecap_v4.analysis.scanner.prompts import (
    SCANNER_PROMPT_VERSION,
    SCANNER_SCHEMA_VERSION,
    SCANNER_SYSTEM_PROMPT,
    build_repair_prompt,
    build_scanner_prompt,
    measure_text_request_bytes,
)
from toolrecap_v4.analysis.scanner.schema import (
    SCANNER_VALIDATION_VERSION,
    ScannerObservation,
    parse_scanner_json,
    validate_scanner_response,
)
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    CancelledError,
    ScannerCapacityError,
    ScannerRepairExhaustedError,
    ScannerResponseError,
    ScannerValidationError,
    GatewayError,
)
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.persistence import atomic_write_json, native_io_path
from toolrecap_v4.progress import ActivityState, WorkflowStage, safe_emit

SCANNER_NORMALIZATION_VERSION = "scanner-normalization-v1"
MAX_RAW_RESPONSE_BYTES = 1_048_576


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ScannerConfig:
    model: str
    reasoning: str = ""
    parallelism: int = 3
    chunk_policy: ScannerChunkPolicy = ScannerChunkPolicy()
    repair_attempts: int = 1
    max_raw_response_bytes: int = MAX_RAW_RESPONSE_BYTES

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Scanner model must be configured.")
        if self.parallelism < 1:
            raise ValueError("Scanner parallelism must be at least 1.")
        if self.repair_attempts < 0:
            raise ValueError("Scanner repair_attempts must be non-negative.")
        if self.max_raw_response_bytes < 1024:
            raise ValueError("Scanner max_raw_response_bytes must be at least 1024.")


@dataclass(frozen=True)
class ScannerChunkResult:
    chunk: ScannerChunk
    observations: tuple[ScannerObservation, ...]
    response_hash: str
    artifact_hash: str
    measurements: tuple[dict[str, Any], ...]
    reused: bool


@dataclass(frozen=True)
class ScannerProjectResult:
    project_id: str
    evidence_revision: str
    episode_counts: dict[str, int]
    total_evidence_count: int
    request_measurements: tuple[dict[str, Any], ...]
    reused_chunk_count: int
    requested_chunk_count: int


def scanner_dependency_signature(episodes: Sequence[PreparedEpisode], config: ScannerConfig) -> dict[str, Any]:
    return {
        "episodes": [{
            "episode_id": episode.episode_id,
            "source_id": episode.source_id,
            "source_fingerprint": episode.source_fingerprint or episode.source_hash,
            "prepared_artifact_hash": episode.artifact_hash,
            "transcript_hash": episode.transcript.provenance_hash,
            "transcript_content_hash": _digest(episode.transcript.to_dict()),
        } for episode in episodes],
        "scanner_model": config.model,
        "scanner_reasoning": config.reasoning,
        "prompt_version": SCANNER_PROMPT_VERSION,
        "schema_version": SCANNER_SCHEMA_VERSION,
        "validation_version": SCANNER_VALIDATION_VERSION,
        "normalization_version": SCANNER_NORMALIZATION_VERSION,
        "chunk_policy_version": CHUNK_POLICY_VERSION,
        "chunk_max_duration_ms": config.chunk_policy.max_duration_ms,
        "chunk_max_request_bytes": config.chunk_policy.max_request_bytes,
    }


def compute_evidence_revision(episodes: Sequence[PreparedEpisode], config: ScannerConfig) -> str:
    """Semantic revision deliberately excludes parallelism, creative prompt, voice, and render settings."""
    return f"evr-{_digest(scanner_dependency_signature(episodes, config))[:24]}"


class _ScannerChunkCache:
    def __init__(self, root: Path, project_id: str, revision: str) -> None:
        self.base = root / "projects" / project_id / "scanner" / revision

    def _dir(self, episode_id: str, chunk_id: str) -> Path:
        return self.base / episode_id / chunk_id

    def save_raw(
        self, episode_id: str, chunk_id: str, attempt: int, raw: str, limit: int,
        measurement: dict[str, Any],
    ) -> Path:
        directory = self._dir(episode_id, chunk_id)
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
                "status": "RECEIVED",
                "response_hash": hashlib.sha256(encoded).hexdigest(),
                "response_bytes": len(encoded),
                "stored_bytes": len(stored),
                "truncated_for_diagnostics": len(stored) != len(encoded),
                "measurement": measurement,
            })
            return path
        except Exception:
            try:
                os.unlink(native_io_path(tmp))
            except FileNotFoundError:
                pass
            raise

    def load_latest_raw(self, episode_id: str, chunk_id: str) -> tuple[int, str, dict[str, Any]] | None:
        directory = self._dir(episode_id, chunk_id)
        manifests = sorted(directory.glob("attempt-*.raw.manifest.json"), reverse=True)
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                attempt = int(manifest_path.name.split("-")[1].split(".")[0])
                raw_path = directory / f"attempt-{attempt:02d}.raw.txt"
                raw_bytes = raw_path.read_bytes()
            except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if manifest.get("status") != "RECEIVED" or manifest.get("truncated_for_diagnostics"):
                continue
            if len(raw_bytes) != manifest.get("stored_bytes") or hashlib.sha256(raw_bytes).hexdigest() != manifest.get("response_hash"):
                continue
            try:
                raw = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            return attempt, raw, dict(manifest.get("measurement", {}))
        return None

    def load(self, episode_id: str, chunk: ScannerChunk) -> ScannerChunkResult | None:
        directory = self._dir(episode_id, chunk.chunk_id)
        data_path = directory / "accepted.json"
        manifest_path = directory / "manifest.json"
        if not data_path.is_file() or not manifest_path.is_file():
            return None
        try:
            data = json.loads(data_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if manifest.get("status") != "COMPLETE" or manifest.get("chunk_hash") != chunk.content_hash:
            return None
        if hashlib.sha256(_canonical_bytes(data)).hexdigest() != manifest.get("artifact_hash"):
            return None
        try:
            observations = tuple(ScannerObservation.from_dict(item) for item in data["observations"])
            return ScannerChunkResult(
                chunk=chunk,
                observations=observations,
                response_hash=str(data["response_hash"]),
                artifact_hash=str(manifest["artifact_hash"]),
                measurements=tuple(dict(item) for item in data.get("measurements", [])),
                reused=True,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def save(
        self,
        episode_id: str,
        result: ScannerChunkResult,
        *,
        cancellation_token: CancellationToken | None,
    ) -> ScannerChunkResult:
        if cancellation_token:
            cancellation_token.check_cancelled()
        directory = self._dir(episode_id, result.chunk.chunk_id)
        directory.mkdir(parents=True, exist_ok=True)
        data = {
            "episode_id": episode_id,
            "chunk_id": result.chunk.chunk_id,
            "chunk_hash": result.chunk.content_hash,
            "response_hash": result.response_hash,
            "observations": [item.to_dict() for item in result.observations],
            "measurements": list(result.measurements),
        }
        artifact_hash = hashlib.sha256(_canonical_bytes(data)).hexdigest()
        atomic_write_json(directory / "accepted.json", data)
        if cancellation_token:
            cancellation_token.check_cancelled()
        atomic_write_json(directory / "manifest.json", {
            "status": "COMPLETE",
            "episode_id": episode_id,
            "chunk_id": result.chunk.chunk_id,
            "chunk_hash": result.chunk.content_hash,
            "artifact_hash": artifact_hash,
            "observation_count": len(result.observations),
        })
        return ScannerChunkResult(
            result.chunk, result.observations, result.response_hash, artifact_hash,
            result.measurements, False,
        )


class ScannerService:
    """Text-only Scanner with bounded work submission and per-chunk recovery."""

    def __init__(
        self,
        gateway_client: GatewayClient,
        storage_root: Path | str,
        config: ScannerConfig,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.gateway_client = gateway_client
        self.storage_root = Path(storage_root).resolve()
        self.config = config
        self.progress_callback = progress_callback

    def _emit(self, **payload: Any) -> None:
        safe_emit(self.progress_callback, stage=WorkflowStage.SCANNER.value, **payload)

    def _prompt_for(self, episode: PreparedEpisode, chunk_id: str, start_ms: int, end_ms: int, parts: Sequence[Any]) -> str:
        return build_scanner_prompt(
            episode_id=episode.episode_id,
            source_id=episode.source_id,
            source_basename=episode.source_basename or episode.source_path.name,
            episode_duration_ms=episode.duration_ms,
            chunk_id=chunk_id,
            chunk_start_ms=start_ms,
            chunk_end_ms=end_ms,
            transcript_source=episode.transcript.source_type,
            transcript_format=episode.transcript.source_format,
            transcript_hash=episode.transcript.provenance_hash,
            parts=parts,
        )

    def _plan(self, episode: PreparedEpisode) -> tuple[ScannerChunk, ...]:
        def measure(chunk_id: str, start_ms: int, end_ms: int, parts: Sequence[Any]) -> int:
            prompt = self._prompt_for(episode, chunk_id, start_ms, end_ms, parts)
            return measure_text_request_bytes(model=self.config.model, reasoning=self.config.reasoning, user_prompt=prompt)
        return plan_scanner_chunks(episode, self.config.chunk_policy, measure)

    def _repair_prompt_that_fits(self, original: str, invalid: str, errors: Sequence[str]) -> str:
        text = invalid
        while True:
            prompt = build_repair_prompt(original_prompt=original, invalid_response=text, validation_errors=errors)
            size = measure_text_request_bytes(model=self.config.model, reasoning=self.config.reasoning, user_prompt=prompt)
            if size <= self.config.chunk_policy.max_request_bytes:
                return prompt
            if not text:
                raise ScannerCapacityError("Technical repair request cannot fit Scanner request capacity.")
            text = text[: len(text) // 2]

    def _scan_one(
        self,
        episode: PreparedEpisode,
        chunk: ScannerChunk,
        cache: _ScannerChunkCache,
        cancellation_token: CancellationToken | None,
    ) -> ScannerChunkResult:
        if cancellation_token:
            cancellation_token.check_cancelled()
        original_prompt = self._prompt_for(episode, chunk.chunk_id, chunk.start_ms, chunk.end_ms, chunk.parts)
        prompt = original_prompt
        last_errors: tuple[str, ...] = ()
        measurements: list[dict[str, Any]] = []
        start_attempt = 0
        end_attempt = self.config.repair_attempts
        recovered = cache.load_latest_raw(episode.episode_id, chunk.chunk_id)
        if recovered is not None:
            recovered_attempt, recovered_raw, recovered_measurement = recovered
            try:
                parsed = parse_scanner_json(recovered_raw, episode_id=episode.episode_id, chunk_id=chunk.chunk_id)
                observations = validate_scanner_response(
                    parsed,
                    episode_id=episode.episode_id,
                    source_id=episode.source_id,
                    episode_duration_ms=episode.duration_ms,
                    chunk=chunk,
                )
                pending = ScannerChunkResult(
                    chunk, observations, hashlib.sha256(recovered_raw.encode("utf-8")).hexdigest(), "",
                    (recovered_measurement,), False,
                )
                return cache.save(episode.episode_id, pending, cancellation_token=cancellation_token)
            except ScannerValidationError as exc:
                last_errors = exc.issue_codes
            except ScannerResponseError as exc:
                last_errors = ("malformed_json", str(exc))
            if self.config.repair_attempts == 0:
                raise ScannerRepairExhaustedError(
                    f"Scanner repair exhausted for {chunk.chunk_id}.",
                    validation_errors=last_errors,
                    episode_id=episode.episode_id,
                    chunk_id=chunk.chunk_id,
                    request_phase="repair",
                    retry_count=recovered_attempt,
                )
            prompt = self._repair_prompt_that_fits(original_prompt, recovered_raw, last_errors)
            start_attempt = recovered_attempt + 1
            end_attempt = start_attempt + self.config.repair_attempts - 1

        for attempt in range(start_attempt, end_attempt + 1):
            if cancellation_token:
                cancellation_token.check_cancelled()
            phase = "scanner" if recovered is None and attempt == 0 else "scanner_repair"
            self._emit(
                state=ActivityState.REPAIRING.value if phase == "scanner_repair" else ActivityState.WAITING_FOR_AI.value,
                activity_text=(
                    f"Repairing Scanner response — attempt {attempt} / {end_attempt}"
                    if phase == "scanner_repair" else "Waiting for Scanner response..."
                ),
                episode_id=episode.episode_id,
                chunk_id=chunk.chunk_id,
                current_item=chunk.chunk_id,
                waiting_for="AI",
                retry_attempt=attempt if phase == "scanner_repair" else None,
                retry_limit=end_attempt if phase == "scanner_repair" else None,
                item_event="start",
                item_key=chunk.chunk_id,
            )
            try:
                result = self.gateway_client.submit_text_chat(
                    prompt=prompt,
                    model=self.config.model,
                    system_prompt=SCANNER_SYSTEM_PROMPT,
                    reasoning_effort=self.config.reasoning or None,
                    stream=True,
                    expect_json=False,
                    cancellation_token=cancellation_token,
                    phase=phase,
                )
            except (CancelledError, ScannerCapacityError):
                raise
            except GatewayError as exc:
                raise ScannerResponseError(
                    f"Gateway failed for Scanner chunk {chunk.chunk_id}: {exc}",
                    episode_id=episode.episode_id,
                    chunk_id=chunk.chunk_id,
                    request_phase=phase,
                    retry_count=attempt,
                ) from exc
            measurement = {
                "phase": phase,
                "episode_id": episode.episode_id,
                "chunk_id": chunk.chunk_id,
                "model": self.config.model,
                "request_bytes": result.bytes_sent,
                "transcript_part_count": len(chunk.parts),
                "retry_attempt": attempt,
                "http_status": (result.metadata or {}).get("status_code"),
                "duration_ms": (result.metadata or {}).get("duration_ms"),
            }
            cache.save_raw(
                episode.episode_id, chunk.chunk_id, attempt, result.raw_response,
                self.config.max_raw_response_bytes, measurement,
            )
            measurements.append(measurement)
            self._emit(
                state=ActivityState.LOCAL_PROCESSING.value,
                activity_text="Scanner response received; validating response...",
                episode_id=episode.episode_id,
                chunk_id=chunk.chunk_id,
                current_item=chunk.chunk_id,
            )
            try:
                parsed = parse_scanner_json(result.raw_response, episode_id=episode.episode_id, chunk_id=chunk.chunk_id)
                observations = validate_scanner_response(
                    parsed,
                    episode_id=episode.episode_id,
                    source_id=episode.source_id,
                    episode_duration_ms=episode.duration_ms,
                    chunk=chunk,
                )
                response_hash = hashlib.sha256(result.raw_response.encode("utf-8")).hexdigest()
                pending = ScannerChunkResult(chunk, observations, response_hash, "", tuple(measurements), False)
                if cancellation_token:
                    cancellation_token.check_cancelled()
                return cache.save(episode.episode_id, pending, cancellation_token=cancellation_token)
            except ScannerValidationError as exc:
                last_errors = exc.issue_codes
            except ScannerResponseError as exc:
                last_errors = ("malformed_json", str(exc))
            if attempt < end_attempt:
                prompt = self._repair_prompt_that_fits(original_prompt, result.raw_response, last_errors)

        raise ScannerRepairExhaustedError(
            f"Scanner repair exhausted for {chunk.chunk_id}.",
            validation_errors=last_errors,
            episode_id=episode.episode_id,
            chunk_id=chunk.chunk_id,
            request_phase="repair",
            retry_count=end_attempt,
        )

    def _scan_episode_chunks(
        self,
        episode: PreparedEpisode,
        chunks: Sequence[ScannerChunk],
        cache: _ScannerChunkCache,
        cancellation_token: CancellationToken | None,
        on_requested_complete: Callable[[ScannerChunkResult], None] | None = None,
    ) -> tuple[ScannerChunkResult, ...]:
        results: dict[int, ScannerChunkResult] = {}
        pending_chunks: list[ScannerChunk] = []
        for chunk in chunks:
            cached = cache.load(episode.episode_id, chunk)
            if cached is not None:
                results[chunk.order] = cached
            else:
                pending_chunks.append(chunk)
        if not pending_chunks:
            return tuple(results[index] for index in sorted(results))

        iterator = iter(pending_chunks)
        with ThreadPoolExecutor(max_workers=self.config.parallelism, thread_name_prefix="scanner") as pool:
            active: dict[Future[ScannerChunkResult], ScannerChunk] = {}
            for _ in range(min(self.config.parallelism, len(pending_chunks))):
                chunk = next(iterator)
                active[pool.submit(self._scan_one, episode, chunk, cache, cancellation_token)] = chunk
            try:
                while active:
                    if cancellation_token:
                        cancellation_token.check_cancelled()
                    completed, _ = wait(active, return_when=FIRST_COMPLETED, timeout=0.1)
                    if not completed:
                        continue
                    for future in completed:
                        chunk = active.pop(future)
                        result = future.result()
                        results[chunk.order] = result
                        if on_requested_complete is not None:
                            on_requested_complete(result)
                        try:
                            next_chunk = next(iterator)
                        except StopIteration:
                            continue
                        active[pool.submit(self._scan_one, episode, next_chunk, cache, cancellation_token)] = next_chunk
            except BaseException:
                for future in active:
                    future.cancel()
                raise
        return tuple(results[index] for index in sorted(results))

    def inspect_project_progress(
        self,
        project_id: str,
        episodes: Sequence[PreparedEpisode],
    ) -> dict[str, Any]:
        """Inspect deterministic plans and validated chunk checkpoints without Gateway calls."""
        ordered = tuple(sorted(episodes, key=lambda item: item.episode_id))
        revision = compute_evidence_revision(ordered, self.config)
        cache = _ScannerChunkCache(self.storage_root, project_id, revision)
        total = completed = completed_episodes = 0
        next_chunk = next_episode = None
        episode_totals: dict[str, dict[str, int]] = {}
        for episode in ordered:
            chunks = self._plan(episode)
            episode_complete = 0
            total += len(chunks)
            for chunk in chunks:
                if cache.load(episode.episode_id, chunk) is not None:
                    completed += 1
                    episode_complete += 1
                elif next_chunk is None:
                    next_chunk = chunk.chunk_id
                    next_episode = episode.episode_id
            if episode_complete == len(chunks):
                completed_episodes += 1
            episode_totals[episode.episode_id] = {"completed": episode_complete, "total": len(chunks)}
        return {
            "completed": completed,
            "total": total,
            "percent": completed * 100.0 / total if total else None,
            "completed_episodes": completed_episodes,
            "total_episodes": len(ordered),
            "next_chunk": next_chunk,
            "next_episode": next_episode,
            "episodes": episode_totals,
            "evidence_revision": revision,
        }

    def _allocate_evidence(
        self,
        episode: PreparedEpisode,
        chunk_results: Sequence[ScannerChunkResult],
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[Evidence, ...]:
        records: list[tuple[tuple[Any, ...], ScannerObservation, ScannerChunkResult]] = []
        for chunk_result in chunk_results:
            for observation in chunk_result.observations:
                if cancellation_token:
                    cancellation_token.check_cancelled()
                content_digest = _digest({
                    "observation": " ".join(observation.observation.split()),
                    "dialogue": [item.to_dict() for item in observation.dialogue],
                    "entities": list(observation.entities),
                    "uncertainty": list(observation.uncertainty),
                    "modality": observation.modality,
                })
                key = (
                    chunk_result.chunk.order,
                    observation.start_ms,
                    observation.end_ms,
                    observation.category,
                    content_digest,
                    observation.response_index,
                )
                records.append((key, observation, chunk_result))
        records.sort(key=lambda item: item[0])
        evidence: list[Evidence] = []
        for number, (_, observation, chunk_result) in enumerate(records, start=1):
            if cancellation_token:
                cancellation_token.check_cancelled()
            evidence.append(Evidence(
                evidence_id=f"{episode.episode_id}-EV-{number:03d}",
                episode_id=episode.episode_id,
                source_id=episode.source_id,
                start_ms=observation.start_ms,
                end_ms=observation.end_ms,
                category=observation.category,
                observation=observation.observation,
                dialogue=observation.dialogue,
                entities=observation.entities,
                modality=observation.modality,
                confidence=observation.confidence,
                uncertainty=observation.uncertainty,
                visual_refs=(),
                provenance={
                    "chunk_id": chunk_result.chunk.chunk_id,
                    "chunk_hash": chunk_result.chunk.content_hash,
                    "scanner_artifact_hash": chunk_result.artifact_hash,
                    "scanner_response_hash": chunk_result.response_hash,
                    "transcript_hash": episode.transcript.provenance_hash,
                },
            ))
        return tuple(evidence)

    def scan_project(
        self,
        project_id: str,
        episodes: Sequence[PreparedEpisode],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ScannerProjectResult:
        if not episodes:
            raise ValueError("At least one prepared episode is required.")
        if cancellation_token:
            cancellation_token.check_cancelled()
        ordered = tuple(sorted(episodes, key=lambda item: item.episode_id))
        revision = compute_evidence_revision(ordered, self.config)
        signature = scanner_dependency_signature(ordered, self.config)
        cache = _ScannerChunkCache(self.storage_root, project_id, revision)
        store = EvidenceStore(self.storage_root, project_id)
        episode_counts: dict[str, int] = {}
        measurements: list[dict[str, Any]] = []
        reused = requested = 0
        plans = {episode.episode_id: self._plan(episode) for episode in ordered}
        initial = self.inspect_project_progress(project_id, ordered)
        completed_chunks = initial["completed"]
        self._emit(
            state=ActivityState.RUNNING.value,
            activity_text=(
                f"Reused {completed_chunks} validated Scanner chunks; continuing from {initial['next_chunk']}."
                if completed_chunks else "Scanner chunk plan ready."
            ),
            episode_id=initial["next_episode"],
            chunk_id=initial["next_chunk"],
            current_item=initial["next_chunk"],
            completed=completed_chunks,
            total=initial["total"],
            unit="chunks",
            reused=completed_chunks,
        )

        def _requested_complete(result: ScannerChunkResult) -> None:
            nonlocal completed_chunks
            completed_chunks += 1
            self._emit(
                state=ActivityState.RUNNING.value,
                activity_text="Scanner chunk completed and checkpoint saved.",
                chunk_id=result.chunk.chunk_id,
                current_item=result.chunk.chunk_id,
                completed=completed_chunks,
                total=initial["total"],
                unit="chunks",
                reused=initial["completed"],
                item_event="complete",
                item_key=result.chunk.chunk_id,
            )

        for episode in ordered:
            if cancellation_token:
                cancellation_token.check_cancelled()
            chunks = plans[episode.episode_id]
            chunk_results = self._scan_episode_chunks(
                episode, chunks, cache, cancellation_token,
                on_requested_complete=_requested_complete,
            )
            evidence = self._allocate_evidence(episode, chunk_results, cancellation_token)
            store.save_episode(
                revision,
                episode.episode_id,
                evidence,
                dependency_digest=_digest({
                    "project": signature,
                    "episode": episode.episode_id,
                    "chunks": [chunk.content_hash for chunk in chunks],
                }),
                cancellation_token=cancellation_token,
            )
            episode_counts[episode.episode_id] = len(evidence)
            for result in chunk_results:
                if result.reused:
                    reused += 1
                else:
                    requested += 1
                    measurements.extend(result.measurements)
        if cancellation_token:
            cancellation_token.check_cancelled()
        store.commit_revision(
            revision,
            [episode.episode_id for episode in ordered],
            dependency_signature=signature,
            cancellation_token=cancellation_token,
        )
        self._emit(
            state=ActivityState.LOCAL_PROCESSING.value,
            activity_text="Full Episode Evidence checkpoint completed.",
            completed=initial["total"], total=initial["total"], unit="chunks",
            reused=reused, stage_status="complete",
        )
        return ScannerProjectResult(
            project_id=project_id,
            evidence_revision=revision,
            episode_counts=episode_counts,
            total_evidence_count=sum(episode_counts.values()),
            request_measurements=tuple(measurements),
            reused_chunk_count=reused,
            requested_chunk_count=requested,
        )
