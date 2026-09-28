"""Bounded Season Planner rounds, exact Evidence fetch, repair, persistence, and resume."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer.catalog import SeasonEvidenceCatalog
from toolrecap_v4.analysis.finalizer.evidence_fetch import EvidenceFetcher
from toolrecap_v4.analysis.finalizer.planner import (
    PLANNER_SYSTEM_PROMPT,
    PlannerAction,
    PlannerDraft,
    build_followup_planner_prompt,
    build_initial_planner_prompt,
    build_planner_repair_prompt,
    parse_planner_json,
    planner_dependency_signature,
    planner_session_id,
    validate_planner_action,
)
from toolrecap_v4.analysis.finalizer.planner_store import PlannerStore
from toolrecap_v4.analysis.scanner.prompts import measure_text_request_bytes
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    CancelledError,
    GatewayError,
    PayloadContextError,
    PlannerCapacityError,
    PlannerRepairExhaustedError,
    PlannerResponseError,
    PlannerRoundLimitError,
    PlannerSessionError,
    PlannerValidationError,
)
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.progress import ActivityState, WorkflowStage, safe_emit

MAX_PLANNER_RAW_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class PlannerConfig:
    model: str
    reasoning: str = ""
    max_rounds: int = 3
    repair_attempts: int = 1
    max_request_bytes: int | None = None
    max_raw_response_bytes: int = MAX_PLANNER_RAW_RESPONSE_BYTES

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Planner model must be configured.")
        if self.max_rounds < 1:
            raise ValueError("Planner max_rounds must be at least 1.")
        if self.repair_attempts < 0:
            raise ValueError("Planner repair_attempts must be non-negative.")
        if self.max_request_bytes is not None and self.max_request_bytes < 1024:
            raise ValueError("Planner max_request_bytes must be null or at least 1024.")


@dataclass(frozen=True)
class PlannerRunResult:
    project_id: str
    session_id: str
    dependency_digest: str
    draft: PlannerDraft
    round_count: int
    request_measurements: tuple[dict[str, Any], ...]
    reused: bool
    draft_path: Path


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cumulative_evidence_fetch(
    catalog: SeasonEvidenceCatalog, fetched_rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    """Carry every previously fetched Full Evidence item into the next round."""
    by_id: dict[str, dict[str, Any]] = {}
    round_ids: list[str] = []
    for fetched in fetched_rounds:
        if (not isinstance(fetched, dict)
                or fetched.get("evidence_revision") != catalog.evidence_revision
                or not isinstance(fetched.get("completeness"), dict)
                or fetched["completeness"].get("complete") is not True
                or not isinstance(fetched.get("items"), list)):
            raise PlannerSessionError("Saved Planner Evidence fetch is incomplete or stale.")
        round_ids.append(str(fetched["round_id"]))
        for item in fetched["items"]:
            if not isinstance(item, dict) or not isinstance(item.get("evidence_id"), str):
                raise PlannerSessionError("Saved Planner Evidence item is malformed.")
            evidence_id = item["evidence_id"]
            if evidence_id in by_id and by_id[evidence_id] != item:
                raise PlannerSessionError(f"Conflicting Full Evidence for {evidence_id} across Planner rounds.")
            by_id[evidence_id] = item
    ordered = [by_id.pop(item.evidence_id) for item in catalog.items if item.evidence_id in by_id]
    if by_id:
        raise PlannerSessionError("Planner fetch contains Evidence absent from the active Catalog.")
    payload = {
        "protocol_version": "planner-evidence-cumulative-v1",
        "evidence_revision": catalog.evidence_revision,
        "source_round_ids": round_ids,
        "items": ordered,
        "completeness": {
            "complete": True, "retrieval_round_count": len(round_ids),
            "unique_item_count": len(ordered),
        },
    }
    payload["result_hash"] = _digest(payload)
    return payload


class PlannerService:
    def __init__(
        self, gateway_client: GatewayClient, storage_root: Path | str, config: PlannerConfig,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.gateway_client = gateway_client
        self.storage_root = Path(storage_root).resolve()
        self.config = config
        self.progress_callback = progress_callback

    def _emit(self, **payload: Any) -> None:
        stage = payload.pop("stage", WorkflowStage.PLANNER.value)
        safe_emit(self.progress_callback, stage=stage, **payload)

    def _capacity(self, prompt: str) -> tuple[int, str]:
        size = measure_text_request_bytes(
            model=self.config.model, reasoning=self.config.reasoning,
            user_prompt=prompt, system_prompt=PLANNER_SYSTEM_PROMPT,
        )
        if self.config.max_request_bytes is None:
            return size, "UNKNOWN"
        if size > self.config.max_request_bytes:
            raise PlannerCapacityError(
                f"Complete Planner request is {size} bytes, exceeding configured limit {self.config.max_request_bytes}."
            )
        return size, "FIT"

    def _repair_prompt(self, original: str, invalid: str, errors: tuple[str, ...]) -> str:
        text = invalid
        while True:
            prompt = build_planner_repair_prompt(
                original_prompt=original, invalid_response=text, validation_errors=errors,
            )
            try:
                self._capacity(prompt)
                return prompt
            except PlannerCapacityError:
                if not text:
                    raise
                text = text[: len(text) // 2]

    def _request_round(
        self,
        *,
        project_id: str,
        round_id: str,
        prompt: str,
        catalog: SeasonEvidenceCatalog,
        dependency_digest: str,
        store: PlannerStore,
        cancellation_token: CancellationToken | None,
        measurements: list[dict[str, Any]],
        require_draft: bool = False,
    ) -> PlannerAction:
        def _validate_response(raw: str) -> PlannerAction:
            parsed = parse_planner_json(raw, project_id=project_id, round_id=round_id)
            action = validate_planner_action(
                parsed, project_id=project_id, round_id=round_id, catalog=catalog,
            )
            if require_draft and action.action != "PLANNER_DRAFT":
                raise PlannerValidationError(
                    "Final Planner synthesis round must return PLANNER_DRAFT.",
                    issue_codes=("final_draft_required",),
                    project_id=project_id, round_id=round_id,
                )
            return action

        recovered = store.load_latest_raw(round_id, dependency_digest)
        next_attempt = 0
        invalid_raw = ""
        errors: tuple[str, ...] = ()
        if recovered is not None:
            recovered_attempt, invalid_raw = recovered
            try:
                return _validate_response(invalid_raw)
            except PlannerValidationError as exc:
                errors = exc.issue_codes
            except PlannerResponseError as exc:
                errors = ("malformed_json", str(exc))
            next_attempt = recovered_attempt + 1
            if self.config.repair_attempts == 0:
                raise PlannerRepairExhaustedError(
                    f"Planner repair exhausted for {round_id}.", project_id=project_id, round_id=round_id
                )

        request_count = self.config.repair_attempts if recovered is not None else self.config.repair_attempts + 1
        current_prompt = self._repair_prompt(prompt, invalid_raw, errors) if recovered is not None else prompt
        for offset in range(request_count):
            attempt = next_attempt + offset
            if cancellation_token:
                cancellation_token.check_cancelled()
            measured, capacity = self._capacity(current_prompt)
            phase = "season_planner" if recovered is None and offset == 0 else "season_planner_repair"
            self._emit(
                state=ActivityState.REPAIRING.value if phase.endswith("repair") else ActivityState.WAITING_FOR_AI.value,
                activity_text=(
                    f"Repairing Planner response — attempt {attempt} / {next_attempt + request_count - 1}"
                    if phase.endswith("repair") else f"Waiting for Planner response ({round_id})..."
                ),
                current_item=round_id, waiting_for="AI",
                retry_attempt=attempt if phase.endswith("repair") else None,
                retry_limit=next_attempt + request_count - 1 if phase.endswith("repair") else None,
                item_event="start", item_key=round_id,
            )
            try:
                result = self.gateway_client.submit_text_chat(
                    prompt=current_prompt, model=self.config.model,
                    system_prompt=PLANNER_SYSTEM_PROMPT,
                    reasoning_effort=self.config.reasoning or None,
                    stream=True, expect_json=False,
                    cancellation_token=cancellation_token, phase=phase,
                )
            except PayloadContextError as exc:
                raise PlannerCapacityError(
                    f"Gateway rejected the complete Planner request for context capacity: {exc}",
                    project_id=project_id, round_id=round_id,
                ) from exc
            except CancelledError:
                raise
            except GatewayError as exc:
                raise PlannerResponseError(
                    f"Gateway failed for Planner {round_id}: {exc}", project_id=project_id, round_id=round_id
                ) from exc
            measurement = {
                "phase": phase, "round_id": round_id, "model": self.config.model,
                "reasoning": self.config.reasoning, "request_bytes": result.bytes_sent,
                "preflight_bytes": measured, "capacity_status": capacity,
                "catalog_item_count": len(catalog.items), "retry_attempt": attempt,
                "http_status": (result.metadata or {}).get("status_code"),
                "duration_ms": (result.metadata or {}).get("duration_ms"),
            }
            measurements.append(measurement)
            self._emit(
                state=ActivityState.LOCAL_PROCESSING.value,
                activity_text=f"Planner response received; validating {round_id}...",
                current_item=round_id,
            )
            store.save_raw(
                round_id, attempt, result.raw_response,
                limit=self.config.max_raw_response_bytes,
                dependency_digest=dependency_digest, measurement=measurement,
            )
            try:
                return _validate_response(result.raw_response)
            except PlannerValidationError as exc:
                errors = exc.issue_codes
            except PlannerResponseError as exc:
                errors = ("malformed_json", str(exc))
            if offset + 1 < request_count:
                current_prompt = self._repair_prompt(prompt, result.raw_response, errors)
        raise PlannerRepairExhaustedError(
            f"Planner repair exhausted for {round_id}: {', '.join(errors)}",
            project_id=project_id, round_id=round_id,
        )

    def run(
        self,
        *,
        project_id: str,
        raw_recap_prompt: str,
        catalog: SeasonEvidenceCatalog,
        cancellation_token: CancellationToken | None = None,
    ) -> PlannerRunResult:
        if not isinstance(raw_recap_prompt, str):
            raise TypeError("raw_recap_prompt must be a string")
        if cancellation_token:
            cancellation_token.check_cancelled()
        dependencies = planner_dependency_signature(
            project_id=project_id, raw_recap_prompt=raw_recap_prompt, catalog=catalog,
            model=self.config.model, reasoning=self.config.reasoning,
        )
        dependency_digest = _digest(dependencies)
        session_id = planner_session_id(dependencies)
        store = PlannerStore(self.storage_root, project_id, session_id)
        store.initialize(dependency_digest, dependencies)
        cached_draft = store.load_draft(dependency_digest)
        if cached_draft is not None:
            round_id = cached_draft.get("round_id", "")
            action = validate_planner_action(
                cached_draft, project_id=project_id, round_id=round_id, catalog=catalog,
            )
            self._emit(
                state=ActivityState.RUNNING.value,
                activity_text="Reused validated Planner Draft checkpoint.",
                current_item=round_id, reused=1, stage_status="complete",
            )
            return PlannerRunResult(
                project_id, session_id, dependency_digest, action.draft,
                int(round_id.rsplit("-", 1)[-1]), (), True, store.draft_path,
            )

        evidence_store = EvidenceStore(self.storage_root, project_id)
        fetcher = EvidenceFetcher(
            project_id=project_id, evidence_revision=catalog.evidence_revision,
            catalog=catalog, evidence_store=evidence_store,
        )
        measurements: list[dict[str, Any]] = []
        fetched_rounds: list[dict[str, Any]] = []
        previous_round_id = ""
        # The configured rounds bound Evidence retrieval. One final synthesis
        # round receives every fetched item and must produce a Planner Draft.
        for round_number in range(1, self.config.max_rounds + 2):
            if cancellation_token:
                cancellation_token.check_cancelled()
            round_id = f"round-{round_number:03d}"
            final_draft_round = round_number > self.config.max_rounds
            self._emit(
                state=ActivityState.RUNNING.value,
                activity_text=f"Planner round {round_number}: building request...",
                current_item=round_id,
            )
            completed = store.load_round(round_id, dependency_digest)
            if completed is not None:
                action = validate_planner_action(
                    completed["action"], project_id=project_id, round_id=round_id, catalog=catalog,
                )
                fetch_dict = completed["fetch"]
            else:
                if round_number == 1:
                    prompt = build_initial_planner_prompt(
                        project_id=project_id, round_id=round_id,
                        raw_recap_prompt=raw_recap_prompt, catalog=catalog,
                        evidence_round_budget=self.config.max_rounds,
                    )
                else:
                    prompt = build_followup_planner_prompt(
                        project_id=project_id, round_id=round_id,
                        raw_recap_prompt=raw_recap_prompt, catalog=catalog,
                        prior_round_id=previous_round_id,
                        evidence_fetch=_cumulative_evidence_fetch(catalog, fetched_rounds),
                        final_draft_round=final_draft_round,
                        evidence_rounds_remaining=max(0, self.config.max_rounds - round_number + 1),
                    )
                action = self._request_round(
                    project_id=project_id, round_id=round_id, prompt=prompt,
                    catalog=catalog, dependency_digest=dependency_digest, store=store,
                    cancellation_token=cancellation_token, measurements=measurements,
                    require_draft=final_draft_round,
                )
                fetch_dict = None
                if action.action == "REQUEST_EVIDENCE":
                    if final_draft_round:
                        raise PlannerRoundLimitError(
                            "Planner requested Evidence after its final synthesis round.",
                            project_id=project_id, round_id=round_id,
                        )
                    requested = len(action.requests)
                    self._emit(
                        state=ActivityState.LOCAL_PROCESSING.value,
                        stage=WorkflowStage.EVIDENCE.value,
                        activity_text=f"Fetching {requested} requested Full Evidence groups...",
                        current_item=round_id, completed=0, total=requested, unit="requests",
                    )
                    fetch = fetcher.fetch(
                        round_id=round_id, requests=action.requests,
                        cancellation_token=cancellation_token,
                    )
                    fetch_dict = fetch.to_dict()
                    self._emit(
                        state=ActivityState.LOCAL_PROCESSING.value,
                        stage=WorkflowStage.EVIDENCE.value,
                        activity_text="Requested Full Evidence fetched.",
                        current_item=round_id, completed=requested, total=requested,
                        unit="requests", stage_status="complete",
                    )
                store.save_round(
                    round_id, dependency_digest=dependency_digest,
                    action=action.to_dict(), fetch=fetch_dict,
                    cancellation_token=cancellation_token,
                )
            if action.action == "PLANNER_DRAFT":
                store.save_draft(
                    dependency_digest=dependency_digest, action=action.to_dict(),
                    cancellation_token=cancellation_token,
                )
                self._emit(
                    state=ActivityState.LOCAL_PROCESSING.value,
                    activity_text="Planner Draft validated and checkpointed.",
                    current_item=round_id, stage_status="complete",
                    item_event="complete", item_key=round_id,
                )
                return PlannerRunResult(
                    project_id, session_id, dependency_digest, action.draft,
                    round_number, tuple(measurements), False, store.draft_path,
                )
            if final_draft_round:
                raise PlannerRoundLimitError(
                    "Planner requested Evidence after its final synthesis round.",
                    project_id=project_id, round_id=round_id,
                )
            if fetch_dict is None:
                raise PlannerSessionError(
                    f"Saved Planner request round {round_id} lacks its Evidence fetch.",
                    project_id=project_id, round_id=round_id,
                )
            fetched_rounds.append(fetch_dict)
            previous_round_id = round_id
        raise PlannerRoundLimitError("Planner round limit exhausted.", project_id=project_id)
