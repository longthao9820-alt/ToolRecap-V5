"""Exact, non-fuzzy Full Evidence retrieval for Season Planner requests."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Sequence

from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer.catalog import SeasonEvidenceCatalog, catalog_detail_hash
from toolrecap_v4.analysis.models import Evidence
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    PlannerEvidenceIntegrityError,
    PlannerEvidenceNotFoundError,
    PlannerEvidenceRequestError,
    PlannerEvidenceRevisionError,
)

EVIDENCE_FETCH_PROTOCOL_VERSION = "planner-evidence-fetch-v1"


@dataclass(frozen=True)
class EvidenceRequest:
    request_id: str
    type: str
    evidence_ids: tuple[str, ...] = ()
    episode_id: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.type == "evidence_ids":
            return {"request_id": self.request_id, "type": self.type, "evidence_ids": list(self.evidence_ids)}
        return {
            "request_id": self.request_id, "type": self.type, "episode_id": self.episode_id,
            "start_ms": self.start_ms, "end_ms": self.end_ms,
        }


@dataclass(frozen=True)
class EvidenceRequestResult:
    request: EvidenceRequest
    evidence_ids: tuple[str, ...]
    total_count: int
    returned_count: int
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(), "evidence_ids": list(self.evidence_ids),
            "total_count": self.total_count, "returned_count": self.returned_count,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class EvidenceFetchResult:
    protocol_version: str
    round_id: str
    evidence_revision: str
    request_results: tuple[EvidenceRequestResult, ...]
    items: tuple[Evidence, ...]
    completeness: dict[str, Any]
    result_hash: str

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version, "round_id": self.round_id,
            "evidence_revision": self.evidence_revision,
            "request_results": [item.to_dict() for item in self.request_results],
            "items": [item.to_dict() for item in self.items],
            "completeness": dict(self.completeness),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.semantic_dict()
        value["result_hash"] = self.result_hash
        return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class EvidenceFetcher:
    """Fetch exact IDs/ranges from authoritative Full Evidence without ranking or substitution."""

    def __init__(
        self,
        *,
        project_id: str,
        evidence_revision: str,
        catalog: SeasonEvidenceCatalog,
        evidence_store: EvidenceStore,
    ) -> None:
        if catalog.project_id != project_id:
            raise PlannerEvidenceRevisionError("Catalog belongs to another project.", project_id=project_id)
        if catalog.evidence_revision != evidence_revision:
            raise PlannerEvidenceRevisionError("Catalog Evidence revision is stale.", project_id=project_id)
        evidence_store.verify_revision(evidence_revision)
        self.project_id = project_id
        self.evidence_revision = evidence_revision
        self.catalog = catalog
        self.evidence_store = evidence_store
        self.catalog_items = {item.evidence_id: item for item in catalog.items}
        self.episodes = {episode.episode_id: episode for episode in catalog.ordered_episodes}
        self.catalog_order = {item.evidence_id: index for index, item in enumerate(catalog.items)}

    def _verify_full(self, evidence: Evidence) -> None:
        item = self.catalog_items.get(evidence.evidence_id)
        if item is None:
            raise PlannerEvidenceNotFoundError(
                f"Evidence ID is not present in active Catalog: {evidence.evidence_id}",
                project_id=self.project_id,
            )
        if evidence.episode_id != item.episode_id or evidence.source_id != item.source_id:
            raise PlannerEvidenceIntegrityError("Full Evidence identity differs from Catalog.", project_id=self.project_id)
        if catalog_detail_hash(evidence) != item.detail_hash:
            raise PlannerEvidenceIntegrityError(
                f"Full Evidence detail hash mismatch: {evidence.evidence_id}", project_id=self.project_id
            )

    def _validate_request(self, request: EvidenceRequest) -> None:
        if not isinstance(request.request_id, str) or not request.request_id:
            raise PlannerEvidenceRequestError("Evidence request_id must be a non-empty string.")
        if request.type == "evidence_ids":
            if not request.evidence_ids:
                raise PlannerEvidenceRequestError("Evidence ID request cannot be empty.")
            if len(request.evidence_ids) != len(set(request.evidence_ids)):
                raise PlannerEvidenceRequestError("Duplicate requested Evidence IDs are forbidden.")
            for evidence_id in request.evidence_ids:
                if not re.fullmatch(r"E\d+-EV-\d+", evidence_id):
                    raise PlannerEvidenceNotFoundError(f"Malformed Evidence ID: {evidence_id}")
                if evidence_id not in self.catalog_items:
                    raise PlannerEvidenceNotFoundError(f"Unknown Evidence ID: {evidence_id}")
        elif request.type == "episode_range":
            episode = self.episodes.get(request.episode_id or "")
            if episode is None:
                raise PlannerEvidenceRequestError(f"Unknown episode_id: {request.episode_id}")
            if type(request.start_ms) is not int or type(request.end_ms) is not int:
                raise PlannerEvidenceRequestError("Range timestamps must be integers and not booleans.")
            if request.start_ms < 0 or request.end_ms <= request.start_ms or request.end_ms > episode.duration_ms:
                raise PlannerEvidenceRequestError("Range must satisfy 0 <= start_ms < end_ms <= episode duration.")
        else:
            raise PlannerEvidenceRequestError(f"Unsupported Evidence request type: {request.type}")

    def fetch(
        self,
        *,
        round_id: str,
        requests: Sequence[EvidenceRequest],
        cancellation_token: CancellationToken | None = None,
    ) -> EvidenceFetchResult:
        if not requests:
            raise PlannerEvidenceRequestError("Planner Evidence request list cannot be empty.")
        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise PlannerEvidenceRequestError("Planner request IDs must be unique.")
        all_items: dict[str, Evidence] = {}
        results: list[EvidenceRequestResult] = []
        for request in requests:
            if cancellation_token:
                cancellation_token.check_cancelled()
            self._validate_request(request)
            if request.type == "evidence_ids":
                fetched: list[Evidence] = []
                for evidence_id in request.evidence_ids:
                    full = self.evidence_store.get(evidence_id, self.evidence_revision)
                    if full is None:
                        raise PlannerEvidenceNotFoundError(f"Full Evidence not found: {evidence_id}")
                    self._verify_full(full)
                    fetched.append(full)
                fetched.sort(key=lambda item: self.catalog_order[item.evidence_id])
            else:
                query = self.evidence_store.query_range(
                    request.episode_id or "", request.start_ms, request.end_ms, self.evidence_revision
                )
                fetched = list(query.items)
                for full in fetched:
                    self._verify_full(full)
                fetched.sort(key=lambda item: self.catalog_order[item.evidence_id])
            for full in fetched:
                all_items[full.evidence_id] = full
            ids = tuple(item.evidence_id for item in fetched)
            results.append(EvidenceRequestResult(request, ids, len(ids), len(ids), True))
        ordered = tuple(sorted(all_items.values(), key=lambda item: self.catalog_order[item.evidence_id]))
        completeness = {
            "request_count": len(requests), "completed_request_count": len(results),
            "unique_item_count": len(ordered), "complete": True,
        }
        semantic = {
            "protocol_version": EVIDENCE_FETCH_PROTOCOL_VERSION,
            "round_id": round_id, "evidence_revision": self.evidence_revision,
            "request_results": [item.to_dict() for item in results],
            "items": [item.to_dict() for item in ordered], "completeness": completeness,
        }
        return EvidenceFetchResult(
            EVIDENCE_FETCH_PROTOCOL_VERSION, round_id, self.evidence_revision,
            tuple(results), ordered, completeness, _digest(semantic),
        )
