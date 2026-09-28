"""Semantic validation and durable AI request storage for Highlight planning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from toolrecap_v4.persistence import atomic_write_json, read_json
from .models import build_highlight_project


def validate_candidates(candidates: list[dict], payload: dict) -> None:
    sources = payload.get("sources") or [payload["source"]]
    build_highlight_project(project_id="validation", prompt="", dependency_revision="validation",
                            candidates=candidates, sources=sources)
    rows = payload.get("season_evidence")
    if rows is None:
        rows = {payload["source"]["episode_id"]: payload.get("factual_evidence", [])}
    if payload.get("evidence_index"):
        rows = {**rows, "_prior_references": payload["evidence_index"]}
    evidence = {row["evidence_id"]: row for items in rows.values() for row in items}
    for index, candidate in enumerate(candidates):
        for clip_index, clip in enumerate(candidate["clips"]):
            path = f"$.candidates[{index}].clips[{clip_index}]"
            refs = clip["evidence_ids"]
            if not refs:
                raise ValueError(f"{path}: at least one supplied Evidence ID is required")
            overlapping = False
            for ref in refs:
                item = evidence.get(ref)
                if item is None:
                    raise ValueError(f"{path}: unknown Evidence ID {ref}")
                if item.get("episode_id") != clip["episode_id"] or item.get("source_id") != clip["source_id"]:
                    raise ValueError(f"{path}: Evidence ID {ref} belongs to another episode/source")
                overlapping |= item["start_ms"] < clip["end_ms"] and item["end_ms"] > clip["start_ms"]
            if not overlapping:
                raise ValueError(f"{path}: cited evidence does not overlap the selected clip")


class PlanningResponses:
    """Persist complete raw responses before parsing; verify hashes on reuse."""

    def __init__(self, directory: Path):
        self.directory = directory

    def key(self, request: dict) -> str:
        return hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def load(self, request: dict) -> str | None:
        try:
            stored = read_json(self.directory / f"{self.key(request)}.json")
            raw = stored["raw"]
            if stored["hash"] == hashlib.sha256(raw.encode()).hexdigest():
                return raw
        except (OSError, ValueError, KeyError, TypeError):
            pass
        except Exception as exc:
            from toolrecap_v4.errors import PersistenceError
            if not isinstance(exc, PersistenceError):
                raise
        return None

    def save(self, request: dict, raw: str) -> None:
        atomic_write_json(self.directory / f"{self.key(request)}.json", {
            "raw": raw, "hash": hashlib.sha256(raw.encode()).hexdigest(),
        })
