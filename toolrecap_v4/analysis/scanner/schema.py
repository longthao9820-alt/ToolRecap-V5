"""Strict Scanner response schema and deterministic validation."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from toolrecap_v4.analysis.models import DialogueReference
from toolrecap_v4.analysis.scanner.chunking import ScannerChunk
from toolrecap_v4.errors import ScannerResponseError, ScannerValidationError

SCANNER_VALIDATION_VERSION = "scanner-validation-v1"
ALLOWED_CATEGORIES = frozenset({"action", "dialogue", "reaction", "event", "chronology", "entity", "location", "other"})
ALLOWED_MODALITIES = frozenset({"subtitle", "ocr", "stt", "mixed"})


@dataclass(frozen=True)
class ScannerObservation:
    start_ms: int
    end_ms: int
    category: str
    observation: str
    dialogue: tuple[DialogueReference, ...]
    entities: tuple[str, ...]
    modality: str
    confidence: float | None
    uncertainty: tuple[str, ...]
    response_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "category": self.category,
            "observation": self.observation,
            "dialogue": [item.to_dict() for item in self.dialogue],
            "entities": list(self.entities),
            "modality": self.modality,
            "confidence": self.confidence,
            "uncertainty": list(self.uncertainty),
            "response_index": self.response_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScannerObservation:
        return cls(
            start_ms=data["start_ms"],
            end_ms=data["end_ms"],
            category=str(data["category"]),
            observation=str(data["observation"]),
            dialogue=tuple(DialogueReference.from_dict(item) for item in data.get("dialogue", [])),
            entities=tuple(str(item) for item in data.get("entities", [])),
            modality=str(data["modality"]),
            confidence=data.get("confidence"),
            uncertainty=tuple(str(item) for item in data.get("uncertainty", [])),
            response_index=int(data.get("response_index", 0)),
        )


def parse_scanner_json(raw_response: str, *, episode_id: str, chunk_id: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ScannerValidationError(
                    f"Scanner response contains duplicate JSON key: {key}",
                    issue_codes=("duplicate_json_key",), episode_id=episode_id,
                    chunk_id=chunk_id, request_phase="validation",
                )
            value[key] = item
        return value

    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    try:
        parsed = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except ScannerValidationError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise ScannerResponseError(
            f"Scanner returned malformed JSON for {chunk_id}: {exc}",
            episode_id=episode_id,
            chunk_id=chunk_id,
            request_phase="validation",
        ) from exc
    if isinstance(parsed, dict) and set(parsed) == {"choices"}:
        choices = parsed.get("choices")
        if (not isinstance(choices, list) or len(choices) != 1 or
            not isinstance(choices[0], dict) or
            not isinstance(choices[0].get("message"), dict) or
            not isinstance(choices[0]["message"].get("content"), str)):
            raise ScannerValidationError(
                "Scanner provider wrapper is malformed.", issue_codes=("provider_wrapper",),
                episode_id=episode_id, chunk_id=chunk_id, request_phase="validation",
            )
        return parse_scanner_json(choices[0]["message"]["content"], episode_id=episode_id, chunk_id=chunk_id)
    if not isinstance(parsed, dict):
        raise ScannerValidationError(
            "Scanner response root must be an object.",
            issue_codes=("root_type",),
            episode_id=episode_id,
            chunk_id=chunk_id,
            request_phase="validation",
        )
    return parsed


def _string_list(value: Any, field: str, issues: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list):
        issues.append(f"{field}_type")
        return ()
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            issues.append(f"{field}_item")
            continue
        result.append(item.strip())
    return tuple(result)


def validate_scanner_response(
    data: dict[str, Any],
    *,
    episode_id: str,
    source_id: str,
    episode_duration_ms: int,
    chunk: ScannerChunk,
) -> tuple[ScannerObservation, ...]:
    """Reject every identity, timestamp, enum, type, and provenance defect without coercion."""
    issues: list[str] = []
    root_required = {"episode_id", "source_id", "chunk_id", "observations"}
    root_extra = set(data) - root_required
    if root_required - set(data):
        issues.append("root_missing_fields")
    if root_extra:
        issues.append("root_extra_fields")
    if data.get("episode_id") != episode_id:
        issues.append("episode_id_mismatch")
    if data.get("source_id") != source_id:
        issues.append("source_id_mismatch")
    if data.get("chunk_id") != chunk.chunk_id:
        issues.append("chunk_id_mismatch")
    raw_observations = data.get("observations")
    if not isinstance(raw_observations, list):
        issues.append("observations_type")
        raw_observations = []

    part_lookup = {(part.cue_id, part.part_index): part for part in chunk.parts}
    observations: list[ScannerObservation] = []
    for index, item in enumerate(raw_observations):
        prefix = f"observation_{index}"
        if not isinstance(item, dict):
            issues.append(f"{prefix}_type")
            continue
        observation_required = {
            "start_ms", "end_ms", "category", "observation", "cue_refs",
            "entities", "modality", "confidence", "uncertainty",
        }
        if observation_required - set(item):
            issues.append(f"{prefix}_missing_fields")
        if set(item) - observation_required:
            issues.append(f"{prefix}_extra_fields")
        start = item.get("start_ms")
        end = item.get("end_ms")
        if type(start) is not int:
            issues.append(f"{prefix}_start_ms_type")
        if type(end) is not int:
            issues.append(f"{prefix}_end_ms_type")
        valid_times = type(start) is int and type(end) is int
        if valid_times:
            if start < 0:
                issues.append(f"{prefix}_start_negative")
            if end <= start:
                issues.append(f"{prefix}_range_order")
            if end > episode_duration_ms:
                issues.append(f"{prefix}_episode_range")
            if start < chunk.start_ms or end > chunk.end_ms:
                issues.append(f"{prefix}_chunk_range")

        category = item.get("category")
        if not isinstance(category, str) or category not in ALLOWED_CATEGORIES:
            issues.append(f"{prefix}_category")
        observation = item.get("observation")
        if not isinstance(observation, str) or not observation.strip():
            issues.append(f"{prefix}_observation")
        modality = item.get("modality")
        if not isinstance(modality, str) or modality not in ALLOWED_MODALITIES:
            issues.append(f"{prefix}_modality")

        entities = _string_list(item.get("entities", []), f"{prefix}_entities", issues)
        uncertainty = _string_list(item.get("uncertainty", []), f"{prefix}_uncertainty", issues)
        confidence = item.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                issues.append(f"{prefix}_confidence_type")
            elif not 0.0 <= float(confidence) <= 1.0:
                issues.append(f"{prefix}_confidence_range")

        refs_raw = item.get("cue_refs")
        refs: list[DialogueReference] = []
        if not isinstance(refs_raw, list) or not refs_raw:
            issues.append(f"{prefix}_cue_refs")
        else:
            for ref_index, ref in enumerate(refs_raw):
                if not isinstance(ref, dict):
                    issues.append(f"{prefix}_cue_ref_{ref_index}_type")
                    continue
                if set(ref) != {"cue_id", "part_index"}:
                    issues.append(f"{prefix}_cue_ref_{ref_index}_fields")
                    continue
                cue_id = ref.get("cue_id")
                part_index = ref.get("part_index")
                if not isinstance(cue_id, str) or type(part_index) is not int:
                    issues.append(f"{prefix}_cue_ref_{ref_index}_fields")
                    continue
                part = part_lookup.get((cue_id, part_index))
                if part is None:
                    issues.append(f"{prefix}_cue_ref_{ref_index}_unknown")
                    continue
                refs.append(DialogueReference(part.cue_id, part.part_index, part.part_count, part.text))
            if valid_times and refs:
                ref_start = min(part_lookup[(ref.cue_id, ref.part_index)].start_ms for ref in refs)
                ref_end = max(part_lookup[(ref.cue_id, ref.part_index)].end_ms for ref in refs)
                if start < ref_start or end > ref_end:
                    issues.append(f"{prefix}_cue_grounding")

        if not any(code.startswith(prefix) for code in issues):
            observations.append(ScannerObservation(
                start_ms=start,
                end_ms=end,
                category=category,
                observation=observation.strip(),
                dialogue=tuple(refs),
                entities=entities,
                modality=modality,
                confidence=float(confidence) if confidence is not None else None,
                uncertainty=uncertainty,
                response_index=index,
            ))

    if issues:
        raise ScannerValidationError(
            f"Scanner response failed validation for {chunk.chunk_id}: {', '.join(issues)}",
            issue_codes=tuple(issues),
            episode_id=episode_id,
            chunk_id=chunk.chunk_id,
            request_phase="validation",
        )
    return tuple(observations)
