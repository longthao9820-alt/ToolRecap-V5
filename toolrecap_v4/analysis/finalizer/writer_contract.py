"""Single authoritative Writer Draft response protocol."""
from __future__ import annotations

import json
from typing import Any

WRITER_DRAFT_VERSION = "writer-draft-v1"
WRITER_ROOT_FIELDS = frozenset({
    "writer_draft_version", "project_id", "season_plan_hash", "output_id",
    "title", "narration", "segments", "writer_notes",
})
WRITER_SEGMENT_FIELDS = frozenset({
    "segment_id", "narration_text", "episode_ids", "evidence_ids",
    "visual_evidence_ids", "source_clips", "editorial_intent", "uncertainty",
})
SOURCE_CLIP_FIELDS = frozenset({"episode_id", "source_id", "start_ms", "end_ms"})


def writer_draft_skeleton(project_id: str, plan_hash: str, output_id: str) -> dict[str, Any]:
    return {
        "writer_draft_version": WRITER_DRAFT_VERSION,
        "project_id": project_id,
        "season_plan_hash": plan_hash,
        "output_id": output_id,
        "title": "<nonempty title>",
        "narration": {"text": "<nonempty complete narration>"},
        "segments": [{
            "segment_id": "<unique nonempty segment id>",
            "narration_text": "<nonempty narration for this segment>",
            "episode_ids": ["<allowed episode id>"],
            "evidence_ids": ["<allowed Evidence id>"],
            "visual_evidence_ids": [],
            "source_clips": [{
                "episode_id": "<allowed episode id>",
                "source_id": "<matching source id>",
                "start_ms": "<integer>",
                "end_ms": "<integer greater than start_ms>",
            }],
            "editorial_intent": "<nonempty explanation>",
            "uncertainty": [],
        }],
        "writer_notes": {},
    }


def writer_response_protocol(project_id: str, plan_hash: str, output_id: str) -> dict[str, Any]:
    return {
        "precedence_rule": "This response protocol overrides any response-format instruction inside editorial guidance for this stage.",
        "stage": "ONE_PER_OUTPUT_WRITER_DRAFT",
        "required_root_fields_exactly": sorted(WRITER_ROOT_FIELDS),
        "optional_root_fields": [],
        "forbidden_root_fields": ["outputs", "schema_version", "sources", "mode"],
        "required_segment_fields_exactly": sorted(WRITER_SEGMENT_FIELDS),
        "required_source_clip_fields_exactly": sorted(SOURCE_CLIP_FIELDS),
        "identity": {"project_id": project_id, "season_plan_hash": plan_hash, "output_id": output_id},
        "skeleton": writer_draft_skeleton(project_id, plan_hash, output_id),
        "final_instruction": [
            "Return JSON only.",
            "Return exactly ONE Writer Draft object for the current output.",
            "DO NOT return {\"outputs\":[...]}.",
            "DO NOT return Final JSON.",
            "DO NOT return a list or a multi-output/season wrapper.",
            "Editorial guidance controls WHAT to write; this protocol controls HOW the response is structured.",
        ],
    }


def parse_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            text = text.lstrip()[4:].lstrip() if text.lstrip().lower().startswith("json") else text
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None


def safely_unwrap_single_writer(data: dict[str, Any], project_id: str, plan_hash: str, output_id: str) -> dict[str, Any] | None:
    """Losslessly unwrap only a single already-conforming Writer Draft."""
    allowed_wrapper = {
        "writer_draft_version", "project_id", "season_plan_hash", "output_id",
        "output_language", "mode", "outputs",
    }
    if not set(data).issubset(allowed_wrapper) or not isinstance(data.get("outputs"), list) or len(data["outputs"]) != 1:
        return None
    nested = data["outputs"][0]
    if not isinstance(nested, dict) or set(nested) != WRITER_ROOT_FIELDS:
        return None
    expected = {"project_id": project_id, "season_plan_hash": plan_hash, "output_id": output_id}
    for key, value in expected.items():
        if data.get(key) not in (None, value) or nested.get(key) != value:
            return None
    if nested.get("writer_draft_version") != WRITER_DRAFT_VERSION:
        return None
    return nested


ISSUE_PATHS = {
    "MALFORMED_JSON": "response is not one complete JSON object",
    "SCHEMA_FIELD_MISSING_OR_EXTRA": "root fields must match the schema exactly; unexpected root field: $.outputs when present",
    "SCHEMA_VERSION_INVALID": "invalid value: $.writer_draft_version",
    "PROJECT_ID_MISMATCH": "identity mismatch: $.project_id",
    "SEASON_PLAN_MISMATCH": "identity mismatch: $.season_plan_hash",
    "OUTPUT_ID_MISMATCH": "identity mismatch: $.output_id",
    "MISSING_TITLE": "required root field missing/empty: $.title",
    "MISSING_NARRATION": "required root field missing/empty: $.narration.text",
    "SOURCE_DIALOGUE_ROUTED_TO_NARRATION": "source character dialogue is forbidden from $.segments[*].narration_text",
    "SEGMENTS_REQUIRED": "required root field missing/empty: $.segments",
}


def validation_diagnostics(issue_codes: tuple[str, ...], parsed: dict[str, Any] | None) -> tuple[str, ...]:
    details = [ISSUE_PATHS.get(code, f"validation issue: {code}") for code in issue_codes]
    if isinstance(parsed, dict):
        for field in sorted(set(parsed) - WRITER_ROOT_FIELDS):
            details.append(f"unexpected root field: $.{field}")
        for field in sorted(WRITER_ROOT_FIELDS - set(parsed)):
            details.append(f"required root field missing: $.{field}")
    return tuple(dict.fromkeys(details))
