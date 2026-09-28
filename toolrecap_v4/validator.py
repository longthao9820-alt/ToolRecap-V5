"""Technical validator for ToolRecap V4 projects and outputs."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Set

import jsonschema

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.discovery import WINDOWS_RESERVED_NAMES, is_windows_reserved_stem
from toolrecap_v4.errors import (
    DuplicateIdError,
    InvalidCharacterError,
    JSON_SOURCE_NOT_FOUND,
    JsonSourceNotFoundError,
    NarrationFitError,
    SchemaValidationError,
    SecretExposureError,
    SourceNotFoundError,
    TimestampBoundaryError,
    WindowsCollisionError,
    WindowsNameError,
    WindowsReservedNameError,
)
from toolrecap_v4.schemas.schema import get_schema_validator

# Windows forbidden filename characters: < > : " / \ | ? * and ASCII control 0-31
INVALID_WINDOWS_CHARS = set('<>:"/\\|?*')
CONTROL_CHARS = {chr(i) for i in range(32)}
ALL_INVALID_CHARS = INVALID_WINDOWS_CHARS | CONTROL_CHARS

SECRET_FIELD_TOKENS = frozenset({
    "secret", "secrets", "token", "password", "passwd", "auth",
    "credential", "credentials",
})
SECRET_FIELD_EXACT = frozenset({"api_key", "apikey", "private_key", "privatekey"})
FIELD_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def sanitize_publication_title(value: str, *, fallback: str = "Video") -> str:
    """Turn an editorial title into a stable Windows filename stem.

    This is a technical filename transformation. It preserves words and
    punctuation where Windows allows them, rather than spending another AI
    request to remove a colon, slash, quote, control character, or trailing dot.
    """
    if not isinstance(value, str):
        raise WindowsNameError("Publication title must be a string")
    chars = []
    for char in value:
        if char in CONTROL_CHARS:
            chars.append(" ")
        elif char in INVALID_WINDOWS_CHARS:
            chars.append("-")
        else:
            chars.append(char)
    clean = re.sub(r"\s+", " ", "".join(chars)).strip(" .")
    clean = re.sub(r"\s*-\s*", " - ", clean)
    clean = re.sub(r"(?:\s+-){2,}", " -", clean).strip(" .-")
    if not clean:
        clean = fallback
    if is_windows_reserved_stem(clean):
        clean = f"{clean} Video"
    validate_windows_name(clean, "publication title")
    return clean


def unique_publication_title(value: str, used: set[str], ordinal: int) -> str:
    clean = sanitize_publication_title(value, fallback=f"Video {ordinal}")
    candidate = clean
    number = 2
    while candidate.casefold() in used:
        candidate = f"{clean} ({number})"
        number += 1
    used.add(candidate.casefold())
    return candidate


def _is_secret_field_name(key: str) -> bool:
    """Recognize credential field identifiers without treating data keys as fields.

    Mapping keys such as source basenames may legitimately contain words like
    ``Secrets``.  A secret-bearing field must first look like a programming/
    JSON field identifier; filenames with spaces or extensions are therefore
    data identities, not schema field names.
    """
    if not FIELD_IDENTIFIER.fullmatch(key):
        return False
    normalized = key.casefold().replace("-", "_")
    if normalized in SECRET_FIELD_EXACT:
        return True
    tokens = tuple(part for part in normalized.split("_") if part)
    if any(token in SECRET_FIELD_TOKENS for token in tokens):
        return True
    return len(tokens) >= 2 and tokens[-2:] in (("api", "key"), ("private", "key"))


def validate_windows_name(name: str, field_name: str = "name") -> None:
    """Validate that a title or filename conforms strictly to Windows naming rules.
    
    Rejects:
    - Empty or whitespace-only names
    - Reserved Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    - Forbidden characters (<>:"/\\|?* and control characters 0-31)
    - Trailing dots or spaces
    """
    if not isinstance(name, str):
        raise WindowsNameError(f"{field_name} must be a string, got {type(name).__name__}")

    stripped = name.strip()
    if not stripped:
        raise WindowsNameError(f"{field_name} cannot be empty or whitespace-only.")

    # Check invalid characters
    found_invalid = set(name) & ALL_INVALID_CHARS
    if found_invalid:
        display_chars = ", ".join(repr(c) for c in sorted(found_invalid))
        raise InvalidCharacterError(
            f"{field_name} '{name}' contains illegal Windows characters: {display_chars}"
        )

    # Check trailing dot or space
    if name.endswith(".") or name.endswith(" "):
        raise WindowsNameError(
            f"{field_name} '{name}' cannot end with a dot or space under Windows rules."
        )

    # Check reserved device names
    if is_windows_reserved_stem(name):
        raise WindowsReservedNameError(
            f"{field_name} '{name}' matches Windows reserved device name."
        )


def check_for_secrets(data: Any, path: str = "$") -> None:
    """Recursively check for secret-like keys in data, rejecting if found."""
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            if _is_secret_field_name(k):
                raise SecretExposureError(
                    f"Forbidden secret key detected at {path}.{k}. Secrets must never be stored in project JSON."
                )
            check_for_secrets(v, f"{path}.{k}")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            check_for_secrets(item, f"{path}[{idx}]")


def validate_project(
    project_data: Dict[str, Any],
    source_durations: Optional[Dict[str, int]] = None,
    cancellation_token: Optional[CancellationToken] = None,
) -> None:
    """Validate a complete ToolRecap V4 project dictionary.
    
    Invariants:
    - Schema immutable: never repairs or mutates input data.
    - Probed source durations and exact basename mappings are authoritative.
    - AI source durations never substitute or overwrite probed durations.
    - When source_durations is supplied, every JSON source must exist in actual map.
    - Enforces exact basename matching (no casefold mapping).
    - Rejects unsafe/reserved Windows titles and duplicate casefold names/IDs.
    - Validates finite timestamps and source duration boundaries.
    - Subtitle cues are validated relative to segment duration.
    - Enforces narration fit policy as technical error (no trimming/altering clips).
    - Ensures no secrets are present in project state.
    """
    if cancellation_token:
        cancellation_token.check_cancelled()

    if not isinstance(project_data, dict):
        raise SchemaValidationError("Project data must be a dictionary.")

    # 1. Reject secrets immediately
    check_for_secrets(project_data)

    # 2. Validate against JSON schema
    validator = get_schema_validator()
    errors = list(validator.iter_errors(project_data))
    if errors:
        first = errors[0]
        path_str = ".".join(str(p) for p in first.path) or "$"
        raise SchemaValidationError(
            f"Schema validation failed at {path_str}: {first.message}"
        )

    if cancellation_token:
        cancellation_token.check_cancelled()

    # 3. Validate project_name
    validate_windows_name(project_data["project_name"], "project_name")

    # 4. Validate sources and casefold uniqueness
    known_sources_exact: Set[str] = set()
    known_sources_casefold: Dict[str, str] = {}  # casefold -> original

    for src in project_data.get("sources", []):
        src_file = src["source_file"]
        validate_windows_name(src_file, "source_file")
        cf = src_file.casefold()
        if cf in known_sources_casefold:
            raise WindowsCollisionError(
                f"Duplicate source_file under casefold: '{src_file}' conflicts with '{known_sources_casefold[cf]}'"
            )
        known_sources_casefold[cf] = src_file
        known_sources_exact.add(src_file)

    # Actual probed durations and project source mapping are authoritative.
    # When source_durations is supplied:
    # 1. Require every JSON source exists in actual map with exact basename (category JSON_SOURCE_NOT_FOUND)
    # 2. AI duration never substitutes or overwrites probed duration.
    if source_durations is not None:
        for src_file in known_sources_exact:
            if src_file not in source_durations:
                raise JsonSourceNotFoundError(
                    f"JSON source '{src_file}' not found in actual source durations map [{JSON_SOURCE_NOT_FOUND}].",
                    category=JSON_SOURCE_NOT_FOUND,
                )
        known_source_durations: Dict[str, int] = dict(source_durations)
    else:
        # Fallback to declared JSON duration only when probed source_durations is None (offline/schema testing)
        known_source_durations: Dict[str, int] = {}
        for src in project_data.get("sources", []):
            if "duration_ms" in src:
                known_source_durations[src["source_file"]] = src["duration_ms"]

    # 5. Validate outputs
    known_render_ids: Dict[str, str] = {}  # casefold -> original
    known_output_titles: Dict[str, str] = {}  # casefold -> original

    outputs = project_data.get("outputs", [])
    if not outputs:
        raise SchemaValidationError("Project must contain at least one output.")

    for output_idx, output in enumerate(outputs):
        if cancellation_token:
            cancellation_token.check_cancelled()

        render_id = output["render_id"]
        title = output["title"]

        validate_windows_name(render_id, f"outputs[{output_idx}].render_id")
        validate_windows_name(title, f"outputs[{output_idx}].title")

        rid_cf = render_id.casefold()
        if rid_cf in known_render_ids:
            raise DuplicateIdError(
                f"Duplicate render_id under casefold: '{render_id}' conflicts with '{known_render_ids[rid_cf]}'"
            )
        known_render_ids[rid_cf] = render_id

        title_cf = title.casefold()
        if title_cf in known_output_titles:
            raise WindowsCollisionError(
                f"Duplicate output title under casefold: '{title}' conflicts with '{known_output_titles[title_cf]}'"
            )
        known_output_titles[title_cf] = title

        segments = output.get("segments", [])
        from toolrecap_v4.narration_groups import narration_groups
        try:
            narration_groups(segments)
        except ValueError as exc:
            raise SchemaValidationError(str(exc)) from exc
        if not segments:
            raise SchemaValidationError(
                f"Output '{render_id}' must contain at least one segment."
            )

        known_segment_ids: Dict[str, str] = {}  # casefold -> original

        for seg_idx, seg in enumerate(segments):
            if cancellation_token:
                cancellation_token.check_cancelled()

            seg_id = seg["segment_id"]
            validate_windows_name(seg_id, f"output[{render_id}].segments[{seg_idx}].segment_id")

            sid_cf = seg_id.casefold()
            if sid_cf in known_segment_ids:
                raise DuplicateIdError(
                    f"Duplicate segment_id under casefold in output '{render_id}': '{seg_id}' conflicts with '{known_segment_ids[sid_cf]}'"
                )
            known_segment_ids[sid_cf] = seg_id

            # Source file mapping check: exact basename enforced, no casefold mapping
            seg_source = seg["source_file"]
            if seg_source not in known_sources_exact:
                if seg_source.casefold() in known_sources_casefold:
                    raise JsonSourceNotFoundError(
                        f"Segment '{seg_id}' references source_file '{seg_source}' with case mismatch. "
                        f"Exact basename match required: '{known_sources_casefold[seg_source.casefold()]}' [{JSON_SOURCE_NOT_FOUND}].",
                        category=JSON_SOURCE_NOT_FOUND,
                    )
                raise JsonSourceNotFoundError(
                    f"Segment '{seg_id}' references unknown source_file '{seg_source}' [{JSON_SOURCE_NOT_FOUND}].",
                    category=JSON_SOURCE_NOT_FOUND,
                )

            if source_durations is not None and seg_source not in source_durations:
                raise JsonSourceNotFoundError(
                    f"Segment '{seg_id}' references source_file '{seg_source}' not found in actual source durations map [{JSON_SOURCE_NOT_FOUND}].",
                    category=JSON_SOURCE_NOT_FOUND,
                )

            # Timestamp validation
            start_ms = seg["start_ms"]
            end_ms = seg["end_ms"]

            # Finite non-negative integers
            if not isinstance(start_ms, int) or isinstance(start_ms, bool) or start_ms < 0:
                raise TimestampBoundaryError(
                    f"Segment '{seg_id}' start_ms must be a non-negative integer, got {start_ms}."
                )
            if not isinstance(end_ms, int) or isinstance(end_ms, bool) or end_ms <= start_ms:
                raise TimestampBoundaryError(
                    f"Segment '{seg_id}' end_ms ({end_ms}) must be strictly greater than start_ms ({start_ms})."
                )

            segment_duration_ms = end_ms - start_ms

            # Boundary validation against source duration
            if seg_source in known_source_durations:
                max_duration = known_source_durations[seg_source]
                if end_ms > max_duration:
                    raise TimestampBoundaryError(
                        f"Segment '{seg_id}' end_ms ({end_ms}ms) exceeds source video duration ({max_duration}ms) for '{seg_source}'."
                    )

            # Subtitle cues relative to segment
            subtitles = seg.get("subtitles", [])
            for cue_idx, cue in enumerate(subtitles):
                cue_start = cue["start_ms"]
                cue_end = cue["end_ms"]
                cue_text = cue.get("text", "")

                if not isinstance(cue_start, int) or isinstance(cue_start, bool) or cue_start < 0:
                    raise TimestampBoundaryError(
                        f"Segment '{seg_id}' cue[{cue_idx}] start_ms must be a non-negative integer, got {cue_start}."
                    )
                if not isinstance(cue_end, int) or isinstance(cue_end, bool) or cue_end <= cue_start:
                    raise TimestampBoundaryError(
                        f"Segment '{seg_id}' cue[{cue_idx}] end_ms ({cue_end}) must be strictly greater than cue start_ms ({cue_start})."
                    )
                # Cues are relative to segment: cue_end cannot exceed segment_duration_ms
                if cue_end > segment_duration_ms:
                    raise TimestampBoundaryError(
                        f"Segment '{seg_id}' cue[{cue_idx}] end_ms ({cue_end}ms) exceeds segment duration ({segment_duration_ms}ms)."
                    )

            # Narration fit policy: technical error, no trim or alteration of clips
            if seg["type"] == "narration":
                narration_duration_ms = seg.get("narration_duration_ms")
                if narration_duration_ms is not None:
                    if not isinstance(narration_duration_ms, int) or narration_duration_ms < 0:
                        raise TimestampBoundaryError(
                            f"Segment '{seg_id}' narration_duration_ms must be a non-negative integer."
                        )
                    if narration_duration_ms > segment_duration_ms:
                        raise NarrationFitError(
                            f"Narration fit policy violation in segment '{seg_id}': narration audio duration ({narration_duration_ms}ms) "
                            f"exceeds visual segment duration ({segment_duration_ms}ms). "
                            f"Trimming or altering editorial clips is strictly prohibited."
                        )
