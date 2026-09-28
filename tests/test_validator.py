"""Tests for technical validator, schema immutability, and boundary rules."""

import copy
import pytest

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    CancelledError,
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
from toolrecap_v4.validator import check_for_secrets, validate_project, validate_windows_name


def get_base_valid_project() -> dict:
    return {
        "schema_version": "3.0",
        "project_id": "proj-001",
        "project_name": "Project_Alpha",
        "sources": [
            {
                "source_file": "episode_01.mp4",
                "fingerprint": {
                    "basename": "episode_01.mp4",
                    "path": "C:/videos/episode_01.mp4",
                    "size_bytes": 1024,
                    "mtime_ns": 1600000000,
                    "sha256": "abc123def456",
                    "extension": ".mp4",
                },
                "duration_ms": 60000,
            }
        ],
        "outputs": [
            {
                "render_id": "render-01",
                "title": "Recap_Part_1",
                "segments": [
                    {
                        "segment_id": "seg-01",
                        "source_file": "episode_01.mp4",
                        "start_ms": 0,
                        "end_ms": 5000,
                        "type": "narration",
                        "narration": "In the beginning...",
                        "source_audio": False,
                        "narration_duration_ms": 4500,
                        "subtitles": [
                            {"start_ms": 0, "end_ms": 4000, "text": "In the beginning..."}
                        ],
                    }
                ],
            }
        ],
    }


def test_schema_immutable_no_repair() -> None:
    """Verify validator never repairs, mutates, or modifies input project data."""
    data = get_base_valid_project()
    original_snapshot = copy.deepcopy(data)

    validate_project(data)

    # Project data must be strictly identical before and after
    assert data == original_snapshot


def test_validator_rejects_without_repair() -> None:
    """Verify validator rejects invalid data rather than attempting repair."""
    data = get_base_valid_project()
    # Invalid: trailing dot in title
    data["outputs"][0]["title"] = "Invalid_Title."
    snapshot = copy.deepcopy(data)

    with pytest.raises(WindowsNameError):
        validate_project(data)

    # Verify input data was NOT repaired or stripped
    assert data == snapshot
    assert data["outputs"][0]["title"] == "Invalid_Title."


@pytest.mark.parametrize("reserved", [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "LPT9",
    "con.mp4", "aux.json", "Nul.txt"
])
def test_reject_windows_reserved_names(reserved: str) -> None:
    """Verify Windows reserved device names are strictly rejected."""
    with pytest.raises(WindowsReservedNameError):
        validate_windows_name(reserved, "test_field")


@pytest.mark.parametrize("invalid_name", [
    'video<1>', 'video>1', 'video:1', 'video"1', 'video/1', 'video\\1', 'video|1', 'video?1', 'video*1',
    "control\x00char", "control\x1fchar",
])
def test_reject_invalid_windows_characters(invalid_name: str) -> None:
    """Verify forbidden Windows filename characters are strictly rejected."""
    with pytest.raises(InvalidCharacterError):
        validate_windows_name(invalid_name, "test_field")


@pytest.mark.parametrize("bad_name", [
    "title.", "title ", "name..", "   ", ""
])
def test_reject_trailing_dot_or_space_or_empty(bad_name: str) -> None:
    """Verify trailing dot, trailing space, or empty names are rejected."""
    with pytest.raises(WindowsNameError):
        validate_windows_name(bad_name, "test_field")


def test_reject_duplicate_casefold_sources() -> None:
    """Verify sources with casefold collision are rejected."""
    data = get_base_valid_project()
    data["sources"].append({
        "source_file": "EPISODE_01.MP4",
        "duration_ms": 60000,
    })

    with pytest.raises(WindowsCollisionError, match="Duplicate source_file under casefold"):
        validate_project(data)


def test_reject_duplicate_casefold_render_id() -> None:
    """Verify outputs with casefold collision on render_id are rejected."""
    data = get_base_valid_project()
    output2 = copy.deepcopy(data["outputs"][0])
    output2["render_id"] = "RENDER-01"
    output2["title"] = "Different_Title"
    data["outputs"].append(output2)

    with pytest.raises(DuplicateIdError, match="Duplicate render_id under casefold"):
        validate_project(data)


def test_reject_duplicate_casefold_output_titles() -> None:
    """Verify outputs with casefold collision on title are rejected."""
    data = get_base_valid_project()
    output2 = copy.deepcopy(data["outputs"][0])
    output2["render_id"] = "render-02"
    output2["title"] = "recap_part_1"  # matches Recap_Part_1
    data["outputs"].append(output2)

    with pytest.raises(WindowsCollisionError, match="Duplicate output title under casefold"):
        validate_project(data)


def test_reject_duplicate_casefold_segment_id() -> None:
    """Verify segments with casefold collision on segment_id are rejected."""
    data = get_base_valid_project()
    seg2 = copy.deepcopy(data["outputs"][0]["segments"][0])
    seg2["segment_id"] = "SEG-01"
    data["outputs"][0]["segments"].append(seg2)

    with pytest.raises(DuplicateIdError, match="Duplicate segment_id under casefold"):
        validate_project(data)


def test_reject_negative_or_inverted_timestamps() -> None:
    """Verify finite timestamp validation for segments."""
    # start_ms < 0
    data = get_base_valid_project()
    data["outputs"][0]["segments"][0]["start_ms"] = -1
    with pytest.raises((TimestampBoundaryError, SchemaValidationError)):
        validate_project(data)

    # end_ms <= start_ms
    data = get_base_valid_project()
    data["outputs"][0]["segments"][0]["start_ms"] = 5000
    data["outputs"][0]["segments"][0]["end_ms"] = 5000
    with pytest.raises(TimestampBoundaryError, match="must be strictly greater than start_ms"):
        validate_project(data)


def test_reject_timestamps_exceeding_source_duration() -> None:
    """Verify segment end_ms exceeding source duration is rejected."""
    data = get_base_valid_project()
    # Source duration is 60000ms; set segment end_ms to 60001ms
    data["outputs"][0]["segments"][0]["end_ms"] = 60001

    with pytest.raises(TimestampBoundaryError, match="exceeds source video duration"):
        validate_project(data)


def test_subtitle_cues_relative_to_segment() -> None:
    """Verify subtitle cues must be relative to segment duration."""
    data = get_base_valid_project()
    # Segment is 0 to 5000ms (duration 5000ms)
    # Cue exceeding segment duration (end_ms: 5001ms)
    data["outputs"][0]["segments"][0]["subtitles"] = [
        {"start_ms": 0, "end_ms": 5001, "text": "Too long cue"}
    ]

    with pytest.raises(TimestampBoundaryError, match="exceeds segment duration"):
        validate_project(data)

    # Inverted cue (end_ms <= start_ms)
    data["outputs"][0]["segments"][0]["subtitles"] = [
        {"start_ms": 3000, "end_ms": 2000, "text": "Inverted"}
    ]
    with pytest.raises(TimestampBoundaryError, match="strictly greater than cue start_ms"):
        validate_project(data)


def test_narration_fit_policy_technical_error() -> None:
    """Verify narration fit policy raises NarrationFitError and does NOT trim clips.
    
    If narration audio duration exceeds visual segment duration, it is a technical error.
    Editorial clips must not be altered or trimmed.
    """
    data = get_base_valid_project()
    # Segment duration = 5000ms - 0ms = 5000ms
    # Narration duration = 5500ms -> exceeds segment
    data["outputs"][0]["segments"][0]["narration_duration_ms"] = 5500

    snapshot = copy.deepcopy(data)

    with pytest.raises(NarrationFitError, match="Narration fit policy violation"):
        validate_project(data)

    # Ensure clips were not modified or trimmed
    assert data["outputs"][0]["segments"][0]["start_ms"] == snapshot["outputs"][0]["segments"][0]["start_ms"]
    assert data["outputs"][0]["segments"][0]["end_ms"] == snapshot["outputs"][0]["segments"][0]["end_ms"]


def test_narration_fit_policy_passes_when_fits() -> None:
    """Verify narration segment passes when narration fits visual segment."""
    data = get_base_valid_project()
    # Segment duration: 5000ms, narration: 4800ms
    data["outputs"][0]["segments"][0]["narration_duration_ms"] = 4800
    validate_project(data)


def test_secrets_rejected_in_validator() -> None:
    """Verify validator strictly rejects any secret keys in project data."""
    data = get_base_valid_project()
    data["api_key"] = "sk-123456789"

    with pytest.raises(SecretExposureError, match="Forbidden secret key detected"):
        validate_project(data)


def test_source_filename_containing_secrets_is_not_a_credential_field() -> None:
    data = get_base_valid_project()
    original = data["sources"][0]["source_file"]
    filename = "Blue Bloods - 3x09 - Secrets and Lies.mp4"
    data["sources"][0]["source_file"] = filename
    data["outputs"][0]["segments"][0]["source_file"] = filename
    data["source_fingerprints"] = {
        filename: {"basename": filename, "path": "D:/TV/" + filename}
    }
    # Secret scanning must accept arbitrary source identity keys.  Schema may
    # independently reject extra project fields, so exercise the guard itself.
    check_for_secrets(data)


def test_cancellation_in_validator() -> None:
    """Verify validator responds to cancellation token."""
    data = get_base_valid_project()
    token = CancellationToken()
    token.cancel()

    with pytest.raises(CancelledError):
        validate_project(data, cancellation_token=token)


def test_forged_long_json_duration_cannot_bypass_actual_duration() -> None:
    """Verify forged long JSON duration cannot bypass actual probed duration.
    
    AI sources.duration_ms must never overwrite or substitute actual probed duration.
    """
    data = get_base_valid_project()
    # Forged duration in JSON: 999999 ms
    data["sources"][0]["duration_ms"] = 999999
    # Actual probed duration: 10000 ms
    actual_source_durations = {"episode_01.mp4": 10000}
    # Segment end_ms: 15000 ms (fits within forged 999999ms, but exceeds probed 10000ms)
    data["outputs"][0]["segments"][0]["end_ms"] = 15000

    with pytest.raises(TimestampBoundaryError, match="exceeds source video duration \\(10000ms\\)"):
        validate_project(data, source_durations=actual_source_durations)


def test_probed_duration_authoritative_over_short_json_duration() -> None:
    """Verify probed duration is authoritative even when JSON duration is shorter."""
    data = get_base_valid_project()
    # Short duration in JSON: 5000 ms
    data["sources"][0]["duration_ms"] = 5000
    # Actual probed duration: 20000 ms
    actual_source_durations = {"episode_01.mp4": 20000}
    # Segment end_ms: 12000 ms (exceeds JSON 5000ms, but within actual probed 20000ms)
    data["outputs"][0]["segments"][0]["end_ms"] = 12000

    # Must pass because actual probed duration governs
    validate_project(data, source_durations=actual_source_durations)


def test_missing_actual_file_rejected_with_category() -> None:
    """Verify missing source in actual map raises JsonSourceNotFoundError with JSON_SOURCE_NOT_FOUND."""
    data = get_base_valid_project()
    # JSON references episode_01.mp4, but actual source map contains different file
    actual_source_durations = {"other_episode.mp4": 60000}

    with pytest.raises(JsonSourceNotFoundError) as exc_info:
        validate_project(data, source_durations=actual_source_durations)

    assert exc_info.value.category == JSON_SOURCE_NOT_FOUND
    assert exc_info.value.code == JSON_SOURCE_NOT_FOUND
    assert "JSON_SOURCE_NOT_FOUND" in str(exc_info.value)
    assert "episode_01.mp4" in str(exc_info.value)
    assert isinstance(exc_info.value, SchemaValidationError)


def test_exact_basename_enforced_clip_case_mismatch() -> None:
    """Verify exact basename is enforced for clip source_file; casefold mapping is prohibited."""
    data = get_base_valid_project()
    # Source defined as episode_01.mp4
    actual_source_durations = {"episode_01.mp4": 60000}
    # Clip references uppercase EPISODE_01.MP4
    data["outputs"][0]["segments"][0]["source_file"] = "EPISODE_01.MP4"

    with pytest.raises(JsonSourceNotFoundError) as exc_info:
        validate_project(data, source_durations=actual_source_durations)

    assert exc_info.value.category == JSON_SOURCE_NOT_FOUND
    assert "case mismatch" in str(exc_info.value).lower()
    assert "Exact basename match required" in str(exc_info.value)


def test_exact_basename_enforced_when_source_durations_none() -> None:
    """Verify exact basename is enforced even when source_durations is None (offline mode)."""
    data = get_base_valid_project()
    # Source defined as episode_01.mp4
    # Clip references different casing Episode_01.mp4
    data["outputs"][0]["segments"][0]["source_file"] = "Episode_01.mp4"

    with pytest.raises(JsonSourceNotFoundError) as exc_info:
        validate_project(data)

    assert exc_info.value.category == JSON_SOURCE_NOT_FOUND
    assert "Exact basename match required" in str(exc_info.value)


def test_exact_basename_enforced_between_json_source_and_actual_map() -> None:
    """Verify JSON source must match actual probe map with exact basename (case-sensitive)."""
    data = get_base_valid_project()
    # Probe map has lowercase
    actual_source_durations = {"episode_01.mp4": 60000}
    # JSON has uppercase
    data["sources"][0]["source_file"] = "EPISODE_01.MP4"
    data["outputs"][0]["segments"][0]["source_file"] = "EPISODE_01.MP4"

    with pytest.raises(JsonSourceNotFoundError) as exc_info:
        validate_project(data, source_durations=actual_source_durations)

    assert exc_info.value.category == JSON_SOURCE_NOT_FOUND
    assert "EPISODE_01.MP4" in str(exc_info.value)


def test_segment_unknown_source_rejected() -> None:
    """Verify segment referencing nonexistent source file is rejected."""
    data = get_base_valid_project()
    actual_source_durations = {"episode_01.mp4": 60000}
    data["outputs"][0]["segments"][0]["source_file"] = "nonexistent.mp4"

    with pytest.raises(JsonSourceNotFoundError) as exc_info:
        validate_project(data, source_durations=actual_source_durations)

    assert exc_info.value.category == JSON_SOURCE_NOT_FOUND
    assert "nonexistent.mp4" in str(exc_info.value)


def test_json_source_not_found_error_hierarchy() -> None:
    """Verify JsonSourceNotFoundError and SourceNotFoundError inheritance and attributes."""
    assert issubclass(JsonSourceNotFoundError, SchemaValidationError)
    assert issubclass(JsonSourceNotFoundError, Exception)
    assert SourceNotFoundError is JsonSourceNotFoundError

    err = JsonSourceNotFoundError("missing file test")
    assert err.category == JSON_SOURCE_NOT_FOUND
    assert err.code == JSON_SOURCE_NOT_FOUND
    assert str(err) == "missing file test"


def test_narration_source_audio_boolean_semantics() -> None:
    """Verify narration segment accepts source_audio=True and False per schema without editorial rules."""
    data = get_base_valid_project()
    actual_source_durations = {"episode_01.mp4": 60000}

    # source_audio = True on narration segment
    data["outputs"][0]["segments"][0]["source_audio"] = True
    validate_project(data, source_durations=actual_source_durations)

    # source_audio = False on narration segment
    data["outputs"][0]["segments"][0]["source_audio"] = False
    validate_project(data, source_durations=actual_source_durations)


def test_valid_project_with_source_durations_passes_and_preserves_json() -> None:
    """Verify valid project with matching source_durations passes and preserves JSON exactly."""
    data = get_base_valid_project()
    snapshot = copy.deepcopy(data)
    actual_source_durations = {"episode_01.mp4": 60000, "extra_unused.mp4": 120000}

    validate_project(data, source_durations=actual_source_durations)

    # Invariant: preserve JSON exactly
    assert data == snapshot
