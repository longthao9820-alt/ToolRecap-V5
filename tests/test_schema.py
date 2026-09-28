"""Tests for JSON schema structure and validation."""

import pytest
import jsonschema

from toolrecap_v4.schemas.schema import get_project_schema, get_schema_validator


def make_valid_project_payload() -> dict:
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


def test_schema_itself_is_valid() -> None:
    """Verify schema is valid JSON Schema Draft 2020-12."""
    schema = get_project_schema()
    validator = get_schema_validator()
    assert validator.is_valid(make_valid_project_payload())


def test_schema_rejects_missing_required_fields() -> None:
    """Verify schema rejects payloads missing mandatory fields."""
    validator = get_schema_validator()
    payload = make_valid_project_payload()

    # Missing schema_version
    del payload["schema_version"]
    assert not validator.is_valid(payload)

    # Missing project_name
    payload = make_valid_project_payload()
    del payload["project_name"]
    assert not validator.is_valid(payload)

    # Missing sources
    payload = make_valid_project_payload()
    del payload["sources"]
    assert not validator.is_valid(payload)

    # Missing outputs
    payload = make_valid_project_payload()
    del payload["outputs"]
    assert not validator.is_valid(payload)


def test_schema_rejects_invalid_schema_version() -> None:
    """Verify schema rejects non-3.0 versions."""
    validator = get_schema_validator()
    payload = make_valid_project_payload()
    payload["schema_version"] = "2.0"
    assert not validator.is_valid(payload)


def test_schema_rejects_invalid_segment_type() -> None:
    """Verify segment type must be 'narration' or 'original_dialogue'."""
    validator = get_schema_validator()
    payload = make_valid_project_payload()
    payload["outputs"][0]["segments"][0]["type"] = "invalid_type"
    assert not validator.is_valid(payload)


def test_schema_rejects_empty_outputs() -> None:
    """Verify outputs list cannot have minItems < 1."""
    validator = get_schema_validator()
    payload = make_valid_project_payload()
    payload["outputs"][0]["segments"] = []
    assert not validator.is_valid(payload)
