from __future__ import annotations

import copy

import pytest

from toolrecap_v4.analysis.scanner.chunking import ScannerChunk, TranscriptPart
from toolrecap_v4.analysis.scanner.schema import parse_scanner_json, validate_scanner_response
from toolrecap_v4.errors import ScannerResponseError, ScannerValidationError


@pytest.fixture
def chunk() -> ScannerChunk:
    return ScannerChunk(
        "E03-CH-001", 1, 1000, 5000,
        (TranscriptPart("E03-CUE-001", 1, 1, 1000, 5000, "Beth says hello."),),
        2000, "chunkhash",
    )


@pytest.fixture
def valid() -> dict:
    return {
        "episode_id": "E03", "source_id": "src_003", "chunk_id": "E03-CH-001",
        "observations": [{
            "start_ms": 1000, "end_ms": 5000, "category": "dialogue",
            "observation": "Beth says hello.",
            "cue_refs": [{"cue_id": "E03-CUE-001", "part_index": 1}],
            "entities": ["Beth"], "modality": "subtitle", "confidence": None, "uncertainty": [],
        }],
    }


def _validate(data: dict, chunk: ScannerChunk):
    return validate_scanner_response(data, episode_id="E03", source_id="src_003", episode_duration_ms=10_000, chunk=chunk)


def test_valid_response_hydrates_exact_dialogue_provenance(valid, chunk):
    observation = _validate(valid, chunk)[0]
    assert observation.dialogue[0].text == "Beth says hello."
    assert observation.confidence is None


@pytest.mark.parametrize("mutator,code", [
    (lambda d: d.update(episode_id="E99"), "episode_id_mismatch"),
    (lambda d: d.update(source_id="wrong"), "source_id_mismatch"),
    (lambda d: d["observations"][0].update(start_ms=-1), "start_negative"),
    (lambda d: d["observations"][0].update(end_ms=1000), "range_order"),
    (lambda d: d["observations"][0].update(end_ms=10_001), "episode_range"),
    (lambda d: d["observations"][0].update(start_ms=True), "start_ms_type"),
    (lambda d: d["observations"][0].update(start_ms=1000.0), "start_ms_type"),
    (lambda d: d["observations"][0].update(start_ms=0), "chunk_range"),
    (lambda d: d["observations"][0].update(cue_refs=[{"cue_id": "missing", "part_index": 1}]), "unknown"),
    (lambda d: d["observations"][0].pop("observation"), "observation"),
    (lambda d: d["observations"][0].update(category="story"), "category"),
    (lambda d: d["observations"][0].update(modality="video"), "modality"),
    (lambda d: d["observations"][0].update(uncertainty="maybe"), "uncertainty_type"),
])
def test_strict_rejections_without_clamping_or_identity_rewrite(valid, chunk, mutator, code):
    data = copy.deepcopy(valid)
    mutator(data)
    with pytest.raises(ScannerValidationError) as raised:
        _validate(data, chunk)
    assert any(code in issue for issue in raised.value.issue_codes)
    assert data.get("episode_id") != "E03" or code != "episode_id_mismatch"  # input was not rewritten


def test_malformed_json_rejected():
    with pytest.raises(ScannerResponseError):
        parse_scanner_json("{not json", episode_id="E03", chunk_id="E03-CH-001")
