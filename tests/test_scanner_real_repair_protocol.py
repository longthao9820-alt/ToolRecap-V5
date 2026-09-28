"""Sanitized real Scanner grounding failure and general transport/recovery coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolrecap_v4.analysis.models import PreparedEpisode, Transcript, TranscriptCue
from toolrecap_v4.analysis.scanner import ScannerChunkPolicy, ScannerConfig, ScannerService
from toolrecap_v4.analysis.scanner.schema import parse_scanner_json, validate_scanner_response
from toolrecap_v4.analysis.scanner.service import compute_evidence_revision
from toolrecap_v4.errors import ScannerRepairExhaustedError, ScannerValidationError, ScannerResponseError
from toolrecap_v4.gateway import GatewayResult


def _episode(episode_id: str, cue_count: int, *, long: bool = False) -> PreparedEpisode:
    cues = tuple(TranscriptCue(
        f"{episode_id}-CUE-{index:03d}", index * 5_000, index * 5_000 + 2_000,
        ("A long synthetic factual line. " * 100) if long and index == 1 else f"Synthetic fact {index}.",
    ) for index in range(1, cue_count + 1))
    return PreparedEpisode(
        episode_id=episode_id, source_id=f"source-{episode_id}",
        source_path=Path(f"C:/synthetic/{episode_id}.mkv"), source_basename=f"{episode_id}.mkv",
        duration_ms=(cue_count + 2) * 5_000, canvas_width=320, canvas_height=240,
        transcript=Transcript(episode_id=episode_id, source_type="sidecar", source_format="srt", cues=cues,
                              provenance_hash=f"transcript-{episode_id}"),
        artifact_hash=f"prepared-{episode_id}", source_fingerprint=f"source-hash-{episode_id}",
    )


def _response(request: dict, *, invalid_grounding: bool = False, empty: bool = False, multiple: bool = False) -> dict:
    parts = request["transcript_parts"]
    if empty:
        rows = []
    else:
        first = parts[0]
        end = parts[-1]["end_ms"] if invalid_grounding and len(parts) > 1 else first["end_ms"]
        rows = [{
            "start_ms": first["start_ms"], "end_ms": end, "category": "event",
            "observation": "A synthetic event occurs.",
            "cue_refs": [{"cue_id": first["cue_id"], "part_index": first["part_index"]}],
            "entities": [], "modality": "subtitle", "confidence": None, "uncertainty": [],
        }]
        if multiple and len(parts) > 1:
            second = parts[1]
            rows.append({**rows[0], "start_ms": second["start_ms"], "end_ms": second["end_ms"],
                         "observation": "A second synthetic event occurs.",
                         "cue_refs": [{"cue_id": second["cue_id"], "part_index": second["part_index"]}]})
    return {"episode_id": request["episode_id"], "source_id": request["source_id"],
            "chunk_id": request["chunk_id"], "observations": rows}


class Gateway:
    def __init__(self, *, failed_chunk: str | None = None, fixed_repair: bool = False, always_invalid: bool = False):
        self.failed_chunk = failed_chunk
        self.fixed_repair = fixed_repair
        self.always_invalid = always_invalid
        self.calls: list[dict] = []

    def submit_text_chat(self, **kwargs):
        self.calls.append(kwargs)
        outer = json.loads(kwargs["prompt"])
        request = outer.get("original_request", outer)
        chunk_id = request["chunk_id"]
        assert "images" not in kwargs
        assert "Recap Prompt" not in kwargs["prompt"]
        invalid = self.always_invalid or chunk_id == self.failed_chunk
        if kwargs["phase"] == "scanner_repair" and self.fixed_repair:
            assert "cue_grounding" in " ".join(outer["validation_errors"])
            assert "earliest cited cue part" in " ".join(outer["validation_explanations"])
            assert outer["original_request"]["chunk_id"] == chunk_id
            invalid = False
        raw = json.dumps(_response(request, invalid_grounding=invalid))
        return GatewayResult(raw_response=raw, bytes_sent=len(kwargs["prompt"].encode()),
                             metadata={"status_code": 200})


def _service(gateway, root, *, max_duration: int = 30_000, max_bytes: int = 10_000):
    return ScannerService(gateway, root, ScannerConfig(
        model="synthetic-model", parallelism=2, repair_attempts=1,
        chunk_policy=ScannerChunkPolicy(max_duration_ms=max_duration, max_request_bytes=max_bytes),
    ))


def test_sanitized_real_grounding_failure_repair_and_cache_resume(tmp_path):
    episode = _episode("E07", 6)
    first = Gateway(failed_chunk="E07-CH-003")
    service = _service(first, tmp_path, max_duration=8_000)
    with pytest.raises(ScannerRepairExhaustedError):
        service.scan_project("synthetic-project", [episode])
    revision = compute_evidence_revision([episode], service.config)
    scanner_root = tmp_path / "projects" / "synthetic-project" / "scanner" / revision / "E07"
    prior_valid = [path for path in scanner_root.glob("*/manifest.json") if json.loads(path.read_text())["status"] == "COMPLETE"]
    assert prior_valid
    failed = scanner_root / "E07-CH-003"
    assert (failed / "attempt-00.raw.txt").is_file() and (failed / "attempt-01.raw.txt").is_file()
    assert not (failed / "manifest.json").is_file()

    resumed = Gateway(failed_chunk="E07-CH-003", fixed_repair=True)
    result = _service(resumed, tmp_path, max_duration=8_000).scan_project("synthetic-project", [episode])
    assert result.reused_chunk_count == len(prior_valid)
    assert [json.loads(call["prompt"])["original_request"]["chunk_id"] for call in resumed.calls] == ["E07-CH-003"]
    assert resumed.calls[0]["phase"] == "scanner_repair"
    assert (failed / "attempt-02.raw.txt").is_file()
    assert (failed / "manifest.json").is_file()


@pytest.mark.parametrize("episode_id,cue_count", [("E01", 1), ("E11", 5), ("E23", 3)])
def test_variable_episode_chunk_ids_and_empty_or_multiple_results(tmp_path, episode_id, cue_count):
    episode = _episode(episode_id, cue_count)
    service = _service(Gateway(), tmp_path)
    chunks = service._plan(episode)
    assert all(chunk.chunk_id.startswith(f"{episode_id}-CH-") for chunk in chunks)
    chunk = chunks[0]
    request = json.loads(service._prompt_for(episode, chunk.chunk_id, chunk.start_ms, chunk.end_ms, chunk.parts))
    for empty,multiple in ((True,False),(False,True)):
        response = _response(request, empty=empty, multiple=multiple)
        accepted = validate_scanner_response(response, episode_id=episode_id, source_id=episode.source_id,
                                             episode_duration_ms=episode.duration_ms, chunk=chunk)
        assert len(accepted) == (0 if empty else min(2,len(chunk.parts)))


def test_long_cue_splitting_stays_lossless_with_grounding_contract(tmp_path):
    episode = _episode("E12", 1, long=True)
    service = _service(Gateway(), tmp_path, max_bytes=2_200)
    parts = [part for chunk in service._plan(episode) for part in chunk.parts]
    assert len(parts) > 1
    assert "".join(part.text for part in parts) == episode.transcript.cues[0].text


def test_fence_wrapper_duplicates_and_strict_identity_rejection(tmp_path):
    episode = _episode("E14", 2)
    service = _service(Gateway(), tmp_path)
    chunk = service._plan(episode)[0]
    request = json.loads(service._prompt_for(episode, chunk.chunk_id, chunk.start_ms, chunk.end_ms, chunk.parts))
    valid = _response(request)
    raw = json.dumps(valid)
    assert parse_scanner_json(f"```json\n{raw}\n```", episode_id=episode.episode_id, chunk_id=chunk.chunk_id) == valid
    assert parse_scanner_json(json.dumps({"choices":[{"message":{"content":raw}}]}), episode_id=episode.episode_id, chunk_id=chunk.chunk_id) == valid
    with pytest.raises(ScannerResponseError):
        parse_scanner_json("{bad", episode_id=episode.episode_id, chunk_id=chunk.chunk_id)
    with pytest.raises(ScannerValidationError) as duplicate:
        parse_scanner_json('{"source_id":"a","source_id":"b"}', episode_id=episode.episode_id, chunk_id=chunk.chunk_id)
    assert "duplicate_json_key" in duplicate.value.issue_codes
    for field,wrong,issue in (("source_id","invented-source","source_id_mismatch"),
                              ("episode_id","E99","episode_id_mismatch")):
        invalid = {**valid, field: wrong}
        with pytest.raises(ScannerValidationError) as error:
            validate_scanner_response(invalid, episode_id=episode.episode_id, source_id=episode.source_id,
                                      episode_duration_ms=episode.duration_ms, chunk=chunk)
        assert issue in error.value.issue_codes
    invalid = json.loads(raw)
    invalid["observations"][0]["end_ms"] = episode.duration_ms + 1
    with pytest.raises(ScannerValidationError) as range_error:
        validate_scanner_response(invalid, episode_id=episode.episode_id, source_id=episode.source_id,
                                  episode_duration_ms=episode.duration_ms, chunk=chunk)
    assert any("range" in code for code in range_error.value.issue_codes)


def test_truthful_repair_exhaustion_is_still_bounded(tmp_path):
    episode = _episode("E17", 2)
    gateway = Gateway(always_invalid=True)
    with pytest.raises(ScannerRepairExhaustedError):
        _service(gateway, tmp_path).scan_project("synthetic-project", [episode])
    assert [call["phase"] for call in gateway.calls] == ["scanner", "scanner_repair"]
