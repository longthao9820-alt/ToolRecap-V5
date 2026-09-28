from __future__ import annotations

from pathlib import Path

import pytest

from toolrecap_v4.analysis.models import PreparedEpisode, Transcript, TranscriptCue
from toolrecap_v4.analysis.scanner import ScannerChunkPolicy, ScannerConfig, ScannerService
from toolrecap_v4.analysis.scanner.prompts import measure_text_request_bytes
from toolrecap_v4.errors import ScannerChunkPlanningError


def _episode(cues: tuple[TranscriptCue, ...], duration: int = 600_000) -> PreparedEpisode:
    return PreparedEpisode(
        episode_id="E01", source_id="src_001", source_path=Path("C:/media/episode01.mkv"),
        source_basename="episode01.mkv", duration_ms=duration, canvas_width=1920, canvas_height=1080,
        transcript=Transcript(
            episode_id="E01", source_type="sidecar", source_format="srt", cues=cues,
            provenance_hash="transcript-hash",
        ), artifact_hash="prepared-hash", source_fingerprint="source-hash",
    )


def _service(max_bytes: int = 3500, max_duration: int = 300_000) -> ScannerService:
    return ScannerService(
        gateway_client=object(), storage_root=Path("."),
        config=ScannerConfig(
            model="router-model", parallelism=2,
            chunk_policy=ScannerChunkPolicy(max_duration_ms=max_duration, max_request_bytes=max_bytes),
        ),
    )


def test_normal_chunking_is_deterministic_and_cue_aware():
    episode = _episode(tuple(
        TranscriptCue(f"E01-CUE-{i:03d}", i * 10_000, i * 10_000 + 5_000, f"line {i}")
        for i in range(12)
    ))
    first = _service(max_duration=30_000)._plan(episode)
    second = _service(max_duration=30_000)._plan(episode)
    assert first == second
    assert [chunk.chunk_id for chunk in first] == [f"E01-CH-{i:03d}" for i in range(1, len(first) + 1)]
    assert all(chunk.request_bytes <= 3500 for chunk in first)
    assert [part.cue_id for chunk in first for part in chunk.parts] == [cue.cue_id for cue in episode.transcript.cues]


def test_long_single_cue_is_split_losslessly_without_cap_marker():
    original = "αβγ🙂 long dialogue " * 800
    episode = _episode((TranscriptCue("E01-CUE-112", 1000, 90_000, original),))
    chunks = _service(max_bytes=2200)._plan(episode)
    parts = [part for chunk in chunks for part in chunk.parts]
    assert len(parts) > 1
    assert "".join(part.text for part in parts) == original
    assert all(part.cue_id == "E01-CUE-112" for part in parts)
    assert [part.part_index for part in parts] == list(range(1, len(parts) + 1))
    assert all(part.part_count == len(parts) for part in parts)
    assert all(part.start_ms == 1000 and part.end_ms == 90_000 for part in parts)
    assert "... [capped]" not in "".join(part.text for part in parts)
    assert all(chunk.request_bytes <= 2200 for chunk in chunks)


def test_exact_duration_boundary_stays_together_then_splits():
    cues = (
        TranscriptCue("c1", 0, 10_000, "one"),
        TranscriptCue("c2", 20_000, 30_000, "two"),
        TranscriptCue("c3", 30_001, 40_000, "three"),
    )
    chunks = _service(max_bytes=5000, max_duration=30_000)._plan(_episode(cues))
    assert [[part.cue_id for part in chunk.parts] for chunk in chunks] == [["c1", "c2"], ["c3"]]


@pytest.mark.parametrize("transcript", [
    Transcript(episode_id="E01", cues=()),
    Transcript(episode_id="WRONG", cues=(TranscriptCue("c1", 0, 1, "x"),)),
    Transcript(episode_id="E01", cues=(TranscriptCue("c1", 0, 1, "x"), TranscriptCue("c1", 2, 3, "y"))),
    Transcript(episode_id="E01", cues=(TranscriptCue("c1", 0, 1, ""),)),
])
def test_malformed_or_empty_transcript_rejected(transcript: Transcript):
    episode = _episode((TranscriptCue("ok", 0, 1, "x"),))
    object.__setattr__(episode, "transcript", transcript)
    with pytest.raises(ScannerChunkPlanningError):
        _service()._plan(episode)


def test_request_payload_has_only_controlled_source_identifier():
    episode = _episode((TranscriptCue("c1", 0, 1000, "literal dialogue"),))
    service = _service()
    chunk = service._plan(episode)[0]
    prompt = service._prompt_for(episode, chunk.chunk_id, chunk.start_ms, chunk.end_ms, chunk.parts)
    assert "episode01.mkv" in prompt
    assert "C:/media" not in prompt and "C:\\media" not in prompt
    assert "literal dialogue" in prompt
    for forbidden in ("Recap Prompt", "EditorialPolicy", "Candidate", "Season Connection", "output quota"):
        assert forbidden not in prompt
    assert chunk.request_bytes == measure_text_request_bytes(
        model=service.config.model,
        reasoning=service.config.reasoning,
        user_prompt=prompt,
    )
