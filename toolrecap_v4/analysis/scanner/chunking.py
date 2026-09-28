"""Deterministic, cue-aware and lossless Scanner chunk planning."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Sequence

from toolrecap_v4.analysis.models import PreparedEpisode, Transcript, TranscriptCue
from toolrecap_v4.errors import ScannerCapacityError, ScannerChunkPlanningError

CHUNK_POLICY_VERSION = "scanner-chunk-v1"


@dataclass(frozen=True)
class ScannerChunkPolicy:
    max_duration_ms: int = 300_000
    max_request_bytes: int = 131_072

    def __post_init__(self) -> None:
        if type(self.max_duration_ms) is bool or self.max_duration_ms <= 0:
            raise ValueError("max_duration_ms must be a positive integer")
        if type(self.max_request_bytes) is bool or self.max_request_bytes < 1024:
            raise ValueError("max_request_bytes must be an integer of at least 1024")


@dataclass(frozen=True)
class TranscriptPart:
    cue_id: str
    part_index: int
    part_count: int
    start_ms: int
    end_ms: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "part_index": self.part_index,
            "part_count": self.part_count,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
        }


@dataclass(frozen=True)
class ScannerChunk:
    chunk_id: str
    order: int
    start_ms: int
    end_ms: int
    parts: tuple[TranscriptPart, ...]
    request_bytes: int
    content_hash: str

    @property
    def part_count(self) -> int:
        return len(self.parts)


def _validate_transcript(transcript: Transcript, episode_id: str, duration_ms: int) -> None:
    if transcript.episode_id != episode_id:
        raise ScannerChunkPlanningError(
            f"Transcript episode_id '{transcript.episode_id}' does not match '{episode_id}'.",
            episode_id=episode_id,
            request_phase="planning",
        )
    if type(duration_ms) is bool or not isinstance(duration_ms, int) or duration_ms <= 0:
        raise ScannerChunkPlanningError("Episode duration must be a positive integer.", episode_id=episode_id)
    if not transcript.cues:
        raise ScannerChunkPlanningError("Transcript contains no cues; factual text scanning cannot proceed.", episode_id=episode_id)
    seen: set[str] = set()
    last_start = -1
    for cue in transcript.cues:
        if not isinstance(cue, TranscriptCue):
            raise ScannerChunkPlanningError("Transcript contains a malformed cue.", episode_id=episode_id)
        if not cue.cue_id.strip() or cue.cue_id in seen:
            raise ScannerChunkPlanningError("Transcript cue IDs must be non-empty and unique.", episode_id=episode_id)
        if not cue.text:
            raise ScannerChunkPlanningError(f"Transcript cue '{cue.cue_id}' has empty text.", episode_id=episode_id)
        if cue.start_ms < 0 or cue.end_ms > duration_ms:
            raise ScannerChunkPlanningError(f"Transcript cue '{cue.cue_id}' is outside episode bounds.", episode_id=episode_id)
        if cue.start_ms < last_start:
            raise ScannerChunkPlanningError("Transcript cues must be ordered by start_ms.", episode_id=episode_id)
        seen.add(cue.cue_id)
        last_start = cue.start_ms


def _lossless_split_cue(
    cue: TranscriptCue,
    fits_single: Callable[[TranscriptPart], bool],
) -> list[TranscriptPart]:
    whole = TranscriptPart(cue.cue_id, 1, 1, cue.start_ms, cue.end_ms, cue.text)
    if fits_single(whole):
        return [whole]

    # Choose conservative pieces with a six-digit part-count envelope, then label exactly.
    pieces: list[str] = []
    remaining = cue.text
    while remaining:
        lo, hi, best = 1, len(remaining), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            probe = TranscriptPart(cue.cue_id, 999999, 999999, cue.start_ms, cue.end_ms, remaining[:mid])
            if fits_single(probe):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best == 0:
            raise ScannerCapacityError(
                f"Scanner request envelope cannot fit any text from cue '{cue.cue_id}'.",
                request_phase="planning",
            )
        pieces.append(remaining[:best])
        remaining = remaining[best:]

    count = len(pieces)
    result = [
        TranscriptPart(cue.cue_id, index, count, cue.start_ms, cue.end_ms, text)
        for index, text in enumerate(pieces, start=1)
    ]
    if "".join(part.text for part in result) != cue.text:
        raise AssertionError("Internal lossless cue splitting invariant failed")
    return result


def plan_scanner_chunks(
    episode: PreparedEpisode,
    policy: ScannerChunkPolicy,
    request_size_for: Callable[[str, int, int, Sequence[TranscriptPart]], int],
) -> tuple[ScannerChunk, ...]:
    """Plan chunks on cue/technical-part boundaries using the actual serialized request size."""
    _validate_transcript(episode.transcript, episode.episode_id, episode.duration_ms)

    def fits_single(part: TranscriptPart) -> bool:
        return request_size_for(f"{episode.episode_id}-CH-999999", part.start_ms, part.end_ms, (part,)) <= policy.max_request_bytes

    all_parts: list[TranscriptPart] = []
    for cue in episode.transcript.cues:
        all_parts.extend(_lossless_split_cue(cue, fits_single))

    groups: list[list[TranscriptPart]] = []
    current: list[TranscriptPart] = []
    for part in all_parts:
        candidate = [*current, part]
        order = len(groups) + 1
        chunk_id = f"{episode.episode_id}-CH-{order:03d}"
        start_ms = min(item.start_ms for item in candidate)
        end_ms = max(item.end_ms for item in candidate)
        too_long = bool(current) and end_ms - start_ms > policy.max_duration_ms
        too_large = request_size_for(chunk_id, start_ms, end_ms, candidate) > policy.max_request_bytes
        if current and (too_long or too_large):
            groups.append(current)
            current = [part]
        else:
            current = candidate
    if current:
        groups.append(current)

    chunks: list[ScannerChunk] = []
    for order, parts in enumerate(groups, start=1):
        chunk_id = f"{episode.episode_id}-CH-{order:03d}"
        start_ms = min(part.start_ms for part in parts)
        end_ms = max(part.end_ms for part in parts)
        measured = request_size_for(chunk_id, start_ms, end_ms, parts)
        if measured > policy.max_request_bytes:
            raise ScannerCapacityError(
                f"Chunk '{chunk_id}' is {measured} bytes, above {policy.max_request_bytes} bytes.",
                episode_id=episode.episode_id,
                chunk_id=chunk_id,
                request_phase="planning",
            )
        digest_input = "\n".join(
            f"{p.cue_id}|{p.part_index}|{p.part_count}|{p.start_ms}|{p.end_ms}|{p.text}" for p in parts
        )
        chunks.append(ScannerChunk(
            chunk_id=chunk_id,
            order=order,
            start_ms=start_ms,
            end_ms=end_ms,
            parts=tuple(parts),
            request_bytes=measured,
            content_hash=hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
        ))
    return tuple(chunks)
