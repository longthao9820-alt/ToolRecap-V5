"""One speech recording spanning adjacent source clips, without repeated TTS."""
from __future__ import annotations

from pathlib import Path
import wave

from toolrecap_v4.errors import NarrationFitError


def narration_groups(segments: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    closed: set[str] = set()
    previous = None
    for segment in segments:
        group = segment.get("narration_group")
        if previous and group != previous:
            closed.add(previous)
        if group:
            if group in closed:
                raise ValueError("Narration group clips must be adjacent")
            if segment.get("type") != "narration":
                raise ValueError("Narration groups may contain narration segments only")
            groups.setdefault(group, []).append(segment)
        previous = group
    for group, members in groups.items():
        if not members[0].get("narration", "").strip():
            raise ValueError(f"Narration group {group} must have text on its first clip")
        if any(member.get("narration", "").strip() for member in members[1:]):
            raise ValueError(f"Narration group {group} must contain speech text exactly once")
    return groups


def slice_group_audio(source: Path, members: list[dict], directory: Path) -> tuple[dict[str, Path], int]:
    """Split a PCM recording at exact sample positions on the planned timeline.

    Never extend into an unselected scene to fit speech. Excess speech requires
    an editorial change; the full source recording remains in the voice cache.
    """
    with wave.open(str(source), "rb") as audio:
        params = audio.getparams()
        frames = audio.getnframes()
        total_ms = sum(m["end_ms"] - m["start_ms"] for m in members)
        if frames > round(total_ms * params.framerate / 1000):
            raise NarrationFitError(
                f"Narration for {members[0]['segment_id']} exceeds the combined selected clips "
                f"({frames / params.framerate:.2f}s > {total_ms / 1000:.2f}s). "
                "Shorten the narration or select more relevant footage; cached speech is preserved."
            )
        result = {}
        elapsed_ms = 0
        for index, member in enumerate(members):
            start = min(frames, round(elapsed_ms * params.framerate / 1000))
            elapsed_ms += member["end_ms"] - member["start_ms"]
            end = min(frames, round(elapsed_ms * params.framerate / 1000))
            if end <= start:
                continue
            audio.setpos(start)
            target = directory / f"group_{members[0]['segment_id']}_{index:04d}.wav"
            with wave.open(str(target), "wb") as part:
                part.setparams(params)
                part.writeframes(audio.readframes(end - start))
            result[member["segment_id"]] = target
    return result, round(frames * 1000 / params.framerate)
