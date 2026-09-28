"""Managed downstream narration artifacts derived only from canonical Final JSON."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid
import tempfile
from typing import Any, Callable

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import SourceDialogueNarrationError, VoiceStudioUnavailableError
from toolrecap_v4.persistence import atomic_write_json, native_io_path
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.voice_studio import VoiceStudioAdapter, validate_wav_bytes
from toolrecap_v4.media import find_binary,run_command
from toolrecap_v4.progress import ActivityState, WorkflowStage, safe_emit
from toolrecap_v4.original_dialogue import source_dialogue_overlaps
from toolrecap_v4.subtitles import SubtitleCue

VOICE_CACHE_VERSION = "downstream-voice-v1"

def apply_commentary_speed(wav_bytes:bytes,speed:float,*,ffmpeg_path:Path|str|None=None)->bytes:
    """Apply only the user-selected global speed using pitch-preserving atempo."""
    speed=round(float(speed),2)
    if speed==1.0:return wav_bytes
    if not 0.80<=speed<=1.30:raise ValueError("Unsupported commentary reading speed")
    ffmpeg=find_binary("ffmpeg",ffmpeg_path)
    with tempfile.TemporaryDirectory(prefix="toolrecap_voice_speed_") as temporary:
        source=Path(temporary)/"input.wav";target=Path(temporary)/"output.wav";source.write_bytes(wav_bytes)
        run_command([str(ffmpeg),"-y","-i",str(source),"-filter:a",f"atempo={speed:.2f}",str(target)],timeout=60.0,check=True)
        data=target.read_bytes();validate_wav_bytes(data);return data


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class VoiceCacheResult:
    narration_audio_map: dict[str, Path]
    artifact_hashes: dict[str, str]
    reused_count: int
    synthesized_count: int


class DownstreamVoiceCache:
    """Per-segment validated WAV cache independent of mix/render settings."""

    def __init__(
        self, storage_root: Path | str, project_id: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.base = Path(storage_root).resolve() / "projects" / project_id / "downstream" / "voice"
        self.progress_callback = progress_callback

    def _emit(self, **payload: Any) -> None:
        safe_emit(self.progress_callback, stage=WorkflowStage.VOICE.value, **payload)

    def prepare_output(
        self,
        *,
        output_def: dict[str, Any],
        final_json_hash: str,
        settings: AppSettings,
        voice_adapter: VoiceStudioAdapter | None,
        source_dialogue_map: dict[str, tuple[SubtitleCue, ...]] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> VoiceCacheResult:
        result: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        reused = synthesized = 0
        narration_segments = [
            segment for segment in output_def.get("segments", [])
            if segment.get("type") == "narration" and segment.get("narration")
        ]
        total = len(narration_segments)
        self._emit(
            state=ActivityState.RUNNING.value,
            activity_text=f"Preparing {total} narration segments at {settings.commentary_reading_speed:.2f}x for {output_def.get('render_id')}...",
            output_id=output_def.get("render_id"), completed=0, total=total, unit="segments",
        )
        for segment in narration_segments:
            if cancellation_token:
                cancellation_token.check_cancelled()
            segment_id = segment["segment_id"]
            covered_segments = [
                item for item in output_def.get("segments", [])
                if item is segment or (
                    segment.get("narration_group") and item.get("narration_group") == segment["narration_group"]
                )
            ]
            dialogue_texts = [
                cue.text for item in covered_segments
                for cue in (source_dialogue_map or {}).get(str(item.get("source_file", "")), ())
                if cue.start_ms < int(item.get("end_ms", 0)) and cue.end_ms > int(item.get("start_ms", 0))
            ]
            blocked_matches = [m for m in source_dialogue_overlaps(str(segment["narration"]), dialogue_texts) if m["blocking"]]
            if blocked_matches:
                raise SourceDialogueNarrationError(
                    f"Segment '{segment_id}' contains verified source dialogue and is forbidden from VoiceStudio: "
                    f"{blocked_matches[0]['source_text']!r} ({blocked_matches[0]['rule']})."
                )
            dependencies = {
                "version": VOICE_CACHE_VERSION,
                "final_json_hash": final_json_hash,
                "output_id": output_def.get("render_id"),
                "segment_id": segment_id,
                "narration": segment["narration"],
                "voice_mode": settings.voice_mode,
                "voice_endpoints": (
                    {"local": settings.voice_local_url}
                    if settings.voice_mode == "local"
                    else {"remote": settings.voice_remote_url}
                    if settings.voice_mode == "remote"
                    else {"local": settings.voice_local_url, "remote": settings.voice_remote_url}
                ),
                "voice_id": settings.voice_id,
                "voice_model": settings.voice_model,
                "voice_language": settings.voice_language,
                "voice_style": settings.voice_style,
                "commentary_reading_speed": round(float(settings.commentary_reading_speed),2),
            }
            key = _digest(dependencies)
            directory = self.base / output_def["render_id"] / segment_id / key[:24]
            wav_path = directory / "narration.wav"
            manifest_path = directory / "manifest.json"
            if self._valid_hit(wav_path, manifest_path, key):
                digest = hashlib.sha256(wav_path.read_bytes()).hexdigest()
                result[segment_id] = wav_path
                hashes[segment_id] = digest
                reused += 1
                self._emit(
                    state=ActivityState.RUNNING.value,
                    activity_text=f"Narration {segment_id} reused from cache.",
                    output_id=output_def.get("render_id"), current_item=segment_id,
                    completed=reused + synthesized, total=total, unit="segments",
                    reused=reused, item_event="complete", item_key=segment_id, item_reused=True,
                )
                continue
            if voice_adapter is None:
                raise VoiceStudioUnavailableError(
                    f"Narration segment '{segment_id}' requires configured VoiceStudio."
                )
            self._emit(
                state=ActivityState.RUNNING.value,
                activity_text=f"Generating narration {segment_id} with VoiceStudio...",
                output_id=output_def.get("render_id"), current_item=segment_id,
                completed=reused + synthesized, total=total, unit="segments",
                reused=reused, waiting_for="VoiceStudio", item_event="start", item_key=segment_id,
            )
            wav_bytes = voice_adapter.synthesize(
                segment["narration"],
                voice=settings.voice_id,
                model=settings.voice_model,
                language=settings.voice_language,
                style=settings.voice_style,
                cancellation_token=cancellation_token,
            )
            wav_bytes=apply_commentary_speed(wav_bytes,settings.commentary_reading_speed)
            duration = validate_wav_bytes(wav_bytes)
            digest = hashlib.sha256(wav_bytes).hexdigest()
            directory.mkdir(parents=True, exist_ok=True)
            tmp = directory / f"narration.wav.tmp.{uuid.uuid4().hex}"
            try:
                with open(native_io_path(tmp), "wb") as handle:
                    handle.write(wav_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                if cancellation_token:
                    cancellation_token.check_cancelled()
                os.replace(native_io_path(tmp), native_io_path(wav_path))
                atomic_write_json(manifest_path, {
                    "status": "COMPLETE",
                    "dependency_digest": key,
                    "wav_hash": digest,
                    "wav_size": len(wav_bytes),
                    "duration_seconds": duration,
                    "commentary_reading_speed": round(float(settings.commentary_reading_speed),2),
                })
            finally:
                try:
                    os.unlink(native_io_path(tmp))
                except FileNotFoundError:
                    pass
            result[segment_id] = wav_path
            hashes[segment_id] = digest
            synthesized += 1
            self._emit(
                state=ActivityState.RUNNING.value,
                activity_text=f"Narration {segment_id} generated and cached.",
                output_id=output_def.get("render_id"), current_item=segment_id,
                completed=reused + synthesized, total=total, unit="segments",
                reused=reused, item_event="complete", item_key=segment_id,
            )
        self._emit(
            state=ActivityState.LOCAL_PROCESSING.value,
            activity_text=f"Voice preparation complete for {output_def.get('render_id')}.",
            output_id=output_def.get("render_id"), completed=total, total=total,
            unit="segments", reused=reused, stage_status="complete" if total else "skipped",
        )
        return VoiceCacheResult(result, hashes, reused, synthesized)

    @staticmethod
    def _valid_hit(wav_path: Path, manifest_path: Path, expected_dependency: str) -> bool:
        if not wav_path.is_file() or not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data = wav_path.read_bytes()
            return (
                manifest.get("status") == "COMPLETE"
                and manifest.get("dependency_digest") == expected_dependency
                and manifest.get("wav_size") == len(data)
                and manifest.get("wav_hash") == hashlib.sha256(data).hexdigest()
                and validate_wav_bytes(data) > 0
            )
        except Exception:
            return False
