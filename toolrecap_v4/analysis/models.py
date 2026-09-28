"""Core data models for ToolRecap V4 analysis, prepared episodes, transcripts, and stream selections."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from toolrecap_v4.analysis.source_prep.subtitles.models import SubtitleStreamInfo, SubtitleTrack
from toolrecap_v4.media import AudioStreamInfo, VideoStreamInfo


@dataclass(frozen=True)
class DialogueReference:
    """Exact transcript-part provenance retained by an Evidence observation."""

    cue_id: str
    part_index: int
    part_count: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DialogueReference:
        return cls(
            cue_id=str(data["cue_id"]),
            part_index=int(data["part_index"]),
            part_count=int(data["part_count"]),
            text=str(data["text"]),
        )


@dataclass(frozen=True)
class Evidence:
    """Application-owned, factual evidence stored under one evidence revision."""

    evidence_id: str
    episode_id: str
    source_id: str
    start_ms: int
    end_ms: int
    category: str
    observation: str
    dialogue: tuple[DialogueReference, ...] = field(default_factory=tuple)
    entities: tuple[str, ...] = field(default_factory=tuple)
    modality: str = "subtitle"
    confidence: float | None = None
    uncertainty: tuple[str, ...] = field(default_factory=tuple)
    visual_refs: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.start_ms) is bool or type(self.end_ms) is bool:
            raise TypeError("Evidence timestamps cannot be boolean")
        if not isinstance(self.start_ms, int) or not isinstance(self.end_ms, int):
            raise TypeError("Evidence timestamps must be integers")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("Evidence requires 0 <= start_ms < end_ms")
        for name in ("evidence_id", "episode_id", "source_id", "category", "observation", "modality"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Evidence {name} must be a non-empty string")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise TypeError("Evidence confidence must be numeric or null")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise ValueError("Evidence confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "episode_id": self.episode_id,
            "source_id": self.source_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "category": self.category,
            "observation": self.observation,
            "dialogue": [item.to_dict() for item in self.dialogue],
            "entities": list(self.entities),
            "modality": self.modality,
            "confidence": self.confidence,
            "uncertainty": list(self.uncertainty),
            "visual_refs": list(self.visual_refs),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        return cls(
            evidence_id=str(data["evidence_id"]),
            episode_id=str(data["episode_id"]),
            source_id=str(data["source_id"]),
            start_ms=data["start_ms"],
            end_ms=data["end_ms"],
            category=str(data["category"]),
            observation=str(data["observation"]),
            dialogue=tuple(DialogueReference.from_dict(item) for item in data.get("dialogue", [])),
            entities=tuple(str(item) for item in data.get("entities", [])),
            modality=str(data["modality"]),
            confidence=data.get("confidence"),
            uncertainty=tuple(str(item) for item in data.get("uncertainty", [])),
            visual_refs=tuple(str(item) for item in data.get("visual_refs", [])),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass(frozen=True)
class TranscriptCue:
    """Individual subtitle or transcript cue with strict millisecond boundaries."""

    cue_id: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if type(self.start_ms) is bool or type(self.end_ms) is bool:
            raise TypeError("start_ms and end_ms cannot be boolean")
        if not isinstance(self.start_ms, int) or not isinstance(self.end_ms, int):
            raise TypeError("start_ms and end_ms must be integers")
        if self.end_ms <= self.start_ms:
            raise ValueError(f"Invalid cue bounds: end_ms ({self.end_ms}) <= start_ms ({self.start_ms})")

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptCue:
        return cls(
            cue_id=str(data.get("cue_id", "")),
            start_ms=int(data.get("start_ms", 0)),
            end_ms=int(data.get("end_ms", 0)),
            text=str(data.get("text", "")),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        )


@dataclass(frozen=True)
class Transcript:
    """Immutable sequence of speech/subtitle cues representing an episode dialogue transcript."""

    episode_id: str = ""
    source_type: str = "empty"  # 'sidecar' | 'embedded' | 'ocr' | 'stt' | 'empty'
    source_format: str = "none"  # 'srt' | 'ass' | 'vtt' | 'pgs' | 'vobsub' | 'whisper' | 'none'
    cues: tuple[TranscriptCue, ...] = field(default_factory=tuple)
    has_speech: bool = True
    language: str = "und"
    provenance_hash: str = ""
    dropped_cues_count: int = 0
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def cue_count(self) -> int:
        return len(self.cues)

    @property
    def duration_ms(self) -> int:
        if not self.cues:
            return 0
        return max(c.end_ms for c in self.cues) - min(c.start_ms for c in self.cues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_type": self.source_type,
            "source_format": self.source_format,
            "cues": [c.to_dict() for c in self.cues],
            "has_speech": self.has_speech,
            "language": self.language,
            "provenance_hash": self.provenance_hash,
            "dropped_cues_count": self.dropped_cues_count,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        cues_raw = data.get("cues", [])
        cues = tuple(
            TranscriptCue.from_dict(c) if isinstance(c, dict) else c
            for c in cues_raw
        )
        return cls(
            episode_id=str(data.get("episode_id", "")),
            source_type=str(data.get("source_type", "empty")),
            source_format=str(data.get("source_format", "none")),
            cues=cues,
            has_speech=bool(data.get("has_speech", True)),
            language=str(data.get("language", "und")),
            provenance_hash=str(data.get("provenance_hash", "")),
            dropped_cues_count=int(data.get("dropped_cues_count", 0)),
            diagnostics=tuple(str(d) for d in data.get("diagnostics", [])),
        )


@dataclass(frozen=True)
class AudioSelection:
    """Audio stream selection result preserving both global stream index and audio stream ordinal.
    
    Invariants:
    - global_index matches container stream index for explicit mapping (e.g. 0:<global_index>).
    - audio_ordinal represents audio-specific 0-based stream index.
    - If no audio exists, global_index and audio_ordinal are -1 and selected_stream is None.
    - Mapping helper raises ValueError if no audio stream exists.
    """

    selected_stream: AudioStreamInfo | None = None
    global_index: int = -1
    audio_ordinal: int = -1
    warning: str | None = None
    reason: str = ""

    @property
    def has_audio(self) -> bool:
        return self.selected_stream is not None and self.global_index >= 0

    @property
    def has_warning(self) -> bool:
        return self.warning is not None

    def ffmpeg_map_spec(self) -> str:
        """Explicit stream mapping specifier using container global stream index."""
        if not self.has_audio or self.global_index < 0:
            raise ValueError("Cannot generate FFmpeg map specifier for empty/no-audio selection")
        return f"0:{self.global_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_stream": self.selected_stream.to_dict() if self.selected_stream else None,
            "global_index": self.global_index,
            "audio_ordinal": self.audio_ordinal,
            "has_audio": self.has_audio,
            "has_warning": self.has_warning,
            "warning": self.warning,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioSelection:
        stream_raw = data.get("selected_stream")
        stream = None
        if isinstance(stream_raw, dict):
            stream = AudioStreamInfo(
                index=int(stream_raw.get("index", 0)),
                codec=str(stream_raw.get("codec", "")),
                channels=int(stream_raw.get("channels", 0)),
                sample_rate=int(stream_raw.get("sample_rate", 0)),
                language=str(stream_raw.get("language", "")),
                title=str(stream_raw.get("title", "")),
                handler_name=str(stream_raw.get("handler_name", "")),
                disposition=dict(stream_raw.get("disposition", {})),
                duration=float(stream_raw["duration"]) if stream_raw.get("duration") is not None else None,
                bitrate=int(stream_raw["bitrate"]) if stream_raw.get("bitrate") is not None else None,
            )
        elif isinstance(stream_raw, AudioStreamInfo):
            stream = stream_raw

        return cls(
            selected_stream=stream,
            global_index=int(data.get("global_index", -1)),
            audio_ordinal=int(data.get("audio_ordinal", -1)),
            warning=data.get("warning"),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class PreparedEpisode:
    """Fully probed and prepared local episode source with deterministic transcript and stream invariants."""

    episode_id: str
    source_id: str
    source_path: Path
    duration_ms: int
    canvas_width: int
    canvas_height: int
    source_basename: str = ""
    source_fingerprint: str = ""
    video_streams: tuple[VideoStreamInfo, ...] = field(default_factory=tuple)
    audio_streams: tuple[AudioStreamInfo, ...] = field(default_factory=tuple)
    subtitle_streams: tuple[SubtitleStreamInfo, ...] = field(default_factory=tuple)
    primary_video: VideoStreamInfo | None = None
    primary_audio: AudioStreamInfo | None = None
    video_info: VideoStreamInfo | None = None
    audio_info: AudioStreamInfo | None = None  # None if source has no audio (no fake audio info!)
    audio_selection: AudioSelection | None = None
    selected_audio: AudioStreamInfo | None = None
    audio_ordinal: int | None = None
    selected_subtitle: SubtitleTrack | None = None
    transcript: Transcript = field(default_factory=Transcript)
    transcript_method: str = "none"
    artifact_hash: str = ""
    dependency_signature: dict[str, Any] = field(default_factory=dict)
    status: str = "ready"
    uncertainty: bool = False
    fallback: bool = False
    source_hash: str = ""
    source_timeline: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_basename and self.source_path:
            object.__setattr__(self, "source_basename", Path(self.source_path).name)
        if not self.source_fingerprint and self.source_hash:
            object.__setattr__(self, "source_fingerprint", self.source_hash)
        elif not self.source_hash and self.source_fingerprint:
            object.__setattr__(self, "source_hash", self.source_fingerprint)
        if self.video_info is None and self.primary_video is not None:
            object.__setattr__(self, "video_info", self.primary_video)
        if self.primary_video is None and self.video_info is not None:
            object.__setattr__(self, "primary_video", self.video_info)
        if self.audio_info is None and self.selected_audio is not None:
            object.__setattr__(self, "audio_info", self.selected_audio)
        if self.selected_audio is None and self.audio_info is not None:
            object.__setattr__(self, "selected_audio", self.audio_info)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_id": self.source_id,
            "source_path": str(self.source_path),
            "source_basename": self.source_basename,
            "source_fingerprint": self.source_fingerprint,
            "source_hash": self.source_hash,
            "duration_ms": self.duration_ms,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "video_streams": [v.to_dict() for v in self.video_streams],
            "audio_streams": [a.to_dict() for a in self.audio_streams],
            "subtitle_streams": [s.to_dict() for s in self.subtitle_streams],
            "primary_video": self.primary_video.to_dict() if self.primary_video else None,
            "primary_audio": self.primary_audio.to_dict() if self.primary_audio else None,
            "video_info": self.video_info.to_dict() if self.video_info else None,
            "audio_info": self.audio_info.to_dict() if self.audio_info else None,
            "audio_selection": self.audio_selection.to_dict() if self.audio_selection else None,
            "selected_audio": self.selected_audio.to_dict() if self.selected_audio else None,
            "audio_ordinal": self.audio_ordinal,
            "selected_subtitle": self.selected_subtitle.to_dict() if self.selected_subtitle else None,
            "transcript": self.transcript.to_dict(),
            "transcript_method": self.transcript_method,
            "artifact_hash": self.artifact_hash,
            "dependency_signature": dict(self.dependency_signature),
            "status": self.status,
            "uncertainty": self.uncertainty,
            "fallback": self.fallback,
            "source_timeline": dict(self.source_timeline),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreparedEpisode:
        from toolrecap_v4.analysis.source_prep.subtitles.models import SubtitleStreamInfo, SubtitleTrack
        def _parse_video(v_raw: dict[str, Any] | None) -> VideoStreamInfo | None:
            if not v_raw:
                return None
            return VideoStreamInfo(
                index=int(v_raw.get("index", 0)),
                codec=str(v_raw.get("codec", "")),
                width=int(v_raw.get("width", 0)),
                height=int(v_raw.get("height", 0)),
                fps=float(v_raw.get("fps", 0.0)),
                fps_text=str(v_raw.get("fps_text", "")),
                duration=float(v_raw.get("duration", 0.0)),
                aspect_ratio=str(v_raw.get("aspect_ratio", "")),
                rotation=int(v_raw.get("rotation", 0)),
                bitrate=int(v_raw["bitrate"]) if v_raw.get("bitrate") is not None else None,
                sar=str(v_raw.get("sar", "")),
                dar=str(v_raw.get("dar", "")),
                pixel_format=str(v_raw.get("pixel_format", "")),
                avg_frame_rate=str(v_raw.get("avg_frame_rate", "")),
                real_frame_rate=str(v_raw.get("real_frame_rate", "")),
                time_base=str(v_raw.get("time_base", "")),
                start_time=float(v_raw.get("start_time", 0.0)),
            )

        def _parse_audio(a_raw: dict[str, Any] | None) -> AudioStreamInfo | None:
            if not a_raw:
                return None
            return AudioStreamInfo(
                index=int(a_raw.get("index", 0)),
                codec=str(a_raw.get("codec", "")),
                channels=int(a_raw.get("channels", 0)),
                sample_rate=int(a_raw.get("sample_rate", 0)),
                language=str(a_raw.get("language", "")),
                title=str(a_raw.get("title", "")),
                handler_name=str(a_raw.get("handler_name", "")),
                disposition=dict(a_raw.get("disposition", {})),
                duration=float(a_raw["duration"]) if a_raw.get("duration") is not None else None,
                bitrate=int(a_raw["bitrate"]) if a_raw.get("bitrate") is not None else None,
            )

        v_info = _parse_video(data.get("video_info")) or _parse_video(data.get("primary_video"))
        a_info = _parse_audio(data.get("audio_info")) or _parse_audio(data.get("selected_audio")) or _parse_audio(data.get("primary_audio"))

        v_streams = tuple(
            _parse_video(v) for v in data.get("video_streams", []) if v
        )
        a_streams = tuple(
            _parse_audio(a) for a in data.get("audio_streams", []) if a
        )
        s_streams = tuple(
            SubtitleStreamInfo(**s) if isinstance(s, dict) else s
            for s in data.get("subtitle_streams", [])
        )

        sel_raw = data.get("audio_selection")
        audio_sel = AudioSelection.from_dict(sel_raw) if isinstance(sel_raw, dict) else sel_raw

        t_raw = data.get("transcript", {})
        transcript = Transcript.from_dict(t_raw) if isinstance(t_raw, dict) else t_raw

        return cls(
            episode_id=str(data.get("episode_id", "")),
            source_id=str(data.get("source_id", "")),
            source_path=Path(data.get("source_path", "")),
            source_basename=str(data.get("source_basename", "")),
            source_fingerprint=str(data.get("source_fingerprint", data.get("source_hash", ""))),
            duration_ms=int(data.get("duration_ms", 0)),
            canvas_width=int(data.get("canvas_width", 0)),
            canvas_height=int(data.get("canvas_height", 0)),
            video_streams=tuple(v for v in v_streams if v is not None),
            audio_streams=tuple(a for a in a_streams if a is not None),
            subtitle_streams=s_streams,
            primary_video=v_info,
            primary_audio=a_info,
            video_info=v_info,
            audio_info=a_info,
            audio_selection=audio_sel,
            selected_audio=a_info,
            audio_ordinal=data.get("audio_ordinal"),
            transcript=transcript,
            transcript_method=str(data.get("transcript_method", "none")),
            artifact_hash=str(data.get("artifact_hash", "")),
            dependency_signature=dict(data.get("dependency_signature", {})),
            status=str(data.get("status", "ready")),
            uncertainty=bool(data.get("uncertainty", False)),
            fallback=bool(data.get("fallback", False)),
            source_hash=str(data.get("source_hash", data.get("source_fingerprint", ""))),
            source_timeline=dict(data.get("source_timeline", {})),
        )
