"""Focused deterministic unit tests for ToolRecap V4 SubtitlePipeline and SourcePreparationPipeline."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from PIL import Image

from toolrecap_v4.analysis.cache import AnalysisCacheManager
from toolrecap_v4.analysis.dependencies import (
    NORMALIZATION_VERSION,
    PGS_DECODER_VERSION,
    PIPELINE_VERSION,
    SUBTITLE_PARSER_VERSION,
    VOBSUB_DECODER_VERSION,
    compute_inventory_hash,
    compute_sidecar_signature,
    compute_source_fingerprint,
    compute_source_signature,
    compute_vobsub_signature,
    hash_file_content,
)
from toolrecap_v4.analysis.models import (
    AudioSelection,
    PreparedEpisode,
    Transcript,
    TranscriptCue,
)
from toolrecap_v4.analysis.source_prep.audio import AudioStreamInfo
from toolrecap_v4.analysis.source_prep.pipeline import (
    SourcePreparationPipeline,
    compute_pipeline_cache_key,
    prepare_episode_source,
)
from toolrecap_v4.analysis.source_prep.probe import EpisodeProbeResult
from toolrecap_v4.analysis.source_prep.stt import (
    DEFAULT_STT_MANIFEST,
    SttResult,
    SttStatus,
)
from toolrecap_v4.analysis.source_prep.subtitles.cache import (
    SubtitleCacheManager,
    compute_subtitle_cache_key,
)
from toolrecap_v4.analysis.source_prep.subtitles.discovery import build_embedded_tracks, discover_sidecars, select_best_english_subtitles
from toolrecap_v4.analysis.source_prep.subtitles.models import (
    SubtitleCue,
    SubtitleStreamInfo,
    SubtitleTrack,
)
from toolrecap_v4.analysis.source_prep.subtitles.ocr import OcrAdapter, OcrResult
from toolrecap_v4.analysis.source_prep.subtitles.parsers import parse_ass
from toolrecap_v4.analysis.source_prep.subtitles.pgs import create_minimal_pgs_sup
from toolrecap_v4.analysis.source_prep.subtitles.pipeline import (
    SubtitlePipeline,
    SubtitlePipelineResult,
    extract_embedded_subtitle_stream,
)
from toolrecap_v4.analysis.source_prep.subtitles.vobsub import create_synthetic_vobsub
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, ToolRecapError
from toolrecap_v4.media import CommandResult, VideoStreamInfo


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,000
Hello from SRT sidecar!

2
00:00:04,500 --> 00:00:06,000
Second subtitle line.
"""

SAMPLE_ASS = """[Script Info]
Title: Test ASS
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello from ASS sidecar!
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,Second line in ASS.
"""

SAMPLE_VTT = """WEBVTT - Test File

00:01.000 --> 00:03.000
Hello from VTT sidecar!

00:04.500 --> 00:06.000
Second line in VTT.
"""

SAMPLE_SSA = """[Script Info]
Title: Real SSA Fixture
ScriptType: v4.00

[V4 Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding
Style: Default,Arial,20,16777215,255,0,0,0,0,1,2,0,2,10,10,10,0,1

[Events]
Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: Marked=0,0:00:01.00,0:00:03.00,Default,,0000,0000,0000,,Hello from real SSA.
Dialogue: Marked=0,0:00:04.00,0:00:06.00,Default,,0000,0000,0000,,Second SSA line.
"""


def test_real_ssa_fixture_discovery_parse_and_pipeline_selection(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    ssa_path = tmp_path / "show.s01e01.eng.ssa"
    ssa_path.write_text(SAMPLE_SSA, encoding="utf-8")

    tracks = discover_sidecars(video, episode_id="show.s01e01")
    assert len(tracks) == 1
    assert tracks[0].source_format == "ssa"
    assert tracks[0].is_full is True

    cues = parse_ass(
        ssa_path,
        source_format="ssa",
        language="eng",
        episode_id="show.s01e01",
        source_duration_ms=60_000,
    )
    assert [cue.text for cue in cues] == ["Hello from real SSA.", "Second SSA line."]
    assert all(cue.source_format == "ssa" for cue in cues)

    pipeline = SubtitlePipeline(cache_manager=SubtitleCacheManager(tmp_path / "cache"))
    result = pipeline.extract_cues(
        tracks[0],
        source_video=video,
        episode_id="show.s01e01",
        source_duration_ms=60_000,
    )
    assert result.status == "success"
    assert result.source_format == "ssa"
    assert len(result.cues) == 2


def _make_dummy_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free")
    return path


def _make_default_probe(
    path: Path,
    *,
    has_audio: bool = True,
    sub_streams: Sequence[SubtitleStreamInfo] = (),
) -> EpisodeProbeResult:
    v = VideoStreamInfo(
        index=0,
        codec="h264",
        width=1920,
        height=1080,
        fps=24.0,
        fps_text="24",
        duration=60.0,
        aspect_ratio="16:9",
        rotation=0,
    )
    a = None
    audio_list: list[AudioStreamInfo] = []
    if has_audio:
        a = AudioStreamInfo(
            index=1,
            codec="aac",
            channels=2,
            sample_rate=48000,
            language="eng",
            title="English Stereo",
            disposition={"default": 1},
            duration=60.0,
        )
        audio_list.append(a)

    return EpisodeProbeResult(
        path=path,
        duration_ms=60000,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        video_streams=(v,),
        audio_streams=tuple(audio_list),
        subtitle_streams=tuple(sub_streams),
        canvas_width=1920,
        canvas_height=1080,
        primary_video=v,
        primary_audio=a,
    )


class MockOcrAdapter(OcrAdapter):
    """Deterministic OCR adapter returning configured text."""

    def __init__(self, text: str = "Bitmap text line", confidence: float = 0.95, status: str = "success") -> None:
        super().__init__()
        self.mock_text = text
        self.mock_confidence = confidence
        self.mock_status = status
        self.call_count = 0

    def ocr_image(self, image: Image.Image, cancellation_token: CancellationToken | None = None) -> OcrResult:
        self.call_count += 1
        is_valid = self.mock_status == "success"
        return OcrResult(
            text=self.mock_text if is_valid else "",
            confidence=self.mock_confidence if is_valid else 0.0,
            is_valid=is_valid,
            reason="mock",
            source="rapidocr" if is_valid else "rejected",
            raw_text=self.mock_text,
        )


def _mock_transcribe_success(video_path: Any, selection: Any, duration_ms: int = 0, **kwargs: Any) -> SttResult:
    return SttResult(
        status=SttStatus.SUCCESS,
        cues=(
            SubtitleCue(
                start_ms=1000,
                end_ms=3000,
                text="Transcribed speech cue 1",
                source_type="stt",
                source_format="whisper",
                language="eng",
                confidence=0.92,
            ),
            SubtitleCue(
                start_ms=4000,
                end_ms=6000,
                text="Transcribed speech cue 2",
                source_type="stt",
                source_format="whisper",
                language="eng",
                confidence=0.95,
            ),
        ),
        device_used="cpu",
    )


def _mock_transcribe_silent(video_path: Any, selection: Any, duration_ms: int = 0, **kwargs: Any) -> SttResult:
    return SttResult(
        status=SttStatus.SILENT,
        cues=(),
        device_used="cpu",
        diagnostics=("Audio measured silent",),
    )


# ============================================================================
# 1. SubtitlePipeline: Text Formats & Embedded Demuxing
# ============================================================================


def test_subtitle_pipeline_sidecar_srt(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    srt_file = tmp_path / "show.s01e01.en.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    track = SubtitleTrack(
        track_id="sidecar:show.s01e01.en.srt",
        source_type="sidecar",
        source_format="srt",
        language="eng",
        source_file=str(srt_file),
    )

    cache = SubtitleCacheManager(tmp_path / "sub_cache")
    pipeline = SubtitlePipeline(cache_manager=cache)

    res = pipeline.extract_cues(track, source_video=video, episode_id="show.s01e01")
    assert res.status == "success"
    assert res.cached is False
    assert len(res.cues) == 2
    assert res.cues[0].text == "Hello from SRT sidecar!"
    assert res.cues[1].text == "Second subtitle line."

    # Second call must load from atomic cache
    res2 = pipeline.extract_cues(track, source_video=video, episode_id="show.s01e01")
    assert res2.status == "success"
    assert res2.cached is True
    assert len(res2.cues) == 2


def test_subtitle_pipeline_sidecar_ass(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    ass_file = tmp_path / "show.s01e01.en.ass"
    ass_file.write_text(SAMPLE_ASS, encoding="utf-8")

    track = SubtitleTrack(
        track_id="sidecar:show.s01e01.en.ass",
        source_type="sidecar",
        source_format="ass",
        language="eng",
        source_file=str(ass_file),
    )

    cache = SubtitleCacheManager(tmp_path / "sub_cache")
    pipeline = SubtitlePipeline(cache_manager=cache)

    res = pipeline.extract_cues(track, source_video=video, episode_id="show.s01e01")
    assert res.status == "success"
    assert len(res.cues) == 2
    assert res.cues[0].text == "Hello from ASS sidecar!"


def test_subtitle_pipeline_sidecar_vtt(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    vtt_file = tmp_path / "show.s01e01.en.vtt"
    vtt_file.write_text(SAMPLE_VTT, encoding="utf-8")

    track = SubtitleTrack(
        track_id="sidecar:show.s01e01.en.vtt",
        source_type="sidecar",
        source_format="vtt",
        language="eng",
        source_file=str(vtt_file),
    )

    cache = SubtitleCacheManager(tmp_path / "sub_cache")
    pipeline = SubtitlePipeline(cache_manager=cache)

    res = pipeline.extract_cues(track, source_video=video, episode_id="show.s01e01")
    assert res.status == "success"
    assert len(res.cues) == 2
    assert res.cues[0].text == "Hello from VTT sidecar!"


def test_subtitle_pipeline_embedded_extraction_mapping(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")

    track = SubtitleTrack(
        track_id="embedded:2:srt",
        source_type="embedded",
        source_format="srt",
        language="eng",
        stream_index=2,
    )

    recorded_cmds: list[list[str]] = []

    def mock_run_cmd(cmd: list[str], **kwargs: Any) -> CommandResult:
        recorded_cmds.append(cmd)
        # Write dummy output file specified as last argument
        out_path = Path(cmd[-1])
        out_path.write_text(SAMPLE_SRT, encoding="utf-8")
        return CommandResult(exit_code=0, stdout="", stderr="")

    cache = SubtitleCacheManager(tmp_path / "sub_cache")
    pipeline = SubtitlePipeline(cache_manager=cache, run_command_fn=mock_run_cmd, temp_dir=tmp_path / "tmp_subs")

    res = pipeline.extract_cues(track, source_video=video, episode_id="show.s01e01")
    assert res.status == "success"
    assert len(res.cues) == 2
    assert res.cues[0].text == "Hello from SRT sidecar!"

    # Verify container global stream mapping: -map 0:2
    assert len(recorded_cmds) == 1
    cmd = recorded_cmds[0]
    assert "-map" in cmd
    map_idx = cmd.index("-map")
    assert cmd[map_idx + 1] == "0:2"


# ============================================================================
# 2. SubtitlePipeline: Bitmap Formats (PGS & VobSub) and OCR Quality Gate
# ============================================================================


def test_subtitle_pipeline_sidecar_pgs_ocr(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    sup_path = tmp_path / "show.s01e01.en.sup"
    sup_path.write_bytes(create_minimal_pgs_sup())

    track = SubtitleTrack(
        track_id="sidecar:show.s01e01.en.sup",
        source_type="sidecar",
        source_format="pgs",
        language="eng",
        is_bitmap=True,
        source_file=str(sup_path),
    )

    mock_ocr = MockOcrAdapter(text="Decoded PGS line", confidence=0.94)
    pipeline = SubtitlePipeline(
        cache_manager=SubtitleCacheManager(tmp_path / "sub_cache"),
        ocr_adapter=mock_ocr,
    )

    res = pipeline.extract_cues(track, source_video=video, episode_id="show.s01e01")
    assert res.status == "success"
    assert len(res.cues) == 1
    assert res.cues[0].text == "Decoded PGS line"
    assert res.cues[0].source_format == "pgs"
    assert mock_ocr.call_count == 1


def test_subtitle_pipeline_sidecar_vobsub_ocr(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    idx_path = tmp_path / "show.s01e01.en.idx"
    create_synthetic_vobsub(idx_path, idx_path.with_suffix(".sub"))

    track = SubtitleTrack(
        track_id="sidecar:show.s01e01.en.idx",
        source_type="sidecar",
        source_format="vobsub",
        language="eng",
        is_bitmap=True,
        source_file=str(idx_path),
    )

    mock_ocr = MockOcrAdapter(text="Decoded VobSub subtitle", confidence=0.91)
    pipeline = SubtitlePipeline(
        cache_manager=SubtitleCacheManager(tmp_path / "sub_cache"),
        ocr_adapter=mock_ocr,
    )

    res = pipeline.extract_cues(track, source_video=video, episode_id="show.s01e01")
    assert res.status == "success"
    assert len(res.cues) == 1
    assert res.cues[0].text == "Decoded VobSub subtitle"
    assert res.cues[0].source_format == "vobsub"


def test_subtitle_pipeline_ocr_quality_gate_dropped_cues(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    idx_path = tmp_path / "show.s01e01.en.idx"
    create_synthetic_vobsub(idx_path, idx_path.with_suffix(".sub"))

    track = SubtitleTrack(
        track_id="sidecar:show.s01e01.en.idx",
        source_type="sidecar",
        source_format="vobsub",
        language="eng",
        is_bitmap=True,
        source_file=str(idx_path),
    )

    # OCR returns empty / low-confidence text
    bad_ocr = MockOcrAdapter(text="", confidence=0.2, status="empty")
    pipeline = SubtitlePipeline(
        cache_manager=SubtitleCacheManager(tmp_path / "sub_cache"),
        ocr_adapter=bad_ocr,
    )

    res = pipeline.extract_cues(track, source_video=video, episode_id="show.s01e01")
    assert res.status == "empty"
    assert len(res.cues) == 0
    assert res.dropped_cues_count == 1
    assert "mock" in res.diagnostics
    again = pipeline.extract_cues(track, source_video=video, episode_id="show.s01e01")
    assert again.cached is False
    assert bad_ocr.call_count == 2


# ============================================================================
# 3. SourcePreparationPipeline: Fallback Hierarchy & Formats
# ============================================================================


def test_source_prep_sidecar_text_preferred(tmp_path: Path):
    # Setup: video with embedded subtitle AND sidecar SRT in same folder
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    srt_file = tmp_path / "show.s01e01.en.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    sub_stream = SubtitleStreamInfo(
        index=2,
        subtitle_index=0,
        codec="subrip",
        language="eng",
        title="Embedded English",
        default=True,
    )
    probe_res = _make_default_probe(video, sub_streams=[sub_stream])

    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=MagicMock(side_effect=AssertionError("STT should not be called!")),
    )

    ep = pipeline.prepare_episode(video, episode_id="show.s01e01")
    assert ep.transcript_method == "sidecar"
    assert ep.fallback is False
    assert ep.status == "ready"
    assert len(ep.transcript.cues) == 2
    assert ep.transcript.cues[0].text == "Hello from SRT sidecar!"
    assert ep.selected_subtitle is not None
    assert ep.selected_subtitle.source_type == "sidecar"


def test_source_prep_embedded_text_used_when_no_sidecar(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")

    sub_stream = SubtitleStreamInfo(
        index=2,
        subtitle_index=0,
        codec="subrip",
        language="eng",
        title="Embedded English",
        default=True,
    )
    probe_res = _make_default_probe(video, sub_streams=[sub_stream])

    def mock_run_cmd(cmd: list[str], **kwargs: Any) -> CommandResult:
        out_path = Path(cmd[-1])
        out_path.write_text(SAMPLE_SRT, encoding="utf-8")
        return CommandResult(exit_code=0, stdout="", stderr="")

    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        run_command_fn=mock_run_cmd,
        transcribe_fn=MagicMock(side_effect=AssertionError("STT should not be called!")),
    )

    ep = pipeline.prepare_episode(video, episode_id="show.s01e01")
    assert ep.transcript_method == "embedded"
    assert ep.fallback is False
    assert ep.status == "ready"
    assert len(ep.transcript.cues) == 2


def test_source_prep_embedded_mov_text_is_transcoded_to_srt_without_stt(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "Blue Bloods - 3x01 - Family Business.mp4")
    sub_stream = SubtitleStreamInfo(
        index=2,
        subtitle_index=0,
        codec="mov_text",
        language="eng",
        title="English",
        default=True,
    )
    probe_res = _make_default_probe(video, sub_streams=[sub_stream])
    captured_commands: list[list[str]] = []

    def mock_run_cmd(cmd: list[str], **kwargs: Any) -> CommandResult:
        captured_commands.append(cmd)
        output_path = Path(cmd[-1])
        output_path.write_text(SAMPLE_SRT, encoding="utf-8")
        return CommandResult(exit_code=0, stdout="", stderr="")

    transcribe = MagicMock(side_effect=AssertionError("mov_text must not fall back to STT"))
    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        subtitle_pipeline=SubtitlePipeline(
            cache_manager=SubtitleCacheManager(tmp_path / "subtitle_cache"),
            run_command_fn=mock_run_cmd,
        ),
        probe_fn=lambda *args, **kwargs: probe_res,
        run_command_fn=mock_run_cmd,
        transcribe_fn=transcribe,
    )

    episode = pipeline.prepare_episode(video, episode_id="E01")

    assert episode.transcript_method == "embedded"
    assert episode.fallback is False
    assert episode.selected_subtitle is not None
    assert episode.selected_subtitle.source_format == "srt"
    assert episode.transcript.cue_count == 2
    assert transcribe.call_count == 0
    assert len(captured_commands) == 1
    command = captured_commands[0]
    assert command[command.index("-map") + 1] == "0:2"
    assert command[command.index("-c:s") + 1] == "srt"
    assert Path(command[-1]).suffix == ".srt"


def test_explicit_english_track_title_recovers_missing_language_tag():
    streams = [
        SubtitleStreamInfo(index=2, subtitle_index=0, codec="subrip", language="und", title="English SDH"),
        SubtitleStreamInfo(index=3, subtitle_index=1, codec="subrip", language="fra", title="French"),
    ]
    tracks = build_embedded_tracks(streams, source_video="episode.mkv")
    chosen = select_best_english_subtitles(tracks).best_english_full
    assert chosen is not None and chosen.stream_index == 2 and chosen.language == "eng"


def test_source_prep_bitmap_ocr_used_when_no_text(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    idx_path = tmp_path / "show.s01e01.en.idx"
    create_synthetic_vobsub(idx_path, idx_path.with_suffix(".sub"))

    probe_res = _make_default_probe(video, sub_streams=[])
    mock_ocr = MockOcrAdapter(text="VobSub OCR cue", confidence=0.96)

    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        ocr_adapter=mock_ocr,
        transcribe_fn=MagicMock(side_effect=AssertionError("STT should not be called!")),
    )

    ep = pipeline.prepare_episode(video, episode_id="show.s01e01")
    assert ep.transcript_method == "ocr"
    assert ep.fallback is False
    assert ep.status == "ready"
    assert len(ep.transcript.cues) == 1
    assert ep.transcript.cues[0].text == "VobSub OCR cue"


def test_v5_source_prep_rejects_bitmap_subtitles_without_running_ocr(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    idx_path = tmp_path / "show.s01e01.en.idx"
    create_synthetic_vobsub(idx_path, idx_path.with_suffix(".sub"))
    probe_res = _make_default_probe(video, sub_streams=[])
    mock_ocr = MockOcrAdapter(text="Must not be used", confidence=0.99)
    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        ocr_adapter=mock_ocr,
        allow_stt=False,
        allow_bitmap_ocr=False,
        transcribe_fn=MagicMock(side_effect=AssertionError("STT should not be called")),
    )
    with pytest.raises(ToolRecapError, match="OCR, which is disabled in V5"):
        pipeline.prepare_episode(video, episode_id="show.s01e01")
    assert mock_ocr.call_count == 0


def test_preparation_tries_next_english_track_and_reuses_result(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    (tmp_path / "show.s01e01.en.srt").write_text("not a valid SRT", encoding="utf-8")
    embedded = SubtitleStreamInfo(
        index=2, subtitle_index=0, codec="subrip", language="eng", default=True,
    )
    probe_result = _make_default_probe(video, sub_streams=[embedded])
    commands = []

    def demux(cmd: list[str], **_kwargs: Any) -> CommandResult:
        commands.append(cmd)
        Path(cmd[-1]).write_text(SAMPLE_SRT, encoding="utf-8")
        return CommandResult(exit_code=0, stdout="", stderr="")

    stt = MagicMock(side_effect=AssertionError("STT must not run"))
    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "analysis"),
        subtitle_pipeline=SubtitlePipeline(
            cache_manager=SubtitleCacheManager(tmp_path / "subtitles"), run_command_fn=demux,
        ),
        probe_fn=lambda *args, **kwargs: probe_result,
        transcribe_fn=stt,
    )
    first = pipeline.prepare_episode(video, episode_id="E01")
    second = pipeline.prepare_episode(video, episode_id="E01")
    assert first.selected_subtitle.track_id == "embedded:2:srt"
    assert first.transcript.cue_count == 2
    assert second.artifact_hash == first.artifact_hash
    assert len(commands) == 1
    stt.assert_not_called()


def test_default_preparation_reports_missing_subtitles_without_whisper(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    probe_result = _make_default_probe(video, sub_streams=[])
    stt = MagicMock(side_effect=AssertionError("Whisper must not run"))
    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "analysis"),
        probe_fn=lambda *args, **kwargs: probe_result,
        transcribe_fn=stt,
    )
    with pytest.raises(ToolRecapError, match="No English full subtitle track"):
        pipeline.prepare_episode(video, episode_id="E01")
    stt.assert_not_called()


def test_source_prep_stt_fallback_when_no_subtitles(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "show.s01e01.mkv")
    probe_res = _make_default_probe(video, sub_streams=[])

    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=_mock_transcribe_success,
        allow_stt=True,
    )

    ep = pipeline.prepare_episode(video, episode_id="show.s01e01")
    assert ep.transcript_method == "stt"
    assert ep.fallback is True
    assert ep.status == "ready"
    assert len(ep.transcript.cues) == 2
    assert ep.transcript.cues[0].text == "Transcribed speech cue 1"
    assert ep.selected_subtitle is None


# ============================================================================
# 4. Cache Invalidation: New Better Sidecar Invalidates Old STT
# ============================================================================


def test_source_prep_new_sidecar_invalidates_old_stt(tmp_path: Path):
    video_dir = tmp_path / "episodes"
    video = _make_dummy_video(video_dir / "series.s01e01.mkv")
    probe_res = _make_default_probe(video, sub_streams=[])
    cache_mgr = AnalysisCacheManager(tmp_path / "ep_cache")

    transcribe_mock = MagicMock(side_effect=_mock_transcribe_success)

    pipeline = SourcePreparationPipeline(
        cache_manager=cache_mgr,
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=transcribe_mock,
        allow_stt=True,
    )

    # 1. Run 1: No sidecars -> runs STT and caches artifact
    ep1 = pipeline.prepare_episode(video, episode_id="series.s01e01")
    assert ep1.transcript_method == "stt"
    assert transcribe_mock.call_count == 1

    # 2. Add new English full text sidecar to folder!
    new_sidecar = video_dir / "series.s01e01.en.srt"
    new_sidecar.write_text(SAMPLE_SRT, encoding="utf-8")

    # 3. Run 2: Fresh inventory scan must detect new sidecar and invalidate old STT cache!
    ep2 = pipeline.prepare_episode(video, episode_id="series.s01e01")
    assert ep2.transcript_method == "sidecar"
    assert ep2.fallback is False
    assert len(ep2.transcript.cues) == 2
    assert ep2.transcript.cues[0].text == "Hello from SRT sidecar!"

    # STT must NOT have been called a second time
    assert transcribe_mock.call_count == 1


def test_source_prep_content_change_invalidates_cache(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "series.s01e02.mkv")
    srt_file = tmp_path / "series.s01e02.en.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")
    probe_res = _make_default_probe(video)
    cache_mgr = AnalysisCacheManager(tmp_path / "ep_cache")

    pipeline = SourcePreparationPipeline(
        cache_manager=cache_mgr,
        probe_fn=lambda *args, **kwargs: probe_res,
    )

    ep1 = pipeline.prepare_episode(video, episode_id="series.s01e02")
    assert ep1.transcript.cues[0].text == "Hello from SRT sidecar!"

    # Modify sidecar text
    modified_srt = SAMPLE_SRT.replace("Hello from SRT sidecar!", "Updated text line!")
    srt_file.write_text(modified_srt, encoding="utf-8")

    # Second preparation must detect content hash change and re-parse
    ep2 = pipeline.prepare_episode(video, episode_id="series.s01e02")
    assert ep2.transcript.cues[0].text == "Updated text line!"


def test_source_prep_cache_corruption_treated_as_miss(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "series.s01e03.mkv")
    srt_file = tmp_path / "series.s01e03.en.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")
    probe_res = _make_default_probe(video)
    cache_mgr = AnalysisCacheManager(tmp_path / "ep_cache")

    pipeline = SourcePreparationPipeline(
        cache_manager=cache_mgr,
        probe_fn=lambda *args, **kwargs: probe_res,
    )

    ep1 = pipeline.prepare_episode(video, episode_id="series.s01e03")
    assert ep1.status == "ready"

    # Corrupt the cached data file on disk
    for data_file in (tmp_path / "ep_cache").glob("*.data.json"):
        data_file.write_bytes(b"CORRUPTED TRUNCATED DATA")

    # Run again: corruption is detected as miss and re-prepared cleanly
    ep2 = pipeline.prepare_episode(video, episode_id="series.s01e03")
    assert ep2.status == "ready"
    assert len(ep2.transcript.cues) == 2


# ============================================================================
# 5. Acoustic Edge Cases: No-Audio & Silent Audio
# ============================================================================


def test_source_prep_no_audio_handling(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "silent_video.mkv")
    # Probe result has NO audio streams
    probe_res = _make_default_probe(video, has_audio=False, sub_streams=[])

    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=MagicMock(side_effect=AssertionError("STT cannot run on no-audio!")),
    )

    ep = pipeline.prepare_episode(video, episode_id="silent_video")
    assert ep.status == "no_audio"
    assert ep.transcript_method == "none"
    assert ep.transcript.has_speech is False
    assert len(ep.transcript.cues) == 0
    assert ep.audio_info is None
    assert ep.selected_audio is None
    assert ep.audio_ordinal is None
    assert ep.audio_selection is not None
    assert ep.audio_selection.has_audio is False


def test_source_prep_silent_audio_handling(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "quiet.mkv")
    probe_res = _make_default_probe(video, has_audio=True, sub_streams=[])

    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=_mock_transcribe_silent,
        allow_stt=True,
    )

    ep = pipeline.prepare_episode(video, episode_id="quiet")
    assert ep.status == "silent"
    assert ep.transcript_method == "stt"
    assert ep.transcript.has_speech is False
    assert len(ep.transcript.cues) == 0
    assert ep.fallback is True


# ============================================================================
# 6. Cancellation Safety, Progress, and PreparedEpisode Invariants
# ============================================================================


def test_source_prep_cancellation_never_promotes_cache(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "cancel_test.mkv")
    probe_res = _make_default_probe(video)
    cache_mgr = AnalysisCacheManager(tmp_path / "ep_cache")

    token = CancellationToken()

    def cancel_during_probe(*args: Any, **kwargs: Any) -> EpisodeProbeResult:
        token.cancel()
        token.check_cancelled()
        return probe_res

    pipeline = SourcePreparationPipeline(
        cache_manager=cache_mgr,
        probe_fn=cancel_during_probe,
    )

    with pytest.raises(CancelledError):
        pipeline.prepare_episode(video, episode_id="cancel_test", cancellation_token=token)

    # Verify no cache artifacts promoted
    assert list((tmp_path / "ep_cache").glob("*.data.json")) == []
    assert list((tmp_path / "ep_cache").glob("*.manifest.json")) == []


def test_source_prep_progress_reporting(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "progress_test.mkv")
    srt_file = tmp_path / "progress_test.en.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")
    probe_res = _make_default_probe(video)

    phases_recorded: list[tuple[str, float]] = []

    def on_progress(phase: str, pct: float, msg: str) -> None:
        phases_recorded.append((phase, pct))

    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
    )

    ep = pipeline.prepare_episode(video, episode_id="progress_test", on_progress=on_progress)
    assert ep.status == "ready"

    # Verify key progression checkpoints were notified
    phase_names = [p[0] for p in phases_recorded]
    assert "probe" in phase_names
    assert "inventory" in phase_names
    assert "subtitles" in phase_names
    assert "complete" in phase_names

    # Verify percentage monotonicity
    percentages = [p[1] for p in phases_recorded]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 1.0


def test_prepared_episode_immutability_and_roundtrip(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "roundtrip.mkv")
    srt_file = tmp_path / "roundtrip.en.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")
    probe_res = _make_default_probe(video)

    ep = prepare_episode_source(
        video,
        episode_id="roundtrip",
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
    )

    # Immutability
    with pytest.raises(FrozenInstanceError):
        ep.status = "tampered"  # type: ignore

    # Serialization roundtrip
    data = ep.to_dict()
    ep2 = PreparedEpisode.from_dict(data)

    assert ep2.episode_id == ep.episode_id
    assert ep2.duration_ms == ep.duration_ms
    assert ep2.canvas_width == ep.canvas_width
    assert ep2.canvas_height == ep.canvas_height
    assert ep2.transcript_method == ep.transcript_method
    assert ep2.artifact_hash == ep.artifact_hash
    assert len(ep2.transcript.cues) == len(ep.transcript.cues)
    assert ep2.transcript.cues[0].text == ep.transcript.cues[0].text


# ============================================================================
# 7. Prime Review Regression Tests
# ============================================================================


def test_transcribe_episode_stt_correct_kwargs_and_duration(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "stt_kwargs.mkv")
    probe_res = _make_default_probe(video, sub_streams=[])

    captured_kwargs: dict[str, Any] = {}

    def mock_transcribe(**kwargs: Any) -> SttResult:
        captured_kwargs.update(kwargs)
        return SttResult(
            status=SttStatus.SUCCESS,
            cues=(
                SubtitleCue(
                    start_ms=1000,
                    end_ms=3000,
                    text="Mock line",
                    source_type="stt",
                    source_format="whisper",
                    language="eng",
                ),
            ),
            device_used="cpu",
        )

    pipeline = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=mock_transcribe,
        allow_stt=True,
    )

    ep = pipeline.prepare_episode(video, episode_id="stt_kwargs")
    assert ep.transcript_method == "stt"

    # Verify exact argument names passed to transcribe_episode_stt:
    assert "selection" in captured_kwargs
    assert isinstance(captured_kwargs["selection"], AudioSelection)
    assert "duration_ms" in captured_kwargs
    assert captured_kwargs["duration_ms"] == 60000
    assert "audio_selection" not in captured_kwargs


def test_stt_status_mapping_and_technical_failures_never_ready_empty(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "stt_status.mkv")
    probe_res = _make_default_probe(video, sub_streams=[])

    # 1. SttStatus.CANCELLED propagates as CancelledError
    def transcribe_cancelled(**kwargs: Any) -> SttResult:
        return SttResult(status=SttStatus.CANCELLED, cues=(), error="Operation was cancelled")

    pipeline_cancel = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=transcribe_cancelled,
        allow_stt=True,
    )
    with pytest.raises(CancelledError):
        pipeline_cancel.prepare_episode(video, episode_id="stt_cancel")

    # 2. SttStatus.FAILED raises ToolRecapError (never silently succeeds as empty)
    def transcribe_failed(**kwargs: Any) -> SttResult:
        return SttResult(status=SttStatus.FAILED, cues=(), error="CUDA out of memory", diagnostics=("OOM error",))

    pipeline_failed = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=transcribe_failed,
        allow_stt=True,
    )
    with pytest.raises(ToolRecapError) as exc_info:
        pipeline_failed.prepare_episode(video, episode_id="stt_fail")
    assert "CUDA out of memory" in str(exc_info.value)

    # 3. SttStatus.RUNTIME_UNAVAILABLE raises ToolRecapError
    def transcribe_runtime_unavail(**kwargs: Any) -> SttResult:
        return SttResult(status=SttStatus.RUNTIME_UNAVAILABLE, cues=(), error="Model missing", diagnostics=("Snapshot not found",))

    pipeline_unavail = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=transcribe_runtime_unavail,
        allow_stt=True,
    )
    with pytest.raises(ToolRecapError):
        pipeline_unavail.prepare_episode(video, episode_id="stt_unavail")

    # 4. SttStatus.SUCCESS_EMPTY remains distinguishable from ordinary ready output
    def transcribe_success_empty(**kwargs: Any) -> SttResult:
        return SttResult(status=SttStatus.SUCCESS_EMPTY, cues=(), diagnostics=("Legitimately empty speech",))

    pipeline_empty = SourcePreparationPipeline(
        cache_manager=AnalysisCacheManager(tmp_path / "ep_cache"),
        probe_fn=lambda *args, **kwargs: probe_res,
        transcribe_fn=transcribe_success_empty,
        allow_stt=True,
    )
    ep = pipeline_empty.prepare_episode(video, episode_id="stt_empty")
    assert ep.status == "empty_transcript"
    assert len(ep.transcript.cues) == 0
    assert ep.transcript.has_speech is False


def test_fallback_order_sidecar_text_over_higher_score_embedded(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "pref_test.mkv")
    srt_file = tmp_path / "pref_test.en.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    # Embedded ASS stream with default=1 gets score 100 + 10 + 5 = 115
    embedded_track = SubtitleTrack(
        track_id="embedded:2:ass",
        source_type="embedded",
        source_format="ass",
        language="eng",
        stream_index=2,
        score=115.0,
    )
    # Sidecar SRT gets score 100 + 10 = 110
    sidecar_track = SubtitleTrack(
        track_id="sidecar:pref_test.en.srt",
        source_type="sidecar",
        source_format="srt",
        language="eng",
        source_file=str(srt_file),
        score=110.0,
    )

    # select_best_english_subtitles must pick sidecar text over embedded text regardless of score!
    disc = select_best_english_subtitles([embedded_track, sidecar_track], video_path=str(video))
    assert disc.best_english_full == sidecar_track
    assert disc.best_english_full.source_type == "sidecar"


def test_fallback_order_embedded_text_over_bitmap(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "embed_vs_bitmap.mkv")
    embedded_text = SubtitleTrack(
        track_id="embedded:2:srt",
        source_type="embedded",
        source_format="srt",
        language="eng",
        stream_index=2,
        is_bitmap=False,
        score=100.0,
    )
    bitmap_sidecar = SubtitleTrack(
        track_id="sidecar:embed_vs_bitmap.en.idx",
        source_type="sidecar",
        source_format="vobsub",
        language="eng",
        source_file="embed_vs_bitmap.en.idx",
        is_bitmap=True,
        score=65.0,
    )

    disc = select_best_english_subtitles([embedded_text, bitmap_sidecar], video_path=str(video))
    assert disc.best_english_full == embedded_text


def test_inventory_hashing_diagnostics_on_missing_pair_and_cancellation(tmp_path: Path):
    # 1. Malformed VobSub (.idx without .sub) does NOT crash and is not silently lost
    lonely_idx = tmp_path / "lonely.idx"
    lonely_idx.write_text("# VobSub index\n", encoding="utf-8")

    track = SubtitleTrack(
        track_id="sidecar:lonely.idx",
        source_type="sidecar",
        source_format="vobsub",
        language="eng",
        source_file=str(lonely_idx),
    )

    inv_hash, items = compute_inventory_hash([track])
    assert len(items) == 1
    assert items[0]["name"] == "lonely.idx"
    assert "error" in items[0]
    assert items[0]["hash"] == "malformed_pair"

    # 2. Cancellation token is not swallowed
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        compute_inventory_hash([track], cancellation_token=token)


def test_source_fingerprint_uses_content_hash(tmp_path: Path):
    f1 = tmp_path / "v1.bin"
    f2 = tmp_path / "v2.bin"

    # Same size, different bytes
    f1.write_bytes(b"A" * 1024)
    f2.write_bytes(b"B" * 1024)

    fp1 = compute_source_fingerprint(f1)
    fp2 = compute_source_fingerprint(f2)

    assert fp1 != fp2
    assert fp1 == hash_file_content(f1)
    assert fp2 == hash_file_content(f2)

    # Supplied fingerprint is preserved
    supplied = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    assert compute_source_fingerprint(f1, supplied_hash=supplied) == supplied


def test_semantic_version_invalidation_pipeline_and_subtitle_keys():
    # 1. Pipeline cache key changes when semantic version or policies change
    k1 = compute_pipeline_cache_key(
        episode_id="ep1",
        source_fingerprint="abc",
        inventory_hash="inv1",
        selected_track_id="t1",
        selected_method="sidecar",
        target_hash="tgt1",
        pipeline_version="1.0.0",
        parser_version="1.0.0",
    )
    k2 = compute_pipeline_cache_key(
        episode_id="ep1",
        source_fingerprint="abc",
        inventory_hash="inv1",
        selected_track_id="t1",
        selected_method="sidecar",
        target_hash="tgt1",
        pipeline_version="1.1.0",  # Bump
        parser_version="1.0.0",
    )
    assert k1 != k2

    # 2. Subtitle cache key changes when parser version changes
    track = SubtitleTrack(
        track_id="sidecar:ep1.en.srt",
        source_type="sidecar",
        source_format="srt",
        language="eng",
        source_file="ep1.en.srt",
    )
    sk1 = compute_subtitle_cache_key("ep1", "video.mkv", track, parser_version="1.0.0")
    sk2 = compute_subtitle_cache_key("ep1", "video.mkv", track, parser_version="1.0.1")
    assert sk1 != sk2


def test_embedded_vobsub_demux_paired_output_validation(tmp_path: Path):
    video = _make_dummy_video(tmp_path / "vob_demux.mkv")
    out_idx = tmp_path / "out.idx"

    # Mock runner that writes only .idx without .sub -> must raise ToolRecapError
    def mock_bad_vobsub(cmd: list[str], **kwargs: Any) -> CommandResult:
        out_idx.write_text("# idx only", encoding="utf-8")
        return CommandResult(exit_code=0, stdout="", stderr="")

    with pytest.raises(ToolRecapError) as exc_info:
        extract_embedded_subtitle_stream(
            video,
            stream_index=2,
            output_path=out_idx,
            source_format="vobsub",
            run_command_fn=mock_bad_vobsub,
        )
    assert "paired VobSub files" in str(exc_info.value)
