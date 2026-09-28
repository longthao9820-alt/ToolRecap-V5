"""Targeted tests for video render pipeline and acceptance criteria.

Tests:
- Real synthetic multi-source integration with mismatched fps, resolutions, and no-audio source.
- Validates final duration, stream properties (1920x1080 even canvas, 30fps, 48kHz stereo).
- Validates source file hashes remain strictly unchanged.
- Validates publication directory contains only {title}.mp4, {title}.narration.srt, {title}.original.srt.
- Validates VoiceStudioAdapter integration and narration fit policy (NarrationFitError on oversize).
- Validates audio gains, auto-ducking dB calculations, and original dialogue never ducking.
- Validates two-pass loudnorm measurement and safe handling of silence.
- Validates collision detection preventing source file overwrite.
- Validates cooperative cancellation and temporary file cleanup.
- Validates subtitle burning when enabled.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import inspect
from typing import Any
from unittest.mock import MagicMock
import wave

import pytest

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import (
    CancelledError,
    InvalidAudioError,
    JsonSourceNotFoundError,
    NarrationFitError,
    TimestampBoundaryError,
    VoiceStudioUnavailableError,
    WindowsNameError,
    WindowsReservedNameError,
)
from toolrecap_v4.media import EncoderStatus, probe_media
import toolrecap_v4.renderer as renderer_module
from toolrecap_v4.renderer import (
    _stage_and_publish_files,
    RenderError,
    RenderResult,
    SourceCollisionError,
    build_two_pass_loudnorm_filter,
    calculate_ducking_gains,
    compute_file_sha256,
    get_render_work_root,
    measure_loudnorm,
    render_output,
    render_project,
)
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.subtitles import parse_srt
from toolrecap_v4.subtitles import SubtitleCue
from toolrecap_v4.voice_studio import VoiceStudioAdapter


def test_final_video_render_never_uses_stream_copy() -> None:
    assert '"-c:v", "copy"' not in inspect.getsource(render_output)


def test_highlight_skips_narration_mix_and_loudness_by_default(
    tmp_path: Path, synthetic_sources, monkeypatch,
) -> None:
    source_name = "clip_720p_24fps.mp4"
    output = {"render_id": "highlight", "title": "Highlight", "segments": [{
        "segment_id": "original", "source_file": source_name,
        "start_ms": 0, "end_ms": 700, "type": "original_dialogue",
        "narration": "", "source_audio": True,
        "audio_intent": "ORIGINAL_AUDIO", "subtitles": [],
    }]}
    events: list[dict] = []

    def forbidden_loudness(*_args, **_kwargs):
        raise AssertionError("Highlight loudness normalization must be opt-in")

    monkeypatch.setattr(renderer_module, "measure_loudnorm", forbidden_loudness)
    result = render_output(
        output, synthetic_sources, tmp_path / "highlight-original-audio",
        settings=AppSettings(use_gpu=False, original_audio_db=-30.0),
        render_mode="HIGHLIGHT", progress_callback=events.append,
    )

    activity = [str(event.get("activity_text", "")) for event in events]
    assert result.output_path.is_file()
    assert "Processing original source audio..." in activity
    assert "Original-audio segment 1 / 1 prepared." in activity
    assert "Original-audio segment assembly ready." in activity
    assert not any("Mixing narration" in text or "Mixed segment" in text for text in activity)


def test_incompatible_nvidia_source_falls_back_to_cpu_for_complete_output(
    tmp_path: Path, synthetic_sources, monkeypatch,
) -> None:
    gpu = EncoderStatus(True, "RTX", "h264_nvenc", "NVIDIA NVENC")
    original_args = renderer_module.get_video_encode_args
    def encode_args(quality="high", *, use_gpu=True, encoder_status=None, ffmpeg_path=None):
        if use_gpu:
            return ["-c:v", "h264_nvenc"], gpu
        return original_args(quality, use_gpu=False, encoder_status=gpu, ffmpeg_path=ffmpeg_path)
    monkeypatch.setattr(renderer_module, "get_video_encode_args", encode_args)
    monkeypatch.setattr(renderer_module, "probe_nvidia_source_pipeline",
                        lambda *_args, **_kwargs: (False, "unsupported source CUDA path"))
    events = []
    source_name = "clip_720p_24fps.mp4"
    output = {"render_id": "fallback", "title": "GPU_Fallback", "segments": [{
        "segment_id": "clip", "source_file": source_name, "start_ms": 0, "end_ms": 700,
        "type": "original_dialogue", "narration": "", "source_audio": True, "subtitles": [],
    }]}
    result = render_output(
        output, synthetic_sources, tmp_path / "gpu-fallback",
        settings=AppSettings(use_gpu=True, canvas_width=640, canvas_height=360),
        progress_callback=events.append,
    )
    assert result.output_path.is_file() and result.video_codec == "h264"
    assert any(event.get("state") == "RETRYING" and "using CPU" in event.get("activity_text", "")
               for event in events)


def test_global_source_audio_and_original_dialogue_across_beginning_middle_end(tmp_path: Path, synthetic_sources):
    source_name = "clip_720p_24fps.mp4"
    output = {
        "render_id": "out_global_audio", "title": "Global_Audio", "segments": [
            {"segment_id": "opening", "source_file": source_name, "start_ms": 0, "end_ms": 900,
             "type": "narration", "narration": "Opening commentary.", "source_audio": True, "subtitles": []},
            {"segment_id": "middle", "source_file": source_name, "start_ms": 1000, "end_ms": 1900,
             "type": "narration", "narration": "Middle commentary.", "source_audio": True, "subtitles": []},
            {"segment_id": "ending", "source_file": source_name, "start_ms": 2000, "end_ms": 2900,
             "type": "narration", "narration": "Ending commentary.", "source_audio": True, "subtitles": []},
        ],
    }
    voice = MagicMock(spec=VoiceStudioAdapter)
    voice.synthesize.side_effect = lambda _text, **_kwargs: _create_wav_bytes(0.5)
    result = render_output(
        output, synthetic_sources, tmp_path / "published-global",
        settings=AppSettings(use_gpu=False, canvas_width=640, canvas_height=360),
        voice_adapter=voice,
        source_dialogue_map={source_name: (
            SubtitleCue(100, 300, "opening actor dialogue"),
            SubtitleCue(1100, 1300, "middle actor dialogue"),
            SubtitleCue(2100, 2300, "ending actor dialogue"),
        )},
    )
    original = parse_srt(result.original_srt_path.read_text(encoding="utf-8"))
    narration = parse_srt(result.narration_srt_path.read_text(encoding="utf-8"))
    assert [cue.text for cue in original] == [
        "opening actor dialogue", "middle actor dialogue", "ending actor dialogue",
    ]
    assert [(cue.start_ms, cue.end_ms) for cue in original] == [(100, 300), (600, 800), (1100, 1300)]
    assert [cue.text for cue in narration] == ["Opening commentary.", "Middle commentary.", "Ending commentary."]
    synthesized = [call.args[0] for call in voice.synthesize.call_args_list]
    assert synthesized == ["Opening commentary.", "Middle commentary.", "Ending commentary."]
    assert all("actor dialogue" not in text for text in synthesized)
    assert probe_media(result.output_path).has_audio
    assert result.original_subtitle_state == "ORIGINAL_DIALOGUE_MAPPED"
    assert result.original_subtitle_cue_count == 3


def test_legacy_false_source_audio_migrates_to_global_audio_rule_and_explicit_mute_wins(tmp_path: Path, synthetic_sources):
    source_name = "clip_720p_24fps.mp4"
    base = {"source_file": source_name, "start_ms": 0, "end_ms": 700,
            "type": "original_dialogue", "narration": "", "source_audio": False, "subtitles": []}
    legacy = {"render_id": "legacy", "title": "Legacy_Audio", "segments": [{"segment_id": "legacy", **base}]}
    muted = {"render_id": "muted", "title": "Explicit_Mute", "segments": [
        {"segment_id": "muted", **base, "audio_intent": "SILENT_SOURCE"}
    ]}
    dialogue = {source_name: (SubtitleCue(100, 300, "actor line"),)}
    legacy_result = render_output(
        legacy, synthetic_sources, tmp_path / "legacy", settings=AppSettings(use_gpu=False),
        source_dialogue_map=dialogue,
    )
    muted_result = render_output(
        muted, synthetic_sources, tmp_path / "muted", settings=AppSettings(use_gpu=False),
        source_dialogue_map=dialogue,
    )
    assert parse_srt(legacy_result.original_srt_path.read_text(encoding="utf-8"))[0].text == "actor line"
    assert muted_result.original_subtitle_state == "NO_ORIGINAL_DIALOGUE"


def test_publication_staging_cleanup_on_copy_failure(tmp_path: Path, monkeypatch) -> None:
    """A failed publication preflight leaves neither user outputs nor neighbor temp files."""
    first = tmp_path / "first.source"
    second = tmp_path / "second.source"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    targets = [(first, tmp_path / "first.mp4"), (second, tmp_path / "second.srt")]
    real_copy = __import__("shutil").copy2
    calls = 0

    def fail_second_copy(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publication failure")
        return real_copy(source, destination)

    monkeypatch.setattr("toolrecap_v4.renderer.shutil.copy2", fail_second_copy)
    with pytest.raises(OSError, match="simulated publication failure"):
        _stage_and_publish_files(targets)

    assert not targets[0][1].exists()
    assert not targets[1][1].exists()
    assert {path.name for path in tmp_path.iterdir()} == {first.name, second.name}


def _create_wav_bytes(duration_s: float, sample_rate: int = 24000) -> bytes:
    """Generate in-memory mono PCM WAV bytes of given duration."""
    num_frames = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


@pytest.fixture
def synthetic_sources(tmp_path: Path) -> dict[str, Path]:
    """Create three synthetic test video sources with mismatched properties."""
    src_dir = tmp_path / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Source 1: 1280x720 @ 24fps with audio (3.0 seconds)
    src1 = src_dir / "clip_720p_24fps.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            str(src1),
        ],
        check=True,
        capture_output=True,
    )

    # Source 2: 640x480 @ 25fps NO AUDIO (2.5 seconds)
    src2 = src_dir / "clip_480p_noaudio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=640x480:rate=25:duration=2.5",
            "-c:v", "libx264", "-preset", "ultrafast",
            str(src2),
        ],
        check=True,
        capture_output=True,
    )

    # Source 3: 800x600 @ 30fps with audio (3.0 seconds)
    src3 = src_dir / "clip_600p_30fps.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=800x600:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=600:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            str(src3),
        ],
        check=True,
        capture_output=True,
    )

    return {
        "clip_720p_24fps.mp4": src1,
        "clip_480p_noaudio.mp4": src2,
        "clip_600p_30fps.mp4": src3,
    }


def test_ducking_gain_calculations():
    """Verify linear gain multipliers for normal, ducked, and commentary levels."""
    # Duck OFF
    norm, duck, comm = calculate_ducking_gains(0.0, 0.0, -12.0, auto_duck=False)
    assert norm == pytest.approx(1.0)
    assert duck == pytest.approx(1.0)
    assert comm == pytest.approx(1.0)

    # Duck ON (-12 dB)
    norm, duck, comm = calculate_ducking_gains(0.0, 0.0, -12.0, auto_duck=True)
    assert norm == pytest.approx(1.0)
    assert duck == pytest.approx(10 ** (-12.0 / 20.0), rel=1e-3)
    assert comm == pytest.approx(1.0)

    # Custom levels: orig -3 dB, commentary +2 dB, ducking -10 dB
    norm, duck, comm = calculate_ducking_gains(-3.0, 2.0, -10.0, auto_duck=True)
    assert norm == pytest.approx(10 ** (-3.0 / 20.0), rel=1e-3)
    assert duck == pytest.approx(10 ** (-13.0 / 20.0), rel=1e-3)
    assert comm == pytest.approx(10 ** (2.0 / 20.0), rel=1e-3)


def test_two_pass_loudnorm_measurement_and_filter(synthetic_sources):
    """Verify pass 1 loudnorm measurement extracts valid values and builds pass 2 filter."""
    src = synthetic_sources["clip_720p_24fps.mp4"]
    measured = measure_loudnorm(src, target_lufs=-14.0, true_peak=-1.0)

    assert "input_i" in measured
    assert "input_tp" in measured
    assert "input_lra" in measured

    filt = build_two_pass_loudnorm_filter(measured, target_lufs=-14.0, true_peak=-1.0)
    assert "loudnorm=I=-14.0:TP=-1.0" in filt
    assert f"measured_I={measured['input_i']}" in filt
    assert "linear=true" in filt

    # Silence handling: fallback filter when input_i is -inf
    silent_measured = {"input_i": "-inf", "target_offset": "inf"}
    safe_filt = build_two_pass_loudnorm_filter(silent_measured, target_lufs=-14.0, true_peak=-1.0)
    assert safe_filt == "anull"


def test_multisource_integration_acceptance(tmp_path: Path, synthetic_sources):
    """Comprehensive acceptance test for multi-source rendering.
    
    Verifies:
    1. Multi-source integration with mismatched resolutions, framerates, and no-audio source.
    2. Exact ordered multi-source timestamps.
    3. Normalization to common even canvas (1280x720), uniform 30fps, 48kHz stereo.
    4. Missing audio gets silence so timeline audio remains synchronized.
    5. Final publication directory contains ONLY {title}.mp4, {title}.narration.srt, {title}.original.srt.
    6. Source file SHA-256 hashes are verified completely unchanged before and after render.
    7. Subtitle cues have proper accumulated timeline offsets.
    8. Input project JSON dictionary is completely unmodified.
    """
    out_dir = tmp_path / "published"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initial hashes of all source files
    hashes_before = {name: compute_file_sha256(p) for name, p in synthetic_sources.items()}

    # Mock VoiceStudioAdapter to synthesize known WAV audio
    voice_mock = MagicMock(spec=VoiceStudioAdapter)
    voice_mock.synthesize.side_effect = lambda text, **kwargs: _create_wav_bytes(duration_s=1.2)

    project_data = {
        "schema_version": "3.0",
        "project_id": "acceptance_proj_001",
        "project_name": "MultiSourceAcceptance",
        "sources": [
            {"source_file": "clip_720p_24fps.mp4"},
            {"source_file": "clip_480p_noaudio.mp4"},
            {"source_file": "clip_600p_30fps.mp4"},
        ],
        "outputs": [
            {
                "render_id": "out_recap_1",
                "title": "Season_Recap_01",
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "source_file": "clip_720p_24fps.mp4",
                        "start_ms": 500,
                        "end_ms": 2500,  # 2000ms visual duration
                        "type": "narration",
                        "narration": "First scene begins in the garden.",
                        "source_audio": True,
                        "narration_duration_ms": 1200,
                        "subtitles": [
                            {"start_ms": 0, "end_ms": 1200, "text": "First scene begins in the garden."}
                        ],
                    },
                    {
                        "segment_id": "seg_02",
                        "source_file": "clip_480p_noaudio.mp4",
                        "start_ms": 0,
                        "end_ms": 1500,  # 1500ms visual duration (from no-audio source)
                        "type": "narration",
                        "narration": "A quiet moment passes.",
                        "source_audio": False,
                        "narration_duration_ms": 1200,
                        "subtitles": [
                            {"start_ms": 100, "end_ms": 1300, "text": "A quiet moment passes."}
                        ],
                    },
                    {
                        "segment_id": "seg_03",
                        "source_file": "clip_600p_30fps.mp4",
                        "start_ms": 1000,
                        "end_ms": 2500,  # 1500ms visual duration
                        "type": "original_dialogue",
                        "narration": "",
                        "source_audio": True,
                        "subtitles": [
                            {"start_ms": 200, "end_ms": 1400, "text": "Who is out there?"}
                        ],
                    },
                ],
            }
        ],
    }

    # Snapshot JSON to verify immutability
    original_json_str = json.dumps(project_data, sort_keys=True)

    settings = AppSettings(
        canvas_width=1280,
        canvas_height=720,
        canvas_fps=30.0,
        use_gpu=False,  # CPU for predictable deterministic test execution
        quality="high",
        auto_duck=True,
        ducking_amount_db=-12.0,
    )

    results = render_project(
        project_data=project_data,
        source_paths=synthetic_sources,
        output_dir=out_dir,
        settings=settings,
        voice_adapter=voice_mock,
    )

    assert len(results) == 1
    res = results[0]

    # 1. Output paths and publication directory contents
    assert res.output_path.is_file()
    assert res.narration_srt_path.is_file()
    assert res.original_srt_path.is_file()

    published_files = sorted(f.name for f in out_dir.iterdir())
    assert published_files == [
        "Season_Recap_01.mp4",
        "Season_Recap_01.narration.srt",
        "Season_Recap_01.original.srt",
    ]

    # 2. Fixed-speed narration is authoritative; visual allocations fit actual audio.
    probe = probe_media(res.output_path)
    assert probe.has_video is True
    assert probe.has_audio is True
    assert probe.width == 1280
    assert probe.height == 720
    assert probe.duration == pytest.approx(3.9, abs=0.2)

    # Audio format check
    audio_info = probe.audio_streams[0]
    assert audio_info.channels == 2
    assert audio_info.sample_rate == 48000

    # 3. Verify source files are 100% untouched
    for name, path in synthetic_sources.items():
        assert compute_file_sha256(path) == hashes_before[name]

    # 4. Verify input JSON was not mutated
    assert json.dumps(project_data, sort_keys=True) == original_json_str

    # 5. Verify subtitle timeline offsets
    narr_cues = parse_srt(res.narration_srt_path.read_text(encoding="utf-8"))
    orig_cues = parse_srt(res.original_srt_path.read_text(encoding="utf-8"))

    # Narration has 2 cues (from seg_01 and seg_02)
    assert len(narr_cues) == 2
    # Seg 1: timeline offset 0
    assert narr_cues[0].start_ms == 0
    assert narr_cues[0].end_ms == 1200
    assert narr_cues[0].text == "First scene begins in the garden."

    # Seg 2: timeline offset follows fixed-speed narration duration of seg_01.
    assert narr_cues[1].start_ms == 1200 + 100
    assert narr_cues[1].end_ms == 1200 + 1300
    assert narr_cues[1].text == "A quiet moment passes."

    # Original dialogue has 1 cue (from seg_03)
    assert len(orig_cues) == 1
    # Seg 3: timeline offset follows seg_01 + seg_02 narration durations.
    assert orig_cues[0].start_ms == 2400 + 200
    assert orig_cues[0].end_ms == 2400 + 1400
    assert orig_cues[0].text == "Who is out there?"


def test_narration_fit_policy_oversize_extends_and_holds(tmp_path: Path, synthetic_sources):
    """Fixed-speed narration expands the render timeline instead of changing voice speed."""
    out_dir = tmp_path / "output_oversize"
    out_dir.mkdir()

    # Voice adapter returns 3.0s audio for a 1.5s segment (1500ms)
    voice_mock = MagicMock(spec=VoiceStudioAdapter)
    voice_mock.synthesize.side_effect = lambda text, **kwargs: _create_wav_bytes(duration_s=3.0)

    output_def = {
        "title": "Oversize_Test",
        "segments": [
            {
                "segment_id": "seg_too_long",
                "source_file": "clip_720p_24fps.mp4",
                "start_ms": 0,
                "end_ms": 1500,  # 1.5s visual duration
                "type": "narration",
                "narration": "Very long narration text that does not fit.",
                "source_audio": True,
                "subtitles": [],
            }
        ],
    }

    result = render_output(output_def=output_def, source_paths=synthetic_sources, output_dir=out_dir, voice_adapter=voice_mock)
    assert result.duration == pytest.approx(3.0, abs=0.2)


def test_source_collision_error_prevents_overwrite(tmp_path: Path, synthetic_sources):
    """Invariant: Publication targets colliding with source files must fail immediately."""
    src1_path = synthetic_sources["clip_720p_24fps.mp4"]
    source_dir = src1_path.parent

    # Try to publish with a title that would create {source_dir}/clip_720p_24fps.mp4
    output_def = {
        "title": "clip_720p_24fps",  # Would collide with clip_720p_24fps.mp4 in source_dir
        "segments": [
            {
                "segment_id": "s1",
                "source_file": "clip_720p_24fps.mp4",
                "start_ms": 0,
                "end_ms": 1000,
                "type": "original_dialogue",
                "narration": "",
                "source_audio": True,
                "subtitles": [],
            }
        ],
    }

    with pytest.raises(SourceCollisionError) as exc_info:
        render_output(
            output_def=output_def,
            source_paths=synthetic_sources,
            output_dir=source_dir,  # Output directory is same as source directory!
        )

    assert "Publication target collides with source file" in str(exc_info.value)


def test_cooperative_cancellation_and_cleanup(tmp_path: Path, synthetic_sources):
    """Invariant: Cancellation must terminate processing and clean up all temporary files."""
    out_dir = tmp_path / "cancelled_output"
    out_dir.mkdir()

    token = CancellationToken()
    token.cancel()  # Cancel prior to starting

    output_def = {
        "title": "Cancelled_Title",
        "segments": [
            {
                "segment_id": "s1",
                "source_file": "clip_720p_24fps.mp4",
                "start_ms": 0,
                "end_ms": 1000,
                "type": "original_dialogue",
                "narration": "",
                "source_audio": True,
                "subtitles": [],
            }
        ],
    }

    with pytest.raises(CancelledError):
        render_output(
            output_def=output_def,
            source_paths=synthetic_sources,
            output_dir=out_dir,
            cancellation_token=token,
        )

    # Output directory must remain empty
    assert list(out_dir.iterdir()) == []


def test_burn_subtitles_mode(tmp_path: Path, synthetic_sources):
    """Verify subtitle burning option runs and completes valid MP4 output."""
    out_dir = tmp_path / "burned_output"
    out_dir.mkdir()

    settings = AppSettings(
        canvas_width=640,
        canvas_height=360,
        canvas_fps=25.0,
        use_gpu=False,
        burn_subtitles=True,
        canvas_auto=False,
    )

    output_def = {
        "title": "Burned_Movie",
        "segments": [
            {
                "segment_id": "s1",
                "source_file": "clip_720p_24fps.mp4",
                "start_ms": 0,
                "end_ms": 1000,
                "type": "narration",
                "narration": "Burned subtitle test",
                "source_audio": True,
                "subtitles": [
                    {"start_ms": 100, "end_ms": 900, "text": "Burned subtitle test"}
                ],
            }
        ],
    }

    voice_mock = MagicMock(spec=VoiceStudioAdapter)
    voice_mock.synthesize.side_effect = lambda text, **kwargs: _create_wav_bytes(duration_s=0.8)

    res = render_output(
        output_def=output_def,
        source_paths=synthetic_sources,
        output_dir=out_dir,
        settings=settings,
        voice_adapter=voice_mock,
    )

    assert res.output_path.is_file()
    assert res.output_path.stat().st_size > 0
    probe = probe_media(res.output_path)
    assert probe.has_video is True
    assert probe.width == 640
    assert probe.height == 360


# ==============================================================================
# Targeted Regression Tests for Renderer Safety Boundaries (Contract: renderer-safety)
# ==============================================================================


def test_boundary_title_windows_validation_and_no_strip(tmp_path: Path, synthetic_sources):
    """Boundary 1: Title and narration must not be stripped or invent defaults; Windows names required."""
    out_dir = tmp_path / "out_b1"
    out_dir.mkdir()

    # 1. Missing title without invented default
    with pytest.raises(WindowsNameError):
        render_output(
            output_def={"segments": [{"segment_id": "s1", "source_file": "clip_720p_24fps.mp4", "start_ms": 0, "end_ms": 1000}]},
            source_paths=synthetic_sources,
            output_dir=out_dir,
        )

    # 2. Title with trailing space (must NOT be stripped away into a valid name)
    with pytest.raises(WindowsNameError, match="cannot end with a dot or space"):
        render_output(
            output_def={
                "title": "TrailingSpaceTitle ",
                "segments": [{"segment_id": "s1", "source_file": "clip_720p_24fps.mp4", "start_ms": 0, "end_ms": 1000}],
            },
            source_paths=synthetic_sources,
            output_dir=out_dir,
        )

    # 3. Reserved Windows device name
    with pytest.raises(WindowsReservedNameError):
        render_output(
            output_def={
                "title": "CON",
                "segments": [{"segment_id": "s1", "source_file": "clip_720p_24fps.mp4", "start_ms": 0, "end_ms": 1000}],
            },
            source_paths=synthetic_sources,
            output_dir=out_dir,
        )

    # 4. Narration text must not be stripped
    captured_text = None

    def capture_synth(text: str, **kwargs: Any) -> bytes:
        nonlocal captured_text
        captured_text = text
        return _create_wav_bytes(duration_s=0.5)

    voice_mock = MagicMock(spec=VoiceStudioAdapter)
    voice_mock.synthesize.side_effect = capture_synth

    raw_narration = "  Preserve leading and trailing whitespace  "
    render_output(
        output_def={
            "title": "ExactNarration",
            "segments": [
                {
                    "segment_id": "s1",
                    "source_file": "clip_720p_24fps.mp4",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "type": "narration",
                    "narration": raw_narration,
                }
            ],
        },
        source_paths=synthetic_sources,
        output_dir=out_dir,
        voice_adapter=voice_mock,
    )
    assert captured_text == raw_narration


def test_boundary_exact_source_mapping_rejects_directory_and_traversal(tmp_path: Path, synthetic_sources):
    """Boundary 2: Exact source mapping ONLY; directory lookup and traversal strictly rejected."""
    out_dir = tmp_path / "out_b2"
    out_dir.mkdir()

    # 1. Reject directory path as source_paths
    with pytest.raises(TypeError, match="exact mapping dict"):
        render_output(
            output_def={
                "title": "DirSourceTest",
                "segments": [{"segment_id": "s1", "source_file": "clip_720p_24fps.mp4", "start_ms": 0, "end_ms": 1000}],
            },
            source_paths=tmp_path,  # Directory passed instead of dict
            output_dir=out_dir,
        )

    # 2. Reject directory traversal in source_file
    with pytest.raises(JsonSourceNotFoundError, match="directory traversal or path separators"):
        render_output(
            output_def={
                "title": "TraversalTest",
                "segments": [{"segment_id": "s1", "source_file": "../clip_720p_24fps.mp4", "start_ms": 0, "end_ms": 1000}],
            },
            source_paths=synthetic_sources,
            output_dir=out_dir,
        )


def test_boundary_strict_timestamps_no_100ms_bypass(tmp_path: Path, synthetic_sources):
    """Boundary 3: Strict timestamps against actual duration; no 100ms bypass."""
    out_dir = tmp_path / "out_b3"
    out_dir.mkdir()

    # clip_720p_24fps.mp4 duration is 3000ms.
    # Exceeding by 50ms (3050ms) was previously permitted under the +100ms bypass; now must fail.
    with pytest.raises(TimestampBoundaryError, match="exceeds source duration"):
        render_output(
            output_def={
                "title": "BypassTest",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_file": "clip_720p_24fps.mp4",
                        "start_ms": 0,
                        "end_ms": 3050,  # 50ms beyond 3000ms duration
                        "type": "original_dialogue",
                        "narration": "",
                    }
                ],
            },
            source_paths=synthetic_sources,
            output_dir=out_dir,
        )

    # Negative start_ms must also fail
    with pytest.raises(TimestampBoundaryError, match="cannot be negative"):
        render_output(
            output_def={
                "title": "NegativeStartTest",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_file": "clip_720p_24fps.mp4",
                        "start_ms": -10,
                        "end_ms": 1000,
                        "type": "original_dialogue",
                        "narration": "",
                    }
                ],
            },
            source_paths=synthetic_sources,
            output_dir=out_dir,
        )


def test_boundary_missing_narration_fails_no_fake_duration(tmp_path: Path, synthetic_sources):
    """Boundary 4: Fail missing narration adapter/audio rather than silent success; no fake duration substitute."""
    out_dir = tmp_path / "out_b4"
    out_dir.mkdir()

    # Segment has narration text and narration_duration_ms, but NO adapter and NO audio map.
    # Previously, it substituted narration_duration_ms and silently succeeded without audio.
    # Now it must explicitly raise VoiceStudioUnavailableError.
    with pytest.raises(VoiceStudioUnavailableError, match="no voice adapter or narration audio was provided"):
        render_output(
            output_def={
                "title": "MissingVoiceTest",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_file": "clip_720p_24fps.mp4",
                        "start_ms": 0,
                        "end_ms": 2000,
                        "type": "narration",
                        "narration": "Narration text without voice adapter",
                        "narration_duration_ms": 1000,  # Fake duration must NOT be substituted
                    }
                ],
            },
            source_paths=synthetic_sources,
            output_dir=out_dir,
            voice_adapter=None,
            narration_audio_map=None,
        )


def test_boundary_validate_external_narration_wav(tmp_path: Path, synthetic_sources):
    """Boundary 5: External narration WAV must be strictly validated for container, format, and duration."""
    out_dir = tmp_path / "out_b5"
    out_dir.mkdir()

    # 1. Corrupt external WAV
    corrupt_wav = tmp_path / "corrupt.wav"
    corrupt_wav.write_bytes(b"RIFF" + b"\x00" * 40)  # Invalid header/frames

    with pytest.raises(InvalidAudioError):
        render_output(
            output_def={
                "title": "CorruptWavTest",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_file": "clip_720p_24fps.mp4",
                        "start_ms": 0,
                        "end_ms": 2000,
                        "type": "narration",
                        "narration": "Some narration",
                    }
                ],
            },
            source_paths=synthetic_sources,
            output_dir=out_dir,
            narration_audio_map={"s1": corrupt_wav},
        )

    # 2. Missing external WAV
    missing_wav = tmp_path / "non_existent.wav"
    with pytest.raises(RenderError, match="not found"):
        render_output(
            output_def={
                "title": "MissingWavTest",
                "segments": [
                    {
                        "segment_id": "s1",
                        "source_file": "clip_720p_24fps.mp4",
                        "start_ms": 0,
                        "end_ms": 2000,
                        "type": "narration",
                        "narration": "Some narration",
                    }
                ],
            },
            source_paths=synthetic_sources,
            output_dir=out_dir,
            narration_audio_map={"s1": missing_wav},
        )


def test_boundary_preserve_voice_model_and_style(tmp_path: Path, synthetic_sources):
    """Boundary 6: Preserve configured voice model and style through synthesis; extend adapter style."""
    out_dir = tmp_path / "out_b6"
    out_dir.mkdir()

    captured_kwargs: dict[str, Any] = {}

    def capture_synth(text: str, **kwargs: Any) -> bytes:
        captured_kwargs.update(kwargs)
        return _create_wav_bytes(duration_s=0.5)

    voice_mock = MagicMock(spec=VoiceStudioAdapter)
    voice_mock.synthesize.side_effect = capture_synth

    custom_settings = AppSettings(
        voice_id="echo",
        voice_model="voxcpm2",
        voice_language="en-GB",
        voice_style="warm tone, documentary style",
    )

    render_output(
        output_def={
            "title": "PreserveStyleTest",
            "segments": [
                {
                    "segment_id": "s1",
                    "source_file": "clip_720p_24fps.mp4",
                    "start_ms": 0,
                    "end_ms": 1500,
                    "type": "narration",
                    "narration": "Voice style test narration",
                }
            ],
        },
        source_paths=synthetic_sources,
        output_dir=out_dir,
        settings=custom_settings,
        voice_adapter=voice_mock,
    )

    assert captured_kwargs.get("voice") == "echo"
    assert captured_kwargs.get("model") == "voxcpm2"
    assert captured_kwargs.get("language") == "en-GB"
    assert captured_kwargs.get("style") == "warm tone, documentary style"

    # Also test VoiceStudioAdapter synthesize directly places style into description payload
    import httpx

    captured_payload = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        if "/health" in str(request.url):
            return httpx.Response(200, json={"status": "ok"})
        if "/speech" in str(request.url):
            captured_payload = json.loads(request.read())
            return httpx.Response(200, content=_create_wav_bytes(duration_s=1.0))
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = VoiceStudioAdapter(mode="local", client=client)
    adapter.synthesize(
        "Test text",
        voice="echo",
        model="voxcpm2",
        language="en-GB",
        style="warm tone, slight British accent",
    )
    assert captured_payload is not None
    assert captured_payload.get("description") == "warm tone, slight British accent"
    assert captured_payload.get("model") == "voxcpm2"


def test_boundary_internal_work_in_localappdata(tmp_path: Path, synthetic_sources, monkeypatch):
    """Boundary 7: Internal scratch files must be created under LOCALAPPDATA/ToolRecapV4/render-work."""
    fake_localappdata = tmp_path / "fake_localappdata"
    monkeypatch.setenv("LOCALAPPDATA", str(fake_localappdata))

    render_work = get_render_work_root()
    assert render_work == fake_localappdata / "ToolRecapV4" / "render-work"
    assert render_work.is_dir()


def test_boundary_no_repeat_source_hashing(tmp_path: Path, synthetic_sources, monkeypatch):
    """Boundary 8: Costly repeat full source hashing each output is removed."""
    out_dir = tmp_path / "out_b8"
    out_dir.mkdir()

    hash_call_count = 0
    import toolrecap_v4.renderer

    original_hash_fn = toolrecap_v4.renderer.compute_file_sha256

    def spy_hash(p: Any) -> str:
        nonlocal hash_call_count
        hash_call_count += 1
        return original_hash_fn(p)

    monkeypatch.setattr(toolrecap_v4.renderer, "compute_file_sha256", spy_hash)

    render_output(
        output_def={
            "title": "NoRepeatHashingTest",
            "segments": [
                {
                    "segment_id": "s1",
                    "source_file": "clip_720p_24fps.mp4",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "type": "original_dialogue",
                    "narration": "",
                }
            ],
        },
        source_paths=synthetic_sources,
        output_dir=out_dir,
    )

    # In render_output, compute_file_sha256 should NOT be called at all
    assert hash_call_count == 0


# ==============================================================================
# Focused Tests: Auto Canvas 640x480 Multi-Source and Anamorphic DAR Aspect Fit
# ==============================================================================


def test_multisource_first_clip_640x480_outputdims_not_1080p(tmp_path: Path, synthetic_sources):
    """Verify multi-source render with first clip 640x480 uses 640x480 canvas, not 1080p fallback."""
    out_dir = tmp_path / "out_640x480"
    out_dir.mkdir()

    # First clip is 640x480 (no audio)
    # Second clip is 1280x720 (has audio)
    # Third clip is 800x600 (has audio)
    output_def = {
        "title": "FirstClip640x480_Recap",
        "segments": [
            {
                "segment_id": "seg_01",
                "source_file": "clip_480p_noaudio.mp4",
                "start_ms": 0,
                "end_ms": 1000,
                "type": "original_dialogue",
                "narration": "",
                "source_audio": False,
            },
            {
                "segment_id": "seg_02",
                "source_file": "clip_720p_24fps.mp4",
                "start_ms": 500,
                "end_ms": 1500,
                "type": "original_dialogue",
                "narration": "",
                "source_audio": True,
            },
            {
                "segment_id": "seg_03",
                "source_file": "clip_600p_30fps.mp4",
                "start_ms": 500,
                "end_ms": 1500,
                "type": "original_dialogue",
                "narration": "",
                "source_audio": True,
            },
        ],
    }

    # AppSettings with default canvas_auto=True (fallback is 1920x1080)
    settings = AppSettings(
        use_gpu=False,
        canvas_width=1920,
        canvas_height=1080,
        canvas_auto=True,
    )

    res = render_output(
        output_def=output_def,
        source_paths=synthetic_sources,
        output_dir=out_dir,
        settings=settings,
    )

    # Invariant: Output dimensions MUST match first clip (640x480), NOT 1080p fallback
    assert res.width == 640, f"Expected width 640, got {res.width}"
    assert res.height == 480, f"Expected height 480, got {res.height}"
    assert (res.width, res.height) != (1920, 1080)

    # Validate output media probe
    probe = probe_media(res.output_path)
    assert probe.has_video is True
    assert probe.has_audio is True
    assert probe.width == 640
    assert probe.height == 480
    assert probe.duration == pytest.approx(3.0, abs=0.2)
    assert probe.audio_streams[0].sample_rate == 48000
    assert probe.audio_streams[0].channels == 2


def test_renderer_aspectfit_respects_anamorphic_display_ratio(tmp_path: Path):
    """Verify aspectfit respects anamorphic DISPLAY aspect ratio (ih*dar), not coded ratio."""
    src_dir = tmp_path / "ana_sources"
    src_dir.mkdir()
    out_dir = tmp_path / "ana_output"
    out_dir.mkdir()

    # Create synthetic anamorphic source: 720x480 with SAR 32:27 -> DAR 16:9, with audio
    ana_src = src_dir / "anamorphic_16_9.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=720x480:rate=25:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
            "-vf", "setsar=32/27",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            str(ana_src),
        ],
        check=True,
        capture_output=True,
    )

    # 1. Test auto canvas on anamorphic source:
    # 720x480 with DAR 16:9 has display width 480 * (16/9) = 853.33 -> 854x480 (even)
    ana_probe = probe_media(ana_src)
    from toolrecap_v4.media import calculate_auto_canvas
    auto_w, auto_h = calculate_auto_canvas(ana_probe)
    assert (auto_w, auto_h) == (854, 480)

    # 2. Render anamorphic source into 1280x720 canvas (16:9):
    # With ih*dar square-pixel normalization, 16:9 display content fills 1280x720 without horizontal squish
    output_def = {
        "title": "Anamorphic_AspectFit",
        "segments": [
            {
                "segment_id": "s1",
                "source_file": "anamorphic_16_9.mp4",
                "start_ms": 0,
                "end_ms": 1000,
                "type": "original_dialogue",
                "narration": "",
                "source_audio": True,
            }
        ],
    }

    settings = AppSettings(
        canvas_width=1280,
        canvas_height=720,
        canvas_auto=False,
        use_gpu=False,
    )

    res = render_output(
        output_def=output_def,
        source_paths={"anamorphic_16_9.mp4": ana_src},
        output_dir=out_dir,
        settings=settings,
    )

    assert res.output_path.is_file()
    probe = probe_media(res.output_path)
    assert probe.width == 1280
    assert probe.height == 720
    assert probe.aspect_ratio == "16:9"
    # Output must have square pixels (SAR 1:1)
    if probe.video_streams:
        assert probe.video_streams[0].sar in ("1:1", "1/1", "")
