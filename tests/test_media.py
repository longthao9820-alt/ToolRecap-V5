"""Targeted tests for media primitives: process runner, probe, audio selector, and GPU detection."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time
import pytest

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError
from toolrecap_v4.media import (
    AudioSelectionResult,
    AudioStreamInfo,
    BinaryNotFoundError,
    CommandResult,
    EncoderStatus,
    MediaError,
    MediaProbeResult,
    ProbeError,
    SubprocessError,
    SubprocessTimeoutError,
    VideoStreamInfo,
    assess_source_timeline,
    inspect_frame_timing,
    calculate_auto_canvas,
    detect_gpu_encoder,
    find_binary,
    get_video_encode_args,
    nvidia_decode_args,
    nvidia_scale_pad_filter,
    probe_nvidia_source_pipeline,
    is_commentary_or_descriptive,
    is_english_language,
    is_main_audio,
    probe_duration,
    probe_encoder_usable,
    probe_media,
    resolve_sources_audio,
    run_command,
    select_audio_for_source,
    select_audio_stream,
)


def test_source_timeline_preflight_detects_vfr_and_nonzero_start():
    stream = VideoStreamInfo(
        0, "h264", 1920, 1080, 29.97, "30000/1001", 60.0, "16:9",
        avg_frame_rate="30000/1001", real_frame_rate="60/1", time_base="1/90000", start_time=0.125,
    )
    probe = MediaProbeResult(
        Path("episode.mkv"), 60.0, "matroska", (stream,), (),
        has_video=True, start_time=0.125,
    )
    assessment = assess_source_timeline(probe)
    assert assessment.status == "needs_frame_inspection"
    assert assessment.requires_cfr_output is True
    assert assessment.recommended_fps == "30000/1001"
    assert len(assessment.reasons) == 2


def test_source_timeline_preflight_accepts_stable_cfr():
    stream = VideoStreamInfo(
        0, "h264", 1920, 1080, 23.976, "24000/1001", 60.0, "16:9",
        avg_frame_rate="24000/1001", real_frame_rate="24000/1001", time_base="1/24000", start_time=0.0,
    )
    assessment = assess_source_timeline(MediaProbeResult(
        Path("episode.mkv"), 60.0, "matroska", (stream,), (), has_video=True,
    ))
    assert assessment.status == "compatible"
    assert assessment.requires_cfr_output is False


def test_frame_timing_sampling_detects_irregular_pts(monkeypatch, tmp_path):
    source = tmp_path / "episode.mp4"; source.write_bytes(b"fixture")
    monkeypatch.setattr("toolrecap_v4.media.find_binary", lambda *_args: Path("ffprobe"))
    monkeypatch.setattr("toolrecap_v4.media.run_command", lambda *_args, **_kwargs:
                        CommandResult(0, "0.000\n0.040\n0.080\n0.160\n0.200\n", ""))
    sample = inspect_frame_timing(source, 60_000)
    assert sample["irregular_timing_observed"] is True
    assert len(sample["sampled_windows"]) == 3


# ============================================================================
# 1. Subprocess Runner Tests: Safe Args, Timeout, Cancellation, Child Cleanup
# ============================================================================


def test_subprocess_runner_safe_args_rejection() -> None:
    """Verify runner rejects string or bytes arguments and empty sequences."""
    with pytest.raises(TypeError, match="sequence of strings"):
        run_command("ffmpeg -version")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="sequence of strings"):
        run_command(b"ffmpeg -version")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot be empty"):
        run_command([])


def test_subprocess_runner_basic_execution() -> None:
    """Verify normal command execution returns CommandResult with exit_code, stdout, stderr."""
    result = run_command(["python", "-c", "import sys; sys.stdout.write('hello'); sys.stderr.write('world')"])
    assert result.exit_code == 0
    assert result.stdout == "hello"
    assert result.stderr == "world"


def test_subprocess_runner_check_flag() -> None:
    """Verify check=True raises SubprocessError on non-zero exit."""
    with pytest.raises(SubprocessError, match="exit code 42"):
        run_command(["python", "-c", "import sys; sys.exit(42)"], check=True)


def test_subprocess_runner_timeout_terminates_child() -> None:
    """Verify finite timeout triggers SubprocessTimeoutError and cleans up child."""
    start_time = time.time()
    with pytest.raises(SubprocessTimeoutError, match="timed out after 0.5s"):
        run_command(["python", "-c", "import time; time.sleep(10)"], timeout=0.5)
    elapsed = time.time() - start_time
    assert elapsed < 5.0, "Subprocess runner did not enforce timeout promptly"


def test_subprocess_runner_cancellation_before_start() -> None:
    """Verify pre-cancelled token raises CancelledError without launching process."""
    token = CancellationToken()
    token.cancel()

    with pytest.raises(CancelledError, match="Operation was cancelled"):
        run_command(["python", "-c", "import time; time.sleep(5)"], cancellation_token=token)


def test_subprocess_runner_cancellation_during_execution() -> None:
    """Verify token cancellation during execution terminates child and raises CancelledError."""
    token = CancellationToken()

    def _cancel_after_delay() -> None:
        time.sleep(0.3)
        token.cancel()

    thread = threading.Thread(target=_cancel_after_delay)
    thread.daemon = True
    thread.start()

    start_time = time.time()
    with pytest.raises(CancelledError, match="Subprocess cancelled"):
        run_command(
            ["python", "-c", "import time; time.sleep(10)"],
            timeout=10.0,
            cancellation_token=token,
        )
    elapsed = time.time() - start_time
    assert elapsed < 3.0, "Subprocess was not cancelled and terminated promptly"


# ============================================================================
# 2. Binary Discovery Tests
# ============================================================================


def test_find_binary_system_path() -> None:
    """Verify system PATH discovers ffmpeg and ffprobe."""
    ffmpeg_bin = find_binary("ffmpeg")
    assert ffmpeg_bin.is_file()
    assert "ffmpeg" in ffmpeg_bin.name.lower()

    ffprobe_bin = find_binary("ffprobe")
    assert ffprobe_bin.is_file()
    assert "ffprobe" in ffprobe_bin.name.lower()


def test_find_binary_custom_path(tmp_path: Path) -> None:
    """Verify custom path parameter overrides discovery."""
    dummy_bin = tmp_path / "custom_ffmpeg.exe"
    dummy_bin.write_bytes(b"dummy")

    found = find_binary("ffmpeg", custom_path=dummy_bin)
    assert found == dummy_bin.resolve()


def test_find_binary_not_found() -> None:
    """Verify non-existent binary raises BinaryNotFoundError."""
    with pytest.raises(BinaryNotFoundError, match="Required binary 'nonexistent_binary_xyz' not found"):
        find_binary("nonexistent_binary_xyz")


# ============================================================================
# 3. Media Probing Tests (Actual Generated Media & Independent Probing)
# ============================================================================


@pytest.fixture(scope="module")
def generated_test_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a real 1-second test video with video (320x240) and audio (sine wave)."""
    tmp_dir = tmp_path_factory.mktemp("media_test")
    video_path = tmp_dir / "test_sample.mp4"

    ffmpeg = find_binary("ffmpeg")
    cmd = [
        str(ffmpeg),
        "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1.0:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1.0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "44100",
        "-ac", "2",
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:a:0", "title=Main Audio",
        str(video_path),
    ]
    res = run_command(cmd, timeout=30.0)
    assert res.exit_code == 0, f"FFmpeg sample generation failed: {res.stderr}"
    assert video_path.is_file()
    return video_path


def test_actual_media_probe(generated_test_video: Path) -> None:
    """Verify probe_media extracts valid metadata from real generated media file."""
    result = probe_media(generated_test_video)

    assert isinstance(result, MediaProbeResult)
    assert result.path == generated_test_video.resolve()
    assert 0.8 <= result.duration <= 1.2
    assert result.has_video is True
    assert result.has_audio is True
    assert result.width == 320
    assert result.height == 240
    assert result.fps == 25.0
    assert result.aspect_ratio == "4:3"

    # Streams
    assert len(result.video_streams) == 1
    assert result.video_streams[0].codec in ("h264", "libx264")
    assert result.video_streams[0].width == 320
    assert result.video_streams[0].height == 240

    assert len(result.audio_streams) == 1
    assert result.audio_streams[0].codec == "aac"
    assert result.audio_streams[0].channels == 2
    assert result.audio_streams[0].sample_rate == 44100
    assert result.audio_streams[0].language == "eng"
    assert result.audio_streams[0].title == "Main Audio"

    # Dictionary access compatibility
    assert result["duration"] == result.duration
    assert result["width"] == 320
    assert result.get("height") == 240


def test_probe_duration(generated_test_video: Path) -> None:
    """Verify probe_duration extracts format duration accurately."""
    dur = probe_duration(generated_test_video)
    assert 0.8 <= dur <= 1.2


def test_probe_media_missing_file() -> None:
    """Verify probe_media raises MediaError for missing file."""
    with pytest.raises(MediaError, match="Media source not found"):
        probe_media("C:/NonExistentPath/video_123.mp4")


def test_probe_duration_missing_file() -> None:
    """Verify probe_duration raises MediaError for missing file."""
    with pytest.raises(MediaError, match="Media source not found"):
        probe_duration("C:/NonExistentPath/video_123.mp4")


# ============================================================================
# 4. Audio Selector Invariant Tests (Scenarios A - D & Per-Source Resolution)
# ============================================================================


def test_audio_selector_scenario_a_english_main_vs_commentary() -> None:
    """Scenario A: English main excluding commentary/descriptive.
    
    When an English Main track and an English Commentary track are present,
    English Main must be selected and commentary must be excluded.
    """
    streams = [
        AudioStreamInfo(
            index=1,
            codec="aac",
            channels=2,
            sample_rate=48000,
            language="eng",
            title="Director's Commentary",
            disposition={"default": 1, "commentary": 1},
        ),
        AudioStreamInfo(
            index=2,
            codec="aac",
            channels=6,
            sample_rate=48000,
            language="eng",
            title="Main Audio 5.1",
            disposition={"default": 0, "commentary": 0},
        ),
    ]

    result = select_audio_stream(streams)
    assert result.selected_stream is not None
    assert result.selected_index == 2
    assert result.selected_stream.title == "Main Audio 5.1"
    assert result.reason == "english_main"
    assert result.warning is None


def test_audio_selector_scenario_b_english_default() -> None:
    """Scenario B: English normal/default before foreign language.
    
    When an English Default track is present alongside other languages,
    English Default is selected without warning.
    """
    streams = [
        AudioStreamInfo(
            index=1,
            codec="ac3",
            channels=2,
            sample_rate=48000,
            language="fra",
            title="French",
            disposition={"default": 0},
        ),
        AudioStreamInfo(
            index=2,
            codec="aac",
            channels=2,
            sample_rate=48000,
            language="en-US",
            title="English Stereo",
            disposition={"default": 1},
        ),
    ]

    result = select_audio_stream(streams)
    assert result.selected_index == 2
    assert result.selected_stream is not None
    assert result.selected_stream.language == "en-US"
    assert result.reason == "english_default"
    assert result.warning is None


def test_audio_selector_scenario_c_english_normal_vs_foreign_default() -> None:
    """Scenario C: English normal track preferred over foreign default track.
    
    Even if a foreign language is marked default=1, a normal English track
    must take precedence.
    """
    streams = [
        AudioStreamInfo(
            index=1,
            codec="aac",
            channels=2,
            sample_rate=48000,
            language="spa",
            title="Spanish",
            disposition={"default": 1},
        ),
        AudioStreamInfo(
            index=2,
            codec="aac",
            channels=2,
            sample_rate=48000,
            language="en-GB",
            title="Stereo Track",
            disposition={"default": 0},
        ),
    ]

    result = select_audio_stream(streams)
    assert result.selected_index == 2
    assert result.selected_stream is not None
    assert result.selected_stream.language == "en-GB"
    assert result.reason == "english_normal"
    assert result.warning is None


def test_audio_selector_prefers_english_among_many_foreign_tracks() -> None:
    streams = [
        AudioStreamInfo(1, "aac", 2, 48000, language="ukr", title="Ukrainian Dub", disposition={"default": 1}),
        AudioStreamInfo(2, "aac", 2, 48000, language="rus", title="Russian Dub", disposition={"default": 0}),
        AudioStreamInfo(3, "aac", 2, 48000, language="jpn", title="Japanese Dub", disposition={"default": 0}),
        AudioStreamInfo(4, "aac", 6, 48000, language="eng", title="English Original", disposition={"default": 0}),
        AudioStreamInfo(5, "aac", 2, 48000, language="eng", title="English Commentary", disposition={"commentary": 1}),
    ]

    result = select_audio_stream(streams)

    assert result.selected_index == 4
    assert result.selected_stream is not None
    assert result.selected_stream.language == "eng"
    assert result.selected_stream.title == "English Original"
    assert result.warning is None


def test_audio_selector_scenario_d_fallback_default_with_warning() -> None:
    """Scenario D: No English tracks; fall back to default track with warning.
    
    When no English tracks exist, the default foreign track is selected,
    and a clear warning is returned.
    """
    streams = [
        AudioStreamInfo(
            index=1,
            codec="aac",
            channels=2,
            sample_rate=48000,
            language="jpn",
            title="Japanese Audio",
            disposition={"default": 1},
        ),
        AudioStreamInfo(
            index=2,
            codec="aac",
            channels=2,
            sample_rate=48000,
            language="jpn",
            title="Japanese Commentary",
            disposition={"default": 0, "commentary": 1},
        ),
    ]

    result = select_audio_stream(streams)
    assert result.selected_index == 1
    assert result.selected_stream is not None
    assert result.reason == "fallback_default"
    assert result.warning is not None
    assert "No suitable English audio track found" in result.warning
    assert "falling back to default audio track" in result.warning


def test_audio_selector_scenario_d_fallback_first_with_warning() -> None:
    """Scenario D (no default flag): fall back to first audio track with warning."""
    streams = [
        AudioStreamInfo(
            index=3,
            codec="mp3",
            channels=2,
            sample_rate=44100,
            language="deu",
            title="German",
            disposition={"default": 0},
        ),
        AudioStreamInfo(
            index=4,
            codec="mp3",
            channels=2,
            sample_rate=44100,
            language="ita",
            title="Italian",
            disposition={"default": 0},
        ),
    ]

    result = select_audio_stream(streams)
    assert result.selected_index == 3
    assert result.selected_stream is not None
    assert result.reason == "fallback_first"
    assert result.warning is not None
    assert "falling back to first audio track" in result.warning


def test_audio_selector_language_equivalents() -> None:
    """Verify equivalent English codes (eng, en, en-US, en-GB, ENG, etc.) are recognized."""
    assert is_english_language("eng") is True
    assert is_english_language("en") is True
    assert is_english_language("en-US") is True
    assert is_english_language("en-GB") is True
    assert is_english_language("en_us") is True
    assert is_english_language("ENG") is True
    assert is_english_language("English") is True
    assert is_english_language("fra") is False
    assert is_english_language("jpn") is False
    assert is_english_language("") is False
    assert is_english_language(None) is False


def test_audio_selector_commentary_and_descriptive_detection() -> None:
    """Verify commentary and descriptive tracks are recognized across flags and keywords."""
    # Disposition flags
    assert is_commentary_or_descriptive(
        AudioStreamInfo(1, "aac", 2, 48000, disposition={"commentary": 1})
    ) is True
    assert is_commentary_or_descriptive(
        AudioStreamInfo(1, "aac", 2, 48000, disposition={"descriptions": 1})
    ) is True
    assert is_commentary_or_descriptive(
        AudioStreamInfo(1, "aac", 2, 48000, disposition={"visual_impaired": 1})
    ) is True

    # Title keywords
    assert is_commentary_or_descriptive(
        AudioStreamInfo(1, "aac", 2, 48000, title="Audio Description")
    ) is True
    assert is_commentary_or_descriptive(
        AudioStreamInfo(1, "aac", 2, 48000, title="Director Commentary")
    ) is True
    assert is_commentary_or_descriptive(
        AudioStreamInfo(1, "aac", 2, 48000, handler_name="DVS Track")
    ) is True

    # Clean track
    assert is_commentary_or_descriptive(
        AudioStreamInfo(1, "aac", 2, 48000, title="Main Dialogue", disposition={"default": 1})
    ) is False


def test_audio_selector_per_source_index_resolution() -> None:
    """Verify per-source audio index resolution handles multiple sources independently."""
    # Source 1: English Main at stream index 1
    probe_1 = MediaProbeResult(
        path=Path("ep01.mp4"),
        duration=100.0,
        container="mp4",
        video_streams=(VideoStreamInfo(0, "h264", 1920, 1080, 24.0, "24/1", 100.0, "16:9"),),
        audio_streams=(
            AudioStreamInfo(1, "aac", 2, 48000, language="eng", title="Main"),
            AudioStreamInfo(2, "aac", 2, 48000, language="eng", title="Commentary"),
        ),
        width=1920,
        height=1080,
        fps=24.0,
        aspect_ratio="16:9",
        has_video=True,
        has_audio=True,
    )

    # Source 2: English Default at stream index 3 (stream 1 and 2 are French/Spanish)
    probe_2 = MediaProbeResult(
        path=Path("ep02.mp4"),
        duration=120.0,
        container="mp4",
        video_streams=(VideoStreamInfo(0, "h264", 1920, 1080, 24.0, "24/1", 120.0, "16:9"),),
        audio_streams=(
            AudioStreamInfo(1, "aac", 2, 48000, language="fra", disposition={"default": 0}),
            AudioStreamInfo(2, "aac", 2, 48000, language="spa", disposition={"default": 0}),
            AudioStreamInfo(3, "aac", 2, 48000, language="en", disposition={"default": 1}),
        ),
        width=1920,
        height=1080,
        fps=24.0,
        aspect_ratio="16:9",
        has_video=True,
        has_audio=True,
    )

    # Source 3: Japanese only at stream index 1 (fallback default with warning)
    probe_3 = MediaProbeResult(
        path=Path("ep03.mp4"),
        duration=90.0,
        container="mp4",
        video_streams=(VideoStreamInfo(0, "h264", 1920, 1080, 24.0, "24/1", 90.0, "16:9"),),
        audio_streams=(
            AudioStreamInfo(1, "aac", 2, 48000, language="jpn", disposition={"default": 1}),
        ),
        width=1920,
        height=1080,
        fps=24.0,
        aspect_ratio="16:9",
        has_video=True,
        has_audio=True,
    )

    results = resolve_sources_audio([probe_1, probe_2, probe_3])
    assert len(results) == 3

    assert results[0].selected_index == 1
    assert results[0].reason == "english_main"
    assert results[0].warning is None

    assert results[1].selected_index == 3
    assert results[1].reason == "english_default"
    assert results[1].warning is None

    assert results[2].selected_index == 1
    assert results[2].reason == "fallback_default"
    assert results[2].warning is not None


# ============================================================================
# 5. GPU Usable Encoder Detection Tests (Actual Smoke & Fallback)
# ============================================================================


def test_gpu_usable_encoder_actual_smoke() -> None:
    """Verify detect_gpu_encoder runs an actual encode probe on the host machine."""
    status = detect_gpu_encoder(refresh=True)
    assert isinstance(status, EncoderStatus)
    # The machine has an RTX 3060; verify NVENC is detected and usable
    if status.available:
        assert status.encoder in ("h264_nvenc", "h264_amf", "h264_qsv")
        assert status.label in ("NVIDIA NVENC", "AMD AMF", "Intel Quick Sync")
    else:
        assert status.encoder == "libx264"
        assert status.label == "CPU"


def test_get_video_encode_args_matching_quality() -> None:
    """Verify video encode arguments generate expected flags for different qualities."""
    status = detect_gpu_encoder()

    for quality in ("standard", "high", "source"):
        if status.available:
            args, selected = get_video_encode_args(quality, use_gpu=True, encoder_status=status)
            assert "-c:v" in args
            assert selected.encoder == status.encoder
        else:
            with pytest.raises(MediaError, match="no working hardware encoder"):
                get_video_encode_args(quality, use_gpu=True, encoder_status=status)

    # Forced CPU fallback
    cpu_args, cpu_status = get_video_encode_args("high", use_gpu=False, encoder_status=status)
    assert "-c:v" in cpu_args
    assert "libx264" in cpu_args
    assert "-crf" in cpu_args
    assert cpu_status.encoder == "libx264"


def test_nvidia_selected_uses_nvenc_decode_and_never_silent_cpu_fallback() -> None:
    nvenc = EncoderStatus(True, "NVIDIA GeForce RTX 3060", "h264_nvenc", "NVIDIA NVENC")
    args, selected = get_video_encode_args("high", use_gpu=True, encoder_status=nvenc)
    assert selected.encoder == "h264_nvenc" and "h264_nvenc" in args and "libx264" not in args
    decode = nvidia_decode_args("h264", nvenc)
    assert decode == [
        "-hwaccel", "cuda", "-hwaccel_device", "0",
        "-hwaccel_output_format", "cuda", "-c:v", "h264_cuvid",
    ]
    unavailable = EncoderStatus(False, "NVIDIA GeForce RTX 3060", "libx264", "CPU", "NVENC failed")
    with pytest.raises(MediaError, match="NVENC failed"):
        get_video_encode_args("high", use_gpu=True, encoder_status=unavailable)
    graph = nvidia_scale_pad_filter(
        canvas_width=1920, canvas_height=1080, fps=30.0,
        source_duration=4.25, hold_duration=0.75, hardware_input=True,
    )
    assert "scale_cuda" in graph and "hwdownload" in graph
    assert "format=nv12:passthrough=0" in graph
    assert "pad=1920:1080" in graph and "tpad=stop_mode=clone" in graph


def test_nvidia_source_probe_uses_explicit_10bit_to_nv12_conversion(tmp_path: Path) -> None:
    source = tmp_path / "main10.mkv"; source.write_bytes(b"source identity")
    calls = []
    def runner(args, **_kwargs):
        calls.append(args)
        return CommandResult(0, "", "")
    ok, detail = probe_nvidia_source_pipeline(
        "ffmpeg", source, source_codec="hevc", pixel_format="yuv420p10le",
        canvas_width=1920, canvas_height=1080, fps=30.0, command_runner=runner,
    )
    assert ok and not detail and len(calls) == 1
    command = calls[0]
    graph = command[command.index("-vf") + 1]
    assert "hevc_cuvid" in command
    assert "format=nv12:passthrough=0" in graph


def test_encoder_usable_real_and_invalid() -> None:
    """Verify probe_encoder_usable returns True for libx264 and False for invalid encoder."""
    ffmpeg = find_binary("ffmpeg")
    assert probe_encoder_usable(ffmpeg, "libx264") is True
    assert probe_encoder_usable(ffmpeg, "invalid_nonexistent_encoder_xyz") is False


# ============================================================================
# 6. Auto Canvas Calculation Tests (DAR, SAR, Rotation, Even Dimensions)
# ============================================================================


def test_auto_canvas_dar_sar_rotation_even() -> None:
    """Verify calculate_auto_canvas respects DAR, SAR, rotation, and guarantees even dimensions."""
    dummy_path = Path("dummy.mp4")

    # 1. DAR Anamorphic: 720x480 with DAR 16:9 -> 853.33 -> 853 -> rounded up to 854x480 (even)
    v_dar = VideoStreamInfo(
        index=0,
        codec="h264",
        width=720,
        height=480,
        fps=25.0,
        fps_text="25/1",
        duration=1.0,
        aspect_ratio="3:2",
        dar="16:9",
        sar="32:27",
    )
    probe_dar = MediaProbeResult(
        path=dummy_path,
        duration=1.0,
        container="mp4",
        video_streams=(v_dar,),
        audio_streams=(),
        width=720,
        height=480,
        dar="16:9",
        sar="32:27",
        has_video=True,
    )
    w, h = calculate_auto_canvas(probe_dar)
    assert (w, h) == (854, 480)
    assert w % 2 == 0 and h % 2 == 0

    # 2. SAR Anamorphic without DAR tag: 720x480 with SAR 32:27 -> 854x480
    v_sar = VideoStreamInfo(
        index=0,
        codec="h264",
        width=720,
        height=480,
        fps=25.0,
        fps_text="25/1",
        duration=1.0,
        aspect_ratio="3:2",
        dar="",
        sar="32:27",
    )
    probe_sar = MediaProbeResult(
        path=dummy_path,
        duration=1.0,
        container="mp4",
        video_streams=(v_sar,),
        audio_streams=(),
        width=720,
        height=480,
        dar="",
        sar="32:27",
        has_video=True,
    )
    w, h = calculate_auto_canvas(probe_sar)
    assert (w, h) == (854, 480)
    assert w % 2 == 0 and h % 2 == 0

    # 3. Rotation 90 degrees: 1920x1080 rot 90 -> 1080x1920
    v_rot90 = VideoStreamInfo(
        index=0,
        codec="h264",
        width=1920,
        height=1080,
        fps=30.0,
        fps_text="30/1",
        duration=1.0,
        aspect_ratio="16:9",
        rotation=90,
    )
    probe_rot90 = MediaProbeResult(
        path=dummy_path,
        duration=1.0,
        container="mp4",
        video_streams=(v_rot90,),
        audio_streams=(),
        width=1920,
        height=1080,
        has_video=True,
    )
    w, h = calculate_auto_canvas(probe_rot90)
    assert (w, h) == (1080, 1920)
    assert w % 2 == 0 and h % 2 == 0

    # 4. Rotation 270 degrees: 1920x1080 rot 270 -> 1080x1920
    v_rot270 = VideoStreamInfo(
        index=0,
        codec="h264",
        width=1920,
        height=1080,
        fps=30.0,
        fps_text="30/1",
        duration=1.0,
        aspect_ratio="16:9",
        rotation=270,
    )
    probe_rot270 = MediaProbeResult(
        path=dummy_path,
        duration=1.0,
        container="mp4",
        video_streams=(v_rot270,),
        audio_streams=(),
        width=1920,
        height=1080,
        has_video=True,
    )
    w, h = calculate_auto_canvas(probe_rot270)
    assert (w, h) == (1080, 1920)
    assert w % 2 == 0 and h % 2 == 0

    # 5. Rotation 90 with anamorphic: 720x480 with SAR 32:27 rot 90 -> (480, 854)
    v_rot_anamorphic = VideoStreamInfo(
        index=0,
        codec="h264",
        width=720,
        height=480,
        fps=25.0,
        fps_text="25/1",
        duration=1.0,
        aspect_ratio="3:2",
        sar="32:27",
        rotation=90,
    )
    probe_rot_ana = MediaProbeResult(
        path=dummy_path,
        duration=1.0,
        container="mp4",
        video_streams=(v_rot_anamorphic,),
        audio_streams=(),
        width=720,
        height=480,
        sar="32:27",
        has_video=True,
    )
    w, h = calculate_auto_canvas(probe_rot_ana)
    assert (w, h) == (480, 854)
    assert w % 2 == 0 and h % 2 == 0

    # 6. Standard 640x480: returns (640, 480)
    v_480p = VideoStreamInfo(
        index=0,
        codec="h264",
        width=640,
        height=480,
        fps=25.0,
        fps_text="25/1",
        duration=1.0,
        aspect_ratio="4:3",
    )
    probe_480p = MediaProbeResult(
        path=dummy_path,
        duration=1.0,
        container="mp4",
        video_streams=(v_480p,),
        audio_streams=(),
        width=640,
        height=480,
        has_video=True,
    )
    w, h = calculate_auto_canvas(probe_480p)
    assert (w, h) == (640, 480)
    assert w % 2 == 0 and h % 2 == 0

    # 7. No video stream fallback with odd dimensions: rounds up to even
    probe_no_video = MediaProbeResult(
        path=dummy_path,
        duration=0.0,
        container="unknown",
        video_streams=(),
        audio_streams=(),
        has_video=False,
    )
    w, h = calculate_auto_canvas(probe_no_video, fallback_w=1919, fallback_h=1079)
    assert (w, h) == (1920, 1080)
    assert w % 2 == 0 and h % 2 == 0
