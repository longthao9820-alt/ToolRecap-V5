"""Focused tests for ToolRecap V4 core analysis, source preparation, and subtitle modules."""
from __future__ import annotations

import json
from pathlib import Path
import struct
from unittest.mock import MagicMock, patch
import wave

import pytest
from PIL import Image

from toolrecap_v4.analysis.cache import AnalysisCacheManager, safe_cache_filename
from toolrecap_v4.analysis.dependencies import (
    compute_sidecar_signature,
    compute_vobsub_signature,
    hash_file_content,
)
from toolrecap_v4.analysis.models import (
    AudioSelection,
    PreparedEpisode,
    Transcript,
    TranscriptCue,
)
from toolrecap_v4.analysis.source_prep.audio import (
    AudioSignalResult,
    check_audio_has_signal,
    check_audio_has_speech,
    get_audio_map_arg,
    measure_audio_signal,
    select_episode_audio_stream,
)
from toolrecap_v4.analysis.source_prep.probe import EpisodeProbeResult, probe_episode_source
from toolrecap_v4.analysis.source_prep.subtitles import (
    SubtitleCue,
    SubtitleTrack,
    compute_subtitle_cache_key,
    create_minimal_pgs_sup,
    create_synthetic_vobsub,
    discover_sidecars,
    extract_vobsub_events,
    match_episode,
    parse_ass,
    parse_pgs_sup,
    parse_srt,
    parse_vtt,
    select_best_english_subtitles,
)
from toolrecap_v4.analysis.source_prep.transcript import (
    build_transcript,
    compute_transcript_hash,
    validate_and_normalize_cue,
)
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError
from toolrecap_v4.media import AudioStreamInfo, CommandResult, MediaProbeResult, VideoStreamInfo


# ============================================================================
# 1. Audio Selection: Global Index vs Audio Ordinal & No-Audio Handling
# ============================================================================


def test_audio_selection_global_index_vs_ordinal():
    # Video stream index=0; audio stream 1 (index=1, ordinal=0); audio stream 2 (index=2, ordinal=1)
    s1 = AudioStreamInfo(
        index=1,
        codec="ac3",
        channels=6,
        sample_rate=48000,
        language="spa",
        title="Spanish",
    )
    s2 = AudioStreamInfo(
        index=2,
        codec="aac",
        channels=2,
        sample_rate=48000,
        language="eng",
        title="English Main",
        disposition={"default": 1},
    )

    selection = select_episode_audio_stream([s1, s2])
    assert selection.selected_stream == s2
    assert selection.global_index == 2
    assert selection.audio_ordinal == 1
    assert selection.has_audio is True
    assert selection.has_warning is False

    # Mapping helper must produce 0:<global_index>, never 0:a:<ordinal>
    map_args = get_audio_map_arg(selection)
    assert map_args == ["-map", "0:2"]
    assert selection.ffmpeg_map_spec() == "0:2"


def test_audio_selection_no_audio_streams():
    selection = select_episode_audio_stream([])
    assert selection.selected_stream is None
    assert selection.global_index == -1
    assert selection.audio_ordinal == -1
    assert selection.has_audio is False
    assert selection.has_warning is True

    with pytest.raises(ValueError, match="Cannot map audio for empty or invalid"):
        get_audio_map_arg(selection)

    with pytest.raises(ValueError, match="Cannot generate FFmpeg map specifier"):
        selection.ffmpeg_map_spec()


def test_audio_selection_unknown_language_fallback():
    s_und = AudioStreamInfo(
        index=3,
        codec="aac",
        channels=2,
        sample_rate=44100,
        language="und",
        title="",
        disposition={"default": 1},
    )
    selection = select_episode_audio_stream([s_und])
    assert selection.selected_stream == s_und
    assert selection.has_warning is True
    assert "undefined language" in (selection.warning or "")
    assert "not confirmed English" in selection.reason


# ============================================================================
# 2. PreparedEpisode Contract: Inventories, Serialization, & No-Audio
# ============================================================================


def test_prepared_episode_contract_roundtrip(tmp_path: Path):
    video_file = tmp_path / "Show.S01E01.mkv"
    video_file.touch()

    v_stream = VideoStreamInfo(
        index=0,
        codec="h264",
        width=1920,
        height=1080,
        fps=24.0,
        fps_text="24/1",
        duration=120.0,
        aspect_ratio="16:9",
    )
    a_stream = AudioStreamInfo(
        index=1,
        codec="aac",
        channels=2,
        sample_rate=48000,
        language="eng",
    )
    audio_sel = AudioSelection(
        selected_stream=a_stream,
        global_index=1,
        audio_ordinal=0,
    )

    t_cue = TranscriptCue(cue_id="E01-CUE-0001", start_ms=1000, end_ms=2500, text="Hello world")
    transcript = Transcript(
        episode_id="E01",
        source_type="sidecar",
        source_format="srt",
        cues=(t_cue,),
        language="eng",
        provenance_hash="fake-prov-hash",
    )

    prep = PreparedEpisode(
        episode_id="E01",
        source_id="SRC-001",
        source_path=video_file,
        duration_ms=120000,
        canvas_width=1920,
        canvas_height=1080,
        source_fingerprint="sig-12345",
        video_streams=(v_stream,),
        audio_streams=(a_stream,),
        video_info=v_stream,
        audio_info=a_stream,
        audio_selection=audio_sel,
        transcript=transcript,
        transcript_method="sidecar",
        artifact_hash="art-6789",
        dependency_signature={"type": "sidecar", "hash": "abc"},
        status="ready",
    )

    assert prep.source_basename == "Show.S01E01.mkv"
    d = prep.to_dict()
    reconstructed = PreparedEpisode.from_dict(d)

    assert reconstructed.episode_id == "E01"
    assert reconstructed.source_path == video_file
    assert reconstructed.source_basename == "Show.S01E01.mkv"
    assert reconstructed.audio_info is not None
    assert reconstructed.audio_info.index == 1
    assert reconstructed.transcript.cue_count == 1
    assert reconstructed.transcript.cues[0].text == "Hello world"


def test_prepared_episode_no_audio_without_fake_stream(tmp_path: Path):
    video_file = tmp_path / "Silent.S01E01.mkv"
    video_file.touch()

    v_stream = VideoStreamInfo(
        index=0,
        codec="h264",
        width=1280,
        height=720,
        fps=30.0,
        fps_text="30/1",
        duration=60.0,
        aspect_ratio="16:9",
    )
    empty_sel = AudioSelection(
        selected_stream=None,
        global_index=-1,
        audio_ordinal=-1,
        warning="No audio",
    )

    prep = PreparedEpisode(
        episode_id="E01",
        source_id="SRC-002",
        source_path=video_file,
        duration_ms=60000,
        canvas_width=1280,
        canvas_height=720,
        video_streams=(v_stream,),
        audio_streams=(),
        video_info=v_stream,
        audio_info=None,  # No fake audio stream!
        audio_selection=empty_sel,
    )

    assert prep.audio_info is None
    assert prep.audio_selection.has_audio is False

    d = prep.to_dict()
    reconstructed = PreparedEpisode.from_dict(d)
    assert reconstructed.audio_info is None


# ============================================================================
# 3. Transcript: Strict Boundaries, No-Clamp, & Diagnostics
# ============================================================================


def test_transcript_strict_boundaries_and_diagnostics():
    # 1. Reject boolean timestamp (isinstance(True, int) is True in Python)
    res, diag = validate_and_normalize_cue(True, 2000, "Text")
    assert res is None
    assert "boolean" in (diag or "")

    # 2. Reject non-integer timestamp
    res, diag = validate_and_normalize_cue("1000", 2000, "Text")
    assert res is None
    assert "not integer" in (diag or "")

    # 3. Reject cue exceeding source duration without silent clamp
    res, diag = validate_and_normalize_cue(1000, 5500, "Text", source_duration_ms=5000)
    assert res is None
    assert "material clamping forbidden" in (diag or "")

    # 4. Normalize minor negative start jitter within tolerance (-20ms -> 0ms)
    res, diag = validate_and_normalize_cue(-20, 1500, "Text")
    assert res is not None
    assert res[0] == 0
    assert res[1] == 1500

    # 5. Reject materially negative start (< -50ms)
    res, diag = validate_and_normalize_cue(-80, 1500, "Text")
    assert res is None
    assert "materially negative" in (diag or "")

    # 6. Reject non-positive duration (end_ms <= start_ms)
    res, diag = validate_and_normalize_cue(2000, 2000, "Text")
    assert res is None
    assert "non-positive duration" in (diag or "")

    # 7. Reject empty text after tag stripping
    res, diag = validate_and_normalize_cue(1000, 2000, "  <i></i>  ")
    assert res is None
    assert "empty after tag stripping" in (diag or "")


def test_build_transcript_diagnostics_and_provenance_hash():
    raw_cues = [
        {"start_ms": 1000, "end_ms": 2500, "text": "<b>Good cue</b>"},
        {"start_ms": 3000, "end_ms": 2000, "text": "Bad reversed cue"},  # invalid
        {"start_ms": 4000, "end_ms": 6000, "text": "Extends past duration"},  # invalid
    ]

    transcript = build_transcript(
        episode_id="E01",
        source_type="sidecar",
        source_format="srt",
        raw_cues=raw_cues,
        language="eng",
        source_duration_ms=5000,
    )

    assert transcript.cue_count == 1
    assert transcript.cues[0].cue_id == "E01-CUE-0001"
    assert transcript.cues[0].text == "Good cue"
    assert transcript.dropped_cues_count == 2
    assert len(transcript.diagnostics) == 2
    assert transcript.provenance_hash != ""


# ============================================================================
# 4. Probe: Never Swallow CancelledError & Subtitle Probe Diagnostics
# ============================================================================


def test_probe_propagates_cancellation(tmp_path: Path):
    dummy_file = tmp_path / "test.mkv"
    dummy_file.touch()

    token = CancellationToken()
    token.cancel()

    with pytest.raises(CancelledError):
        probe_episode_source(dummy_file, cancellation_token=token)


def test_probe_subtitle_failure_diagnostic(tmp_path: Path):
    dummy_file = tmp_path / "test.mkv"
    dummy_file.touch()

    fake_video = VideoStreamInfo(
        index=0, codec="h264", width=1920, height=1080, fps=24.0, fps_text="24/1", duration=10.0, aspect_ratio="16:9"
    )
    fake_probe = MediaProbeResult(
        path=dummy_file,
        duration=10.0,
        container="matroska",
        video_streams=(fake_video,),
        audio_streams=(),
    )

    with patch("toolrecap_v4.analysis.source_prep.probe.probe_media", return_value=fake_probe), \
         patch("toolrecap_v4.analysis.source_prep.probe.run_command", return_value=CommandResult(exit_code=1, stdout="", stderr="ffprobe error demo")):
        result = probe_episode_source(dummy_file)
        assert len(result.subtitle_streams) == 0
        assert result.subtitle_probe_error is not None
        assert "ffprobe subtitle query exit 1" in result.subtitle_probe_error


# ============================================================================
# 5. Audio Signal: Technical Measurement (Bounded RAM)
# ============================================================================


def test_measure_audio_signal_bounded(tmp_path: Path):
    wav_file = tmp_path / "test_tone.wav"
    # Generate 16-bit mono PCM WAV (0.5s tone)
    sample_rate = 16000
    num_samples = sample_rate // 2
    with wave.open(str(wav_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # 1000Hz amplitude 5000
        samples = [int(5000 * (1 if (i // 16) % 2 == 0 else -1)) for i in range(num_samples)]
        raw_bytes = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(raw_bytes)

    res = measure_audio_signal(wav_file, min_rms_threshold=100.0, chunk_frames=4096)
    assert isinstance(res, AudioSignalResult)
    assert res.has_signal is True
    assert res.rms_peak > 1000.0
    assert check_audio_has_signal(wav_file, min_rms_threshold=100.0) is True
    assert check_audio_has_speech(wav_file, min_rms_threshold=100.0) is True


# ============================================================================
# 6. Cache: Collision Resistance, Manifest Verification, & Cancellation
# ============================================================================


def test_safe_cache_filename_collision_resistance():
    name1 = safe_cache_filename("item:1")
    name2 = safe_cache_filename("item_1")
    assert name1 != name2


def test_cache_verify_size_and_key_and_hash(tmp_path: Path):
    cache = AnalysisCacheManager(tmp_path / "cache")
    key = "test-key-alpha"
    data = {"result": "ok", "value": 42}

    saved_path = cache.save_artifact(key, data)
    assert saved_path.is_file()

    # Valid load
    loaded = cache.load_artifact(key)
    assert loaded == data

    # 1. Tamper cache_key in manifest
    manifest_file = tmp_path / "cache" / f"{safe_cache_filename(key)}.manifest.json"
    manifest_data = json.loads(manifest_file.read_text("utf-8"))
    manifest_data["cache_key"] = "wrong-key"
    manifest_file.write_text(json.dumps(manifest_data), "utf-8")
    assert cache.load_artifact(key) is None

    # Restore key, tamper data_size
    manifest_data["cache_key"] = key
    manifest_data["data_size"] = 99999
    manifest_file.write_text(json.dumps(manifest_data), "utf-8")
    assert cache.load_artifact(key) is None

    # Restore size, tamper data content
    manifest_data["data_size"] = len(saved_path.read_bytes())
    manifest_file.write_text(json.dumps(manifest_data), "utf-8")
    saved_path.write_text(json.dumps({"tampered": True}), "utf-8")
    assert cache.load_artifact(key) is None


def test_cache_cancellation_aborts_promotion(tmp_path: Path):
    cache = AnalysisCacheManager(tmp_path / "cache")
    token = CancellationToken()
    token.cancel()

    with pytest.raises(CancelledError):
        cache.save_artifact("cancel-key", {"data": 1}, cancellation_token=token)

    # Verify no promoted files remain
    files = list((tmp_path / "cache").glob("*.data.json"))
    assert len(files) == 0


# ============================================================================
# 7. Sidecars: Content Hash Verification (Same Metadata, Different Content)
# ============================================================================


def test_sidecar_content_hash_different_content(tmp_path: Path):
    sidecar_a = tmp_path / "sub_a.srt"
    sidecar_b = tmp_path / "sub_b.srt"

    # Same size, different content
    sidecar_a.write_text("1\n00:00:01,000 --> 00:00:02,000\nAlpha\n", encoding="utf-8")
    sidecar_b.write_text("1\n00:00:01,000 --> 00:00:02,000\nBravo\n", encoding="utf-8")

    sig_a = compute_sidecar_signature(sidecar_a)
    sig_b = compute_sidecar_signature(sidecar_b)

    assert sig_a["content_hash"] != sig_b["content_hash"]


# ============================================================================
# 8. Discovery: Strict Episode Match & Forced-Only Isolation
# ============================================================================


def test_discovery_strict_episode_matching_rejects_wrong_episode():
    # Matching episode
    assert match_episode("Show.S01E02.1080p.mkv", "Show.S01E02.srt", episode_id="E02") is True
    # Wrong episode number: must be rejected to prevent cross-episode contamination!
    assert match_episode("Show.S01E02.1080p.mkv", "Show.S01E03.srt", episode_id="E02") is False


def test_discovery_forced_only_isolation():
    forced_track = SubtitleTrack(
        track_id="stream:2:srt",
        source_type="embedded",
        source_format="srt",
        language="eng",
        is_forced=True,
        is_full=False,
    )
    result = select_best_english_subtitles([forced_track])
    # Forced track must NEVER be chosen as best full track
    assert result.best_english_full is None
    assert result.stt_required is True
    assert len(result.forced_tracks) == 1


# ============================================================================
# 9. Direct Subtitle Parsers: SRT, VTT, ASS with Tag Stripping
# ============================================================================


def test_parsers_srt_vtt_ass_tag_stripping_and_duration():
    srt_content = """1
00:00:01,000 --> 00:00:02,500
<font color="red">Hello <b>World</b></font>

2
00:00:05,000 --> 00:00:08,000
Second cue
"""
    cues_srt = parse_srt(srt_content, source_duration_ms=6000)
    assert len(cues_srt) == 1
    assert cues_srt[0].start_ms == 1000
    assert cues_srt[0].end_ms == 2500
    assert cues_srt[0].text == "Hello World"

    vtt_content = """WEBVTT

00:01.000 --> 00:02.500
<c.yellow>VTT Cue</c>
"""
    cues_vtt = parse_vtt(vtt_content)
    assert len(cues_vtt) == 1
    assert cues_vtt[0].text == "VTT Cue"

    ass_content = """[Script Info]
Title: Test
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,{\\pos(100,200)}ASS text\\NSecond line
"""
    cues_ass = parse_ass(ass_content)
    assert len(cues_ass) == 1
    assert cues_ass[0].start_ms == 1000
    assert cues_ass[0].end_ms == 3500
    assert cues_ass[0].text == "ASS text\nSecond line"


# ============================================================================
# 10. PGS & VobSub: Valid, Malformed, Missing Pair, & Pair Hashing
# ============================================================================


def test_pgs_valid_and_malformed():
    valid_pgs = create_minimal_pgs_sup(start_ms=1000, end_ms=3000)
    events = parse_pgs_sup(valid_pgs)
    assert len(events) == 1
    assert events[0].start_ms == 1000
    assert events[0].end_ms == 3000
    assert isinstance(events[0].image, Image.Image)

    # Malformed / truncated binary stream: must not crash
    truncated_pgs = valid_pgs[:35]
    malformed_events = parse_pgs_sup(truncated_pgs)
    assert isinstance(malformed_events, list)


def test_vobsub_valid_missing_pair_and_pair_hash(tmp_path: Path):
    idx_file = tmp_path / "test.idx"
    sub_file = tmp_path / "test.sub"

    create_synthetic_vobsub(idx_file, sub_file, start_ms=1500)
    assert idx_file.is_file()
    assert sub_file.is_file()

    # 1. Valid extraction
    events = extract_vobsub_events(idx_file, sub_file)
    assert len(events) == 1
    assert events[0].start_ms == 1500
    assert isinstance(events[0].image, Image.Image)

    # 2. Pair signature and hash
    sig_1 = compute_vobsub_signature(idx_file, sub_file)
    assert "pair_hash" in sig_1

    # Modify .sub content and verify pair_hash changes
    sub_file.write_bytes(sub_file.read_bytes() + b"\x00")
    sig_2 = compute_vobsub_signature(idx_file, sub_file)
    assert sig_1["pair_hash"] != sig_2["pair_hash"]

    # 3. Missing paired .sub file must raise FileNotFoundError
    sub_file.unlink()
    with pytest.raises(FileNotFoundError, match="not found"):
        compute_vobsub_signature(idx_file, sub_file)

    with pytest.raises(FileNotFoundError, match="matching .sub file not found"):
        extract_vobsub_events(idx_file, sub_file)
