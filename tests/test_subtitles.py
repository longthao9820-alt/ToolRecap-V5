"""Targeted tests for subtitle generation and timing offsets."""

from pathlib import Path

import pytest

from toolrecap_v4.subtitles import (
    SubtitleCue,
    build_srt,
    escape_ffmpeg_subtitles_path,
    extract_subtitles_for_output,
    format_srt_time,
    generate_subtitles,
    parse_srt,
    parse_srt_time,
    write_srt,
)


def test_format_and_parse_srt_time():
    """Verify timestamp conversion between milliseconds and SRT string."""
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(500) == "00:00:00,500"
    assert format_srt_time(61500) == "00:01:01,500"
    assert format_srt_time(3661500) == "01:01:01,500"
    assert format_srt_time(-100) == "00:00:00,000"

    assert parse_srt_time("00:00:00,000") == 0
    assert parse_srt_time("00:00:00,500") == 500
    assert parse_srt_time("00:01:01,500") == 61500
    assert parse_srt_time("01:01:01,500") == 3661500
    assert parse_srt_time("00:00:01.250") == 1250  # Handles dot separator


def test_build_and_parse_srt_roundtrip():
    """Verify build_srt output matches standard SRT format and roundtrips."""
    cues = [
        SubtitleCue(start_ms=1000, end_ms=3500, text="First subtitle line"),
        SubtitleCue(start_ms=4000, end_ms=6200, text="Second line\nwith multiline"),
    ]

    srt_text = build_srt(cues)
    expected = (
        "1\n00:00:01,000 --> 00:00:03,500\nFirst subtitle line\n\n"
        "2\n00:00:04,000 --> 00:00:06,200\nSecond line\nwith multiline\n"
    )
    assert srt_text == expected

    parsed = parse_srt(srt_text)
    assert len(parsed) == 2
    assert parsed[0].start_ms == 1000
    assert parsed[0].end_ms == 3500
    assert parsed[0].text == "First subtitle line"
    assert parsed[1].start_ms == 4000
    assert parsed[1].end_ms == 6200
    assert parsed[1].text == "Second line\nwith multiline"


def test_extract_subtitles_offsets():
    """Verify subtitle cues are offset by accumulated segment durations across timeline."""
    output_def = {
        "render_id": "out_01",
        "title": "Episode_1",
        "segments": [
            {
                "segment_id": "seg_1",
                "source_file": "ep1.mp4",
                "start_ms": 1000,
                "end_ms": 5000,  # duration 4000ms
                "type": "narration",
                "narration": "Narration text 1",
                "source_audio": True,
                "subtitles": [
                    {"start_ms": 0, "end_ms": 2000, "text": "Seg1 cue 1"},
                    {"start_ms": 2000, "end_ms": 3800, "text": "Seg1 cue 2"},
                ],
            },
            {
                "segment_id": "seg_2",
                "source_file": "ep1.mp4",
                "start_ms": 10000,
                "end_ms": 13000,  # duration 3000ms
                "type": "original_dialogue",
                "narration": "",
                "source_audio": True,
                "subtitles": [
                    {"start_ms": 500, "end_ms": 2500, "text": "Seg2 dialogue"},
                ],
            },
            {
                "segment_id": "seg_3",
                "source_file": "ep2.mp4",
                "start_ms": 0,
                "end_ms": 2000,  # duration 2000ms
                "type": "narration",
                "narration": "Seg3 synthesized narration",
                "source_audio": False,
                "subtitles": [],  # synthesized from narration
            },
        ],
    }

    narr_cues, orig_cues = extract_subtitles_for_output(output_def)

    # Seg 1 narration cues: timeline offset 0
    assert len(narr_cues) == 3
    assert narr_cues[0].start_ms == 0
    assert narr_cues[0].end_ms == 2000
    assert narr_cues[0].text == "Seg1 cue 1"

    assert narr_cues[1].start_ms == 2000
    assert narr_cues[1].end_ms == 3800
    assert narr_cues[1].text == "Seg1 cue 2"

    # Seg 2 original dialogue: timeline offset = 4000ms
    assert len(orig_cues) == 1
    assert orig_cues[0].start_ms == 4000 + 500  # 4500
    assert orig_cues[0].end_ms == 4000 + 2500   # 6500
    assert orig_cues[0].text == "Seg2 dialogue"

    # Seg 3 narration cue: timeline offset = 4000 + 3000 = 7000ms
    assert narr_cues[2].start_ms == 7000
    assert narr_cues[2].end_ms == 7000 + 2000   # 9000
    assert narr_cues[2].text == "Seg3 synthesized narration"


def test_generate_subtitles_creates_both_files(tmp_path: Path):
    """Verify both {title}.narration.srt and {title}.original.srt are created."""
    output_def = {
        "title": "TestMovie",
        "segments": [
            {
                "segment_id": "s1",
                "source_file": "src.mp4",
                "start_ms": 0,
                "end_ms": 2000,
                "type": "narration",
                "narration": "Welcome to the show",
                "source_audio": True,
                "subtitles": [
                    {"start_ms": 0, "end_ms": 1500, "text": "Welcome to the show"}
                ],
            }
        ],
    }

    narr_p, orig_p = generate_subtitles(output_def, tmp_path)

    assert narr_p.is_file()
    assert orig_p.is_file()
    assert narr_p.name == "TestMovie.narration.srt"
    assert orig_p.name == "TestMovie.original.srt"

    # Narration has cues
    narr_content = narr_p.read_text(encoding="utf-8")
    assert "Welcome to the show" in narr_content

    # Original has 0 cues, should be empty string
    orig_content = orig_p.read_text(encoding="utf-8")
    assert orig_content == ""


def test_escape_ffmpeg_subtitles_path():
    """Verify path escaping for FFmpeg subtitles filter on Windows."""
    escaped = escape_ffmpeg_subtitles_path(r"C:\Users\Example\sub.srt")
    assert r"C\:/Users/Example/sub.srt" in escaped
