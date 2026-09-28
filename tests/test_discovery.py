"""Tests for source video discovery, deterministic natural sorting, and collision detection."""

from pathlib import Path
import pytest

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.discovery import (
    SUPPORTED_EXTENSIONS,
    compute_file_fingerprint,
    create_source_map,
    discover_sources,
    natural_sort_key,
)
from toolrecap_v4.errors import (
    CancelledError,
    DiscoveryError,
    WindowsCollisionError,
    WindowsReservedNameError,
)


def test_supported_8_formats_discovered(tmp_path: Path) -> None:
    """Verify exact 8 supported formats are discovered and other files ignored."""
    # Create all 8 supported formats
    for ext in [".mp4", ".mkv", ".mov", ".m4v", ".avi", ".ts", ".m2ts", ".webm"]:
        (tmp_path / f"video{ext}").write_bytes(b"dummy video content")

    # Create unsupported files
    (tmp_path / "notes.txt").write_text("notes")
    (tmp_path / "poster.jpg").write_bytes(b"jpg")
    (tmp_path / "audio.mp3").write_bytes(b"mp3")
    (tmp_path / "data.json").write_text("{}")

    results = discover_sources(tmp_path)
    discovered_exts = {Path(r.path).suffix.lower() for r in results}

    assert discovered_exts == SUPPORTED_EXTENSIONS
    assert len(results) == 8


def test_nonrecursive_nested_exclusion(tmp_path: Path) -> None:
    """Verify discovery does NOT recurse into subdirectories (nested exclusion)."""
    # Top-level video
    (tmp_path / "top_level.mp4").write_bytes(b"top")

    # Subdirectory with videos
    sub_dir = tmp_path / "sub_folder"
    sub_dir.mkdir()
    (sub_dir / "nested1.mp4").write_bytes(b"nested1")
    (sub_dir / "nested2.mkv").write_bytes(b"nested2")

    # Deeply nested directory
    deep_dir = sub_dir / "deep"
    deep_dir.mkdir()
    (deep_dir / "deep.webm").write_bytes(b"deep")

    results = discover_sources(tmp_path)

    assert len(results) == 1
    assert results[0].basename == "top_level.mp4"


def test_case_insensitive_extensions(tmp_path: Path) -> None:
    """Verify video formats are matched case-insensitively."""
    (tmp_path / "clip1.MP4").write_bytes(b"1")
    (tmp_path / "clip2.Mkv").write_bytes(b"2")
    (tmp_path / "clip3.WebM").write_bytes(b"3")

    results = discover_sources(tmp_path)
    basenames = [r.basename for r in results]

    assert "clip1.MP4" in basenames
    assert "clip2.Mkv" in basenames
    assert "clip3.WebM" in basenames
    assert len(results) == 3


def test_natural_ordering_deterministic(tmp_path: Path) -> None:
    """Verify natural sorting handles numeric chunks naturally and deterministically."""
    files = [
        "ep10.mp4",
        "ep1.mp4",
        "ep2.mp4",
        "ep20.mp4",
        "ep3.mp4",
        "ep02.mp4",
    ]
    for f in files:
        (tmp_path / f).write_bytes(b"sample")

    results = discover_sources(tmp_path)
    sorted_names = [r.basename for r in results]

    # ep1 < ep02 < ep2 < ep3 < ep10 < ep20
    assert sorted_names == [
        "ep1.mp4",
        "ep02.mp4",
        "ep2.mp4",
        "ep3.mp4",
        "ep10.mp4",
        "ep20.mp4",
    ]


def test_natural_sort_key_behavior() -> None:
    """Directly test natural_sort_key function."""
    items = ["item2", "item10", "item1", "Item1"]
    sorted_items = sorted(items, key=natural_sort_key)
    # item1 and Item1 casefolded equal; tie-breaker places Item1 before item1 (ASCII 'I' < 'i')
    assert sorted_items == ["Item1", "item1", "item2", "item10"]


def test_exact_basename_source_mapping(tmp_path: Path) -> None:
    """Verify exact basename mapping maps each file to its fingerprint."""
    f1 = tmp_path / "scene_01.mp4"
    f2 = tmp_path / "scene_02.mkv"
    f1.write_bytes(b"scene1_bytes")
    f2.write_bytes(b"scene2_bytes")

    fingerprints = discover_sources(tmp_path)
    source_map = create_source_map(fingerprints)

    assert "scene_01.mp4" in source_map
    assert "scene_02.mkv" in source_map
    assert source_map["scene_01.mp4"].size_bytes == len(b"scene1_bytes")
    assert source_map["scene_02.mkv"].size_bytes == len(b"scene2_bytes")


def test_windows_collision_in_source_mapping() -> None:
    """Verify source mapping detects casefold collision and raises WindowsCollisionError."""
    from toolrecap_v4.discovery import SourceFingerprint

    fp1 = SourceFingerprint("video.mp4", "/path/video.mp4", 100, 1000, "hash1", ".mp4")
    fp2 = SourceFingerprint("VIDEO.MP4", "/path/VIDEO.MP4", 100, 1000, "hash2", ".mp4")

    with pytest.raises(WindowsCollisionError, match="Duplicate source basename under casefold"):
        create_source_map([fp1, fp2])


def test_preserve_source_bytes(tmp_path: Path) -> None:
    """Verify computing fingerprints strictly preserves source bytes and returns correct sha256."""
    import hashlib

    content = b"exact byte preservation test content \x00\xff\xfe"
    video_file = tmp_path / "source.mp4"
    video_file.write_bytes(content)

    expected_sha = hashlib.sha256(content).hexdigest()

    fp = compute_file_fingerprint(video_file)

    assert fp.sha256 == expected_sha
    assert fp.size_bytes == len(content)
    assert video_file.read_bytes() == content  # Bytes unchanged


def test_windows_reserved_stem_in_discovery(tmp_path: Path) -> None:
    """Verify discovery rejects Windows reserved device names (e.g., con.mp4, aux.mkv)."""
    # Note: On Windows NTFS, creating CON.mp4 may fail at OS level, but let's test is_windows_reserved_stem
    # and discovery logic
    from toolrecap_v4.discovery import is_windows_reserved_stem

    assert is_windows_reserved_stem("CON")
    assert is_windows_reserved_stem("con.mp4")
    assert is_windows_reserved_stem("AUX")
    assert is_windows_reserved_stem("nul")
    assert is_windows_reserved_stem("COM1")
    assert is_windows_reserved_stem("lpt9")
    assert not is_windows_reserved_stem("concert.mp4")
    assert not is_windows_reserved_stem("auxiliary.mkv")


def test_discovery_cancellation(tmp_path: Path) -> None:
    """Verify discovery respects cancellation token."""
    (tmp_path / "clip.mp4").write_bytes(b"data")

    token = CancellationToken()
    token.cancel()

    with pytest.raises(CancelledError):
        discover_sources(tmp_path, cancellation_token=token)


def test_discovery_invalid_directory() -> None:
    """Verify discovery raises DiscoveryError for nonexistent directory."""
    with pytest.raises(DiscoveryError):
        discover_sources("C:/NonExistentPath_XYZ_123")
