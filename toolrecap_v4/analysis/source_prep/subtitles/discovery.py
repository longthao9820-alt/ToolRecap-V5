"""Subtitle discovery, episode/stem matching, scoring, and track selection."""
from __future__ import annotations

from pathlib import Path
import re

from .models import (
    SubtitleDiscoveryResult,
    SubtitleStreamInfo,
    SubtitleTrack,
    normalize_language_code,
)


def extract_episode_identifiers(text: str) -> set[tuple[int | None, int]]:
    """Extract (season, episode) pairs from a filename, stem, or episode_id."""
    results: set[tuple[int | None, int]] = set()
    s = text.strip()

    # Pattern: S01E02 or s1e2 or S01-E02
    for m in re.finditer(r"[sS](\d+)\s*[-_.]?\s*[eE](\d+)", s):
        results.add((int(m.group(1)), int(m.group(2))))

    # Pattern: 1x02
    for m in re.finditer(r"(\d+)x(\d+)", s):
        results.add((int(m.group(1)), int(m.group(2))))

    # Pattern: E02, Ep02, Ep.02, Episode 02
    for m in re.finditer(r"(?:^|[\b_.\-\[\(])(?:ep|episode|e)\.?\s*(\d+)(?:$|[\b_.\-\]\)])", s, re.IGNORECASE):
        results.add((None, int(m.group(1))))

    # Pattern: [02] or (02)
    for m in re.finditer(r"[\[\(](\d{1,3})[\]\)]", s):
        results.add((None, int(m.group(1))))

    # Pattern: ' - 02'
    for m in re.finditer(r"\s+-\s+(\d{1,3})(?:[\b_.\-\s]|$)", s):
        results.add((None, int(m.group(1))))

    return results


def extract_show_prefix(filename: str) -> str:
    """Extract show/series name prefix appearing before episode markers."""
    stem = Path(filename).stem
    m = re.search(
        r"[sS]\d+\s*[-_.]?\s*[eE]\d+|\d+x\d+|(?:^|[\b_.\-\[\(])(?:ep|episode|e)\.?\s*\d+",
        stem,
        re.IGNORECASE,
    )
    if m and m.start() > 0:
        prefix = stem[:m.start()].strip("._- ")
        return re.sub(r"[._\- ]+", " ", prefix).strip().lower()
    return ""


def match_episode(video_name: str, sidecar_name: str, episode_id: str = "") -> bool:
    """Determine whether a sidecar file matches the video and episode without cross-episode contamination."""
    v_ids = extract_episode_identifiers(video_name)
    if episode_id:
        v_ids.update(extract_episode_identifiers(episode_id))

    s_ids = extract_episode_identifiers(sidecar_name)

    # 1. If sidecar contains episode numbers
    if s_ids:
        if v_ids:
            matched = False
            for v_season, v_ep in v_ids:
                for s_season, s_ep in s_ids:
                    if v_season is not None and s_season is not None:
                        if v_season == s_season and v_ep == s_ep:
                            matched = True
                            break
                    else:
                        if v_ep == s_ep:
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                # Definite episode mismatch! Never accept wrong episode!
                return False

            v_prefix = extract_show_prefix(video_name)
            s_prefix = extract_show_prefix(sidecar_name)
            if v_prefix and s_prefix:
                if v_prefix != s_prefix and v_prefix not in s_prefix and s_prefix not in v_prefix:
                    return False

            return True
        else:
            # Sidecar has episode number but video has none: mismatch
            return False

    # 2. Sidecar has NO episode number: check base stem matching
    v_stem = Path(video_name).stem.lower()
    s_stem = Path(sidecar_name).stem.lower()

    if s_stem == v_stem:
        return True

    if s_stem.startswith(v_stem):
        rem = s_stem[len(v_stem):]
        if rem.startswith((".", "_", "-")):
            return True

    return False


def detect_sidecar_metadata(sidecar_path: Path, video_stem: str) -> tuple[str, bool, str]:
    """Detect language, is_forced flag, and format from sidecar filename and extension."""
    suffix = sidecar_path.suffix.lower()
    ext_format_map = {
        ".srt": "srt",
        ".ass": "ass",
        ".ssa": "ssa",
        ".vtt": "vtt",
        ".idx": "vobsub",
    }
    fmt = ext_format_map.get(suffix, "srt")

    stem = sidecar_path.stem.lower()
    v_stem = video_stem.lower()

    if stem.startswith(v_stem):
        tag_part = stem[len(v_stem):]
    else:
        tag_part = stem

    tokens = [t for t in re.split(r"[\._\-\s]+", tag_part) if t]
    is_forced = any(tok in ("forced", "foreign", "supplemental") for tok in tokens)

    lang = "und"
    for tok in tokens:
        if tok in ("forced", "foreign", "supplemental"):
            continue
        norm = normalize_language_code(tok)
        if norm != "und":
            lang = norm
            break

    return lang, is_forced, fmt


def discover_sidecars(
    video_path: Path | str,
    episode_id: str = "",
) -> list[SubtitleTrack]:
    """Discover valid sidecar subtitle files in the SAME directory only.

    Supports .srt, .ass, .ssa, .vtt, and paired .idx/.sub.
    Rejects files with wrong episode numbers or unrelated stems.
    """
    v_path = Path(video_path).resolve()
    if not v_path.is_file():
        return []

    parent_dir = v_path.parent
    v_stem = v_path.stem
    tracks: list[SubtitleTrack] = []

    supported_exts = {".srt", ".ass", ".ssa", ".vtt", ".idx"}

    try:
        entries = list(parent_dir.iterdir())
    except OSError:
        return []

    for entry in entries:
        if not entry.is_file():
            continue
        if entry.resolve() == v_path:
            continue
        ext = entry.suffix.lower()
        if ext not in supported_exts:
            continue

        # Strict episode and stem matching
        if not match_episode(v_path.name, entry.name, episode_id=episode_id):
            continue

        # For VobSub (.idx), must have paired .sub in the same directory
        if ext == ".idx":
            paired_sub = entry.with_suffix(".sub")
            if not paired_sub.is_file():
                continue

        lang, is_forced, fmt = detect_sidecar_metadata(entry, v_stem)
        is_bitmap = (fmt == "vobsub")

        # Base score calculation
        score = 100.0 if not is_bitmap else 50.0
        if entry.stem.lower() == v_stem.lower():
            score += 15.0
        elif entry.stem.lower().startswith(v_stem.lower()):
            score += 10.0

        track = SubtitleTrack(
            track_id=f"sidecar:{entry.name}",
            source_type="sidecar",
            source_format=fmt,
            language=lang,
            title=entry.name,
            stream_index=None,
            source_file=str(entry.resolve()),
            is_forced=is_forced,
            is_full=(not is_forced and lang == "eng"),
            is_bitmap=is_bitmap,
            score=score,
        )
        tracks.append(track)

    return tracks


def build_embedded_tracks(
    streams: list[SubtitleStreamInfo],
    source_video: str,
) -> list[SubtitleTrack]:
    """Convert probed embedded subtitle stream info into SubtitleTrack records."""
    tracks: list[SubtitleTrack] = []
    bitmap_codecs = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvdsub", "pgs"}

    for s in streams:
        codec = s.codec.lower()
        is_bitmap = codec in bitmap_codecs or s.is_bitmap
        if codec in ("subrip", "srt", "mov_text", "tx3g"):
            fmt = "srt"
        elif codec in ("ass", "ssa"):
            fmt = "ass"
        elif codec == "webvtt":
            fmt = "vtt"
        elif codec in ("hdmv_pgs_subtitle", "pgs"):
            fmt = "pgs"
        elif codec in ("dvd_subtitle", "dvdsub"):
            fmt = "vobsub"
        else:
            fmt = codec

        lang = normalize_language_code(s.language)
        if lang == "und" and re.search(r"(?i)(?<![a-z])(?:english|eng|en-us|en-gb)(?![a-z])", s.title):
            # Some containers omit the language tag but label the track
            # explicitly in its title. Unknown titles remain unknown.
            lang = "eng"
        title_lower = s.title.lower()
        is_forced = s.forced or ("forced" in title_lower) or ("foreign" in title_lower)

        score = 100.0 if not is_bitmap else 50.0
        if s.default:
            score += 10.0
        if fmt == "ass":
            score += 5.0

        track = SubtitleTrack(
            track_id=f"embedded:{s.index}:{fmt}",
            source_type="embedded",
            source_format=fmt,
            language=lang,
            title=s.title or f"Stream #{s.index} ({fmt})",
            stream_index=s.index,
            source_file=source_video,
            is_forced=is_forced,
            is_full=(not is_forced and lang == "eng"),
            is_bitmap=is_bitmap,
            score=score,
        )
        tracks.append(track)

    return tracks


def select_best_english_subtitles(
    tracks: list[SubtitleTrack],
    video_path: str = "",
    episode_id: str = "",
) -> SubtitleDiscoveryResult:
    """Select best English Full subtitle track from discovered sidecars and embedded streams.

    Invariants:
    - Forced subtitles are supplemental only; never chosen as Full.
    - Preference hierarchy: English Full text -> English Full bitmap -> STT fallback.
    - If only forced or no English subtitles available, sets best_english_full=None and stt_required=True.
    """
    forced_tracks: list[SubtitleTrack] = [t for t in tracks if t.is_forced]
    full_candidates: list[SubtitleTrack] = [t for t in tracks if not t.is_forced and t.language == "eng"]

    # 1. Check Sidecar English Full Text candidates (preferred over embedded text regardless raw score)
    sidecar_text = [t for t in full_candidates if not t.is_bitmap and t.source_type == "sidecar"]
    if sidecar_text:
        best = max(sidecar_text, key=lambda t: t.score)
        return SubtitleDiscoveryResult(
            video_path=video_path,
            episode_id=episode_id,
            all_tracks=tracks,
            best_english_full=best,
            forced_tracks=forced_tracks,
            stt_required=False,
            selection_reason=f"Selected English Full sidecar text ({best.source_format}, score={best.score:.1f})",
        )

    # 2. Check Embedded English Full Text candidates (preferred over bitmap regardless raw score)
    embedded_text = [t for t in full_candidates if not t.is_bitmap and t.source_type == "embedded"]
    if embedded_text:
        best = max(embedded_text, key=lambda t: t.score)
        return SubtitleDiscoveryResult(
            video_path=video_path,
            episode_id=episode_id,
            all_tracks=tracks,
            best_english_full=best,
            forced_tracks=forced_tracks,
            stt_required=False,
            selection_reason=f"Selected English Full embedded text ({best.source_format}, score={best.score:.1f})",
        )

    # 3. Check English Full Bitmap candidates (PGS / VobSub; sidecar over embedded, then score)
    bitmap_candidates = [t for t in full_candidates if t.is_bitmap]
    if bitmap_candidates:
        best = max(bitmap_candidates, key=lambda t: (1 if t.source_type == "sidecar" else 0, t.score))
        return SubtitleDiscoveryResult(
            video_path=video_path,
            episode_id=episode_id,
            all_tracks=tracks,
            best_english_full=best,
            forced_tracks=forced_tracks,
            stt_required=False,
            selection_reason=f"Selected English Full bitmap ({best.source_type}:{best.source_format}, score={best.score:.1f})",
        )

    # 3. No English Full tracks: forced only, foreign only, or no tracks -> STT fallback
    reason = (
        "No English Full subtitles found (only forced or foreign available); internal STT fallback required."
        if forced_tracks or tracks
        else "No subtitle tracks discovered; internal STT fallback required."
    )
    return SubtitleDiscoveryResult(
        video_path=video_path,
        episode_id=episode_id,
        all_tracks=tracks,
        best_english_full=None,
        forced_tracks=forced_tracks,
        stt_required=True,
        selection_reason=reason,
    )
