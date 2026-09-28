"""VobSub (.idx / .sub) subtitle parser, pairing validation, and event extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import struct
from typing import Any, Callable

from PIL import Image

from toolrecap_v4.cancellation import CancellationToken
from .models import VobSubEvent


@dataclass
class VobSubTrackInfo:
    language: str
    track_index: int
    id_str: str = ""
    timestamps: list[tuple[int, int]] = field(default_factory=list)  # (timestamp_ms, filepos)


def parse_vobsub_timestamp(ts_str: str) -> int:
    """Parse VobSub timestamp 'HH:MM:SS:mmm' to milliseconds."""
    parts = ts_str.strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid VobSub timestamp: '{ts_str}'")
    h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    return (h * 3600 + m * 60 + s) * 1000 + ms


def parse_vobsub_idx(idx_path: str | Path) -> dict[str, Any]:
    """Parse VobSub .idx index file into structured metadata."""
    p = Path(idx_path)
    if not p.is_file():
        raise FileNotFoundError(f"VobSub index file not found: {p}")

    content = p.read_text(encoding="utf-8-sig", errors="replace")
    lines = content.splitlines()

    metadata: dict[str, Any] = {
        "size": (720, 480),
        "palette": [],
        "tracks": [],
    }

    current_track: VobSubTrackInfo | None = None
    tracks: list[VobSubTrackInfo] = []
    ts_pattern = re.compile(r"timestamp:\s*(\d{2}:\d{2}:\d{2}:\d{3}),\s*filepos:\s*([0-9a-fA-F]+)")

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.lower().startswith("size:"):
            size_part = stripped[5:].strip()
            w_h = size_part.split("x")
            if len(w_h) == 2:
                try:
                    metadata["size"] = (int(w_h[0]), int(w_h[1]))
                except ValueError:
                    pass
            continue

        if stripped.lower().startswith("palette:"):
            raw_pal = stripped[8:].strip().split(",")
            metadata["palette"] = [color.strip() for color in raw_pal if color.strip()]
            continue

        if stripped.lower().startswith("id:"):
            m_id = re.search(r"id:\s*([a-zA-Z\-]+)(?:,\s*index:\s*(\d+))?", stripped, re.IGNORECASE)
            if m_id:
                lang = m_id.group(1).lower()
                idx_num = int(m_id.group(2)) if m_id.group(2) else len(tracks)
                current_track = VobSubTrackInfo(language=lang, track_index=idx_num, id_str=stripped)
                tracks.append(current_track)
            continue

        m_ts = ts_pattern.search(stripped)
        if m_ts and current_track is not None:
            try:
                ts_ms = parse_vobsub_timestamp(m_ts.group(1))
                filepos = int(m_ts.group(2), 16)
                current_track.timestamps.append((ts_ms, filepos))
            except ValueError:
                continue

    metadata["tracks"] = tracks
    return metadata


def parse_spu_packet(
    buf: bytes,
    palette_rgbs: list[tuple[int, int, int]],
) -> tuple[Image.Image | None, int, int, int, int]:
    """Parse DVD SPU (Subpicture) packet into RGBA Image cropped to non-transparent bounding box.

    Returns (cropped_image, crop_x, crop_y, crop_w, crop_h).
    """
    if len(buf) < 4:
        return None, 0, 0, 0, 0

    try:
        total_size, dcsq_offset = struct.unpack_from(">HH", buf, 0)
    except struct.error:
        return None, 0, 0, 0, 0

    if dcsq_offset >= len(buf):
        return None, 0, 0, 0, 0

    cur_offset = dcsq_offset
    top_offset = 4
    bottom_offset = 4
    width = 0
    height = 0
    crop_x = 0
    crop_y = 0
    color_map = (0, 1, 2, 3)
    alpha_map = (0, 255, 255, 255)

    while cur_offset + 4 <= len(buf):
        try:
            delay, next_offset = struct.unpack_from(">HH", buf, cur_offset)
        except struct.error:
            break
        cmd_pos = cur_offset + 4
        while cmd_pos < len(buf):
            cmd = buf[cmd_pos]
            cmd_pos += 1
            if cmd == 0xFF:  # CMD_END
                break
            elif cmd in (0x00, 0x01, 0x02):  # FSTA_DSP, STA_DSP, STP_DSP
                continue
            elif cmd == 0x03:  # SET_COLOR
                if cmd_pos + 2 <= len(buf):
                    b0, b1 = buf[cmd_pos], buf[cmd_pos + 1]
                    color_map = (b1 & 0x0F, (b1 >> 4) & 0x0F, b0 & 0x0F, (b0 >> 4) & 0x0F)
                    cmd_pos += 2
            elif cmd == 0x04:  # SET_CONTR
                if cmd_pos + 2 <= len(buf):
                    b0, b1 = buf[cmd_pos], buf[cmd_pos + 1]
                    alpha_map = (
                        (b1 & 0x0F) * 17,
                        ((b1 >> 4) & 0x0F) * 17,
                        (b0 & 0x0F) * 17,
                        ((b0 >> 4) & 0x0F) * 17,
                    )
                    cmd_pos += 2
            elif cmd == 0x05:  # SET_DAREA
                if cmd_pos + 6 <= len(buf):
                    b = buf[cmd_pos : cmd_pos + 6]
                    x_start = (b[0] << 4) | (b[1] >> 4)
                    x_end = ((b[1] & 0x0F) << 8) | b[2]
                    y_start = (b[3] << 4) | (b[4] >> 4)
                    y_end = ((b[4] & 0x0F) << 8) | b[5]
                    width = max(0, x_end - x_start + 1)
                    height = max(0, y_end - y_start + 1)
                    crop_x = x_start
                    crop_y = y_start
                    cmd_pos += 6
            elif cmd == 0x06:  # SET_DSPXA
                if cmd_pos + 4 <= len(buf):
                    try:
                        top_offset, bottom_offset = struct.unpack_from(">HH", buf, cmd_pos)
                    except struct.error:
                        pass
                    cmd_pos += 4
            else:
                break

        if next_offset == cur_offset or next_offset >= len(buf):
            break
        cur_offset = next_offset

    if width <= 0 or height <= 0 or top_offset >= len(buf) or bottom_offset >= len(buf):
        return None, 0, 0, 0, 0

    rgba_colors = []
    for i in range(4):
        pal_idx = color_map[i]
        r, g, b = palette_rgbs[pal_idx] if pal_idx < len(palette_rgbs) else (255, 255, 255)
        a = alpha_map[i]
        rgba_colors.append((r, g, b, a))

    pixels = bytearray(width * height * 4)

    class _BitReader:
        def __init__(self, buffer: bytes, offset: int) -> None:
            self.buf = buffer
            self.byte_pos = offset
            self.bit_pos = 0

        def read_bits(self, n: int) -> int:
            val = 0
            for _ in range(n):
                if self.byte_pos >= len(self.buf):
                    return val
                b = self.buf[self.byte_pos]
                bit = (b >> (7 - self.bit_pos)) & 1
                val = (val << 1) | bit
                self.bit_pos += 1
                if self.bit_pos == 8:
                    self.bit_pos = 0
                    self.byte_pos += 1
            return val

        def byte_align(self) -> None:
            if self.bit_pos != 0:
                self.bit_pos = 0
                self.byte_pos += 1

    def _decode_field(reader: _BitReader, start_y: int, step_y: int) -> None:
        for y in range(start_y, height, step_y):
            line_pixels = 0
            while line_pixels < width:
                v = reader.read_bits(4)
                if (v & 0x0C) != 0:
                    length = v >> 2
                    color = v & 0x03
                else:
                    v = (v << 4) | reader.read_bits(4)
                    if (v & 0xF0) != 0:
                        length = v >> 2
                        color = v & 0x03
                    else:
                        v = (v << 4) | reader.read_bits(4)
                        if (v & 0xFC) != 0:
                            length = v >> 2
                            color = v & 0x03
                        else:
                            v = (v << 4) | reader.read_bits(4)
                            length = v >> 2
                            color = v & 0x03
                            if length == 0:
                                length = width - line_pixels

                length = min(length, width - line_pixels)
                c_rgba = rgba_colors[color]
                for x in range(line_pixels, line_pixels + length):
                    idx = (y * width + x) * 4
                    pixels[idx] = c_rgba[0]
                    pixels[idx + 1] = c_rgba[1]
                    pixels[idx + 2] = c_rgba[2]
                    pixels[idx + 3] = c_rgba[3]
                line_pixels += length
            reader.byte_align()

    try:
        _decode_field(_BitReader(buf, top_offset), 0, 2)
        _decode_field(_BitReader(buf, bottom_offset), 1, 2)
    except Exception:
        return None, 0, 0, 0, 0

    img = Image.frombytes("RGBA", (width, height), bytes(pixels))
    bbox = img.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        return cropped, crop_x + bbox[0], crop_y + bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
    return img, crop_x, crop_y, width, height


def extract_spu_from_sub(sub_path: str | Path, filepos: int) -> bytes:
    """Extract and reassemble DVD SPU packet bytes from a .sub file at filepos."""
    p = Path(sub_path)
    if not p.is_file():
        raise FileNotFoundError(f"VobSub .sub file not found: {p}")

    with p.open("rb") as f:
        f.seek(filepos)
        header = f.read(2048)
        if not header:
            return b""

        # Check for MPEG-2 PS pack or PES packet
        if header.startswith(b"\x00\x00\x01\xBA") or header.startswith(b"\x00\x00\x01\xBD"):
            pos = filepos
            chunks = bytearray()
            spu_size: int | None = None

            while True:
                f.seek(pos)
                blk = f.read(65536)
                if not blk:
                    break

                idx = 0
                if blk[idx : idx + 4] == b"\x00\x00\x01\xBA":
                    if idx + 14 > len(blk):
                        break
                    stuffing = blk[idx + 13] & 0x07
                    idx += 14 + stuffing

                if idx + 9 > len(blk) or blk[idx : idx + 4] != b"\x00\x00\x01\xBD":
                    break

                pes_len, = struct.unpack_from(">H", blk, idx + 4)
                if pes_len == 0:
                    break
                header_len = blk[idx + 8]
                payload_start = idx + 9 + header_len
                payload_data = blk[payload_start + 1 : idx + 6 + pes_len]
                chunks.extend(payload_data)

                if spu_size is None and len(chunks) >= 2:
                    spu_size, = struct.unpack_from(">H", chunks, 0)

                if spu_size is not None and len(chunks) >= spu_size:
                    return bytes(chunks[:spu_size])

                pos += idx + 6 + pes_len

            return bytes(chunks)

        # Raw SPU fallback
        f.seek(filepos)
        hdr = f.read(4)
        if len(hdr) < 4:
            return b""
        total_size, = struct.unpack_from(">H", hdr, 0)
        f.seek(filepos)
        return f.read(total_size)


def extract_vobsub_events(
    idx_path: str | Path,
    sub_path: str | Path | None = None,
    *,
    language: str = "eng",
    cancellation_token: CancellationToken | None = None,
) -> list[VobSubEvent]:
    """Extract VobSub events with bounding-box cropped images.

    Invariants:
    - Pairs .idx with .sub; raises FileNotFoundError if .sub is missing.
    - Decodes SPU packets into cropped RGBA images.
    - Skips corrupt SPU packets gracefully without crashing the whole stream.
    """
    idx_p = Path(idx_path).resolve()
    if not idx_p.is_file():
        raise FileNotFoundError(f"VobSub .idx file not found: {idx_p}")

    sub_p = Path(sub_path).resolve() if sub_path else idx_p.with_suffix(".sub")
    if not sub_p.is_file():
        raise FileNotFoundError(f"VobSub pairing failed: matching .sub file not found for {idx_p}")

    meta = parse_vobsub_idx(idx_p)
    tracks: list[VobSubTrackInfo] = meta.get("tracks", [])

    raw_palette = meta.get("palette", [])
    palette_rgbs: list[tuple[int, int, int]] = []
    for c in raw_palette:
        c_clean = c.strip().lstrip("#")
        if len(c_clean) >= 6:
            r = int(c_clean[0:2], 16)
            g = int(c_clean[2:4], 16)
            b = int(c_clean[4:6], 16)
            palette_rgbs.append((r, g, b))
        else:
            palette_rgbs.append((255, 255, 255))

    selected_track = next((t for t in tracks if t.language == language or t.language == language[:2]), None)
    if not selected_track and tracks:
        selected_track = tracks[0]

    if not selected_track or not selected_track.timestamps:
        return []

    events: list[VobSubEvent] = []
    ts_list = selected_track.timestamps

    for idx, (ts_ms, filepos) in enumerate(ts_list):
        if cancellation_token:
            cancellation_token.check_cancelled()

        if idx + 1 < len(ts_list):
            next_ts = ts_list[idx + 1][0]
            duration = min(next_ts - ts_ms, 5000)
            end_ms = ts_ms + max(500, duration)
        else:
            end_ms = ts_ms + 3000

        spu_bytes = extract_spu_from_sub(sub_p, filepos)
        img, crop_x, crop_y, w, h = parse_spu_packet(spu_bytes, palette_rgbs)

        if img is None:
            # Skip unparseable packet safely
            continue

        events.append(
            VobSubEvent(
                start_ms=ts_ms,
                end_ms=end_ms,
                image=img,
                x=crop_x,
                y=crop_y,
                width=w,
                height=h,
                filepos=filepos,
            )
        )

    return events


def create_synthetic_vobsub(
    idx_path: str | Path,
    sub_path: str | Path,
    start_ms: int = 1000,
    width: int = 60,
    height: int = 30,
    x_pos: int = 100,
    y_pos: int = 200,
) -> None:
    """Generate a minimal valid synthetic VobSub .idx and .sub pair for testing."""
    top_lines = (height + 1) // 2
    bottom_lines = height // 2
    top_rle = b"\x00\x01" * top_lines
    bottom_rle = b"\x00\x01" * bottom_lines
    top_offset = 4
    bottom_offset = top_offset + len(top_rle)
    dcsq_offset = bottom_offset + len(bottom_rle)

    x_start = x_pos
    x_end = x_pos + width - 1
    y_start = y_pos
    y_end = y_pos + height - 1

    darea = bytearray(6)
    darea[0] = (x_start >> 4) & 0xFF
    darea[1] = ((x_start & 0x0F) << 4) | ((x_end >> 8) & 0x0F)
    darea[2] = x_end & 0xFF
    darea[3] = (y_start >> 4) & 0xFF
    darea[4] = ((y_start & 0x0F) << 4) | ((y_end >> 8) & 0x0F)
    darea[5] = y_end & 0xFF

    dspxa = struct.pack(">HH", top_offset, bottom_offset)

    dcsq = bytearray()
    dcsq.extend(struct.pack(">HH", 0, dcsq_offset))
    dcsq.append(0x01)  # STA_DSP
    dcsq.extend(b"\x03\x32\x10")  # SET_COLOR
    dcsq.extend(b"\x04\xff\xf0")  # SET_CONTR
    dcsq.append(0x05)  # SET_DAREA
    dcsq.extend(darea)
    dcsq.append(0x06)  # SET_DSPXA
    dcsq.extend(dspxa)
    dcsq.append(0xFF)  # CMD_END

    total_size = 4 + len(top_rle) + len(bottom_rle) + len(dcsq)
    spu = bytearray()
    spu.extend(struct.pack(">HH", total_size, dcsq_offset))
    spu.extend(top_rle)
    spu.extend(bottom_rle)
    spu.extend(dcsq)

    p_idx = Path(idx_path)
    p_sub = Path(sub_path)

    sec = start_ms // 1000
    ms = start_ms % 1000
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    ts_str = f"{hh:02d}:{mm:02d}:{ss:02d}:{ms:03d}"

    idx_text = f"""# VobSub index file
size: 720x480
palette: 000000, ffffff, 101010, 808080
id: en, index: 0
timestamp: {ts_str}, filepos: 000000000
"""
    p_idx.write_text(idx_text, encoding="utf-8")
    p_sub.write_bytes(bytes(spu))
