"""Pure Python Blu-ray PGS SUP subtitle parser and segment decoder with bounding box cropping."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct
from typing import Any

from PIL import Image

from .models import PgsSubtitleEvent

SEGMENT_PDS = 0x14
SEGMENT_ODS = 0x15
SEGMENT_PCS = 0x16
SEGMENT_WDS = 0x17
SEGMENT_END = 0x80


def ycbcr_to_rgba(y: int, cr: int, cb: int, alpha: int) -> tuple[int, int, int, int]:
    """Convert YCbCr and Alpha to standard sRGB RGBA tuple."""
    if alpha == 0:
        return 0, 0, 0, 0
    Y = float(y)
    Cr = float(cr) - 128.0
    Cb = float(cb) - 128.0
    r = int(min(255, max(0, round(Y + 1.402 * Cr))))
    g = int(min(255, max(0, round(Y - 0.344136 * Cb - 0.714136 * Cr))))
    b = int(min(255, max(0, round(Y + 1.772 * Cb))))
    return r, g, b, alpha


def decode_pgs_rle(rle_data: bytes, width: int, height: int) -> bytearray:
    """Decode Blu-ray PGS run-length encoded bitmap into indexed 8-bit bytearray."""
    total_pixels = width * height
    pixels = bytearray(total_pixels)
    idx = 0
    pos = 0
    n = len(rle_data)

    while pos < n and idx < total_pixels:
        b = rle_data[pos]
        pos += 1
        if b != 0:
            pixels[idx] = b
            idx += 1
        else:
            if pos >= n:
                break
            b2 = rle_data[pos]
            pos += 1
            if b2 == 0:
                # End of scanline marker
                line_rem = width - (idx % width)
                if line_rem != width and line_rem > 0:
                    idx += line_rem
            else:
                flag = b2 & 0xC0
                if flag == 0x00:
                    run = b2 & 0x3F
                    color = 0
                elif flag == 0x40:
                    if pos >= n:
                        break
                    b3 = rle_data[pos]
                    pos += 1
                    run = ((b2 & 0x3F) << 8) | b3
                    color = 0
                elif flag == 0x80:
                    if pos >= n:
                        break
                    b3 = rle_data[pos]
                    pos += 1
                    run = b2 & 0x3F
                    color = b3
                else:  # flag == 0xC0
                    if pos + 1 >= n:
                        break
                    b3 = rle_data[pos]
                    b4 = rle_data[pos + 1]
                    pos += 2
                    run = ((b2 & 0x3F) << 8) | b3
                    color = b4

                limit = min(run, total_pixels - idx)
                if color != 0:
                    pixels[idx : idx + limit] = bytes([color]) * limit
                idx += limit

    return pixels


@dataclass
class _PgsObjectEntry:
    object_id: int
    window_id: int
    is_forced: bool
    x: int
    y: int


@dataclass
class _PgsComposition:
    pts: int
    width: int
    height: int
    composition_number: int
    composition_state: int
    palette_id: int
    objects: list[_PgsObjectEntry] = field(default_factory=list)


def parse_pgs_sup(data_or_path: bytes | str | Path) -> list[PgsSubtitleEvent]:
    """Parse a Blu-ray PGS SUP stream into a list of PgsSubtitleEvents with cropped bounding boxes.
    
    Invariants:
    - Handles malformed or truncated binary data safely.
    - Decodes PDS palette and ODS bitmap runs.
    - Crops rendered subpictures to non-transparent bounding boxes.
    """
    if isinstance(data_or_path, (str, Path)):
        p = Path(data_or_path)
        if not p.is_file():
            raise FileNotFoundError(f"PGS SUP file not found: {p}")
        data = p.read_bytes()
    else:
        data = data_or_path

    events: list[PgsSubtitleEvent] = []
    offset = 0
    n = len(data)

    current_palettes: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    current_objects: dict[int, tuple[int, int, bytes]] = {}  # id -> (w, h, rle_data)
    pending_event: PgsSubtitleEvent | None = None
    last_composition: _PgsComposition | None = None

    while offset + 13 <= n:
        if data[offset : offset + 2] != b"PG":
            next_pg = data.find(b"PG", offset + 1)
            if next_pg == -1:
                break
            offset = next_pg
            continue

        try:
            pts, dts = struct.unpack_from(">II", data, offset + 2)
            segment_type = data[offset + 10]
            seg_len, = struct.unpack_from(">H", data, offset + 11)
        except struct.error:
            break

        offset += 13
        if offset + seg_len > n:
            # Truncated segment: stop safely
            break

        seg_data = data[offset : offset + seg_len]
        offset += seg_len
        pts_ms = int(round(pts / 90.0))

        if segment_type == SEGMENT_PDS:
            if len(seg_data) >= 2:
                pal_id = seg_data[0]
                entries = current_palettes.setdefault(pal_id, {})
                idx = 2
                while idx + 5 <= len(seg_data):
                    entry_id, y, cr, cb, a = seg_data[idx : idx + 5]
                    entries[entry_id] = ycbcr_to_rgba(y, cr, cb, a)
                    idx += 5

        elif segment_type == SEGMENT_ODS:
            if len(seg_data) >= 11:
                try:
                    obj_id, obj_ver, seq_flag = struct.unpack_from(">HBB", seg_data, 0)
                    w, h = struct.unpack_from(">HH", seg_data, 7)
                    rle_chunk = seg_data[11:]
                    if seq_flag & 0x80:  # First sequence
                        current_objects[obj_id] = (w, h, rle_chunk)
                    else:
                        if obj_id in current_objects:
                            prev_w, prev_h, prev_data = current_objects[obj_id]
                            current_objects[obj_id] = (prev_w, prev_h, prev_data + rle_chunk)
                except struct.error:
                    continue

        elif segment_type == SEGMENT_PCS:
            if len(seg_data) >= 11:
                try:
                    w, h, fps, comp_num, comp_state, pal_upd, pal_id, num_objs = struct.unpack_from(
                        ">HHBHBBBB", seg_data, 0
                    )
                    comp = _PgsComposition(
                        pts=pts,
                        width=w,
                        height=h,
                        composition_number=comp_num,
                        composition_state=comp_state,
                        palette_id=pal_id,
                    )
                    idx = 11
                    for _ in range(num_objs):
                        if idx + 8 <= len(seg_data):
                            o_id, win_id, flags, ox, oy = struct.unpack_from(">HBBHH", seg_data, idx)
                            is_forced = bool(flags & 0x40)
                            is_cropped = bool(flags & 0x80)
                            idx += 8
                            if is_cropped:
                                idx += 8  # skip crop coordinates
                            comp.objects.append(
                                _PgsObjectEntry(
                                    object_id=o_id,
                                    window_id=win_id,
                                    is_forced=is_forced,
                                    x=ox,
                                    y=oy,
                                )
                            )
                    last_composition = comp

                    # If composition clears the screen (num_objs == 0)
                    if num_objs == 0:
                        if pending_event is not None:
                            pending_event.end_ms = max(pending_event.start_ms + 1, pts_ms)
                            events.append(pending_event)
                            pending_event = None
                    else:
                        if pending_event is not None:
                            pending_event.end_ms = max(pending_event.start_ms + 1, pts_ms)
                            events.append(pending_event)
                            pending_event = None
                except struct.error:
                    continue

        elif segment_type == SEGMENT_END:
            if last_composition and last_composition.objects:
                pts_ms = int(round(last_composition.pts / 90.0))
                for obj_entry in last_composition.objects:
                    if obj_entry.object_id in current_objects:
                        obj_w, obj_h, rle_bytes = current_objects[obj_entry.object_id]
                        if obj_w <= 0 or obj_h <= 0:
                            continue
                        pal = current_palettes.get(last_composition.palette_id, {})

                        pixel_indices = decode_pgs_rle(rle_bytes, obj_w, obj_h)
                        rgba_bytes = bytearray(obj_w * obj_h * 4)
                        for p_idx, color_idx in enumerate(pixel_indices):
                            r, g, b, a = pal.get(color_idx, (0, 0, 0, 0))
                            base_offset = p_idx * 4
                            rgba_bytes[base_offset] = r
                            rgba_bytes[base_offset + 1] = g
                            rgba_bytes[base_offset + 2] = b
                            rgba_bytes[base_offset + 3] = a

                        img = Image.frombytes("RGBA", (obj_w, obj_h), bytes(rgba_bytes))
                        bbox = img.getbbox()
                        if bbox:
                            cropped_img = img.crop(bbox)
                            crop_x = obj_entry.x + bbox[0]
                            crop_y = obj_entry.y + bbox[1]
                            crop_w = bbox[2] - bbox[0]
                            crop_h = bbox[3] - bbox[1]
                        else:
                            cropped_img = img
                            crop_x = obj_entry.x
                            crop_y = obj_entry.y
                            crop_w = obj_w
                            crop_h = obj_h

                        pending_event = PgsSubtitleEvent(
                            start_ms=pts_ms,
                            end_ms=pts_ms + 3000,
                            image=cropped_img,
                            x=crop_x,
                            y=crop_y,
                            width=crop_w,
                            height=crop_h,
                            composition_number=last_composition.composition_number,
                            is_forced=obj_entry.is_forced,
                        )
                        break

    if pending_event is not None:
        events.append(pending_event)

    return events


def create_minimal_pgs_sup(
    start_ms: int = 1000,
    end_ms: int = 3000,
    width: int = 1920,
    height: int = 1080,
    sub_w: int = 100,
    sub_h: int = 40,
    sub_x: int = 910,
    sub_y: int = 900,
    is_forced: bool = False,
) -> bytes:
    """Generate a minimal valid Blu-ray PGS SUP binary stream for testing."""
    start_pts = int(round(start_ms * 90.0))
    end_pts = int(round(end_ms * 90.0))

    out = bytearray()

    def _write_segment(pts: int, seg_type: int, payload: bytes) -> None:
        out.extend(b"PG")
        out.extend(struct.pack(">IIBH", pts, pts, seg_type, len(payload)))
        out.extend(payload)

    # 1. SHOW Display Set
    flag_byte = 0x40 if is_forced else 0x00
    pcs_payload = struct.pack(
        ">HHBHBBBBHBBHH",
        width,
        height,
        0x20,  # 24fps
        0,     # comp_num
        0x80,  # Epoch start
        0,     # pal_upd
        0,     # pal_id
        1,     # 1 object
        0,     # obj_id
        0,     # win_id
        flag_byte,
        sub_x,
        sub_y,
    )
    _write_segment(start_pts, SEGMENT_PCS, pcs_payload)

    # PDS (0x14)
    # Entry 0: transparent, Entry 1: White opaque (Y=235, Cr=128, Cb=128, A=255)
    pds_payload = bytearray([0, 0])
    pds_payload.extend([0, 0, 128, 128, 0])
    pds_payload.extend([1, 235, 128, 128, 255])
    _write_segment(start_pts, SEGMENT_PDS, bytes(pds_payload))

    # ODS (0x15)
    rle = bytearray()
    for _ in range(sub_h):
        rem = sub_w
        while rem > 0:
            chunk = min(rem, 63)
            rle.extend([0x00, 0x80 | chunk, 0x01])
            rem -= chunk
        rle.extend([0x00, 0x00])

    data_len = len(rle) + 4
    ods_header = struct.pack(
        ">HBBBBBHH",
        0,     # obj_id
        0,     # obj_ver
        0xC0,  # first & last
        (data_len >> 16) & 0xFF,
        (data_len >> 8) & 0xFF,
        data_len & 0xFF,
        sub_w,
        sub_h,
    )
    _write_segment(start_pts, SEGMENT_ODS, ods_header + bytes(rle))
    _write_segment(start_pts, SEGMENT_END, b"")

    # 2. CLEAR Display Set
    clear_pcs = struct.pack(
        ">HHBHBBBB",
        width,
        height,
        0x20,
        1,     # comp_num
        0x00,  # normal
        0,     # pal_upd
        0,     # pal_id
        0,     # 0 objects = clear
    )
    _write_segment(end_pts, SEGMENT_PCS, clear_pcs)
    _write_segment(end_pts, SEGMENT_END, b"")

    return bytes(out)
