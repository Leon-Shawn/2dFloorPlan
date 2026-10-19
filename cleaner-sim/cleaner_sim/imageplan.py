"""Uploaded floorplan image -> FineMask (stage 1 of the module-1 pipeline).

A real-estate floorplan image is a picture: thick dark walls, light rooms, and a
light page background that is *not* part of the flat. Turning that into a walkable
grid is a small pipeline of its own:

    1. threshold      Otsu (automatic) or a user-supplied level -> wall mask
    2. seal gaps      morphological closing, so a hairline break in the wall outline
                      does not leak the flood fill into the whole page
    3. find outside   flood fill from the image border through non-wall pixels
    4. free space     = not wall and not outside  (this is what makes the page
                      background disappear and leaves only the flat's interior)
    5. drop islands   remove free specks (noise, text, dimension arrows)
    6. crop + scale   crop to the floor's bounding box and convert to metres using
                      the real-world width the user provides (an image has no scale)

The browser decodes the uploaded PNG/JPEG (Canvas has full image support) and posts
a downsampled 8-bit grayscale buffer, so the Python side needs no image codec — but
every geometric decision above still happens here, in Python.
"""
from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass, field

from . import grid as G
from .rasterize import FineMask

GRAY_MAGIC = b"CLEANERGRAY1"


# ------------------------------------------------------------------- thresholding
def otsu_threshold(gray) -> int:
    """Otsu's method: the level maximising between-class variance."""
    hist = [0] * 256
    for v in gray:
        hist[v] += 1
    total = len(gray)
    if total == 0:
        return 128
    sum_all = sum(i * h for i, h in enumerate(hist))
    sum_b = 0.0
    w_b = 0
    best_var = -1.0
    best_t = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return best_t


@dataclass
class ImageParseParams:
    threshold: int | None = None      # None -> Otsu
    invert: bool = False              # True when walls are drawn light on dark
    seal_px: int = 2                  # closing radius that seals outline breaks
    min_island_px: int = 40           # drop free components smaller than this
    real_width_m: float = 10.0        # known real-world size of the plan
    real_height_m: float | None = None
    scale_axis: str = "x"             # which axis real_width_m measures


def _flood_outside(wall: bytearray, width: int, height: int) -> bytearray:
    """Mark every non-wall pixel reachable from the image border (4-connected)."""
    outside = bytearray(width * height)
    q = deque()

    def seed(idx):
        if not wall[idx] and not outside[idx]:
            outside[idx] = 1
            q.append(idx)

    for x in range(width):
        seed(x)
        seed((height - 1) * width + x)
    for y in range(height):
        seed(y * width)
        seed(y * width + width - 1)

    while q:
        idx = q.popleft()
        y, x = divmod(idx, width)
        if y > 0:
            n = idx - width
            if not wall[n] and not outside[n]:
                outside[n] = 1
                q.append(n)
        if y + 1 < height:
            n = idx + width
            if not wall[n] and not outside[n]:
                outside[n] = 1
                q.append(n)
        if x > 0:
            n = idx - 1
            if not wall[n] and not outside[n]:
                outside[n] = 1
                q.append(n)
        if x + 1 < width:
            n = idx + 1
            if not wall[n] and not outside[n]:
                outside[n] = 1
                q.append(n)
    return outside


def parse_floorplan_image(gray, width: int, height: int,
                          params: ImageParseParams) -> tuple[FineMask, dict]:
    """Grayscale floorplan image -> (FineMask, diagnostics)."""
    if width < 8 or height < 8:
        raise ValueError("image too small to parse")
    if len(gray) != width * height:
        raise ValueError("grayscale buffer size does not match width*height")

    threshold = params.threshold if params.threshold is not None else otsu_threshold(gray)
    threshold = max(0, min(255, int(threshold)))

    # 1. threshold -> walls
    wall = bytearray(width * height)
    for i, v in enumerate(gray):
        dark = v <= threshold
        wall[i] = 1 if (dark != params.invert) else 0
    wall_fraction = G.count(wall) / (width * height)

    # 2. close the wall outline so the flood fill cannot leak through pinholes
    if params.seal_px > 0:
        d = G.dilate_square(wall, width, height, params.seal_px)
        wall = G.erode_square(d, width, height, params.seal_px)

    # 3+4. outside the building vs inside it
    outside = _flood_outside(wall, width, height)
    free = bytearray(1 if (not wall[i] and not outside[i]) else 0
                     for i in range(width * height))

    warnings = []
    free_fraction = sum(free) / (width * height)
    if free_fraction > 0.7:
        warnings.append("外轮廓疑似有缺口，墙外区域被误判为室内：请增大「封口半径」或调整阈值")
    if free_fraction < 0.02:
        warnings.append("几乎没有识别到室内区域：请检查阈值，或勾选「墙体为浅色」")

    # 5. drop specks (text, dimension arrows, dust) and keep the real floor
    labels, sizes = G.label_components(free, width, height, 4)
    kept = {lab for lab, size in sizes.items() if size >= params.min_island_px}
    if not kept and sizes:
        kept = {max(sizes.items(), key=lambda kv: kv[1])[0]}
    if kept != set(sizes):
        free = bytearray(1 if (labels[i] and labels[i] in kept) else 0
                         for i in range(width * height))

    # 6. crop to the floor and convert pixels to metres
    xs = [i % width for i, v in enumerate(free) if v]
    ys = [i // width for i, v in enumerate(free) if v]
    if not xs:
        raise ValueError("未能在图中找到封闭的户型区域：请调低/调高阈值，或勾选「墙体为浅色」")

    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    cropped = bytearray(cw * ch)
    for y in range(ch):
        src = (y + y0) * width + x0
        cropped[y * cw:(y + 1) * cw] = free[src:src + cw]

    if params.scale_axis == "y":
        ref_px = ch
        ref_m = params.real_height_m or params.real_width_m or 10.0
    else:
        ref_px = cw
        ref_m = params.real_width_m or params.real_height_m or 10.0
    meters_per_px = ref_m / max(1, ref_px)

    meta = {
        "id": "",
        "room_num": 0,
        "size_x_m": round(cw * meters_per_px, 2),
        "size_y_m": round(ch * meters_per_px, 2),
        "area_m2": round(cw * meters_per_px * ch * meters_per_px, 2),
        "n_verts": 0,
        "categories": [],
        "tier": "custom",
        "kind": "image",
    }
    fine = FineMask(inside=cropped, width=cw, height=ch,
                    meters_per_px=meters_per_px, label="upload", kind="image",
                    meta=meta)

    diagnostics = {
        "threshold": threshold,
        "threshold_auto": params.threshold is None,
        "invert": params.invert,
        "seal_px": params.seal_px,
        "image_px": [width, height],
        "crop_px": [cw, ch],
        "crop_box": [x0, y0, x1, y1],
        "wall_fraction": round(wall_fraction, 4),
        "free_px": sum(cropped),
        "dropped_islands": max(0, len(sizes) - len(kept)),
        "meters_per_px": round(meters_per_px, 5),
        "floor_area_m2": meta["area_m2"],
        "free_fraction": round(free_fraction, 4),
        "warnings": warnings,
    }
    return fine, diagnostics


# ------------------------------------------------------- grayscale cache on disk
def save_gray(path, gray, width: int, height: int) -> None:
    """Persist the uploaded grayscale so the user can re-parse with new settings."""
    with open(path, "wb") as fh:
        fh.write(GRAY_MAGIC)
        fh.write(struct.pack("<II", width, height))
        fh.write(bytes(gray))


def load_gray(path) -> tuple[bytes, int, int]:
    with open(path, "rb") as fh:
        magic = fh.read(len(GRAY_MAGIC))
        if magic != GRAY_MAGIC:
            raise ValueError("not a cached grayscale upload")
        width, height = struct.unpack("<II", fh.read(8))
        return fh.read(width * height), width, height
