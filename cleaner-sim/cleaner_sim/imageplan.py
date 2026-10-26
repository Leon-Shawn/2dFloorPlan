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
from .rasterize import FineMask, _pad_mask

GRAY_MAGIC = b"CLEANERGRAY1"

# 自动阈值重试：Otsu 对"深色主墙线 + 浅灰细节线（窗线等）"的图容易取低，
# 导致外轮廓不闭合、洪泛泄漏。识别失败时按此阶梯向上试探。
_AUTO_THRESHOLD_STEPS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
_THRESHOLD_CAP = 250          # 255 会把整图判成墙，留出余量


def _parse_quality_ok(diag: dict) -> bool:
    """判断一次解析是否"像一张正常户型图"：室内占比合理且墙体不过量。"""
    return (0.15 <= diag["free_fraction"] <= 0.70
            and diag["wall_fraction"] <= 0.5)


def _crop_mask(mask: bytearray, width: int, height: int, pad: int) -> bytearray:
    """Inverse of `rasterize._pad_mask`: strip a `pad`-wide border again."""
    if pad <= 0:
        return mask
    nw = width + 2 * pad
    out = bytearray(width * height)
    for y in range(height):
        src = (y + pad) * nw + pad
        out[y * width:(y + 1) * width] = mask[src:src + width]
    return out


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


def _parse_at_threshold(gray, width: int, height: int,
                        params: ImageParseParams, threshold: int) -> tuple[FineMask, dict]:
    """One full parse pass at a fixed threshold level."""
    # 1. threshold -> walls
    wall = bytearray(width * height)
    for i, v in enumerate(gray):
        dark = v <= threshold
        wall[i] = 1 if (dark != params.invert) else 0
    wall_fraction = G.count(wall) / (width * height)

    # 2. close the wall outline so the flood fill cannot leak through pinholes.
    #    Pad the mask first: `erode_square` treats the map border as outside, so
    #    without padding a wall running along the image edge would be erased
    #    entirely — re-opening exactly the gap the closing is meant to seal.
    if params.seal_px > 0:
        r = params.seal_px
        padded, pw, ph = _pad_mask(wall, width, height, r)
        closed = G.dilate_square(padded, pw, ph, r)
        closed = G.erode_square(closed, pw, ph, r)
        wall = _crop_mask(closed, width, height, r)

    # 3+4. outside the building vs inside it
    outside = _flood_outside(wall, width, height)
    free = bytearray(1 if (not wall[i] and not outside[i]) else 0
                     for i in range(width * height))

    warnings = []
    free_fraction = sum(free) / (width * height)
    if wall_fraction > 0.5:
        if params.invert:
            warnings.append("已勾选「墙体为浅色」，但识别出的墙体超过一半："
                            "这张图很可能是白底深色墙线的常规户型图——"
                            "请「取消勾选」后重新解析")
        else:
            warnings.append("识别出的墙体超过一半：原图可能是深底浅色墙线——"
                            "请勾选「墙体为浅色」，或检查阈值")
    if free_fraction > 0.7:
        warnings.append("外轮廓疑似有缺口，墙外区域被误判为室内："
                        "请增大「封口半径」或调高阈值")
    if free_fraction < 0.02:
        if params.invert:
            warnings.append("几乎没有识别到室内区域（且已勾选「墙体为浅色」）："
                            "若原图是白底深色墙线，请先「取消勾选」；否则请调整阈值")
        else:
            warnings.append("几乎没有识别到室内区域：墙线在窗户/门洞处可能有缺口——"
                            "请调高阈值或增大「封口半径」")

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
        if params.invert:
            raise ValueError(
                "未能在图中找到封闭的户型区域（已勾选「墙体为浅色」）："
                "若原图是白底深色墙线，请「取消勾选」后重试；否则请调整阈值")
        raise ValueError(
            "未能在图中找到封闭的户型区域：请调低/调高阈值，"
            "或勾选「墙体为浅色」（仅当原图为深底浅色墙线时）")

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
        "threshold_auto_adjusted": False,
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


def parse_floorplan_image(gray, width: int, height: int,
                          params: ImageParseParams) -> tuple[FineMask, dict]:
    """Grayscale floorplan image -> (FineMask, diagnostics).

    自动阈值（params.threshold=None）分两步：
      1) 先用 Otsu 跑一遍；
      2) 若结果不像一张正常户型图（室内占比 <15% 或 >70%，或墙体超过一半），
         通常是墙线细节（如浅灰色的窗线）落在阈值之上、外轮廓未闭合导致的
         洪泛泄漏——按阶梯向上试探候选阈值，取第一个"像正常户型图"的结果，
         并在 diagnostics 中标记 threshold_auto_adjusted。
    手动指定阈值时不重试，尊重用户选择。
    """
    if width < 8 or height < 8:
        raise ValueError("image too small to parse")
    if len(gray) != width * height:
        raise ValueError("grayscale buffer size does not match width*height")

    if params.threshold is not None:
        return _parse_at_threshold(gray, width, height, params,
                                   max(0, min(255, int(params.threshold))))

    base = max(0, min(255, int(otsu_threshold(gray))))
    base_error = None
    try:
        fine, diag = _parse_at_threshold(gray, width, height, params, base)
    except ValueError as exc:
        fine, diag, base_error = None, None, exc

    if fine is not None and _parse_quality_ok(diag):
        return fine, diag

    candidates = sorted({min(_THRESHOLD_CAP, round(base + (_THRESHOLD_CAP - base) * f))
                         for f in _AUTO_THRESHOLD_STEPS})
    for t in candidates:
        if t <= base:
            continue
        try:
            cand_fine, cand_diag = _parse_at_threshold(gray, width, height, params, t)
        except ValueError:
            continue
        if _parse_quality_ok(cand_diag):
            cand_diag["threshold_auto_adjusted"] = True
            cand_diag["threshold_auto_base"] = base
            cand_diag["warnings"].insert(
                0, f"自动阈值已从 {base} 优化为 {t}"
                   f"（原阈值下墙线轮廓未闭合，外部区域会泄漏进室内）")
            return cand_fine, cand_diag

    if fine is None:
        raise base_error
    return fine, diag


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
