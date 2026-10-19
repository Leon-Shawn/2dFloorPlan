"""Module 1 — floorplan parsing and rasterisation pipeline.

Stages (each independently testable):

    1. source       a HouseExpo outline polygon, or an uploaded raster image, is
                    reduced to a *fine* binary "inside the house" mask plus its
                    physical scale (see `FineMask`, `imageplan.py`)
    2. rasterise    polygon -> fine mask, via an active-edge-table even-odd scanline
                    fill at a supersampled resolution
    3. downsample   fine mask -> occupancy grid at the requested cell size, using a
                    wall-fraction threshold so thin walls survive but door openings
                    stay open
    4. analyse      connectivity of the floor, dead-zone detection
    5. robot model  erode the floor by the robot's square footprint to get legal
                    robot positions ("center_ok"), pick the reachable region, and
                    work out which floor cells can actually be cleaned

The result of stages 2-5 is `Occupancy` + `RobotView`, which the planner,
simulator and dashboard all consume. Because stage 1 is pluggable, an uploaded
floorplan image goes through exactly the same stages 3-5 as the built-in dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import grid as G
from .houseexpo import Floorplan

# ------------------------------------------------------------------ stage 2: fill
def _edge_table(points):
    edges = []
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if y1 == y2:
            continue
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        edges.append((y1, y2, x1, (x2 - x1) / (y2 - y1)))
    edges.sort(key=lambda e: e[0])
    return edges


def fill_polygon(target: bytearray, width: int, height: int, points, value: int = 1) -> None:
    """Even-odd scanline polygon fill using an active edge table."""
    if len(points) < 3:
        return
    edges = _edge_table(points)
    if not edges:
        return

    y_start = max(0, int(edges[0][0]))
    y_end = min(height - 1, int(max(e[1] for e in edges)))
    active: list = []
    ei = 0
    fill = bytes([value])

    for y in range(y_start, y_end + 1):
        yc = y + 0.5
        while ei < len(edges) and edges[ei][0] <= yc:
            active.append(edges[ei])
            ei += 1
        if active:
            active = [e for e in active if e[1] > yc]
        if not active:
            continue

        xs = sorted(e[2] + (yc - e[0]) * e[3] for e in active)
        row = y * width
        for i in range(0, len(xs) - 1, 2):
            xa = int(round(xs[i]))
            xb = int(round(xs[i + 1]))
            if xb < 0 or xa >= width:
                continue
            xa = max(0, xa)
            xb = min(width - 1, xb)
            if xb >= xa:
                target[row + xa:row + xb + 1] = fill * (xb - xa + 1)


def rasterise_fine(plan: Floorplan, pixels_per_m: float, pad_px: int = 0):
    """Outline polygon -> fine mask where 1 = inside the house (walkable)."""
    verts = plan.verts
    xs = [v[0] * pixels_per_m for v in verts]
    ys = [v[1] * pixels_per_m for v in verts]
    x_min, y_min = min(xs), min(ys)
    width = int(max(xs) - x_min) + 2 * pad_px + 2
    height = int(max(ys) - y_min) + 2 * pad_px + 2

    pts = [(x - x_min + pad_px, y - y_min + pad_px) for x, y in zip(xs, ys)]
    inside = bytearray(width * height)
    fill_polygon(inside, width, height, pts)
    return inside, width, height


# ---------------------------------------------------------------- stage 1: source
@dataclass
class FineMask:
    """A floorplan reduced to a fine binary mask plus its physical scale.

    This is the common currency between the HouseExpo polygon loader and the
    uploaded-image parser: everything downstream only needs the mask, its
    resolution, and how many metres one mask pixel spans.
    """
    inside: bytearray
    width: int
    height: int
    meters_per_px: float
    label: str = "floorplan"
    kind: str = "houseexpo"
    meta: dict = field(default_factory=dict)

    @property
    def size_x_m(self) -> float:
        return round(self.width * self.meters_per_px, 2)

    @property
    def size_y_m(self) -> float:
        return round(self.height * self.meters_per_px, 2)


def fine_from_plan(plan: Floorplan, pixels_per_m: float, pad_px: int = 0) -> FineMask:
    inside, w, h = rasterise_fine(plan, pixels_per_m, pad_px=pad_px)
    return FineMask(inside=inside, width=w, height=h,
                    meters_per_px=1.0 / pixels_per_m, label=plan.id,
                    kind="houseexpo", meta=plan.meta())


# -------------------------------------------------------------- stage 3: downsample
def downsample_wall(inside_fine: bytearray, fw: int, fh: int, factor: int,
                    wall_threshold: float = 0.5):
    """Fine inside-mask -> coarse wall mask using a wall-fraction threshold.

    A coarse cell becomes a wall when at least `wall_threshold` of its sub-cells
    are wall. Keeping the threshold at 0.5 preserves real walls while leaving
    door openings (typically far wider than one coarse cell) open.
    """
    factor = max(1, int(factor))
    cw, ch = fw // factor, fh // factor
    wall = bytearray(cw * ch)
    sub = factor * factor
    for cy in range(ch):
        fy0 = cy * factor
        for cx in range(cw):
            fx0 = cx * factor
            wall_sub = 0
            for dy in range(factor):
                row = (fy0 + dy) * fw + fx0
                for dx in range(factor):
                    if not inside_fine[row + dx]:
                        wall_sub += 1
            if wall_sub >= wall_threshold * sub:
                wall[cy * cw + cx] = 1
    return wall, cw, ch


def _pad_mask(mask: bytearray, width: int, height: int, pad: int):
    if pad <= 0:
        return mask, width, height
    nw, nh = width + 2 * pad, height + 2 * pad
    out = bytearray(nw * nh)
    for y in range(height):
        out[(y + pad) * nw + pad:(y + pad) * nw + pad + width] = mask[y * width:(y + 1) * width]
    return out, nw, nh


# ------------------------------------------------------------------- data holders
@dataclass
class Occupancy:
    width: int
    height: int
    cell_cm: float
    wall: bytearray
    walkable: bytearray
    source_meta: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def cell_m(self) -> float:
        return self.cell_cm / 100.0

    @property
    def cell_area_m2(self) -> float:
        return self.cell_m * self.cell_m

    def area_m2(self, mask) -> float:
        return round(G.count(mask) * self.cell_area_m2, 3)

    def to_png(self, palette=None) -> bytes:
        palette = palette or {0: (32, 34, 42), 1: (238, 240, 245), 2: (255, 214, 120)}
        idx = [0 if self.wall[i] else 1 for i in range(self.width * self.height)]
        from .png import png_from_palette
        return png_from_palette(self.width, self.height, idx, palette)


@dataclass
class RobotView:
    radius_cells: int
    diameter_cm: float
    center_ok: bytearray
    reachable: bytearray
    start: int
    cleanable: bytearray
    dead: bytearray
    # dead-area split by cause (see build_robot_view)
    all_cleanable: bytearray = field(default_factory=bytearray)
    unreachable: bytearray = field(default_factory=bytearray)
    tight: bytearray = field(default_factory=bytearray)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------- stages 4 and 5
def build_occupancy_from_fine(fine: FineMask, cell_cm: float = 10.0,
                              wall_threshold: float = 0.5, pad_cells: int = 2) -> Occupancy:
    """Fine mask at any resolution -> occupancy grid at the requested cell size."""
    cell_m = cell_cm / 100.0
    factor = max(1, int(round(cell_m / fine.meters_per_px)))
    # An uploaded image has its own pixel pitch, so the achievable cell size is a
    # multiple of it. Report (and use) the true value so all downstream areas and
    # speeds stay consistent with the geometry actually built.
    effective_cell_cm = factor * fine.meters_per_px * 100.0

    padded, fw, fh = _pad_mask(fine.inside, fine.width, fine.height, pad_cells * factor)
    wall, w, h = downsample_wall(padded, fw, fh, factor, wall_threshold)
    walkable = bytearray(1 if not v else 0 for v in wall)

    occ = Occupancy(width=w, height=h, cell_cm=effective_cell_cm, wall=wall,
                    walkable=walkable, source_meta=dict(fine.meta))
    occ.stats.update({
        "grid_w": w,
        "grid_h": h,
        "cell_cm": round(effective_cell_cm, 3),
        "requested_cell_cm": cell_cm,
        "source": fine.kind,
        "source_label": fine.label,
        "source_px": [fine.width, fine.height],
        "downsample_factor": factor,
        "wall_threshold": wall_threshold,
        "walkable_cells": G.count(walkable),
        "walkable_area_m2": occ.area_m2(walkable),
    })
    return occ


def build_occupancy(source, cell_cm: float = 10.0, supersample: int = 4,
                    wall_threshold: float = 0.5, pad_cells: int = 2) -> Occupancy:
    """Floorplan (polygon) or FineMask -> coarse occupancy grid."""
    if isinstance(source, FineMask):
        return build_occupancy_from_fine(source, cell_cm=cell_cm,
                                         wall_threshold=wall_threshold,
                                         pad_cells=pad_cells)
    cell_m = cell_cm / 100.0
    fine = fine_from_plan(source, supersample / cell_m,
                          pad_px=pad_cells * supersample)
    occ = build_occupancy_from_fine(fine, cell_cm=cell_cm,
                                    wall_threshold=wall_threshold, pad_cells=0)
    occ.stats["supersample"] = supersample
    return occ


def analyse_occupancy(occ: Occupancy) -> dict:
    """Stage 4: connectivity / dead-zone statistics of the raw floor."""
    labels, sizes = G.label_components(occ.walkable, occ.width, occ.height, 4)
    ranked = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
    info = {
        "floor_regions": len(sizes),
        "largest_region_cells": ranked[0][1] if ranked else 0,
        "region_sizes": [s for _, s in ranked[:8]],
        "isolated_regions": max(0, len(sizes) - 1),
        "isolated_area_m2": round(sum(s for _, s in ranked[1:]) * occ.cell_area_m2, 3),
    }
    occ.stats.update(info)
    return info


def build_robot_view(occ: Occupancy, robot_cm: float = 34.0) -> RobotView:
    """Stage 5: legal robot positions, reachable region, cleanable vs dead floor.

    The robot is modelled as a square footprint of side ``2 * radius_cells + 1``
    cells: that is the area it occupies (so it must fit without touching a wall)
    and the area it cleans at every position. Square footprints tile the sweep
    bands exactly, which is what lets a plain boustrophedon reach the walls.
    """
    radius_cells = max(1, int(round((robot_cm / 2.0) / occ.cell_cm)))
    center_ok = G.erode_square(occ.walkable, occ.width, occ.height, radius_cells)

    labels, sizes = G.label_components(center_ok, occ.width, occ.height, 4)
    if sizes:
        main = max(sizes.items(), key=lambda kv: kv[1])[0]
        reachable = bytearray(1 if v == main else 0 for v in labels)
    else:
        reachable = bytearray(occ.width * occ.height)

    # (a) everything the robot could clean from *some* legal position
    all_cleanable = G.intersection(
        G.dilate_square(center_ok, occ.width, occ.height, radius_cells), occ.walkable)
    # (b) everything cleanable while staying inside the start region
    cleanable = G.intersection(
        G.dilate_square(reachable, occ.width, occ.height, radius_cells), occ.walkable)

    dead = G.difference(occ.walkable, cleanable)
    # Coverable only from another robot region -> separated floor (needs its own run);
    # not coverable from anywhere -> a genuinely dead corner for this body size.
    unreachable = G.difference(all_cleanable, cleanable)
    tight = G.difference(occ.walkable, all_cleanable)

    big_regions = sorted((s for s in sizes.values() if s >= 4), reverse=True)

    start = G.most_open_cell(reachable, occ.width, occ.height)
    if start is None:
        start = next((i for i, ok in enumerate(reachable) if ok), 0)

    view = RobotView(
        radius_cells=radius_cells,
        diameter_cm=robot_cm,
        center_ok=center_ok,
        reachable=reachable,
        start=start,
        cleanable=cleanable,
        dead=dead,
        all_cleanable=all_cleanable,
        unreachable=unreachable,
        tight=tight,
    )
    view.stats = {
        "robot_diameter_cm": robot_cm,
        "robot_radius_cells": radius_cells,
        "footprint_cells": 2 * radius_cells + 1,
        "footprint_cm": round((2 * radius_cells + 1) * occ.cell_cm, 1),
        "robot_positions": G.count(reachable),
        "robot_regions": len(big_regions),
        "robot_region_sizes": big_regions[:6],
        "cleanable_cells": G.count(cleanable),
        "cleanable_area_m2": occ.area_m2(cleanable),
        "dead_cells": G.count(dead),
        "dead_area_m2": occ.area_m2(dead),
        "dead_ratio": round(G.count(dead) / max(1, G.count(occ.walkable)), 4),
        "separated_cells": G.count(unreachable),
        "separated_area_m2": occ.area_m2(unreachable),
        "tight_cells": G.count(tight),
        "tight_area_m2": occ.area_m2(tight),
    }
    return view


def build_world(source, cell_cm: float = 10.0, robot_cm: float = 34.0,
                supersample: int = 4, wall_threshold: float = 0.5):
    """Convenience wrapper: floorplan source -> (Occupancy, RobotView), fully analysed."""
    occ = build_occupancy(source, cell_cm=cell_cm, supersample=supersample,
                          wall_threshold=wall_threshold)
    analyse_occupancy(occ)
    view = build_robot_view(occ, robot_cm=robot_cm)
    occ.stats.update(view.stats)
    return occ, view


# ------------------------------------------------------------------------ preview
def png_from_masks(wall, width: int, height: int,
                   wall_rgb=(38, 41, 51), floor_rgb=(236, 239, 244)) -> bytes:
    from .png import png_rgb
    rgb = bytearray(width * height * 3)
    for i, is_wall in enumerate(wall):
        r, g, b = wall_rgb if is_wall else floor_rgb
        rgb[i * 3] = r
        rgb[i * 3 + 1] = g
        rgb[i * 3 + 2] = b
    return png_rgb(width, height, rgb)


def thumbnail_from_fine(fine: FineMask, target_w: int = 160) -> bytes:
    """Small RGB preview of a parsed floorplan for the browser's picker."""
    if fine.size_x_m <= 0:
        factor = 1
    else:
        factor = max(1, int(round(fine.width / max(1, target_w))))
    wall, w, h = downsample_wall(fine.inside, fine.width, fine.height, factor, 0.5)
    return png_from_masks(wall, w, h)


def thumbnail_png(plan: Floorplan, target_w: int = 160) -> bytes:
    ppm = max(1e-6, plan.size_x_m)
    ppm = target_w / ppm
    fine = fine_from_plan(plan, ppm * 3, pad_px=3)
    return thumbnail_from_fine(fine, target_w)
