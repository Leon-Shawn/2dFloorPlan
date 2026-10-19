"""Render a finished cleaning run to a PNG — visual check of the planner output.

    python tools/render_result.py [floorplan-id] [--scale 2] [--out out.png]

Colours: wall dark · cleaned teal · cleanable-but-dirty light · robot-passed path
brighter · separated (needs its own run) violet · genuine dead corner amber.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.coverage import analyse_coverage
from cleaner_sim.houseexpo import FloorplanLibrary
from cleaner_sim.planner import plan_coverage
from cleaner_sim.png import png_rgb
from cleaner_sim.rasterize import build_world

WALL = (38, 46, 60)
CLEAN = (34, 195, 166)
DIRTY = (214, 224, 238)
PATH = (150, 220, 255)
SEP = (107, 75, 143)
TIGHT = (240, 169, 43)


def render(fid: str, cell_cm: float, robot_cm: float, out: Path, scale: int = 1):
    lib = FloorplanLibrary()
    plan = lib.get(fid)
    occ, view = build_world(plan, cell_cm=cell_cm, robot_cm=robot_cm)
    result = plan_coverage(occ, view, target_cov=0.995)
    coverage = analyse_coverage(occ, view, result.covered)

    w, h = occ.width, occ.height
    on_path = bytearray(w * h)
    for c in result.path:
        on_path[c] = 1

    rgb = bytearray(w * h * 3)
    for i in range(w * h):
        if occ.wall[i]:
            col = WALL
        elif result.covered[i]:
            col = PATH if on_path[i] else CLEAN
        elif view.cleanable[i]:
            col = DIRTY
        elif view.unreachable[i]:
            col = SEP
        else:
            col = TIGHT
        rgb[i * 3] = col[0]
        rgb[i * 3 + 1] = col[1]
        rgb[i * 3 + 2] = col[2]

    if scale > 1:
        sw, sh = w * scale, h * scale
        big = bytearray(sw * sh * 3)
        for y in range(h):
            for x in range(w):
                o = (y * w + x) * 3
                px = (rgb[o], rgb[o + 1], rgb[o + 2])
                for dy in range(scale):
                    base = ((y * scale + dy) * sw + x * scale) * 3
                    for dx in range(scale):
                        p = base + dx * 3
                        big[p] = px[0]
                        big[p + 1] = px[1]
                        big[p + 2] = px[2]
        png = png_rgb(sw, sh, big)
    else:
        png = png_rgb(w, h, rgb)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)

    print(f"{fid}  {plan.room_num} rooms  {plan.size_x_m}x{plan.size_y_m} m  "
          f"grid {w}x{h} @ {cell_cm:g}cm")
    print(f"  steps={len(result.path)}  path={result.stats['steps']}  "
          f"coverage_reachable={coverage['coverage_reachable']:.4f}  "
          f"coverage_walkable={coverage['coverage_walkable']:.4f}")
    print(f"  cleanable={coverage['cleanable_area_m2']} m2  "
          f"separated={coverage['separated_area_m2']} m2  tight={coverage['tight_area_m2']} m2")
    print(f"  -> {out}  ({len(png)} bytes, {sw if scale > 1 else w}x{sh if scale > 1 else h})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id", nargs="?", default=None)
    ap.add_argument("--cell", type=float, default=10.0)
    ap.add_argument("--robot", type=float, default=34.0)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lib = FloorplanLibrary()
    fid = args.id or lib.ids()[0]
    out = Path(args.out) if args.out else Path("previews") / f"result_{fid[:10]}.png"
    render(fid, args.cell, args.robot, out, args.scale)


if __name__ == "__main__":
    main()
