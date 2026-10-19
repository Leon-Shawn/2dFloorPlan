"""Round-trip test for the image upload pipeline.

Renders a HouseExpo floorplan into a realistic floorplan *image* (dark walls on a
light page with a margin), then feeds it back through the upload parser and compares
the resulting walkable area with the vector pipeline. This exercises exactly what an
uploaded PNG/JPEG goes through, without needing a real photograph.

    python tools/test_image_pipeline.py [--cell 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.houseexpo import FloorplanLibrary
from cleaner_sim.imageplan import ImageParseParams, parse_floorplan_image
from cleaner_sim.rasterize import build_world, fine_from_plan, png_from_masks

PX_PER_M = 40.0            # 2.5 cm per pixel, like a decent scan
WALL_M = 0.16              # drawn wall thickness
MARGIN_M = 0.9             # white page margin around the plan


def render_floorplan_image(plan, px_per_m=PX_PER_M, wall_m=WALL_M, margin_m=MARGIN_M):
    """Produce (gray, width, height) for a synthetic but realistic floorplan image.

    The outline is rasterised with padding equal to the wall thickness so the drawn
    wall ring closes all the way around — without it the interior would touch the
    crop edge and the outside flood fill would leak in, exactly as it would for a
    tightly cropped real scan.
    """
    wall_px = max(1, int(round(wall_m * px_per_m)))
    fine = fine_from_plan(plan, px_per_m, pad_px=wall_px + 2)
    inside = fine.inside
    fw, fh = fine.width, fine.height

    inside_d = G.dilate_square(inside, fw, fh, wall_px)     # interior grown by the wall
    wall_band = G.difference(inside_d, inside)              # the wall itself

    margin = int(round(margin_m * px_per_m))
    width, height = fw + 2 * margin, fh + 2 * margin
    gray = bytearray([250]) * (width * height)              # light page
    for y in range(fh):
        base = y * fw
        dst = (y + margin) * width + margin
        for x in range(fw):
            if wall_band[base + x]:
                gray[dst + x] = 25
    return gray, width, height, fine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=float, default=10.0)
    ap.add_argument("--robot", type=float, default=34.0)
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    lib = FloorplanLibrary()
    ids = lib.ids()[: args.limit]
    print(f"round-tripping {len(ids)} floorplans through the image pipeline "
          f"({PX_PER_M:g} px/m, {WALL_M * 100:g} cm walls)\n")
    print(f"{'floorplan':<14}{'size':>12}{'vector m2':>11}{'image m2':>10}"
          f"{'delta':>9}{'threshold':>11}{'islands':>9}")

    worst = 0.0
    for fid in ids:
        plan = lib.get(fid)
        occ_v, view_v = build_world(plan, cell_cm=args.cell, robot_cm=args.robot)
        vector_area = occ_v.area_m2(occ_v.walkable)

        gray, w, h, fine = render_floorplan_image(plan)
        # The parser crops to the floor's bounding box, so the real-world width to
        # hand it is the outline's own width -- NOT fine.size_x_m, which includes the
        # padding the renderer added around the outline.
        real_w = max(v[0] for v in plan.verts) - min(v[0] for v in plan.verts)
        params = ImageParseParams(real_width_m=real_w, scale_axis="x")
        parsed, diag = parse_floorplan_image(gray, w, h, params)
        occ_i, view_i = build_world(parsed, cell_cm=args.cell, robot_cm=args.robot)
        image_area = occ_i.area_m2(occ_i.walkable)

        delta = (image_area - vector_area) / max(1e-6, vector_area)
        worst = max(worst, abs(delta))
        size = f"{plan.size_x_m}x{plan.size_y_m}"
        print(f"{fid[:12]:<14}{size:>12}{vector_area:>11.1f}{image_area:>10.1f}"
              f"{delta * 100:>8.1f}%{diag['threshold']:>11}{diag['dropped_islands']:>9}")

    print(f"\nmax area deviation: {worst * 100:.1f}%")
    ok = worst < 0.20
    print("PASS: uploaded-image parsing reproduces the vector geometry"
          if ok else "FAIL: image parse deviates too much from the vector pipeline")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
