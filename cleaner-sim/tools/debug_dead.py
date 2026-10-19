"""Inspect floorplans whose walkable area is largely unreachable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.houseexpo import FloorplanLibrary
from cleaner_sim.rasterize import build_world

lib = FloorplanLibrary()
rows = []
for meta in lib.list_meta():
    occ, view = build_world(lib.get(meta["id"]), cell_cm=10.0, robot_cm=34.0)
    dead = view.stats["dead_ratio"]
    if dead > 0.05:
        rows.append((dead, meta, occ, view))

rows.sort(reverse=True, key=lambda r: r[0])
print(f"{len(rows)} floorplans with >5% dead area\n")
for dead, meta, occ, view in rows:
    w, h = occ.width, occ.height
    labels, sizes = G.label_components(occ.walkable, w, h, 4)
    clabels, csizes = G.label_components(view.center_ok, w, h, 4)
    print(f"{meta['id']}  rooms={meta['room_num']}  {meta['size_x_m']}x{meta['size_y_m']}m  "
          f"grid={w}x{h}")
    print(f"   dead={dead:.3f}  walkable={G.count(occ.walkable)}  "
          f"reachable_positions={G.count(view.reachable)}  cleanable={G.count(view.cleanable)}")
    print(f"   walkable components={len(sizes)} sizes={sorted(sizes.values(), reverse=True)[:6]}")
    print(f"   robot-position components={len(csizes)} sizes={sorted(csizes.values(), reverse=True)[:6]}")
    print()
