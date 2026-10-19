"""Diagnostic: where do the steps go, and what does the main sweep leave behind?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.houseexpo import FloorplanLibrary
from cleaner_sim.planner import invert, plan_coverage
from cleaner_sim.rasterize import build_world

lib = FloorplanLibrary()
metas = {m["id"]: m for m in lib.list_meta()}
targets = [lib.ids()[0], lib.ids()[12], lib.ids()[-1]]

for fid in targets:
    meta = metas[fid]
    plan = lib.get(fid)
    occ, view = build_world(plan, cell_cm=10.0, robot_cm=34.0)
    r = view.radius_cells
    cleanable = G.count(view.cleanable)

    sweep = plan_coverage(occ, view, max_retreat_passes=0)
    uncovered = G.intersection(view.cleanable, invert(sweep.covered))
    labels, sizes = G.label_components(uncovered, occ.width, occ.height, 4)
    total_uncovered = G.count(uncovered)

    full = plan_coverage(occ, view, max_retreat_passes=4)

    disk = len(G.disk_offsets(r))
    swath = 2 * r + 1
    ideal_steps = cleanable / swath
    print(f"--- {meta['room_num']:>2} rooms, {meta['size_x_m']}x{meta['size_y_m']}m, "
          f"grid {occ.width}x{occ.height}, r={r}")
    print(f"    cleanable={cleanable:>6}  ideal_steps≈{ideal_steps:>7.0f}")
    print(f"    main sweep only : steps={len(sweep.path):>6}  "
          f"uncovered={total_uncovered:>5} ({total_uncovered/cleanable:.1%})  "
          f"clusters={len(sizes)}")
    if sizes:
        top = sorted(sizes.values(), reverse=True)[:8]
        print(f"    largest uncovered clusters: {top}")
    print(f"    with retreat    : steps={len(full.path):>6}  "
          f"retreat_added={len(full.path) - len(sweep.path):>6}  "
          f"phases={len(full.phases) - 1}  "
          f"coverage={full.stats['coverage_reachable']:.4f}")
    print(f"    ratio steps/ideal = {len(full.path)/ideal_steps:.2f}")
