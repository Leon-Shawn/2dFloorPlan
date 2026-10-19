"""Diagnostic: locate the first non-contiguous / duplicated step in the planned path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.houseexpo import FloorplanLibrary
from cleaner_sim.planner import plan_coverage, serpentine, select_sweep_lines
from cleaner_sim.rasterize import build_world

lib = FloorplanLibrary()
plan = lib.get(lib.ids()[0])
occ, view = build_world(plan, cell_cm=10.0, robot_cm=34.0)
print(f"grid {occ.width}x{occ.height}  radius_cells={view.radius_cells}  "
      f"cleanable={G.count(view.cleanable)}  reachable={G.count(view.reachable)}")

res = plan_coverage(occ, view)
path = res.path
print(f"path steps={len(path)}  phases={[(p['name'], p['start_step'], p['end_step']) for p in res.phases]}")

bad = []
for i in range(1, len(path)):
    y0, x0 = divmod(path[i - 1], occ.width)
    y1, x1 = divmod(path[i], occ.width)
    if max(abs(x1 - x0), abs(y1 - y0)) != 1:
        bad.append(i)

print(f"non-contiguous steps: {len(bad)} -> first 10 indices {bad[:10]}")
for i in bad[:5]:
    a, b = path[i - 1], path[i]
    ay, ax = divmod(a, occ.width)
    by, bx = divmod(b, occ.width)
    phase = next((p["name"] for p in res.phases
                  if p["start_step"] <= i <= p["end_step"]), "?")
    print(f"  step {i}: {a}(x={ax},y={ay}) -> {b}(x={bx},y={by})   phase={phase}")

# also check the raw serpentine output in isolation
route, deferred = serpentine(view.reachable, occ.width, occ.height,
                             max(1, 2 * view.radius_cells),
                             lambda a, b: G.astar(view.reachable, occ.width, occ.height, a, b,
                                                  max_expand=60000, diagonal=True),
                             tail_slack=view.radius_cells)
bad2 = [i for i in range(1, len(route))
        if max(abs(divmod(route[i], occ.width)[1] - divmod(route[i - 1], occ.width)[1]),
               abs(divmod(route[i], occ.width)[0] - divmod(route[i - 1], occ.width)[0])) != 1]
print(f"raw serpentine: steps={len(route)} non-contiguous={len(bad2)} deferred={len(deferred)}")
print(f"  sweep lines selected={len(select_sweep_lines(view.reachable, occ.width, occ.height, max(1, 2*view.radius_cells), view.radius_cells))}")
