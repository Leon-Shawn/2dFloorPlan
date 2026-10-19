"""ASCII dump of one floorplan: walls, cleaned, missed, dead zones."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.houseexpo import FloorplanLibrary
from cleaner_sim.planner import decompose_regions, invert, plan_coverage
from cleaner_sim.rasterize import build_world

lib = FloorplanLibrary()
fid = lib.ids()[0] if len(sys.argv) < 2 else sys.argv[1]
plan = lib.get(fid)
occ, view = build_world(plan, cell_cm=10.0, robot_cm=34.0)
w, h = occ.width, occ.height
res = plan_coverage(occ, view, max_retreat_passes=0)

covered = res.covered
uncovered = G.intersection(view.cleanable, invert(covered))
dead = view.dead
radius = view.radius_cells

region, sizes, adj, cent = decompose_regions(view.reachable, w, h, radius)
members: dict[int, list[int]] = {}
for i, lab in enumerate(region):
    if lab:
        members.setdefault(lab, []).append(i)

print(f"{fid}  grid {w}x{h}  radius={radius}  cleanable={G.count(view.cleanable)}  "
      f"uncovered={G.count(uncovered)}  dead={G.count(dead)}  regions={len(sizes)}  "
      f"steps(main only)={len(res.path)}")
print(f"region sizes: {sorted(sizes.values(), reverse=True)[:10]}")
print()

for y in range(h):
    row = []
    for x in range(w):
        i = y * w + x
        if occ.wall[i]:
            ch = "#"
        elif uncovered[i]:
            ch = "O"
        elif covered[i]:
            ch = "." if view.reachable[i] else ","
        elif dead[i]:
            ch = "-"
        else:
            ch = " "
        row.append(ch)
    print(f"{y:>3} {''.join(row)}")

print("\nlegend: # wall   . cleaned(robot passed)   , cleaned(edge)   O missed   - dead")
