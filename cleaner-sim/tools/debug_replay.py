"""Diagnostic: does re-encoding/replaying the planned route reproduce `covered`?

The browser never sees the server's covered mask — it replays the direction codes and
stamps the same footprint. This checks that round trip for every floorplan and prints
the exact cells that disagree, which pinpoints any encoding or stamping mismatch.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.houseexpo import FloorplanLibrary
from cleaner_sim.planner import plan_coverage
from cleaner_sim.rasterize import build_world
from cleaner_sim.simulate import replay_payload

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)]


def replay(mask_wall, mask_cleanable, width, height, radius, payload):
    dirs = base64.b64decode(payload["dirs_b64"])
    x, y = payload["start"]
    cells = [y * width + x]
    for code in dirs:
        dx, dy = DIRS[code]
        x += dx
        y += dy
        cells.append(y * width + x)

    cleaned = bytearray(width * height)
    for c in cells:
        cy, cx = divmod(c, width)
        for ny in range(max(0, cy - radius), min(height - 1, cy + radius) + 1):
            base = ny * width
            for nx in range(max(0, cx - radius), min(width - 1, cx + radius) + 1):
                j = base + nx
                if not mask_wall[j]:
                    cleaned[j] = 1
    clean_total = sum(mask_cleanable)
    hit = sum(1 for i in range(width * height) if mask_cleanable[i] and cleaned[i])
    return cells, cleaned, hit, clean_total


def main():
    lib = FloorplanLibrary()
    bad = 0
    print(f"{'floorplan':<14}{'steps':>7}{'replay cov':>12}{'server cov':>12}{'diff cells':>11}")
    for fid in lib.ids():
        plan = lib.get(fid)
        occ, view = build_world(plan, cell_cm=10.0, robot_cm=34.0)
        res = plan_coverage(occ, view)
        if not res.path:
            continue
        payload = replay_payload(occ, res, res.path[0])
        wall = occ.wall
        cleanable = view.cleanable
        cells, cleaned, hit, total = replay(wall, cleanable, occ.width, occ.height,
                                            view.radius_cells, payload)
        server_hit = G.count(G.intersection(res.covered, cleanable))
        server_cov = server_hit / max(1, total)
        replay_cov = hit / max(1, total)

        # where do the two masks disagree (restricted to cleanable cells)?
        diff = [i for i in range(occ.width * occ.height)
                if cleanable[i] and (cleaned[i] != res.covered[i])]
        if diff or len(cells) != len(res.path):
            bad += 1
            print(f"{fid[:12]:<14}{len(res.path):>7}{replay_cov:>12.6f}{server_cov:>12.6f}"
                  f"{len(diff):>11}")
            for i in diff[:6]:
                y, x = divmod(i, occ.width)
                print(f"     cell (x={x}, y={y})  replay={cleaned[i]} server={res.covered[i]}"
                      f"  walkable={occ.walkable[i]} cleanable={cleanable[i]}"
                      f"  reachable={view.reachable[i]}")
            if len(cells) != len(res.path):
                print(f"     CELL COUNT differs: replay={len(cells)} server={len(res.path)}")

    print()
    if bad:
        print(f"FAIL: {bad} floorplans disagree between the planned route and its replay")
        return 1
    print("PASS: replaying the encoded route reproduces the server's covered mask exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
