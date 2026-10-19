"""Coverage path planning: region decomposition + per-room boustrophedon sweep.

This is the engine that makes modules 4/5/7/9 meaningful — without a coverage path
there is nothing to animate, count or replay. (In the course brief this corresponds
to module 3, which is not among the five chosen, so it lives here as the internal
engine and the dashboard reports on it.)

Pipeline
--------
1. **Region decomposition (区域划分)** — the robot's legal positions are eroded by a
   passage radius so that doorways and other narrow gaps break into separate
   components, then a multi-source BFS (morphological watershed) grows those cores
   back out to partition the floor into room-like regions. Sweeping a whole flat
   row-by-row would make the robot zig-zag between rooms on every single pass; a
   per-region sweep keeps it inside one room at a time.
2. **Per-region sweep orientation** — each region is swept along its longer bounding
   box axis, which minimises the number of passes.
3. **Serpentine traversal** with adaptive sweep lines (spaced ``2 * radius`` so
   passes overlap by one cell), starting from real geometry so L-shaped and concave
   rooms stay correct.
4. **Obstacle detour** — consecutive runs are linked by A* through legal robot
   positions; runs that turn out to be unreachable are deferred.
5. **Region visiting order** — BFS over the region adjacency graph from the robot's
   start region, i.e. the robot walks room to room through the doorways.
6. **Dead-zone retreat (死角回退)** — whatever is still dirty is clustered, and each
   cluster is re-planned as a local serpentine over exactly those robot positions
   whose cleaning disk can touch it. This is the "fall back and re-divide the
   region" mechanism the brief calls the hardest part, and it lifts coverage from
   ~95% to ~99%+ around corners and inside narrow nooks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from . import grid as G
from .rasterize import Occupancy, RobotView


def invert(mask) -> bytearray:
    return bytearray(1 if not v else 0 for v in mask)


# --------------------------------------------------------------------- sweep lines
def _runs_1d(values, length: int):
    """Maximal runs of truthy values as inclusive (start, end) pairs."""
    runs = []
    start = None
    for i in range(length):
        if values(i):
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, length - 1))
    return runs


def build_sweep_lines(mask: bytearray, width: int, height: int, vertical: bool):
    """Ordered sweep lines; each line is a list of runs, each run a list of indices."""
    lines = []
    if not vertical:
        for y in range(height):
            base = y * width
            runs = _runs_1d(lambda x, b=base: mask[b + x], width)
            if runs:
                lines.append({"index": y,
                              "runs": [[base + i for i in range(a, b + 1)] for a, b in runs]})
    else:
        for x in range(width):
            runs = _runs_1d(lambda y, xx=x: mask[y * width + xx], height)
            if runs:
                lines.append({"index": x,
                              "runs": [[i * width + x for i in range(a, b + 1)] for a, b in runs]})
    return lines


def select_lines(lines: list[dict], spacing: int, tail_slack: int) -> list[dict]:
    """First line, then every `spacing` lines, plus the last one if it is far enough."""
    if not lines:
        return []
    selected = [lines[0]]
    for line in lines[1:]:
        if line["index"] - selected[-1]["index"] >= spacing:
            selected.append(line)
    if lines[-1]["index"] - selected[-1]["index"] > tail_slack:
        selected.append(lines[-1])
    return selected


def serpentine_over(lines: list[dict], spacing: int, link, tail_slack: int = 0):
    """Serpentine route over pre-built sweep lines. Returns (route, deferred_runs)."""
    route: list[int] = []
    deferred: list[list[int]] = []
    current = None

    for line_no, line in enumerate(select_lines(lines, spacing, tail_slack)):
        left_to_right = (line_no % 2 == 0)
        runs = line["runs"] if left_to_right else line["runs"][::-1]
        for run in runs:
            cells = run if left_to_right else run[::-1]
            if current is None:
                route.extend(cells)
            else:
                connector = link(current, cells[0])
                if connector is None:
                    deferred.append(cells)
                    continue
                # connector runs [current, ..., cells[0]]; drop both ends so the run
                # supplies its own first element without duplicating it
                route.extend(connector[1:-1])
                route.extend(cells)
            current = cells[-1]
    return route, deferred


def serpentine(mask: bytearray, width: int, height: int, spacing: int, link,
               tail_slack: int = 0, vertical: bool = False):
    """Convenience wrapper: build lines from a mask, then traverse them."""
    return serpentine_over(build_sweep_lines(mask, width, height, vertical),
                           spacing, link, tail_slack)


# ------------------------------------------------------------------ region division
def decompose_regions(reachable: bytearray, width: int, height: int, radius: int):
    """Partition legal robot positions into room-like regions.

    Erode by a passage radius so doorways sever, label the cores, then grow them
    back simultaneously (watershed) to label every reachable cell.
    """
    passage_radius = max(radius + 1, int(round(2.2 * radius)))
    core = G.erode(reachable, width, height, passage_radius)
    labels, sizes = G.label_components(core, width, height, 4)

    region = [0] * (width * height)
    q = deque()
    for i, lab in enumerate(labels):
        if lab:
            region[i] = lab
            q.append(i)

    while q:
        idx = q.popleft()
        lab = region[idx]
        y, x = divmod(idx, width)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height:
                n = ny * width + nx
                if reachable[n] and not region[n]:
                    region[n] = lab
                    q.append(n)

    # Reachable blobs too small to survive the erosion get their own regions so the
    # sweep still services them (a robot can legally stand there).
    leftover = bytearray(1 if (reachable[i] and not region[i]) else 0
                         for i in range(width * height))
    if G.count(leftover):
        extra, _ = G.label_components(leftover, width, height, 4)
        offset = max(labels) if labels else 0
        for i, lab in enumerate(extra):
            if lab:
                region[i] = offset + lab

    # True region sizes (the watershed grows cores back out, so core sizes alone
    # would under-report). These are the numbers the visiting order sorts on.
    sizes: dict[int, int] = {}
    for lab in region:
        if lab:
            sizes[lab] = sizes.get(lab, 0) + 1

    # adjacency between regions (4-neighbour contact)
    adjacency: dict[int, set[int]] = {}
    for y in range(height):
        base = y * width
        for x in range(width):
            a = region[base + x]
            if not a:
                continue
            if x + 1 < width:
                b = region[base + x + 1]
                if b and b != a:
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)
            if y + 1 < height:
                b = region[base + x + width]
                if b and b != a:
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)

    # centroid per region (for ordering fallbacks)
    sums: dict[int, list[float]] = {}
    for i, lab in enumerate(region):
        if lab:
            y, x = divmod(i, width)
            acc = sums.setdefault(lab, [0, 0, 0])
            acc[0] += x
            acc[1] += y
            acc[2] += 1
    centroids = {lab: (acc[0] / acc[2], acc[1] / acc[2]) for lab, acc in sums.items()}

    return region, dict(sizes), adjacency, centroids


def order_regions(region, sizes, adjacency, centroids, start_cell: int) -> list[int]:
    """Visit regions breadth-first through the doorways, starting where the robot is."""
    if not sizes:
        return []
    start_region = region[start_cell] or next(iter(sizes))
    order: list[int] = []
    seen = {start_region}
    q = deque([start_region])
    while q:
        rid = q.popleft()
        order.append(rid)
        neighbours = sorted(adjacency.get(rid, ()),
                            key=lambda n: -sizes.get(n, 0))
        for n in neighbours:
            if n not in seen:
                seen.add(n)
                q.append(n)

    # any region not reachable through adjacency: nearest centroid first
    remaining = [r for r in sizes if r not in seen]
    cursor = centroids.get(start_region, (0.0, 0.0))
    while remaining:
        nxt = min(remaining, key=lambda r: (centroids[r][0] - cursor[0]) ** 2
                  + (centroids[r][1] - cursor[1]) ** 2)
        remaining.remove(nxt)
        seen.add(nxt)
        order.append(nxt)
        cursor = centroids[nxt]
    return order


# --------------------------------------------------------------------------- result
@dataclass
class PlanResult:
    path: list[int] = field(default_factory=list)
    newly: list[int] = field(default_factory=list)   # new cells cleaned at each step
    touch: list[int] = field(default_factory=list)   # walkable cells swept over
    phases: list[dict] = field(default_factory=list)
    covered: bytearray = field(default_factory=bytearray)
    stats: dict = field(default_factory=dict)


class _Builder:
    """Accumulates the route while tracking what has been cleaned."""

    def __init__(self, occ: Occupancy, view: RobotView):
        self.occ = occ
        self.view = view
        self.w, self.h = occ.width, occ.height
        self.radius = view.radius_cells
        self.covered = bytearray(self.w * self.h)
        self.path: list[int] = []
        self.newly: list[int] = []
        self.touch: list[int] = []

    def step(self, idx: int) -> None:
        """Move to `idx` and clean its whole square footprint."""
        y, x = divmod(idx, self.w)
        r = self.radius
        new = 0
        touched = 0
        walkable = self.occ.walkable
        covered = self.covered
        for ny in range(max(0, y - r), min(self.h - 1, y + r) + 1):
            base = ny * self.w
            for nx in range(max(0, x - r), min(self.w - 1, x + r) + 1):
                j = base + nx
                if walkable[j]:
                    touched += 1
                    if not covered[j]:
                        covered[j] = 1
                        new += 1
        self.path.append(idx)
        self.newly.append(new)
        self.touch.append(touched)

    def extend(self, cells):
        last = self.path[-1] if self.path else None
        for c in cells:
            if c == last:          # never emit a zero-length move
                continue
            self.step(c)
            last = c

    def coverage(self) -> float:
        cleanable = G.count(self.view.cleanable)
        if not cleanable:
            return 0.0
        return G.count(G.intersection(self.covered, self.view.cleanable)) / cleanable


def _bbox_of(mask_cells, width: int):
    ys = [c // width for c in mask_cells]
    xs = [c % width for c in mask_cells]
    return min(xs), min(ys), max(xs), max(ys)


def plan_coverage(occ: Occupancy, view: RobotView, target_cov: float = 0.995,
                  max_retreat_passes: int = 4, link_budget: int = 60_000,
                  spacing_mode: str = "overlap") -> PlanResult:
    """Produce a full-coverage route over the reachable floor."""
    w, h = occ.width, occ.height
    result = PlanResult()
    cleanable = G.count(view.cleanable)
    if cleanable == 0:
        result.covered = bytearray(w * h)
        result.stats = {"error": "no cleanable floor for this robot size"}
        return result

    builder = _Builder(occ, view)
    radius = view.radius_cells
    spacing = max(1, 2 * radius + 1) if spacing_mode == "exact" else max(1, 2 * radius)
    link = lambda a, b: G.astar(view.reachable, w, h, a, b,      # noqa: E731
                                max_expand=link_budget, diagonal=True)

    # -------------------------------------------------- region decomposition
    region, sizes, adjacency, centroids = decompose_regions(view.reachable, w, h, radius)
    order = order_regions(region, sizes, adjacency, centroids, view.start)
    if not order:
        result.stats = {"error": "no reachable robot positions"}
        return result

    members: dict[int, list[int]] = {}
    for i, lab in enumerate(region):
        if lab:
            members.setdefault(lab, []).append(i)

    builder.extend([view.start])
    cursor = view.start
    deferred_total = 0

    for rid in order:
        cells = members.get(rid)
        if not cells:
            continue
        x0, y0, x1, y1 = _bbox_of(cells, w)
        vertical = (y1 - y0) > (x1 - x0)          # sweep along the longer axis
        mask = bytearray(w * h)
        for c in cells:
            mask[c] = 1

        lines = build_sweep_lines(mask, w, h, vertical)
        route, deferred = serpentine_over(lines, spacing, link, tail_slack=0)
        deferred_total += len(deferred)
        if not route:
            continue

        # Enter the serpentine at whichever end is closer to where the robot is, so
        # it does not cross the room just to reach the nominal starting corner.
        if len(route) > 1:
            cy, cx = divmod(cursor, w)
            ey, ex = divmod(route[0], w)
            ly, lx = divmod(route[-1], w)
            if (abs(ly - cy) + abs(lx - cx)) < (abs(ey - cy) + abs(ex - cx)):
                route.reverse()

        connector = G.astar(view.reachable, w, h, cursor, route[0],
                            max_expand=link_budget, diagonal=True)
        if connector is None:
            continue

        start_step = len(builder.path)
        builder.extend(connector[1:-1])
        builder.extend(route)
        cursor = route[-1]

        result.phases.append({
            "name": f"区域 #{len([p for p in result.phases if p['kind'] == 'region'])+1}"
                    f"（{len(cells)} 格 · {'纵向' if vertical else '横向'}扫描）",
            "kind": "region",
            "region": rid,
            "cells": len(cells),
            "orientation": "vertical" if vertical else "horizontal",
            "start_step": start_step,
            "end_step": len(builder.path) - 1,
            "coverage": round(builder.coverage(), 4),
        })

    result.phases.insert(0, {
        "name": "主扫（区域划分 + 牛耕式全覆盖）",
        "kind": "sweep",
        "start_step": 0,
        "end_step": len(builder.path) - 1,
        "coverage": round(builder.coverage(), 4),
        "regions": len([p for p in result.phases if p["kind"] == "region"]),
    })

    # ------------------------------------------------------- dead-zone retreats
    for pass_no in range(max_retreat_passes):
        if builder.coverage() >= target_cov:
            break
        uncovered = G.intersection(view.cleanable, invert(builder.covered))
        if not G.count(uncovered):
            break

        labels, cluster_sizes = G.label_components(uncovered, w, h, 4)
        clusters = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)
        progressed = False

        for cluster_id, size in clusters:
            cluster_cells = [i for i, lab in enumerate(labels) if lab == cluster_id]
            # earlier trips in this pass may already have cleaned part of it
            remaining = [c for c in cluster_cells if not builder.covered[c]]
            if not remaining:
                continue

            cluster_mask = bytearray(w * h)
            for c in remaining:
                cluster_mask[c] = 1

            # Re-divide the neighbourhood: a legal position is a candidate iff its
            # cleaning square can touch this cluster. Expressed as a dilation of the
            # dirty cells, which is one separable pass instead of a per-cell search.
            candidate = G.intersection(
                view.reachable,
                G.dilate_square(cluster_mask, w, h, radius))
            if not G.count(candidate):
                continue

            # sweep the local island along its longer axis
            bxs = [c % w for c in remaining]
            bys = [c // w for c in remaining]
            vertical = (max(bys) - min(bys)) > (max(bxs) - min(bxs))
            local_route, _ = serpentine(candidate, w, h, spacing, link,
                                        tail_slack=0, vertical=vertical)
            if not local_route:
                continue

            connector = G.astar(view.reachable, w, h, builder.path[-1], local_route[0],
                                max_expand=link_budget, diagonal=True)
            if connector is None:
                continue

            start_step = len(builder.path)
            builder.extend(connector[1:-1])
            builder.extend(local_route)
            progressed = True

            result.phases.append({
                "name": f"死区回退 #{pass_no + 1}（{size} 格）",
                "kind": "retreat",
                "start_step": start_step,
                "end_step": len(builder.path) - 1,
                "coverage": round(builder.coverage(), 4),
                "cluster_cells": size,
            })

        if not progressed:
            break

    result.path = builder.path
    result.newly = builder.newly
    result.touch = builder.touch
    result.covered = builder.covered

    covered_cleanable = G.count(G.intersection(builder.covered, view.cleanable))
    result.stats = {
        "steps": len(result.path),
        "sweep_spacing_cells": spacing,
        "regions": len([p for p in result.phases if p["kind"] == "region"]),
        "deferred_intervals": deferred_total,
        "retreat_passes": sum(1 for p in result.phases if p["kind"] == "retreat"),
        "covered_cleanable_cells": covered_cleanable,
        "cleanable_cells": cleanable,
        "coverage_reachable": round(covered_cleanable / cleanable, 4),
        "target_coverage": target_cov,
        "reached_target": covered_cleanable / cleanable >= target_cov,
    }
    return result
