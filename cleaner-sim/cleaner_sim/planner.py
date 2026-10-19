"""Coverage path planning — Boustrophedon Cellular Decomposition (BCD).

Replaces the earlier watershed-region planner (`planner_watershed.py`, kept for
A/B benchmarking). See README §"路径算法" for the measured improvement.

Why BCD
-------
A cell decomposition splits the free space into pieces that can each be swept by a
simple back-and-forth (lawn-mower) motion, so the robot never has to zig-zag between
distant parts of the map. The classic construction
(Choset & Pignon, *Coverage Path Planning: The Boustrophedon Cellular
Decomposition*, FSR 1998) sweeps an imaginary line across the workspace and starts a
new cell wherever the free-space connectivity along that line changes.

Implementation notes
--------------------
* Sweeping along ``x`` and reading the free runs in ``y``: because the run structure
  is *constant* inside a slab, every cell is exactly ``(slab, run_index)`` — a
  rectangle — so there is no need to label connected components per slab. The whole
  decomposition is one O(w·h) pass.
* Both sweep axes are evaluated and the cheaper decomposition is kept.
* **Cell merging**: greedily fuse adjacent cells while the union stays monotone along
  an axis (at most one run per row/column). A monotone region is swept by a single
  serpentine, so merging can only remove a connector — never add travel.
* **Sweep spacing** defaults to ``2r+1`` (the exact footprint width). A square
  footprint tiles perfectly at that spacing, whereas ``2r`` overlaps every pass by
  one row and costs ~20% extra steps for nothing.
* **Cell ordering** is greedy nearest-neighbour over each cell's two entry/exit ends,
  then improved by 2-opt (segment reversal with orientation flip) — an open
  path-TSP with free entry/exit per cell.
* **Dead-zone retreat** stays from the previous planner: cluster whatever is still
  dirty, re-divide locally, and sweep it with a local serpentine.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from . import grid as G
from .rasterize import Occupancy, RobotView


def invert(mask) -> bytearray:
    return bytearray(1 if not v else 0 for v in mask)


# ------------------------------------------------------------ sweep-line helpers
def runs_along(mask: bytearray, width: int, height: int, index: int, axis: str):
    """Inclusive (start, end) runs of set cells on one sweep line.

    ``axis='x'`` means the line is the column ``x=index`` and runs are read in ``y``.
    """
    runs = []
    start = None
    if axis == "x":
        for y in range(height):
            if mask[y * width + index]:
                if start is None:
                    start = y
            elif start is not None:
                runs.append((start, y - 1))
                start = None
        if start is not None:
            runs.append((start, height - 1))
    else:
        base = index * width
        for x in range(width):
            if mask[base + x]:
                if start is None:
                    start = x
            elif start is not None:
                runs.append((start, x - 1))
                start = None
        if start is not None:
            runs.append((start, width - 1))
    return runs


def build_sweep_lines(mask: bytearray, width: int, height: int, vertical: bool):
    """Ordered sweep lines; each line holds a list of runs, each run a list of indices."""
    lines = []
    if not vertical:
        for y in range(height):
            base = y * width
            runs = []
            start = None
            for x in range(width):
                if mask[base + x]:
                    if start is None:
                        start = x
                elif start is not None:
                    runs.append([base + i for i in range(start, x)])
                    start = None
            if start is not None:
                runs.append([base + i for i in range(start, width)])
            if runs:
                lines.append({"index": y, "runs": runs})
    else:
        for x in range(width):
            runs = []
            start = None
            for y in range(height):
                if mask[y * width + x]:
                    if start is None:
                        start = y
                elif start is not None:
                    runs.append([i * width + x for i in range(start, y)])
                    start = None
            if start is not None:
                runs.append([i * width + x for i in range(start, height)])
            if runs:
                lines.append({"index": x, "runs": runs})
    return lines


def select_lines(lines: list[dict], spacing: int, tail_slack: int) -> list[dict]:
    """First line, then every ``spacing`` lines, plus the last one if it is far enough."""
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
                route.extend(connector[1:-1])
                route.extend(cells)
            current = cells[-1]
    return route, deferred


def serpentine(mask: bytearray, width: int, height: int, spacing: int, link,
               tail_slack: int = 0, vertical: bool = False):
    return serpentine_over(build_sweep_lines(mask, width, height, vertical),
                           spacing, link, tail_slack)


# ============================================================================
#  Boustrophedon Cellular Decomposition
# ============================================================================
@dataclass
class Cell:
    cid: int
    x0: int
    y0: int
    x1: int
    y1: int
    count: int
    # runs per row (for merge checks): y -> list of (xa, xb)
    rows: dict = field(default_factory=dict)
    # runs per column: x -> list of (ya, yb)
    cols: dict = field(default_factory=dict)


def decompose_bcd(mask: bytearray, width: int, height: int, axis: str):
    """Partition `mask` into boustrophedon cells by sweeping along `axis`.

    Returns ``(labels, cells)`` where ``labels[i]`` is a 1-based cell id (0 outside)
    and ``cells`` maps id -> :class:`Cell`. Sweep ``axis='x'`` means the sweep line
    moves in +x and free runs are read along y.
    """
    n_lines = width if axis == "x" else height
    structure = [tuple(runs_along(mask, width, height, i, axis)) for i in range(n_lines)]

    critical = [0]
    for i in range(1, n_lines):
        if structure[i] != structure[i - 1]:
            critical.append(i)
    critical.append(n_lines)

    labels = [0] * (width * height)
    cells: dict[int, Cell] = {}
    cid = 0

    for s in range(len(critical) - 1):
        i0, i1 = critical[s], critical[s + 1] - 1
        if i1 < i0:
            continue
        for run_index, (a, b) in enumerate(structure[i0]):
            cid += 1
            cell = Cell(cid=cid, x0=0, y0=0, x1=0, y1=0, count=0)
            if axis == "x":
                cell.x0, cell.x1 = i0, i1
                cell.y0, cell.y1 = a, b
            else:
                cell.y0, cell.y1 = i0, i1
                cell.x0, cell.x1 = a, b
            for i in range(i0, i1 + 1):
                for t in range(a, b + 1):
                    idx = t * width + i if axis == "x" else i * width + t
                    labels[idx] = cid
            cell.count = (i1 - i0 + 1) * (b - a + 1)
            cells[cid] = cell

    _index_runs(labels, width, height, cells)
    return labels, cells


def _index_runs(labels, width: int, height: int, cells: dict) -> None:
    """Fill each cell's per-row and per-column run tables (used by the merge test)."""
    for cell in cells.values():
        cell.rows = {}
        cell.cols = {}
    for y in range(height):
        base = y * width
        x = 0
        while x < width:
            cid = labels[base + x]
            if not cid:
                x += 1
                continue
            start = x
            while x + 1 < width and labels[base + x + 1] == cid:
                x += 1
            cells[cid].rows.setdefault(y, []).append((start, x))
            x += 1
    for x in range(width):
        y = 0
        while y < height:
            cid = labels[y * width + x]
            if not cid:
                y += 1
                continue
            start = y
            while y + 1 < height and labels[(y + 1) * width + x] == cid:
                y += 1
            cells[cid].cols.setdefault(x, []).append((start, y))
            y += 1


def _coalesce_all(runs):
    """Sort and fuse overlapping/adjacent runs into a minimal run list."""
    runs = sorted(runs)
    out = [list(runs[0])]
    for a, b in runs[1:]:
        if a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(r) for r in out]


def _union_runs(a: dict, b: dict, monotone: bool):
    """Fuse two run tables.

    Returns None when `monotone` is requested and some key ends up with more than
    one run (i.e. the union is no longer sweepable by a single serpentine on that
    axis).
    """
    out = {}
    for key in set(a) | set(b):
        fused = _coalesce_all(list(a.get(key, [])) + list(b.get(key, [])))
        if monotone and len(fused) > 1:
            return None
        out[key] = fused
    return out


def merge_cells(labels: list, cells: dict, width: int, height: int):
    """Greedily fuse adjacent cells while the union stays monotone on some axis.

    A monotone region is swept by a single serpentine, so a merge can only remove a
    connector between cells — it can never add travel.
    """
    parent = {cid: cid for cid in cells}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    pairs = set()
    for y in range(height):
        base = y * width
        for x in range(width):
            a = labels[base + x]
            if not a:
                continue
            if x + 1 < width:
                b = labels[base + x + 1]
                if b and b != a:
                    pairs.add((min(a, b), max(a, b)))
            if y + 1 < height:
                b = labels[base + x + width]
                if b and b != a:
                    pairs.add((min(a, b), max(a, b)))

    groups = {cid: cell for cid, cell in cells.items()}
    changed = True
    while changed:
        changed = False
        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            ga, gb = groups[ra], groups[rb]
            applied = None
            for axis in ("h", "v"):
                rows = _union_runs(ga.rows, gb.rows, monotone=(axis == "h"))
                if rows is None:
                    continue
                cols = _union_runs(ga.cols, gb.cols, monotone=(axis == "v"))
                if cols is None:
                    continue
                applied = (rows, cols)
                break
            if applied is None:
                continue

            keep, drop = (ra, rb) if ga.count >= gb.count else (rb, ra)
            target, other = groups[keep], groups[drop]
            target.rows, target.cols = applied
            target.x0 = min(target.x0, other.x0)
            target.y0 = min(target.y0, other.y0)
            target.x1 = max(target.x1, other.x1)
            target.y1 = max(target.y1, other.y1)
            target.count += other.count
            parent[drop] = keep
            del groups[drop]
            changed = True

    # Relabel to dense ids and, importantly, key the returned dict by those ids so
    # callers can look cells up with the labels array directly.
    remap = {}
    out_cells = {}
    for next_id, cid in enumerate(sorted(groups), start=1):
        cell = groups[cid]
        cell.cid = next_id
        remap[cid] = next_id
        out_cells[next_id] = cell

    out_labels = [0] * (width * height)
    for i, cid in enumerate(labels):
        if cid:
            out_labels[i] = remap[find(cid)]
    return out_labels, out_cells


# --------------------------------------------------------------- cell sweeping
def _lines_for_cell(labels, width: int, cid: int, cell: Cell, vertical: bool):
    """Sweep lines restricted to the cell's bounding box (cheap, no full-grid mask)."""
    lines = []
    if not vertical:
        for y in range(cell.y0, cell.y1 + 1):
            base = y * width
            runs = []
            start = None
            for x in range(cell.x0, cell.x1 + 1):
                if labels[base + x] == cid:
                    if start is None:
                        start = x
                elif start is not None:
                    runs.append([base + i for i in range(start, x)])
                    start = None
            if start is not None:
                runs.append([base + i for i in range(start, cell.x1 + 1)])
            if runs:
                lines.append({"index": y, "runs": runs})
    else:
        for x in range(cell.x0, cell.x1 + 1):
            runs = []
            start = None
            for y in range(cell.y0, cell.y1 + 1):
                if labels[y * width + x] == cid:
                    if start is None:
                        start = y
                elif start is not None:
                    runs.append([i * width + x for i in range(start, y)])
                    start = None
            if start is not None:
                runs.append([i * width + x for i in range(start, cell.y1 + 1)])
            if runs:
                lines.append({"index": x, "runs": runs})
    return lines


def _cheap_cell_route(lines: list[dict], spacing: int, width: int):
    """Manhattan-only estimate of a cell's serpentine, with no A* calls.

    Returns ``(steps, first_cell, last_cell)`` where `steps` counts both the cleaned
    cells and the repositioning between runs, using Manhattan distance as the proxy
    for connector length. Cheap enough to evaluate several candidate decompositions.
    """
    selected = select_lines(lines, spacing, 0)
    clean = 0
    moves = 0
    current = None
    first = None
    for line_no, line in enumerate(selected):
        forward = (line_no % 2 == 0)
        runs = line["runs"] if forward else line["runs"][::-1]
        for run in runs:
            start = run[0] if forward else run[-1]
            end = run[-1] if forward else run[0]
            if first is None:
                first = start
            if current is not None:
                moves += _manhattan(current, start, width)
            clean += len(run)
            current = end
    if first is None:
        return 0, 0, None, None
    return clean, moves, first, current


def greedy_order(entries: list[dict], start: int, width: int):
    """Greedy nearest-neighbour ordering over ``{first,last}`` entries."""
    remaining = list(range(len(entries)))
    tour = []
    cursor = start
    while remaining:
        best = None
        for k in remaining:
            e = entries[k]
            d0 = _manhattan(cursor, e["first"], width)
            d1 = _manhattan(cursor, e["last"], width)
            flipped = d1 < d0
            d = d1 if flipped else d0
            if best is None or d < best[0]:
                best = (d, k, flipped)
        _, k, flipped = best
        remaining.remove(k)
        tour.append((k, flipped))
        cursor = entries[k]["last"] if flipped else entries[k]["first"]
    return tour


def _tour_gap(entries: list[dict], tour, start: int, width: int) -> int:
    total = 0
    cursor = start
    for k, flipped in tour:
        e = entries[k]
        entry = e["last"] if flipped else e["first"]
        exit_ = e["first"] if flipped else e["last"]
        total += _manhattan(cursor, entry, width)
        cursor = exit_
    return total


# ------------------------------------------------------------- order (open TSP)
def _manhattan(a: int, b: int, width: int) -> int:
    ay, ax = divmod(a, width)
    by, bx = divmod(b, width)
    return abs(ax - bx) + abs(ay - by)


def order_routes(routes: list[list[int]], start: int, width: int, optimize: bool = True):
    """Greedy nearest-neighbour over route ends, then 2-opt with orientation flips.

    Returns a list of ``(route_index, flipped)``.
    """
    n = len(routes)
    if n == 0:
        return []
    remaining = set(range(n))
    tour: list[tuple[int, bool]] = []
    cursor = start

    while remaining:
        best = None
        for k in remaining:
            route = routes[k]
            d0 = _manhattan(cursor, route[0], width)
            d1 = _manhattan(cursor, route[-1], width)
            flipped = d1 < d0
            d = d1 if flipped else d0
            if best is None or d < best[0]:
                best = (d, k, flipped)
        _, k, flipped = best
        remaining.discard(k)
        tour.append((k, flipped))
        cursor = routes[k][-1] if flipped else routes[k][0]

    if not optimize or n < 3:
        return tour

    def entry(t):
        route = routes[t[0]]
        return route[-1] if t[1] else route[0]

    def exit_(t):
        route = routes[t[0]]
        return route[0] if t[1] else route[-1]

    def leg(a, b):
        return _manhattan(a, b, width)

    for _ in range(40):
        improved = False
        for i in range(n - 1):
            prev = start if i == 0 else exit_(tour[i - 1])
            for j in range(i + 1, n):
                nxt = None if j + 1 >= n else entry(tour[j + 1])
                before = leg(prev, entry(tour[i]))
                after = leg(prev, exit_(tour[j]))
                if nxt is not None:
                    before += leg(exit_(tour[j]), nxt)
                    after += leg(entry(tour[i]), nxt)
                if after + 1e-9 < before:
                    segment = [(k, not f) for k, f in reversed(tour[i:j + 1])]
                    tour[i:j + 1] = segment
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return tour


# --------------------------------------------------------------------------- result
@dataclass
class PlanResult:
    path: list[int] = field(default_factory=list)
    newly: list[int] = field(default_factory=list)
    touch: list[int] = field(default_factory=list)
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
            if c == last:
                continue
            self.step(c)
            last = c

    def coverage(self) -> float:
        cleanable = G.count(self.view.cleanable)
        if not cleanable:
            return 0.0
        return G.count(G.intersection(self.covered, self.view.cleanable)) / cleanable


SPACING_MODES = {"exact": lambda r: 2 * r + 1, "overlap": lambda r: max(1, 2 * r)}


def plan_coverage(occ: Occupancy, view: RobotView, target_cov: float = 0.995,
                  max_retreat_passes: int = 4, link_budget: int = 60_000,
                  spacing_mode: str = "exact", merge: bool = True,
                  optimize_order: bool = True) -> PlanResult:
    """Full-coverage route over the reachable floor, using BCD + lawn-mower cells."""
    w, h = occ.width, occ.height
    result = PlanResult()
    cleanable = G.count(view.cleanable)
    if cleanable == 0:
        result.covered = bytearray(w * h)
        result.stats = {"error": "no cleanable floor for this robot size"}
        return result

    builder = _Builder(occ, view)
    radius = view.radius_cells
    spacing = SPACING_MODES.get(spacing_mode, SPACING_MODES["exact"])(radius)
    link = lambda a, b: G.astar(view.reachable, w, h, a, b,       # noqa: E731
                                max_expand=link_budget, diagonal=True)

    # ------------------------------- decomposition: build cheap candidates, score
    def evaluate(cand_axis, cand_labels, cand_cells, merged):
        entries = []
        clean_total = 0
        move_total = 0
        for cid in sorted(cand_cells):
            cell = cand_cells[cid]
            if not cell.count:
                continue
            vertical = (cell.y1 - cell.y0) > (cell.x1 - cell.x0)   # sweep the long axis
            lines = _lines_for_cell(cand_labels, w, cid, cell, vertical)
            clean, moves, first, last = _cheap_cell_route(lines, spacing, w)
            if first is None:
                continue
            entries.append({"cid": cid, "vertical": vertical, "cells": cell.count,
                            "lines": lines, "first": first, "last": last,
                            "clean": clean, "moves": moves})
            clean_total += clean
            move_total += moves
        if not entries:
            return None
        tour = greedy_order(entries, view.start, w)
        gap = _tour_gap(entries, tour, view.start, w)
        return {"axis": cand_axis, "merged": merged, "entries": entries,
                "tour": tour, "estimate": clean_total + move_total + gap,
                "clean": clean_total, "moves": move_total + gap}

    candidates = []
    for cand_axis in ("x", "y"):
        lab, cls = decompose_bcd(view.reachable, w, h, cand_axis)
        if cls:
            cand = evaluate(cand_axis, lab, cls, False)
            if cand:
                candidates.append(cand)
        if merge:
            lab2, cls2 = decompose_bcd(view.reachable, w, h, cand_axis)
            if cls2:
                ml, mc = merge_cells(lab2, cls2, w, h)
                cand = evaluate(cand_axis, ml, mc, True)
                if cand:
                    candidates.append(cand)

    if not candidates:
        result.stats = {"error": "no reachable robot positions"}
        return result

    chosen = min(candidates, key=lambda c: c["estimate"])
    axis = chosen["axis"]
    entries = chosen["entries"]
    tried = {f"{c['axis']}/{'merged' if c['merged'] else 'raw'}": c["estimate"]
             for c in candidates}

    # ------------------------------------ real routes (A* connectors) for winner
    routes: list[list[int]] = []
    route_meta: list[dict] = []
    deferred_total = 0
    for e in entries:
        route, deferred = serpentine_over(e["lines"], spacing, link, tail_slack=0)
        deferred_total += len(deferred)
        if not route:
            continue
        routes.append(route)
        route_meta.append({"cid": e["cid"], "vertical": e["vertical"], "cells": e["cells"]})

    if not routes:
        result.stats = {"error": "no sweepable cells"}
        return result

    # ------------------------------------------------------------- sequencing
    tour = order_routes(routes, view.start, w, optimize=optimize_order)

    builder.extend([view.start])
    cursor = view.start
    for seq, (k, flipped) in enumerate(tour):
        route = routes[k][::-1] if flipped else routes[k]
        connector = G.astar(view.reachable, w, h, cursor, route[0],
                            max_expand=link_budget, diagonal=True)
        if connector is None:
            continue
        start_step = len(builder.path)
        builder.extend(connector[1:-1])
        builder.extend(route)
        cursor = route[-1]
        meta = route_meta[k]
        result.phases.append({
            "name": f"单元 #{seq + 1}（{meta['cells']} 格 · "
                    f"{'纵向' if meta['vertical'] else '横向'}扫描）",
            "kind": "region",
            "region": meta["cid"],
            "cells": meta["cells"],
            "orientation": "vertical" if meta["vertical"] else "horizontal",
            "start_step": start_step,
            "end_step": len(builder.path) - 1,
            "coverage": round(builder.coverage(), 4),
        })

    result.phases.insert(0, {
        "name": "主扫（BCD 单元分解 + 牛耕式）",
        "kind": "sweep",
        "start_step": 0,
        "end_step": len(builder.path) - 1,
        "coverage": round(builder.coverage(), 4),
        "regions": len(routes),
    })

    # ------------------------------------------------------- dead-zone retreats
    for pass_no in range(max_retreat_passes):
        if builder.coverage() >= target_cov:
            break
        uncovered = G.intersection(view.cleanable, invert(builder.covered))
        if not G.count(uncovered):
            break

        clusters_labels, cluster_sizes = G.label_components(uncovered, w, h, 4)
        clusters = sorted(cluster_sizes.items(), key=lambda kv: kv[1], reverse=True)
        progressed = False

        for cluster_id, size in clusters:
            cluster_cells = [i for i, lab in enumerate(clusters_labels) if lab == cluster_id]
            remaining = [c for c in cluster_cells if not builder.covered[c]]
            if not remaining:
                continue

            cluster_mask = bytearray(w * h)
            for c in remaining:
                cluster_mask[c] = 1

            candidate = G.intersection(view.reachable,
                                       G.dilate_square(cluster_mask, w, h, radius))
            if not G.count(candidate):
                continue

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
    sweep_steps = chosen["clean"]
    result.stats = {
        "steps": len(result.path),
        "algorithm": "bcd",
        "spacing_mode": spacing_mode,
        "sweep_spacing_cells": spacing,
        "decomposition_axis": axis,
        "cells": len(routes),
        "decomposition_estimates": tried,
        "merged": chosen["merged"],
        "estimate_steps": chosen["estimate"],
        # where the steps go: pure cleaning motion vs repositioning
        "sweep_steps": sweep_steps,
        "reposition_steps": max(0, len(result.path) - sweep_steps),
        "reposition_ratio": round(max(0, len(result.path) - sweep_steps)
                                  / max(1, len(result.path)), 4),
        "ideal_steps": round(cleanable / spacing, 1),
        "efficiency": round(cleanable / spacing / max(1, len(result.path)), 4),
        "deferred_intervals": deferred_total,
        "retreat_passes": sum(1 for p in result.phases if p["kind"] == "retreat"),
        "covered_cleanable_cells": covered_cleanable,
        "cleanable_cells": cleanable,
        "coverage_reachable": round(covered_cleanable / cleanable, 4),
        "target_coverage": target_cov,
        "reached_target": covered_cleanable / cleanable >= target_cov,
    }
    return result
