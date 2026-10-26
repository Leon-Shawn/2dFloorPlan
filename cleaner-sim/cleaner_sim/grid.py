"""Core raster-grid primitives.

Everything operates on flat `bytearray` masks of length ``width * height`` in
row-major order (index = ``y * width + x``), which keeps the whole pipeline
dependency-free and fast enough for pure Python on the grid sizes this project
uses (tens of thousands of cells).

Conventions used across the project
-----------------------------------
* mask value ``1`` = true/in-set, ``0`` = false.
* ``passable``  = floor cells (the area that *should* be cleaned).
* ``center_ok`` = cells where the robot's body fits without touching a wall
  (``passable`` eroded by the robot's square footprint). Only these can be robot
  positions.
"""
from __future__ import annotations

import base64
import heapq
from collections import deque
from functools import lru_cache


# --------------------------------------------------------------------------- disks
@lru_cache(maxsize=64)
def disk_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """Integer offsets of a filled disk of `radius` cells (Euclidean rule)."""
    r2 = radius * radius
    return tuple(
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= r2
    )


def stamp(mask: bytearray, width: int, height: int, cx: int, cy: int,
          offsets, value: int = 1) -> int:
    """Paint `offsets` around (cx, cy) into `mask`; returns cells actually changed."""
    changed = 0
    for dx, dy in offsets:
        x, y = cx + dx, cy + dy
        if 0 <= x < width and 0 <= y < height:
            i = y * width + x
            if mask[i] != value:
                mask[i] = value
                changed += 1
    return changed


def erode(src: bytearray, width: int, height: int, radius: int) -> bytearray:
    """Cells whose whole disk of `radius` lies inside `src` (and inside the map)."""
    out = bytearray(width * height)
    if radius <= 0:
        out[:] = src
        return out
    offsets = disk_offsets(radius)
    for y in range(radius, height - radius):
        row = y * width
        for x in range(radius, width - radius):
            if not src[row + x]:
                continue
            for dx, dy in offsets:
                if not src[(y + dy) * width + x + dx]:
                    break
            else:
                out[row + x] = 1
    return out


def dilate(src: bytearray, width: int, height: int, radius: int) -> bytearray:
    """Cells within `radius` (Euclidean) of any set cell in `src`."""
    out = bytearray(width * height)
    offsets = disk_offsets(radius)
    for y in range(height):
        row = y * width
        for x in range(width):
            if src[row + x]:
                stamp(out, width, height, x, y, offsets)
    return out


# --------------------------------------------------- square footprint (separable)
# The robot is modelled as a square footprint of side (2r+1) cells, which is what
# grid coverage planners conventionally assume and what the machine physically
# projects. Unlike a disk it tiles the plane exactly, so a boustrophedon sweep with
# spacing 2r leaves no scalloped corners along the walls. Both operations are
# separable, so they run in a single O(w*h) pass per axis.
def erode_square(src: bytearray, width: int, height: int, radius: int) -> bytearray:
    """Cells whose whole (2r+1)^2 square lies inside `src` and inside the map."""
    if radius <= 0:
        out = bytearray(src)
        return out
    k = 2 * radius + 1
    tmp = bytearray(width * height)
    for y in range(height):
        base = y * width
        run = 0
        for x in range(width):
            if src[base + x]:
                run += 1
            if x >= k and src[base + x - k]:
                run -= 1
            if x >= k - 1 and run == k:
                tmp[base + x - radius] = 1

    out = bytearray(width * height)
    for x in range(width):
        run = 0
        for y in range(height):
            if tmp[y * width + x]:
                run += 1
            if y >= k and tmp[(y - k) * width + x]:
                run -= 1
            if y >= k - 1 and run == k:
                out[(y - radius) * width + x] = 1
    return out


def dilate_square(src: bytearray, width: int, height: int, radius: int) -> bytearray:
    """Cells whose (2r+1)^2 square intersects any set cell in `src`."""
    if radius <= 0:
        return bytearray(src)
    k = 2 * radius + 1
    tmp = bytearray(width * height)
    for y in range(height):
        base = y * width
        run = 0
        for x in range(width):
            if src[base + x]:
                run += 1
            if x >= k and src[base + x - k]:
                run -= 1
            if x >= k - 1 and run:
                tmp[base + x - radius] = 1

    out = bytearray(width * height)
    for x in range(width):
        run = 0
        for y in range(height):
            if tmp[y * width + x]:
                run += 1
            if y >= k and tmp[(y - k) * width + x]:
                run -= 1
            if y >= k - 1 and run:
                out[(y - radius) * width + x] = 1
    return out


def stamp_square(mask: bytearray, width: int, height: int, cx: int, cy: int,
                 radius: int, value: int = 1) -> None:
    """Paint the (2r+1)^2 square centred on (cx, cy) using row slices."""
    x0 = max(0, cx - radius)
    x1 = min(width - 1, cx + radius)
    if x1 < x0:
        return
    fill = bytes([value]) * (x1 - x0 + 1)
    for y in range(max(0, cy - radius), min(height - 1, cy + radius) + 1):
        base = y * width
        mask[base + x0:base + x1 + 1] = fill


def _check_same_len(a, b) -> None:
    """Guard the zip-based set operations below against silent truncation."""
    if len(a) != len(b):
        raise ValueError(f"mask size mismatch: {len(a)} != {len(b)}")


def union(a: bytearray, b: bytearray) -> bytearray:
    _check_same_len(a, b)
    return bytearray(1 if (x or y) else 0 for x, y in zip(a, b))


def intersection(a: bytearray, b: bytearray) -> bytearray:
    _check_same_len(a, b)
    return bytearray(1 if (x and y) else 0 for x, y in zip(a, b))


def difference(a: bytearray, b: bytearray) -> bytearray:
    _check_same_len(a, b)
    return bytearray(1 if (x and not y) else 0 for x, y in zip(a, b))


def count(mask) -> int:
    return sum(mask)


# ------------------------------------------------------------------- connectivity
def label_components(mask: bytearray, width: int, height: int,
                     connectivity: int = 4):
    """Label connected components of `mask`.

    Returns ``(labels, sizes)`` where ``labels[i]`` is 0 for cells outside the
    mask and otherwise a 1-based component id, and ``sizes`` maps id -> cell count.
    """
    if connectivity == 8:
        neighbours = ((-1, -1), (0, -1), (1, -1), (-1, 0),
                      (1, 0), (-1, 1), (0, 1), (1, 1))
    else:
        neighbours = ((0, -1), (-1, 0), (1, 0), (0, 1))

    labels = [0] * (width * height)
    sizes: dict[int, int] = {}
    current = 0
    for start in range(width * height):
        if not mask[start] or labels[start]:
            continue
        current += 1
        labels[start] = current
        q = deque([start])
        size = 0
        while q:
            idx = q.popleft()
            size += 1
            y, x = divmod(idx, width)
            for dx, dy in neighbours:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    n = ny * width + nx
                    if mask[n] and not labels[n]:
                        labels[n] = current
                        q.append(n)
        sizes[current] = size
    return labels, sizes


def distance_from_walls(passable: bytearray, width: int, height: int) -> list[int]:
    """4-connected distance (in cells) from every cell to the nearest non-passable cell."""
    n = width * height
    dist = [-1] * n
    q = deque()
    for i in range(n):
        if not passable[i]:
            dist[i] = 0
            q.append(i)
    while q:
        idx = q.popleft()
        y, x = divmod(idx, width)
        nd = dist[idx] + 1
        if y > 0 and dist[idx - width] < 0:
            dist[idx - width] = nd
            q.append(idx - width)
        if y + 1 < height and dist[idx + width] < 0:
            dist[idx + width] = nd
            q.append(idx + width)
        if x > 0 and dist[idx - 1] < 0:
            dist[idx - 1] = nd
            q.append(idx - 1)
        if x + 1 < width and dist[idx + 1] < 0:
            dist[idx + 1] = nd
            q.append(idx + 1)
    return dist


def most_open_cell(center_ok: bytearray, width: int, height: int, candidates=None) -> int | None:
    """The reachable cell furthest from any wall — a natural robot start point."""
    dist = distance_from_walls(center_ok, width, height)
    best, best_d = None, -1
    rng = candidates if candidates is not None else range(width * height)
    for i in rng:
        if center_ok[i] and dist[i] > best_d:
            best, best_d = i, dist[i]
    return best


# --------------------------------------------------------------------------- A*
_DIAG = 1.4142


def astar(passable: bytearray, width: int, height: int, start: int, goal: int,
          max_expand: int = 400_000, diagonal: bool = True) -> list[int] | None:
    """Shortest 4/8-connected path through `passable`, as a list of cell indices.

    Returns ``[start]`` when start == goal, or ``None`` when no route exists
    (or the expansion budget runs out, which keeps pathological maps bounded).
    """
    if start == goal:
        return [start]
    if not passable[start] or not passable[goal]:
        return None

    if diagonal:
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, _DIAG), (1, -1, _DIAG), (-1, 1, _DIAG), (-1, -1, _DIAG))

        def h(idx: int) -> float:
            y, x = divmod(idx, width)
            gy, gx = divmod(goal, width)
            dx, dy = abs(x - gx), abs(y - gy)
            return (dx + dy) + (_DIAG - 2) * min(dx, dy)
    else:
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0))

        def h(idx: int) -> float:
            y, x = divmod(idx, width)
            gy, gx = divmod(goal, width)
            return abs(x - gx) + abs(y - gy)

    g_score = {start: 0.0}
    came: dict[int, int] = {}
    open_heap = [(h(start), 0.0, start)]
    closed = set()
    expanded = 0

    while open_heap:
        _, g, idx = heapq.heappop(open_heap)
        if idx == goal:
            path = [idx]
            while idx in came:
                idx = came[idx]
                path.append(idx)
            path.reverse()
            return path
        if idx in closed:
            continue
        closed.add(idx)
        expanded += 1
        if expanded > max_expand:
            return None

        y, x = divmod(idx, width)
        for dx, dy, cost in steps:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            n = ny * width + nx
            if not passable[n]:
                continue
            if diagonal and dx and dy:
                # never squeeze diagonally through a wall corner
                if not (passable[y * width + nx] or passable[ny * width + x]):
                    continue
            ng = g + cost
            if ng < g_score.get(n, float("inf")):
                g_score[n] = ng
                came[n] = idx
                heapq.heappush(open_heap, (ng + h(n), ng, n))
    return None


# --------------------------------------------------------------------- encoding
def pack_bits(mask) -> bytes:
    """Pack a 0/1 mask into one bit per cell (LSB-first within each byte)."""
    out = bytearray((len(mask) + 7) // 8)
    for i, v in enumerate(mask):
        if v:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# movement codes shared with the browser client (see web/app.js)
DIR_CODES = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3,
             (1, -1): 4, (1, 1): 5, (-1, 1): 6, (-1, -1): 7}


def encode_path(path: list[int], width: int) -> bytes:
    """Encode a list of adjacent cell indices as one movement code per step."""
    out = bytearray()
    for a, b in zip(path, path[1:]):
        ay, ax = divmod(a, width)
        by, bx = divmod(b, width)
        code = DIR_CODES.get((bx - ax, by - ay))
        if code is None:
            raise ValueError(f"path is not contiguous: {a} -> {b}")
        out.append(code)
    return bytes(out)


def downsample_max(mask: bytearray, width: int, height: int, factor: int):
    """Box-downsample keeping the maximum — thin walls survive the reduction."""
    if factor <= 1:
        return mask, width, height
    nw, nh = max(1, width // factor), max(1, height // factor)
    out = bytearray(nw * nh)
    for y in range(nh):
        for x in range(nw):
            acc = 0
            for dy in range(factor):
                sy = y * factor + dy
                if sy >= height:
                    break
                row = sy * width
                for dx in range(factor):
                    sx = x * factor + dx
                    if sx < width and mask[row + sx]:
                        acc = 1
                        break
                if acc:
                    break
            out[y * nw + x] = acc
    return out, nw, nh
