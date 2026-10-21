"""A* 网格最短路 —— 模块 3 步骤 3（单元间转场）与步骤 4（死角补扫）

八连通，允许斜向移动（斜向代价 √2，对齐栅格地图的物理意义：
斜着走比直走远）。

**防震荡第一道防线就落在代价函数里**：
``代价 = 移动代价 + 已访问栅格 × w``
走过的格子会越来越贵，从根本上抑制「来回绕圈」。

其他两道防线在 :mod:`backend.planning.deadzone` 里。
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from ..core.rasterize import GridCell
from .grid_ops import DIRS8

# 斜向移动的实际代价（√2 ≈ 1.414），使启发函数保持可采纳（admissible）
_DIAG = math.sqrt(2.0)


def _heuristic(r: float, c: float, goal: tuple[int, int]) -> float:
    """八连通下的曼哈顿式启发（用对角距离），保证不高估最优代价。"""
    dr = abs(r - goal[0])
    dc = abs(c - goal[1])
    return (dr + dc) + (_DIAG - 2.0) * min(dr, dc)


def astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
          visited: np.ndarray | None = None,
          revisit_penalty: float = 0.0,
          allow_diagonal: bool = True) -> list[tuple[int, int]] | None:
    """在栅格图上求 ``start → goal`` 的最短路。

    参数
    ----
    grid: 栅格矩阵，``GridCell.BLOCKED`` 不可通行。
    start / goal: ``(row, col)``。
    visited: 已访问掩码；与 ``revisit_penalty`` 配合实现
        「重复访问代价」——防震荡第一道防线。``None`` 表示不启用。
    revisit_penalty: 每经过一个已访问栅格追加的代价。0 表示不惩罚。
    allow_diagonal: 是否允许八连通。

    返回
    ----
    路径点列表（含起点与终点），不可达时返回 ``None``。
    """
    h, w = grid.shape
    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None
    if grid[sr, sc] == GridCell.BLOCKED or grid[gr, gc] == GridCell.BLOCKED:
        return None
    if start == goal:
        return [start]

    nbrs = DIRS8 if allow_diagonal else DIRS8[:4]
    use_penalty = visited is not None and revisit_penalty > 0

    best: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap: list[tuple[float, float, tuple[int, int]]] = [
        (_heuristic(sr, sc, goal), 0.0, start)]
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            return _reconstruct(parent, cur)
        closed.add(cur)
        r, c = cur

        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc] == GridCell.BLOCKED:
                continue
            # 斜向移动不许「穿墙角」：两侧正交格必须至少一侧可通行
            if dr and dc:
                if grid[r + dr, c] == GridCell.BLOCKED and \
                        grid[r, c + dc] == GridCell.BLOCKED:
                    continue
            move = _DIAG if (dr and dc) else 1.0
            if use_penalty and visited[nr, nc]:
                move += revisit_penalty
            ng = g + move

            if ng < best.get((nr, nc), math.inf) - 1e-9:
                best[(nr, nc)] = ng
                parent[(nr, nc)] = cur
                heapq.heappush(open_heap, (ng + _heuristic(nr, nc, goal), ng, (nr, nc)))

    return None


def _reconstruct(parent: dict, node: tuple[int, int]) -> list[tuple[int, int]]:
    path = [node]
    while node in parent:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path


def astar_nearest(grid: np.ndarray, start: tuple[int, int],
                  targets: list[tuple[int, int]],
                  visited: np.ndarray | None = None,
                  revisit_penalty: float = 0.0) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """找 ``targets`` 中离 ``start`` 最近且可达的一个，返回其路径。

    转场时下一个单元的起点往往不是一个确定的点（可能是它边界上的任意格），
    所以需要「在候选集合里挑最近的可达点」。
    """
    best: tuple[float, list, tuple[int, int]] | None = None
    for t in targets:
        path = astar(grid, start, t, visited, revisit_penalty)
        if path is None:
            continue
        cost = float(len(path))
        if best is None or cost < best[0]:
            best = (cost, path, t)
    if best is None:
        return None
    return best[1], best[2]
