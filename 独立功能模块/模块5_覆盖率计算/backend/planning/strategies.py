"""覆盖策略接口与三种实现 —— 任务 3-7

对应规划文档 2.3。抽象基类 + 三种策略：

- :class:`BoustrophedonStrategy`—— 整体直接蛇形，适合简单矩形房间；
- :class:`BlockBoustrophedonStrategy`—— 先 BCD 分解再逐单元蛇形，
  **非凸户型的正式策略**；
- :class:`RandomWalkStrategy`—— 随机游走，作为对比实验的对照组。

架构价值（报告 2.1 可写）：策略可插拔，新增 DARP、Hilbert 曲线
等算法只需实现一个类，无需改动仿真主流程。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..core.rasterize import GridCell
from .astar import astar, astar_nearest
from .bcd import Cell, cover_cell_boustrophedon, decompose_bcd
from .deadzone import AntiDither, dead_zone_ratio, find_dead_zones, sweep_dead_zone
from .grid_ops import bfs_reachable, find_start, passable_mask
from .trajectory import (
    Trajectory, area_time_cost, coverage_ratio,
    mark_visited, repeat_rate, swath_mask,
)


@dataclass
class SimParams:
    """仿真参数 —— 全部可从前端调，对应任务组 3 的参数面板。"""

    clean_width: int = 2          # 清扫幅宽（格）。必须 ≥ 扫描行距
    scan_axis: str = "row"        # 'row' 逐行 / 'col' 逐列
    start: tuple[int, int] | None = None
    step_time: float = 1.0        # 单步耗时（秒），用于单位面积耗时
    cell_area: float = 1.0# 单格面积（m²），栅格未标定时按 1
    sweep_dead: bool = True       # 是否做死区补扫
    max_retry: int = 3            # 死区最大补扫次数
    revisit_penalty: float = 0.8  # A* 重复访问代价

    def normalized(self) -> "SimParams":
        return SimParams(
            clean_width=int(np.clip(self.clean_width, 1, 12)),
            scan_axis="col" if self.scan_axis == "col" else "row",
            start=self.start,
            step_time=float(np.clip(self.step_time, 0.01, 100)),
            cell_area=float(np.clip(self.cell_area, 1e-4, 1e4)),
            sweep_dead=bool(self.sweep_dead),
            max_retry=int(np.clip(self.max_retry, 0, 10)),
            revisit_penalty=float(np.clip(self.revisit_penalty, 0.0, 20.0)),
        )


@dataclass
class SimResult:
    """一次仿真的完整结果。"""

    path: list                     # 轨迹点序列
    grid: np.ndarray               # 仿真后的栅格（已清扫标记为 2）
    cleaned: np.ndarray            # 清扫带掩码
    visited: np.ndarray            # 中心线掩码
    reachable: np.ndarray          # 可达区掩码
    metrics: dict                  # 指标
    cells: list = field(default_factory=list)   # BCD 单元（调试与可视化）
    dead_zones: list = field(default_factory=list)
    guard_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics,
            "grid": self.grid.tolist(),
            "cleaned": self.cleaned.astype(int).tolist(),
            "reachable": self.reachable.astype(int).tolist(),
            "path": [[int(r), int(c)] for r, c in self.path],
            "cells": [{"bbox": list(c.bbox), "area": c.area} for c in self.cells],
            "dead_zones": [z.to_dict() for z in self.dead_zones],
            "guard": self.guard_stats,
        }


class CoverageStrategy(ABC):
    """覆盖策略抽象接口（任务 3-7）。

    子类只需实现 :meth:`plan`，把「可通行栅格 + 起点」变成一条轨迹。
    指标的统计与渲染由基类的 :meth:`simulate` 统一处理，
    保证不同策略的指标口径完全一致 —— 否则对比实验就没有可比性。
    """

    name = "base"
    description = "抽象基类"

    @abstractmethod
    def plan(self, grid: np.ndarray, start: tuple[int, int],
             params: SimParams, reachable: np.ndarray,
             guard: AntiDither) -> tuple[list[tuple[int, int]], list]:
        """规划轨迹。

        返回 ``(path, cells)``：``path`` 为轨迹点序列，
        ``cells`` 为解出的单元（可为空，仅用于可视化）。
        """

    # ---- 公共流程：所有策略共用，保证指标口径一致 ----
    def simulate(self, grid: np.ndarray, params: SimParams) -> SimResult:
        p = params.normalized()
        g = grid.copy()
        start = find_start(g, p.start)
        reachable = bfs_reachable(g, start)

        if not reachable.any():
            return _empty_result(g, start)

        guard = AntiDither(max_retry=p.max_retry, revisit_penalty=p.revisit_penalty)
        path, cells = self.plan(g, start, p, reachable, guard)

        visited = mark_visited(g, path)
        cleaned = swath_mask(g, path, p.clean_width)

        # --- 死区处理（步骤 4） ---
        dead_zones: list = []
        if p.sweep_dead and path:
            dead_zones = find_dead_zones(cleaned, g, reachable)
            visited_count = np.zeros(g.shape, dtype=np.int32)
            cur = path[-1]
            for z in dead_zones:
                if not z.reachable:
                    continue      # 物理死区，补扫也没用
                extra = sweep_dead_zone(z, g, cur, visited, visited_count,
                                        p.clean_width, guard)
                if extra is None:
                    continue
                path.extend(extra)
                cur = extra[-1]
            if len(path) != len(mark_visited(g, path)):
                visited = mark_visited(g, path)
                cleaned = swath_mask(g, path, p.clean_width)

        # --- 路径连续化（补齐「跳跃点」） ---
        # 各策略把多条扫描线直接 extend 到一起，拼接处会残留「不相邻」的
        # 跳跃点：数据上压墙步数为 0（两端都是可通行格），但前端把两点
        # 连成直线就会穿墙。在统一出口处一次性补齐，比各策略各修一遍可靠。
        #
        # 顺序很关键：**必须在算指标之前**补齐。
        # 早期版本把补齐放在指标之后，想让重复率维持在低位，理由是
        # 「补齐点本就在覆盖掩码里，不该计入分母」。但这条路走不通：
        #   ① `module9.py` 造回放帧用的是补齐**后**的路径长度，
        #      而返回的 total_steps 是补齐**前**的，前端进度条会显示
        #      「4377 / 3646 步」并卡在 88%；
        #   ② 「假装机器人能瞬移」在物理上不成立，重复率正是用来
        #      度量这种冗余的，掩盖它等于把指标作废。
        # 补齐点代表机器人真实走过的过渡步（单元间转场、换扫描线），
        # 计入后语义才对得上。现在三者完全自洽：
        # 路径点数 = total_steps + 1 = 前端帧号上限。
        path = _densify_path(g, path, p.revisit_penalty)

        # 补齐可能引入新的访问格，须重算掩码
        visited = mark_visited(g, path)
        cleaned = swath_mask(g, path, p.clean_width)

        # --- 指标（口径见规划文档 2.5） ---
        total_steps = max(0, len(path) - 1)

        metrics = {
            "coverage": coverage_ratio(cleaned, reachable),
            "repeat_rate": repeat_rate(visited, total_steps),
            "dead_zone_ratio": dead_zone_ratio(g, reachable),
            "total_steps": total_steps,
            "area_time_cost": area_time_cost(total_steps, p.step_time,
                                             int(np.count_nonzero(reachable)),
                                             p.cell_area),
            "reachable_cells": int(np.count_nonzero(reachable)),
            "cleaned_cells": int(np.count_nonzero(cleaned & reachable)),
            "passable_cells": int(np.count_nonzero(passable_mask(g))),
            "unique_visited": int(np.count_nonzero(visited)),
            "cell_count": int(cleaned.size),
            "strategy": self.name,
            "start": [int(start[0]), int(start[1])],
            "uncovered_cells": int(np.count_nonzero(reachable & ~cleaned)),
            "algo_miss": sum(1 for z in dead_zones if z.reachable),
        }

        # 输出栅格：已清扫标记为 2，供前端三色渲染
        out = g.copy()
        out[cleaned] = GridCell.CLEANED

        return SimResult(
            path=path, grid=out, cleaned=cleaned, visited=visited,
            reachable=reachable, metrics=metrics, cells=cells,
            dead_zones=dead_zones, guard_stats=guard.stats(),
        )


def _empty_result(grid: np.ndarray, start: tuple[int, int]) -> SimResult:
    empty = np.zeros(grid.shape, dtype=bool)
    return SimResult(
        path=[start], grid=grid, cleaned=empty, visited=empty.copy(),
        reachable=empty.copy(),
        metrics={"coverage": 0.0, "repeat_rate": 0.0, "dead_zone_ratio": 1.0,
                 "total_steps": 0, "area_time_cost": 0.0,
                 "reachable_cells": 0, "cleaned_cells": 0,
                 "passable_cells": 0, "unique_visited": 0,
                 "cell_count": int(grid.size), "strategy": "none",
                 "start": [int(start[0]), int(start[1])],
                 "uncovered_cells": 0, "algo_miss": 0},
    )


def _transfer(grid: np.ndarray, cur: tuple[int, int],
              targets: list[tuple[int, int]],
              visited_count: np.ndarray, penalty: float) -> list[tuple[int, int]] | None:
    """单元间转场 —— 步骤 3：A* 规划到下一单元最近可达点的最短路。"""
    if not targets:
        return None
    res = astar_nearest(grid, cur, targets,
                        visited=visited_count > 0, revisit_penalty=penalty)
    if res is None:
        return None
    path, _ = res
    for p in path:
        visited_count[p[0], p[1]] += 1
    return path


class BoustrophedonStrategy(CoverageStrategy):
    """基础牛耕式：不做 BCD 区域分解，仅按扫描线分段蛇形。

    适合**凸的矩形房间**。非凸户型覆盖率会掉下来——因为不分解区域，
    扫描线断开后要靠 A* 跨过去，绕路会拉长路径。

    实现说明（曾修过一个穿墙 bug）
    --------------------------------
    早期版本把整个可达区当成**一个 Cell** 丢给
    :func:`cover_cell_boustrophedon`，而该函数的``_connect()``
    在跨行时返回空序列（它假设调用方会补A* 转场）。
    分块策略确实补了，但基础策略没补，于是相邻扫描线之间
    残留了「跳跃点」—— 数据里压墙步数为 0（点本身都在可通行格），
    可前端把两点连成直线就穿墙了。实测 real_fp_02 上有 42 处这样的跳跃。

    修法：把每条扫描线当作一个独立块，块间用 A* 转场连接。
    仍然不做 BCD 分解（这是它与分块策略的本质区别），
    但保证了轨迹连续、不穿墙。
    """

    name = "boustrophedon"
    description = "基础牛耕式（整体蛇形，适合矩形房间）"

    def plan(self, grid, start, params, reachable, guard):
        whole = Cell(np.argwhere(reachable), index=0)
        seg = cover_cell_boustrophedon(reachable, whole,
                                       params.clean_width, start,
                                       params.scan_axis)
        if not seg:
            return [start], [whole]

        path: list[tuple[int, int]] = [start]
        visited_count = np.zeros(grid.shape, dtype=np.int32)
        visited_count[start[0], start[1]] += 1
        cur = start

        # 把扫描线序列切成「连续段」，段间用 A* 转场，
        # 避免相邻路径点不相邻导致前端连线穿墙。
        # 起点必须一并纳入切分判定：若 seg 的首点离 start 有距离，
        # 单独成块，由第一次 A* 转场填过去。
        for chunk in _split_by_jump([start] + list(seg)):
            if not chunk:
                continue
            transfer = _transfer(grid, cur, list(chunk), visited_count,
                                 params.revisit_penalty)
            if transfer:
                path.extend(transfer[1:])
                cur = transfer[-1]
            path.extend(chunk)
            cur = chunk[-1]

        return path, [whole]


def _split_by_jump(seg: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """把路径点序列切成「相邻点连续」的若干块。

    切分依据：若 seg[i+1] 与 seg[i] 的切比雪夫距离 > 1，
    说明两点之间存在跳跃（换行时没有补 A*），此处必须断开，
    由调用方用 A* 转场填补，否则可视化时直线会穿墙。
    """
    chunks: list[list[tuple[int, int]]] = []
    for p in seg:
        pt = (int(p[0]), int(p[1]))
        if chunks:
            pr, pc = chunks[-1][-1]
            if max(abs(pt[0] - pr), abs(pt[1] - pc)) > 1:
                chunks.append([pt])
                continue
        else:
            chunks.append([pt])
            continue
        chunks[-1].append(pt)
    return chunks


class BlockBoustrophedonStrategy(CoverageStrategy):
    """分块牛耕式：**非凸户型的正式策略**（任务 3-4）。

    流程：BCD 分解 → 逐单元蛇形 → 单元间 A* 转场。
    每个单元内部可 100% 覆盖，单元间空隙由转场补上，
    因此整体覆盖率能接近 100%。
    """

    name = "block_boustrophedon"
    description = "分块牛耕式（BCD 分解 + A* 转场，正式策略）"

    def plan(self, grid, start, params, reachable, guard):
        cells = decompose_bcd(reachable)
        if not cells:
            cells = [Cell(np.argwhere(reachable), index=0)]

        path: list[tuple[int, int]] = [start]
        visited_count = np.zeros(grid.shape, dtype=np.int32)
        visited_count[start[0], start[1]] += 1
        cur = start

        # 先扫起点所在的单元，避免一开始就长途转场
        cells.sort(key=lambda c: 0 if start in {tuple(p) for p in c.cells} else 1)

        for cell in cells:
            if not cell.is_scanable():
                # 退化单元（太窄）：直接 A* 穿过去
                targets = [tuple(p) for p in cell.cells]
                seg = _transfer(grid, cur, targets, visited_count,
                                params.revisit_penalty)
                if seg:
                    path.extend(seg[1:])
                    cur = seg[-1]
                continue

            seg = cover_cell_boustrophedon(reachable, cell,
                                          params.clean_width, None,
                                          params.scan_axis)
            if not seg:
                continue
            entry = seg[0]
            # 转场到本单元的入口
            transfer = _transfer(grid, cur, list(seg), visited_count,
                                 params.revisit_penalty)
            if transfer:
                path.extend(transfer[1:])
                cur = transfer[-1]
            path.extend(seg)
            cur = seg[-1]

        return path, cells


class RandomWalkStrategy(CoverageStrategy):
    """随机游走对照组（任务 3-10）。

    用途：作为对比实验的基线，证明 BCD 策略在覆盖率与重复率上确有优势。
    实现为「向邻近可通行格随机移动」，但强制朝未覆盖格走，
    否则会长时间原地打转 —— 那样对比就不公平了。
    """

    name = "random_walk"
    description = "随机游走（对照组）"

    def plan(self, grid, start, params, reachable, guard, seed: int = 42):
        rng = np.random.default_rng(seed)
        cur = start
        path = [start]
        visited_count = np.zeros(grid.shape, dtype=np.int32)
        visited_count[cur] = 1
        h, w = grid.shape
        budget = int(np.count_nonzero(reachable)) * 6   # 步数上限，防跑飞

        for _ in range(budget):
            # 候选邻格：可通行优先，未覆盖优先
            nbrs = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != GridCell.BLOCKED:
                    nbrs.append((nr, nc))
            if not nbrs:
                break
            fresh = [p for p in nbrs if visited_count[p] == 0]
            pool = fresh if fresh else nbrs
            cur = pool[int(rng.integers(len(pool)))]
            path.append(cur)
            visited_count[cur] += 1
            if not fresh:
                # 陷入死胡同就跳回起点重来
                cur = start
                path.append(cur)
        return path, []


STRATEGIES: dict[str, type[CoverageStrategy]] = {
    BoustrophedonStrategy.name: BoustrophedonStrategy,
    BlockBoustrophedonStrategy.name: BlockBoustrophedonStrategy,
    RandomWalkStrategy.name: RandomWalkStrategy,
}


def get_strategy(name: str) -> CoverageStrategy:
    cls = STRATEGIES.get(name, BlockBoustrophedonStrategy)
    return cls()


def _densify_path(grid: np.ndarray, path: list[tuple[int, int]],
                  penalty: float) -> list[tuple[int, int]]:
    """补齐路径中所有「跳跃点」，使相邻点恒为八邻接。

    背景
    ----
    各策略把多条扫描线/ 多个单元的蛇形点序列直接 ``extend`` 到一起，
    但扫描线之间的切换点并不总是相邻的—— ``cover_cell_boustrophedon``
    的 ``_connect()`` 在跨行时故意返回空序列，假设调用方会补 A*转场；
    而有些调用路径（基础牛耕整体蛇形、死区补扫拼接）忘了补。
    数据上这些跳跃点**压墙步数为 0**（两端都是可通行格），
    可前端把两点连成直线就会穿墙—— 属于「指标正常但显示错误」的隐蔽缺陷。

    做法
    ----
    在仿真统一出口处扫一遍，凡相邻点不八邻接就用 A* 补齐。
    补不出来的（理论不该发生，A* 不可达）也照样记录，
    由 :mod:`backend.scripts.check_path` 的断言去暴露。
    """
    if len(path) < 2:
        return [tuple(map(int, p)) for p in path]

    out: list[tuple[int, int]] = [(int(path[0][0]), int(path[0][1]))]
    for i in range(1, len(path)):
        cur = (int(path[i][0]), int(path[i][1]))
        prev = out[-1]
        if cur == prev:
            continue
        if max(abs(cur[0] - prev[0]), abs(cur[1] - prev[1])) > 1:
            seg = astar(grid, prev, cur, revisit_penalty=penalty)
            if seg:
                out.extend((int(p[0]), int(p[1])) for p in seg[1:])
                continue
        out.append(cur)
    return out
