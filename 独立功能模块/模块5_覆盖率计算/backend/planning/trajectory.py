"""轨迹与清扫带 —— 模块 3 的轨迹数据结构

**关键概念：清扫幅宽（cleaning swath）**

机器人的实际覆盖不是「中心线经过的格子」，而是「中心线两侧
``clean_width`` 范围内的所有格子」。扫地机器人的刷盘直径约 18cm，
对应栅格就是若干格。

这一点决定了覆盖率怎么算：
- 若按「路径点集合」算覆盖率，蛇形隔行扫描最多只能到 50%，
  看起来像是算法有bug，其实只是没考虑刷盘宽度；
- 按「路径膨胀 clean_width 后的覆盖带」算，才与物理一致，
  也才能和规划文档 2.5 节的口径对上。

同时，清扫带宽度必须 ≥ 扫描行距，否则一定漏扫 —— 这是
规划文档 2.2 步骤 2 强调「行距必须 ≤ c」的原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.rasterize import GridCell
from .grid_ops import DIRS4


@dataclass
class Trajectory:
    """一条仿真轨迹。

    属性
    ----
    path: 机器人中心经过的格子序列 ``[(r,c), ...]``。
    cleaned: 被实际清扫到的格子布尔掩码（路径按幅宽膨胀后的并集）。
    visited: 中心线经过过的格子布尔掩格，用于重复率统计。
    """

    path: list[tuple[int, int]] = field(default_factory=list)
    cleaned: np.ndarray | None = None
    visited: np.ndarray | None = None

    @property
    def total_steps(self) -> int:
        """总步数 —— 沿轨迹移动的次数。

        指标口径要求 ``total_steps`` **包含单元间转场步数**：
        转场必然重复经过已清扫栅格，如实统计才反映真实效率。
        """
        return max(0, len(self.path) - 1)

    @property
    def unique_steps(self) -> int:
        return int(np.count_nonzero(self.visited)) if self.visited is not None else 0

    def append(self, cell: tuple[int, int]) -> None:
        self.path.append(cell)


def mark_visited(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    """轨迹中心线经过的格子掩码。"""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    for r, c in path:
        if 0 <= r < h and 0 <= c < w:
            visited[r, c] = True
    return visited


def swath_mask(grid: np.ndarray, path: list[tuple[int, int]],
               clean_width: int = 2) -> np.ndarray:
    """轨迹的清扫带 —— 中心线按 ``clean_width`` 膨胀后的并集。

    参数
    ----
    clean_width: 清扫幅宽（格）。1 表示只覆盖中心格与四邻，
        2 表示覆盖 5×5 范围内的格，以此类推。

    实现：距离变换。对可通行区算到最近路径点的距离，
    距离 ≤ clean_width 的格子即为被清扫到。
    这样做比逐点膨胀高效，且天然贴合栅格连通性。
    """
    import cv2

    h, w = grid.shape
    if not path:
        return np.zeros((h, w), dtype=bool)

    # 距离变换算的是「到最近 0 像素的距离」，所以：
    #   轨迹点 = 0（作为距离源），其余 = 1
    # 注意这里**不能**把可通行区也置 0，否则距离处处为 0，清扫带会铺满全图。
    src = np.ones((h, w), dtype=np.uint8)
    for r, c in path:
        if 0 <= r < h and 0 <= c < w:
            src[r, c] = 0

    if not (src == 0).any():
        return np.zeros((h, w), dtype=bool)

    dist = cv2.distanceTransform(src, cv2.DIST_L2, 3)
    passable = (grid != GridCell.BLOCKED)
    return (dist <= float(clean_width)) & passable


def coverage_ratio(cleaned: np.ndarray, reachable: np.ndarray) -> float:
    """覆盖率 = |cleaned ∩ reachable| / |reachable|（规划文档 2.5 定版口径）。

    分母用 ``reachable`` 而非全部可通行格：
    被家具完全包围的死区物理上不可达，算进去会让覆盖率永远到不了 100%，
    且分不清是「算法漏扫」还是「地图断了」。
    """
    denom = int(np.count_nonzero(reachable))
    if denom == 0:
        return 0.0
    hit = int(np.count_nonzero(cleaned & reachable))
    return hit / denom


def repeat_rate(visited: np.ndarray, total_steps: int) -> float:
    """重复率 = 1 − |unique_visited| / |total_steps|。

    ``total_steps`` 含单元间转场步数 —— 转场必然重复经过已清扫栅格，
    如实统计才能反映真实的路径效率。转场越远、单元划分越碎，重复率越高。
    """
    if total_steps <= 0:
        return 0.0
    unique = int(np.count_nonzero(visited))
    return max(0.0, 1.0 - unique / total_steps)


def area_time_cost(total_steps: int, step_time: float,
                   reachable_cells: int, cell_area: float = 1.0) -> float:
    """单位面积耗时 = total_steps × step_time / (|reachable| × cell_area)。"""
    denom = reachable_cells * cell_area
    if denom <= 0:
        return 0.0
    return total_steps * step_time / denom
