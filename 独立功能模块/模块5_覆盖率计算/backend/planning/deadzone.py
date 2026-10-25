"""死区识别、成因分类与防震荡三道防线 —— 模块 3 步骤 4（任务 3-5 / 3-6）

对应规划文档 2.2 步骤 4。

**核心设计原则**（规划文档 2.4 明确要求）：
覆盖率不到 100% 时，系统必须能**自证原因**——是「地图断了」
（物理死区，有清单和成因）还是「算法漏了」（bug，可定位到具体坐标）。
这个自证能力既是调试利器，也是报告里的加分项。

因此本模块把所有未覆盖栅格分成两类，**绝不混淆**：

============ ====================================================
类型          判定依据
============ ====================================================
物理死区      从起点 BFS 根本到不了 → 不可能覆盖，不算算法失败
算法漏扫      BFS 能到但轨迹没经过 → 这是 bug，必须能定位到坐标
============ ====================================================

防震荡三道防线
--------------
1. **重复访问代价**：落在 :func:`backend.planning.astar.astar` 的代价函数里；
2. **最大重试次数**：同一死区补扫超K 次则标记放弃；
3. **震荡检测**：滑动窗口内高频进出同一区域 → 判定震荡 → 重新划分区域。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.rasterize import GridCell
from .astar import astar
from .grid_ops import bbox_of, components_of, passable_mask

# 死区成因标签
CAUSE_FURNITURE = "被家具包围"
CAUSE_NARROW = "通道窄于机身"
CAUSE_SEALED = "门洞未打通"
CAUSE_ALGO = "算法漏扫（可达但未覆盖）"


@dataclass
class DeadZone:
    """一处死区及其诊断信息。"""

    cells: np.ndarray           # (N,2) 坐标
    cause: str                  # 成因标签
    reachable: bool             # False = 物理死区，不可达
    retried: int = 0            # 已补扫次数
    exhausted: bool = False     # 补扫超限放弃

    @property
    def area(self) -> int:
        return int(len(self.cells))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return bbox_of(self.cells)

    def to_dict(self) -> dict:
        r0, c0, r1, c1 = self.bbox
        return {
            "area": self.area,
            "cause": self.cause,
            "reachable": self.reachable,
            "retried": self.retried,
            "exhausted": self.exhausted,
            "bbox": [r0, c0, r1, c1],
            "centroid": [float(self.cells[:, 0].mean()),
                         float(self.cells[:, 1].mean())],
        }


def classify_cause(zone_cells: np.ndarray, grid: np.ndarray,
                   reachable: np.ndarray) -> str:
    """判定死区成因（规划文档 2.4 的排查顺序）。

    判据按优先级：

    1. BFS 可达 → 是算法漏扫，不是死区（真正的 bug）；
    2. 周围一圈是障碍且障碍很「厚」 → 被家具/墙体包围；
    3. 连通域很小且离主区远 → 窄通道末端的死角；
    4. 其余 → 门洞未打通。
    """
    if reachable[zone_cells[:, 0], zone_cells[:, 1]].any():
        return CAUSE_ALGO

    h, w = grid.shape
    r0, c0, r1, c1 = bbox_of(zone_cells)

    # 统计死区四周的障碍厚度：向外扩一圈看有几层实心障碍
    from .grid_ops import dilate_mask
    zone_mask = np.zeros((h, w), dtype=bool)
    zone_mask[zone_cells[:, 0], zone_cells[:, 1]] = True

    ring1 = dilate_mask(zone_mask, 1) & ~zone_mask
    blocked_ring1 = np.count_nonzero(ring1 & (grid == GridCell.BLOCKED))
    ring2 = dilate_mask(dilate_mask(zone_mask, 1), 1) & ~zone_mask
    blocked_ring2 = np.count_nonzero(ring2 & (grid == GridCell.BLOCKED))

    # 包围圈很完整（内外两圈都是障碍）→ 被家具/墙体包住
    if blocked_ring1 > 0.55 * ring1.size and blocked_ring2 > 0.35 * ring2.size:
        return CAUSE_FURNITURE

    # 死区很小（几个格）且不在房间中央 → 多半是窄通道尽头
    if len(zone_cells) <= 12:
        return CAUSE_NARROW

    return CAUSE_SEALED


def find_dead_zones(cleaned: np.ndarray, grid: np.ndarray,
                    reachable: np.ndarray,
                    min_area: int = 1) -> list[DeadZone]:
    """找出所有未覆盖栅格并分类成因。

    参数
    ----
    cleaned: 轨迹清扫带掩码。
    reachable: 从起点 BFS 的可达掩码（判定物理死区的依据）。
    min_area: 小于该面积的连通域忽略（多为孤立噪点）。

    返回
    -------
    :class:`DeadZone` 列表，按面积降序，便于前端优先展示大问题。
    """
    uncovered = reachable & ~cleaned
    if not uncovered.any():
        # 连不可达的区域也要报告 —— 它们同样是「扫不到」的原因
        unreachable = passable_mask(grid) & ~reachable
        if not unreachable.any():
            return []
        comps = components_of(unreachable)
    else:
        comps = components_of(uncovered)

    zones: list[DeadZone] = []
    for cells in comps:
        if len(cells) < min_area:
            continue
        sub_reach = np.zeros_like(reachable)
        sub_reach[cells[:, 0], cells[:, 1]] = reachable[cells[:, 0], cells[:, 1]]
        cause = classify_cause(cells, grid, sub_reach)
        zones.append(DeadZone(cells=cells, cause=cause,
                              reachable=bool(sub_reach.any())))
    zones.sort(key=lambda z: z.area, reverse=True)
    return zones


def dead_zone_ratio(grid: np.ndarray, reachable: np.ndarray) -> float:
    """死区率 = |passable − reachable| / |passable|。"""
    passable = passable_mask(grid)
    total = int(np.count_nonzero(passable))
    if total == 0:
        return 0.0
    dead = int(np.count_nonzero(passable & ~reachable))
    return dead / total


# --------------------------------------------------------------------------
# 防震荡三道防线
# --------------------------------------------------------------------------
class AntiDither:
    """防震荡控制器（规划文档 2.2 的三道防线中除代价函数外的两道）。

    参数
    ----
    max_retry: 同一死区最多补扫次数（防线二）。
    window: 震荡检测的滑动窗口半径（防线三）。
    window_threshold: 窗口内进出次数超过此值判定为震荡。
    revisit_penalty: 传给 A* 的重复访问代价（防线一，见 astar.py）。
    """

    def __init__(self, max_retry: int = 3, window: int = 6,
                 window_threshold: int = 4, revisit_penalty: float = 0.8):
        self.max_retry = max_retry
        self.window = window
        self.window_threshold = window_threshold
        self.revisit_penalty = revisit_penalty
        self._visits: dict[tuple[int, int], int] = {}
        self.dither_events: list[dict] = []

    def record(self, cell: tuple[int, int]) -> None:
        self._visits[cell] = self._visits.get(cell, 0) + 1

    def check_dither(self, cell: tuple[int, int]) -> bool:
        """震荡检测：窗口内进出次数超阈值 → 判定震荡。

        返回 True 表示本次进入应被视为震荡，调用方应重新规划
        而不是继续硬扫。
        """
        r, c = cell
        k = 0
        for (vr, vc), n in self._visits.items():
            if abs(vr - r) <= self.window and abs(vc - c) <= self.window:
                k += n
        self._visits[(r, c)] = self._visits.get((r, c), 0) + 1
        if k > self.window_threshold:
            self.dither_events.append({"cell": [r, c], "window_hits": k})
            return True
        return False

    def can_retry(self, zone: DeadZone) -> bool:
        """最大重试次数判据。"""
        if zone.retried >= self.max_retry:
            zone.exhausted = True
            return False
        return True

    def stats(self) -> dict:
        return {
            "max_retry": self.max_retry,
            "revisit_penalty": self.revisit_penalty,
            "dither_events": len(self.dither_events),
            "window": self.window,
        }


def sweep_dead_zone(zone: DeadZone, grid: np.ndarray, current: tuple[int, int],
                    visited: np.ndarray, visited_count: np.ndarray,
                    clean_width: int, guard: AntiDither) -> list[tuple[int, int]] | None:
    """对单个死区做 A* 定向补扫。

    做法：取死区里离当前点最近的一个格作为目标，
    带重复访问代价规划过去，从而抑制绕圈。

    返回补扫路径；不可达或触发重试上限时返回 ``None``。
    """
    if not guard.can_retry(zone):
        return None

    # 目标取「离当前点最近」的死区格
    d = (zone.cells[:, 0] - current[0]) ** 2 + (zone.cells[:, 1] - current[1]) ** 2
    goal = (int(zone.cells[int(np.argmin(d))][0]),
            int(zone.cells[int(np.argmin(d))][1]))

    if not guard.check_dither(goal):
        path = astar(grid, current, goal,
                     visited=visited_count > 0,
                     revisit_penalty=guard.revisit_penalty)
        if path is not None:
            zone.retried += 1
            for p in path:
                visited_count[p[0], p[1]] += 1
            return path
    return None
