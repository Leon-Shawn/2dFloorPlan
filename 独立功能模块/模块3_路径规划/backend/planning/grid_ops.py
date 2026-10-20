"""栅格基础运算 —— 模块 3 的地基（任务 3-1）

提供连通域、可达性、邻居扩展等所有策略共用的原语。
指标口径与规划文档 2.5 节完全一致，不允许各策略自行定义。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..core.rasterize import GridCell


# 方向向量：四连通（用于连通性判定，口径与指标定义一致）
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
# 八连通（用于 A* 移动，允许斜穿但代价更高）
DIRS8 = DIRS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def passable_mask(grid: np.ndarray) -> np.ndarray:
    """可通行掩码。``2``（已清扫）也算可通行 —— 重复经过是允许的。"""
    return (grid != GridCell.BLOCKED)


def label_components(passable: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """给可通行区做连通域标号。

    自己实现 BFS 而不用 ``cv2.connectedComponents``，原因是要保证
    **与 A* 的移动口径完全一致**（四连通），并且要能返回每个域的成员列表。

    返回 ``(labels, count)``，``labels`` 中0 表示不可通行。
    """
    h, w = passable.shape
    labels = np.zeros((h, w), dtype=np.int32)
    nbrs = DIRS4 if connectivity == 4 else DIRS8
    count = 0

    for sr in range(h):
        for sc in range(w):
            if not passable[sr, sc] or labels[sr, sc]:
                continue
            count += 1
            labels[sr, sc] = count
            q = deque([(sr, sc)])
            while q:
                r, c = q.popleft()
                for dr, dc in nbrs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not labels[nr, nc]:
                        labels[nr, nc] = count
                        q.append((nr, nc))
    return labels, count


def bfs_reachable(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """从起点做 BFS，返回可达栅格的布尔掩码。

    这就是指标口径里的 ``reachable``：机器人从起点出发物理上能到达的地方。
    覆盖率的分母必须用它，而不是「全部可通行格」——
    否则被家具完全包围的死区会让覆盖率永远上不到 100%，
    而且分不清是「算法漏扫」还是「地图断了」。
    """
    h, w = grid.shape
    reach = np.zeros((h, w), dtype=bool)
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w):
        return reach
    if grid[sr, sc] == GridCell.BLOCKED:
        return reach

    reach[sr, sc] = True
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] \
                    and grid[nr, nc] != GridCell.BLOCKED:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def find_start(grid: np.ndarray, prefer: tuple[int, int] | None = None) -> tuple[int, int]:
    """挑选机器人起点。

    优先用调用方指定的点；无效时自动挑一个。
    自动挑选策略：靠近可通行区包围盒中心的位置 —— 从中心出发
    到各个房间的距离比较均衡，不会一上来就卡在角落。
    """
    h, w = grid.shape
    if prefer is not None:
        r, c = prefer
        if 0 <= r < h and 0 <= c < w and grid[r, c] != GridCell.BLOCKED:
            return r, c

    passable = passable_mask(grid)
    if not passable.any():
        return h // 2, w // 2
    rs, cs = np.nonzero(passable)
    # 取距离几何中心最近的可通行点
    cr, cc = (rs.min() + rs.max()) // 2, (cs.min() + cs.max()) // 2
    d = (rs - cr) ** 2 + (cs - cc) ** 2
    i = int(np.argmin(d))
    return int(rs[i]), int(cs[i])


def components_of(mask: np.ndarray) -> list[np.ndarray]:
    """把布尔掩码拆成若干连通域，每域返回坐标数组 ``(N,2)``，顺序为 (r, c)。"""
    labels, count = label_components(mask, connectivity=4)
    out = []
    for i in range(1, count + 1):
        rs, cs = np.nonzero(labels == i)
        out.append(np.stack([rs, cs], axis=1))
    return out


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """布尔掩码膨胀（四连通，radius 格）。用于安全间隙判断。"""
    out = mask.copy()
    for _ in range(max(0, radius)):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def bbox_of(cells: np.ndarray) -> tuple[int, int, int, int]:
    """返回 ``(min_r, min_c, max_r, max_c)``。"""
    return int(cells[:, 0].min()), int(cells[:, 1].min()), \
           int(cells[:, 0].max()), int(cells[:, 1].max())
