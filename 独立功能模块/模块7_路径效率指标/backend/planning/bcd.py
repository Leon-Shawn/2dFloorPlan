"""Boustrophedon Cell Decomposition（BCD 牛耕式分解）—— 模块 3 步骤 1、2

对应规划文档 2.2 步骤 1（区域分解）与步骤 2（单元内牛耕式覆盖）。

为什么需要 BCD
--------------
非凸房间（L 形、凹形、带承重柱的厅）直接对整体做蛇形扫描，
必然有「扫不到的死角」——因为扫描线会横穿障碍，或者某些行被障碍切断。
BCD 的思路是：先沿变化点把房间切成若干**单调单元**（内部不含夹断结构），
每个单元各自做蛇形遍历。这样每个单元内部都能 100% 覆盖，
单元之间的空隙由步骤 3 的转场路径负责。

实现要点
--------
- **变化点检测**：逐列记录该列可达栅格的连通段数量，
  相邻两列的段数或段的位置发生拓扑变化处即为切分点。
  只在「段数变化」和「段首段尾坐标跳变」两种情形切，避免过度切分。
- **退化单元防护**：切出的单元若没有可通行格则丢弃；
  单元内可通行行数 < 2 时无法做蛇形，交给 A* 直穿处理。
"""

from __future__ import annotations

import numpy as np

from collections import deque

from ..core.rasterize import GridCell
from .grid_ops import bbox_of, passable_mask


class Cell:
    """一个单调单元（BCD 的「cell」）。

    属性
    ----
    cells: 单元内可通行格的坐标数组 ``(N,2)``，顺序 (r, c)。
    bbox: ``(min_r, min_c, max_r, max_c)``。
    rows: 单元内可通行出现的行号集合（升序）。
    """

    __slots__ = ("cells", "bbox", "rows", "index")

    def __init__(self, cells: np.ndarray, index: int = 0):
        self.cells = cells
        self.bbox = bbox_of(cells)
        self.rows = np.unique(cells[:, 0])
        self.index = index

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def area(self) -> int:
        return int(len(self.cells))

    def is_scanable(self) -> bool:
        """能否做蛇形扫描：至少要有 2 个可通行行，否则退化成一条线。"""
        return len(self.rows) >= 2

    def __repr__(self) -> str:
        return (f"Cell(idx={self.index}, bbox={self.bbox}, "
                f"area={self.area}, rows={len(self.rows)})")


def _column_segments(col_mask: np.ndarray) -> list[tuple[int, int]]:
    """一列中连续可通行段的 ``[(起, 止), ...]``。"""
    segs = []
    start = None
    for i, v in enumerate(col_mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(col_mask) - 1))
    return segs


def decompose_bcd(passable: np.ndarray, min_cell_area: int = 4) -> list[Cell]:
    """BCD 垂直切片分解。

    参数
    ----
    passable: 布尔可通行掩码（通常是 ``BFS`` 可达区）。
    min_cell_area: 小于该面积的单元直接丢弃（多为噪声）。

    返回
    ----
    :class:`Cell` 列表。
    """
    h, w = passable.shape
    if not passable.any():
        return []

    # 限制在有效包围盒内，避免全图空白列干扰扫描
    rs, cs = np.nonzero(passable)
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    # 每列的连通段序列
    col_segs: list[list[tuple[int, int]]] = []
    for c in range(c0, c1 + 1):
        col_segs.append(_column_segments(passable[r0:r1 + 1, c]))

    # 找切分点：段数变化，或段的首尾位置发生跳变
    cuts: list[int] = [c0]                      # 起始列
    for i in range(1, len(col_segs)):
        prev, cur = col_segs[i - 1], col_segs[i]
        if len(prev) != len(cur):
            cuts.append(c0 + i)
            continue
        # 段数相同但拓扑变了（例如一个段分裂成两个位置不同的段）
        changed = any(p[0] != c[0] or p[1] != c[1] for p, c in zip(prev, cur))
        if changed:
            # 只有「跳变幅度明显」才切，避免边界抖动导致过度切分
            if _is_significant_shift(prev, cur):
                cuts.append(c0 + i)
    cuts.append(c1 + 1)                        # 结束列（开区间上界）

    # 按切分点划分列区间，逐区间收集可通行格
    cells: list[Cell] = []
    for i in range(len(cuts) - 1):
        ca, cb = cuts[i], cuts[i + 1]
        if cb <= ca:
            continue
        sub = passable[r0:r1 + 1, ca:cb]
        rr, cc = np.nonzero(sub)
        if len(rr) == 0:
            continue
        coords = np.stack([rr + r0, cc + ca], axis=1)
        if len(coords) < min_cell_area:
            continue
        cells.append(Cell(coords, index=len(cells)))

    return cells


def _is_significant_shift(prev: list[tuple[int, int]],
                          cur: list[tuple[int, int]],
                          tol: int = 1) -> bool:
    """段数相同时，段的起止坐标跳变是否算「拓扑变化」。

    容差取 1 格：容差太小会把噪声当成变化导致单元碎片化，
    太大会漏掉真正的夹断结构。
    """
    for (ps, pe), (cs, ce) in zip(prev, cur):
        if abs(ps - cs) > tol or abs(pe - ce) > tol:
            return True
    return False


def cover_cell_boustrophedon(passable: np.ndarray, cell: Cell,
                             clean_width: int = 2,
                             entry: tuple[int, int] | None = None,
                             scan_axis: str = "row") -> list[tuple[int, int]]:
    """单元内牛耕式（蛇形）覆盖 —— 模块 3 步骤 2。

    核心要求（规划文档 2.2 步骤 2）：**行距必须 ≤ 清扫宽度 c**，
    否则会留下漏扫带 —— 这是覆盖率上不去最常见的原因。

    参数
    ----
    passable: 全图可通行掩码（用于判断某一行能不能走通）。
    cell: 待扫单元。
    clean_width: 清扫宽度（格）。
    entry: 进入单元的落点；为 None 时取单元内离质心最近的格。
    scan_axis: ``'row'`` 逐行扫描（默认）；``'col'`` 逐列扫描。

    返回
    ----
    蛇形路径点序列（含换行时的横移点）。
    """
    cells = set(map(tuple, cell.cells))
    if not cells:
        return []

    stride = max(1, int(clean_width))

    if scan_axis == "row":
        lines = _scan_lines_rows(cells, stride)
        axis = 0
    else:
        lines = _scan_lines_cols(cells, stride)
        axis = 1

    if not lines:
        return []

    # 进入点：若指定了 entry，把它所在的那条扫描线排到最前
    order = list(range(len(lines)))
    if entry is not None and entry in cells:
        for li, line in enumerate(lines):
            if entry in line:
                order.remove(li)
                order.insert(0, li)
                break

    path: list[tuple[int, int]] = []
    for n, li in enumerate(order):
        line = lines[li]
        if not line:
            continue
        # 奇偶行反向，形成蛇形；反向时还要整体倒序走
        rev = (n % 2 == 1) if axis == 0 else (n % 2 == 0)
        seq = sorted(line, key=lambda p: p[axis], reverse=rev)
        path.extend(_connect(path[-1] if path else None, seq[0], cells))
        path.extend(seq)

    return _close_jumps(path, cells)


def _close_jumps(path: list[tuple[int, int]], cells: set) -> list[tuple[int, int]]:
    """把单元内相邻但不相邻的两点用直线补齐（仅在单元内部）。

    历史 bug
    --------
    ``_connect()`` 在跨行时返回空序列，注释写的是「由调用方的 A* 转场负责」。
    但**单元内部**根本没有调用方补A* —— 调用方只负责**单元之间**的转场。
    于是单元内每换一条扫描线就残留一个跳跃点（如 (5,11) → (5,13)）。
    表现：压墙步数为 0（两端都是可通行格），但前端连线直接穿过中间墙体。

    做法
    ----
    扫描线按 ``stride`` 间隔取，单元内相邻两线的纵向距离等于 ``stride``。
    先尝试直线插值：若整条桥都落在本单元的可通行格内（同单元无障碍夹断，
    这正是 BCD 单调单元的定义），直接采用；
    否则退化为在「本单元的可通行格」里做 BFS 找折线，
    仍找不到（理论上不该发生）才保留跳跃，交给外层 ``_densify_path`` 兜底。
    """
    if len(path) < 2:
        return path
    out: list[tuple[int, int]] = [(int(path[0][0]), int(path[0][1]))]
    for i in range(1, len(path)):
        prev = out[-1]
        cur = (int(path[i][0]), int(path[i][1]))
        if cur == prev:
            continue
        dr = cur[0] - prev[0]
        dc = cur[1] - prev[1]
        if max(abs(dr), abs(dc)) > 1:
            steps = max(abs(dr), abs(dc))
            sr = (dr > 0) - (dr < 0)
            sc = (dc > 0) - (dc < 0)
            bridge = [(prev[0] + sr * t, prev[1] + sc * t)
                      for t in range(1, steps)]
            if bridge and all(pt in cells for pt in bridge):
                out.extend(bridge)
            else:
                out.extend(_bfs_within(prev, cur, cells))
        out.append(cur)
    return out


def _bfs_within(src: tuple[int, int], dst: tuple[int, int],
                cells: set, max_nodes: int = 40000) -> list[tuple[int, int]]:
    """在给定格子集合内 BFS 求 src→dst 的八邻接折线（不含首点）。

    直线插值失败时兜底：只能在该单元的可通行格内绕，
    保证「单元内也是连续路径」，不把跳跃留给可视化阶段。
    """
    if src == dst:
        return []
    seen = {src}
    q: deque = deque([(src, [src])])
    while q:
        node, trace = q.popleft()
        if len(seen) > max_nodes:
            break
        r, c = node
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if nxt in seen or nxt not in cells:
                    continue
                if nxt == dst:
                    return trace[1:] + [dst]
                seen.add(nxt)
                q.append((nxt, trace + [nxt]))
    return []


def _scan_lines_rows(cells: set, stride: int) -> list[list[tuple[int, int]]]:
    """按行分组：每隔 ``stride`` 行取一条扫描线。

    优先从最小行号开始，保证全覆盖；行内连续段分别成线，
    这样遇到障碍断开时会生成多条线段而不是连成一条穿墙的线。
    """
    rows = sorted({r for r, _ in cells})
    lines: list[list[tuple[int, int]]] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        line = [p for p in cells if p[0] == r]
        if line:
            lines.append(sorted(line))
        # 跳到下一个「距当前行 ≥ stride」的行
        j = i + 1
        while j < len(rows) and rows[j] - r < stride:
            j += 1
        i = j
    return _split_contiguous(lines)


def _scan_lines_cols(cells: set, stride: int) -> list[list[tuple[int, int]]]:
    """按列分组，与 :func:`_scan_lines_rows` 对称。"""
    cols = sorted({c for _, c in cells})
    lines: list[list[tuple[int, int]]] = []
    i = 0
    while i < len(cols):
        c = cols[i]
        line = [p for p in cells if p[1] == c]
        if line:
            lines.append(sorted(line))
        j = i + 1
        while j < len(cols) and cols[j] - c < stride:
            j += 1
        i = j
    return _split_contiguous(lines)


def _split_contiguous(lines: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    """把被障碍断开的扫描线拆成连续段。

    不拆的话会生成一条横穿障碍的「假蛇形线」，
    机器人执行时会撞墙 —— 这是很容易被忽略的 bug。

    历史 bug：连续性判据写成了 ``列差 == 1 or 行差 == 1``。
    其中「行差 == 1」是错的 —— 它会把**上一行末点与下一行首点**
    （斜对角关系）判为连续，于是同一条扫描线被墙断成的两段
    又被粘回一起，产生跨墙跳跃。实测在 basic策略的
    real_fp_02 上造成 42 处可视化穿墙。

    正确判据：只有**同行且列差恰为 1** 才算连续。
    """
    out: list[list[tuple[int, int]]] = []
    for line in lines:
        line = sorted(line)
        if not line:
            continue
        seg: list[tuple[int, int]] = [line[0]]
        for p in line[1:]:
            # 必须同行且列相邻。跨行的两段不是连续段，
            # 交由 A* 转场处理。
            if p[0] == seg[-1][0] and abs(p[1] - seg[-1][1]) == 1:
                seg.append(p)
            else:
                out.append(seg)
                seg = [p]
        if seg:
            out.append(seg)
    return out


def _connect(a: tuple[int, int] | None, b: tuple[int, int],
             cells: set) -> list[tuple[int, int]]:
    """换行时的横移点。

    这里只在**同一行内相邻**时直连；跨行时返回空，
    由调用方的 A* 转场负责（保证不会穿墙）。
    """
    if a is None or a == b:
        return []
    if a[0] == b[0] and abs(a[1] - b[1]) == 1:
        return []
    return []
