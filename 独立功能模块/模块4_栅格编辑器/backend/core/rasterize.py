"""模块 1 · 户型图转栅格地图 —— 栅格矩阵构建（任务 1-5 ~ 1-9）

流程
----
    二值墙体掩码
        ↓ 1-5  墙体检测（连通域筛选，滤掉噪点与家具轮廓）
        ↓ 1-6  墙厚膨胀填充（把线状墙体填成实心障碍带）
        ↓ 1-7  门洞检测（墙体掩码的缺口 → 局部打通）
        ↓ 1-8  家具/障碍标记
        ↓ 1-9  输出栅格矩阵【格式定版】

栅格矩阵格式（定版，此后只许追加不许改）
------------------------------------------
``grid`` : ``np.ndarray[uint8]``，形状 ``(rows, cols)``，行优先。
取值常量见 :class:`GridCell`：

===== =========== ==========================================
值含义        说明
===== =========== ==========================================
0    不可通行     墙体、家具、图外区域
1    可通行       机器人可达的空闲栅格
2    已清扫       仿真运行后标记（初态不存在此值）
===== =========== ==========================================

配套元数据 :class:`GridMeta` 记录像素↔栅格的缩放关系，
使Canvas 能准确反算像素坐标（任务 1-2 画布↔栅格坐标映射的基础）。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .binarize import BinarizeParams, binarize


class GridCell:
    """栅格取值常量。定版后不得修改数值。"""

    BLOCKED = 0
    PASSABLE = 1
    CLEANED = 2

    NAMES = {0: "不可通行", 1: "可通行", 2: "已清扫"}


@dataclass
class GridMeta:
    """栅格与原图的坐标映射关系。

    栅格 (r, c) 的中心对应原图像素：
        x_px = (c + 0.5) * scale
        y_px = (r + 0.5) * scale
    """

    cell_px: int          # 每个栅格对应多少像素（边长）
    src_width: int        # 原图宽
    src_height: int       # 原图高
    grid_width: int# 栅格列数
    grid_height: int      # 栅格行数
    door_cells: list = None# 检测到的门洞栅格坐标（诊断用）
    furniture_cells: int = 0    # 家具占据的栅格数

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grid_width"], d["grid_height"] = self.grid_width, self.grid_height
        return d


@dataclass
class RasterizeParams:
    """栅格化参数（除二值化外的部分）。"""

    cell_px: int = 10          # 栅格边长（像素）。10px 兼顾精度与计算量
    wall_dilate: int = 1# 墙体膨胀核尺寸（0=不膨胀）
    door_width_px: int = 110  # 门洞判定：墙体缺口短于此值则视为门洞（真实门宽约 80-100cm）
    door_bridge_px: int = 26  # 门洞打通：在缺口中心连接的栅格厚度
    furniture_ratio: float = 0.30  # 家具判定：轮廓面积/包围盒面积 低于此值且非墙体
    robot_radius_px: int = 14      # 机器人等效半径（像素），用于窄通道判定
    keep_border: bool = True       # 是否保留图外边框为不可通行

    def normalized(self) -> "RasterizeParams":
        return RasterizeParams(
            cell_px=int(np.clip(self.cell_px, 4, 60)),
            wall_dilate=int(np.clip(self.wall_dilate, 0, 9)),
            door_width_px=int(np.clip(self.door_width_px, 8, 400)),
            door_bridge_px=int(np.clip(self.door_bridge_px, 2, 200)),
            furniture_ratio=float(np.clip(self.furniture_ratio, 0.05, 0.95)),
            robot_radius_px=int(np.clip(self.robot_radius_px, 2, 100)),
            keep_border=bool(self.keep_border),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RasterizeResult:
    """栅格化完整产物。"""

    grid: np.ndarray
    meta: GridMeta
    stages: dict# 中间图（原图 / 二值 / 墙体掩码 / 门洞后）
    params: dict              # 实际生效的参数
    degraded: bool
    note: str
    stats: dict               # 诊断统计

    def grid_list(self) -> list:
        """转成 JSON 可序列化的二维列表（前端直接渲染）。"""
        return self.grid.tolist()


# --------------------------------------------------------------------------
# 1-5 墙体检测
# --------------------------------------------------------------------------
def detect_walls(binary: np.ndarray, min_area: int = 60) -> np.ndarray:
    """从二值图中提取墙体（任务 1-5）。

    用**骨架化 + 长度**而非连通域面积来判定，原因：
    真实户型图里所有墙体经由门洞串成**一个大连通域**，
    按连通域面积筛选会把整张图当成一堵墙，无法区分内外墙。

    做法：抽取线条的骨架（skeleton），骨架总长度与图幅尺寸相关，
    远大于家具轮廓的骨架长度。据此判定「结构性线状物」。
    """
    mask = np.zeros_like(binary)

    # 骨架化：把粗线压成 1px 中心线，消除线宽对面积的影响
    skel = _skeletonize(binary)
    skel_len = int(np.count_nonzero(skel))

    # 连通域筛选：去掉孤立噪点
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        # 长条结构（墙体）判定：跨度大 或 长宽比高
        span = max(cw, ch) / max(binary.shape)
        aspect = max(cw / max(ch, 1), ch / max(cw, 1))
        if span >= 0.25 or aspect >= 2.5:
            mask[labels == i] = 255

    # 骨架总长度兜底：若因门洞把墙串成一体导致上面全部漏判，
    # 则用「面积足够大且骨架够长」的条件整体保留
    if np.count_nonzero(mask) < binary.size * 0.002 and skel_len > 0.05 * binary.shape[1]:
        num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num2):
            if stats2[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels2 == i] = 255
    return mask


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen 细化算法，把二值线条压成1px 骨架。

    用于长度/拓扑特征提取，与 OpenCV 的形态学操作互补。
    """
    img = (binary > 0).astype(np.uint8)
    # 先做一次开运算去掉毛刺，让细化更稳
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    skel = np.zeros_like(img)
    h, w = img.shape
    padded = np.pad(img, 1)

    while True:
        changed = False
        for step in (0, 1):
            p = padded
            p2 = img
            # 8 邻域计数
            p1 = p[:-2, :-2]; p2_ = p[:-2, 1:-1]; p3 = p[:-2, 2:]
            p4 = p[1:-1, :-2]; p6 = p[1:-1, 2:]
            p7 = p[2:, :-2];  p8_ = p[2:, 1:-1]; p9 = p[2:, 2:]

            b = (p1 + p2_ + p3 + p4 + p6 + p7 + p8_ + p9)
            # 邻居数 2..6
            cond_a = np.logical_and(b >= 2, b <= 6)
            # 0→1 的顺序数（避免断裂）
            seq = [p1, p2_, p3, p6, p9, p8_, p7, p4, p1]
            transitions = sum(
                ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8)
            )
            cond_b = transitions == 1
            if step == 0:
                cond_c = (p2_ * p4 * p6) == 0
            else:
                cond_c = (p2_ * p4 * p8_) == 0
            remove = np.logical_and(np.logical_and(cond_a, cond_b), cond_c) & (img == 1)
            if remove.any():
                img[remove] = 0
                changed = True
            padded = np.pad(img, 1)
        if not changed:
            break

    skel[img == 1] = 255
    return skel



# --------------------------------------------------------------------------
# 1-8 家具 / 障碍标记
# --------------------------------------------------------------------------
def detect_furniture(binary: np.ndarray, wall_mask: np.ndarray,
                     ratio: float = 0.30, min_area: int = 200) -> np.ndarray:
    """标记家具/障碍（任务 1-8）—— 占位签名，真实实现在下方灰度版。

    保留此函数是为了让 :func:`rasterize` 的调用签名统一；
    实际检测走 :func:`detect_furniture_gray`，因为家具的**填充色**
    （浅灰 205）与背景（250）亮度接近，在二值图里往往被阈值滤掉，
    只能靠「灰色块 + 闭合轮廓」在灰度域识别。
    """
    furniture = np.zeros_like(binary)
    residual = cv2.bitwise_and(binary, cv2.bitwise_not(wall_mask))
    residual = cv2.bitwise_not(residual)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    h, w = binary.shape
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        fill = area / max(cw * ch, 1)
        if fill >= ratio and area <= 0.20 * h * w:
            furniture[labels == i] = 255
    return furniture


def detect_furniture_gray(gray: np.ndarray, wall_mask: np.ndarray,
                          cell_px: int = 10,
                          area_ratio: float = 0.30, min_area: int = 200) -> np.ndarray:
    """在灰度域检测家具/障碍轮廓（任务 1-8）。

    家具与墙的三个可分特征：
      1. **亮度**：家具填充是浅灰，明显暗于背景，但亮于墙；
      2. **闭合轮廓**：家具是封闭矩形/圆角矩形，有完整外轮廓；
      3. **尺寸**：家具占图幅比例小，不会贯穿全图。

    实现：以「比背景暗、比墙亮」为前景 →闭运算补全→ 找轮廓，
    再按面积和长宽比筛掉墙体自身的轮廓（墙极长长宽比大）。
    """
    h, w = gray.shape[:2]
    # 家具亮度区间。门限取 100 而非 120：家具轮廓线本身是 120 灰，
    # 若门限卡在 120 会把轮廓排除在家具掩码之外，
    # 轮廓就会残留成「墙」，相邻两件家具之间的空隙随即被误判为门洞
    #（实测 fp_04 的床与衣柜之间出现假门洞）。
    v = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if gray.ndim == 3 else gray
    furn_gray = ((v > 100) & (v < 225)).astype(np.uint8) * 255

    # 闭运算把家具内部的零散灰块连成一体
    furn_gray = cv2.morphologyEx(
        furn_gray, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(cell_px, 5),) * 2),
    )

    # 去掉墙体区域，避免把墙的粗轮廓当家具
    furn_gray = cv2.bitwise_and(furn_gray, cv2.bitwise_not(cv2.dilate(
        wall_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))))

    furniture = np.zeros_like(furn_gray)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(furn_gray, connectivity=8)
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        # 家具：长宽比不能太极端（排除细长条），面积不超过图幅 20%
        aspect = max(cw / max(ch, 1), ch / max(cw, 1))
        if aspect <= 2.2 and area <= 0.20 * h * w:
            furniture[labels == i] = 255
    return furniture



# --------------------------------------------------------------------------
def detect_doors(binary: np.ndarray, max_gap_px: int = 110,
                 bridge_px: int = 26) -> tuple[np.ndarray, list]:
    """检测门洞并给出打通掩码（任务 1-7）。

    方法：**线段缺口检测**。

    户型图里的门洞没有显式标注，物理表现是「墙体线段上的一个缺口」。
    因此直接在线段层面找缺口：

    1. 用细长的方向性核做开运算，分别抽出**水平线段**与**竖直线段**
       （长度 ≥ ``max_gap_px`` 才保留，短碎片直接丢弃）。
    2. 对同一条墙上的线段两两配对：若二者**共线**（同一y 或同一 x）、
       间隙在 ``(0, max_gap_px]`` 之间 → 该间隙即门洞。
    3. 在缺口中心画一条垂直于墙的连通带，把两侧打通。

    为什么不用形态学闭运算：门洞宽约 70px，而墙厚仅 5px。
    要用闭运算合上门洞，核必须 ≥70px，此时 5px 薄墙会被膨胀整个吞掉，
    内外空间合并，形态学无法逆恢复 —— 这是本方法的核心取舍。

    参数
    ----
    binary: 墙体二值图（255 为墙）。
    max_gap_px: 门洞最大宽度门限，同时是最短线段长度。
    bridge_px: 打通时连接缺口中心的厚度。

    返回
    ----
    ``(doors_mask, door_points)``，door_points 为门洞中心像素坐标。
    """
    h, w = binary.shape
    door_pts: list[tuple[int, int]] = []

    # --- 水平墙上的缺口 ---
    # min_len 取小值：门洞常靠近墙角，断口外侧的墙只剩几十像素的短茬，
    # 过滤太严会把短茬丢掉，门洞就永远配不上对。
    hsegs = _extract_segments(binary, axis="h", min_len=12)
    for (y1, a1, b1), (y2, a2, b2) in _pairwise(hsegs):
        if abs(y1 - y2) > _wall_align_tol(binary):
            continue          # 不共线，不是同一条墙
        gap = max(a1, a2) - min(b1, b2)
        if not (0 < gap <= max_gap_px):
            continue
        lo, hi = sorted((b1, a2)) if b1 < a2 else sorted((b2, a1))
        cx, cy = (lo + hi) // 2, (y1 + y2) // 2
        # 复核：缺口必须是贯通的（两侧都是墙，中间是空的）
        if not _gap_is_open(binary, cx, cy, "h", gap):
            continue
        door_pts.append((cx, cy))

    # --- 竖直墙上的缺口 ---
    vsegs = _extract_segments(binary, axis="v", min_len=12)
    for (x1, a1, b1), (x2, a2, b2) in _pairwise(vsegs):
        if abs(x1 - x2) > _wall_align_tol(binary):
            continue
        gap = max(a1, a2) - min(b1, b2)
        if not (0 < gap <= max_gap_px):
            continue
        lo, hi = sorted((b1, a2)) if b1 < a2 else sorted((b2, a1))
        cx, cy = (x1 + x2) // 2, (lo + hi) // 2
        if not _gap_is_open(binary, cx, cy, "v", gap):
            continue
        door_pts.append((cx, cy))

    door_pts = _dedup_points(door_pts, tol=max(10, max_gap_px // 4))

    doors = np.zeros((h, w), dtype=np.uint8)
    for cx, cy in door_pts:
        _bridge_gap(binary, cx, cy, doors, bridge_px)

    return doors, door_pts


def _extract_segments(binary: np.ndarray, axis: str,
                      min_len: int = 20) -> list[tuple[int, int, int]]:
    """抽取指定方向的直线段。

    用细长核做开运算保住长直线，短线段被核抹掉。
    返回 ``[(轴坐标, 起点, 终点), ...]``：
      - axis='h' 时轴坐标是线段的 y，起点/终点是 x；
      - axis='v' 时轴坐标是线段的 x，起点/终点是 y。
    """
    L = max(int(min_len), 3)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if axis == "h" else (1, L))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    num, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    segs: list[tuple[int, int, int]] = []
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if area < L:               # 开运算后仍是碎片，丢弃
            continue
        if axis == "h":
            if cw < L:
                continue
            # 只接受「细长」的：厚度不超过墙厚的数倍
            if ch > max(ch * 0 + 12, cw * 0.12):
                continue
            segs.append(((y + ch // 2), x, x + cw))
        else:
            if ch < L:
                continue
            if cw > max(cw * 0 + 12, ch * 0.12):
                continue
            segs.append(((x + cw // 2), y, y + ch))
    return segs


def _pairwise(segs: list) -> list[tuple]:
    """所有两两组合（线段数量在真实户型图里是个位数，O(n²) 完全可接受）。"""
    out = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            out.append((segs[i], segs[j]))
    return out


def _wall_align_tol(binary: np.ndarray) -> int:
    """共线判定容差：允许中心相差一个墙厚。

    墙的两侧像素被二值化后可能不在同一行，故按图幅的0.6% 取容差，
    对 1000px 图约为 6px，正好覆盖墙厚。
    """
    return max(3, int(binary.shape[1] * 0.006))


def _gap_is_open(binary: np.ndarray, cx: int, cy: int, axis: str,
                 gap: int = 0) -> bool:
    """复核门洞候选点：是否真的是「两端有墙、中间贯通」的缺口。

    做法：沿**墙的走向**取一条横切线，要求「墙 → 空 → 空 → 墙」。

    方向约定容易搞反，这里明确：
        - ``axis='h'``：墙是**水平**的（沿 x 延伸），门洞在 x 方向断开，
          所以要检查缺口的**左右**两侧是否有墙；
        - ``axis='v'``：墙是**竖直**的（沿 y 延伸），要检查**上下**两侧。

    采样偏移必须落在缺口**之外**才能取到墙，因此取 ``gap/2 + 余量``；
    用固定小偏移会永远落在门洞内部，导致两侧都取不到墙而全盘漏判。

    参数
    ----
    gap: 实测缺口宽度（沿墙走向）。
    """
    h, w = binary.shape
    off = max(int(gap) // 2 + 6, 10)   # 保证采到门洞两侧的墙

    if axis == "h":
        # 水平墙 → 检查左右
        if cx - off < 0 or cx + off >= w:
            return False
        side_a = binary[cy, cx - off] > 0
        side_b = binary[cy, cx + off] > 0
        mid = binary[cy, max(cx - off // 2, 0):cx + off // 2 + 1] > 0
    else:
        # 竖直墙 → 检查上下
        if cy - off < 0 or cy + off >= h:
            return False
        side_a = binary[cy - off, cx] > 0
        side_b = binary[cy + off, cx] > 0
        mid = binary[max(cy - off // 2, 0):cy + off // 2 + 1, cx] > 0

    return bool(side_a) and bool(side_b) and not bool(mid.any())


def _bridge_gap(binary: np.ndarray, cx: int, cy: int,
                doors: np.ndarray, bridge_px: int) -> None:
    """在门洞中心画一条连通带，把缺口两侧的空间打通。

    方向由调用方传下来的上下文不可得时，用「上下左右哪两侧是墙」来判定：
    门是竖直墙上的洞 → 水平打通；反之竖直打通。
    """
    h, w = binary.shape
    r = 4
    thick = max(int(bridge_px), 7)

    y0, y1 = max(cy - r, 0), min(cy + r + 1, h)
    x0, x1 = max(cx - r, 0), min(cx + r + 1, w)

    up_wall = bool(binary[max(cy - 12, 0), x0:x1].any())
    down_wall = bool(binary[min(cy + 12, h - 1), x0:x1].any())
    left_wall = bool(binary[y0:y1, max(cx - 12, 0)].any())
    right_wall = bool(binary[y0:y1, min(cx + 12, w - 1)].any())

    vertical_pair = up_wall and down_wall      # 上下有墙 → 竖直墙 → 水平打通
    horizontal_pair = left_wall and right_wall  # 左右有墙 → 水平墙 → 竖直打通

    if vertical_pair and not horizontal_pair:
        cv2.line(doors, (max(cx - thick, 0), cy), (min(cx + thick, w - 1), cy), 255, thick)
    elif horizontal_pair and not vertical_pair:
        cv2.line(doors, (cx, max(cy - thick, 0)), (cx, min(cy + thick, h - 1)), 255, thick)
    else:
        # 角部门或判定不明：挖一个方块，稳妥地确保连通
        cv2.circle(doors, (cx, cy), max(thick // 2 + 2, 6), 255, -1)


def _dedup_points(pts: list[tuple[int, int]], tol: int = 20) -> list[tuple[int, int]]:
    """按欧氏距离合并邻近的重复点。"""
    out: list[tuple[int, int]] = []
    for p in pts:
        if not any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= tol * tol for q in out):
            out.append(p)
    return out


# --------------------------------------------------------------------------
# 1-9栅格矩阵输出
# --------------------------------------------------------------------------
def to_grid(blocked: np.ndarray, cell_px: int,
            keep_border: bool = True) -> tuple[np.ndarray, tuple[int, int]]:
    """把像素级障碍掩码降采样成栅格矩阵（max-pooling）。

    判据：**一个栅格内只要出现任何一个障碍像素，该栅格即不可通行**。

    为什么不能用覆盖率阈值（如「>25% 才算障碍」）：
    户型图的墙只有 5px 宽，而栅格边长 10px，一堵墙在它穿过的栅格里
    最多只占 50%，并且常常因为墙恰好压在栅格边界上而只占到 20%。
    用任何 <50% 的阈值，薄墙都会在降采样时**整段消失** ——
    实测 fp_02 的隔墙就是这样被抹掉的，导致三个房间连成一片，
    覆盖率与连通性分析全部失真。

    max-pooling 相当于给墙「加粗一格」，既保证墙不会消失，
    也不会误伤正常通道（门洞宽 70px，足以留出通道）。
    """
    h, w = blocked.shape
    rows, cols = h // cell_px, w // cell_px
    if rows < 1 or cols < 1:
        return np.zeros((1, 1), dtype=np.uint8), (1, 1)

    grid = np.zeros((rows, cols), dtype=np.uint8)
    occ = blocked[:rows * cell_px, :cols * cell_px].reshape(rows, cell_px, cols, cell_px)
    any_occ = occ.max(axis=(1, 3)) > 0
    grid[any_occ] = GridCell.BLOCKED
    grid[~any_occ] = GridCell.PASSABLE

    if keep_border and rows > 2 and cols > 2:
        grid[0, :] = grid[-1, :] = GridCell.BLOCKED
        grid[:, 0] = grid[:, -1] = GridCell.BLOCKED

    return grid, (cols, rows)


def exclude_exterior(obstacle: np.ndarray) -> tuple[np.ndarray, int]:
    """剔除建筑轮廓外的空白背景（任务 1-9 的一部分）。

    为什么必须做：户型图四周通常留有几十像素的白边，
    它在栅格上是连片的「可通行」区域。若不剔除：
      - 可通行区会多出一个与户型无关的大连通域；
      - 更致命的是 ``reachable = BFS(start) ∩ passable_grid`` 的分母
        会被这块室外空白撑大，导致**覆盖率被系统性稀释**。

    做法：从图像外围做一次flood fill，把与画布边界连通的自由区
    全部标记为室外并转为障碍。

    返回
    ----
    ``(masked_obstacle, exterior_px)``
    """
    h, w = obstacle.shape[:2]
    free = cv2.bitwise_not(obstacle)

    # 外围补一圈自由边，保证 flood fill 有确定的起点，
    # 即使原图四角恰好被墙压住也能启动
    padded = cv2.copyMakeBorder(free, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 128)

    exterior = (padded == 128)[1:-1, 1:-1]
    masked = obstacle.copy()
    masked[exterior] = 255
    return masked, int(np.count_nonzero(exterior))


def narrow_corridor_mask(grid: np.ndarray, robot_radius_cells: int) -> np.ndarray:
    """标记窄于机身的通道（任务 1-12 鲁棒性 + README 选图建议第 5 类）。

    判据：某可通行栅格的**横向可通行宽度**小于机器人直径，
    则该栅格实际不可通行（机器人过不去）。
    实现：对可通行区做距离变换，dist <= radius 的即为窄处。
    """
    passable = (grid == GridCell.PASSABLE).astype(np.uint8)
    if passable.sum() == 0:
        return np.zeros_like(passable, dtype=bool)
    dist = cv2.distanceTransform(passable, cv2.DIST_L2, 3)
    narrow = (dist <= robot_radius_cells) & (passable > 0)
    return narrow


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def rasterize(img: np.ndarray,
              bin_params: BinarizeParams | None = None,
              ras_params: RasterizeParams | None = None) -> RasterizeResult:
    """户型图 → 栅格矩阵。模块 1 的完整管线。

    参数
    ----
    img: 输入户型图，灰度或 BGR。
    bin_params: 二值化参数，为 None 用默认。
    ras_params: 栅格化参数，为 None 用默认。

    返回
    ----
    :class:`RasterizeResult`
    """
    bp = (bin_params or BinarizeParams()).normalized()
    rp = (ras_params or RasterizeParams()).normalized()

    # --- 1-3/1-4二值化 ---
    binres = binarize(img, bp)
    binary = binres.binary

    # 二值化对过小的图会做放大（否则自适应阈值的邻域比图像还大），
    # 这里把原图同步到二值图的尺寸，否则后续灰度域的家具检测
    # 会因形状不匹配而抛 OpenCV 的 binary_op 异常。
    if img.shape[:2] != binary.shape[:2]:
        img = cv2.resize(img, (binary.shape[1], binary.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    # --- 1-5 墙体检测 ---
    walls = detect_walls(binary)

    # --- 1-8 家具标记（在灰度域做，见 detect_furniture_gray 说明） ---
    furniture = detect_furniture_gray(img, walls, rp.cell_px)

    # 墙体 + 家具 = 静态障碍
    obstacle = cv2.bitwise_or(walls, furniture)

    # --- 1-6 墙厚膨胀填充 ---
    if rp.wall_dilate >= 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (rp.wall_dilate * 2 + 1,) * 2)
        walls_dilated = cv2.dilate(walls, k)
    else:
        walls_dilated = walls.copy()
    obstacle_dilated = cv2.bitwise_or(walls_dilated, furniture)

    # --- 1-7 门洞检测与打通 ---
    # 关键：门洞检测必须只看**结构墙**，不能看全图。
    # 家具也是闭合矩形轮廓，相邻两件家具之间的空隙会被误判成门洞
    #（实测 fp_04 床与衣柜之间出现假门洞），故先把家具区域从证据里剔除。
    structural = cv2.bitwise_and(
        binary,
        cv2.bitwise_not(cv2.dilate(
            furniture, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))),
    )
    doors, door_pts = detect_doors(structural, rp.door_width_px, rp.door_bridge_px)
    # 门洞区域从障碍中扣除 —— 打通连通性
    final_obstacle = cv2.bitwise_and(obstacle_dilated, cv2.bitwise_not(doors))

    # --- 1-9 前置：剔除建筑轮廓外的室外空白 ---
    # 顺序很关键：必须在门洞打通之后做，否则门外的空白会通过门洞
    # 渗进室内，把室内外的连通性重新连起来。
    final_obstacle, exterior_px = exclude_exterior(final_obstacle)

    # --- 1-9 栅格矩阵 ---
    grid, (cols, rows) = to_grid(final_obstacle, rp.cell_px, rp.keep_border)

    # 窄通道压测：过不去的窄道也判为不可通行
    radius_cells = max(1, rp.robot_radius_px // rp.cell_px)
    narrow = narrow_corridor_mask(grid, radius_cells)
    grid[narrow] = GridCell.BLOCKED

    meta = GridMeta(
        cell_px=rp.cell_px,
        src_width=img.shape[1],
        src_height=img.shape[0],
        grid_width=cols,
        grid_height=rows,
        door_cells=[[int(x), int(y)] for x, y in door_pts],
        furniture_cells=int(np.count_nonzero(furniture) / (rp.cell_px ** 2)),
    )

    stats = {
        "ink_ratio": round(binres.ink_ratio, 5),
        "wall_px": int(np.count_nonzero(walls)),
        "furniture_px": int(np.count_nonzero(furniture)),
        "door_count": len(door_pts),
        "exterior_px": exterior_px,
        "passable_cells": int(np.count_nonzero(grid == GridCell.PASSABLE)),
        "blocked_cells": int(np.count_nonzero(grid == GridCell.BLOCKED)),
        "narrow_blocked_cells": int(np.count_nonzero(narrow)),
    }

    stages = {
        "binary": binary,
        "walls": walls,
        "furniture": furniture,
        "doors": doors,
        "obstacle": final_obstacle,
    }

    return RasterizeResult(
        grid=grid,
        meta=meta,
        stages=stages,
        params={"binarize": bp.to_dict(), "rasterize": rp.to_dict()},
        degraded=binres.degraded,
        note=binres.note,
        stats=stats,
    )


def load_image(path: str | Path) -> np.ndarray:
    """读图，兼容中文路径（与生成脚本中的写入方式对应）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"户型图不存在: {p}")
    buf = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图像: {p}")
    return img


def imencode_png(img: np.ndarray) -> bytes:
    """把图编码为 PNG 字节流（供 API 直接返回 base64，避免中文路径问题）。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return buf.tobytes()
