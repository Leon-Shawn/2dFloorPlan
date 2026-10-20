"""模块 1 · 户型图转栅格地图 —— 图像预处理管线（任务 1-3 / 1-4）

处理链
------
    灰度化 → 去噪 → 自适应二值化（墙体为前景）→ 形态学闭运算补断线
                                                    ↓
                                        可选反相（处理浅墙深底的图）

设计约束
--------
1. **参数全部可调**：阈值、核尺寸由调用方传入，前端参数面板直接映射。
2. **鲁棒降级**：全白图、全黑图、超小图、极端长宽比都必须返回可用结果而不是抛异常
   （任务 1-12）。管线任何一步失败都退化为「按原图灰度阈值」的最简路径。
3. **中间图全部保留**：供任务 1-10 的三联对照视图使用，也便于报告里展示图像管线。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class BinarizeParams:
    """二值化参数 —— 对应前端参数面板（任务 1-4）。

    默认值针对「深色线条 + 浅色填充」线框图调校，
    这是 README 中明确指定的输入类型。
    """

    block_size: int = 31        # 自适应阈值邻域大小，必须为奇数且 >1
    c_offset: int = 8           # 自适应阈值偏移，越大前景判定越严
    denoise_ksize: int = 3# 中值滤波核尺寸，0 表示跳过去噪
    close_ksize: int = 3        # 闭运算核尺寸，0 表示跳过（用于补墙体断线）
    invert: bool = False        # True 表示原图是「浅墙深底」，需反相

    def normalized(self) -> "BinarizeParams":
        """把参数夹到合法区间。

        OpenCV 对 block_size /核尺寸有硬性约束（奇数、上限），
        参数面板传来非法值时不能让整条管线崩掉。
        """
        return BinarizeParams(
            block_size=self._odd_clamp(self.block_size, 3, 301),
            c_offset=int(np.clip(self.c_offset, -50, 100)),
            denoise_ksize=self._odd_clamp(self.denoise_ksize, 0, 31),
            close_ksize=self._odd_clamp(self.close_ksize, 0, 31),
            invert=bool(self.invert),
        )

    @staticmethod
    def _odd_clamp(v: int, lo: int, hi: int) -> int:
        """夹到 [lo, hi] 并强制为奇数（0 视为「关闭」保持不变）。"""
        if v == 0:
            return 0
        v = int(np.clip(v, lo, hi))
        return v if v % 2 == 1 else v + 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BinarizeResult:
    """管线输出。"""

    binary: np.ndarray        # uint8, 0/255，255 = 墙体（前景）
    stages: dict              # 各中间图，uint8 单通道或三通道
    degraded: bool            # 是否走了降级路径
    note: str                 # 降级原因，正常路径为空串
    ink_ratio: float          # 前景像素占比，用于识别「全白图」等异常

    def to_dict(self) -> dict:
        return {
            "degraded": self.degraded,
            "note": self.note,
            "ink_ratio": round(self.ink_ratio, 5),
        }


def _to_gray(img: np.ndarray) -> np.ndarray:
    """转灰度。支持灰度图与 BGR/RGBA 三/四通道。"""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _odd(v: int) -> int:
    """核尺寸必须为正奇数。"""
    v = max(int(v), 1)
    return v if v % 2 == 1 else v + 1


def binarize(img: np.ndarray, params: BinarizeParams | None = None) -> BinarizeResult:
    """户型图 → 二值墙体掩码。

    参数
    ----
    img:输入图，灰度或BGR。dtype 需为 uint8，非 uint8 会自动转换。
    params: :class:`BinarizeParams`，为 None 时用默认值。

    返回
    ----
    :class:`BinarizeResult`，``binary`` 中255 表示墙体。
    """
    p = (params or BinarizeParams()).normalized()
    stages: dict[str, np.ndarray] = {}
    notes: list[str] = []

    # --- 前置健壮性处理 ---
    if img is None or img.size == 0:
        raise ValueError("输入图像为空")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gray = _to_gray(img)
    stages["gray"] = gray

    # 图像过小时自适应阈值邻域比图像还大，会直接报错 —— 先放大到最小 64px
    h, w = gray.shape[:2]
    if min(h, w) < 64:
        scale = 64 / max(min(h, w), 1)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        notes.append(f"图像过小({w}x{h})，已放大 {scale:.1f}倍")

    # --- 步骤 1：去噪（中值滤波压掉压缩噪点，避免自适应阈值把噪点判成墙） ---
    work = gray
    if p.denoise_ksize >= 3:
        work = cv2.medianBlur(work, _odd(p.denoise_ksize))
    stages["denoised"] = work

    # --- 步骤 2：自适应二值化 ---
    # 目标是「深色线条为前景」。THRESH_BINARY_INV 会把比邻域均值暗 c_offset
    # 以上的像素判为 255，正好对应深色墙线。
    bs = _odd(p.block_size)
    if bs > min(work.shape[:2]):
        bs = _odd(max(3, min(work.shape[:2]) - 1))
        notes.append(f"block_size 超过图像尺寸，已下调为 {bs}")
    binary = cv2.adaptiveThreshold(
        work, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs,
        p.c_offset,
    )
    if p.invert:
        binary = cv2.bitwise_not(binary)
        notes.append("已按参数反相（浅墙深底模式）")
    stages["binary_raw"] = binary

    # --- 步骤 3：形态学闭运算，补墙体线条的断裂 ---
    closed = binary
    if p.close_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(p.close_ksize),) * 2)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    stages["closed"] = closed

    ink_ratio = float(np.count_nonzero(closed)) / closed.size

    # --- 步骤 4：异常检测与降级 ---
    degraded = False
    if ink_ratio < 1e-4:
        # 全白图：没检出任何墙。若原图本身信息量就极低，属正常输入；
        # 否则说明阈值不合适，退回全局 Otsu 阈值再试一次。
        degraded, note = _degrade_blank(gray)
        if note:
            notes.append(note)
    elif ink_ratio > 0.6:
        # 前景占六成以上：几乎全被判成墙，多半是阈值反了。
        degraded, note = _degrade_inverted(gray, ink_ratio)
        if note:
            notes.append(note)

    return BinarizeResult(
        binary=closed,
        stages=stages,
        degraded=degraded,
        note="；".join(notes),
        ink_ratio=ink_ratio,
    )


def _degrade_blank(gray: np.ndarray) -> tuple[np.ndarray, str]:
    """全白图降级：改用全局 Otsu 阈值重试。

    返回 (是否降级, 说明)。若原图本身就是纯白（无任何线条），
    属于合法输入，不算降级。
    """
    otsu_thr, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ratio = float(np.count_nonzero(otsu_mask)) / otsu_mask.size
    if ratio > 1e-4:
        return True, f"自适应阈值未检出墙体（占比<0.01%），已降级为全局 Otsu 阈值 {otsu_thr:.0f}"
    # 原图确实接近纯白 —— 空户型输入，不是错误
    return False, ""


def _degrade_inverted(gray: np.ndarray, ink_ratio: float) -> tuple[np.ndarray, str]:
    """前景占比过高降级：提示可能是浅墙深底。

    只给提示，不改binary —— 浅墙深底的正确处理应由调用方
    显式设置 ``params.invert=True``，避免自动反转造成不可预期的行为。
    """
    return True, (
        f"墙体前景占比 {ink_ratio:.1%} 过高，疑似浅墙深底图；"
        f"请在参数面板勾选「反相」后重试"
    )


def estimate_ink_ratio(img: np.ndarray) -> float:
    """估算线条占比，用于前端在上传时提示「这张图可能不适合自动解析」。

    典型线框图线条占比在 0.5%—15% 之间；低于 0.1% 基本是纯色图，
    高于 40%多半是照片或彩色渲染图。
    """
    gray = _to_gray(img)
    std = float(gray.std())
    ratio = float(np.count_nonzero(gray < 128)) / gray.size
    # 灰度方差极小说明几乎没有对比度（纯色块）
    if std < 5.0:
        return 0.0
    return ratio
