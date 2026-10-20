"""合成户型平面图生成器 —— 开发期基准数据集

为什么需要它
------------
原计划用HouseExpo 的真实户型图，但其图片托管在 Google Drive，
国内访问不稳定，且原始 PNG 风格不统一会让二值化调参变成玄学。

本模块用 OpenCV 直接绘制「深色线条 + 浅色填充」的线框图，
与 README 中「输入类型优先线框图」的要求一致，风格可控、可复现，
用于：
  1. 模块 1 栅格化管线的开发与回归测试；
  2. ground truth 标注（墙体/门洞/家具位置已知，IoU 基线可信）；
  3. HouseExpo 图片到位前的过渡。

用法::

    python -m backend.scripts.make_synthetic_plans
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "data" / "floorplans"

# 统一画布尺寸：足够大以体现细节，又不会让二值化变慢
CANVAS = (1000, 750)
BG = (250, 250, 250)        # 浅色填充（三通道必须显式给元组，传标量会被 OpenCV 解释成 (v,0,0) 即蓝色）
INK = (40, 40, 40)          # 深色墙线
WALL = 5                    # 墙线线宽


def _canvas() -> np.ndarray:
    """空白画布。"""
    img = np.full((CANVAS[1], CANVAS[0], 3), BG, dtype=np.uint8)
    return img


def _rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
          thickness: int = WALL) -> None:
    cv2.rectangle(img, p1, p2, INK, thickness)


def _line(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
          thickness: int = WALL) -> None:
    cv2.line(img, p1, p2, INK, thickness)


def _door(img: np.ndarray, center: tuple[int, int], vertical: bool,
          width: int = 70, thickness: int = WALL) -> None:
    """在墙上挖一个门洞：把线段擦成背景色即形成缺口。

    真实户型图的门洞表现为墙体线段的**缺口**（HouseExpo 矢量数据同样如此），
    没有显式门洞标注，因此模块 1 只能靠形态学检测缺口 —— 这正是任务 1-7。

    几何约定（弄反会让门洞变成「横切墙体的细缝」，而不是「墙上的门」）：

        门在**竖直墙**上：墙沿 y 延伸 → 门宽沿 y，厚度沿 x
            擦除区域 x∈[xc-t, xc+t], y∈[yc-w/2, yc+w/2]
        门在**水平墙**上：墙沿 x 延伸 → 门宽沿 x，厚度沿 y
            擦除区域 x∈[xc-w/2, xc+w/2], y∈[yc-t, yc+t]
    """
    x, y = center
    half = width // 2
    if vertical:
        cv2.rectangle(img, (x - thickness, y - half), (x + thickness, y + half),
                      BG, -1)
    else:
        cv2.rectangle(img, (x - half, y - thickness), (x + half, y + thickness),
                      BG, -1)


def _furniture(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> None:
    """家具/障碍：浅灰填充 + 深色边框，与墙体明显区分。"""
    cv2.rectangle(img, p1, p2, (205, 205, 205), -1)
    cv2.rectangle(img, p1, p2, (120, 120, 120), 2)


# --------------------------------------------------------------------------
# fp_01单房间方形 —— 最简单，验证全链路能跑通
# --------------------------------------------------------------------------
def fp_01() -> np.ndarray:
    img = _canvas()
    _rect(img, (120, 90), (880, 660))
    return img


# --------------------------------------------------------------------------
# fp_02  三房间 + 门洞 —— 验证门洞检测（任务 1-7）是否有效
# --------------------------------------------------------------------------
def fp_02() -> np.ndarray:
    img = _canvas()
    # 外墙
    _rect(img, (100, 80), (900, 670))
    # 中间竖墙，把房子分成左右两半，竖墙下方开门洞
    _line(img, (520, 80), (520, 670))
    _door(img, (520, 600), vertical=True)
    # 右半部分再切一刀，上方开门洞
    _line(img, (520, 360), (900, 360))
    _door(img, (700, 360), vertical=False)
    return img


# --------------------------------------------------------------------------
# fp_03  L 形/ 凹形 —— 验证 BCD 区域分解能否正确切分单元（任务 2-1）
# --------------------------------------------------------------------------
def fp_03() -> np.ndarray:
    img = _canvas()
    # L 形：主体 + 右下角缺口
    pts = [(120, 90), (880, 90), (880, 420), (520, 420), (520, 670), (120, 670)]
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, INK, WALL)
    return img


# --------------------------------------------------------------------------
# fp_04  家具密集 —— 验证障碍物标记（任务 1-8）
# --------------------------------------------------------------------------
def fp_04() -> np.ndarray:
    img = _canvas()
    _rect(img, (100, 80), (900, 670))
    _furniture(img, (180, 150), (420, 320))     # 床
    _furniture(img, (180, 400), (380, 610))     # 衣柜
    _furniture(img, (520, 150), (830, 300))     # 沙发
    _furniture(img, (600, 420), (830, 610))     # 餐桌
    _line(img, (460, 80), (460, 670))           # 一道隔断墙
    _door(img, (460, 300), vertical=True)
    return img


# --------------------------------------------------------------------------
# fp_05  窄通道 —— 通道窄于机身，应判为不可通行，压测用
# --------------------------------------------------------------------------
def fp_05() -> np.ndarray:
    img = _canvas()
    _rect(img, (100, 80), (900, 670))
    # 三条竖墙构成两条 22px 的窄通道（远小于典型机身宽度）
    for x in (330, 356, 640, 666):
        _line(img, (x, 80), (x, 670))
    # 门洞开在各隔断墙顶部
    _door(img, (330, 200), vertical=True)
    _door(img, (640, 470), vertical=True)
    return img


BUILDERS = {
    "fp_01": fp_01,
    "fp_02": fp_02,
    "fp_03": fp_03,
    "fp_04": fp_04,
    "fp_05": fp_05,
}

DESCRIPTIONS = {
    "fp_01": "单房间方形，无门洞无遮挡，验证全链路",
    "fp_02": "三房间 + 2 个门洞，验证门洞检测",
    "fp_03": "L 形凹形，验证 BCD 区域分解",
    "fp_04": "家具密集，验证障碍物标记",
    "fp_05": "含窄通道（22px），压测不可通行判定",
}


def _imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """写PNG，兼容中文路径。

    OpenCV 的 cv2.imwrite 在 Windows 上无法处理非ASCII 路径，
    失败时**不抛异常、也不写文件**，只是静默返回 False。
    这里改用 imencode + tofile 绕开该限制。
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {path}")
    buf.tofile(str(path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        img = fn()
        path = OUT_DIR / f"{name}.png"
        _imwrite_unicode(path, img)
        print(f"[ok] {path.name}  {img.shape[1]}x{img.shape[0]}  {DESCRIPTIONS[name]}")

    manifest = {
        name: {
            "file": f"{name}.png",
            "description": DESCRIPTIONS[name],
            "width": CANVAS[0],
            "height": CANVAS[1],
            "source": "synthetic",
        }
        for name in BUILDERS
    }
    mpath = OUT_DIR / "synthetic_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {mpath.name}")


if __name__ == "__main__":
    main()
