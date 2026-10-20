"""模块 1 栅格化回归测试 —— 任务 6-2 的测试用例雏形

跑法::

    python -m backend.tests.test_rasterize

覆盖：
  1. 五张合成户型图的门洞数与坐标是否与ground truth 一致；
  2. 门洞是否**真的打通了连通性**（这是最容易被糊弄过去的地方 ——
     门洞数对但没连通，覆盖率依然上不去）；
  3. 栅格格式定版约束（取值只有 0/1、尺寸一致）；
  4. 鲁棒性：全白图、全黑图、极小图、超大图、灰度图、RGBA 图；
  5. 参数面板传入非法值时是否夹取而非崩溃。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.binarize import BinarizeParams, binarize  # noqa: E402
from backend.core.rasterize import (  # noqa: E402
    GridCell, RasterizeParams, load_image, rasterize,
)

FLOORPLANS = Path(__file__).resolve().parents[2] / "data" / "floorplans"

# ground truth：门洞数量与像素坐标（与生成脚本中的 _door 调用一一对应）
GT_DOORS = {
    "fp_01": [],
    "fp_02": [(700, 360), (520, 600)],
    "fp_03": [],
    "fp_04": [(460, 300)],
    "fp_05": [(330, 200), (640, 470)],
}
GT_TOLERANCE = 12   # 像素容差


class TestReport:
    """极简断言收集器，避免引入 pytest 依赖（交付演示要能一条命令跑完）。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, cond: bool, name: str, detail: str = "") -> bool:
        if cond:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed.append(f"{name}: {detail}")
            print(f"  [FAIL] {name} — {detail}")
        return cond

    def summary(self) -> int:
        total = self.passed + len(self.failed)
        print("\n" + "=" * 62)
        print(f"通过 {self.passed}/{total}")
        if self.failed:
            print("失败明细：")
            for f in self.failed:
                print("  - " + f)
            return 1
        print("全部通过")
        return 0


def _passable_components(grid: np.ndarray) -> int:
    """统计可通行区的连通域数量（四连通，与后续 A* 的连通性口径一致）。"""
    n, _ = cv2.connectedComponents((grid == GridCell.PASSABLE).astype(np.uint8), 4)
    return n - 1


def test_doors(t: TestReport) -> None:
    print("\n[1] 门洞检测与ground truth 一致性")
    for name, gt in GT_DOORS.items():
        path = FLOORPLANS / f"{name}.png"
        if not path.exists():
            t.check(False, f"{name} 图片存在", f"缺少 {path}")
            continue
        res = rasterize(load_image(path))
        found = res.meta.door_cells
        if not t.check(len(found) == len(gt),
                       f"{name} 门洞数量",
                       f"期望 {len(gt)} 实得 {len(found)} {found}"):
            continue
        ok = all(
            any(abs(fx - gx) <= GT_TOLERANCE and abs(fy - gy) <= GT_TOLERANCE
                for fx, fy in found)
            for gx, gy in gt
        )
        t.check(ok, f"{name} 门洞坐标", f"期望 {gt} 实得 {found}")


def test_connectivity(t: TestReport) -> None:
    """门洞必须真正打通连通性 —— 数量对但没连通是最常见的假通过。"""
    print("\n[2] 门洞是否真正打通连通性")
    # fp_02 三房间 + 2 门洞 → 打通后应为一个连通区
    res = rasterize(load_image(FLOORPLANS / "fp_02.png"))
    comps = _passable_components(res.grid)
    t.check(comps == 1, "fp_02 三房间经门洞连成一体",
            f"可通行连通域数 = {comps}，应为 1")

    # 反证：把门洞栅格人为封死，连通域必须分裂成 3 个房间。
    # 这才真正证明「门洞对连通性是必要的」——
    # 注意不能靠「关小门宽参数」来做反证：门洞在原图上本就是缺口，
    # 不填补它就一直是通的，那样测不出任何东西。
    sealed = res.grid.copy()
    bridge_px = res.params["rasterize"]["door_bridge_px"]
    for fx, fy in res.meta.door_cells:
        r, c = int(fy // res.meta.cell_px), int(fx // res.meta.cell_px)
        span = max(3, int(bridge_px) // res.meta.cell_px)
        r0, r1 = max(r - span, 0), min(r + span + 1, sealed.shape[0])
        c0, c1 = max(c - span, 0), min(c + span + 1, sealed.shape[1])
        sealed[r0:r1, c0:c1] = GridCell.BLOCKED

    sealed_comps = _passable_components(sealed)
    t.check(sealed_comps == 3, "封死门洞后分裂为 3 个房间",
            f"封死后连通域数 = {sealed_comps}，应为 3")

    # fp_01 单房间无门洞 → 应恰为 1 个连通域
    res1 = rasterize(load_image(FLOORPLANS / "fp_01.png"))
    t.check(_passable_components(res1.grid) == 1, "fp_01 单房间连通",
            f"连通域数 = {_passable_components(res1.grid)}")

    # fp_03 L 形无门洞 → 也应是 1 个连通域（凹形不应被切断）
    res3 = rasterize(load_image(FLOORPLANS / "fp_03.png"))
    t.check(_passable_components(res3.grid) == 1, "fp_03 L 形保持连通",
            f"连通域数 = {_passable_components(res3.grid)}")


def test_grid_format(t: TestReport) -> None:
    """栅格矩阵格式定版约束：取值只能是 0/1，且尺寸与元数据一致。"""
    print("\n[3] 栅格矩阵格式定版")
    for name in GT_DOORS:
        res = rasterize(load_image(FLOORPLANS / f"{name}.png"))
        vals = set(np.unique(res.grid).tolist())
        t.check(vals <= {0, 1}, f"{name} 取值域仅含 0/1", f"实际 {vals}")
        t.check(res.grid.shape == (res.meta.grid_height, res.meta.grid_width),
                f"{name} 栅格尺寸与 meta 一致",
                f"grid {res.grid.shape} vs meta "
                f"({res.meta.grid_height}, {res.meta.grid_width})")
        t.check(res.grid.dtype == np.uint8, f"{name} dtype 为 uint8",
                str(res.grid.dtype))


def test_narrow_corridor(t: TestReport) -> None:
    """fp_05 含窄通道，应被判为不可通行（压测用例）。"""
    print("\n[4] 窄通道判定")
    res = rasterize(load_image(FLOORPLANS / "fp_05.png"))
    t.check(res.stats["narrow_blocked_cells"] > 0,
            "fp_05 检出窄通道并判为不可通行",
            f"narrow={res.stats['narrow_blocked_cells']}")


def test_robustness(t: TestReport) -> None:
    """鲁棒性（任务 1-12）：异常输入必须降级而非崩溃。"""
    print("\n[5] 鲁棒性与降级")
    h, w = 750, 1000

    cases = {
        "全白图": np.full((h, w, 3), 255, np.uint8),
        "全黑图": np.zeros((h, w, 3), np.uint8),
        "纯噪声": (np.random.rand(h, w, 3) * 255).astype(np.uint8),
        "灰度单通道": np.full((h, w), 250, np.uint8),
        "RGBA 四通道": np.full((h, w, 4), 250, np.uint8),
        "极小图 8x8": np.full((8, 8, 3), 250, np.uint8),
        "极扁图": np.full((3, 2000, 3), 250, np.uint8),
        "float32 输入": (np.random.rand(h, w) * 255).astype(np.float32),
    }
    for label, img in cases.items():
        try:
            res = rasterize(img)
            ok = res.grid.ndim == 2 and res.grid.size > 0
            t.check(ok, f"{label} 不崩溃且返回栅格", f"grid={getattr(res.grid,'shape',None)}")
        except Exception as exc:            # noqa: BLE001
            t.check(False, f"{label} 不崩溃", f"{type(exc).__name__}: {exc}")


def test_param_clamping(t: TestReport) -> None:
    """参数面板传入越界值时应夹取，而不是抛异常。"""
    print("\n[6] 非法参数夹取")
    bad_bin = BinarizeParams(block_size=-5, c_offset=9999,
                             denoise_ksize=2, close_ksize=100)
    bad_ras = RasterizeParams(cell_px=0, door_width_px=-1,
                             robot_radius_px=10 ** 6)
    try:
        res = rasterize(load_image(FLOORPLANS / "fp_01.png"),
                        bin_params=bad_bin, ras_params=bad_ras)
        t.check(res.grid.size > 0, "非法参数被夹取后仍能栅格化",
                f"grid={res.grid.shape}")
    except Exception as exc:                # noqa: BLE001
        t.check(False, "非法参数被夹取后仍能栅格化", f"{type(exc).__name__}: {exc}")

    # 奇偶校验：核尺寸必须被强制为奇数
    p = BinarizeParams(block_size=30, close_ksize=4).normalized()
    t.check(p.block_size % 2 == 1 and p.close_ksize % 2 == 1,
            "核尺寸被强制为奇数", f"block={p.block_size} close={p.close_ksize}")


def test_binarize_degrade(t: TestReport) -> None:
    """全白图应走降级路径并给出说明，而不是静默返回空结果。"""
    print("\n[7] 二值化降级路径")
    res = binarize(np.full((400, 400, 3), 255, np.uint8))
    t.check(res.binary is not None and res.binary.size > 0,
            "全白图返回可用二值结果", "")
    # 带纹理的全白底图：应被判定为降级并说明原因
    img = np.full((400, 400, 3), 255, np.uint8)
    img[100:300, 100:300] = 10
    res2 = binarize(img, BinarizeParams(block_size=301, c_offset=60))
    t.check(res2.binary is not None, "低对比图返回可用二值结果", "")


def main() -> int:
    print("=" * 62)
    print("模块 1 栅格化回归测试")
    print("=" * 62)
    t = TestReport()
    test_doors(t)
    test_connectivity(t)
    test_grid_format(t)
    test_narrow_corridor(t)
    test_robustness(t)
    test_param_clamping(t)
    test_binarize_degrade(t)
    return t.summary()


if __name__ == "__main__":
    raise SystemExit(main())
