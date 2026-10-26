"""模块 9 API 路由 —— 仿真回放

端点
----
``POST /api/replay/frames``  轨迹 → 分帧数据（前端逐帧播放动画）
``POST /api/replay/summary`` 回放摘要：分段耗时、死区时间点

设计要点：帧数据用**增量**（每帧新增的清扫格）而不是全量快照，
否则 2000 步 × 200×300 栅格的 JSON 会到几百 MB。
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.rasterize import GridCell
from ..planning.strategies import SimParams, get_strategy
from .common import grid_from_payload

router = APIRouter(prefix="/api/replay", tags=["module9"])

MAX_FRAMES = 600
MAX_ADD_PER_FRAME = 6000     # 单帧最多携带的新增格数
MAX_SAFE_CELLS = 900_000     # 全部帧 add 坐标对总量上限（约 6~10MB JSON）


class ReplayRequest(BaseModel):
    grid: list
    strategy: str = Field("block_boustrophedon")
    clean_width: int = Field(2, ge=1, le=12)
    scan_axis: str = Field("row")
    step_time: float = Field(1.0, ge=0.01, le=100)
    cell_area: float = Field(1.0, ge=1e-4, le=1e4)
    frames: int = Field(240, ge=2, le=MAX_FRAMES)
    sweep_dead: bool = Field(True)
    max_retry: int = Field(3)
    revisit_penalty: float = Field(0.8)


def _cleaned_prefix(grid: np.ndarray, path: list, upto: int, width: int) -> np.ndarray:
    """前 ``upto`` 步轨迹的清扫带掩码。

    用 ``scipy`` 距离变换的等价实现（cv2.distanceTransform）
    与 trajectory.swath_mask 保持同一口径。
    """
    from ..planning.trajectory import swath_mask
    return swath_mask(grid, path[:upto + 1], width)


@router.post("/frames")
def frames_endpoint(req: ReplayRequest) -> dict:
    """生成回放帧序列。

    返回 ``frames[i] = {step, pos:[r,c], add:[[r,c],...]}``：
    ``add`` 是第i 帧相对上一帧**新增**的清扫格，
    前端按累加方式上色即可，无需每帧重传整张掩码。
    """
    from ..core.rasterize import GridCell as GC
    grid = grid_from_payload(req.grid)
    if not np.any(grid != GC.BLOCKED):
        raise HTTPException(400, "栅格全为障碍，无可通行区域")

    params = SimParams(
        clean_width=req.clean_width, scan_axis=req.scan_axis,
        step_time=req.step_time, cell_area=req.cell_area,
        sweep_dead=req.sweep_dead, max_retry=req.max_retry,
        revisit_penalty=req.revisit_penalty,
    ).normalized()
    res = get_strategy(req.strategy).simulate(grid, params)

    total = len(res.path)
    n = min(req.frames, max(2, total))
    idx = np.unique(np.linspace(0, total - 1, n).astype(int))

    # 增量帧的总体积上限。前端靠累加 add 上色，一旦某帧被截断，
    # 累积结果就会缺格、画布出现永远补不上的空白。
    # 因此先按「轨迹总长× 幅宽」估一个安全帧数，宁可帧少也不能截断。
    est_cells = int(np.count_nonzero(res.cleaned))
    safe_frames = max(2, min(n,MAX_SAFE_CELLS // max(1, est_cells // len(idx))))
    truncated = safe_frames < n
    if truncated:
        idx = np.unique(np.linspace(0, total - 1, safe_frames).astype(int))
        n = safe_frames

    prev = np.zeros(grid.shape, dtype=bool)
    frames = []
    truncated_cells = 0
    for i in idx:
        cur = _cleaned_prefix(grid, res.path, int(i), params.clean_width)
        add = cur & ~prev
        rr, cc = np.nonzero(add)
        if rr.size > MAX_ADD_PER_FRAME:
            #仍会超单帧上限时如实上报，让前端知道着色不完整
            truncated_cells += int(rr.size - MAX_ADD_PER_FRAME)
        frames.append({
            "step": int(i),
            "pos": [int(res.path[int(i)][0]), int(res.path[int(i)][1])],
            "add": [[int(a), int(b)] for a, b in zip(rr[:MAX_ADD_PER_FRAME],
                                                    cc[:MAX_ADD_PER_FRAME])],
            "add_count": int(add.sum()),
        })
        prev = cur

    return {
        "strategy": req.strategy,
        "shape": [int(grid.shape[0]), int(grid.shape[1])],
        "total_steps": int(res.metrics["total_steps"]),
        "frames": frames,
        "metrics": res.metrics,
        "dead_zones": [z.to_dict() for z in res.dead_zones],
        "cells": [{"bbox": list(c.bbox), "area": c.area} for c in res.cells],
        "duration_s": round(res.metrics["area_time_cost"], 2),
        # 如实告知前端是否有格子因体积上限被丢弃
        "complete": truncated_cells == 0,
        "dropped_cells": truncated_cells,
    }


@router.post("/summary")
def summary_endpoint(req: ReplayRequest) -> dict:
    """回放摘要 —— 报告 3.x「仿真过程分析」直接引用这张表。"""
    grid = grid_from_payload(req.grid)
    params = SimParams(
        clean_width=req.clean_width, scan_axis=req.scan_axis,
        step_time=req.step_time, cell_area=req.cell_area,
        sweep_dead=req.sweep_dead, max_retry=req.max_retry,
        revisit_penalty=req.revisit_penalty,
    ).normalized()
    res = get_strategy(req.strategy).simulate(grid, params)

    total = max(1, res.metrics["total_steps"])
    stage = [
        {"name": "单元数（BCD）", "value": len(res.cells)},
        {"name": "总步数", "value": res.metrics["total_steps"]},
        {"name": "轨迹点数", "value": len(res.path)},
        {"name": "唯一访问格", "value": res.metrics["unique_visited"]},
        {"name": "清扫覆盖格", "value": res.metrics["cleaned_cells"]},
        {"name": "可达格", "value": res.metrics["reachable_cells"]},
        {"name": "算法漏扫格", "value": res.metrics["algo_miss"]},
        {"name": "物理死区数", "value": sum(1 for z in res.dead_zones if not z.reachable)},
        {"name": "平均步长(格)", "value": round(total / max(1, len(res.cells)), 1)},
    ]

    return {
        "stages": stage,
        "metrics": res.metrics,
        "guard": res.guard_stats,
        "dead_zones": [
            {**z.to_dict(), "cause_label": z.cause}   # cause 本身就是中文标签
            for z in res.dead_zones
        ],
    }


@router.get("/health")
def replay_health() -> dict:
    return {"status": "ok", "module": "module9-replay"}
