"""模块 7「路径效率指标」—— 独立包接口

模块归属
--------
覆盖率回答「扫没扫全」，本模块回答「扫得亏不亏」。三个指标：

1. **重复率** ``repeat_rate = 1 − |unique_visited| / |total_steps|``
   路径冗余程度。转场越远、单元划分越碎，重复率越高。

2. **单位面积耗时** ``total_steps × step_time / (|reachable| × cell_area)``
   把步数换算成时间与面积的比值，跨户型可比。

3. **对比实验** 三策略同图同参数横向对比

口径说明（重要）
----------------
``total_steps`` 取**补齐后**的路径长度，即机器人真实走过的步数。

早期版本取补齐前的长度，理由是「补齐点本就在覆盖掩码里，
不该计入分母」。这条路走不通：

- 前端回放帧用的是补齐后的长度，而 total_steps 字段用补齐前的值，
  两者不一致导致进度条显示「4377 / 3646 步」并停在 88%；
- 补齐点代表机器人真实走过的过渡步（单元间转场、换扫描线），
  排除它们等于假设机器人能瞬移。重复率正是度量这类冗余的指标，
  排除它就使指标失去意义。

现在三者完全自洽：``len(path) == total_steps + 1 == 回放帧号上限``。
代价是分块牛耕的重复率从 17~28% 变为 33~41%——
这个「数字变差」其实是变对了，旧数字建立在瞬移假设上。
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.rasterize import GridCell
from ..planning.strategies import SimParams, get_strategy
from .common import grid_from_payload

router = APIRouter(prefix="/api/efficiency", tags=["module7"])


class EfficiencyRequest(BaseModel):
    """栅格直传的效率指标请求。"""

    grid: list
    strategy: str = Field("block_boustrophedon")
    clean_width: int = Field(2, ge=1, le=12)
    scan_axis: str = Field("row")
    start: list[int] | None = None
    step_time: float = Field(1.0, gt=0, le=100)
    cell_area: float = Field(1.0, gt=0, le=1e4)
    with_comparison: bool = Field(True, description="是否附带三策略对比")


@router.post("/compute")
def compute_endpoint(req: EfficiencyRequest) -> dict:
    """计算路径效率指标。"""
    grid = grid_from_payload(req.grid)
    if not np.any(grid != GridCell.BLOCKED):
        raise HTTPException(400, "栅格全为障碍，无可通行区域")

    params = SimParams(
        clean_width=req.clean_width, scan_axis=req.scan_axis,
        start=tuple(req.start) if req.start else None,
        step_time=req.step_time, cell_area=req.cell_area,
    ).normalized()
    res = get_strategy(req.strategy).simulate(grid, params)
    m = res.metrics

    h, w = grid.shape
    on_wall = sum(1 for r, c in res.path
                  if 0 <= r < h and 0 <= c < w and grid[r][c] == GridCell.BLOCKED)

    return {
        "strategy": m["strategy"],
        "total_steps": m["total_steps"],
        "unique_visited": m["unique_visited"],
        "repeat_rate": m["repeat_rate"],
        "repeat_rate_pct": round(m["repeat_rate"] * 100, 2),
        "area_time_cost": m["area_time_cost"],
        "reachable_cells": m["reachable_cells"],
        "path_points": len(res.path),
        # 自洽性：这三个数必须吻合，否则说明口径又分叉了
        "self_consistent": len(res.path) == m["total_steps"] + 1,
        "path_on_wall_steps": on_wall,
        "formula": {
            "repeat_rate": "1 − |unique_visited| / |total_steps|",
            "area_time_cost": "total_steps × step_time / (|reachable| × cell_area)",
        },
    }


@router.post("/compare")
def compare_endpoint(req: EfficiencyRequest) -> dict:
    """三策略同图同参数横向对比 —— 报告 2.3(C) 的数据来源。"""
    grid = grid_from_payload(req.grid)
    if not np.any(grid != GridCell.BLOCKED):
        raise HTTPException(400, "栅格全为障碍，无可通行区域")

    rows = []
    for name in ("boustrophedon", "block_boustrophedon", "random_walk"):
        params = SimParams(
            clean_width=req.clean_width, scan_axis=req.scan_axis,
            start=tuple(req.start) if req.start else None,
            step_time=req.step_time, cell_area=req.cell_area,
        ).normalized()
        res = get_strategy(name).simulate(grid, params)
        m = res.metrics
        rows.append({
            "strategy": name,
            "role": "对照组" if name == "random_walk" else "正式策略",
            "coverage": m["coverage"],
            "repeat_rate": m["repeat_rate"],
            "total_steps": m["total_steps"],
            "area_time_cost": m["area_time_cost"],
            "algo_miss": m["algo_miss"],
        })

    best_cov = max(rows, key=lambda r: r["coverage"])
    best_rep = min(rows, key=lambda r: r["repeat_rate"])
    return {
        "rows": rows,
        "best_coverage": best_cov["strategy"],
        "best_repeat_rate": best_rep["strategy"],
        "conclusion": (
            f"覆盖率最高：{best_cov['strategy']}（{best_cov['coverage'] * 100:.1f}%）；"
            f"重复率最低：{best_rep['strategy']}"
            f"（{best_rep['repeat_rate'] * 100:.1f}%）"
        ),
    }