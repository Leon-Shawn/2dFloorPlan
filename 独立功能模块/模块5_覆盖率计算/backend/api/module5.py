"""模块 5「覆盖率计算与可视化」—— 独立包接口

模块归属
--------
覆盖率是题目直接命名的指标，本模块负责三件事：

1. **可达性BFS 连通域分析** —— 这是算法**固有环节**，不是补救措施。
   它给覆盖率一个数学上可达的上界（100%），并且在覆盖率掉下来时
   能区分两种完全不同的情况：
     - 地图断了（该区域 BFS 不可达，属物理死区，补扫也没用）
     - 算法漏了（该区域可达但没被覆盖，属缺陷，必须修）
   没有这一步，覆盖率只是个来路不明的百分数。

2. **覆盖率与去重统计** —— ``coverage = |cleaned ∩ reachable| / |reachable|``

3. **三色热力图数据** —— 输出0/1/2 栅格供前端Canvas 上色

函数实现在 ``planning/trajectory.py``（覆盖率与掩码）
与 ``planning/deadzone.py``（死区分类），本文件只做 HTTP 暴露。
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.rasterize import GridCell
from ..planning.strategies import SimParams, get_strategy
from ..planning.trajectory import coverage_ratio
from .common import grid_from_payload

router = APIRouter(prefix="/api/coverage", tags=["module5"])


class CoverageRequest(BaseModel):
    """栅格直传的覆盖率请求。"""

    grid: list
    strategy: str = Field("block_boustrophedon")
    clean_width: int = Field(2, ge=1, le=12)
    scan_axis: str = Field("row")
    start: list[int] | None = None
    sweep_dead: bool = Field(True)
    max_retry: int = Field(3, ge=0, le=10)


@router.post("/compute")
def compute_endpoint(req: CoverageRequest) -> dict:
    """计算覆盖率并给出完整归因。

    返回里的 ``diagnosis`` 字段是本模块的核心价值：
    覆盖率低时直接告诉你是「地图断了」还是「算法漏了」。
    """
    grid = grid_from_payload(req.grid)
    if not np.any(grid != GridCell.BLOCKED):
        raise HTTPException(400, "栅格全为障碍，无可通行区域")

    params = SimParams(
        clean_width=req.clean_width, scan_axis=req.scan_axis,
        start=tuple(req.start) if req.start else None,
        sweep_dead=req.sweep_dead, max_retry=req.max_retry,
    ).normalized()
    res = get_strategy(req.strategy).simulate(grid, params)
    m = res.metrics

    reach = int(np.count_nonzero(res.reachable))
    cleaned_reach = int(np.count_nonzero(res.cleaned & res.reachable))
    physical = sum(1 for z in res.dead_zones if not z.reachable)
    missed = sum(1 for z in res.dead_zones if z.reachable)

    if m["coverage"] > 0.999:
        diagnosis = "全部可达区域已覆盖，无遗漏"
    elif missed > 0:
        diagnosis = (
            f"存在 {missed} 处**算法漏扫**（BFS 可达但未覆盖），"
            f"属可修复缺陷。另有 {physical} 处物理死区，属地图本身断开，补扫无效。"
        )
    else:
        diagnosis = (
            f"覆盖率未满但无算法漏扫，缺口全部来自 {physical} 处物理死区"
            f"（门洞未打通等）。属地图输入问题，应回到模块 1/4 修正。"
        )

    return {
        "coverage": m["coverage"],
        "coverage_pct": round(m["coverage"] * 100, 2),
        "reachable_cells": reach,
        "cleaned_cells": cleaned_reach,
        "unique_visited": m["unique_visited"],
        "uncovered_cells": m["uncovered_cells"],
        "dead_zone_ratio": m["dead_zone_ratio"],
        "physical_dead_zones": physical,
        "algo_missed_zones": missed,
        "algo_miss": m["algo_miss"],
        "diagnosis": diagnosis,
        "heatmap_grid": res.grid.tolist(),
        "cell_names": {str(v): k for v, k in GridCell.NAMES.items()},
        "strategy": req.strategy,
    }


@router.post("/verify")
def verify_endpoint(req: CoverageRequest) -> dict:
    """独立校验覆盖率口径 —— 不信任 metrics，直接重算。

    存在的理由与项目里另一处校验工具同源：
    ``coverage = |visited ∩ reachable| / |reachable|`` 这个式子
    在路径误入墙格时**照样显示 100%**，指标本身有盲区。
    这里从 ``cleaned`` 与 ``reachable`` 两个掩码重新算一遍，
    与 ``metrics["coverage"]`` 对比，不一致就说明有问题。
    """
    grid = grid_from_payload(req.grid)
    if not np.any(grid != GridCell.BLOCKED):
        raise HTTPException(400, "栅格全为障碍，无可通行区域")

    params = SimParams(
        clean_width=req.clean_width, scan_axis=req.scan_axis,
        start=tuple(req.start) if req.start else None,
    ).normalized()
    res = get_strategy(req.strategy).simulate(grid, params)

    recomputed = coverage_ratio(res.cleaned, res.reachable)
    reported = res.metrics["coverage"]
    h, w = grid.shape
    on_wall = sum(1 for r, c in res.path
                  if 0 <= r < h and 0 <= c < w and grid[r][c] == GridCell.BLOCKED)
    jumps = sum(1 for i in range(1, len(res.path))
                if max(abs(res.path[i][0] - res.path[i - 1][0]),
                       abs(res.path[i][1] - res.path[i - 1][1])) > 1)

    return {
        "reported_coverage": reported,
        "recomputed_coverage": recomputed,
        "consistent": abs(recomputed - reported) < 1e-9,
        "path_on_wall_steps": on_wall,
        "path_jump_points": jumps,
        "note": (
            "覆盖率指标对「路径误入墙格」不敏感——它只统计掩码交集。"
            "故path_on_wall_steps 与 path_jump_points 必须独立看。"
        ),
    }