"""模块 3 API 路由 —— 全覆盖路径规划仿真

端点
----
``GET  /api/strategies``   列出三种覆盖策略（前端下拉）
``POST /api/simulate``     栅格（或图片/示例名）+ 策略 → 轨迹 + 指标
``POST /api/compare``      同一张栅格跑三种策略，返回对比表（模块 7 对比实验）
``POST /api/progress``     只算覆盖率曲线，不返回完整轨迹（前端 ECharts 用）
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import numpy as np

from ..planning.strategies import (
    STRATEGIES, SimParams, get_strategy,
)
from ..planning.trajectory import mark_visited, swath_mask
from ..core.rasterize import GridCell
from .common import grid_from_payload, grid_from_upload

router = APIRouter(prefix="/api", tags=["module3"])


@router.get("/strategies")
def strategies() -> dict:
    """列出可选覆盖策略 —— 任务 3-7「策略可插拔」的前端入口。"""
    return {
        "strategies": [
            {"name": cls.name, "description": cls.description,
             "official": cls.name == "block_boustrophedon",
             "role": "对照组" if cls.name == "random_walk" else "正式策略"}
            for cls in STRATEGIES.values()
        ],
        "default": "block_boustrophedon",
    }


class SimulateRequest(BaseModel):
    """直接提交栅格矩阵的仿真请求（模块 4 编辑器改完地图后走这条）。"""

    grid: list = Field(..., description="二维 0/1/2 数组")
    strategy: str = Field("block_boustrophedon")
    clean_width: int = Field(2)
    scan_axis: str = Field("row")
    start: list[int] | None = Field(None, description="起点 [row, col]，null=自动")
    step_time: float = Field(1.0)
    cell_area: float = Field(1.0)
    sweep_dead: bool = Field(True)
    max_retry: int = Field(3)
    revisit_penalty: float = Field(0.8)


def _params(
    clean_width: int = Form(2),
    scan_axis: str = Form("row"),
    start_r: int = Form(-1),
    start_c: int = Form(-1),
    step_time: float = Form(1.0),
    cell_area: float = Form(1.0),
    sweep_dead: bool = Form(True),
    max_retry: int = Form(3),
    revisit_penalty: float = Form(0.8),
) -> SimParams:
    """把表单参数打包成 SimParams（起点 -1 表示自动挑选）。"""
    sp = None
    if start_r >= 0 and start_c >= 0:
        sp = (start_r, start_c)
    return SimParams(
        clean_width=clean_width, scan_axis=scan_axis, start=sp,
        step_time=step_time, cell_area=cell_area, sweep_dead=sweep_dead,
        max_retry=max_retry, revisit_penalty=revisit_penalty,
    ).normalized()


def _run(grid: np.ndarray, strategy: str, params: SimParams) -> dict:
    res = get_strategy(strategy).simulate(grid, params)
    out = res.to_dict()
    out["source"] = "payload"
    return out


@router.post("/simulate")
async def simulate_endpoint(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
    strategy: str = Form("block_boustrophedon"),
    clean_width: int = Form(2),
    scan_axis: str = Form("row"),
    start_r: int = Form(-1),
    start_c: int = Form(-1),
    step_time: float = Form(1.0),
    cell_area: float = Form(1.0),
    sweep_dead: bool = Form(True),
    max_retry: int = Form(3),
    revisit_penalty: float = Form(0.8),
    cell_px: int = Form(10),
    block_size: int = Form(31),
    c_offset: int = Form(8),
    close_ksize: int = Form(3),
    wall_dilate: int = Form(1),
    door_width_px: int = Form(110),
    robot_radius_px: int = Form(14),
) -> dict:
    """执行一次覆盖仿真。

    输入三选一：``grid``（编辑器产出）、``file``（上传图）、``sample``（示例图）。
    后两者会先走一遍模块 1 的栅格化管线。

    返回轨迹点序列、清扫掩码、指标、BCD 单元、死区与防震荡统计。
    """
    params = _params(clean_width, scan_axis, start_r, start_c,
                     step_time, cell_area, sweep_dead, max_retry, revisit_penalty)

    if file is not None or sample:
        res, source = await grid_from_upload(
            file, sample, cell_px=cell_px, block_size=block_size,
            c_offset=c_offset, close_ksize=close_ksize,
            wall_dilate=wall_dilate, door_width_px=door_width_px,
            robot_radius_px=robot_radius_px,
        )
        out = get_strategy(strategy).simulate(res.grid, params).to_dict()
        out["source"] = source
        out["meta"] = res.meta.to_dict()
        return out

    raise HTTPException(400, "请提供 sample、file 之一（栅格直传请用 POST /api/simulate-grid）")


@router.post("/simulate-grid")
def simulate_grid_endpoint(req: SimulateRequest) -> dict:
    """直接提交栅格矩阵仿真 —— 模块 4 编辑器「运行仿真」按钮走这条。"""
    grid = grid_from_payload(req.grid)
    params = SimParams(
        clean_width=req.clean_width, scan_axis=req.scan_axis,
        start=tuple(req.start) if req.start else None,
        step_time=req.step_time, cell_area=req.cell_area,
        sweep_dead=req.sweep_dead, max_retry=req.max_retry,
        revisit_penalty=req.revisit_penalty,
    ).normalized()
    if not np.any(grid != GridCell.BLOCKED):
        raise HTTPException(400, "栅格全为障碍，无可通行区域")
    return _run(grid, req.strategy, params)


@router.post("/compare")
async def compare_endpoint(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
    clean_width: int = Form(2),
    scan_axis: str = Form("row"),
    step_time: float = Form(1.0),
    cell_area: float = Form(1.0),
    sweep_dead: bool = Form(True),
    max_retry: int = Form(3),
    revisit_penalty: float = Form(0.8),
    cell_px: int = Form(10),
) -> dict:
    """同一张栅格跑全部三种策略，返回对比表。

    任务 3-10 对比实验：证明 BCD 策略在覆盖率/重复率上优于随机游走。
    随机游走步数上限为 6×可达格数，因此耗时明显长于正式策略。
    """
    if file is None and not sample:
        raise HTTPException(400, "请提供 file 或 sample 之一")
    res, source = await grid_from_upload(file, sample, cell_px=cell_px)

    rows = []
    for name in STRATEGIES:
        p = SimParams(clean_width=clean_width, scan_axis=scan_axis,
                      step_time=step_time, cell_area=cell_area,
                      sweep_dead=sweep_dead, max_retry=max_retry,
                      revisit_penalty=revisit_penalty).normalized()
        r = get_strategy(name).simulate(res.grid, p)
        rows.append({"strategy": name, **r.metrics})

    return {"source": source, "meta": res.meta.to_dict(), "rows": rows}


@router.post("/progress")
async def progress_endpoint(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
    strategy: str = Form("block_boustrophedon"),
    clean_width: int = Form(2),
    points: int = Form(120),
    cell_px: int = Form(10),
) -> dict:
    """覆盖率 / 重复率随步数的收敛曲线 —— 前端 ECharts 折线图。

    抽稀计算：轨迹可能有上万步，逐��算距离变换太慢，
    因此只在下采样后的采样点处计算曲线。
    """
    if file is None and not sample:
        raise HTTPException(400, "请提供 file 或 sample 之一")
    res, source = await grid_from_upload(file, sample, cell_px=cell_px)
    params = SimParams(clean_width=clean_width).normalized()
    sim = get_strategy(strategy).simulate(res.grid, params)

    total = len(sim.path)
    n = max(2, min(points, total))
    idx = np.unique(np.linspace(0, total - 1, n).astype(int))

    cov_curve, rep_curve, steps_axis = [], [], []
    from ..planning.grid_ops import bfs_reachable
    reachable = sim.reachable
    area = max(1, int(np.count_nonzero(reachable)))
    for i in idx:
        prefix = sim.path[: int(i) + 1]
        cleaned = swath_mask(res.grid, prefix, params.clean_width)
        visited = mark_visited(res.grid, prefix)
        cov_curve.append(round(float(np.count_nonzero(cleaned & reachable)) / area, 4))
        steps = max(0, len(prefix) - 1)
        rep_curve.append(round(1.0 - np.count_nonzero(visited) / max(1, steps), 4))
        steps_axis.append(steps)

    return {
        "source": source,
        "strategy": strategy,
        "steps": steps_axis,
        "coverage": cov_curve,
        "repeat_rate": rep_curve,
        "final": sim.metrics,
    }
