"""模块 4 API 路由 —— 栅格地图编辑器

端点
----
``POST /api/editor/grid``       新建空白栅格 / 读取栅格
``POST /api/editor/validate``   校验栅格并给出连通性诊断
``POST /api/editor/fill``       矩形区域刷墙（前端拖拽绘制用）
``POST /api/editor/statistics`` 地图统计（房间数、面积、可达率）
``GET  /api/editor/presets``    预置教学地图
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.rasterize import GridCell
from ..planning.grid_ops import (
    bbox_of, bfs_reachable, find_start, label_components, passable_mask,
)
from .common import grid_from_payload

router = APIRouter(prefix="/api/editor", tags=["module4"])

MAX_DIM = 400


class NewGridRequest(BaseModel):
    height: int = Field(40, ge=4, le=MAX_DIM)
    width: int = Field(60, ge=4, le=MAX_DIM)
    cell_px: int = Field(10, ge=2, le=60)


class FillRequest(BaseModel):
    grid: list
    r0: int
    c0: int
    r1: int
    c1: int
    value: int = Field(1, ge=0, le=2)


class GridOnlyRequest(BaseModel):
    """只需栅格、不需要矩形坐标的端点用这个。"""

    grid: list


def _diagnose(grid: np.ndarray) -> dict:
    """连通性诊断 —— 编辑器的核心反馈：改完地图立刻知道能不能跑。"""
    pm = passable_mask(grid)
    start = find_start(grid, None)
    reachable = bfs_reachable(grid, start)
    labels, n = label_components(pm)
    unreachable = int(np.count_nonzero(pm)) - int(np.count_nonzero(reachable))
    bb = list(bbox_of(np.argwhere(pm))) if pm.any() else None
    return {
        "start": [int(start[0]), int(start[1])],
        "components": int(n),
        "passable": int(np.count_nonzero(pm)),
        "reachable": int(np.count_nonzero(reachable)),
        "unreachable": unreachable,
        "coverage_base": round(
            float(np.count_nonzero(reachable)) / max(1, int(np.count_nonzero(pm))), 4),
        "bbox": bb,
        "labels": labels.astype(int).tolist(),
        "ok": unreachable == 0 and n >= 1,
    }


@router.post("/grid")
def new_grid(req: NewGridRequest) -> dict:
    """新建空白栅格 —— 四周补墙，内部全可通行。"""
    g = np.ones((req.height, req.width), dtype=np.uint8) * GridCell.PASSABLE
    g[0, :] = g[-1, :] = GridCell.BLOCKED
    g[:, 0] = g[:, -1] = GridCell.BLOCKED
    diag = _diagnose(g)
    return {
        "grid": g.tolist(),
        "cell_px": req.cell_px,
        "height": req.height,
        "width": req.width,
        **diag,
    }


@router.post("/validate")
def validate_endpoint(req: GridOnlyRequest) -> dict:
    """校验一份栅格并返回连通性诊断 —— 前端每次编辑后调用。"""
    g = grid_from_payload(req.grid)
    return {"grid": g.tolist(), **_diagnose(g)}


@router.post("/fill")
def fill_endpoint(req: FillRequest) -> dict:
    """矩形刷格 —— 前端鼠标拖拽画墙/擦除的原子操作。

    坐标自动夹取到栅格范围内，越界不报错（拖拽出界是常态）。
    """
    g = grid_from_payload(req.grid).copy()
    h, w = g.shape
    r0, r1 = sorted((max(0, min(h - 1, req.r0)), max(0, min(h - 1, req.r1))))
    c0, c1 = sorted((max(0, min(w - 1, req.c0)), max(0, min(w - 1, req.c1))))
    g[r0:r1 + 1, c0:c1 + 1] = req.value
    return {"grid": g.tolist(), **_diagnose(g)}


@router.post("/statistics")
def statistics_endpoint(req: GridOnlyRequest) -> dict:
    """地图统计：按连通域切出「房间」，给出每间面积。"""
    g = grid_from_payload(req.grid)
    pm = passable_mask(g)
    labels, n = label_components(pm)
    start = find_start(g, None)
    reachable = bfs_reachable(g, start)

    rooms = []
    for i in range(1, n + 1):
        cells = np.argwhere(labels == i)
        if cells.size == 0:
            continue
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        rooms.append({
            "id": int(i),
            "area": int(cells.shape[0]),
            "bbox": [int(r0), int(c0), int(r1), int(c1)],
            "reachable": bool(np.any(reachable[cells[:, 0], cells[:, 1]])),
        })
    rooms.sort(key=lambda x: -x["area"])

    return {
        "rooms": rooms,
        "room_count": len(rooms),
        "total_passable": int(np.count_nonzero(pm)),
        "reachable": int(np.count_nonzero(reachable)),
        "unreachable": int(np.count_nonzero(pm)) - int(np.count_nonzero(reachable)),
        "has_isolated_room": any(not r["reachable"] for r in rooms),
        **({"start": [int(start[0]), int(start[1])]} if n else {"start": None}),
    }


@router.get("/presets")
def presets() -> dict:
    """预置教学地图 —— 编辑器「载入示例」下拉。

    三个刻意设计的场景：连通 / 隔离房间 / 狭窄通道，
    覆盖死区判定的三种成因。
    """
    out = []

    # 1) L 形开放空间：一切正常
    g = np.full((24, 32), GridCell.PASSABLE, np.uint8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = GridCell.BLOCKED
    g[12:, 18:] = GridCell.BLOCKED
    out.append({"name": "L 形开放空间", "desc": "单一连通域，基准场景", "grid": g.tolist()})

    # 2) 三室两门：中间房间被墙隔开，考验门洞
    g = np.full((24, 32), GridCell.PASSABLE, np.uint8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = GridCell.BLOCKED
    g[12, 1:12] = GridCell.BLOCKED
    g[12, 21:31] = GridCell.BLOCKED
    g[1:12, 17] = GridCell.BLOCKED
    g[13:23, 17] = GridCell.BLOCKED
    out.append({"name": "三室两门", "desc": "两处门洞决定连通性", "grid": g.tolist()})

    # 3) 孤立房间：完全封死，用于演示「物理死区」
    g = np.full((24, 32), GridCell.PASSABLE, np.uint8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = GridCell.BLOCKED
    g[5:14, 22:29] = GridCell.BLOCKED
    out.append({"name": "含孤立房间", "desc": "右上角房间无门，必然漏扫", "grid": g.tolist()})

    # 4) 窄通道：两列障碍夹出的 1 格通道
    g = np.full((24, 32), GridCell.PASSABLE, np.uint8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = GridCell.BLOCKED
    g[4:20, 14] = GridCell.BLOCKED
    g[4:20, 16] = GridCell.BLOCKED
    out.append({"name": "窄通道", "desc": "1 格通道，验证机身半径约束", "grid": g.tolist()})

    return {"presets": out}


@router.get("/health")
def editor_health() -> dict:
    return {"status": "ok", "module": "module4-editor"}
