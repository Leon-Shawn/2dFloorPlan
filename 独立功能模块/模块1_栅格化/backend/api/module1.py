"""模块 1 API 路由 —— 户型图转栅格地图

端点
----
``GET  /api/samples``    列出内置示例户型图
``POST /api/rasterize``  上传图片或指定示例名 → 返回栅格矩阵 + 中间图
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..core.binarize import BinarizeParams, binarize
from ..core.rasterize import (
    GridCell,
    RasterizeParams,
    imencode_png,
    load_image,
    rasterize,
)

router = APIRouter(prefix="/api", tags=["module1"])

BASE_DIR = Path(__file__).resolve().parents[2]
FLOORPLAN_DIR = BASE_DIR / "data" / "floorplans"

# 上传体积上限，防止误传大图把内存打满
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _b64(img: np.ndarray) -> str:
    """图像 → base64 data URI，供前端 <img> 直接显示。

    走内存编码而不是落盘，规避 Windows 中文路径下 cv2.imwrite 静默失败的问题。
    """
    return "data:image/png;base64," + base64.b64encode(imencode_png(img)).decode()


def _sample_images() -> list[dict]:
    """扫描内置示例图目录。"""
    if not FLOORPLAN_DIR.exists():
        return []
    out = []
    for p in sorted(FLOORPLAN_DIR.glob("*.png")):
        out.append({"name": p.stem, "file": p.name})
    return out


@router.get("/samples")
def samples() -> dict:
    """列出内置示例户型图，供前端下拉选择。"""
    return {"samples": _sample_images(), "dir": FLOORPLAN_DIR.name}


class RasterizeRequest(BaseModel):
    """用内置示例图栅格化时的请求体（无需上传文件）。"""

    sample: str = Field(..., description="示例图名称，如 fp_02")


@router.post("/rasterize")
async def rasterize_endpoint(
    file: UploadFile | None = File(None, description="户型图文件"),
    sample: str | None = Form(None, description="内置示例图名称"),
    cell_px: int = Form(10),
    block_size: int = Form(31),
    c_offset: int = Form(8),
    close_ksize: int = Form(3),
    denoise_ksize: int = Form(3),
    invert: bool = Form(False),
    wall_dilate: int = Form(1),
    door_width_px: int = Form(110),
    robot_radius_px: int = Form(14),
    with_stages: bool = Form(True, description="是否返回中间图（三联对照视图用）"),
) -> dict:
    """户型图 → 栅格矩阵。

    两种输入方式：
      - ``file``：上传图片；
      - ``sample``：使用内置示例图名。

    返回
    ----
    ``grid``      二维 0/1 数组（栅格矩阵，格式定版）
    ``meta``      尺寸与像素↔栅格映射，供前端 Canvas 反算坐标
    ``stats``     诊断统计（门洞数、可通行格数等）
    ``stages``    各中间图 base64（原图/二值/墙体/门洞），供三联对照
    """
    # --- 取图 ---
    img: np.ndarray | None = None
    source = ""
    if file is not None:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"图片超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限")
        if not raw:
            raise HTTPException(400, "上传文件为空")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "无法解码该图片，请换PNG/JPG 格式")
        source = file.filename or "upload"
    elif sample:
        path = FLOORPLAN_DIR / f"{sample}.png"
        if not path.exists():
            raise HTTPException(404, f"示例图不存在：{sample}")
        img = load_image(path)
        source = sample
    else:
        raise HTTPException(400, "请提供 file 或 sample 之一")

    bin_params = BinarizeParams(
        block_size=block_size,
        c_offset=c_offset,
        close_ksize=close_ksize,
        denoise_ksize=denoise_ksize,
        invert=invert,
    )
    ras_params = RasterizeParams(
        cell_px=cell_px,
        wall_dilate=wall_dilate,
        door_width_px=door_width_px,
        robot_radius_px=robot_radius_px,
    )

    try:
        res = rasterize(img, bin_params, ras_params)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    out = {
        "source": source,
        "grid": res.grid_list(),
        "meta": res.meta.to_dict(),
        "stats": res.stats,
        "params": res.params,
        "degraded": res.degraded,
        "note": res.note,
        "cell_names": {str(k): v for k, v in GridCell.NAMES.items()},
    }

    if with_stages:
        # 三联对照视图（任务 1-10）：原图 / 二值图 / 障碍掩码
        out["stages"] = {
            "binary": _b64(res.stages["binary"]),
            "obstacle": _b64(res.stages["obstacle"]),
            "doors": _b64(res.stages["doors"]),
        }

    return out


@router.post("/preview")
async def preview_endpoint(
    file: UploadFile = File(...),
    block_size: int = Form(31),
    c_offset: int = Form(8),
    invert: bool = Form(False),
) -> dict:
    """仅跑二值化并返回中间图 —— 供参数面板实时预览调参效果。"""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "上传文件为空")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解码该图片")

    res = binarize(img, BinarizeParams(
        block_size=block_size, c_offset=c_offset, invert=invert))
    return {
        "stages": {"binary": _b64(res.stages["binary"])},
        "degraded": res.degraded,
        "note": res.note,
        "ink_ratio": round(res.ink_ratio, 5),
    }
