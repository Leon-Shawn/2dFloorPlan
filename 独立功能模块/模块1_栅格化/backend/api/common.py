"""API 层公共工具 —— 栅格获取与参数校验

模块 1（栅格化）、模块 3（仿真）、模块 4（编辑器）都需要
「拿到一张栅格矩阵」这一步，统一放这里避免三份重复代码。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from ..core.binarize import BinarizeParams
from ..core.rasterize import RasterizeParams, load_image, rasterize

BASE_DIR = Path(__file__).resolve().parents[2]
FLOORPLAN_DIR = BASE_DIR / "data" / "floorplans"

# 上传体积上限，防止误传大图把内存打满
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


async def grid_from_upload(
    file: UploadFile | None = None,
    sample: str | None = None,
    cell_px: int = 10,
    block_size: int = 31,
    c_offset: int = 8,
    close_ksize: int = 3,
    denoise_ksize: int = 3,
    invert: bool = False,
    wall_dilate: int = 1,
    door_width_px: int = 110,
    robot_radius_px: int = 14,
):
    """上传图片或指定示例名 → 栅格化结果。

    返回 ``(RasterizeResult, source_name)``。

    注意：这里刻意**不用** ``Form(...)`` 作为默认值 ——
    本函数被当作普通 Python 函数直接调用（由各端点显式传参），
    若默认值是 Form 对象会一路传到下游把 numpy 弄崩。
    """
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
            raise HTTPException(400, "无法解码该图片，请换 PNG/JPG 格式")
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
        block_size=block_size, c_offset=c_offset, close_ksize=close_ksize,
        denoise_ksize=denoise_ksize, invert=invert,
    )
    ras_params = RasterizeParams(
        cell_px=cell_px, wall_dilate=wall_dilate,
        door_width_px=door_width_px, robot_radius_px=robot_radius_px,
    )
    try:
        res = rasterize(img, bin_params, ras_params)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return res, source


def grid_from_payload(rows: list) -> np.ndarray:
    """校验前端传来的二维 0/1/2 数组 → np.ndarray。

    模块 4 编辑器保存的栅格会走这条路，因此必须严格校验：
    维度不对会直接让下游 BFS / A* 崩掉。
    """
    if not isinstance(rows, list) or not rows:
        raise HTTPException(400, "grid 必须是非空二维数组")
    width = None
    for i, row in enumerate(rows):
        if not isinstance(row, list) or not row:
            raise HTTPException(400, f"grid 第 {i} 行不是非空数组")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise HTTPException(400, f"grid 第 {i} 行宽度 {len(row)} ≠ {width}，栅格必须矩形")
        for v in row:
            if not isinstance(v, int) or isinstance(v, bool) or v not in (0, 1, 2):
                raise HTTPException(400, f"grid 含有非法格值 {v!r}，只允许 0/1/2")
    if len(rows) > 400 or (width or 0) > 400:
        raise HTTPException(400, "栅格过大（上限 400×400），仿真耗时不可控")
    return np.array(rows, dtype=np.uint8)
