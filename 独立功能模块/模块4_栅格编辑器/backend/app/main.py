"""模块 4「栅格地图编辑器」—— 独立服务入口

本文件是为**单独提交 / 验收**而写的精简入口，只挂载模块 4 的路由。
完整项目的入口在 ``backend/app/main.py``。

模块 4 依赖模块 1 产出的栅格格式（``GridCell``），
故本包内附``core/binarize.py`` 与 ``core/rasterize.py``——
它们在这里只提供 ``GridCell`` 常量与栅格构造工具，
不代表模块 1 的功能被重复计入。

启动::

    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

冒烟验证::

    curl http://127.0.0.1:8000/api/health
    curl -X POST http://127.0.0.1:8000/api/editor/grid \
         -H "Content-Type: application/json" \
         -d '{"width":12,"height":12}'
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..api.module4 import router as module4_router

app = FastAPI(
    title="模块 4 · 栅格地图编辑器",
    description="课程设计选题 15 · 模块 4 独立包（用于单独提交与验收）",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@app.get("/api/health", tags=["system"])
def health() -> dict:
    """健康检查。"""
    return {"status": "ok", "module": "模块4 栅格编辑器", "version": app.version}


@app.get("/api/info", tags=["system"])
def info() -> dict:
    """本包包含的内容与边界，供验收方快速了解范围。"""
    return {
        "module": "模块 4 ·栅格地图编辑器",
        "role": "基础模块",
        "files": {
            "api/module4.py": "编辑器接口：新建/校验/连通性诊断/预置地图",
            "planning/grid_ops.py": "连通域标记、BFS 可达性、包围盒",
            "core/rasterize.py": "GridCell 定义（依赖模块 1 的栅格格式）",
            "api/common.py": "栅格反序列化公共入口",
        },
        "why_needed": (
            "模块 1 的图像解析不可能 100% 准确。编辑器让人工修正解析错误，"
            "把「看运气」的项目变成「必可救」的项目。"
        ),
        "endpoints": [
            "POST /api/editor/grid      新建空白栅格",
            "POST /api/editor/validate  校验并给连通性诊断",
            "GET  /api/editor/presets   列出预置地图",
        ],
        "not_included": [
            "模块 1 图像解析（仅保留 GridCell 常量）",
            "模块 3 路径规划",
            "前端页面（frontend/）",
        ],
    }


app.include_router(module4_router)