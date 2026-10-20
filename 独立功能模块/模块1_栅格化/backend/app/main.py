"""模块 1「户型图转栅格地图」—— 独立服务入口

本文件是为**单独提交/ 验收**而写的精简入口，只挂载模块 1 的路由。
完整项目的入口在``backend/app/main.py``，那里同时挂载了模块 3/4/9/AI。

启动::

    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

冒烟验证::

    curl http://127.0.0.1:8000/api/health
    curl http://127.0.0.1:8000/api/samples
    curl -X POST http://127.0.0.1:8000/api/rasterize -F "sample=fp_02"
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..api.module1 import router as module1_router

app = FastAPI(
    title="模块 1 · 户型图转栅格地图",
    description="课程设计选题 15 · 模块 1 独立包（用于单独提交与验收）",
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
    return {"status": "ok", "module": "模块1 栅格化", "version": app.version}


@app.get("/api/info", tags=["system"])
def info() -> dict:
    """本包包含的内容与边界，供验收方快速了解范围。"""
    return {
        "module": "模块 1 · 户型图转栅格地图",
        "role": "核心模块",
        "files": {
            "core/binarize.py": "二值化与形态学处理",
            "core/rasterize.py": "栅格矩阵构建、门洞打通、GridCell 定义",
            "api/module1.py": "HTTP 接口与参数面板",
            "api/common.py": "图像/示例图加载的公共入口",
            "scripts/make_synthetic_plans.py": "合成基准户型图生成器",
        },
        "tests": "backend/tests/test_rasterize.py（42 条断言）",
        "grid_format": "0=不可通行 1=可通行 2=已清扫（GridCell）",
        "not_included": [
            "模块 3 路径规划（planning/strategies.py、bcd.py）",
            "模块 4 栅格编辑器（api/module4.py）",
            "前端页面（frontend/）",
        ],
    }


app.include_router(module1_router)