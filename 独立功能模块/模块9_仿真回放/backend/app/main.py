"""模块 9「仿真回放」—— 独立服务入口

启动::

    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

冒烟验证::

    curl http://127.0.0.1:8000/api/health
    curl http://127.0.0.1:8000/api/replay/health
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..api.module9 import router as module9_router

app = FastAPI(
    title="模块 9 · 仿真回放",
    description="课程设计选题 15 · 模块 9 独立包（用于单独提交与验收）",
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
    return {"status": "ok", "module": "模块9 回放", "version": app.version}


@app.get("/api/info", tags=["system"])
def info() -> dict:
    return {
        "module": "模块 9 · 仿真回放",
        "role": "基础模块",
        "design": (
            "增量回放帧：每帧只带新增清扫格（add），前端累加上色，"
            "而非每帧重发全图。这样帧体积可控，大栅格也能流畅播放。"
        ),
        "key_fix": (
            "帧体积超限时降帧并如实上报 dropped_cells，"
            "不静默截断——前端据此提示用户哪些格子没着色。"
            "宁可帧少也不能让累积结果缺格、画布出现永远补不上的空白。"
        ),
        "consistency": (
            "末帧 step 必须等于 total_steps。二者曾因路径补齐口径分叉，"
            "导致进度条卡在 88%；现已统一，接口里返回 self_consistent 供核对。"
        ),
        "files": {
            "api/module9.py": "回放帧生成、摘要、健康检查",
        },
        "endpoints": [
            "POST /api/replay/frames  生成增量回放帧",
            "POST /api/replay/summary 回放摘要（报告 3.x 直接引用）",
            "GET  /api/replay/health  健康检查",
        ],
    }


app.include_router(module9_router)