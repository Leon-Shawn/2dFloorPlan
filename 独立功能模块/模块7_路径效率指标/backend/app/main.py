"""模块 7「路径效率指标」—— 独立服务入口

启动::

    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

冒烟验证::

    curl http://127.0.0.1:8000/api/health
    curl -X POST http://127.0.0.1:8000/api/efficiency/compare \
         -H "Content-Type: application/json" \
         -d '{"grid": [[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0]]}'
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..api.module7 import router as module7_router

app = FastAPI(
    title="模块 7 · 路径效率指标",
    description="课程设计选题 15 · 模块 7 独立包（用于单独提交与验收）",
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
    return {"status": "ok", "module": "模块7 路径效率", "version": app.version}


@app.get("/api/info", tags=["system"])
def info() -> dict:
    return {
        "module": "模块 7 · 路径效率指标",
        "role": "基础模块",
        "metrics": {
            "repeat_rate": "1 − |unique_visited| / |total_steps|，路径冗余程度",
            "area_time_cost": "total_steps × step_time / (|reachable| × cell_area)",
            "dead_zone_ratio": "BFS 不可达部分占可通行区的比例",
        },
        "caliber_decision": (
            "total_steps 取补齐后的真实步数。早期取补齐前的值以压低重复率，"
            "但那等于假设机器人能瞬移，且与前端回放帧数不一致"
            "（进度条会显示「4377 / 3646 步」并停在 88%）。"
            "现在 len(path) == total_steps + 1 == 帧号上限，三者自洽。"
        ),
        "measured": {
            "boustrophedon": "重复率 31.7% ~ 70.1%",
            "block_boustrophedon": "重复率 32.6% ~ 41.1%（正式策略，明显更稳）",
            "random_walk": "重复率 98.5% ~ 99.8%（对照组）",
            "sample_count": "7 张图 × 3 策略",
        },
        "files": {
            "api/module7.py": "效率指标计算与三策略对比",
            "planning/trajectory.py": "repeat_rate / area_time_cost 实现",
        },
        "endpoints": [
            "POST /api/efficiency/compute 单策略效率指标",
            "POST /api/efficiency/compare  三策略横向对比",
        ],
    }


app.include_router(module7_router)