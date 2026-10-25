"""模块 5「覆盖率计算与可视化」—— 独立服务入口

启动::

    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

冒烟验证::

    curl http://127.0.0.1:8000/api/health
    curl -X POST http://127.0.0.1:8000/api/coverage/compute \
         -H "Content-Type: application/json" \
         -d '{"grid": [[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0]]}'
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..api.module5 import router as module5_router

app = FastAPI(
    title="模块 5 · 覆盖率计算与可视化",
    description="课程设计选题 15 · 模块 5 独立包（用于单独提交与验收）",
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
    return {"status": "ok", "module": "模块5 覆盖率", "version": app.version}


@app.get("/api/info", tags=["system"])
def info() -> dict:
    return {
        "module": "模块 5 · 覆盖率计算与可视化",
        "role": "核心模块",
        "formula": "coverage = |cleaned ∩ reachable| / |reachable|",
        "key_design": (
            "BFS 可达性分析是算法固有环节，不是补救措施。"
            "它给覆盖率一个数学上可达的上界（100%），"
            "并让覆盖率掉下来时能区分「地图断了」与「算法漏了」。"
        ),
        "why_verify_endpoint": (
            "coverage 只统计掩码交集，路径误入墙格时它照样显示 100%——"
            "指标本身有盲区。故 /api/coverage/verify 从掩码重算一遍做交叉校验，"
            "并独立统计压墙步数与跳跃点数。"
        ),
        "files": {
            "api/module5.py": "覆盖率计算、归因诊断、独立校验",
            "planning/trajectory.py": "coverage_ratio / swath_mask（距离变换）",
            "planning/deadzone.py": "物理死区 vs 算法漏扫分类",
        },
        "endpoints": [
            "POST /api/coverage/compute  覆盖率 + 归因诊断",
            "POST /api/coverage/verify  口径交叉校验 + 路径合规性",
        ],
    }


app.include_router(module5_router)