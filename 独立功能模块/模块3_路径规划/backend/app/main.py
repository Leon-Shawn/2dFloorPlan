"""模块 3「全覆盖路径规划」—— 独立服务入口

本项目**最有分量的设计点**。题面原文点名：
「全覆盖路径规划容易在死角处陷入反复来回，需要设计回退与重新划分区域的机制」。

三套策略
--------
============ ======================================================
策略          说明
============ ======================================================
boustrophedon 基础牛耕式：整体蛇形扫描，简单但重复率高
block_boustro- 分块牛耕式（正式）：BCD 分解为矩形单元，单元内蛇形、
pheddon      单元间 A* 转场，重复率显著更低
random_walk   随机游走（对照）：步数上限 6×可达格数，用于反证
============ ======================================================

三道防震荡防线
--------------
1. **重复访问代价** —— A* 代价函数对已访问格加权，让路径倾向 unexplored 区域
2. **最大重试次数** —— 死区补扫上限 `max_retry`，防止无限循环
3. **滑动窗口震荡检测** —— ``AntiDither`` 识别往返抖动并强制转向

启动::

    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..api.module3 import router as module3_router

app = FastAPI(
    title="模块 3 · 全覆盖路径规划",
    description="课程设计选题 15 · 模块 3 独立包（用于单独提交与验收）",
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
    return {"status": "ok", "module": "模块3 路径规划", "version": app.version}


@app.get("/api/info", tags=["system"])
def info() -> dict:
    return {
        "module": "模块 3 · 全覆盖路径规划",
        "role": "核心模块（本项目最有分量的设计点）",
        "files": {
            "planning/bcd.py": "BCD 单元分解 + 单元内牛耕覆盖",
            "planning/astar.py": "带重复访问代价的 A*",
            "planning/deadzone.py": "死区分类、振荡检测（AntiDither）",
            "planning/strategies.py": "三套策略 + 仿真统一出口",
            "planning/grid_ops.py": "连通域标记与 BFS 可达性",
            "planning/trajectory.py": "清扫带掩码（距离变换实现）",
            "api/module3.py": "HTTP 接口",
        },
        "strategies": [
            {"name": "boustrophedon", "label": "基础牛耕式", "role": "正式策略"},
            {"name": "block_boustrophedon", "label": "分块牛耕式", "role": "正式策略"},
            {"name": "random_walk", "label": "随机游走", "role": "对照组"},
        ],
        "guardrails": [
            "重复访问代价（A* 代价加权）",
            "最大重试次数（死区补扫上限）",
            "滑动窗口震荡检测（AntiDither）",
        ],
        "known_fix": (
            "cover_cell_boustrophedon 的 _connect() 跨行时返回空序列，"
            "注释称由调用方补 A*，但单元内部无调用方补 A*，"
            "导致每换一条扫描线残留跳跃点（real_fp_02 上 165 处，"
            "42 处连线穿墙）。修复分三层：_close_jumps 单元内补齐 → "
            "_bfs_within BFS 兜底 → _densify_path 仿真出口统一兜底。"
        ),
        "endpoints": [
            "GET  /api/strategies    策略清单",
            "POST /api/simulate      图片/示例图 → 仿真",
            "POST /api/simulate-grid 栅格直传 → 仿真",
            "POST /api/compare       三策略对比",
            "POST /api/progress      覆盖率曲线",
        ],
    }


app.include_router(module3_router)