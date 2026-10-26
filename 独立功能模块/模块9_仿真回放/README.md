# 模块 9 · 仿真回放

> 《软件工程理论基础》课程设计 · 选题 15「2D 户型全覆盖清扫仿真」
> 角色：**基础模块** · 独立提交包

---

## 这个包做什么

把仿真结果变成**看得见的行走过程**——轨迹动画本身就是演示主体，
配回放可以反复展示不同策略的对比。

## 核心设计：增量回放帧

每帧**只带新增清扫格**（`add` 字段），前端累加上色，而不是每帧重发全图。

这样帧体积可控，大栅格也能流畅播放。相邻帧的差值天然很小。

## 关键约束一：超限就降帧，如实上报

单帧体积有上限。前端靠累加 `add` 上色，**一旦某帧被截断，
累积结果就会缺格、画布出现永远补不上的空白**。

因此：

1. 先按「轨迹总长 × 幅宽」估一个安全帧数，**宁可帧少也不能截断**
2. 仍超限时如实上报 `dropped_cells` 与 `complete=false`，不静默截断

前端据此提示用户哪些格子没着色，并建议点「全速」看最终结果。
**指标不受影响** —— 少的是动画帧，不是数据。

## 关键约束二：末帧必须等于 total_steps

这是修掉的一个真bug。

`simulate()` 早期先算指标、再对路径做可视化补齐，
导致同一份数据出现两套口径：

- 造回放帧用补齐**后**的路径长度
- 返回的 `total_steps` 字段取补齐**前**的值

后果是前端进度条显示「4377 / 3646 步」并停在 88%。

现在已统一为补齐后的真实步数，实测末帧 step 与 total_steps **差值恒为 0**。
接口同时返回 `metrics` 供交叉核对。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/replay/frames` | 生成增量回放帧 |
| POST | `/api/replay/summary` | 回放摘要（报告 3.x 直接引用） |
| GET | `/api/replay/health` | 健康检查 |

## 目录结构

```
backend/
├── api/
│   ├── module9.py        回放帧生成、摘要、健康检查
│   └── common.py         栅格反序列化公共入口
├── planning/             依赖的仿真与算法模块
├── core/                 GridCell 定义（依赖模块 1 的栅格格式）
└── data/floorplans/      7 张示例图
```

## 跑起来

```bash
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

冒烟验证：

```bash
curl http://127.0.0.1:8000/api/replay/health
curl -X POST http://127.0.0.1:8000/api/replay/frames \
     -H "Content-Type: application/json" \
     -d '{"grid": [[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0]], "frames": 30}'
```

实测输出：`total_steps=17`，末帧 `step=17`，差 0，`complete=true`，`dropped_cells=0`。

## 不包含什么

- 前端播放控件（时间轴、速度选择在完整项目的 `frontend/index.html`）
- 模块 1 图像解析（仅保留 `GridCell` 常量）