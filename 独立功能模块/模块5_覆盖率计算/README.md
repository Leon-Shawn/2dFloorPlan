# 模块 5 · 覆盖率计算与可视化

> 《软件工程理论基础》课程设计 · 选题 15「2D 户型全覆盖清扫仿真」
> 角色：**核心模块** · 独立提交包

---

## 这个包做什么

覆盖率是题目直接命名的指标，是最终交付物的「结论页」。本模块负责：

1. 可达性 BFS 连通域分析
2. 覆盖率与去重统计
3. 三色热力图数据输出
4. **归因诊断** —— 覆盖率低时告诉你是「地图断了」还是「算法漏了」

## 关键设计：BFS 可达性分析不是补救措施

它是**算法固有环节**。作用有两个：

- 给覆盖率一个数学上可达的上界（100%）
- 覆盖率掉下来时，区分两种完全不同的情况：
  - **地图断了**：该区域 BFS 不可达，属物理死区，补扫也没用 → 回去改模块 1/4
  - **算法漏了**：该区域可达但没被覆盖，属缺陷 → 必须修

没有这一步，覆盖率只是个来路不明的百分数。

## 指标口径

```
coverage = |cleaned ∩ reachable| / |reachable|
```

其中 `reachable = BFS(起点) ∩ passable`。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/coverage/compute` | 覆盖率 + 归因诊断 + 热力图数据 |
| POST | `/api/coverage/verify` | 口径交叉校验 + 路径合规性 |

## 为什么需要 verify 接口

**覆盖率指标本身有盲区。** 路径若误入墙格，
`|visited ∩ reachable|` 这个交集不受影响，覆盖率**照样显示 100%**，
只有重复率会略微上升。项目里真实发生过这类 bug
（模块 3 的穿墙问题，165 处跳跃、42 处连线穿墙，指标全绿）。

所以 `verify` 从 `cleaned` 与 `reachable` 两个掩码重新算一遍，
与 `metrics["coverage"]` 对比，并**独立**统计：

- `path_on_wall_steps` —— 压墙步数
- `path_jump_points` —— 跳跃点数（切比雪夫距离 >1）

几何合规类约束必须独立校验，不能指望覆盖率顺带报警。

## 目录结构

```
backend/
├── api/
│   ├── module5.py        覆盖率计算、归因诊断、独立校验
│   └── common.py         栅格反序列化公共入口
├── planning/
│   ├── trajectory.py     coverage_ratio / swath_mask（距离变换）
│   ├── deadzone.py物理死区 vs 算法漏扫分类
│   ├── grid_ops.py连通域标记与 BFS
│   └── strategies.py     仿真入口
└── core/                 GridCell 定义（依赖模块 1 的栅格格式）
```

## 跑起来

```bash
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

冒烟验证：

```bash
curl -X POST http://127.0.0.1:8000/api/coverage/compute \
     -H "Content-Type: application/json" \
     -d '{"grid": [[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0]]}'
```

实测输出：覆盖率 100.0%，可达 18 格，已扫 18 格，诊断「全部可达区域已覆盖，无遗漏」。

## 不包含什么

- 模块 1 图像解析（仅保留 `GridCell` 常量）
- 路径规划策略实现（仅依赖，不重新实现）
- 前端页面（Canvas 上色在完整项目里）