# 模块 1 · 户型图转栅格地图

> 《软件工程理论基础》课程设计 · 选题 15「2D 户型全覆盖清扫仿真」
> 角色：**核心模块** · 独立提交包

---

## 这个包做什么

把一张户型图（PNG/JPG）转成可计算的栅格地图。这是整条链的**唯一输入端**——
没有它，后面的路径规划、覆盖率、回放全是空的。

它也是全题**现成工具最不覆盖**的环节：ROS nav2 / turtlebot 都要求
「已知地图」，从真实户型图反推语义栅格没有成熟方案。

## 目录结构

```
backend/
├── core/
│   ├── binarize.py        225 行  二值化与形态学处理
│   └── rasterize.py       699 行  栅格矩阵构建、门洞打通、GridCell 定义
├── api/
│   ├── module1.py         180 行  HTTP 接口与参数面板
│   └── common.py          103 行  图像/示例图加载的公共入口
├── scripts/
│   └── make_synthetic_plans.py    合成基准户型图生成器
├── tests/
│   └── test_rasterize.py42 条断言
├── app/main.py                     独立服务入口
└── data/floorplans/                7 张内置示例图
    ├── fp_01 ~ fp_05.png           合成基准图
    └── real_fp_01 / real_fp_02.png 真实户型图（HouseExpo数据集）
```

## 栅格格式（已定版，全项目统一）

`GridCell` 定义在 `core/rasterize.py`：

| 值 | 含义 |
| --- | --- |
| `0` | 不可通行（墙体、家具） |
| `1` | 可通行 |
| `2` | 已清扫（**清扫标记，不是家具**） |

> 注意：`2` 常被误读为「家具」。它表示机器人已经走过并清扫过的格子。
> 排查可视化问题时务必先确认这一点，否则会误判成「路线穿墙」。

## 跑起来

```bash
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

冒烟验证：

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/samples                # 列出 7 张示例图
curl -X POST http://127.0.0.1:8000/api/rasterize -F "sample=fp_02"
```

实测输出：`栅格 100 × 75，门洞 2 处`。

## 跑测试

```bash
python -m backend.tests.test_rasterize
# 通过 42/42
```

## 关键参数

栅格化参数全部可通过HTTP 表单调整，无需改代码：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `cell_px` | 10 | 栅格边长（像素），决定分辨率 |
| `block_size` | 31 | 形态学闭运算核，用于补墙体断点 |
| `door_width_px` | 110 | 门洞宽度阈值 |
| `robot_radius_px` | 14 | 机身半径，窄通道据此膨胀 |
| `invert` | false | 黑白反转（浅底深线图需要） |

## 已知限制（如实声明）

只保证对「**深色线条 + 浅色填充**」的 CAD 导出/扫描线框图有效。
彩色渲染图、照片等风格可能导致二值化失效——这是任务书2.3
要求如实说明的输入限制，也是模块 4 编辑器存在的原因。

## 不包含什么

- 模块 3 路径规划（`planning/strategies.py`、`bcd.py`）
- 模块 4 栅格编辑器（`api/module4.py`）
- 前端页面（`frontend/`）

完整项目见主仓库，这个包只是抽出来的独立可运行副本。