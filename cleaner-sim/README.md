# 2D 户型全覆盖清扫仿真（选题 15）

读入一张真实户型图（内置数据集 **或自己上传的图片**）→ 解析成可通行栅格 → 用 **BCD 单元分解**
规划全覆盖清扫路径 → 机器人在网页上跑完全屋 → 实时给出覆盖率曲线与效率看板，并支持任意步数回放。

**零第三方依赖**：后端只用 Python 标准库（HTTP 服务、PNG 编解码、栅格运算、图像二值化全部自研），
前端只用浏览器原生 Canvas，不需要 `pip install`、不需要构建步骤。

```
python run.py          →   http://127.0.0.1:8017
```

---

## 一、模块对照（选题要求的 5 个功能）

| 选题模块 | 实现位置 | 关键接口 |
|---|---|---|
| **① 户型图转栅格地图**（核心） | `houseexpo.py` + `rasterize.py` + `imageplan.py` | `build_occupancy()` 五级管线；**支持向量轮廓与上传图片两种来源，下游完全共用** |
| **④ 机器人运动仿真** | `simulate.py` | `simulate_motion()` 位姿/朝向/耗时；`MotionParams` 速度与转向代价 |
| **⑤ 覆盖率计算与可视化**（核心） | `coverage.py` + `web/app.js` | `analyse_coverage()` 全屋/可达覆盖率、分区覆盖率、死区归类；Canvas 逐格渲染 + 覆盖率曲线 |
| **⑦ 路径效率指标** | `metrics.py` | `efficiency_metrics()` 步数/路程/转向/重复度/**路径效率（相对理论下限）**/单位面积耗时 |
| **⑨ 仿真过程回放** | `simulate.py` + `web/app.js` | `replay_payload()` 紧凑回放协议；播放/暂停/倍速/进度条/阶段跳转 |
| （引擎）全覆盖路径规划 | `planner.py` | **BCD 单元分解 + 逐单元牛耕式 + 单元合并 + 2-opt 排序 + 死区回退** |

---

## 二、路径算法：BCD 单元分解（本轮重点改造）

### 为什么换

原实现用「分水岭区域划分 + 全局逐行牛耕式」，问题有二：

1. 分水岭按通道半径腐蚀，走廊、门洞附近会把一整间房切碎——38 房户型被切成 80 个碎片，每个碎片都要一次进出代价；
2. 扫描间距取 `2r`，方形足迹下每一遍都白白重叠一行，天然多花约 20% 步数。

### 现在的算法

**Boustrophedon Cellular Decomposition**（Choset & Pignon, *Coverage Path Planning: The
Boustrophedon Cellular Decomposition*, FSR 1998）：用一条扫描线扫过地图，**每当自由空间的连通性
沿扫描线发生变化就切开一个单元**。

关键实现点：

- **单元即矩形，一次遍历得到**：因为单元内每个扫描位置的自由区间结构完全相同，单元就等价于
  `(slab, run_index)`，不需要逐 slab 做连通域标注，整体 O(w·h)。
- **两个扫描轴都试，按实测路径长度选**：先用纯曼哈顿估算给每个候选构造出完整路线长度，
  再只对胜出的那个跑真正的 A\* 连接——既避免了错误的方向选择，又不付出双倍 A\* 代价。
- **单元合并**：相邻单元只要并集在某个轴上保持单调（每行/每列至多一段），就合并。
  单调区域能被一条蛇形路径一次扫完，所以合并**只会省掉一次单元间连接，不会增加行程**。
- **扫描间距 = 2r+1**（精确贴合足迹宽度）：方形足迹在这个间距下完美密铺，相邻两遍既不重叠也不留缝；
  需要保守时可在界面切回 `2r`（重叠一遍）。
- **单元排序 = 开放路径 TSP**：每个单元有首尾两个可选出入口，先用最近邻贪心，再用
  2-opt（段反转 + 朝向翻转）改进；进入每个单元时在两端里挑离当前位置更近的一端。
- **死区回退**保留：主扫后仍脏的格子聚类，就地重新划分局部区域、沿长边做局部牛耕式，
  再把机器人 A\* 接过去。

参考文献与参考实现：
[Choset & Pignon 1998](https://github.com/devanys/Coverage-Path-Planning-The-Boustrophedon-Cellular-Decomposition)
（同仓库附原始论文 PDF）、
[RicheyHuang/BoustrophedonCellularDecompositionPathPlanning](https://github.com/RicheyHuang/BoustrophedonCellularDecompositionPathPlanning)、
[atb033/multi_agent_path_planning](https://github.com/atb033/multi_agent_path_planning)（选题说明中提到的参考仓库）。

### 实测收益（45 张真实户型，10cm 栅格 / 34cm 机器人）

用 `tools/bench_planners.py` 把两个独立改动拆开对比，旧实现在 `planner_watershed.py` 里保留：

| 变体 | 平均 steps/ideal | 总步数 | 相对旧版 | 最低覆盖率 | 平均规划耗时 |
|---|---|---|---|---|---|
| `legacy/overlap`（原实现） | 1.837 | 214,385 | — | 0.9953 | 0.75 s |
| `legacy/exact`（只改间距） | 1.628 | 188,482 | −12.1% | 0.9954 | 0.74 s |
| `bcd/overlap`（只改分解） | 1.797 | 200,413 | −6.5% | 0.9951 | 0.48 s |
| **`bcd/exact`（新默认）** | **1.555** | **172,001** | **−19.8%** | 0.9958 | **0.46 s** |

- **总步数 −19.8%**，平均 `steps/ideal` 从 1.837 降到 1.555（−15.3%）；
- **规划耗时 ×1.65**（0.75s → 0.46s），全量 45 张从 45.8s 降到 32.3s；
- 复杂户型收益最大：38 房户型 `2.34 → 1.58`（−0.76 steps/ideal，步数 10449 → 7042，**−33%**）；
- 没有任何户型明显变差（最大劣化 +0.15 steps/ideal）；
- 覆盖率不降反升（最低可达覆盖率 0.9958，仍高于 0.995 目标）。

> `steps/ideal` 的分母 `ideal = 可清扫格数 / (2r+1)` 是**无障碍牛耕式的下界**：足迹宽 W 格的机器人
> 每走一步最多把已清扫前沿推进 W 格。1.0 是物理下限，剩下的差值来自「改变扫描行时不可避免的横向
> 移动」和「单元间连接」——看板里的「重定位占比」正是这部分（约 38.6%）。

---

## 三、在线上传户型图（本轮新增）

左侧面板「上传户型图」支持**拖拽或点选**，两种格式：

### 1) HouseExpo `.json`

直接在 Python 侧校验并注册（必需字段 `id`/`bbox`/`verts`），随即出现在户型列表里，可立即仿真。

### 2) 户型图片（PNG / JPG / WebP / GIF / BMP）

图片没有比例尺、没有墙线数据，所以走了这样一条管线：

```
浏览器 createImageBitmap 解码 → canvas 降采样(≤560px) → 灰度 buffer(base64)
   → POST /api/parse/preview          （实时预览，不落盘）
   → POST /api/upload/image           （确认后落盘并加入列表）
        └─ [Python] imageplan.parse_floorplan_image()
             1. 阈值      Otsu 自动，或手动拖滑块
             2. 封口      形态学闭运算，补上墙线断口
             3. 找外部    从图像四边泛洪，穿过非墙像素
             4. 取室内    = 非墙 且 非外部   ← 页面空白背景因此消失
             5. 去碎片    丢掉文字、标注箭头、噪点形成的小连通块
             6. 裁切定标   裁到户型外接框，按用户给的真实宽度换算米
        └─ 之后完全复用模块①的降采样 / 机器人建模 / 规划 / 看板
```

**为什么让浏览器解码**：Canvas 原生支持所有常见图片格式，Python 侧因此不需要写 JPEG 解码器；
但**所有几何决策（阈值、封口、内外判定、去碎片、定标）仍然在 Python 里**，模块①的管线没有被绕过。

界面提供实时调参：阈值（自动/手动）、封口半径、户型实际宽度、墙体反色，右侧同步显示解析结果与
「识别面积 / 户型像素 / 每像素实际尺寸 / 墙体占比 / 剔除碎片数」，确认后一键「加入列表并仿真」。

**准确性验证**：把内置户型渲染成带页边距的合成户型图再反向解析，与向量管线对比面积：

```
$ python tools/test_image_pipeline.py --limit 10
floorplan             size  vector m2  image m2    delta  threshold  islands
a24e5d6b82e2     3.72x2.68        8.9       8.7    -1.6%         25        0
79095a00a61f     8.98x9.63       85.9      85.5    -0.4%         25        0
524f0a38058d   15.57x14.83       79.2      78.8    -0.4%         25        0
max area deviation: 1.6%   PASS
```

10 张全部在 **1.6%** 以内（系统性 −0.5%，来自墙带内沿吃掉约 1 像素）。

上传的户型会持久化在 `uploads/`（`index.json` + 原始灰度 `.gray` + 预览 `.png`），
重启后仍在列表里；可随时改参数重新解析，或从列表删除。

---

## 四、快速开始

```bash
cd cleaner-sim
python run.py                      # http://127.0.0.1:8017
python run.py --port 9000
python run.py --data ../HouseExpo/json     # 用解压后的全量 35,126 张户型
```

数据目录自动探测：`--data` → `CLEANER_DATA` → `../HouseExpo/samples/json` → `../HouseExpo/json`。

**验证与基准**

```bash
python -m unittest discover -s tests -v   # 36 个单元测试
python tools/bench_planners.py            # 新旧路径算法对比（45 户型）
python tools/validate_all.py              # 全量跑一遍并输出指标表
python tools/test_image_pipeline.py       # 图片解析 vs 向量管线
python tools/test_upload_api.py           # 上传相关 HTTP 接口端到端
node   tools/verify_replay.mjs 8017       # 浏览器回放与服务端数值一致性
python tools/render_result.py <户型id>     # 把一次清扫渲染成 PNG
```

---

## 五、目录结构

```
cleaner-sim/
├── run.py                      启动脚本
├── cleaner_sim/
│   ├── grid.py                 栅格基元：方形腐蚀/膨胀、连通域、A*、位打包
│   ├── houseexpo.py            数据集载入与索引
│   ├── imageplan.py            ★ 上传图片 → 栅格（阈值/封口/内外判定/去碎片/定标）
│   ├── uploads.py              ★ 上传注册表（内存 + uploads/ 落盘）
│   ├── rasterize.py            ★ 模块①：栅格化 + 机器人建模（来源无关）
│   ├── planner.py              ★ BCD 单元分解 + 牛耕式 + 合并 + 2-opt + 死区回退
│   ├── planner_watershed.py    旧的分水岭规划器（仅用于 A/B 基准对比）
│   ├── simulate.py             ★ 模块④⑨：运动仿真 + 回放协议
│   ├── coverage.py             ★ 模块⑤：覆盖率与死区统计
│   ├── metrics.py              ★ 模块⑦：效率指标
│   ├── pipeline.py             全流程编排 + 结果缓存
│   ├── server.py               标准库 HTTP 服务与 API（含上传）
│   └── png.py                  零依赖 PNG 编码
├── web/                        前端（原生 HTML/CSS/JS + Canvas）
├── uploads/                    用户上传的户型（自动创建）
├── tests/test_cleaner_sim.py   36 个单元测试
├── tools/                      基准、验证与诊断脚本
└── previews/                   渲染结果截图
```

---

## 六、栅格化管线（模块①）

`build_occupancy()` 的五级管线，**向量轮廓与上传图片共用后三级**：

1. **来源**：HouseExpo 的 `verts` 是整户的**单条闭合轮廓多边形**（内墙、门洞都在里面，门洞是轮廓缺口）；
   上传图片则由图三的三～六步得到等价的「室内掩码」。
2. **光栅化**：活动边表 + 奇偶规则扫描线填充，在 4 倍超采样分辨率下画出室内掩码。
3. **降采样**：按「墙体子格占比 ≥ 0.5」判定为墙，细墙不丢、远宽于一格的门洞不会被误封。
   图片来源还会按自身像素间距反推出**真实可达的格边尺寸**并在看板中显示。
4. **连通性分析**：地板连通域数量、最大区域、孤立区域面积。
5. **机器人建模**：地板按机器人**方形足迹**腐蚀得到合法位姿 `center_ok`，取最大连通块为可达区域，
   再膨胀求出可清扫集合。由此把地板严格分成三部分（互斥且完备）：

| 集合 | 含义 |
|---|---|
| `cleanable` | 从起点区域出发能扫到的地板 |
| `unreachable`（不可达区） | 换个区域能扫到，但**通道容不下机器人本体**——需分区多次清扫 |
| `tight`（死角） | **任何**合法位姿的足迹都够不到的地板 |

> 实测：`30b070f2…`（30.8×14.37 m）地板本身连通，但两半之间只有一条细缝，34 cm 机器人过不去，
> 「不可达区」占 49.6%——正是选题点名的**窄通道与门洞通过性**测试项。
> 见 `previews/result_separated.png`。

---

## 七、覆盖率与指标（模块⑤⑦）

| 指标 | 口径 |
|---|---|
| 全屋覆盖率 | 已清扫格 ÷ 全部可通行格（含不可达区域） |
| 可达覆盖率 | 已清扫格 ÷ 可清扫格（默认目标 99.5%） |
| **路径效率** | `ideal / 实际步数`，`ideal = 可清扫格数 / 足迹宽度`，1.0 为物理下限 |
| **重定位占比** | 非清扫移动步数 ÷ 总步数 |
| 足迹重复度 | 每个格子平均被足迹覆盖的次数 − 1（受足迹宽度决定，非算法缺陷） |
| 单位路程面积 | 清扫面积 ÷ 路径长度（m²/m） |
| 单位面积耗时 | 预计耗时 ÷ 清扫面积（s/m²） |
| 死区 | 分「不可达区 / 死角」两类给出面积、包围盒与原因 |

---

## 八、前端与回放协议（模块⑨）

- **上传**：拖拽/点选 → 图片实时调参预览 → 加入列表并直接开跑。
- **户型选择**：缩略图墙（服务端实时栅格化），可按难度筛选、按 ID/房间类型搜索；上传的户型带「上传」角标，可删除。
- **参数**：栅格分辨率、机器人直径、**扫描间距（精确/重叠）**、目标覆盖率、清扫速度。
- **动画**：Canvas 逐格绘制，可切换「清扫热度 / 轨迹 / 不可达区域 / 分区着色」。
- **回放**：播放/暂停/重置、0.5×–8× 倍速、步数进度条、点击阶段跳转。
- **看板**：双环覆盖率、覆盖率-步数曲线、15 项指标卡、区域与死区清单、规划阶段清单、管线各阶段耗时。

**回放协议只有几 KB**：服务端只下发 `grid{wall, cleanable, separated, tight}`（每格 1 bit，位打包）
与 `replay{start, dirs}`（**每步一个方向码字节**，8 种）。浏览器按同一套整数方形足迹规则重新叠印，
因此**动画与服务端数值必然一致**——由自动化测试保证：

```
$ node tools/verify_replay.mjs 8017
rooms   steps  replay-cov   server-cov        diff  verdict
    1     229    1.000000    1.000000   0.00e+0  OK
   38    7041    1.000000    1.000000   0.00e+0  OK
    2     1045   0.998560    0.998600   4.02e-5  OK   ← 服务端把比率四舍五入到 4 位
PASS: browser replay reproduces the server coverage exactly on every case
```

判定以**已清扫格数的整数完全相等**为准，比率另给 1e-4 容差（服务端字段本身四舍五入到 4 位）。
`tools/debug_replay.py` 用纯 Python 复核同一往返，45 张全部逐格一致。

---

## 九、HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单页界面 |
| GET | `/static/<file>` | 前端资源 |
| GET | `/api/health` | 存活 + 数据集/上传目录 |
| GET | `/api/floorplans` | 户型目录（内置 + 上传） |
| GET | `/api/uploads` | 仅上传项及其解析诊断 |
| GET | `/api/thumb?id=&w=` | 户型缩略图（PNG） |
| GET | `/api/run?id=&cell_cm=&robot_cm=&spacing_mode=&target_cov=&max_retreat=&speed_mps=&turn_time_s=` | 跑完整管线 |
| POST | `/api/upload/houseexpo` | `{name, text}` 注册 HouseExpo JSON |
| POST | `/api/upload/image` | `{name, width, height, gray_b64, threshold, invert, seal_px, real_width_m, scale_axis}` |
| POST | `/api/parse/preview` | 同上但不落盘，用于实时预览（返回 PNG） |
| POST | `/api/upload/reparse` | `{id, ...参数}` 改参重解析 |
| POST | `/api/delete` | `{id}` 删除上传 |
| DELETE | `/api/upload/<id>` | 同上 |

`/api/run` 结果按 (户型, 参数) 缓存；上传被修改或删除时相关缓存自动失效。

---

## 十、验证结果

| 验证 | 结果 |
|---|---|
| 单元测试 | **36/36 通过**（栅格基元与暴力实现逐格对齐、A\* 绕障、BCD 单元恰好划分可达集、合并保持单调与划分、窄门洞必须切分区域、覆盖率集合互斥完备、时间轴单调、回放逐格一致、图像解析往返、上传注册与持久化…） |
| 全量户型 | **45/45 规划成功**，平均全屋覆盖率 97.28%，平均 0.72 s，最慢 2.31 s |
| 路径算法基准 | 总步数 **−19.8%**，平均 `steps/ideal` **−15.3%**，规划耗时 **×1.65**，覆盖率不降 |
| 图片解析往返 | 10 张合成户型图，面积偏差 ≤ **1.6%** |
| 上传接口 | `tools/test_upload_api.py` **17/17 通过**（含非法输入拒绝） |
| 回放一致性 | 浏览器重放与服务端**逐格完全一致**（整数计数相等） |

---

## 十一、已知限制与可扩展点

- 覆盖率目标默认 99.5%：剩下部分主要是墙根死角，需「沿墙走一圈」的贴边清扫才能消掉；
  这部分已单独统计为「死角」，不会混进覆盖率。
- 只对机器人位姿连通块中的**最大块**做规划，其余按「不可达区」报告；要做多次任务编排，
  可对每个块各跑一次规划并合并时间轴。
- 机器人运动学是**质点 + 方形足迹**，未建模加减速与轮速约束。
- 上传图片的解析依赖「墙线闭合 + 有页边距」；若平面图被裁到贴边，需手动调大封口半径
  （界面会给出「外轮廓疑似有缺口」提示）。
- 性能瓶颈是纯 Python 的逐格循环；若要用 5cm 栅格或全量 35,126 张，建议把 `grid.py` 的
  腐蚀/膨胀/连通域换成 NumPy 或 C 扩展（接口不变）。

---

## 十二、数据来源与许可

户型数据来自 [HouseExpo](https://github.com/TeaganLi/HouseExpo)（MIT License, Copyright (c) 2019
Tingguang Li；论文 IROS 2020, arXiv:1903.09845）。项目自带 45 张样例位于
`../HouseExpo/samples/json`，完整 35,126 张可用 `../HouseExpo/tools/extract_dataset.ps1`
解压后以 `--data` 指定。使用时请保留上游 `LICENSE` 并按上游 README 引用。

路径算法参考 Choset & Pignon (FSR 1998) 及上述开源实现，均为独立重写，未复制其代码。
