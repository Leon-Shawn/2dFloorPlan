# HouseExpo 资源接入说明

> 面向《软件工程理论基础》课程设计 **选题 15 · 2D 户型全覆盖清扫仿真**
> 本项目需实现的功能模块：**① 户型图转栅格地图（核心）· ④ 机器人运动仿真 · ⑤ 覆盖率计算与可视化（核心）· ⑦ 路径效率指标 · ⑨ 仿真过程回放**

> **⚠️ 2026-10 修订说明（当前工作区实际状态）**
> 本工作区现已收敛为：`HouseExpo/LICENSE`、`HouseExpo/README.md`、`HouseExpo/samples/`
> （45 张样例 JSON + 索引 CSV），以及 `cleaner-sim/` 项目本体。
> 本文档第二、五、六、七、十节中提到的 `json.tar.gz`、`pseudoslam/`、`experiments/`、
> `map_id_sets/`、`tools/`、`previews/` 等目录**均不在当前工作区**（当时为临时落盘，后续已清理）。
> 相关命令与行号映射按**历史记录**保留，仅供参考；如需完整数据集或上游源码，请按第十节命令
> 自行从 GitHub 拉取。清扫仿真主体（`cleaner-sim/`）不依赖这些文件，
> 仅用 `HouseExpo/samples/json/` 即可完整运行与验证。

---

## 一、来源、许可与版本

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/TeaganLi/HouseExpo |
| 分支 / commit | `master` @ `45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c`（2023-09-26） |
| 许可 | **MIT License**，Copyright (c) 2019 Tingguang Li |
| 论文 | HouseExpo: A Large-scale 2D Indoor Layout Dataset for Learning-based Algorithms on Mobile Robots（IROS 2020，arXiv:1903.09845） |
| 本地形态 | **现仅保留** `LICENSE`、`README.md`、`samples/`（45 张样例）；完整克隆与补充目录（`pseudoslam/`、`map_id_sets/`、`tools/`、`previews/` 等）已移除，见文首修订说明 |

**署名要求**：使用该数据集须保留 `LICENSE`（MIT，允许修改与再分发，需保留版权声明），论文/报告中按上游 README 给出的 BibTeX 引用。切勿删除 `LICENSE` 文件。

数据集本身：**35,126 张二维户型图**（源自 SUNCG），共 252,550 个房间，每户房间数均值 7.14、中位数 7.0。

---

## 二、下载清单（历史记录：当时落盘的内容；现多数已不在工作区）

| 路径 | 大小 | 内容 | 对应模块 |
|---|---|---|---|
| `HouseExpo/json.tar.gz` | 24.4 MB | **完整数据集**：35,126 个户型 JSON（压缩包内为 `json/<id>.json`） | 全部 |
| `pseudoslam/` | ~50 KB | 上游 PseudoSLAM 仿真平台全部源码（栅格化、运动仿真、激光、可视化） | ①②④⑤⑦⑨ |
| `experiments/` | ~1.2 MB | 官方复现配置（含 `map_id_train.txt` / `map_id_test.txt` 划分） | ⑦ 批量统计 |
| `samples/` | 0.13 MB | **本地补充**：45 张已解压的户型 JSON + 元数据索引 CSV（免去解压步骤即可开工） | ①②④⑤⑦⑨ |
| `map_id_sets/` | ~1.5 MB | **本地补充**：来自 `dev` 分支的 map id 清单（100 / 1000 / 10000 / train / test） | ⑦ 批量统计 |
| `tools/` | ~16 KB | **本地补充**：3 个即用脚本（见第七节） | ①⑦ |
| `previews/` | 6 张 PNG | **本地补充**：已渲染的栅格地图预览，可直接核对解析效果 | ① |

> 说明：`samples/` 等补充目录不属于上游仓库，当时以未跟踪文件的形态存在于临时克隆中；
> 当前工作区已无 `.git`，且除 `samples/` 外均已移除。**当前实际可用的只有 `samples/`（45 张）**。

---

## 三、数据格式（模块①的解析依据）

每个户型 JSON 只有 5 个字段：

```jsonc
{
  "id": "a24e5d6b82e244357f688ca1a57b5806",   // 唯一房屋 ID
  "room_num": 6,                               // 房间数
  "bbox": { "min": [0.1, 0.1], "max": [11.3, 15.97] },  // 整户包围盒（单位：米）
  "verts": [[4.18, 0.1], [4.17, 0.11], ...],   // 轮廓多边形顶点，单位：米
  "room_category": { "Kitchen": [[x1,y1,x2,y2], ...], "Room": [...] }  // 各房间类别及其包围盒
}
```

三个关键事实（本次已实测确认）：

1. **`verts` 是整户的单条闭合轮廓多边形**，既包含外墙也包含内墙，**门洞体现为轮廓上的缺口**。因此上游用一次 `cv2.drawContours(..., 255, -1)` 填充即可得到「内墙+门洞」都正确的二值图，不需要额外处理门。
2. **`room_category` 的类别在原始数据里不统一**（如 `Room`、`Dining_Room`、`Toilet`、`Gym` 等），上游 `jsonReader._get_room_tp_id()` 会把 `toilet→bathroom`、`guest_room→bedroom`，其余未知类别统一映射为 `indoor`，最终归并为 8 类位掩码：`indoor/kitchen/dining_room/living_room/bathroom/bedroom/office/hallway`。
3. **`room_category` 的包围盒不能当作房间分割**：它们粗粒度、彼此重叠，且实测存在未被任何包围盒覆盖的可通行区域（见第七节 `rooms_overlay_demo.png` 中的白色区域）。上游也只把它 OR 进 `tp_map` 位掩码供逐像素查询（`get_room_tp()`），一个像素可同时命中多个房间类别。

栅格化约定（**全项目统一，模块⑤⑦都依赖**）：

| 值 | 含义 |
|---|---|
| `255` | 可通行 / 自由空间 |
| `0` | 障碍 / 墙体 |

`meter2pixel` 是米→像素比例（上游 README 用例取 100，即 1 米 = 100 像素；官方 demo 配置取 1）。**该值直接决定栅格粒度：越大越精确但仿真越慢。** 本选题做全覆盖清扫仿真，建议 20–50 px/m 起步。

---

## 四、文件 → 模块映射（核心表）

| 模块 | 上游文件 | 关键接口 / 行号 | 用途 |
|---|---|---|---|
| **① 户型图转栅格地图**（核心） | `pseudoslam/envs/simulator/jsonReader.py` | 类 `jsonReader`；`read_json()` L28–64；`_get_room_tp_id()` L7–15 | JSON → `cnt_map`（0/255 二值栅格）+ `tp_map`（房间类型位掩码）。**这就是模块①的参考实现**，建议直接改写而不是重写 |
| ① | `pseudoslam/viz/vis_maps.py` | `draw_map()` L10–29；CLI L31–54 | 批量把 map id 清单渲染成 PNG，用于人工核对墙体/门洞解析是否正确 |
| ① | `samples/json/*.json`、`samples/sample_index.csv` | — | 45 张已解压样例；CSV 含 `room_num`、`n_verts`、尺寸、面积、房间类别、难度分层，**便于按「结构简单 → 复杂」挑选起步样例** |
| ① | `tools/floorplan_preview.py` | `rasterise()`、`connectivity_stats()` | 零依赖（纯标准库）栅格化 + 连通域统计，可在装好 numpy/cv2 之前先跑通数据链路 |
| **④ 机器人运动仿真** | `pseudoslam/envs/simulator/pseudoSlam.py` | 类 `pseudoSlam`；`moveRobot()` L388–431；`robotCrashed()` L454–467；`_randomizeRobotPose()` L221–233；`reset()` L235–253 | 机器人位姿 `[y, x, θ]`、步进、**按 7 像素/10° 过采样防穿墙**、圆形碰撞检测、随机初始位姿 |
| ④ | `pseudoslam/envs/simulator/util.py` | `transform_coord()` L4、`world2mapCoord()` L61 / `map2worldCoord()` L71（世界↔图像坐标，注意 y 轴翻转）、`make_circle()` L86、`within_bound()` L81 | 运动学与几何工具，模块④可直接复用 |
| ④ | `pseudoslam/envs/config/config_navigation.yaml` | `robotRadius` L14、`stepLength.linear/angular` L15–17、`startPose` L18–21 | 机器人半径、每步线速度/角速度、起始位姿等参数口径 |
| **⑤ 覆盖率计算与可视化**（核心） | `pseudoslam/envs/simulator/pseudoSlam.py` | **`measure_ratio()` L480–484**（`已清扫栅格 / 可通行栅格`，就是覆盖率的定义）；`map_color` L11；`world2state()` L433–452（把位姿与朝向画进状态图） | 覆盖率统计的现成口径 + 渲染 |
| ⑤ | `pseudoslam/envs/simulator/pseudoSlam.py` | `build_map()` L372–384、`_build_map_with_rangeCoordMat()` L309–370 | 激光感知与「已探索地图」构建；`cv2.dilate` 填补孔洞（L369）可借鉴来做覆盖热力的形态学平滑 |
| ⑤ | `pseudoslam/viz/vis_maps.py` | 同上 | 出图写文档用 |
| **⑦ 路径效率指标** | `pseudoslam/envs/simulator/pseudoSlam.py` | `self.traj` 记录 L37 / L424（**仅记录前进动作的离散点**）；`stepLength_linear` 用于把步数换算成路程 | 总步数、总路程的原始数据源 |
| ⑦ | `pseudoslam/envs/config/config_navigation.yaml` | `meter2pixel`、`stepLength` | 把「步」换算为「米/秒」的口径 |
| ⑦ | `map_id_sets/`、`experiments/` | `map_id_100/1000/10000.txt`、`map_id_train/test.txt` | 固定可复现的户型子集，用于多户型批量跑指标并对比 |
| ⑦ | `pseudoslam/viz/map_id_set_generator.py` | L30–41 | 从全量数据集中随机抽样生成 map id 清单（支持指定补集，避免与 train 重叠） |
| **⑨ 仿真过程回放** | `pseudoslam/envs/simulator/pseudoSlam.py` | `self.traj`（同上） | 轨迹序列即回放数据源 |
| ⑨ | `pseudoslam/envs/keyboard_navigation.py` | `moveBindings` L12–15；按键循环 L69–101；**存盘 L89–95** | 交互式仿真的最小原型：按 `s` 把 `traj` 存成 `traj_epi_N.txt` 并把轨迹画到图上存 PNG。用 `np.loadtxt` 读回即可实现加速回放 |
| ⑨ | `pseudoslam/envs/robot_exploration_v0.py` | Gym 环境封装 | 若要把「一次清扫」封装成可批量/可重放的环境，可参考其 `reset()/step()` 结构 |

**上游代码里没有、必须自研的部分**（对应选题说明的「自主开发模块」）：

- **全覆盖路径规划**（牛耕式/BCD、区域划分、死角回退）——上游只提供环境与感知，**不含覆盖规划算法**；
- 覆盖率看板（指标聚合与图表）；
- 回放的进度条 / 倍速 / 单步控制；
- 栅格地图编辑器（模块②，本选题不在范围内）。

---

## 五、必须先解决的环境问题（实测）

本机（`Python 3.13.12`）**未安装** `numpy`、`cv2`、`matplotlib`、`PyYAML`、`gym`，上游代码也**不能直接在 NumPy 1.24+ / PyYAML 6 上运行**。落地前请先修掉以下 5 处：

```bash
pip install numpy opencv-python matplotlib pyyaml
```

| # | 问题 | 位置 | 修法 |
|---|---|---|---|
| 1 | `np.int` 在 NumPy ≥ 1.24 已**被移除**（`AttributeError`） | `jsonReader.py` L40、L55；`vis_maps.py` L17；`pseudoSlam.py` L311、L312、L440 | 全部替换为内置 `int`（上游 6 处） |
| 2 | `yaml.load(stream)` 缺少 `Loader`，PyYAML 6 直接报错 | `pseudoSlam.py` L40 | 改为 `yaml.safe_load(stream)` |
| 3 | `np.lib.pad` 已弃用 | `pseudoSlam.py` L143 | 改为 `np.pad` |
| 4 | 键盘 demo 依赖 Linux 终端（`termios/tty/select`） | `keyboard_navigation.py` L10、L18–22；`keyboard_exploration.py` 同上 | Windows 下改用 `msvcrt`/`pynput`，或干脆做 Web 前端（更符合选题「Canvas/SVG 可视化前端」的方向） |
| 5 | 配置里 `json_dir: ./config/`、`map_id_set: ./config/empty.txt` 只指向空样例 | `config_*.yaml` L2–L3 | 改成 `../../HouseExpo/json/` 与 `../../HouseExpo/map_id_10.txt`（路径基准是 `pseudoslam/envs/`，见 `jsonReader.py` L19 与 `pseudoSlam.py` L46 的拼接方式） |

另需注意：

- `pseudoslam/viz/*.py` 的默认路径（`./HouseExpo/json/`）是**相对当前工作目录**解析的，必须从仓库根目录运行。
- 本环境下 PATH 上的 `tar`（PortableGit 的 MSYS tar）曾出现不可用（`NtCreateDirectoryObject ... 0xC0000022`），上游 README 的 `tar -xvzf json.tar.gz` 可能失败；当时改用 `tools/extract_dataset.ps1`（内部调用 `C:\Windows\System32\tar.exe`）绕过——**该脚本当前不在工作区**，如需复现请自行调用系统自带 `C:\Windows\System32\tar.exe` 解压。
- 仿真循环用 `matplotlib.pause(0.001)` 渲染，只是 demo 级别；选题要求实时覆盖率可视化和回放，建议按选题说明换成浏览器 Canvas 前端。

---

## 六、目录结构

```
HouseExpo/
├── LICENSE                       # MIT，务必保留
├── README.md                     # 上游原始说明（含数据格式与仿真参数文档）
├── setup.py
├── HouseExpo/
│   ├── json.tar.gz               # ★ 完整数据集（35,126 个户型 JSON，24.4 MB）
│   └── map_id_10.txt             # 官方 10 张 demo 户型 id
├── pseudoslam/                   # ★ 上游仿真平台源码
│   ├── envs/
│   │   ├── config/*.yaml         # 仿真参数（机器人/激光/SLAM/栅格粒度）
│   │   ├── keyboard_navigation.py    # 交互式仿真 + 轨迹存盘（模块⑨原型）
│   │   ├── keyboard_exploration.py
│   │   ├── robot_exploration_v0.py   # Gym 环境封装
│   │   └── simulator/
│   │       ├── jsonReader.py     # ★ 模块① 核心：JSON → 栅格地图
│   │       ├── pseudoSlam.py     # ★ 模块④⑤⑦⑨ 核心：运动仿真/覆盖率/轨迹
│   │       └── util.py           # 坐标变换与几何工具
│   └── viz/
│       ├── vis_maps.py           # 户型 → PNG（模块① 校验）
│       └── map_id_set_generator.py
├── experiments/                  # 官方复现配置 + train/test 户型划分
├── samples/                      # 【补充】45 张已解压户型 + sample_index.csv
├── map_id_sets/                  # 【补充】100/1000/10000/train/test 户型清单
├── tools/                        # 【补充】即用脚本
│   ├── extract_dataset.ps1       # 解压数据集（绕开本环境损坏的 tar）
│   ├── curate_samples.py         # 从全量数据集中按房间数分层抽样
│   └── floorplan_preview.py      # 零依赖栅格化预览 + 连通域统计
└── previews/                     # 【补充】6 张已渲染的栅格地图预览
```

---

## 七、快速开始

### 1) 不装任何依赖，先验证数据链路（模块① 冒烟测试）

```bash
cd HouseExpo
python tools/floorplan_preview.py --sample-dir samples/json --limit 3 --outdir previews
```

已实测输出（`--sample-dir samples/json --limit 3`）：

```
01e53c5618fd8d4d78c916b9dcc4ff92.json  rooms=6   verts=99   11.20 m x 15.87 m -> 1170 x 1637 px @100px/m
  walkable cells=1736838  walkable ratio=0.907  walkable regions=1
025f15bb7f929b7694ce63b02edb0387.json  rooms=18  verts=222  10.08 m x 15.43 m -> 1058 x 1593 px
  walkable cells=1276904  walkable ratio=0.758  walkable regions=1
071a63d027b148c6db9a4ce1caa9cd1f.json  rooms=6   verts=78   10.95 m x 11.58 m -> 1145 x 1208 px
  walkable cells=830246   walkable ratio=0.600  walkable regions=1
```

`previews/` 中生成的 PNG 可以直接肉眼确认：外墙/内墙为黑、房间为白、**门洞是墙上的缺口**——说明解析正确。`walkable regions=1` 表示可通行区域通过门洞连通成一片，这正是全覆盖清扫可行的前提；**若某户型该值 > 1，说明存在无法抵达的孤立区域，就是选题要求的「死区」**，模块⑤可以直接用这个指标做死区检测。

`previews/rooms_overlay_demo.png` 是叠加 `room_category` 的版本（房间灰、墙体黑）。注意图中**仍有一块白色区域没有任何房间包围盒覆盖**：HouseExpo 的 `room_category` 只是**粗粒度、可重叠的包围盒，不构成对户型的完整划分**。因此它只适合做「某点属于哪类房间」的查询（上游 `jsonReader.get_room_tp()` 即如此实现，一个像素可同时命中多个类别），**不要**拿它当房间分割结果；若要按房间统计覆盖率，需要自行做区域划分。

### 2) 解压数据集（历史命令；`json.tar.gz` 与脚本均不在当前工作区）

> ⚠️ 以下命令仅作历史参考：需要自行拉取上游 `json.tar.gz` 并自备解压脚本。

```powershell
# 只解压 500 张试跑
powershell -ExecutionPolicy Bypass -File tools\extract_dataset.ps1 -Limit 500
# 或在 PowerShell 中直接调用（本机为 Windows PowerShell 5.1，没有 pwsh）
.\tools\extract_dataset.ps1 -Limit 500
# 全量解压（35,126 张，约需 1 分钟）
.\tools\extract_dataset.ps1
```

| 用法 | 命令 |
|---|---|
| 单张（位置参数） | `python tools/floorplan_preview.py samples/json/01e53c56....json --rooms` |
| 批量：`--sample-dir <dir> --limit N` | 见上 |
| 调栅格粒度：`--m2p 50` | 1 米 = 50 像素 |
| 叠加房间类别：`--rooms` | 按 `room_category` 着色（墙体/门洞保持黑色不变） |
| 指定输出：`--out <file>` | 单张模式下显式指定 PNG 路径 |

### 3) 从全量数据集重新抽样（可选）

```bash
python tools/curate_samples.py --source json --target 38
```

---

## 八、本次未下载的内容及原因

| 内容 | 原因 | 替代方案 |
|---|---|---|
| **35,126 张户型 PNG 图集** | 上游放在 Google Drive，体积达数 GB（仓库里不含） | 用 `pseudoslam/viz/vis_maps.py` 或 `tools/floorplan_preview.py` 从 JSON 就地生成等价 PNG；`previews/` 中已有 6 张样例 |
| PseudoSLAM 训练好的模型权重 | 仓库中本就没有（需自行按 README 用 OpenAI Baselines 训练） | 本选题做全覆盖清扫仿真，**不需要**强化学习模型 |
| 动态障碍（移动行人）数据集 | 上游 README 明确标注「尚未发布」 | 不需要 |
| `dev` 分支的 Python 源码 | 与 `master` 仅细微差异，`master` 为正式发布版 | 已单独取回 `dev` 上独有的 `map_id_*.txt` 抽样清单（见 `map_id_sets/`） |

---

## 九、给五个模块的落地建议

1. **模块①**：以 `jsonReader.read_json()` 为骨架改写。它把 `verts` 直接当作单条轮廓填充，得到的是「整户可通行区域」。注意 `room_category` 的包围盒**粗粒度、互相重叠且不覆盖全屋**（实测见第七节），只能用于「某点房间类型」查询，**不能直接当房间分割**；若教学上需要逐房间统计覆盖率，需自行做区域划分（如基于连通域/分水岭的房间切分），或退一步只报全屋覆盖率。建议把栅格化封装成纯函数并配单元测试：输入 45 张样例，断言可通行面积与连通域数量。
2. **模块④**：`moveRobot()` 的**过采样防穿墙**（按 7 像素/10° 细分步进）和 `robotCrashed()` 的圆形碰撞补丁是可直接移植的两个关键实现；注意机器人半径要用像素、且地图四周需 `padding` 一圈障碍（见 `_map_process()` L127–145）。
3. **模块⑤**：覆盖率口径直接采用 `measure_ratio()`（已清扫 / 可通行）；上游在探索模式下把未知区域标成 `-101`，清扫仿真里建议改为独立的一层「已清扫掩码」，避免与感知地图耦合。**可视化建议按选题说明走 Canvas/SVG 前端**，而不是 `matplotlib.pause()`。
4. **模块⑦**：`self.traj` 只记录前进时的离散点，用它算总路程会**低估**；建议改为记录每步位姿序列，再据此计算总步数、重复覆盖率（同一栅格被清扫的次数分布）、单位面积耗时。
5. **模块⑨**：回放数据就是位姿/轨迹序列，`keyboard_navigation.py` L89–95 演示了「轨迹 + 地图 → 存档」的最小闭环；把该序列按帧写入 JSON，前端即可做进度条与倍速回放。
6. **测试口径**（对应选题的「测试重点」）：用 `samples/sample_index.csv` 按 `room_num` 分层挑选窄通道/凹形房间用例；`floorplan_preview.py` 已能直接给出「可通行区域连通域数」，可作为死区回归断言的量化基准。

---

## 十、复现本次下载所用的命令

本环境直连 GitHub 会被沙箱的 TLS 限制拦住（`schannel: SEC_E_NO_CREDENTIALS`），需走本地代理并改用 OpenSSL 后端：

```bash
git -c http.sslBackend=openssl -c http.sslVerify=false \
    -c http.proxy=http://127.0.0.1:62906 \
    clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/TeaganLi/HouseExpo.git HouseExpo
# 再 sparse-checkout（如需省流量可先排除 HouseExpo/json.tar.gz）
```

`map_id_sets/` 取自 `dev` 分支的 `raw.githubusercontent.com/TeaganLi/HouseExpo/dev/HouseExpo/`。
