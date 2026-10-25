# 户型图数据目录

本目录存放户型平面图。

---

## ⭐ 当前用的是合成图（fp_01 ~ fp_05）

`fp_01` ~ `fp_05` 五张基准图由 `backend/scripts/make_synthetic_plans.py`
**确定性生成**，已入库（合计约 90KB）：

| 文件 | 场景 | 验证什么 |
| --- | --- | --- |
| fp_01 | 单房间方形 | 全链路跑通 |
| fp_02 | 三房间 + 2 门洞 | 门洞检测 + 连通性打通 |
| fp_03 | L 形凹形 | BCD 区域分解（模块 3） |
| fp_04 | 家具密集 | 障碍物标记 |
| fp_05 | 含 22px 窄通道 | 窄道不可通行判定 |

**为什么要合成而不是直接用 HouseExpo 真实图**：

1. HouseExpo 图片托管在 Google Drive，国内访问不稳定；
2. 真实图风格不统一（线框/彩色/扫描件混杂），阈值调参会变成玄学；
3. 合成图 **ground truth 已知**（门洞坐标写在代码里），
   IoU 基线可复现，三名成员的结果能完全对齐。

重新生成：

```bat
.venv\Scripts\python.exe -m backend.scripts.make_synthetic_plans
```

>报告里要**如实说明输入支持范围** —— 合成图验证的是管线正确性，
> 接真实图时的表现要单独实验并单列一节。这是加分项，不是减分项。

---

## HouseExpo 真实图（尚未接入）

克隆 GitHub 仓库只会得到 JSON 矢量数据和一个 `map_id_10.txt`，
**下不到任何户型图片**。图片单独放在 Google Drive 上。

克隆 GitHub 仓库只会得到 JSON 矢量数据和一个 `map_id_10.txt`，
**下不到任何户型图片**。图片单独放在 Google Drive 上。

---

## 数据集背景

**HouseExpo**（IROS2020，香港中文大学 Tingguang Li 等人）
- 规模：35,126 张二维户型图，252,550 个房间，平均每套7.14 个房间
- 底座：基于 SUNCG 数据集
- 内容：矢量版JSON（房间顶点坐标 + 房间类别标签，如 bedroom / kitchen / bathroom）
- 许可：仓库含 LICENSE 文件，使用前请自行确认

论文：arXiv:1903.09845

---

## 下载方式

### 方式一：只要5 张图（推荐，最省事）

**PNG 图片全量包（Google Drive）**：
https://drive.google.com/file/d/1gEmTdgZD1pIa8UtaLXz8vm_301zL4L1J/view?usp=sharing

打开后下载，解压后从里面挑 5 张复制到本目录即可。

> ⚠️ Google Drive 在国内可能打不开或很慢。**打不开就用方式三的国内镜像。**

### 方式二：只要矢量 JSON（几 MB，适合自己渲染成图）

```bash
git clone --depth 1 https://github.com/TeaganLi/HouseExpo.git
cd HouseExpo/HouseExpo
tar -xvzf json.tar.gz
```

解压后得到 35,126 个 `.json` 文件，每个文件就是一套户型的矢量描述：

| 字段 | 含义 |
| --- | --- |
| `id` | 户型唯一编号 |
| `room_num` | 房间数量 |
| `bbox` | 整户包围盒 `{"min": (x1,y1), "max": (x2,y2)}` |
| `verts` | 房间顶点坐标列表，单位为**米** |
| `room_category` | 房间类别及其包围盒，例：`"kitchen": (x1,y1,x2,y2)` |

> **这个方案其实更适合我们。** 因为矢量数据里墙体坐标是精确的，
> 我们可以自己用 matplotlib / OpenCV 渲染成任意分辨率的图，
> 不用受PNG 原图的清晰度和风格限制。
> 缺点是需要写一段渲染代码（约 30 行，属于任务组 1 的一部分）。

### 方式三：国内镜像 / 替代源

若Google Drive 无法访问：

1. **搜 HuggingFace 镜像**：搜索 `HouseExpo`，部分数据集被搬运到 HuggingFace Hub，国内可直接下载
2. **搜 Kaggle**：Kaggle 数据集社区可能有搬运版本
3. **仓库镜像**：用国内 GitHub 加速前缀克隆
   ```
   # 例如用 ghproxy 之类的镜像服务
   git clone https://ghproxy.com/https://github.com/TeaganLi/HouseExpo.git
   ```

---

## 选图建议

挑「结构简单」优先。模块 1 的图像解析不可能一次到位，先用简单图跑通管线，再逐步上难的。

| 优先级 | 户型特征 | 用来验证什么 |
| --- | --- | --- |
| 1 | **单房间 / 方形** | 无门洞无遮挡，最容易跑通全链路 |
| 2 | **两三房间 + 简单门洞** | 门洞检测是否有效 |
| 3 | **L 形 / 凹形** | BCD 区域分解能否正确切分 |
| 4 | **家具密集** | 障碍物标记 |
| 5 | **窄通道**（窄于机身） | 应判为不可通行，压测用 |

HouseExpo 平均 7 个房间，所以「单房间」的图比较少，找不到就用房间少、
形状规整的代替。

---

## 文件命名规范

```
fp_01.png     ← 主测试图（最简单）
fp_02.png     ← 带门洞
fp_03.png     ← L 形 / 凹形
fp_04.png     ← 家具密集
fp_05.png     ← 极端户型（窄通道）
```

与任务 1-11 的 ground truth 标注表、IoU 基线表**一一对应**，按序号对齐。

---

## 注意事项

- **输入类型优先线框图**（深色线条 + 浅色填充），这是模块 1 二值化最友好的类型。彩色渲染图和扫描件成功率明显更低。
- 若某些图解析效果差，**在报告里如实说明输入支持范围**——这是加分项不是减分项。模块 4 栅格编辑器就是为此准备的兜底手段。
- 图片分辨率建议控制在 2000×2000 以内，超大图会自动降采样。
- HouseExpo 是**矢量数据**，没有门洞的显式标注——门洞表现为墙体线段的**缺口**，需要靠形态学检测。这正是任务 1-7 的内容。

---

## 参考文献（写报告时按 GB/T 7714 格式引用）

```
[1] LI T, HO D, LI C, et al. HouseExpo: A Large-scale 2D Indoor Layout
    Dataset for Learning-based Algorithms on Mobile Robots[C]//IROS 2020.
```

数据集网址（EB/OL 格式）：
```
[2] HOUSEEXPO. HouseExpo Dataset[EB/OL]. https://github.com/TeaganLi/HouseExpo
```