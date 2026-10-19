# FALH-V2

本目录实现从地图构建、安全区域提取、观察锚点选择，到视觉采集与找物验证的各阶段功能。本 README 记录运行环境、已实现模块的用法及后续任务入口。

研究定义与算法依据见[锚点选择研究记录（含旧版路径约束方案）](../docs/falhv2/path_constrained_anchor_selection_v0_20261013.md)；实施进度和验收要求见[研究路线与交接](../docs/falhv2/research_roadmap_and_handoff.md)。

首次了解当前锚点算法，建议阅读[当前锚点选取图解](../docs/falhv2/anchor_selection_explained_20261019.md)：从安全站位、入镜与遮挡检查讲到动态投票、删冗余和照片选择，配有七张示意图及默认地图的真实运行案例。

## 环境与运行约定

当前使用已有的 **conda 环境 `test`**。运行或开发前，先激活环境并进入仓库根目录：

```bash
conda activate test
cd /home/hri/yutian/Find-Anything-Like-Humans-V2
```

下文所有命令均在仓库根目录执行，使用激活环境中的 `python`。地图编辑器仅依赖 Python 标准库；A1 与锚点规划使用 `test` 环境中的 NumPy、SciPy 和 OpenCV；离线规划不需要连接相机或安装 RealSense SDK。编辑器前端测试另外需要 Node.js。

## 输入与输出管理

所有入口通过 [config.py](config.py) 统一管理默认路径，路径相对 `falhv2` 目录解析，不受启动工作目录影响。

| 目录 / 文件 | 用途与维护方式 |
|---|---|
| `data/map_aabb.json` | 当前地图和物体 AABB，来自编辑器导出；持久输入 |
| `data/robot.json` | 当前机器人的实测尺寸和安全距离；持久输入 |
| `data/camera.d435i.json` | 当前相机内参与人工补全的安装、观察距离；持久输入 |
| `outputs/` | 可删除的生成结果，运行脚本会重建；不作为默认输入来源 |

地图编辑器通过浏览器下载 `map.json`，请将它重命名为 `map_aabb.json` 并放到 `data/map_aabb.json`。相机读取脚本生成 `outputs/camera.d435i.json` 草稿，补全人工项后自行复制到 data；复制完成后草稿可删除。规划器始终读取 data 中的正式输入，不会读取草稿或自动选择其他地图。配置缺失时明确报错；也可使用 `--scene`、`--robot`、`--camera` 显式指定其他文件。

地图、机器人和相机 JSON 已集中到 `data/`，内容保持不变。以后更换输入文件，只需修改 `config.py` 中的 `MAP_PATH`、`ROBOT_PATH`、`CAMERA_PATH`；也可临时通过命令行覆盖。旧 outputs 中的地图不参与默认输入选择。

## 任务与进度

| 阶段 | 内容 | 状态与入口 |
|---|---|---|
| 地图编辑器 | 绘制栅格地图、编辑无类别物体 AABB、显示高度、导入导出 JSON | 已完成，见[地图编辑器](#地图编辑器) |
| A1 安全区域提取 | 根据地图、机器人尺寸和安全距离求全部安全连通区域 | 已实现 EDT + OpenCV 连通域，见[A1](#a1安全区域提取) |
| A2 观察模型 | 物体驱动反向查询，AABB 部分遮挡、入镜和像素门槛 | 已接入 `main`，见[锚点规划](#a2a3物体驱动锚点规划) |
| A3 锚点选择 | 合格覆盖优先、质量平票择优、冗余删除与拍摄方向选择 | 已接入 `anchor_selector.py`，待现场验证 |
| A4 离线验证 | 精确解核对、基线比较、质量与成本评估 | 已完成首轮八场景实验，见[报告](../docs/falhv2/anchor_driver_experiment_20261014.md) |
| B 静态真实实验 | 按锚点采集、检查实际图像质量、检索与找物 | 待实现 |
| C 动态维护 | 物体变化后更新分布、复核锚点、刷新记忆 | 待实现 |
| D 长期评估 | 多轮变化、消融实验与失败分析 | 待实现 |

## 目录结构

```text
falhv2/
├── README.md
├── map_editor.html                  # 地图编辑器页面
├── map_editor.py                # 地图编辑器启动入口
├── regions_extractor.py              # A1：安全格中心与四连通区域
├── camera_parameters.py         # D435i 相机标定导出
├── anchor_selector.py            # A2/A3：物体驱动锚点与拍摄动作
├── observation_model.py          # 实心 AABB 投影采样、遮挡比例与质量门槛
├── anchor_selector.html           # 离线交互展示模板
├── config.py                    # 地图、机器人、相机及输出文件路径
├── data/
│   ├── map_aabb.json            # 当前默认地图和物体 AABB
│   ├── map_point.json           # 保留的旧点地图，需用 --scene 显式选择
│   ├── robot.json               # 当前机器人尺寸与安全距离
│   └── camera.d435i.json        # 当前 D435i 的默认 RGB 内参
├── outputs/                     # 本地产物，Git 忽略
│   ├── safe_regions.json        # A1 安全区域结果
│   ├── safe_regions.svg         # A1 安全区域诊断图
│   ├── anchor_plan.json         # 锚点、拍摄方向与覆盖缺口
│   ├── anchor_plan.html         # 可直接打开的离线交互页面
│   └── anchor_plan.svg          # 静态诊断图
└── tests/                       # 编辑器与规划器测试
```

## 地图编辑器

启动本地服务：

```bash
python falhv2/map_editor.py
```

浏览器打开 `http://127.0.0.1:8765`，可通过 `--port` 指定其他端口。

1. 设置地图宽高和分辨率，新建空白地图。
2. 拖动画障碍或未知区域，用擦除工具恢复空闲格。
3. 设置中心高度 z、X/Y 尺寸和竖直高度，选择“放置物体”并点击地图。XY 点击位置是 Box 中心。
4. 选择已有物体后可修改中心高度和三轴尺寸，也可撤销、删除。旧点物体选中后点击“应用尺寸到所选物体”补全 AABB。
5. 保存地图为 JSON，将导出的 `map.json` 重命名为 `map_aabb.json` 后放入 `falhv2/data/`；打开已有地图时也从该目录选择。

大地图可使用 **1–64 格宽的方形笔刷**连续涂抹，界面同时显示对应的米制宽度。也可选择**矩形填充**，从一个角拖到另一个角，预览后松开鼠标完成整块填充。两种方式都适用于障碍、未知区域和擦除，每次拖动均可一次撤销。例如 0.05 米/格时，10 格笔刷宽度为 0.5 米。

地图支持鼠标滚轮缩放（25%–3200%），缩放以鼠标所在位置为中心。按住中键或右键拖动可平移，左键继续使用当前绘制工具；点击“显示全图”恢复初始视图。缩放和平移只改变显示，不修改栅格分辨率或物体坐标。

导出内容为 `name`、`grid` 和 `objects`。旧场景文件可以打开，但只保留这些地图数据。

### AABB 与高度显示

主画布保持俯视：矩形表示 XY 范围，标签显示“底部～顶部”高度；浅黄到深橙对应顶部由低到高，图例标出自动刻度。选中物体后显示高度侧视条，带地面线和米制刻度，直观看到离地高度与竖直厚度。侧视条宽度仅为示意，所有物体共用高度尺度。

导出格式示例（世界坐标，单位米）：

```json
{
  "id": "o1",
  "position": [2.0, 3.0, 0.85],
  "aabb": {
    "min": [1.8, 2.85, 0.6],
    "max": [2.2, 3.15, 1.1]
  }
}
```

`position` 必须是 min/max 的中心，三轴尺寸均为正，底部不低于地面。Box 与世界坐标轴对齐，不带旋转。编辑器保留旧点地图且不自动假定物体尺寸；混合地图可以编辑，但运行 AABB 规划前需为全部物体补齐 Box。Box 用于视觉遮挡，不自动修改导航栅格；阻挡底盘的区域仍需绘制为障碍。

## A1：安全区域提取

运行示例：

```bash
python -B falhv2/regions_extractor.py
```

默认读取 `data/map_aabb.json` 和 `data/robot.json`，缺失时报错。需要使用其他地图时通过 `--scene` 显式指定。`outputs/` 和备份目录 `outputs_/` 不纳入版本控制；删除 outputs 不影响正式输入。

也可显式指定输入和输出：

```bash
python -B falhv2/regions_extractor.py \
  --scene falhv2/data/map_aabb.json \
  --robot falhv2/data/robot.json \
  --output falhv2/outputs/safe_regions.json \
  --svg falhv2/outputs/safe_regions.svg
```

**输入：** 栅格地图，以及机器人的足迹长、宽和安全距离。A1 无需起点、终点、物体分布或相机参数；读取编辑器导出的场景时，只使用其中的 `grid`。

机器人配置示例：

```json
{
  "robot": {
    "footprint_width_m": 0.28,
    "footprint_length_m": 0.28,
    "safety_margin_m": 0.0
  }
}
```

这些值仅供示例使用，实际应用时应填写机器人的实测尺寸与所需安全距离。

**方法：** 用矩形足迹的外接圆加安全距离做保守碰撞检查，使用半栅格 EDT 提取安全格中心，再用 OpenCV 四连通标记提取区域。障碍、未知格和地图外部均限制净空；白色空闲格还须满足机器人尺寸条件。模型假设底盘可以原地转向。

**输出：** 每个区域的安全格集合、地图与机器人配置、区域数量及安全格数量，另附 SVG 诊断图。没有任何安全格时，命令以非零状态退出并说明原因。

A1 不再生成全覆盖扫描路线，区域中的每个安全格中心直接成为 A2 候选站位。没有起点输入，因此保留所有安全连通区域，不表示机器人从当前位置可以到达每个区域。选站后的避障路径、访问顺序和区域转移留待执行层实现；物体视觉覆盖由 A2/A3 判断。

## 读取 D435i 相机参数

使用 conda `test` 环境中的 RealSense Python SDK `pyrealsense2` 直接读取标定。当前已安装 `2.59.1.11053`；新环境可运行 `python -m pip install pyrealsense2`。在仓库根目录执行：

```bash
python -B falhv2/camera_parameters.py
```

默认读取 RGB **1280×720、BGR8、30 FPS** 对应的内参，保存到本地 `falhv2/outputs/camera.d435i.json`。脚本查询设备标定，不启动视频采集；后续实际图像应使用相同的分辨率及几何处理。

[`data/camera.d435i.json`](data/camera.d435i.json) 使用与读取结果完全相同的精简结构，作为后续 A2 的默认输入。当前已复制一份真机内参，并已填写安装位置 `[0.23, 0, 0.5]` m、零安装角及 0.2–3 m 观察距离；实际采集效果待验证。新用户读取的文件仍用 `null` 标记人工项。交给其他用户时，按以下步骤准备自己的配置：

1. 连接相机，运行脚本生成 `outputs/camera.d435i.json`。
2. 人工填写该文件中 `mount` 与 `observation_quality` 的所有 `null`，含义见下表。
3. 将补全后的文件自行复制到 `data/camera.d435i.json`，供离线规划使用。重复读取只更新指定输出文件，脚本不自动替换默认配置。

| 字段 | 来源或填写方式 |
|---|---|
| `camera.width/height`、`fx/fy/cx/cy`、畸变模型与系数 | Python SDK 自动读取，匹配实际使用的图像分辨率 |
| `mount.position_in_base_m` | **人工填写** `[x, y, z]`，单位米；RGB 光心相对底盘原点的位置，底盘 `+x` 向前、`+y` 向左、`+z` 向上，`z` 为相对底盘原点的高度偏移 |
| `mount.roll_deg` | **人工填写** 横滚角，单位度；绕相机安装参考坐标的前向轴按右手规则为正 |
| `mount.pitch_deg` | **人工填写** 俯仰角，单位度；向下为正，水平为 0 |
| `mount.yaw_deg` | **人工填写** 相对底盘前方的偏航角，单位度；向左为正，正前方为 0 |
| `observation_quality.min_distance_m/max_distance_m` | **人工填写** 合格观察的最近/最远物体中心距离，单位米，依据拍摄效果确定，不直接等同于深度测量量程 |

安装角采用前、左、上参考轴，零角表示镜头水平朝向底盘前方；旋转约定为 `Rz(yaw) Ry(pitch) Rx(roll)`，再转换到相机光学坐标（右、下、前）。所有 `null` 都表示尚未填写，不默认解释为零。底盘原点应与 A1 站位参考点一致；首版底盘原点按地面上的 `(x,y,0)` 表示。

支持通过 `--width`、`--height`、`--fps`、`--format` 指定实际拍摄模式，`--serial` 选择设备；列出支持的 RGB 模式：

```bash
python -B falhv2/camera_parameters.py --list-modes
```

在 `falhv2` 目录内运行时省略命令中的 `falhv2/` 即可。深度流可通过 `--stream depth --width 640 --height 480 --output falhv2/outputs/camera.depth.json` 另存；后续 RGB 观察模型使用 RGB 配置。

裁切、缩放或去畸变后，应使用对应的新内参。畸变模型与系数保持 SDK 返回值。标定含义参考 [RealSense SDK API](https://github.com/realsenseai/librealsense/blob/master/examples/sensor-control/api_how_to.h)及[投影说明](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)。

## A2/A3：物体驱动锚点规划

在仓库根目录运行下一步入口：

```bash
python -B falhv2/anchor_selector.py
```

默认输入与 A1 一致：读取 `data/map_aabb.json`、`data/robot.json`，相机读取 `data/camera.d435i.json`。脚本内部调用 A1 重新计算安全区域并直接生成站位，不读取已有规划输出，因此可独立运行。无需机器人起始位姿、起终点或指定锚点数量。

也可显式指定编辑器保存的地图和配置：

```bash
python -B falhv2/anchor_selector.py \
  --scene falhv2/data/map_aabb.json \
  --robot falhv2/data/robot.json \
  --camera falhv2/data/camera.d435i.json \
  --output falhv2/outputs/anchor_plan.json \
  --save-supports
```

在 `falhv2` 目录内可以直接运行 `python -B anchor_selector.py`。`--yaw-step-deg` 设置底盘拍摄方向间隔，默认 30°，即每站检查 12 个方向。相机安装高度和俯仰固定，底盘可原地转向。

### 处理流程

1. A1 提取安全连通区域，每个安全格中心生成一个候选站位；按区域 ID、行、列固定排序。
2. 对每个物体用 KDTree 查询附近安全站位，保守查询半径为最远观察距离加相机水平安装偏移。
3. 完整 AABB 地图逐方向投影 Box，采样投影区域，通过射线最近交点估计部分遮挡；距离、入镜比例、未遮挡比例和可见像素均达标才设置 `q(i,θ,o)=1`。全点地图沿用旧视场/距离判断并明确标记 `legacy_point`。
4. 每个未覆盖物体对每站最多投一票，优先新增合格覆盖数，平票比较最佳方向质量之和；再删除冗余站位、选择和精简照片。单个物体的多个低质量方向不累加成合格覆盖。
5. 重新检查每个最终拍摄动作，保存逐物体观察指标、遮挡物 ID、拒绝原因及可视化。

### 部分遮挡模型与门槛

所有输入 Box 都作为实心遮挡体，也都属于待覆盖物体。仅有二维栅格的墙不参与三维遮挡；AABB 空隙被当作实体，属于偏保守的场景近似。距离仍为相机光心到 Box 中心的三维欧氏距离，因此 KDTree 中心距离粗筛保持有效，不采用“最近表面距离”。

从投影外框上的均匀网格发射射线，只保留与目标相交的采样。目标最近表面之前先碰到其他 Box 的采样记为遮挡；后方 Box 不遮挡，目标背面不重复计入分母。

- `in_frame_fraction`：画面内目标投影面积 / 完整目标投影面积。
- `visible_fraction`：画面内未遮挡目标面积 / 画面内目标投影面积。
- `visible_pixels`：画面内未遮挡面积，单位像素平方。
- `quality`：入镜比例 × 未遮挡比例，仅用于合格候选平票择优；像素面积采用独立门槛。

可在相机 JSON 的 `observation_quality` 中添加下列可选字段；未填写时使用表中初始值，正式结果保存实际参数。这些默认值尚未通过真实识别实验标定。

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `min_in_frame_fraction` | 0.8 | 最低入镜比例，范围 `(0,1]` |
| `min_visible_fraction` | 0.6 | 画面内最低未遮挡比例，范围 `(0,1]` |
| `min_visible_pixels` | 400 | 最小可见像素面积，须为正 |
| `sample_grid` | 12 | 投影外框每轴初始采样数，范围 4–128 |
| `refinement_grid` | 36 | 接近门槛时每轴加密采样数，不低于初始值 |

比例距离门槛不超过 0.08、或像素面积距离门槛不超过门槛的 20% 时重新加密计算。分批处理并用相机—目标线段包围范围粗筛遮挡体。网格采样是近似，可能遗漏很细的遮挡物；提高采样密度可检查收敛，但不提供解析误差保证。相机位于任一实心 Box 内，或目标跨越相机平面时，保守拒绝并记录原因；后者尚未实现近裁剪面几何裁切。

优先级为新增合格物体数、质量之和、站位/方向编号。站位中每个物体只取最佳合格方向的质量；冗余删除优先尝试低质量项，仍以保持合格覆盖为条件，不承诺全局最优或多视角表面互补覆盖。

### 输出与查看

| 文件 | 内容 |
|---|---|
| `outputs/anchor_plan.json` | 站位编号、区域、格坐标、底盘偏航（弧度）、每张照片覆盖的物体 ID、逐物体比例与遮挡原因、最佳已检查候选诊断、缺口、配置与耗时 |
| `outputs/anchor_plan.html` | 自包含离线交互页面；可缩放、平移、点选物体和站位、查看 Box 投影外框与观察指标、显示安全区域与候选票数，并导出当前地图 PNG |
| `outputs/anchor_plan.svg` | 静态地图、候选票数、锚点与拍摄方向 |
| `outputs/anchor_plan.supports.json` | 加 `--save-supports` 时导出 A2 完整稀疏二值关系；方向键是 `yaws_rad` 的索引，未列出的项为 0；`qualities` 保存合格方向的逐物体质量 |

修改 `--output` 后，其余文件使用相同文件名主干。直接双击生成的 HTML 即可离线查看或转发，无需启动服务；`anchor_selector.html` 是模板，查看结果应打开 outputs 中的页面。

全部输入目标进入统计：全部有支持返回 `complete`；有缺口返回 `partial` 并列出原始物体 ID；空目标返回 `empty_targets`、零锚点和 `null` 覆盖率。缺口指当前离散站位与方向无支持，未证明连续空间无解。无可通行格、未补全相机参数或非零畸变系数会明确报错。

历史实验地图在当时配置下得到 **3 个站位、10 张定向照片，28/28 点覆盖**；该快照保留在实验分支，主线直接使用编辑器保存的地图。选择采用启发式，脚本不承诺全局最少站位或照片。该历史结果来自点模型。当前 AABB 模型评估采样遮挡和投影面积，仍未验证实际识别；二维导航障碍不直接作为相机遮挡。锚点按站位编号排列，仅用于稳定展示，不代表访问顺序。当前包含所有安全连通区域，尚未根据起点筛选，也未规划站位之间的行驶路线。

### 从扫描路线版迁移

2026-10-19 起移除前置全覆盖扫描，`path_planner.py` 替换为 `regions_extractor.py`。原路线覆盖全部安全格中心，所以删除扫描不改变候选位置集合；站位编号改为区域、行、列顺序，平票时选出的具体位置可能变化。

删除扫描时使用 `schema_version: 2`；此次 AABB 升级为 `schema_version: 3`，新增观察模型、实际门槛、质量与指标。移除 `first_route_index`、`route_indices`、`region_transfer` 和路线长度统计，新增 `navigation: not_planned`、未指定起点的可达性说明及 `station_order`。预处理统计改为 `safe_regions_and_station_seconds`、`safe_station_count`。`plan()` 的第二个返回值为安全区域结果。旧路线文件不再生成或读取；查看时请重新运行入口，使用新导出的页面。

本次用当前地图对照改造前后：3 个区域、8,842 个候选位置以及按格坐标对齐的完整方向覆盖关系一致，均得到 3 个站位、10 张照片、28/28 点覆盖。该结果仍属于几何预计覆盖。

### 选型与后续

`main` 仅保留物体驱动实现，不依赖 experiments。位置驱动和多方法比较可在 `experiment/anchor-candidate-drivers` 分支继续复现。历史[实验报告](../docs/falhv2/anchor_driver_experiment_20261014.md)及数据汇总保留为选型依据。下一步是制作完整 AABB 地图、检查遮挡采样精度、现场采集验证与门槛标定，见[交接清单](../docs/falhv2/research_roadmap_and_handoff.md)。

## 数据约定

- 长度单位为米；地图原点在左下角，`+x` 向右、`+y` 向上，JSON 栅格行从上到下保存。
- `.` 表示空闲、`#` 表示障碍、`?` 表示未知；A1 将障碍和未知格均视为不可通行。
- 编辑器宽高支持 4–512 格；例如 300×300 格、0.05 米/格对应 15×15 米。大地图自动缩放显示，格子较小时隐藏网格线；导出分辨率和格子数量保持设置值。
- 每个物体保存稳定的 `id`、中心 `position: [x, y, z]` 与 `aabb.min/max`。全点旧图兼容；混合点/Box 地图拒绝规划，防止默默遗漏遮挡体。
- 二维障碍仅表示导航占据，不包含高度或三维相机遮挡几何。
- 当前安全区域由满足足迹净空的格中心表示；连通性与物体视觉覆盖分别评估。

## 测试

在 `test` 环境、仓库根目录运行：

```bash
python -B -m unittest discover -s falhv2/tests -v
node falhv2/tests/test_editor_ui.cjs
node falhv2/tests/test_anchor_viewer.cjs
```

Python 测试覆盖页面服务、安全净空与连通域、完整站位集合、相机几何、反向筛选与穷举一致性、投票质量择优、全/半遮挡、投影面积、覆盖缺口和正式 CLI 导出；Node.js 测试覆盖编辑器与锚点查看页面交互。仅运行锚点测试可使用 `python -B -m unittest discover -s falhv2/tests -p 'test_anchor*.py' -v`。

## 后续模块维护

后续子任务沿用 `test` 环境。实现新模块时，在本 README 补充其运行命令、输入输出与依赖要求，并同步更新任务状态。详细研究设计和验收记录维护在上方链接的研究文档中。
