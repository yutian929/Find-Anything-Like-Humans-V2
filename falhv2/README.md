# FALH-V2

本目录实现从地图构建、覆盖路线规划、观察锚点选择，到视觉采集与找物验证的各阶段功能。本 README 记录运行环境、已实现模块的用法及后续任务入口。

研究定义与算法依据见[路径约束下的锚点选择](../docs/falhv2/path_constrained_anchor_selection_v0_20261013.md)；实施进度和验收要求见[研究路线与交接](../docs/falhv2/research_roadmap_and_handoff.md)。

## 环境与运行约定

当前使用已有的 **conda 环境 `test`**。运行或开发前，先激活环境并进入仓库根目录：

```bash
conda activate test
cd /home/hri/yutian/Find-Anything-Like-Humans-V2
```

下文所有命令均在仓库根目录执行，使用激活环境中的 `python`。地图编辑器仅依赖 Python 标准库；A1 使用 `test` 环境中的 NumPy、SciPy 和 OpenCV。编辑器前端测试另外需要 Node.js。

## 任务与进度

| 阶段 | 内容 | 状态与入口 |
|---|---|---|
| 地图编辑器 | 绘制栅格地图、放置无类别物体中心点、导入导出 JSON | 已完成，见[地图编辑器](#地图编辑器) |
| A1 区域与路线规划 | 根据地图、机器人尺寸和安全距离求全部可通行区域及区域内路线 | 已实现 EDT + OpenCV 连通域 + 分区扫描，见[A1](#a1区域与路线规划) |
| A2 观察模型 | 共用质量判断；比较物体反向支持区域与路径位置采样 | 下一步，方案已确定，尚未实现 |
| A3 锚点选择 | 物体投票、贪心集合覆盖、冗余删除与拍摄方向选择 | 待实现 |
| A4 离线验证 | 精确解核对、基线比较、质量与成本评估 | 待实现 |
| B 静态真实实验 | 按锚点采集、检查实际图像质量、检索与找物 | 待实现 |
| C 动态维护 | 物体变化后更新分布、复核锚点、刷新记忆 | 待实现 |
| D 长期评估 | 多轮变化、消融实验与失败分析 | 待实现 |

## 目录结构

```text
falhv2/
├── README.md
├── editor.html                  # 地图编辑器页面
├── map_editor.py                # 地图编辑器启动入口
├── path_planner.py              # A1：可通行区域与区域内路线
├── camera_parameters.py    # D435i 相机标定导出
├── config/
│   ├── robot.example.json       # 机器人尺寸与安全距离示例
│   └── camera.d435i.json        # 当前 D435i 的默认 RGB 内参
├── examples/
│   └── map.json                 # 受版本控制的最小示例地图
├── outputs/                     # 本地产物，Git 忽略
│   ├── map.json                 # 用户在编辑器保存的本地地图
│   ├── path_plan.json           # A1 区域与路线结果
│   └── path_diagnostic.svg      # A1 诊断图
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
3. 设置物体高度，选择“放置物体”并点击地图。
4. 选择已有物体后可修改其高度，也可撤销、删除物体。
5. 保存地图为 JSON，或打开已有地图继续编辑。

大地图可使用 **1–64 格宽的方形笔刷**连续涂抹，界面同时显示对应的米制宽度。也可选择**矩形填充**，从一个角拖到另一个角，预览后松开鼠标完成整块填充。两种方式都适用于障碍、未知区域和擦除，每次拖动均可一次撤销。例如 0.05 米/格时，10 格笔刷宽度为 0.5 米。

地图支持鼠标滚轮缩放（25%–3200%），缩放以鼠标所在位置为中心。按住中键或右键拖动可平移，左键继续使用当前绘制工具；点击“显示全图”恢复初始视图。缩放和平移只改变显示，不修改栅格分辨率或物体坐标。

导出内容为 `name`、`grid` 和 `objects`。旧场景文件可以打开，但只保留这些地图数据。

## A1：区域与路线规划

运行示例：

```bash
python -B falhv2/path_planner.py
```

默认优先读取本地 `outputs/map.json`；不存在时使用仓库自带的 `examples/map.json`。`outputs/` 和备份目录 `outputs_/` 不纳入版本控制；测试固定使用 `examples/`，不依赖本地运行产物。

也可显式指定输入和输出：

```bash
python -B falhv2/path_planner.py \
  --scene falhv2/outputs/map.json \
  --robot falhv2/config/robot.example.json \
  --output falhv2/outputs/path_plan.json \
  --svg falhv2/outputs/path_diagnostic.svg
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

**方法：** 用矩形足迹的外接圆加安全距离做保守碰撞检查，使用 EDT 提取安全格中心，再用 OpenCV 四连通标记提取区域。在各区域内进行区间分解和往复扫描，用 A* 连接分区，并比较横向与纵向方案，覆盖全部安全格中心。规划模型假设底盘可以原地转向。

**输出：** 每个区域的格子集合、路线、世界坐标航点、长度及汇总统计，另附 SVG 诊断图。路线首尾由算法确定；断开的区域分别输出。没有任何可通行格时，命令以非零状态退出并说明原因。

区域数量与路线长度取决于输入地图和机器人配置。物体视觉覆盖标记为 `not_evaluated`，留待 A2 判断。实际执行时还需把机器人当前位置接入对应区域路线，并由导航控制器跟踪。

## 读取 D435i 相机参数

使用 conda `test` 环境中的 RealSense Python SDK `pyrealsense2` 直接读取标定。当前已安装 `2.59.1.11053`；新环境可运行 `python -m pip install pyrealsense2`。在仓库根目录执行：

```bash
python -B falhv2/camera_parameters.py
```

默认读取 RGB **1280×720、BGR8、30 FPS** 对应的内参，保存到本地 `falhv2/outputs/camera.d435i.json`。脚本查询设备标定，不启动视频采集；后续实际图像应使用相同的分辨率及几何处理。

[`config/camera.d435i.json`](config/camera.d435i.json) 使用与读取结果完全相同的精简结构，作为后续 A2 的默认输入。当前已复制一份真机内参，待填项仍为 `null`。交给其他用户时，按以下步骤准备自己的配置：

1. 连接相机，运行脚本生成 `outputs/camera.d435i.json`。
2. 人工填写该文件中 `mount` 与 `observation_quality` 的所有 `null`，含义见下表。
3. 将补全后的文件自行复制到 `config/camera.d435i.json`，供离线规划使用。重复读取只更新指定输出文件，脚本不自动替换默认配置。

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

## A2 与 A3 的实现计划

A1 路径固定后，A2 比较两种候选生成方式：**物体驱动**，从每个物体反推合格观察位置和方向，再与路径相交；**位置驱动**，沿路径均匀或随机采样后，检查每个观察能够覆盖哪些物体。首版优先实现前者，以后者作对照。

两条路线使用相同的相机质量模型，输出站位、方向、稀疏覆盖关系和未满足目标。A3 共用动态投票 / 贪心集合覆盖与冗余删除，分别统计站位和照片。相同候选下，两条路线应得到相同覆盖关系；采样密度或随机点数本身不保证质量覆盖。

首版 A2 的物体输入为带稳定 ID 的三维中心点，依据可执行动作、视场、高度和观察距离判断预计覆盖；不要求物体尺寸或 AABB，也不计算物体像素宽高。开始前需确认相机安装位姿和有效观察距离，预计覆盖不等同于已经拍清楚或识别成功。当前 `examples/map.json` 没有物体，需另备观察测试场景。详细方案、缺口处理和公平比较规则见[研究文档第 6 节](../docs/falhv2/path_constrained_anchor_selection_v0_20261013.md#6-a2-与-a3-的两条实现路线)，待办见[交接清单](../docs/falhv2/research_roadmap_and_handoff.md)。

## 数据约定

- 长度单位为米；地图原点在左下角，`+x` 向右、`+y` 向上，JSON 栅格行从上到下保存。
- `.` 表示空闲、`#` 表示障碍、`?` 表示未知；A1 将障碍和未知格均视为不可通行。
- 编辑器宽高支持 4–512 格；例如 300×300 格、0.05 米/格对应 15×15 米。大地图自动缩放显示，格子较小时隐藏网格线；导出分辨率和格子数量保持设置值。
- 每个物体保存稳定的 `id` 和 `position: [x, y, z]`。
- 二维障碍仅表示导航占据，不包含高度或三维相机遮挡几何。
- 当前可通行区域由安全格中心表示；区域路线覆盖与物体视觉覆盖分别评估。

## 测试

在 `test` 环境、仓库根目录运行：

```bash
python -B -m unittest discover -s falhv2/tests -v
node falhv2/tests/test_editor_ui.cjs
```

Python 测试覆盖页面服务、坐标转换、足迹约束、连通区域及路线合法性；Node.js 测试覆盖编辑器交互和导入导出逻辑。

## 后续模块维护

后续子任务沿用 `test` 环境。实现新模块时，在本 README 补充其运行命令、输入输出与依赖要求，并同步更新任务状态。详细研究设计和验收记录维护在上方链接的研究文档中。
