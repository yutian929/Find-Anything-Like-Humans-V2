# FALH-V2

本目录实现从地图构建、安全区域提取、观察锚点选择，到视觉采集与找物验证的各阶段功能。本 README 记录运行环境、已实现模块的用法及后续任务入口。

默认入口运行快、慢系统。在 Lenovo 已具备模型与规划依赖的 depth 环境中：

```bash
conda activate depth
cd falhv2
python main.py
```

快系统持续读取同步 RGB-D 和底盘栅格，调用 Cubify，按深度二维覆盖与框投影的栅格交集删旧存新；不自动移动底盘。
慢系统仅在终端收到 `replan` / `重规划`，或网页点击“重规划锚点”后运行。
每次请求立即冻结一份地图，规划期间快系统继续更新；同一时间仅运行一次重规划，不累计重复请求。
终端输入 `quit` / `退出` 或 Ctrl+C 退出；正在执行的推理/规划会完成清理后退出。

默认实时页面为 http://127.0.0.1:8090，显示栅格、深度覆盖、物体框、相机 RGB 和最近一次锚点。
在本机浏览远程 Lenovo 页面，可先运行 `ssh -N -L 8090:127.0.0.1:8090 hri@10.113.45.27`，再打开本机该地址。
也可显式 `--host 0.0.0.0` 在可信局域网提供访问；此开发页面没有公网认证。

每次启动创建新的 `outputs/live_<时间>_<会话>/`：
- `map.json`：持续原子更新的地图及恢复状态。
- `plans/0001/scene.json`：第一次指令冻结的地图；后续指令递增目录。
- `plans/0001/anchor_plan.json/html/svg` 及 supports、安全区域：独立的重规划结果。
规划 JSON 记录 source_map；网页明确提示锚点对应的旧地图版本，不会悄悄自动重算。

默认自动生成本次应用会话的 map_id / map_revision，并不代表底盘提供了真实 SLAM 版本。
重定位、地图切换、回环坐标修正后必须重启会话；默认不继承上次物体地图。
ROBOT_SOURCE 可以指定会话和设备地址，LIVE 配置页面地址、采集超时和更新间隔；MAP_UPDATE 控制共用覆盖掩膜的深度采样。
只有 main.py 加载配置，`--config /path/to/robot.py` 可选择另一份配置。

原来的离线锚点全流程改为显式选择：

```bash
python falhv2/main.py --task anchors
python falhv2/main.py --task anchors --scene /path/to/map.json
```

离线任务一次完成安全区域提取、观察候选计算、优化、复核和 JSON/HTML/SVG 导出。
默认规划参数和输入路径不变；地图编辑器、相机标定和有限帧建图仍通过各自 --task 启动。

候选观察默认按目标物体使用 **4 个进程**并行，每个任务仍读取全部物体作为遮挡体。也可选择线程或立即回退串行：

```bash
python -B falhv2/main.py --task anchors --observation-executor threads --observation-workers 4
python -B falhv2/main.py --task anchors --observation-workers 1
```

任务结果按原物体顺序合并，MILP 和最终复核保持原流程。三轮完整运行中位数 **23.19 s → 13.86 s（减少 40.2%）**，候选生成 **13.97 s → 4.71 s**；完整 supports、评分、热图、诊断和最终方案一致。4 线程单次为 15.72 s，因此默认选择 4 进程。实验对照与参数说明见[逐物体并行记录](../docs/falhv2/parallel_observation_20261026.md)。

2026-10-26 起，观测计算使用 NumPy 批量矩阵乘法、列式数组与批量统计，以及 Shapely/GEOS 双精度凸包和裁剪。安装依赖仍在同一个环境中执行：

```bash
conda activate test
python -m pip install -r falhv2/requirements.txt
```

依赖统一在 `requirements.txt` 中维护，包含 Open3D。默认射线后端仍为 NumPy Float64；Open3D 为显式实验选项，最终原密度及高密度复核始终使用 NumPy Float64：

```bash
python -B falhv2/main.py --task anchors --ray-backend open3d --output falhv2/outputs/open3d/anchor_plan.json
```

Open3D 使用 Float32；当前地图有少量候选评分差异，且包含启动成本的实测未优于 NumPy，因此不自动切换后端。`configuration.candidate_ray_backend` 和 `verification_ray_backend` 记录实际实现。分阶段过程、精度边界、批次/线程与混合求交实验见[现成库优化记录](../docs/falhv2/library_acceleration_20261026.md)。

研究定义与算法依据见[锚点选择研究记录（含旧版路径约束方案）](../docs/falhv2/path_constrained_anchor_selection_v0_20261013.md)；实施进度和验收要求见[研究路线与交接](../docs/falhv2/research_roadmap_and_handoff.md)。

观察模型与旧加权贪心规则见[锚点选择改进说明](../docs/falhv2/anchor_selection_refine.md)：安装位姿约束观察区域、连续权重和逐物体热力图。当前选点采用下述两阶段整数优化。[旧版锚点选取图解](../docs/falhv2/anchor_selection_explained_20261019.md)保留了改进前的七张示意图和运行案例。

## 真实机器人地图更新

通过 `python falhv2/main.py --task build-map --map-id <地图标识> --map-revision <坐标版本> --frames 10` 获取底盘栅格和同步 RGB-D，调用 Cubify，并按当前深度可见性删除旧框、加入本帧新框，不做关联或融合。
输出规划器兼容的地图 JSON（含恢复状态）及覆盖诊断 SVG；`--resume <地图.json>` 继续更新，`--scene <地图.json>` 用于锚点选择。
只有 main.py 读取 CUBIFY、ROBOT_SOURCE 和 MAP_UPDATE；默认入口为实时快、慢系统；--task anchors 为离线选点。
参数、版本约束与命令见 [地图维护说明](map_maintainers/README.md)。

## 当前选点目标（2026-10-24）

全部输入物体至少有一次合格观察；优先最小化锚点数，再在该锚点数下最小化照片数。`anchor_optimizer.py` 使用已有 SciPy 的 `optimize.milp` 分两阶段求解，第二阶段在全部候选站位和方向上重新选择。完整候选生成沿用原观察模型、门槛和采样。

求解器会作保持最优成本的等价化简：同站去除被另一张合格照片完全覆盖的冗余动作，再合并“全部极大方向覆盖集合”相同的站位；不是仅按站位覆盖并集或空间距离合并。输出仍映射到真实站位/朝向，完整候选 supports 和热力图保持不变。该化简只适用于当前站数/照片数目标，不适用于加入导航、转向或质量奖励后的目标。

旧加权贪心保留为基线，输出包含 `baseline_cost`、`selected_cost`、各阶段状态、下界和 gap。只有两阶段均证明最优且通过最终几何复核时，`optimality` 才为 `finite_pool_lexicographic_optimal`。`complete` 只表示全覆盖。无支持目标仍返回 `partial`，不从覆盖分母中删除。

默认不限求解时间；可以添加 `--selection-time-limit-s 120` 限制全部 MILP 调用的总时间，包括下述复核重选（不含观察计算、建模和最终复核）。限时或求解异常时保留已经验证的最佳可行方案，成本不劣于旧基线，并明确说明未完成证明。新方案几何复核失败时退回通过复核的基线。

默认配置下，完整 AABB 的最终照片额外用至少 96×96 遮挡采样检查全部目标。若出现覆盖缺口，仅在求解副本中删除复核否定的关系，最多重选三次；完整候选与热图不改写。重选失败或预算不足时，退回通过高密度复核的基线；基线也未通过则明确报告复核失败。修正解只有达到原来已证明的成本才继承原候选池最优性。`optimization.high_density_verification` 单独记录初次、最终、重选及基线复核；每张照片的 `high_density_object_metrics` 与原采样 `object_metrics` 分开保存。`complete` 和最优性标签针对原观察矩阵，是否通过高密度检查必须另看上述字段，页面同时显示。

2026-10-25 起，若初始两阶段已证明最优，删边后的重选先在原最优站数和照片数下做限时可行性搜索。找到方案并通过复核即可继承原候选池证明；搜索本身不产生新的最优性证明。搜索失败则恢复完整两阶段优化，预算耗尽则回退已复核基线。每次尝试默认最多 2 秒，并计入全局 MILP 预算；设置 `SAME_COST_REPAIR_TIME_LIMIT_S = 0` 可关闭。`high_density_verification.same_cost_searches` 保存尝试状态及耗时；成功修复对应的 `repairs[].optimization.fixed_cost_feasibility` 保存可行性结果。同成本解的具体站位和朝向可能与旧版不同。

主结果为 schema 5；候选 `.supports.json` 的结构保持 schema 4。新结果包含独立的 `baseline`、`optimization`、`objective_order` 等字段。页面支持新方案/旧基线切换：新方案保留静态热力图，只有基线提供贪心轮次回放。

默认地图验收：旧方案 **5 站14张**，新方案 **4 站10张**；最终 28/28 目标在原模型与 96×96 复核下均合格覆盖。原候选池两级资源下界分别为 4 和 10，gap 均为 0。首次最优解出现一个遮挡采样边界问题，经过一次自动重选修复，成本不变。实现依据见[两阶段选点计划](../docs/falhv2/lexicographic_anchor_selection_plan_20261024.md)，过程、时间和边界见[验收报告](../docs/falhv2/lexicographic_anchor_selection_validation_20261024.md)。

2026-10-25 加速验收：完整命令三次运行中位数由 **74.55 s 降为 35.54 s（约 2.10 倍）**，候选计算由 57.06 s 降为 26.18 s。完整 supports、得分、热图和最佳候选诊断保持一致；最终仍为 4 站10张，28/28 通过高密度复核。具体同成本站位可能变化，见[加速实现与对照报告](../docs/falhv2/observation_acceleration_validation_20261025.md)。

2026-10-26 现成库优化验收：三轮交替完整运行的中位数 **35.37 s → 23.16 s（减少 34.5%）**，候选生成 **26.02 s → 14.03 s**。完整 supports、评分和热图一致；几何面积存在双精度末位差异，最终站位/朝向不变，仍为 4 站10张、28/28 高密度复核通过。Open3D 实验版完整运行 25.07 s，存在 3 条候选评分差异，默认继续使用 NumPy，见[优化过程与验收记录](../docs/falhv2/library_acceleration_20261026.md)。

## 环境与运行约定

实验调参入口为 [`config/defaults.py`](config/defaults.py)，观察门槛与采样默认值保持不变：

| 配置项 | 默认值 | 用途 |
|---|---|---|
| `OBSERVATION_DEFAULTS.min_in_frame_fraction` | 0.8 | 最低入镜比例 |
| `OBSERVATION_DEFAULTS.min_observation_score` | 0.5 | 最低得分，包含等于门槛 |
| `OBSERVATION_DEFAULTS.min_visible_pixels` | 400 | 最低可见像素面积，px² |
| `OBSERVATION_DEFAULTS.radial_inner_weight` / `radial_outer_weight` | 1.0 / 0.5 | 距离权重内外端点，线性下降 |
| `OBSERVATION_DEFAULTS.sample_grid` / `refinement_grid` | 12 / 36 | 常规与临界遮挡采样边长 |
| `YAW_STEP_DEG` | 30 | 每站拍摄方向间隔，度 |
| `OBSERVATION_WORKERS` | 4 | 逐物体候选计算的 worker 数；1 恢复串行 |
| `OBSERVATION_EXECUTOR` | `processes` | 候选阶段使用 `threads` 或 `processes` |
| `SELECTION_TIME_LIMIT_S` | `None` | 全部 MILP 调用共享预算，秒；`None` 不限时 |
| `AUDIT_SAMPLE_GRID` | 96 | 最终复核网格边长；实际不低于临界采样边长 |
| `MAX_AUDIT_RETRIES` | 3 | 复核缺口重选次数；0 时直接尝试基线兜底 |
| `SAME_COST_REPAIR_TIME_LIMIT_S` | 2.0 | 每次同成本修复的 MILP 秒数上限，受全局剩余预算约束；0 关闭 |

`OBSERVATION_DEFAULTS` 是字典，修改其中的数值即可。相机 JSON 的 `observation_quality` 若显式定义同名字段，则覆盖该字典；目前默认相机 JSON 只定义距离范围，所以上表观察参数均由 `config/defaults.py` 生效。CLI 的 `--yaw-step-deg` 和 `--selection-time-limit-s` 覆盖规划默认值。Python `plan()` 未传入或传入 `None` 时采用对应配置值。

`--observation-workers`、`--observation-executor` 覆盖并行配置；`main.plan()` 对应参数为 `observation_workers`、`observation_executor`。进程方式的 Python API 调用应放在 `if __name__ == '__main__':` 入口保护中；交互式环境可选线程或单 worker。空输入、无站位或单目标自动使用串行，实际执行方式记录在 `statistics.candidate_workers/candidate_executor`。

运行结果的 `observation_settings` 保存合并后的观察参数，`configuration` 保存实际方向间隔、时间预算和复核配置。修改后需重新运行规划器，已有 HTML/JSON 不会自动刷新。本文的 0.5 门槛、96×96 复核、三次重选及 4 站10张验收结果均对应默认配置。

端点须满足 `0 < 外端 <= 内端 <= 1`；采样边长为 `[4,128]` 的整数且临界采样不小于常规采样；重选次数为非负整数。非法配置会明确报错。相机标定、机器人安全余量、地图分辨率继续读取原 JSON；加密触发带宽、数值容差及两阶段目标优先级继续保留在算法代码中。

当前使用已有的 **conda 环境 `test`**。运行或开发前，先激活环境并进入仓库根目录：

```bash
conda activate test
cd /home/hri/yutian/Find-Anything-Like-Humans-V2
```

下文所有命令均在仓库根目录执行，使用激活环境中的 `python`。地图编辑器仅依赖 Python 标准库；A1 与锚点规划使用 `test` 环境中的 NumPy、SciPy 和 OpenCV；离线规划不需要连接相机或安装 RealSense SDK。编辑器前端测试另外需要 Node.js。

## 输入与输出管理

只有 main.py 加载项目配置，其余模块通过函数或构造参数接收所需值，不 import config。
默认读取 config/defaults.py；使用 `--config /path/to/experiment.py` 可选择其他 Python 配置。
配置文件中的大写变量覆盖默认项，字典按键合并；未指定的参数沿用默认值，CLI 再覆盖本次运行。
顶层路径参数如 MAP_PATH、OUTPUT_DIR 的相对路径以所选配置文件目录为基准；模型字典中的路径建议写绝对路径。

```bash
python falhv2/main.py --task anchors --config /path/to/experiment.py
python falhv2/main.py --config /path/to/robot.py --task cubify-preview
```

模型仍是一模型一个字典，例如 CUBIFY；无需为本地、远程和预览分别增加配置块。
`main(argv)` 解析命令行、按选定路径读取三份 JSON，再调用 `plan(...)`，最后导出结果。
`plan(...)` 负责安全区域、候选生成、优化和复核；未显式传入的规划参数取自 settings。
方向间隔、并行参数和求解预算以 CLI 显式参数优先；观察门槛以相机 JSON 中显式
提供的同名字段优先于 `OBSERVATION_DEFAULTS`。输入输出路径由配置文件位置解析，
不依赖启动目录。更改配置后需重新运行程序。

main.py 通过 [config/defaults.py](config/defaults.py) 统一管理默认路径，并显式传给各模块，路径相对 `falhv2` 目录解析，不受启动工作目录影响。

| 目录 / 文件 | 用途与维护方式 |
|---|---|
| `data/map_aabb.json` | 当前地图和物体 AABB，来自编辑器导出；持久输入 |
| `data/robot.json` | 当前机器人的实测尺寸和安全距离；持久输入 |
| `data/camera.d435i.json` | 当前相机内参与人工补全的安装、观察距离；持久输入 |
| `outputs/` | 可删除的生成结果，运行脚本会重建；不作为默认输入来源 |

地图编辑器通过浏览器下载 `map.json`，请将它重命名为 `map_aabb.json` 并放到 `data/map_aabb.json`。相机读取脚本生成 `outputs/camera.d435i.json` 草稿，补全人工项后自行复制到 data；复制完成后草稿可删除。规划器始终读取 data 中的正式输入，不会读取草稿或自动选择其他地图。配置缺失时明确报错；也可使用 `--scene`、`--robot`、`--camera` 显式指定其他文件。

地图、机器人和相机 JSON 已集中到 `data/`，内容保持不变。以后更换输入文件，只需修改 `config/defaults.py` 中的 `MAP_PATH`、`ROBOT_PATH`、`CAMERA_PATH`；也可临时通过命令行覆盖。旧 outputs 中的地图不参与默认输入选择。

## 任务与进度

| 阶段 | 内容 | 状态与入口 |
|---|---|---|
| 地图编辑器 | 绘制栅格地图、编辑无类别物体 AABB、显示高度、导入导出 JSON | 已完成，见[地图编辑器](#地图编辑器) |
| A1 安全区域提取 | 根据地图、机器人尺寸和安全距离求全部安全连通区域 | 已实现 EDT + OpenCV 连通域，见[A1](#a1安全区域提取) |
| A2 观察模型 | 安装位姿与视场径向预筛，AABB 几何入镜、部分遮挡和加权得分 | 已在优化分支实现，见[锚点规划](#a2a3物体驱动锚点规划) |
| A3 锚点选择 | 全覆盖约束下先最少锚点，再最少照片 | 两阶段整数优化；保留旧基线和独立回放，待现场验证 |
| A4 离线验证 | 精确解核对、基线比较、质量与成本评估 | 已完成首轮八场景实验，见[报告](../docs/falhv2/anchor_driver_experiment_20261014.md) |
| B 静态真实实验 | 按锚点采集、检查实际图像质量、检索与找物 | 待实现 |
| C 动态维护 | 物体变化后更新分布、复核锚点、刷新记忆 | 待实现 |
| D 长期评估 | 多轮变化、消融实验与失败分析 | 待实现 |

## 目录结构

```text
falhv2/
├── main.py                      # 配置加载、CLI 与离线选锚点全流程编排
├── __init__.py
├── anchor_planners/            # 已实现：现有锚点规划算法
│   ├── regions_extractor.py     # 安全区域
│   ├── anchor_candidates.py     # 观察候选与热图
│   ├── anchor_selector.py       # 基线与结果组装
│   ├── anchor_optimizer.py      # 两阶段 MILP
│   ├── anchor_verification.py   # 复核与修复
│   ├── anchor_export.py         # 导出
│   ├── observation_model.py
│   └── observation_region.py
├── config/defaults.py           # 路径和实验默认参数
├── robot_adapters/base.py               # 新增：采集与可选导航接口
├── model_clients/               # CuTR RGB-D 本地/远程客户端与模型协议
├── map_maintainers/base.py              # 新增：地图融合与快照接口
├── tools/                      # 已实现：独立输入准备工具
│   ├── map_editor.py
│   └── camera_parameters.py
├── assets/                     # 地图编辑器和锚点展示 HTML
├── data/                       # 正式地图、机器人尺寸、相机标定输入
├── outputs/                    # 生成产物，Git 忽略
├── tests/
├── requirements.txt
└── README.md
```

2026-10-28 的整理只改变包结构，选点算法、默认输入、输出 schema 和参数保持不变。
`model_clients` 已接入 Cubify Anything RGB-D 的本地/远程调用；`robot_adapters` 已实现同步 RGB-D 获取、栅格地图读取和底盘定位（见 [调用示例](robot_adapters/README.md)），`map_maintainers` 实现深度约束的物体地图更新，不做多帧融合。使用方法见 [模型客户端](model_clients/README.md)。默认 main.py 运行实时快、慢系统；离线选点使用 --task anchors。
完整设计和后续实施顺序见[机器人建图架构](../docs/falhv2/robot_mapping_architecture.md)。

在仓库根目录运行 `python -B falhv2/main.py` 或 `python -B -m falhv2.main`；
进入 falhv2 后运行 `python -B main.py` 也保持有效。脚本可从任意工作目录用绝对路径执行。
Python 调用使用 `from falhv2.main import plan`；它返回
`(result, regions, stations, candidates)`，不写文件。配置加载与 CLI 调度均实现在
`main.main(argv=None)`，安全区域只计算一次。

旧的顶层模块路径已迁移，不再保留一排转发脚本。
`anchor_planners.anchor_selector` 只提供选点算法，不再提供 `plan()`、CLI 或脚本启动入口。完整流程统一使用 `falhv2.main`。
独立命令统一使用包入口（在仓库根目录执行）：
`python falhv2/main.py --task map-editor`、
`python falhv2/main.py --task camera-parameters`、
`python falhv2/main.py --task safe-regions`。
旧 docs 中的历史运行记录不改写，当前命令以本 README 为准。

## 地图编辑器

启动本地服务：

```bash
python falhv2/main.py --task map-editor
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
python -B falhv2/main.py --task safe-regions
```

默认读取 `data/map_aabb.json` 和 `data/robot.json`，缺失时报错。需要使用其他地图时通过 `--scene` 显式指定。`outputs/` 和备份目录 `outputs_/` 不纳入版本控制；删除 outputs 不影响正式输入。

也可显式指定输入和输出：

```bash
python -B falhv2/main.py --task safe-regions \
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
python -B falhv2/main.py --task camera-parameters
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
python -B falhv2/main.py --task camera-parameters --list-modes
```

在 `falhv2` 目录内运行时省略命令中的 `falhv2/` 即可。深度流可通过 `--stream depth --width 640 --height 480 --output falhv2/outputs/camera.depth.json` 另存；后续 RGB 观察模型使用 RGB 配置。

裁切、缩放或去畸变后，应使用对应的新内参。畸变模型与系数保持 SDK 返回值。标定含义参考 [RealSense SDK API](https://github.com/realsenseai/librealsense/blob/master/examples/sensor-control/api_how_to.h)及[投影说明](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)。

## A2/A3：物体驱动锚点规划

在仓库根目录运行统一入口：

```bash
python -B falhv2/main.py --task anchors
```

默认输入与 A1 一致：读取 `data/map_aabb.json`、`data/robot.json`，相机读取 `data/camera.d435i.json`。脚本内部调用 A1 重新计算安全区域并直接生成站位，不读取已有规划输出，因此可独立运行。无需机器人起始位姿、起终点或指定锚点数量。

也可显式指定编辑器保存的地图和配置：

```bash
python -B falhv2/main.py \
  --scene falhv2/data/map_aabb.json \
  --robot falhv2/data/robot.json \
  --camera falhv2/data/camera.d435i.json \
  --output falhv2/outputs/anchor_plan.json
```

在 `falhv2` 目录内可以直接运行 `python -B main.py`，在仓库根目录也支持 `python -B -m falhv2.main`。`--yaw-step-deg` 设置底盘拍摄方向间隔，默认 30°，即每站检查 12 个方向。相机安装高度和俯仰固定，底盘可原地转向。

### 处理流程

1. A1 提取安全连通区域，每个安全格中心生成一个候选站位；按区域 ID、行、列固定排序。
2. 对每个物体先按相机高度和水平安装偏移查询附近安全站位，再逐方向解析计算径向范围。范围包含完整安装旋转、相机视场、中心距离及 Box 投影外框交叠约束，先排除不可能入镜的方向。
3. 对保留方向求 Box 投影轮廓与画面的几何裁切面积，再采样射线估计未遮挡比例。距离权重在径向内边界为 1、外边界为 0.5，线性递减；最终得分为距离权重 × 未遮挡比例。得分至少 0.5、入镜至少 80%、可见面积至少 400 px² 才算合格。全点地图保留 `legacy_point` 旧视场/距离及单位票模式。
4. 保留旧加权贪心作为对照；在合格支持关系上求最少锚点数，再固定锚点数量，在全部站位和方向上求最少照片数。分数不覆盖资源成本的优先级。
5. 独立复核基线和新方案的拍摄动作，保存逐物体观察指标、遮挡物 ID、覆盖缺口、求解证明状态及可视化。

### 部分遮挡模型与门槛

所有输入 Box 都作为实心遮挡体，也都属于待覆盖物体。仅有二维栅格的墙不参与三维遮挡；AABB 空隙被当作实体，属于偏保守的场景近似。距离仍为相机光心到 Box 中心的三维欧氏距离，因此 KDTree 中心距离粗筛保持有效，不采用“最近表面距离”。

从投影外框上的均匀网格发射射线，只保留与目标相交的采样。目标最近表面之前先碰到其他 Box 的采样记为遮挡；后方 Box 不遮挡，目标背面不重复计入分母。

- `in_frame_fraction`：画面内目标投影面积 / 完整目标投影面积。
- `visible_fraction`：画面内未遮挡目标面积 / 画面内目标投影面积。
- `visible_pixels`：画面内未遮挡面积，单位像素平方。
- `distance_weight`：在含安装姿态和视场约束的几何径向范围内，从 1 线性降到 0.5。
- `observation_score`：距离权重 × 未遮挡比例，用于合格判断、热力图及旧基线投票。
- `quality`：入镜比例 × 未遮挡比例，保留作诊断，不再用于投票。

径向内外边界是固定相机朝向和相机至目标水平方位后的几何包络，不是从当前地图可用站位的最近/最远点估计。AABB 外框与画面相交只是必要条件，包络边界可能被后续 80% 入镜门槛继续裁掉；通过全部门槛的位置才出现在热力图里。详见[定义与边界](../docs/falhv2/anchor_selection_refine.md)。

可在相机 JSON 的 `observation_quality` 中添加下列可选字段；未填写时使用表中初始值，正式结果保存实际参数。这些默认值尚未通过真实识别实验标定。

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `min_in_frame_fraction` | 0.8 | 最低入镜比例，范围 `(0,1]` |
| `min_observation_score` | 0.5 | 最低最终加权得分，范围 `(0,1]` |
| `min_visible_pixels` | 400 | 最小可见像素面积，须为正 |
| `sample_grid` | 12 | 投影外框每轴初始采样数，范围 4–128 |
| `refinement_grid` | 36 | 接近门槛时每轴加密采样数，不低于初始值 |

入镜比例或最终得分距离门槛不超过 0.08、或像素面积距离门槛不超过门槛的 20% 时重新加密遮挡采样。入镜比例用投影凸包裁切计算，不依赖采样点计数；可见面积为裁切后的投影面积 × 采样未遮挡比例。分批处理并粗筛遮挡体；很细的遮挡物仍可能漏采，没有解析遮挡误差保证。相机位于任一实心 Box 内、或目标跨越相机平面时保守拒绝。

观察计算复用粗细采样之间的投影、裁切面积、距离权重和遮挡体范围；采样网格与盒角点缓存，权重批量计算，射线求交按轴处理以减少临时数组。候选阶段先检查入镜比例和可见面积上界，确定不合格的观察跳过遮挡计算；这些内部记录的未计算指标为 `None`，不会作为真实分数导出。若物体没有合格观察，补算被跳过的观察以恢复原来的最佳失败诊断及平票顺序。最终照片仍完整计算全部物体指标。`statistics.occlusion_skipped_pose_pairs` 和 `diagnostic_recomputed_pose_pairs` 分别记录跳过和补算数量。完整候选池、合格分数、热图和原采样/加密规则保持不变。

旧字段 `min_visible_fraction` 已移除；配置仍含此字段时会提示删除，以避免旧 60% 门槛与新规则同时生效。默认相机 JSON 没有该旧字段，无需修改。

仅旧加权贪心基线按新增合格得分之和、新增物体数、站位编号排序。每个物体在同站仅取最佳方向，得分不足配置门槛（默认 0.5）不计票。冗余删除优先尝试低总分项，仍以保持合格覆盖为条件；照片选择保留“新增覆盖数优先、得分平票择优”，减少定向照片。上述基线不承诺全局最优；新默认方案的最优性以 `optimization` 的证明状态为准。两者均不把多个不合格方向相加成覆盖。

### 输出与查看

| 文件 | 内容 |
|---|---|
| `outputs/safe_regions.json` / `safe_regions.svg` | 本次规划中计算的安全区域与诊断图，无需另跑 A1 |
| `outputs/anchor_plan.json` | 站位编号、区域、格坐标、底盘偏航（弧度）、每张照片覆盖的物体 ID、逐物体比例与遮挡原因、最佳已检查候选诊断、缺口、配置与耗时 |
| `outputs/anchor_plan.html` | 自包含离线页面；新旧方案及证明状态、逐物体热力图、静态叠加图和独立基线回放、点击安全站位查看得分和最佳朝向；支持缩放、平移、投影检查和地图 PNG 导出 |
| `outputs/anchor_plan.svg` | 静态地图、初始加权总分、锚点与拍摄方向 |
| `outputs/anchor_plan.supports.json` | 默认导出合格关系、连续 `scores` 和 `object_heatmaps`；方向键为 `yaws_rad` 的索引，未列出的项无有效贡献 |

默认输出路径由 `config/defaults.py` 管理。使用其他 `--output`（例如 `/tmp/run/plan.json`）时，锚点附件使用相同主干，安全区域另存为 `/tmp/run/plan.safe_regions.json` 和 `plan.safe_regions.svg`，不会写回默认目录。`--no-save-supports` 可跳过候选关系导出；原 `--save-supports` 参数仍可使用。直接双击生成的 HTML 即可离线查看或转发，无需启动服务；`anchor_selector.html` 是模板，查看结果应打开 outputs 中的页面。

在“观察热力图”中选择具体物体，或直接点击地图物体/右侧物体按钮，显示该物体每个安全站位的最佳合格得分，色标从实际观察得分门槛到 1，默认是 0.5～1。切到“基线：加权贪心”的“物体观察能力叠加”后，拖动轮次条查看已覆盖物体撤票后的分布；总分色标保持初始尺度，便于比较。回放记录的是删冗余之前的贪心过程，不代表导航顺序。单物体图不随回放轮次消失。无色格表示未通过全部条件，不代表未检查。

全部输入目标进入统计：全部有支持返回 `complete`；有缺口返回 `partial` 并列出原始物体 ID；空目标返回 `empty_targets`、零锚点和 `null` 覆盖率。缺口指当前离散站位与方向无支持，未证明连续空间无解。无可通行格、未补全相机参数或非零畸变系数会明确报错。

历史实验地图在当时配置下得到 **3 个站位、10 张定向照片，28/28 点覆盖**；该快照保留在实验分支，主线直接使用编辑器保存的地图。该历史选择采用启发式；当前两阶段方案的证明仅针对给定有限候选池和观察模型。该历史结果来自点模型。当前 AABB 模型评估采样遮挡和投影面积，仍未验证实际识别；二维导航障碍不直接作为相机遮挡。锚点按站位编号排列，仅用于稳定展示，不代表访问顺序。当前包含所有安全连通区域，尚未根据起点筛选，也未规划站位之间的行驶路线。

### 从扫描路线版迁移

2026-10-19 起移除前置全覆盖扫描，`path_planner.py` 替换为 `regions_extractor.py`。原路线覆盖全部安全格中心，所以删除扫描不改变候选位置集合；站位编号改为区域、行、列顺序，平票时选出的具体位置可能变化。

删除扫描时使用 `schema_version: 2`；此次 AABB 升级为 `schema_version: 3`，新增观察模型、实际门槛、质量与指标。移除 `first_route_index`、`route_indices`、`region_transfer` 和路线长度统计，新增 `navigation: not_planned`、未指定起点的可达性说明及 `station_order`。预处理统计改为 `safe_regions_and_station_seconds`、`safe_station_count`。`plan()` 的第二个返回值为安全区域结果。旧路线文件不再生成或读取；查看时请重新运行入口，使用新导出的页面。

2026-10-19 的加权观察版本升级为 `schema_version: 4`，AABB 模型名为 `aabb_weighted_region`。支持表 `qualities` 改为 `scores`；动作 `quality_by_object` 改为 `score_by_object`；投票记录使用 `weighted_score` 与 `new_object_count`。热力图每条记录为 `[station_index, best_qualified_score, yaw_index, vote_units]`，`vote_units` 是该得分乘一百万后的整数，前后端均用它汇总，避免平票时不同舍入。旧图解中的 4 站 12 张结果属于旧规则。

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
node falhv2/tests/test_live_ui.cjs
```

Python 测试另覆盖两阶段字典序目标、独立动作子集穷举、限时/异常兜底、几何复核、高密度缺口修复及最优性证明的适用范围。原加权贪心测试显式验证旧基线。Python 测试覆盖安全区域、相机几何、俯仰与径向范围解析核对、预筛与独立投影对照、内外边界得分、0.5 合格门槛、加权动态贪心、几何裁切入镜比例、热力图与最终动作得分一致性、覆盖缺口及 CLI 导出。Node.js 测试覆盖编辑器、单物体热图、叠加扣票、轮次回放和最佳朝向展示。新求解器重点测试可运行 `python -B -m unittest discover -s falhv2/tests -p 'test_anchor_optimizer.py' -v`。

统一入口测试另覆盖默认实时调度、显式离线任务的完整输出、A1 只计算一次、配置路径与输入文件保持、从其他工作目录运行脚本、包入口，以及自定义输出文件名不会被安全区域文件覆盖。

## 后续模块维护

后续子任务沿用 `test` 环境。实现新模块时，在本 README 补充其运行命令、输入输出与依赖要求，并同步更新任务状态。详细研究设计和验收记录维护在上方链接的研究文档中。

模型客户端接口测试需额外安装 `falhv2/model_clients/requirements.txt`，无需 torch 或权重。

机器人适配器测试需安装 `pyzmq`、`msgpack`、`requests`，测试使用本地模拟发布器和 HTTP 服务，无需连接真实机器人。

实时系统验证（2026-10-31）：149 项 Python 测试通过；实时页面 Node 交互测试通过。
Lenovo depth 环境补齐 shapely==2.1.2；默认实时入口完成真实 RGB-D / 底盘栅格 / Cubify 连续三帧更新、HTTP 页面与状态读取、quit 正常退出。
快慢并发、冻结快照、重规划失败隔离、实际规划导出在模拟场景中验证；尚未对真实大地图完成一次完整重规划验收。

地图维护已改为二维栅格相交：新旧框使用同一更新区域，网页青色区域即实际更新依据。旧 50% 可见比例规则已移除；切换后启动新会话。
