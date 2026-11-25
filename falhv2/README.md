# FALHv2

外部快系统提供地图。FALHv2 读取一份地图规划锚点与拍摄朝向，接收各视角的到访观测，再用 Valen 在已拍摄的快照里找物。

## 六个任务

在仓库根目录激活 `falhv2` 环境。运行 `python falhv2/main.py --help` 查看任务列表，运行 `python falhv2/main.py --task <任务名> --help` 查看该任务参数。

| 任务 | 输入与结果 | 实现位置 |
|---|---|---|
| `anchors-plan`（默认） | 地图、机器人尺寸、相机标定 → 锚点、拍摄朝向、覆盖复核及可视化 | `anchor_planners/planning.py` |
| `safe-regions` | 地图、机器人尺寸 → 安全连通区域及 SVG | `anchor_planners/regions_extractor.py` |
| `map-editor` | 在本机浏览器编辑栅格和物体框，导出场景 JSON | `tools/map_editor.py` |
| `camera-parameters` | 读取 D435i 支持的模式或导出相机参数 | `tools/camera_parameters.py` |
| `snapshots-updater` | 初始化快照库，按到访记录 first / last，查看采集进度或生成模拟数据 | `snapshots_updater/` |
| `find-object` | 文字描述和快照库 → 最新或历史命中、图像证据及可用的地图坐标估计 | `object_search/` |

## 从规划到找物

```bash
python falhv2/main.py --task anchors-plan
python falhv2/main.py --task snapshots-updater init
python falhv2/main.py --task snapshots-updater record --station 5 --yaw 3 --frame /path/to/arrival
python falhv2/main.py --task snapshots-updater status
python falhv2/main.py --task find-object --query '蓝色马克杯'
```

采集示例中的站点和朝向要取自本次计划。`--frame` 指向一次到访的 RGB、对齐深度及元数据。查找前启动 Valen 服务。完整输入格式、模拟巡检和服务启动命令见[快照与找物](snapshots_updater/README.md)。

## 配置与开发

- 默认路径和规划参数在 [config/defaults.py](config/defaults.py)。用 `--config /path/to/experiment.py` 覆盖本次运行的配置。
- 持久输入在 `data/`。默认规划结果在 `outputs/anchor_plan.json`，快照库在 `outputs/snapshots/`，查找结果在 `outputs/searches/`。
- `main.py` 选择任务并传入配置。Python 调用可继续使用 `from falhv2.main import plan`，返回 `(result, regions, stations, candidates)`。
- `robot_adapters/` 定义观测数据，`model_clients/` 对接模型。规划算法、工具参数和历史实验见[规划与工具参考](planning_reference.md)。
- 回归测试在仓库根目录运行 `python -B -m unittest discover -s falhv2/tests`。
