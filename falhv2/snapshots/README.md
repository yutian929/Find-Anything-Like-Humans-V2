# 离线快照与 Valen 找物

在仓库根目录、`falhv2` 环境中运行。快系统或离线回放提供一次到访的观测，`main.py` 处理快照和查找。采集和查找都可以在只完成部分计划时运行。

## 采集

```bash
python -B falhv2/main.py --task snapshots init
python -B falhv2/main.py --task snapshots record --station 5 --yaw 3 --frame /path/to/arrival
python -B falhv2/main.py --task snapshots status
```

`init` 默认冻结配置中的地图和锚点计划，也可指定 `--map`、`--plan`、`--store`。新版规划文件包含输入地图的 SHA-256，旧计划按场景名和目标 ID 集合检查兼容性。新计划使用新的库目录。索引中的 `expected_object_ids` 表示规划预期。

```text
snapshots/
  map.json
  anchor_plan.json
  index.json                         # 完整计划含未拍摄朝向
  anchors/
    station_0005/yaw_003/
      first/{rgb.png,depth_m.npy,meta.json}
      last/{rgb.png,depth_m.npy,meta.json}
```

`station_index` 和 `yaw_index` 取自规划文件。同一锚点的不同朝向各自保存。一次到访产生一个稳定的 `visit_id`，同一次停留的所有帧和重试复用这个 ID。首次到访只写 `first`，后续到访原子替换 `last`。每个视角最多两份。保存时检查底盘与计划的 XY 距离（≤0.10 m）及偏航误差（≤10°）。同一视角要求采集时钟一致，迟到的旧帧会被跳过。

`record --frame` 的输入目录包含三份文件，可直接参考示例库里的一个 `first/`。机器人适配器也可把同步好的 `RGBDFrame` 和到访 ID 传给 `store.record()`。

| 文件 | 内容 |
|---|---|
| `rgb.png` | 校正后的彩色图 |
| `depth_m.npy` | 与彩色图对齐的 H×W 浮点光轴深度，单位米 |
| `meta.json` | `visit_id`、`frame_id`、`timestamp_ns`、`clock_domain`、`map_id`、`map_revision`、`camera_frame`、`intrinsics`、`T_map_camera`、`T_map_base`、`simulated` |

内参含 width、height、fx、fy、cx、cy。两个 4×4 刚体变换都取自采集时刻，分别将光学相机坐标（右、下、前）和底盘坐标变换到地图系。测试帧标记 `simulated: true`。

## 查找

```bash
python -B falhv2/main.py --task find-object --query '蓝色马克杯'
```

每个已访问的朝向先查最新快照。Valen-Decision-1.5 在整图上同时返回存在概率（predicate）、可辨认程度（score）和候选框（bbox）。通过存在概率和框置信度门槛后，裁出候选框，再用 choice 在“匹配、其他、不确定”中确认。多张确认时优先选择可辨认程度最高的视角。所有最新图均未确认时，继续检索有重访记录的视角的 `first`。模型服务错误会让任务失败，可以用同一命令重试。

默认请求本机 `http://127.0.0.1:18765/v1/decisions`，模型别名为 `valen-0.8b`。可用 `--endpoint` 和 `--model` 覆盖。默认门槛 `--predicate-min 0.65 --bbox-min 0.5`，实际环境需要用标注样本校准。模型接收查询、整图及候选裁图。地图、深度、位姿在本机处理。

RealMan 的本地 CPU 服务和两份 GGUF 权重位于下面的目录。若重启机器或服务已停止，可在另一个终端以前台方式启动，运行查找时保持这个终端打开。

```bash
cd /rm_nvme/recordings/user_home/models_weights/Valen-Decision-1.5-0.8B-GGUF
./serve-local.sh
```

每次运行在 `falhv2/outputs/searches/<唯一编号>/` 保存 `result.json`。也可用 `--output` 指定新目录。结果区分 `found_latest`、`found_historical`、`not_found_in_snapshots` 和 `no_snapshots`，另列未访问视角、各张图的分数和确认结果。命中时附原图和标框图、拍摄位姿和时间。框中央区域的深度有效且一致时，按相机内参和实测位姿给出地图系近似表面点，其他情况记录 null。

## 复现部分巡检

```bash
img=falhv2/outputs/demo-images
mkdir -p "$img"
curl -fL https://valen-team.github.io/cookbook/code/media/mug-counter.jpg -o "$img/mug.jpg"
curl -fL https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/fruits.jpg -o "$img/fruits.jpg"
curl -fL https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/basketball1.png -o "$img/court.png"
store=falhv2/outputs/snapshot-search-demo
python -B falhv2/main.py --task snapshots demo --store "$store" \
  --mug "$img/mug.jpg" --fruit "$img/fruits.jpg" --court "$img/court.png"
python -B falhv2/main.py --task find-object --store "$store" --query 'a blue mug'
```

`demo` 要求一个新目录。RealMan 已生成的样例可以从最后一条查找命令直接开始。模拟先拍水果，停留时重复提交，到另一站拍篮球场，最后两次重访第一站，依次拍篮球场和蓝杯。事件及原图 SHA-256 在 `demo_events.json`。图片是公开样例，深度、相机内参、拍摄时间及底盘/相机位姿是模拟值。

模型支持的四类问答、图像格式和本地服务构建说明见[官方 Decisions API](https://valen-team.github.io/docs/valen/api/decisions/) 和[官方 GGUF 服务文档](https://valen-team.github.io/docs/valen/gguf/)。
