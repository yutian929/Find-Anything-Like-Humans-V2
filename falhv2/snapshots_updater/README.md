# 快照库与 Valen 找物

默认库位于 `falhv2/data/snapshots/`。示例只有一个站点、一个朝向和一张蓝色马克杯照片。深度、标定和位姿均为模拟值。[图片来源](https://valen-team.github.io/cookbook/code/media/mug-counter.jpg)。

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

每份快照包含三份文件，可参考示例库里的 `first/`。机器人适配器也可把同步好的 `RGBDFrame` 和到访 ID 传给 `store.record()`。

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
cd /rm_nvme/recordings/user_home/Find-Anything-Like-Humans-V2/models/Valen-Decision-1.5-0.8B-GGUF
./serve-local.sh
```

每次运行在 `falhv2/outputs/searches/<唯一编号>/` 保存 `result.json`。也可用 `--output` 指定新目录。结果区分 `found_latest`、`found_historical`、`not_found_in_snapshots` 和 `no_snapshots`，另列未访问视角、各张图的分数和确认结果。命中时附原图和标框图、拍摄位姿和时间。框中央区域的深度有效且一致时，按相机内参和实测位姿给出地图系近似表面点，其他情况记录 null。
