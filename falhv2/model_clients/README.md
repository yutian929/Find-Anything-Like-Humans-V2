# 模型客户端

已接入 [Cubify Anything](https://github.com/apple-aiml-research/ml-cubifyanything) 的 **CuTR RGB-D** 模型。
本地和 HTTP 客户端统一实现 `base.CuboidDetector`；`InstanceSegmenter` 保留为独立的 mask 接口。
默认 `python falhv2/main.py` 仍执行原有离线锚点选择流程，不导入 torch、不连接相机、不加载模型。

每个模型使用一个 Python 文件，与旧 `anchor-map/model_clients` 的组织方式一致：
`cubify_anything.py` 包含 Local / Remote 客户端及该模型专用的协议、几何处理；
`base.py` 定义统一接口，`__init__.py` 提供构造入口。HTTP 服务启动器仍在 `tools/cubify_server.py`。

## 配置

默认使用 `falhv2/config/defaults.py` 中的一个 `CUBIFY` 字典，结构与旧项目的 RYNNBRAIN / SAM2 / SAM3 相同。
本地模式读取 path、source_path、device、max_side、score_threshold；远程模式读取 url（含端口）、timeout、score_threshold。
切换远程只需修改该字典中的 mode 和 url，本地字段可以保留；远程推理的权重、设备和输入尺寸以服务端机器的配置为准。
只有 main.py 加载配置，再调用 `build_cubify_anything(model_config)`。客户端不 import 配置，也不自行选择配置文件。
预览和服务启动器接收 main.py 传入的该字典；服务使用其中的本地参数，并默认使用 url 中的端口，始终本地加载模型。
服务绑定地址仍由 --host 控制（默认 127.0.0.1），跨机访问可用 --host 0.0.0.0；--port 可临时覆盖监听端口。
显示框数、点云采样等保留在工具内部，不另设配置块。默认 main.py 仍运行离线锚点选择。

可通过 `python falhv2/main.py --config /path/to/robot.py --task cubify-preview` 选择另一份 Python 配置。
配置文件可以只定义一个 CUBIFY 字典：未指定的字段沿用默认值，命令行参数再覆盖本次调用。

## 本地调用

在仓库根目录、具备 CuTR 依赖的 Python 环境中：

```python
from falhv2.model_clients import build_cubify_anything
from falhv2.robot_adapters.anchor_map import from_anchor_map_frame
from falhv2.map_maintainers.cubify import cuboids_to_map_aabbs

client = build_cubify_anything(model_config)  # model_config 由 main.py 传入
try:
    # legacy_frame 是 anchor-map/frame_source.py 的 Frame。
    # map_id / map_revision 必须由 SLAM 会话提供，不可用常量掩盖回环修正。
    frame = from_anchor_map_frame(
        legacy_frame, map_id=current_map_id, map_revision=current_map_revision)
    result = client.detect(frame)
    observations = cuboids_to_map_aabbs(frame, result)
finally:
    client.close()
```

Lenovo 已找到上述权重（396866874 字节）和源码；`depth` conda 环境已验证可加载并运行。
该源码目录是已有项目保留的 Cubify 包（含优化修改），并非本次下载；
客户端只引用其中的 `cubifyanything` 包，不引用旧项目业务代码。
也可将 `source_path` 指向其他完整上游源码根目录；若已安装该包，可省略。
路径不存在、RGB-only 权重、缺失依赖或非法配置会直接报错，不自动下载或静默切换 backend。
模型版本记录为权重文件的 SHA-256；部署时还应固定所用源码版本。

只使用远程客户端或运行轻量接口测试：
```bash
python -m pip install -r falhv2/model_clients/requirements.txt
```
本地推理还需要匹配设备的 PyTorch / torchvision、timm 及所用 CuTR 源码的依赖。
已有 Lenovo `depth` 环境可直接使用；不要为离线锚点规划强制安装 GPU 依赖。

## 远程调用

在模型所在机器的仓库根目录启动（默认仅监听本机）：
```bash
conda run -n depth python falhv2/main.py --task cubify-server \
  --weights /home/hri/models_weights/cubifyanything/cutr_rgbd.pth \
  --source-path /home/hri/map_test_new_v8_clean \
  --device cuda --port 8093
```
跨机器部署时按实际网络设置 `--host`，或使用 SSH 转发；此启动器供受信任网络中的开发调试使用，
没有实现公网认证。停止服务会释放客户端，不保留后台进程。

机器人侧仅更换配置，`detect(frame)` 与结果类型不变：
```python
client = build_cubify_anything({
    "mode": "remote",
    "url": "http://127.0.0.1:8093",
    "timeout": 60,
    "score_threshold": 0.25,
})
```

协议 v1：`GET /healthz`、`POST /v1/cubify/detect`。
JSON 请求包含 RGB PNG、base64 little-endian float32 米制深度、内参、采集时刻位姿、帧与地图版本。
响应包含 frame_id、timestamp_ns、camera_frame、model_id、model_version、scores 与 Nx8x3 corners_camera_m。
客户端核对帧标识、相机坐标系、时间戳、模型标识、置信度和角点有效性，拒绝错帧响应。
超时、服务不可用、推理异常均抛错；成功空结果仅表示该帧没有达到阈值的预测。
HTTP 超时不会取消服务端已开始的 GPU 推理；本地调用是同步的。

## 数据与几何约定

- 输入 RGB 是 HxWx3 uint8 **RGB**，深度是对齐、去畸变后的 optical-Z 米制浮点数。
  legacy 适配器处理 BGR→RGB、z16×depth_scale；不负责对齐/去畸变。
  无效深度置零，全帧无有效深度报错。
- 相机光学坐标：右、下、前；地图 +Z 必须与物理向上方向一致。
  使用采集时刻的 T_map_camera 提取重力倾斜，移除地图 yaw 的影响。
  当前要求图像正立；侧置、倒置或垂直朝上/下的相机须先校正图像、标定与位姿，否则明确报错。
- 客户端缩放 RGB 到 max_side 内、尺寸对齐到 4；深度使用四分之一分辨率。
  内参按真实横纵缩放比例更新。归一化交给上游 Preprocessor。
- 输出是**无类别有向 3D cuboid** 的八个角点，单位米、原相机光学坐标系，
  尚不是地图 AABB。不得对上游已应用重力旋转的输出再旋转一次。
- `cuboids_to_map_aabbs` 先把所有角点变换到 map，再逐轴取 min/max；
  不能先取相机系 AABB 后只变换两个极值点。
- AABBObservation 仅是单帧观测，没有跨帧稳定物体 ID。
  已通过 main.py --task build-map 实现深度约束的旧框替换、栅格读取和规划地图导出；不做关联/融合，尚不支持回环重投影。
- legacy Frame 的 stamp 是 UNIX 秒浮点数；转换回纳秒不能恢复其已丢失的精度。

## 验证

```bash
python -B -m unittest falhv2.tests.test_cubify_clients -v
```

覆盖 RGB/深度转换、位姿与重力、地图 AABB、损坏/错帧响应、真实 localhost HTTP 传输和异常。
HTTP 单测用可控假模型，不要求权重。Lenovo 的真实权重另已用合成 RGB-D 完成 GPU 推理及真实 HTTP 链路冒烟检查（641x479 非整倍数图像，本地/远程输出一致）；
这不代表已验证真实相机上的检测精度。默认规划入口不使用这些模型配置。

## 当前相机帧可视化

在仓库根目录运行（Lenovo 的 depth 环境已有所需 cv2 / plotly）：
```bash
conda run -n depth python falhv2/main.py --task cubify-preview
```
工具先加载模型，再读取一帧实时同步 RGB-D，生成原图与 3D 框投影对照 PNG、
可旋转的实测点云与预测框 HTML，以及预测 JSON 和原始 RGB-D NPZ。
结果保存在 falhv2/outputs/cubify_时间戳.*；HTML 内嵌 Plotly，可离线打开。
编号按置信度降序排列，默认阈值 0.25，最多显示 20 个框；JSON 保留全部过阈值预测。
可用 --threshold、--max-boxes 调整；--server-url 可改用已部署的模型服务。
该预览使用相机坐标，不进行地图融合或给物体分配跨帧 ID。

需要同步 RGB-D 发布器和底盘网络均正常；底盘定位超时会让发布器停止发布有效帧，
工具会明确报错，不使用旧帧、合成帧或猜测位姿生成效果图。
