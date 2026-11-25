# 模型客户端

已接入 [Cubify Anything](https://github.com/apple-aiml-research/ml-cubifyanything) 的 **CuTR RGB-D** 模型。
本地和 HTTP 客户端统一实现 `base.CuboidDetector`；`InstanceSegmenter` 保留为独立的 mask 接口。
客户端通过 Python API 使用。离线锚点规划直接读取地图快照，无需加载模型。

每个模型使用一个 Python 文件，与旧 `anchor-map/model_clients` 的组织方式一致：
`cubify_anything.py` 包含 Local / Remote 客户端及该模型专用的协议、几何处理；
`base.py` 定义统一接口，`__init__.py` 提供构造入口。

## 配置

调用方将参数字典传给 `build_cubify_anything`。
本地客户端使用 `path`、`source_path`、`device`、`max_side`、`score_threshold` 和 `fp16`。
远程客户端使用 `url`、`timeout` 和 `score_threshold`。

## 本地调用

在仓库根目录、具备 CuTR 依赖的 Python 环境中：

```python
from falhv2.model_clients import build_cubify_anything
from falhv2.robot_adapters.anchor_map import from_anchor_map_frame

client = build_cubify_anything(model_config)  # model_config 由调用方传入
try:
    # legacy_frame 是 anchor-map/frame_source.py 的 Frame。
    # map_id / map_revision 必须由 SLAM 会话提供，不可用常量掩盖回环修正。
    frame = from_anchor_map_frame(
        legacy_frame, map_id=current_map_id, map_revision=current_map_revision)
    result = client.detect(frame)
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

远程客户端连接已部署的兼容服务：

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
- legacy Frame 的 stamp 是 UNIX 秒浮点数；转换回纳秒不能恢复其已丢失的精度。

## 验证

```bash
python -B -m unittest falhv2.tests.test_cubify_clients -v
```

测试覆盖 RGB/深度转换、位姿与重力、地图 AABB、损坏或错帧的协议消息和配置校验。

### Confidence and precision

`score_threshold` (default `0.25`) keeps scores greater than or equal to the threshold in both local and remote clients. `fp16` (default `True`) enables CUDA FP16 autocast; set it to `False` for FP32. CPU always uses FP32. Outputs are converted to FP32 before NumPy conversion. The remote inference service controls its own precision. Restart the process after changing configuration. Ground filtering is not enabled by these settings.
