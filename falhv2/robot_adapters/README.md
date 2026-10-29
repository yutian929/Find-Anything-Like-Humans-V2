# 机器人数据适配

当前实现位于已有的 `anchor_map.py`：`AnchorMapRobot` 订阅 Lenovo 上的 RGB-D/pose 发布器，
并通过 Slamtec REST 获取栅格地图和底盘定位。不会启动相机、发布器或发送运动指令。

在仓库根目录调用：
```python
from falhv2.robot_adapters.anchor_map import AnchorMapRobot

with AnchorMapRobot(
    map_id="your-slam-map",
    map_revision="your-coordinate-revision",
    endpoint="ipc:///tmp/rgbd_pose.ipc",
    chassis_url="http://192.168.11.1:1448",
) as robot:
    frame = robot.capture(timeout_s=2)
    grid = robot.grid_snapshot(timeout_s=2)
    pose = robot.localization(timeout_s=2)

    # frame.rgb: uint8 RGB; frame.depth_m: 对齐的米制 optical-Z 深度
    # frame.T_map_camera / frame.T_map_base: 同一帧采集时刻的同步位姿
    # grid.rows: 从上到下，'.' 自由、'#' 占用、'?' 未知
    # grid.origin: 左下角米制坐标；grid.resolution: 米/格
    print(pose.x, pose.y, pose.yaw)  # 米、米、弧度
```

`capture` 使用协议 v2，原样保留 UNIX 纳秒时间戳，校验编码、尺寸、内参、
深度对齐和同步位姿；拒绝全无效深度、非零未校正畸变、坏帧及过期帧。
默认允许帧龄 1 秒、未来时间偏差最多 0.1 秒；跨机使用须同步系统时钟。
异常帧会跳过并等待后续有效帧，超过 timeout_s 后抛出带最后错误原因的 TimeoutError。

`grid_snapshot` 每次获取最新地图，不永久缓存；接收现有 20 字节头及实机 36 字节头格式，
验证长度，将底部起始栅格翻转成规划器的顶部起始行。未知区域保留为 '?'。
`content_version` 是响应内容 SHA-256，能检测栅格变化，**不是 SLAM 坐标版本**。
timestamp_ns 是接收时间；REST 没有提供地图采集时间。

`localization` 独立调用底盘定位接口，即使相机没有数据也可使用。
它返回完整 T_map_base 和便捷 x/y/yaw。REST 没有测量时间戳或定位置信度，
所以 timestamp_source 标为 http_midpoint，并保留 rtt_s；不能用它替换 frame 内的同步位姿。
HTTP 错误、非法数据会直接抛错，不回退到单位矩阵或旧地图。
HTTP timeout_s 是连接/读取超时；RGB-D timeout_s 是等待有效帧的总时限。

当前发布协议和底盘 REST 都没有 SLAM map_id/map_revision，因此构造时必须由调用方提供。
地图切换、重定位坐标重置或回环修正后，应关闭旧实例并用新的坐标版本重新创建；
此适配器不能自动识别这些事件。三次独立读取也不是原子地图快照。

依赖（不需要 RealSense SDK 或 torch）：
```bash
python -m pip install numpy pyzmq msgpack requests
python -B -m unittest falhv2.tests.test_robot_adapter -v
```

open/读取/close 应在同一线程调用；构造对象和导入模块不会连接硬件。
with 退出会关闭 ZMQ socket/context 和 HTTP session。REST 直连底盘，不继承系统代理。
原有 `from_anchor_map_frame` 转换函数仍可用于旧 Frame，也保留其中可用的 base_pose。

其他机器人以后在本目录按实际通信协议实现 `base.RobotSource`，
保持 capture/grid_snapshot/localization 的数据约定即可，不需要改模型或锚点算法。
