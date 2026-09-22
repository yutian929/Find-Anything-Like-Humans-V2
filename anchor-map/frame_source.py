"""同步 RGB-D/pose 数据源与独立的一次性占据栅格加载。

帧协议见 ``docs/README_rgbd_pose_protocol.md``。发布端已经完成相机时间轴与
底盘位姿的同步，并在同一条 ZMQ multipart 消息中发布 ``T_map_camera``；本模块
只校验、解码和缓存该原子观测，不再向底盘 REST API 单独轮询 pose。

占据栅格不属于逐帧观测。它由独立后台任务从底盘 REST API 加载：启动时尝试，
失败则低频重试，首次成功后停止请求并在内存中长期缓存。
"""
from __future__ import annotations

import hashlib
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

import config

logger = logging.getLogger("frame_source")


@dataclass(frozen=True)
class Frame:
    """一个由发布端完成时间同步的 RGB-D + pose 原子观测。

    ``camera_pose`` 是协议中的 ``T_map_camera``（相机光学系到地图系）。下游应
    直接使用它将相机点变换到 map，不能再用另一时刻的底盘 pose 重算。
    """

    rgb: np.ndarray                         # BGR, HxWx3, uint8
    depth: np.ndarray                       # HxW, uint16，已对齐到 RGB
    camera_pose: np.ndarray                 # T_map_camera, 4x4
    intrinsics: Dict[str, float]            # fx/fy/cx/cy/depth_scale
    stamp: float                            # timestamp_ns 转换的 Unix 秒
    frame_id: int
    base_pose: Optional[np.ndarray] = None   # T_map_base, 4x4（诊断用）
    sync: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OccupancyMap:
    png_bytes: bytes
    resolution: float
    origin: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    width: int = 0
    height: int = 0
    version: str = ""


class FrameSource:
    def latest(self) -> Optional[Frame]:
        raise NotImplementedError

    def occupancy_map(self) -> Optional[OccupancyMap]:
        return None

    def healthz(self) -> Dict[str, Any]:
        frame = self.latest()
        return {
            "connected": frame is not None,
            "has_frame": frame is not None,
            "has_map": self.occupancy_map() is not None,
        }


class PlaceholderFrameSource(FrameSource):
    def latest(self) -> Optional[Frame]:
        return None

    def healthz(self) -> Dict[str, Any]:
        return {
            "connected": False,
            "has_frame": False,
            "has_map": False,
            "note": "PlaceholderFrameSource：真实数据源不可用",
        }


def _matrix4(value: Any, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{field_name} 必须是有限的 4x4 矩阵")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{field_name} 最后一行不是齐次变换格式")
    return matrix


def decode_rgbd_pose_message(parts: Sequence[bytes]) -> Frame:
    """校验并解码一条协议 v1 multipart 消息，便于独立测试协议适配。"""
    import msgpack

    if len(parts) != 4:
        raise ValueError(f"multipart 段数应为 4，实际为 {len(parts)}")
    topic_b, meta_b, rgb_b, depth_b = parts
    expected_topic = config.FRAME_ZMQ_TOPIC.encode("utf-8")
    if topic_b != expected_topic:
        raise ValueError(f"topic 不匹配：{topic_b!r}")

    meta = msgpack.unpackb(meta_b, raw=False)
    if not isinstance(meta, dict):
        raise ValueError("metadata 必须是 map")
    if int(meta.get("version", -1)) != int(config.FRAME_PROTOCOL_VERSION):
        raise ValueError(f"不支持的协议版本：{meta.get('version')!r}")
    if meta.get("type") != "rgbd_pose":
        raise ValueError(f"不支持的消息类型：{meta.get('type')!r}")
    if not bool(meta.get("pose_valid")):
        reason = (meta.get("sync") or {}).get("reason", "unknown")
        raise ValueError(f"当前帧没有有效同步位姿：{reason}")

    height, width = int(meta["height"]), int(meta["width"])
    if height <= 0 or width <= 0:
        raise ValueError(f"非法图像尺寸：{width}x{height}")
    rgb_dtype = np.dtype(meta.get("rgb_dtype", "uint8"))
    depth_dtype = np.dtype(meta.get("depth_dtype", "uint16"))
    if rgb_dtype != np.dtype("uint8") or depth_dtype != np.dtype("uint16"):
        raise ValueError(f"不支持的数据类型：rgb={rgb_dtype}, depth={depth_dtype}")

    rgb = np.frombuffer(rgb_b, dtype=rgb_dtype)
    depth = np.frombuffer(depth_b, dtype=depth_dtype)
    if rgb.size != height * width * 3 or depth.size != height * width:
        raise ValueError("图像字节数与 metadata 尺寸不一致")
    rgb = rgb.reshape(height, width, 3)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    depth = depth.reshape(height, width).copy()

    color_intr = meta.get("color_intr") or {}
    intrinsics = {
        "fx": float(color_intr["fx"]),
        "fy": float(color_intr["fy"]),
        "cx": float(color_intr["ppx"]),
        "cy": float(color_intr["ppy"]),
        "depth_scale": float(meta["depth_scale_m"]),
    }
    if not all(np.isfinite(v) for v in intrinsics.values()):
        raise ValueError("相机内参包含非有限值")
    if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0 or intrinsics["depth_scale"] <= 0:
        raise ValueError("相机焦距和深度尺度必须大于 0")

    timestamp_ns = int(meta["timestamp_ns"])
    if timestamp_ns <= 0:
        raise ValueError("timestamp_ns 必须大于 0")
    camera_pose = _matrix4(meta.get("T_map_camera"), "T_map_camera")
    base_raw = meta.get("T_map_base")
    base_pose = _matrix4(base_raw, "T_map_base") if base_raw is not None else None

    return Frame(
        rgb=bgr,
        depth=depth,
        camera_pose=camera_pose,
        intrinsics=intrinsics,
        stamp=timestamp_ns / 1_000_000_000.0,
        frame_id=int(meta["frame_id"]),
        base_pose=base_pose,
        sync=dict(meta.get("sync") or {}),
    )


class ChassisMapClient:
    """Slamtec REST 客户端，仅负责获取 explore 占据栅格。"""

    def __init__(self, base_url: str, timeout: float = 2.0):
        import requests

        self._session = requests.Session()
        # 底盘是局域网直连设备，不能继承 HTTP_PROXY/HTTPS_PROXY。
        self._session.trust_env = False
        self.base = str(base_url).rstrip("/")
        self.timeout = float(timeout)

    def explore_map(self) -> OccupancyMap:
        response = self._session.get(
            f"{self.base}/api/core/slam/v1/maps/explore", timeout=self.timeout
        )
        response.raise_for_status()
        raw = response.content
        if len(raw) < 20:
            raise ValueError("explore map 响应过短")
        origin_x, origin_y, width, height, resolution = struct.unpack_from("<ffiif", raw, 0)
        cell_count = int(width) * int(height)
        map_numbers = (origin_x, origin_y, resolution)
        if (
            width <= 0
            or height <= 0
            or not all(np.isfinite(v) for v in map_numbers)
            or resolution <= 0
            or len(raw) < 20 + cell_count
        ):
            raise ValueError("explore map 元数据或栅格长度非法")

        body = np.frombuffer(raw[-cell_count:], dtype=np.uint8).reshape(height, width)
        gray = np.full((height, width), 205, dtype=np.uint8)
        gray[body == 0] = 254
        gray[(body != 0) & (body != 127)] = 0
        gray = np.flipud(gray)

        import cv2

        ok, encoded = cv2.imencode(".png", gray)
        if not ok:
            raise ValueError("占据栅格 PNG 编码失败")
        png = encoded.tobytes()
        return OccupancyMap(
            png_bytes=png,
            resolution=float(resolution),
            origin=[float(origin_x), float(origin_y), 0.0],
            width=int(width),
            height=int(height),
            version=hashlib.md5(png).hexdigest()[:12],
        )


class RGBDPoseFrameSource(FrameSource):
    """缓存最新同步观测，并与一次性加载的地图分别维护。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._map: Optional[OccupancyMap] = None
        self._last_frame_monotonic: Optional[float] = None
        self._invalid_frames = 0
        self._last_frame_error: Optional[str] = None
        self._map_attempts = 0
        self._map_error: Optional[str] = None
        self._stop = threading.Event()
        self._map_client = ChassisMapClient(config.CHASSIS_URL, config.CHASSIS_TIMEOUT_S)

        threading.Thread(target=self._zmq_loop, name="fs-rgbd-pose", daemon=True).start()
        threading.Thread(target=self._load_map_once, name="fs-map-once", daemon=True).start()
        logger.info(
            "RGBDPoseFrameSource: zmq=%s topic=%s map=%s (load once)",
            config.FRAME_ZMQ_ENDPOINT,
            config.FRAME_ZMQ_TOPIC,
            config.CHASSIS_URL,
        )

    def _zmq_loop(self) -> None:
        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, config.FRAME_ZMQ_TOPIC.encode("utf-8"))
        socket.setsockopt(zmq.RCVHWM, 2)
        socket.connect(config.FRAME_ZMQ_ENDPOINT)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if not dict(poller.poll(timeout=500)):
                    continue
                try:
                    frame = decode_rgbd_pose_message(socket.recv_multipart())
                except Exception as exc:
                    with self._lock:
                        self._invalid_frames += 1
                        self._last_frame_error = str(exc)
                    logger.warning("rgbd_pose frame rejected: %s", exc)
                    continue
                with self._lock:
                    self._frame = frame
                    self._last_frame_monotonic = time.monotonic()
                    self._last_frame_error = None
        finally:
            socket.close(linger=0)

    def _load_map_once(self) -> None:
        retry = max(0.1, float(config.MAP_LOAD_RETRY_S))
        while not self._stop.is_set():
            with self._lock:
                self._map_attempts += 1
            try:
                occupancy_map = self._map_client.explore_map()
            except Exception as exc:
                with self._lock:
                    self._map_error = str(exc)
                logger.warning("occupancy map load failed; retry in %.1fs: %s", retry, exc)
                self._stop.wait(retry)
                continue
            with self._lock:
                self._map = occupancy_map
                self._map_error = None
            logger.info(
                "occupancy map cached: %dx%d resolution=%.4f version=%s",
                occupancy_map.width,
                occupancy_map.height,
                occupancy_map.resolution,
                occupancy_map.version,
            )
            return

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._frame

    def occupancy_map(self) -> Optional[OccupancyMap]:
        with self._lock:
            return self._map

    def healthz(self) -> Dict[str, Any]:
        with self._lock:
            frame = self._frame
            received_at = self._last_frame_monotonic
            occupancy_map = self._map
            result = {
                "connected": frame is not None,
                "has_frame": frame is not None,
                "has_pose": frame is not None,
                "has_map": occupancy_map is not None,
                "frame_id": frame.frame_id if frame is not None else None,
                "frame_timestamp": frame.stamp if frame is not None else None,
                "frame_age_s": (
                    round(time.monotonic() - received_at, 3) if received_at is not None else None
                ),
                "invalid_frames": self._invalid_frames,
                "last_frame_error": self._last_frame_error,
                "map_load_attempts": self._map_attempts,
                "map_error": self._map_error,
            }
        return result

    def close(self) -> None:
        self._stop.set()


def build_frame_source() -> FrameSource:
    try:
        return RGBDPoseFrameSource()
    except Exception:
        logger.exception("真实数据源构建失败，回退 PlaceholderFrameSource")
        return PlaceholderFrameSource()
