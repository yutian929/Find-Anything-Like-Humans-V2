"""对齐帧数据源（替代原三节点里的 node1 publisher）。

数据获取改为「直接读取 + HTTP」，不再单独起进程。本文件包含：
1. 数据契约：Frame（rgb/depth/位姿/内参/外参）与 OccupancyMap（占据栅格底图）。
2. FrameSource 接口：Engine 只依赖 latest() / occupancy_map() / healthz()。
3. ChassisClient：Slamtec 底盘 REST 客户端（位姿 + explore 栅格地图）。
4. RealSenseChassisFrameSource：真实数据源
   —— RealSense RGB-D（ZMQ SUB，配合 /home/hri/camera/realsense_publisher.py）
      + 底盘位姿（REST）合成 Frame；explore 栅格转 OccupancyMap 给 UI 底图。

坐标/单位约定见 Frame 文档。相机->底盘静态外参 cam_tf 暂留 None（geometry 回退
config.CAM2BASE_* 静态兜底），待提供后可在此填入随帧下发。
"""
from __future__ import annotations

import hashlib
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

import config

logger = logging.getLogger("frame_source")


# =========================================================================== #
# 数据契约
# =========================================================================== #
@dataclass
class Frame:
    """一帧对齐数据。

    - rgb:        BGR, HxWx3, uint8（与 cv2.imread 一致；下游按 BGR 处理）。
    - depth:      uint16, 与 rgb 同分辨率、已对齐到彩色（毫米，尺度在 intrinsics.depth_scale）。
    - pose2d:     底盘在地图系的 2D 位姿 {x, y, yaw}（米/弧度）。
    - intrinsics: {fx, fy, cx, cy, depth_scale}（彩色相机内参）。
    - cam_tf:     相机外参 {xyz_m:[x,y,z], quat_wxyz:[w,x,y,z]}（光学系->底盘）；
                  None 时 geometry 回退 config.CAM2BASE_* 静态兜底。
    - stamp:      帧时间戳（秒）。
    """
    rgb: np.ndarray
    depth: np.ndarray
    pose2d: Dict[str, float]
    intrinsics: Dict[str, float]
    cam_tf: Optional[Dict[str, Any]] = None
    stamp: float = 0.0


@dataclass
class OccupancyMap:
    png_bytes: bytes
    resolution: float
    origin: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    width: int = 0
    height: int = 0
    version: str = ""


class FrameSource:
    """对齐帧数据源接口。子类至少实现 latest()。"""

    def latest(self) -> Optional[Frame]:
        raise NotImplementedError

    def occupancy_map(self) -> Optional[OccupancyMap]:
        return None

    def healthz(self) -> Dict[str, Any]:
        f = self.latest()
        return {"connected": f is not None, "has_frame": f is not None,
                "has_map": self.occupancy_map() is not None}


class PlaceholderFrameSource(FrameSource):
    """占位实现：不产出数据。真实数据源不可用时兜底，使系统能空跑。"""

    def latest(self) -> Optional[Frame]:
        return None

    def healthz(self) -> Dict[str, Any]:
        return {"connected": False, "has_frame": False, "has_map": False,
                "note": "PlaceholderFrameSource：真实数据源不可用"}


# =========================================================================== #
# Slamtec 底盘 REST 客户端
# =========================================================================== #
class ChassisClient:
    """Slamtec Slamware REST API（实测 Hermes，端口 1448）。"""

    def __init__(self, base_url: str, timeout: float = 2.0):
        import requests
        self._requests = requests
        self.base = str(base_url).rstrip("/")
        self.timeout = float(timeout)

    def _get(self, path: str):
        r = self._requests.get(f"{self.base}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r

    def pose(self) -> Dict[str, float]:
        """GET /localization/pose -> {x, y, z, yaw, pitch, roll}（地图系，米/弧度）。"""
        d = self._get("/api/core/slam/v1/localization/pose").json()
        return {"x": float(d["x"]), "y": float(d["y"]), "yaw": float(d["yaw"])}

    def localization_quality(self) -> Optional[int]:
        try:
            return int(self._get("/api/core/slam/v1/localization/quality").json())
        except Exception:
            return None

    def explore_map(self) -> Optional[OccupancyMap]:
        """GET /maps/explore -> 解析二进制栅格并渲染为灰度 PNG 底图。

        二进制布局：origin_x(f32) origin_y(f32) width(i32) height(i32) resolution(f32)
        + 填充，随后 width*height 个栅格字节（0=空闲，127=未知，其余=占据/代价）。
        """
        raw = self._get("/api/core/slam/v1/maps/explore").content
        if len(raw) < 20:
            return None
        ox, oy, w, h, res = struct.unpack_from("<ffiif", raw, 0)
        if w <= 0 or h <= 0 or len(raw) < w * h:
            return None
        body = np.frombuffer(raw[len(raw) - w * h:], dtype=np.uint8).reshape(h, w)

        # 三值灰度：空闲=白(254)、未知=灰(205)、占据=黑(0)；上下翻转到图像行序
        gray = np.full((h, w), 205, dtype=np.uint8)
        gray[body == 0] = 254
        gray[(body != 0) & (body != 127)] = 0
        gray = np.flipud(gray)

        import cv2
        ok, buf = cv2.imencode(".png", gray)
        if not ok:
            return None
        png = buf.tobytes()
        return OccupancyMap(png_bytes=png, resolution=float(res), origin=[float(ox), float(oy), 0.0],
                            width=int(w), height=int(h), version=hashlib.md5(png).hexdigest()[:12])


# =========================================================================== #
# 真实数据源：RealSense(ZMQ) + 底盘(REST)
# =========================================================================== #
class RealSenseChassisFrameSource(FrameSource):
    """两条后台线程：ZMQ 收 RGB-D、REST 轮询位姿+底图；latest() 合成 Frame。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._img: Optional[Dict[str, Any]] = None      # {bgr, depth, intr, stamp}
        self._pose: Optional[Dict[str, float]] = None
        self._map: Optional[OccupancyMap] = None
        self._running = True

        self._chassis = ChassisClient(config.CHASSIS_URL, config.CHASSIS_TIMEOUT_S)

        threading.Thread(target=self._zmq_loop, name="fs-zmq", daemon=True).start()
        threading.Thread(target=self._chassis_loop, name="fs-chassis", daemon=True).start()
        logger.info("RealSenseChassisFrameSource: zmq=%s topic=%s chassis=%s",
                    config.FRAME_ZMQ_ENDPOINT, config.FRAME_ZMQ_TOPIC, config.CHASSIS_URL)

    # ---- RealSense ZMQ 接收（4 段 multipart：topic/meta/rgb/depth）---- #
    def _zmq_loop(self) -> None:
        import msgpack
        import zmq
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, config.FRAME_ZMQ_TOPIC.encode("utf-8"))
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.connect(config.FRAME_ZMQ_ENDPOINT)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while self._running:
            if not dict(poller.poll(timeout=500)):
                continue
            parts = sock.recv_multipart()
            if len(parts) != 4:
                continue
            try:
                _, meta_b, rgb_b, depth_b = parts
                meta = msgpack.unpackb(meta_b, raw=False)
                h, w = int(meta["height"]), int(meta["width"])
                rgb = np.frombuffer(rgb_b, dtype=np.uint8).reshape(h, w, 3)
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])   # 发布端为 RGB，转 BGR 供下游
                depth = np.frombuffer(depth_b, dtype=np.uint16).reshape(h, w).copy()
                ci = meta.get("color_intr", {})
                intr = {
                    "fx": float(ci.get("fx", w)), "fy": float(ci.get("fy", w)),
                    "cx": float(ci.get("ppx", w / 2.0)), "cy": float(ci.get("ppy", h / 2.0)),
                    "depth_scale": float(meta.get("depth_scale_m", config.DEFAULT_DEPTH_SCALE)),
                }
                stamp = float(meta.get("color_hw_ts_ms", 0.0)) / 1000.0 or time.time()
                with self._lock:
                    self._img = {"bgr": bgr, "depth": depth, "intr": intr, "stamp": stamp}
            except Exception:
                logger.exception("zmq frame parse failed")

    # ---- 底盘位姿 + 底图轮询 ---- #
    def _chassis_loop(self) -> None:
        period = 1.0 / max(1.0, float(config.CHASSIS_POSE_HZ))
        last_map = 0.0
        while self._running:
            try:
                pose = self._chassis.pose()
                with self._lock:
                    self._pose = pose
            except Exception as exc:
                logger.debug("chassis pose failed: %s", exc)
            now = time.time()
            if now - last_map >= float(config.CHASSIS_MAP_REFRESH_S):
                last_map = now
                try:
                    occ = self._chassis.explore_map()
                    if occ is not None:
                        with self._lock:
                            self._map = occ
                except Exception as exc:
                    logger.debug("chassis explore_map failed: %s", exc)
            time.sleep(period)

    # ---- 对外 ---- #
    def latest(self) -> Optional[Frame]:
        with self._lock:
            img, pose = self._img, self._pose
        if img is None or pose is None:
            return None
        return Frame(rgb=img["bgr"], depth=img["depth"], pose2d=dict(pose),
                     intrinsics=dict(img["intr"]), cam_tf=None, stamp=img["stamp"])

    def occupancy_map(self) -> Optional[OccupancyMap]:
        with self._lock:
            return self._map

    def healthz(self) -> Dict[str, Any]:
        with self._lock:
            return {"connected": self._img is not None and self._pose is not None,
                    "has_frame": self._img is not None, "has_pose": self._pose is not None,
                    "has_map": self._map is not None,
                    "quality": self._chassis.localization_quality()}

    def close(self) -> None:
        self._running = False


# =========================================================================== #
# 工厂
# =========================================================================== #
def build_frame_source() -> FrameSource:
    """构建真实数据源（RealSense + 底盘）；依赖缺失/构建失败时回退占位。

    真实数据源对断连是容错的：ZMQ 无帧、底盘不可达时 latest() 返回 None，系统空转。
    """
    try:
        return RealSenseChassisFrameSource()
    except Exception:
        logger.exception("真实数据源构建失败，回退 PlaceholderFrameSource")
        return PlaceholderFrameSource()
