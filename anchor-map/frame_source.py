"""对齐帧数据源（替代原三节点里的 node1 publisher）。

数据获取方式改为「直接读取 + HTTP」，不再单独起进程。本文件定义两件事：
1. 数据契约：Frame（一帧对齐的 rgb/depth/位姿/内参/外参）与 OccupancyMap（占据栅格底图）。
2. FrameSource 接口：Engine 只依赖 latest() / occupancy_map() / healthz()。

★ 真正的采集逻辑由你后续补全：继承 FrameSource 实现 latest() 即可
  （从固定路径读、或向机器人 HTTP 拉取，做好 rgb/depth/位姿时间对齐）。
默认提供的 PlaceholderFrameSource 不产出数据（latest() 返回 None），
使整套系统能先空跑起来（MAPPING 态空转、UI 可访问）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Frame:
    """一帧对齐数据。坐标/单位约定：

    - rgb:        BGR, HxWx3, uint8（与 cv2.imread 一致）。
    - depth:      uint16, 与 rgb 同分辨率（毫米，depth_scale 在 intrinsics 里）。
    - pose2d:     底盘在地图系的 2D 位姿 {x, y, yaw}（米/弧度）。
    - intrinsics: {fx, fy, cx, cy, depth_scale}（彩色相机内参）。
    - cam_tf:     动态相机外参 {xyz_m:[x,y,z], quat_wxyz:[w,x,y,z]}（光学系->底盘）；
                  为 None 时 geometry 回退 config.CAM2BASE_* 静态兜底。
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
    """占据栅格底图（供 3D 地图页画底图）。

    - png_bytes:  地图灰度/三值 PNG 的原始字节（直接给前端 <img>）。
    - resolution: 米/像素。
    - origin:     地图左下角在世界系的 (x, y[, z])。
    - version:    版本号；变化时前端才重新拉图。
    """
    png_bytes: bytes
    resolution: float
    origin: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    width: int = 0
    height: int = 0
    version: str = ""


class FrameSource:
    """对齐帧数据源接口。子类至少实现 latest()。"""

    def latest(self) -> Optional[Frame]:
        """返回当前最新一帧；暂无数据时返回 None。"""
        raise NotImplementedError

    def occupancy_map(self) -> Optional[OccupancyMap]:
        """返回当前占据栅格底图；无则 None。"""
        return None

    def healthz(self) -> Dict[str, Any]:
        """数据就绪状态，供 UI 状态栏展示。"""
        frame = self.latest()
        return {
            "connected": frame is not None,
            "has_frame": frame is not None,
            "has_map": self.occupancy_map() is not None,
        }


class PlaceholderFrameSource(FrameSource):
    """占位实现：不产出数据。用于先把整套系统空跑起来，待你替换为真实采集。"""

    def latest(self) -> Optional[Frame]:
        return None

    def occupancy_map(self) -> Optional[OccupancyMap]:
        return None

    def healthz(self) -> Dict[str, Any]:
        return {"connected": False, "has_frame": False, "has_map": False,
                "note": "PlaceholderFrameSource：请实现真实数据源"}


def build_frame_source() -> FrameSource:
    """工厂：当前返回占位数据源。补全真实采集后在此替换。"""
    return PlaceholderFrameSource()
