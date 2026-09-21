"""坐标变换工具（自 anchor-map-v2 原样迁移）。

核心链路：T_cam2map = T_base2map(x, y, yaw) @ T_cam2base
- T_base2map：底盘 2D Pose（x, y, yaw）绕 Z 转 yaw、平移 (x, y, 0)。
- T_cam2base：优先用随帧下发的动态外参（{xyz_m, quat_wxyz}，光学系->底盘）；
  缺省时回退 config.CAM2BASE_* 静态兜底值。
- 不再乘 R_opt2cam：动态/静态四元数均已包含「光学系->机体系」轴变换。
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

import config

_CAM2BASE_CACHE: np.ndarray | None = None


def quat_wxyz_to_R(q: Sequence[float]) -> np.ndarray:
    """四元数 (w, x, y, z) -> 3x3 旋转矩阵。"""
    w, x, y, z = [float(v) for v in q]
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_tf(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    """由旋转矩阵 R(3x3) 与平移 t(3,) 组装 4x4 齐次变换。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = [float(v) for v in t]
    return T


def cam2base_matrix() -> np.ndarray:
    """读取 config 静态外参，组装并缓存 T_cam2base（相机 -> 底盘）。"""
    global _CAM2BASE_CACHE
    if _CAM2BASE_CACHE is None:
        R = quat_wxyz_to_R(config.CAM2BASE_QUAT_WXYZ)
        _CAM2BASE_CACHE = make_tf(R, config.CAM2BASE_XYZ_M)
    return _CAM2BASE_CACHE


def cam2base_matrix_from_tf(tf: Dict) -> np.ndarray | None:
    """由动态 TF 字典 {"xyz_m": [...], "quat_wxyz": [...]} 组装 4x4（光学系->底盘）。

    字段缺失/非法时返回 None，交由调用方回退 config 静态外参。
    """
    if not isinstance(tf, dict):
        return None
    q = tf.get("quat_wxyz")
    t = tf.get("xyz_m")
    if not q or not t or len(q) < 4 or len(t) < 3:
        return None
    try:
        return make_tf(quat_wxyz_to_R(q), t)
    except (TypeError, ValueError):
        return None


def base2map_matrix(x: float, y: float, yaw: float) -> np.ndarray:
    """底盘 2D Pose -> 4x4（绕 Z 转 yaw，平移 (x, y, 0)）。"""
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    return T


def camera_pose_in_map(pose2d: Dict[str, float], cam2base: Dict | None = None) -> np.ndarray:
    """底盘 2D Pose -> 相机光学系到地图系的 4x4。

    cam2base: 可选动态外参 {xyz_m, quat_wxyz}（光学系->底盘）。有效时优先使用，
              否则回退 config.CAM2BASE_* 静态兜底值。
    """
    x = float(pose2d["x"])
    y = float(pose2d["y"])
    yaw = float(pose2d.get("yaw", pose2d.get("theta", 0.0)))
    T_cam2base = cam2base_matrix_from_tf(cam2base) if cam2base else None
    if T_cam2base is None:
        T_cam2base = cam2base_matrix()
    return (base2map_matrix(x, y, yaw) @ T_cam2base).astype(np.float64)


def world_xy_to_pixel(
    x: float, y: float, resolution: float, origin: Sequence[float], img_h: int,
) -> Tuple[int, int]:
    """地图坐标 (x, y, 米) -> 占据栅格图像像素 (px, py)（ROS 约定，origin 为左下角）。"""
    ox = float(origin[0]) if len(origin) > 0 else 0.0
    oy = float(origin[1]) if len(origin) > 1 else 0.0
    px = int(round((float(x) - ox) / float(resolution)))
    py = int(round(int(img_h) - (float(y) - oy) / float(resolution)))
    return px, py
