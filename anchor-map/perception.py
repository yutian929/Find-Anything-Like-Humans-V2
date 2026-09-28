"""感知算法工具（自 anchor-map-v2 core_models.py 拆出的纯函数部分）。

包含：图像/深度解码、bbox 处理、内参/位姿归一化、点云去噪、mask->3D 中心、
3D bbox、SAM3 仲裁用的同图去重/重合度、结果图落盘、JSON 读写。
不含模型类（见 model_clients/）、不含 AnchorStore（见 anchor_store.py）。
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config

logger = logging.getLogger("perception")

# 快照文件名约定
FIRST_SNAPSHOT_NAME = "first"
LAST_SNAPSHOT_NAME = "last"
RGB_FILE_NAME = "rgb.png"
DEPTH_FILE_NAME = "depth.png"
CAMERA_POSE_FILE_NAME = "camera_pose.npy"
CAMERA_INTRINSICS_FILE_NAME = "camera_intrinsics.npy"
ANCHORS_INDEX_FILE_NAME = "anchors.json"

# 点云去噪参数
DEFAULT_SOR_NB = 30
DEFAULT_SOR_STD = 2.0
DEFAULT_DBSCAN_EPS = 0.03
DEFAULT_DBSCAN_MIN_POINTS = 50

# highlight 结果图参数
HIGHLIGHT_ALPHA_THRESHOLD = 8
HIGHLIGHT_PADDING_PX = 16
HIGHLIGHT_GLOW_DILATE_PX = 5
HIGHLIGHT_GLOW_BLUR_PX = 15
HIGHLIGHT_GLOW_COLOR_RGBA = (0, 255, 220, 190)


# =========================================================================== #
# 名称 / JSON 小工具
# =========================================================================== #
def safe_object_name(name: str) -> str:
    raw = str(name).strip()
    safe = re.sub(r"[^0-9A-Za-z_\-]+", "_", raw).strip("_")
    if not safe:
        safe = "object"
    if safe != raw:
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}_{digest}"
    return safe


def read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else dict(default)
    except Exception:
        return dict(default)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# =========================================================================== #
# 图像 / 深度 解码 + bbox
# =========================================================================== #
def decode_image_bytes(image_bytes: bytes, flags: int) -> Optional[np.ndarray]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, flags)


def decode_rgb_part(image_bytes: bytes) -> Optional[np.ndarray]:
    return decode_image_bytes(image_bytes, cv2.IMREAD_COLOR)


def decode_depth_part(depth_bytes: bytes, encoding: str = "image") -> Optional[np.ndarray]:
    encoding = str(encoding or "image").lower()
    if encoding in ("npy", "numpy"):
        try:
            return np.load(io.BytesIO(depth_bytes))
        except Exception as exc:
            logger.warning("decode depth npy failed: %s", exc)
            return None
    return decode_image_bytes(depth_bytes, cv2.IMREAD_UNCHANGED)


def denormalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 / 1000.0 * width)),
        int(round(y1 / 1000.0 * height)),
        int(round(x2 / 1000.0 * width)),
        int(round(y2 / 1000.0 * height)),
    ]


def clip_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    bbox = [int(v) for v in bbox]
    bbox[0] = int(np.clip(bbox[0], 0, width - 1))
    bbox[1] = int(np.clip(bbox[1], 0, height - 1))
    bbox[2] = int(np.clip(bbox[2], bbox[0] + 1, width))
    bbox[3] = int(np.clip(bbox[3], bbox[1] + 1, height))
    return bbox


def normalize_intrinsics(intrinsics: Any, width: int, height: int) -> Dict[str, float]:
    if isinstance(intrinsics, dict):
        fx = float(intrinsics.get("fx", 600.0))
        fy = float(intrinsics.get("fy", 600.0))
        cx = float(intrinsics.get("cx", intrinsics.get("ppx", width / 2.0)))
        cy = float(intrinsics.get("cy", intrinsics.get("ppy", height / 2.0)))
        ds = float(intrinsics.get("depth_scale", config.DEFAULT_DEPTH_SCALE))
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "ppx": cx, "ppy": cy, "depth_scale": ds}
    if isinstance(intrinsics, (list, tuple, np.ndarray)) and len(intrinsics) >= 4:
        v = list(intrinsics)
        fx, fy, cx, cy = [float(t) for t in v[:4]]
        ds = float(v[4]) if len(v) >= 5 else config.DEFAULT_DEPTH_SCALE
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "ppx": cx, "ppy": cy, "depth_scale": ds}
    return {"fx": 600.0, "fy": 600.0, "cx": width / 2.0, "cy": height / 2.0,
            "ppx": width / 2.0, "ppy": height / 2.0, "depth_scale": config.DEFAULT_DEPTH_SCALE}


def pose2d_to_matrix(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    pose = np.eye(4, dtype=np.float64)
    pose[0, 0] = c
    pose[0, 1] = -s
    pose[1, 0] = s
    pose[1, 1] = c
    pose[0, 3] = float(x)
    pose[1, 3] = float(y)
    return pose


def normalize_camera_pose(camera_pose: Any) -> np.ndarray:
    """支持 4x4 矩阵 / 16 元素一维 / [x,y,yaw] / {x,y,yaw}。"""
    if isinstance(camera_pose, dict):
        if "matrix" in camera_pose:
            camera_pose = camera_pose["matrix"]
        elif all(k in camera_pose for k in ("x", "y", "yaw")):
            return pose2d_to_matrix(float(camera_pose["x"]), float(camera_pose["y"]), float(camera_pose["yaw"]))
    arr = np.asarray(camera_pose, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.size >= 16:
        return arr.reshape(4, 4).astype(np.float64)
    if arr.size >= 3:
        return pose2d_to_matrix(float(arr.flat[0]), float(arr.flat[1]), float(arr.flat[2]))
    raise ValueError("camera_pose 必须是 4x4 矩阵或 [x, y, yaw]")


def pose_summary(camera_pose: np.ndarray) -> Dict[str, float]:
    """从相机 4x4 提取用于锚点匹配的 (x, y, yaw)。前向 = 光学 Z 轴（第 2 列）。"""
    x = float(camera_pose[0, 3])
    y = float(camera_pose[1, 3])
    forward_optical = np.asarray([camera_pose[0, 2], camera_pose[1, 2]], dtype=np.float64)
    if float(np.linalg.norm(forward_optical)) > 0.2:
        yaw = float(np.arctan2(forward_optical[1], forward_optical[0]))
    else:
        yaw = float(np.arctan2(camera_pose[1, 0], camera_pose[0, 0]))
    return {"x": x, "y": y, "yaw": yaw}


def yaw_diff_deg(a: float, b: float) -> float:
    diff = np.arctan2(np.sin(a - b), np.cos(a - b))
    return float(abs(np.degrees(diff)))


def transform_camera_to_map(center_cam: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    p = np.array([float(center_cam[0]), float(center_cam[1]), float(center_cam[2]), 1.0], dtype=np.float64)
    out = camera_pose.astype(np.float64) @ p
    return out[:3]


# =========================================================================== #
# 点云去噪 + 3D 中心
# =========================================================================== #
def _denoise_with_colors(
    points_cam: np.ndarray, colors_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Open3D SOR + DBSCAN 去噪，同时保留每点 RGB（缺 open3d 时 numpy 兜底原样返回）。"""
    info: Dict[str, Any] = {"input_points": int(points_cam.shape[0]), "method": ""}
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        info["method"] = "numpy_fallback"
        return points_cam.astype(np.float32), colors_rgb.astype(np.float32), info

    info["method"] = "open3d_sor_dbscan"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cam.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64) / 255.0)
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=DEFAULT_SOR_NB, std_ratio=DEFAULT_SOR_STD)
    clean_xyz = np.asarray(pcd_clean.points)
    clean_rgb = np.asarray(pcd_clean.colors) * 255.0
    info["after_sor"] = int(clean_xyz.shape[0])
    if clean_xyz.shape[0] == 0:
        return clean_xyz.astype(np.float32), clean_rgb.astype(np.float32), info
    labels = np.asarray(pcd_clean.cluster_dbscan(
        eps=DEFAULT_DBSCAN_EPS, min_points=DEFAULT_DBSCAN_MIN_POINTS, print_progress=False,
    ))
    valid = labels[labels >= 0]
    if valid.size == 0:
        return clean_xyz.astype(np.float32), clean_rgb.astype(np.float32), info
    largest = int(np.argmax(np.bincount(valid)))
    keep = labels == largest
    return clean_xyz[keep].astype(np.float32), clean_rgb[keep].astype(np.float32), info


def mask_to_3d_points_center_and_colors(
    depth_map: np.ndarray, mask: np.ndarray, rgb_image: np.ndarray, intrinsics: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any], np.ndarray]:
    """mask + depth -> 去噪后相机系 3D 中心 + 带色点云 (N,6)=[x,y,z,r,g,b]。失败时中心为 None。"""
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics.get("cx", intrinsics.get("ppx", 0.0)))
    cy = float(intrinsics.get("cy", intrinsics.get("ppy", 0.0)))
    depth_scale = float(intrinsics.get("depth_scale", config.DEFAULT_DEPTH_SCALE))

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, {"error": "empty_mask"}, np.zeros((0, 6), dtype=np.float32)
    depths = depth_map[ys, xs].astype(np.float32) * depth_scale
    valid = depths > 0
    xs, ys, depths = xs[valid], ys[valid], depths[valid]
    if len(depths) == 0:
        return None, {"error": "empty_depth"}, np.zeros((0, 6), dtype=np.float32)

    x3 = (xs.astype(np.float32) - cx) * depths / fx
    y3 = (ys.astype(np.float32) - cy) * depths / fy
    points = np.stack((x3, y3, depths), axis=-1)
    colors_rgb = rgb_image[ys, xs].astype(np.float32)

    cleaned_xyz, cleaned_rgb, info = _denoise_with_colors(points, colors_rgb)
    if cleaned_xyz.shape[0] == 0:
        info["error"] = "empty_after_denoise"
        return None, info, np.zeros((0, 6), dtype=np.float32)
    colored_pc = np.hstack([cleaned_xyz, cleaned_rgb]).astype(np.float32)
    center = cleaned_xyz.mean(axis=0).astype(np.float32)
    return center, info, colored_pc


def compute_bbox_3d_corners(pointcloud_world: np.ndarray, trim_percentile: float = 1.0) -> List[Dict[str, float]]:
    """由世界系点云算 AABB 的 8 个角点。"""
    if pointcloud_world.shape[0] < 3:
        return []
    low = max(0.0, min(50.0, float(trim_percentile))) / 2.0
    high = 100.0 - low
    min_vals = np.percentile(pointcloud_world, low, axis=0)
    max_vals = np.percentile(pointcloud_world, high, axis=0)
    corners: List[Dict[str, float]] = []
    for dx in (0, 1):
        x = float(min_vals[0] if dx == 0 else max_vals[0])
        for dy in (0, 1):
            y = float(min_vals[1] if dy == 1 else max_vals[1])
            for dz in (0, 1):
                z = float(min_vals[2] if dz == 0 else max_vals[2])
                corners.append({"x": x, "y": y, "z": z})
    return corners


def pointcloud_cam_to_world(pointcloud_cam: np.ndarray, camera_pose: np.ndarray):
    """相机系带色点云 (N,3或6) -> 世界系点云 + 3D bbox 角点。"""
    n = pointcloud_cam.shape[0]
    has_color = pointcloud_cam.shape[1] >= 6
    if n < 3:
        return np.zeros((0, 6 if has_color else 3), dtype=np.float32), []
    pose = camera_pose.astype(np.float32)
    pc_xyz = pointcloud_cam[:, :3]
    pc_rgb = pointcloud_cam[:, 3:6] if has_color else np.zeros((n, 0), dtype=np.float32)
    ones = np.ones((n, 1), dtype=np.float32)
    world_xyz = (pose @ np.hstack([pc_xyz, ones]).T).T[:, :3]
    world = np.hstack([world_xyz, pc_rgb]) if has_color else world_xyz
    return world.astype(np.float32), compute_bbox_3d_corners(world_xyz)


def pointcloud_for_display(points: np.ndarray, max_points: int = 5000) -> Dict[str, Any]:
    """将地图系点云限量采样为前端可绘制的彩色坐标。"""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        raise ValueError("pointcloud must have shape (N,3) or (N,6)")
    if max_points < 1:
        raise ValueError("max_points must be positive")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=np.int64)]
    return {"points": points.astype(float).tolist(), "colored": points.shape[1] == 6}


# =========================================================================== #
# SAM3 仲裁：同图去重 + 重合度
# =========================================================================== #
def filter_nested_instances(
    masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray, overlap_threshold: float,
) -> List[Dict[str, Any]]:
    """高分优先，移除交集占较小 bbox 面积过高的同图冗余实例。"""
    if len(boxes) == 0:
        return []
    order = np.argsort(np.asarray(scores))[::-1]
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    keep: List[int] = []
    for index in order:
        if areas[index] <= 0:
            continue
        redundant = False
        for kept in keep:
            x1, y1 = max(boxes[index][0], boxes[kept][0]), max(boxes[index][1], boxes[kept][1])
            x2, y2 = min(boxes[index][2], boxes[kept][2]), min(boxes[index][3], boxes[kept][3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            smaller = min(areas[index], areas[kept])
            if smaller > 0 and intersection / smaller > float(overlap_threshold):
                redundant = True
                break
        if not redundant:
            keep.append(int(index))
    return [{"mask": masks[i], "bbox": boxes[i], "score": float(scores[i])} for i in keep]


def bbox_overlap_smaller(box_a: Any, box_b: Any) -> float:
    """两 bbox 重合度 = 交集面积 / 较小框面积（非 IoU）。box: [x1,y1,x2,y2]。"""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    if smaller <= 0.0:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(inter / smaller)


# =========================================================================== #
# 锚点范围筛选 + 内参读取
# =========================================================================== #
def filter_anchors_by_xy(
    anchors: List[Dict[str, Any]],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """按 2D 坐标范围筛选锚点（任一维 None 则不限制）。"""
    if x_range is None and y_range is None:
        return list(anchors)

    def _norm(rng):
        if rng is None:
            return None
        lo, hi = float(rng[0]), float(rng[1])
        return (lo, hi) if lo <= hi else (hi, lo)

    xr, yr = _norm(x_range), _norm(y_range)
    kept: List[Dict[str, Any]] = []
    for anchor in anchors:
        try:
            ax, ay = float(anchor["x"]), float(anchor["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if xr is not None and not (xr[0] <= ax <= xr[1]):
            continue
        if yr is not None and not (yr[0] <= ay <= yr[1]):
            continue
        kept.append(anchor)
    return kept


def load_intrinsics_npy(snapshot_dir: Path, width: int, height: int) -> Dict[str, float]:
    path = snapshot_dir / CAMERA_INTRINSICS_FILE_NAME
    if path.exists():
        try:
            return normalize_intrinsics(np.load(str(path), allow_pickle=True).item(), width, height)
        except Exception:
            try:
                return normalize_intrinsics(np.load(str(path), allow_pickle=True), width, height)
            except Exception:
                pass
    return normalize_intrinsics(None, width, height)


# =========================================================================== #
# 结果图落盘
# =========================================================================== #
def save_mask_rgba(image_rgb: np.ndarray, mask: np.ndarray, save_path: Path) -> None:
    alpha = (mask > 0).astype(np.uint8) * 255
    rgba = np.dstack([image_rgb, alpha])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(save_path)


def save_highlight_rgba(image_rgb: np.ndarray, mask: np.ndarray, save_path: Path) -> None:
    alpha = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(alpha > HIGHLIGHT_ALPHA_THRESHOLD)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if len(xs) == 0 or len(ys) == 0:
        Image.fromarray(np.dstack([image_rgb, alpha]), mode="RGBA").save(save_path)
        return
    h, w = alpha.shape[:2]
    pad = max(0, int(HIGHLIGHT_PADDING_PX))
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + 1 + pad)
    y2 = min(h, int(ys.max()) + 1 + pad)
    rgb_crop = image_rgb[y1:y2, x1:x2]
    alpha_crop = alpha[y1:y2, x1:x2]

    glow_alpha = alpha_crop.copy()
    dilate_px = max(0, int(HIGHLIGHT_GLOW_DILATE_PX))
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        glow_alpha = cv2.dilate(glow_alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    blur_px = max(0, int(HIGHLIGHT_GLOW_BLUR_PX))
    if blur_px > 0:
        k = blur_px * 2 + 1
        glow_alpha = cv2.GaussianBlur(glow_alpha, (k, k), 0)

    gr, gg, gb, ga = [int(v) for v in HIGHLIGHT_GLOW_COLOR_RGBA]
    glow_alpha = np.clip(glow_alpha.astype(np.float32) * (ga / 255.0), 0, 255).astype(np.uint8)
    glow = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
    glow[:, :, 0], glow[:, :, 1], glow[:, :, 2], glow[:, :, 3] = gr, gg, gb, glow_alpha
    obj = np.dstack([rgb_crop, alpha_crop])
    out = Image.alpha_composite(Image.fromarray(glow, mode="RGBA"), Image.fromarray(obj, mode="RGBA"))
    out.save(save_path)


def save_bbox_overlay(image_rgb: np.ndarray, bbox: List[int], save_path: Path) -> str:
    """绘制 bbox 落盘，返回该带框图（JPEG）的 base64。"""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), bgr)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buf).decode("utf-8") if ok else ""


def encode_jpeg_b64(bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return base64.b64encode(buf).decode("utf-8") if ok else ""
