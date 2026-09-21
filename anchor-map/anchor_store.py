"""锚点语义地图存储：建/匹配/刷新/删单个/清空 + 快照落盘与读取。

自 anchor-map-v2 core_models.AnchorStore 迁移，新增：
- delete(anchor_id)：删除指定单个锚点（UI 可精确删除）。
- load_snapshot_frame / snapshot_candidates_split：供 semantic_query 遍历锚点。
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import perception as P

logger = logging.getLogger("anchor_store")

STATUS_CREATED = "created_anchor"
STATUS_REFRESHED = "refreshed_anchor"
STATUS_UNMATCHED = "unmatched"


class AnchorStore:
    def __init__(self, anchors_root: str | Path, xy_thresh: float, yaw_thresh_deg: float):
        self.root = Path(anchors_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / P.ANCHORS_INDEX_FILE_NAME
        self.xy_thresh = float(xy_thresh)
        self.yaw_thresh_deg = float(yaw_thresh_deg)
        self.index = P.read_json(self.index_path, {"current_anchor_id": None, "anchors": []})
        self.index.setdefault("current_anchor_id", None)
        self.index.setdefault("anchors", [])
        self.save_index()

    # --- 落盘 --- #
    def save_index(self) -> None:
        P.write_json_atomic(self.index_path, self.index)

    def anchor_dir(self, anchor_id: str) -> Path:
        return self.root / anchor_id

    def snapshot_dir(self, anchor_id: str, snapshot_name: str) -> Path:
        return self.anchor_dir(anchor_id) / snapshot_name

    def next_anchor_id(self) -> str:
        used = {str(a.get("id", "")) for a in self.index.get("anchors", [])}
        i = 0
        while True:
            aid = f"anchor_{i:04d}"
            if aid not in used:
                return aid
            i += 1

    def write_snapshot(self, anchor_id, snapshot_name, rgb_bgr, depth, camera_pose, intrinsics) -> None:
        snap = self.snapshot_dir(anchor_id, snapshot_name)
        snap.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snap / P.RGB_FILE_NAME), rgb_bgr)
        cv2.imwrite(str(snap / P.DEPTH_FILE_NAME), depth)          # uint16 PNG16 无损
        np.save(snap / P.CAMERA_POSE_FILE_NAME, camera_pose.astype(np.float32))
        np.save(snap / P.CAMERA_INTRINSICS_FILE_NAME, dict(intrinsics))

    # --- 建 / 匹配 / 刷新 --- #
    def create_anchor(self, rgb_bgr, depth, camera_pose, intrinsics) -> Dict[str, Any]:
        anchor_id = self.next_anchor_id()
        anchor = {"id": anchor_id, **P.pose_summary(camera_pose),
                  "first_created_at": time.time(), "last_updated_at": None}
        self.write_snapshot(anchor_id, P.FIRST_SNAPSHOT_NAME, rgb_bgr, depth, camera_pose, intrinsics)
        self.index["anchors"].append(anchor)
        self.index["current_anchor_id"] = anchor_id
        self.save_index()
        return anchor

    def find_match(self, camera_pose: np.ndarray) -> Optional[Dict[str, Any]]:
        cur = P.pose_summary(camera_pose)
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        for a in self.index.get("anchors", []):
            xy = float(np.hypot(cur["x"] - float(a["x"]), cur["y"] - float(a["y"])))
            yaw = P.yaw_diff_deg(cur["yaw"], float(a["yaw"]))
            if xy <= self.xy_thresh and yaw <= self.yaw_thresh_deg:
                score = xy + yaw / 180.0
                if best is None or score < best[0]:
                    best = (score, a)
        return best[1] if best else None

    def refresh_anchor(self, anchor, rgb_bgr, depth, camera_pose, intrinsics) -> Dict[str, Any]:
        anchor_id = str(anchor["id"])
        self.write_snapshot(anchor_id, P.LAST_SNAPSHOT_NAME, rgb_bgr, depth, camera_pose, intrinsics)
        anchor["last_updated_at"] = time.time()
        self.index["current_anchor_id"] = anchor_id
        self.save_index()
        return anchor

    def update(self, rgb_bgr, depth, camera_pose, intrinsics, is_new_anchor: bool) -> Dict[str, Any]:
        """MAPPING 主循环调用：建锚脉冲置位则新建，否则匹配已有锚点并刷新其 last 快照。"""
        if is_new_anchor:
            a = self.create_anchor(rgb_bgr, depth, camera_pose, intrinsics)
            return {"success": True, "status": STATUS_CREATED, "anchor_id": a["id"]}
        match = self.find_match(camera_pose)
        if match is None:
            return {"success": True, "status": STATUS_UNMATCHED, "anchor_id": None}
        a = self.refresh_anchor(match, rgb_bgr, depth, camera_pose, intrinsics)
        return {"success": True, "status": STATUS_REFRESHED, "anchor_id": a["id"]}

    # --- 查询 / 删除 --- #
    def anchors(self) -> List[Dict[str, Any]]:
        return list(self.index.get("anchors", []))

    def delete(self, anchor_id: str) -> Dict[str, Any]:
        """删除指定单个锚点及其快照目录。"""
        anchor_id = str(anchor_id)
        before = len(self.index.get("anchors", []))
        self.index["anchors"] = [a for a in self.index.get("anchors", []) if str(a.get("id", "")) != anchor_id]
        removed = before - len(self.index["anchors"])
        d = self.anchor_dir(anchor_id)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        if str(self.index.get("current_anchor_id")) == anchor_id:
            self.index["current_anchor_id"] = None
        self.save_index()
        return {"success": removed > 0, "removed": removed, "anchor_id": anchor_id}

    def clear(self) -> Dict[str, Any]:
        """清空所有锚点（等价于重新开始建图）。"""
        removed = 0
        for a in list(self.index.get("anchors", [])):
            aid = str(a.get("id", ""))
            if not aid:
                continue
            d = self.anchor_dir(aid)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        self.index = {"current_anchor_id": None, "anchors": []}
        self.save_index()
        return {"removed_anchors": removed}

    # --- semantic_query 遍历辅助 --- #
    def load_snapshot_frame(self, anchor_id: str, snapshot_name: str) -> Optional[Dict[str, Any]]:
        """读取快照 rgb/depth/pose/内参，组装成一帧上下文。缺文件/解码失败返回 None。"""
        snap = self.snapshot_dir(anchor_id, snapshot_name)
        rgb_path = snap / P.RGB_FILE_NAME
        depth_path = snap / P.DEPTH_FILE_NAME
        pose_path = snap / P.CAMERA_POSE_FILE_NAME
        if not (rgb_path.exists() and depth_path.exists() and pose_path.exists()):
            return None
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None:
            return None
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = image.size
        return {
            "anchor_id": anchor_id, "snapshot": snapshot_name, "snapshot_dir": snap,
            "rgb": rgb, "image": image, "depth": depth, "width": width, "height": height,
            "intrinsics": P.load_intrinsics_npy(snap, width, height),
            "camera_pose": np.load(str(pose_path)),
        }

    def snapshot_candidates_split(
        self, anchors: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[float, str, str]], List[Tuple[float, str, str]]]:
        """拆两批候选做「先最新、再兜底」两趟检索：
        primary=每锚点最新快照（有 last 用 last 否则 first）；fallback=有 last 锚点的 first。
        """
        primary: List[Tuple[float, str, str]] = []
        fallback: List[Tuple[float, str, str]] = []
        for anchor in anchors:
            aid = str(anchor.get("id", ""))
            if not aid:
                continue
            last_ts = anchor.get("last_updated_at")
            first_ts = anchor.get("first_created_at")

            def _push(bucket, ts, snapshot_name):
                if ts is None:
                    return
                try:
                    bucket.append((float(ts), aid, snapshot_name))
                except (TypeError, ValueError):
                    pass

            if last_ts is not None:
                _push(primary, last_ts, P.LAST_SNAPSHOT_NAME)
                _push(fallback, first_ts, P.FIRST_SNAPSHOT_NAME)
            else:
                _push(primary, first_ts, P.FIRST_SNAPSHOT_NAME)
        primary.sort(key=lambda item: item[0], reverse=True)
        fallback.sort(key=lambda item: item[0], reverse=True)
        return primary, fallback
