"""状态机 Engine —— 合并原三节点的运行时职责。

单线程主循环：
- MAPPING（默认/空闲）：从 FrameSource 取最新同步帧 -> 直接使用 T_map_camera -> AnchorStore.update
  （匹配则刷新，建锚脉冲置位则新建）。这就是「没事时一直跑匹配刷新」。
- 收到请求：从队列取出，切到对应态处理，把结果回填 Future，再切回 MAPPING。
  因为只有主循环这一个线程碰模型与 AnchorStore，无需加锁。

两类检索请求：
- semantic_query(object, xy_range?)：遍历所有锚点（RynnBrain 召回 + SAM3 仲裁 + SAM2 精分割）。
- instant_detect(object)：只取当前帧做一次检测。
控制类请求：build_anchor / delete_anchor / clear_anchors / clear_history。
"""
from __future__ import annotations

import logging
import queue
import random
import re
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
import perception as P
from anchor_store import AnchorStore
from frame_source import FrameSource
from model_clients import build_rynnbrain, build_sam2, build_sam3

logger = logging.getLogger("engine")

RESULT_URL_PREFIX = "/api/results"


class State(str, Enum):
    BOOTING = "booting"
    MAPPING = "mapping"
    QUERYING = "querying"
    DETECTING = "detecting"
    CONTROL = "control"


@dataclass
class _Request:
    kind: str                       # 'query' | 'detect' | 'build' | 'delete' | 'clear_anchors' | 'clear_history' | 'warmup'
    payload: Dict[str, Any] = field(default_factory=dict)
    state: State = State.CONTROL
    future: Future = field(default_factory=Future)


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", str(text)))


def _result_summary(result: Any) -> Dict[str, Any]:
    """保留指令结果的关键信息，避免将图片 base64 或大量候选写入日志。"""
    if not isinstance(result, dict):
        return {"result": str(result)}
    keys = (
        "success", "found", "object", "anchor_id", "status", "bbox", "center_3d",
        "removed", "removed_anchors", "removed_objects", "message", "error",
    )
    return {key: result[key] for key in keys if key in result}


class Engine:
    def __init__(self, source: FrameSource, store: AnchorStore):
        self.source = source
        self.store = store
        self.state = State.BOOTING

        self._queue: "queue.Queue[_Request]" = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._anchor_pulse = False           # 建锚脉冲（单次，自动复位）
        self._prompt_cache: Dict[str, str] = {}
        self._last_update: Dict[str, Any] = {}

        # 模型（懒加载；由主循环线程独占构建）
        self._rynnbrain = None
        self._sam2 = None
        self._sam3 = None

        # 落盘目录
        self.map_root = Path(config.MAP_ROOT)
        self.results_dir = self.map_root / "results"
        self.history_path = self.results_dir / "history.json"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            P.write_json_atomic(self.history_path, {})

        self._tick = 1.0 / max(0.1, float(config.ENGINE_TICK_HZ))

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()
        if bool(getattr(config, "PRELOAD_MODELS", False)):
            self._submit(_Request("warmup"))   # 由主循环线程预热，避免多线程抢构建

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        logger.info("engine loop started (tick=%.2fs)", self._tick)
        logger.info("state %s -> %s", self.state.value, State.MAPPING.value)
        self.state = State.MAPPING
        while self._running:
            try:
                req = self._queue.get(timeout=self._tick)
            except queue.Empty:
                self._mapping_tick()
                continue
            self._handle(req)
        logger.info("engine loop stopped")

    # ------------------------------------------------------------------ #
    # 对外 API（入队 + 等 Future）
    # ------------------------------------------------------------------ #
    def _submit(self, req: _Request) -> Any:
        self._queue.put(req)
        return req.future.result()

    def semantic_query(self, obj: str, xy_range: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._submit(_Request("query", {"object": obj, "xy_range": xy_range or {}}, State.QUERYING))

    def instant_detect(self, obj: str) -> Dict[str, Any]:
        return self._submit(_Request("detect", {"object": obj}, State.DETECTING))

    def build_anchor(self) -> Dict[str, Any]:
        return self._submit(_Request("build"))

    def delete_anchor(self, anchor_id: str) -> Dict[str, Any]:
        return self._submit(_Request("delete", {"anchor_id": anchor_id}))

    def clear_anchors(self) -> Dict[str, Any]:
        return self._submit(_Request("clear_anchors"))

    def clear_history(self) -> Dict[str, Any]:
        return self._submit(_Request("clear_history"))

    def status(self) -> Dict[str, Any]:
        """只读，不入队。"""
        return {
            "state": self.state.value,
            "num_anchors": len(self.store.anchors()),
            "anchor_pulse": self._anchor_pulse,
            "models_loaded": {
                "rynnbrain": self._rynnbrain is not None,
                "sam2": self._sam2 is not None,
                "sam3": self._sam3 is not None,
            },
            "last_update": dict(self._last_update),
            "source": self.source.healthz(),
        }

    # ------------------------------------------------------------------ #
    # 请求分发
    # ------------------------------------------------------------------ #
    def _handle(self, req: _Request) -> None:
        previous_state = self.state
        self.state = req.state
        logger.info("state %s -> %s (request=%s)", previous_state.value, req.state.value, req.kind)
        try:
            if req.kind == "query":
                result = self._do_query(req.payload["object"], req.payload.get("xy_range") or {})
            elif req.kind == "detect":
                result = self._do_detect(req.payload["object"])
            elif req.kind == "build":
                self._anchor_pulse = True
                result = {"success": True, "message": "下一次 update 将建立新锚点"}
            elif req.kind == "delete":
                result = self.store.delete(req.payload["anchor_id"])
            elif req.kind == "clear_anchors":
                result = {"success": True, **self.store.clear()}
            elif req.kind == "clear_history":
                result = self._clear_history()
            elif req.kind == "warmup":
                result = self._warmup()
            else:
                result = {"success": False, "error": f"unknown request: {req.kind}"}
            logger.info("request %s result=%s", req.kind, _result_summary(result))
            req.future.set_result(result)
        except Exception as exc:  # noqa: BLE001 —— 保证 Future 一定被回填
            logger.exception("handle %s failed", req.kind)
            result = {"success": False, "error": str(exc)}
            logger.info("request %s result=%s", req.kind, _result_summary(result))
            req.future.set_result(result)
        finally:
            self.state = State.MAPPING
            logger.info("state %s -> %s (request=%s)", req.state.value, State.MAPPING.value, req.kind)

    # ------------------------------------------------------------------ #
    # MAPPING：匹配刷新
    # ------------------------------------------------------------------ #
    def _mapping_tick(self) -> None:
        frame = self.source.latest()
        if frame is None:
            return
        try:
            bgr = frame.rgb
            depth = frame.depth
            h, w = bgr.shape[:2]
            camera_pose = frame.camera_pose
            intr = P.normalize_intrinsics(frame.intrinsics, w, h)
            is_new = self._anchor_pulse
            self._anchor_pulse = False
            status = self.store.update(bgr, depth, camera_pose, intr, is_new)
            status["is_new_anchor"] = is_new
            status["ts"] = time.time()
            previous_status = self._last_update.get("status")
            previous_anchor = self._last_update.get("anchor_id")
            self._last_update = status
            if (is_new or status.get("status") != previous_status
                    or status.get("anchor_id") != previous_anchor):
                logger.info("mapping update status=%s anchor_id=%s",
                            status.get("status"), status.get("anchor_id"))
        except Exception:
            logger.exception("mapping tick failed")

    # ------------------------------------------------------------------ #
    # 模型懒加载（仅主循环线程调用）
    # ------------------------------------------------------------------ #
    def _ensure_rynnbrain(self):
        if self._rynnbrain is None:
            self._rynnbrain = build_rynnbrain(config.RYNNBRAIN)
        return self._rynnbrain

    def _ensure_sam2(self):
        if self._sam2 is None:
            self._sam2 = build_sam2(config.SAM2)
        return self._sam2

    def _ensure_sam3(self):
        if self._sam3 is None:
            self._sam3 = build_sam3(config.SAM3)
        return self._sam3

    def _warmup(self) -> Dict[str, Any]:
        t0 = time.time()
        self._ensure_rynnbrain()
        self._ensure_sam2()
        self._ensure_sam3()
        logger.info("模型预热完成，用时 %.1fs", time.time() - t0)
        return {"success": True, "warmup_seconds": round(time.time() - t0, 1)}

    # ------------------------------------------------------------------ #
    # semantic_query
    # ------------------------------------------------------------------ #
    def _do_query(self, obj: str, xy_range: Dict[str, Any]) -> Dict[str, Any]:
        object_name = str(obj).strip()
        if not object_name:
            return {"success": False, "found": False, "center_3d": None, "error": "object is empty"}

        rynnbrain = self._ensure_rynnbrain()
        sam2 = self._ensure_sam2()
        sam3 = self._ensure_sam3()

        x_range = _build_range(xy_range.get("x_min"), xy_range.get("x_max"))
        y_range = _build_range(xy_range.get("y_min"), xy_range.get("y_max"))
        use_range = x_range is not None or y_range is not None

        all_anchors = self.store.anchors()
        anchors = P.filter_anchors_by_xy(all_anchors, x_range, y_range)
        logger.debug("query[%s] anchors=%d/%d range=%s", object_name, len(anchors), len(all_anchors), use_range)
        if use_range and not anchors:
            return {"success": True, "found": False, "object": object_name, "center_3d": None,
                    "message": "坐标范围内没有锚点", "num_anchors_in_range": 0, "num_anchors_total": len(all_anchors)}

        primary, fallback = self.store.snapshot_candidates_split(anchors)
        for snaps in (primary, fallback):
            if not snaps:
                continue
            response = self._run_search_pass(rynnbrain, sam2, sam3, object_name, snaps)
            if response is not None:
                return response
        return self._query_not_found(object_name)

    def _run_search_pass(self, rynnbrain, sam2, sam3, object_name, snapshot_candidates) -> Optional[Dict[str, Any]]:
        # 阶段1+2：RynnBrain 召回（存在性 + bbox），不早退
        candidates = self._collect_rynnbrain_candidates(rynnbrain, object_name, snapshot_candidates)
        if not candidates:
            return None

        # 阶段3：SAM3 辅助打分
        try:
            prompt = self._resolve_prompt(rynnbrain, object_name)
            self._score_candidates_by_sam3(sam3, prompt, candidates)
        except Exception:
            logger.exception("SAM3 scoring failed, fallback to random arbitration")
            prompt = ""
            for c in candidates:
                c.setdefault("sam3_confidence", None)

        num_confirmed = sum(1 for c in candidates if c.get("sam3_confidence") is not None)

        # 阶段4：仲裁 + SAM2 精分割 + 3D（失败则回退下一名重新仲裁）
        remaining = list(candidates)
        while remaining:
            arb = _arbitrate(remaining)
            if arb is None:
                break
            winner = arb["winner"]
            local = self._segment_and_localize(sam2, winner, winner["bbox"])
            if local is None:
                remaining.remove(winner)
                continue
            urls = self._write_result_images(object_name, winner, winner["bbox"], local["mask"],
                                             local.get("pointcloud_world"))
            annotated = urls.pop("annotated_image", "")
            cmap, ccam = local["center_map"], local["center_cam"]
            arbitration = {
                "sam3_prompt": prompt, "num_yes_snapshots": len(candidates),
                "num_sam3_confirmed": num_confirmed, "decision": arb["decision"],
                "chosen_confidence": winner.get("sam3_confidence"),
            }
            result = {
                "name": object_name, "found": True, "anchor_id": winner["anchor_id"],
                "snapshot": winner["snapshot"], "bbox": [int(v) for v in winner["bbox"]],
                "center_3d": {"x": float(cmap[0]), "y": float(cmap[1]), "z": float(cmap[2])},
                "center_xy": [float(cmap[0]), float(cmap[1])],
                "center_camera": [float(ccam[0]), float(ccam[1]), float(ccam[2])],
                **urls, "bbox_3d": local.get("bbox_3d", []), "denoise": local["denoise"],
                "arbitration": arbitration, "candidates": [_brief(c) for c in candidates],
                "updated_at": time.time(),
            }
            self._update_history(object_name, result)
            return {"success": True, "found": True, "object": object_name, **result, "annotated_image": annotated}
        return None

    def _collect_rynnbrain_candidates(self, rynnbrain, object_name, snapshot_candidates) -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        for _, aid, snapshot_name in snapshot_candidates:
            frame = self.store.load_snapshot_frame(aid, snapshot_name)
            if frame is not None:
                frames.append(frame)
        if not frames:
            return []
        bboxes_norm = rynnbrain.recall_batch([f["image"] for f in frames], object_name)
        candidates: List[Dict[str, Any]] = []
        for frame, bbox_norm in zip(frames, bboxes_norm):
            if bbox_norm is None:
                continue
            bbox = P.clip_bbox(P.denormalize_bbox(bbox_norm, frame["width"], frame["height"]),
                               frame["width"], frame["height"])
            frame["bbox"] = bbox
            candidates.append(frame)
        return candidates

    def _score_candidates_by_sam3(self, sam3, prompt: str, candidates: List[Dict[str, Any]]) -> None:
        overlap_thresh = float(config.QUERY_SAM3_OVERLAP_THRESH)
        score_thresh = float(config.QUERY_SAM3_SCORE_THRESH)
        per_image = sam3.detect([c["image"] for c in candidates], prompt)
        for candidate, instances in zip(candidates, per_image):
            best: Optional[float] = None
            for inst in instances:
                score = float(inst.get("score", 0.0))
                if score < score_thresh:
                    continue
                if P.bbox_overlap_smaller(candidate["bbox"], inst["bbox"]) > overlap_thresh and (best is None or score > best):
                    best = score
            candidate["sam3_confidence"] = best

    def _segment_and_localize(self, sam2, frame: Dict[str, Any], bbox: List[int]) -> Optional[Dict[str, Any]]:
        mask = sam2.segment(frame["image"], bbox)
        center_cam, denoise, pc_cam = P.mask_to_3d_points_center_and_colors(
            frame["depth"], mask, frame["rgb"], frame["intrinsics"])
        if center_cam is None:
            logger.warning("3D center failed @ %s/%s: %s", frame["anchor_id"], frame["snapshot"], denoise)
            return None
        center_map = P.transform_camera_to_map(center_cam, frame["camera_pose"])
        pc_world, bbox_3d = P.pointcloud_cam_to_world(pc_cam, frame["camera_pose"])
        return {"mask": mask, "center_cam": center_cam, "center_map": center_map,
                "denoise": denoise, "pointcloud_world": pc_world, "bbox_3d": bbox_3d}

    def _resolve_prompt(self, rynnbrain, object_name: str) -> str:
        name = str(object_name).strip()
        if name in config.OBJECT_PROMPTS:
            return config.OBJECT_PROMPTS[name]
        if name in self._prompt_cache:
            return self._prompt_cache[name]
        if not _has_chinese(name):
            return name
        prompt = rynnbrain.translate(name)
        self._prompt_cache[name] = prompt
        return prompt

    # ------------------------------------------------------------------ #
    # instant_detect
    # ------------------------------------------------------------------ #
    def _do_detect(self, obj: str) -> Dict[str, Any]:
        object_name = str(obj).strip()
        if not object_name:
            return {"success": False, "found": False, "bbox": None, "center_3d": None, "error": "object is empty"}
        frame = self.source.latest()
        if frame is None:
            return {"success": False, "found": False, "center_3d": None, "error": "无当前帧（数据源未就绪）"}

        rynnbrain = self._ensure_rynnbrain()
        sam2 = self._ensure_sam2()

        bgr = frame.rgb
        depth = frame.depth
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        image = Image.fromarray(rgb)
        w, h = image.size
        pose_mat = frame.camera_pose
        intr = P.normalize_intrinsics(frame.intrinsics, w, h)

        bbox_norm = rynnbrain.detect_one(image, object_name)
        if bbox_norm is None:
            return {"success": True, "found": False, "bbox": None, "center_3d": None,
                    "image_b64": P.encode_jpeg_b64(bgr)}
        bbox = P.clip_bbox(P.denormalize_bbox(bbox_norm, w, h), w, h)
        mask = sam2.segment(image, bbox)

        center_cam, denoise, pc_cam = P.mask_to_3d_points_center_and_colors(depth, mask, rgb, intr)
        center_3d = center_xy = center_camera = None
        bbox_3d: List[Dict[str, float]] = []
        if center_cam is not None:
            cmap = P.transform_camera_to_map(center_cam, pose_mat)
            center_3d = {"x": float(cmap[0]), "y": float(cmap[1]), "z": float(cmap[2])}
            center_xy = [float(cmap[0]), float(cmap[1])]
            center_camera = [float(center_cam[0]), float(center_cam[1]), float(center_cam[2])]
            _, bbox_3d = P.pointcloud_cam_to_world(pc_cam, pose_mat)
        else:
            logger.warning("detect[%s] 3D center failed: %s", object_name, denoise)

        overlay = bgr.copy()
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        return {
            "success": True, "found": True, "object": object_name, "bbox": bbox,
            "center_3d": center_3d, "center_xy": center_xy, "center_camera": center_camera,
            "bbox_3d": bbox_3d, "denoise": denoise, "image_b64": P.encode_jpeg_b64(overlay),
        }

    # ------------------------------------------------------------------ #
    # 结果 / 历史 落盘
    # ------------------------------------------------------------------ #
    def _write_result_images(self, object_name, frame, bbox, mask, pointcloud_world=None) -> Dict[str, str]:
        safe = P.safe_object_name(object_name)
        object_dir = self.results_dir / safe
        P.save_highlight_rgba(frame["rgb"], mask, object_dir / "highlight.png")
        P.save_mask_rgba(frame["rgb"], mask, object_dir / "mask_rgba.png")
        annotated = P.save_bbox_overlay(frame["rgb"], bbox, object_dir / "bbox_overlay.png")
        urls = {
            "highlight_image_url": f"{RESULT_URL_PREFIX}/{safe}/highlight.png",
            "mask_image_url": f"{RESULT_URL_PREFIX}/{safe}/mask_rgba.png",
            "bbox_image_url": f"{RESULT_URL_PREFIX}/{safe}/bbox_overlay.png",
            "annotated_image": annotated,
        }
        if pointcloud_world is not None and pointcloud_world.shape[0] >= 3:
            np.save(str(object_dir / "pointcloud.npy"), pointcloud_world.astype(np.float32))
        return urls

    def _update_history(self, object_name: str, result: Dict[str, Any]) -> None:
        history = P.read_json(self.history_path, {})
        record = dict(result)
        record.pop("annotated_image", None)   # base64 大字符串不写入 history
        history[str(object_name)] = record
        P.write_json_atomic(self.history_path, history)

    def _query_not_found(self, object_name: str) -> Dict[str, Any]:
        object_dir = self.results_dir / P.safe_object_name(object_name)
        if object_dir.exists():
            for p in object_dir.iterdir():
                if p.is_file() or p.is_symlink():
                    p.unlink()
        self._update_history(object_name, {"name": object_name, "found": False,
                                            "center_3d": None, "updated_at": time.time()})
        return {"success": True, "found": False, "object": object_name, "center_3d": None}

    def _clear_history(self) -> Dict[str, Any]:
        import shutil
        removed = 0
        for p in list(self.results_dir.iterdir()) if self.results_dir.exists() else []:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        P.write_json_atomic(self.history_path, {})
        return {"success": True, "removed_objects": removed}

    def history(self) -> Dict[str, Any]:
        return P.read_json(self.history_path, {})


# --------------------------------------------------------------------------- #
# 无状态辅助
# --------------------------------------------------------------------------- #
def _build_range(lo: Any, hi: Any) -> Optional[Tuple[float, float]]:
    lo = _to_float(lo)
    hi = _to_float(hi)
    if lo is None and hi is None:
        return None
    return (lo if lo is not None else -float("inf"), hi if hi is not None else float("inf"))


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _arbitrate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """confirmed 恰好1个 -> 选它；>=2 -> 选 SAM3 confidence 最高；0 -> 随机兜底。"""
    if not candidates:
        return None
    confirmed = [c for c in candidates if c.get("sam3_confidence") is not None]
    if len(confirmed) == 1:
        return {"winner": confirmed[0], "decision": "one_confirmed"}
    if len(confirmed) >= 2:
        return {"winner": max(confirmed, key=lambda c: float(c["sam3_confidence"])), "decision": "sam3_top_confidence"}
    return {"winner": random.choice(candidates), "decision": "random_fallback"}


def _brief(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {"anchor_id": candidate["anchor_id"], "snapshot": candidate["snapshot"],
            "bbox": [int(v) for v in candidate["bbox"]], "sam3_confidence": candidate.get("sam3_confidence")}
