"""Cubify Anything RGB-D: local inference and HTTP client in one model module.

Torch and the CuTR SDK are loaded only when constructing the local client.
"""
from __future__ import annotations

import base64
from contextlib import nullcontext
import hashlib
import io
import math
import sys
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np


from ..robot_adapters.base import MAX_PIXELS, transform, validate_frame
from ..robot_adapters.base import CameraIntrinsics, RGBDFrame
from ..model_clients.base import Cuboid3D, CuboidDetectionResult

MODEL_ID = "cubify-anything-rgbd"
PROTOCOL_VERSION = 1



def threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score_threshold must be a number in [0,1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score_threshold must be a number in [0,1]")
    return float(value)


def gravity_rotation(frame):
    """Gravity-aligned cuboid -> optical frame; map +Z MUST be physical up.

    Strip yaw, retain pitch/roll. Like the official x/z tilt convention,
    Rz(roll) @ Rx(pitch) maps [0,1,0] to optical-frame gravity.
    No identity fallback when pose is missing or invalid.
    """
    r = transform(frame.T_map_camera)[:3, :3]
    down = r.T @ np.array([0., 0., -1.])
    if down[1] < abs(down[0]) or np.hypot(down[0], down[1]) < 1e-6:
        raise ValueError("CuTR requires upright RGB-D; rectify sideways/upside-down images and calibration first")
    pitch = math.asin(float(np.clip(down[2], -1, 1)))
    roll = math.atan2(-down[0], down[1])
    cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    return np.array([[cr, -sr*cp, sr*sp], [sr, cr*cp, -cr*sp], [0, sp, cp]], dtype=np.float32)


def make_result(frame, scores, corners, model_version, score_threshold):
    scores = np.asarray(scores, dtype=float)
    corners = np.asarray(corners, dtype=float)
    if scores.shape == (0,) and corners.size == 0:
        corners = corners.reshape(0, 8, 3)
    if scores.ndim != 1 or corners.shape != (len(scores), 8, 3):
        raise ValueError("Model must return N scores and Nx8x3 optical-frame corners")
    if (not np.isfinite(scores).all() or not np.isfinite(corners).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Invalid model scores or cuboid coordinates")
    if not isinstance(model_version, str) or not model_version:
        raise ValueError("model_version must be nonempty")
    if len(corners) and np.any(np.ptp(corners, axis=1) <= 0):
        raise ValueError("Degenerate cuboid")
    selected = np.flatnonzero(scores >= threshold(score_threshold))
    selected = selected[np.argsort(-scores[selected], kind="stable")]
    boxes = tuple(Cuboid3D(float(scores[i]), corners[i].copy()) for i in selected)
    return CuboidDetectionResult(frame.frame_id, frame.timestamp_ns, frame.camera_frame,
                                 MODEL_ID, model_version, boxes)


def encode_frame(frame, score_threshold):
    from PIL import Image
    rgb, depth = validate_frame(frame)
    image = io.BytesIO()
    Image.fromarray(rgb).save(image, format="PNG")
    return {
        "protocol_version": PROTOCOL_VERSION, "score_threshold": threshold(score_threshold),
        "frame": {
            "frame_id": frame.frame_id, "timestamp_ns": frame.timestamp_ns,
            "clock_domain": frame.clock_domain, "map_id": frame.map_id,
            "map_revision": frame.map_revision, "camera_frame": frame.camera_frame,
            "intrinsics": asdict(frame.intrinsics),
            "T_map_camera": np.asarray(frame.T_map_camera).tolist(),
            "rgb_png": base64.b64encode(image.getvalue()).decode("ascii"),
            "depth_f32_le": base64.b64encode(depth.astype("<f4").tobytes()).decode("ascii"),
        }}


def decode_frame(payload):
    from PIL import Image
    try:
        if payload["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("Unsupported protocol_version")
        data = payload["frame"]
        intrinsics = CameraIntrinsics(**data["intrinsics"])
        w, h = intrinsics.width, intrinsics.height
        if type(w) is not int or type(h) is not int or min(w, h) < 4 or w*h > MAX_PIXELS:
            raise ValueError("Invalid image size")
        raw = base64.b64decode(data["depth_f32_le"], validate=True)
        if len(raw) != w*h*4:
            raise ValueError("Depth byte count does not match intrinsics")
        depth = np.frombuffer(raw, dtype="<f4").reshape(h, w).copy()
        image_bytes = base64.b64decode(data["rgb_png"], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as im:
            if im.format != "PNG" or im.size != (w, h) or im.mode != "RGB":
                raise ValueError("RGB must be an RGB-mode PNG matching intrinsics")
            rgb = np.array(im)
        frame = RGBDFrame(
            frame_id=data["frame_id"], timestamp_ns=data["timestamp_ns"],
            clock_domain=data["clock_domain"], map_id=data["map_id"],
            map_revision=data["map_revision"], camera_frame=data["camera_frame"],
            intrinsics=intrinsics, rgb=rgb, depth_m=depth,
            T_map_camera=np.asarray(data["T_map_camera"], dtype=np.float64))
        validate_frame(frame)
        return frame, threshold(payload["score_threshold"])
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError("Malformed Cubify RGB-D request") from exc


def encode_result(result):
    return {"protocol_version": PROTOCOL_VERSION,
            "frame_id": result.frame_id, "timestamp_ns": result.timestamp_ns,
            "camera_frame": result.camera_frame,
            "model_id": result.model_id, "model_version": result.model_version,
            "scores": [box.confidence for box in result.cuboids],
            "corners_camera_m": [np.asarray(box.corners_camera_m).tolist() for box in result.cuboids]}


def decode_result(frame, payload, score_threshold):
    try:
        expected = {"protocol_version": PROTOCOL_VERSION, "frame_id": frame.frame_id,
                    "timestamp_ns": frame.timestamp_ns, "camera_frame": frame.camera_frame,
                    "model_id": MODEL_ID}
        if any(payload[k] != value for k, value in expected.items()):
            raise ValueError("Cubify response metadata does not match the request/model")
        return make_result(frame, payload["scores"], payload["corners_camera_m"],
                           payload["model_version"], score_threshold)
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed Cubify response") from exc


def _checkpoint_version(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


class CubifyAnythingLocal:
    """Single-frame cuboids in the ORIGINAL optical frame, with metric scale.

    source_path points to a directory containing cubifyanything/. It is an
    explicit dependency location, never discovered/downloaded automatically.
    Map +Z must be gravity aligned; images must already be upright/rectified.
    """

    def __init__(self, path, *, source_path=None, device=None, score_threshold=None,
                 max_side=None, fp16=True):
        device = "auto" if device is None else device
        max_side = 768 if max_side is None else max_side
        score_threshold = .25 if score_threshold is None else score_threshold
        self.score_threshold = threshold(score_threshold)
        if isinstance(max_side, bool) or not isinstance(max_side, int) or not 256 <= max_side <= 1024:
            raise ValueError("max_side must be an integer in [256,1024]")
        self.max_side = max_side
        if not isinstance(fp16, bool):
            raise ValueError("fp16 must be a boolean")
        self.fp16 = fp16
        weights = Path(path).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"CuTR RGB-D weights not found: {weights}")
        if source_path is not None:
            source = Path(source_path).expanduser().resolve()
            if not (source / "cubifyanything" / "cubify_transformer.py").is_file():
                raise ValueError("source_path must contain cubifyanything/cubify_transformer.py")
            loaded = sys.modules.get("cubifyanything")
            if loaded is not None and not any(Path(p).resolve() == source / "cubifyanything"
                                              for p in getattr(loaded, "__path__", ())):
                raise RuntimeError("A different cubifyanything package is already loaded")
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
        try:
            import torch
            from cubifyanything.cubify_transformer import make_cubify_transformer
            from cubifyanything.preprocessor import Augmentor, Preprocessor, move_input_to_current_device
            from cubifyanything.measurement import ImageMeasurementInfo, DepthMeasurementInfo
            from cubifyanything.sensor import PosedSensorInfo, SensorArrayInfo
        except ImportError as exc:
            raise ImportError("CuTR dependencies missing; use the documented inference environment/source_path") from exc
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        state = torch.load(weights, map_location="cpu", weights_only=True)["model"]
        if not any(k.startswith("backbone.0.patch_embed_depth.") for k in state):
            raise ValueError("Expected RGB-D weights, received an RGB-only checkpoint")
        dimension = int(state["backbone.0.patch_embed.proj.weight"].shape[0])
        self._model = make_cubify_transformer(dimension=dimension, depth_model=True)
        self._model.load_state_dict(state, strict=True)
        self._model.eval().to(self.device)
        self._augmentor = Augmentor(("wide/image", "wide/depth"))
        self._preprocessor = Preprocessor()
        self._move = move_input_to_current_device
        self._ImageInfo, self._DepthInfo = ImageMeasurementInfo, DepthMeasurementInfo
        self._SensorInfo, self._ArrayInfo = PosedSensorInfo, SensorArrayInfo
        self.model_version = _checkpoint_version(weights)
        self._lock = threading.Lock()
        self._closed = False

    def _sample(self, frame):
        # Quarter-resolution depth matches the released RGB-D model's windows.
        from PIL import Image
        torch = self._torch
        rgb, depth = validate_frame(frame)
        h, w = depth.shape
        scale = min(1., self.max_side / max(w, h))
        # Multiples of four preserve the precise RGB/depth ratio.
        nw, nh = max(4, int(w*scale)//4*4), max(4, int(h*scale)//4*4)
        if (nw, nh) != (w, h):
            rgb = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR))
        depth = np.asarray(Image.fromarray(depth).resize((nw//4, nh//4), Image.Resampling.NEAREST))
        if not np.any(depth > 0):
            raise ValueError("No valid depth remains after model downsampling")
        k = frame.intrinsics
        K = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]], dtype=np.float32)
        K[0] *= nw/w
        K[1] *= nh/h
        Kd = K.copy()
        Kd[:2] /= 4
        wide = self._SensorInfo()
        wide.image = self._ImageInfo(size=(nw, nh), K=torch.from_numpy(K)[None])
        wide.depth = self._DepthInfo(size=(nw//4, nh//4), K=torch.from_numpy(Kd)[None])
        wide.RT = torch.eye(4)[None]
        wide.T_gravity = torch.from_numpy(gravity_rotation(frame))[None]
        sensors = self._ArrayInfo(wide=wide)
        return {"wide": {
            "image": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None],
            "depth": torch.from_numpy(np.array(depth, dtype=np.float32, copy=True))[None]},
            "sensor_info": sensors}

    def detect(self, frame: RGBDFrame, *, score_threshold: float | None = None) -> CuboidDetectionResult:
        score_threshold = self.score_threshold if score_threshold is None else threshold(score_threshold)
        with self._lock:
            if self._closed:
                raise RuntimeError("Cubify client is closed")
            sample = self._sample(frame)
            packaged = self._augmentor.package(sample)
            packaged = self._move(packaged, self._model.pixel_mean)
            batch = self._preprocessor.preprocess([packaged])
            autocast = (self._torch.autocast("cuda", dtype=self._torch.float16)
                        if self.fp16 and self.device.type == "cuda" else nullcontext())
            with self._torch.inference_mode(), autocast:
                output = self._model(batch)[0]
            # Upstream applies T_gravity to orientation already. Do NOT rotate twice.
            return make_result(frame, output.scores.detach().float().cpu().numpy(),
                               output.pred_boxes_3d.corners.detach().float().cpu().numpy(),
                               self.model_version, score_threshold)

    def healthz(self):
        return {"ok": not self._closed, "model_id": MODEL_ID,
                "model_version": self.model_version, "device": str(self.device),
                "precision": "fp16" if self.fp16 and self.device.type == "cuda" else "fp32"}

    def close(self):
        with self._lock:
            self._model = None
            self._closed = True


class CubifyAnythingRemote:
    def __init__(self, server_url, *, timeout_s=None, score_threshold=None):
        timeout_s = 60. if timeout_s is None else timeout_s
        score_threshold = .25 if score_threshold is None else score_threshold
        from urllib.parse import urlparse
        import requests
        if not isinstance(server_url, str) or urlparse(server_url).scheme not in ("http", "https") or not urlparse(server_url).netloc:
            raise ValueError("server_url must be an HTTP(S) base URL")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.score_threshold = threshold(score_threshold)
        self._session = requests.Session()
        self._closed = False

    def detect(self, frame: RGBDFrame, *, score_threshold: float | None = None) -> CuboidDetectionResult:
        if self._closed:
            raise RuntimeError("Cubify client is closed")
        score = self.score_threshold if score_threshold is None else threshold(score_threshold)
        response = self._session.post(self.server_url + "/v1/cubify/detect",
                                      json=encode_frame(frame, score), timeout=self.timeout_s)
        response.raise_for_status()
        return decode_result(frame, response.json(), score)

    def healthz(self):
        if self._closed:
            raise RuntimeError("Cubify client is closed")
        response = self._session.get(self.server_url + "/healthz", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def close(self):
        self._session.close()
        self._closed = True
