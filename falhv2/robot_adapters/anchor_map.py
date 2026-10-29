"""RGB-D publisher and Slamtec read-only adapter; independent of old global config."""
from __future__ import annotations

import hashlib
import math
import struct
import threading
import time

import numpy as np

from .base import (BasePose, CameraIntrinsics, GridSnapshot, MAX_PIXELS,
                   RGBDFrame, transform, validate_frame)


def from_anchor_map_frame(frame, *, map_id, map_revision, camera_frame="color_optical"):
    """Convert the legacy BGR/z16 Frame; caller supplies the SLAM revision.

    Legacy stamp is UNIX seconds as float, so original nanosecond precision
    cannot be recovered. camera_pose must be the synchronized T_map_camera.
    Depth must already be aligned to the rectified color image.
    """
    bgr, raw_depth = np.asarray(frame.rgb), np.asarray(frame.depth)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Legacy frame.rgb must be HxWx3 uint8 BGR")
    if raw_depth.dtype != np.uint16 or raw_depth.shape != bgr.shape[:2]:
        raise ValueError("Legacy depth must be aligned HxW uint16 z16")
    intr = frame.intrinsics
    scale = float(intr["depth_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale must be a positive meter-per-unit value")
    stamp = float(frame.stamp)
    if not math.isfinite(stamp) or stamp <= 0:
        raise ValueError("Frame stamp must be positive UNIX seconds")
    h, w = raw_depth.shape
    result = RGBDFrame(
        frame_id=str(frame.frame_id), timestamp_ns=round(stamp * 1_000_000_000),
        clock_domain="unix", map_id=map_id, map_revision=map_revision,
        camera_frame=camera_frame,
        intrinsics=CameraIntrinsics(w, h, float(intr["fx"]), float(intr["fy"]),
                                    float(intr["cx"]), float(intr["cy"])),
        rgb=np.ascontiguousarray(bgr[..., ::-1]),
        depth_m=raw_depth.astype(np.float32) * scale,
        T_map_camera=np.array(frame.camera_pose, dtype=np.float64, copy=True),
        T_map_base=(transform(frame.base_pose).copy()
                    if getattr(frame, "base_pose", None) is not None else None))
    validate_frame(result)
    return result

# Current Lenovo transport: synchronized RGB-D over ZMQ, SLAM data over REST.


def _positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def decode_rgbd_pose_message(parts, *, map_id, map_revision, topic="rgbd.pose"):
    """Decode publisher v2 directly, preserving the original Unix nanoseconds."""
    import msgpack
    if len(parts) != 4 or parts[0] != topic.encode():
        raise ValueError("Expected topic + metadata + RGB + depth")
    m = msgpack.unpackb(parts[1], raw=False)
    if not isinstance(m, dict) or m.get("version") != 2 or m.get("type") != "rgbd_pose":
        raise ValueError("Unsupported RGB-D protocol")
    if m.get("pose_valid") is not True or not isinstance(m.get("sync"),dict) or m["sync"].get("valid") is not True:
        raise ValueError("Publisher has no valid synchronized pose")
    for key, expected in (("rgb_encoding","rgb8"), ("depth_encoding","z16"),
                          ("depth_aligned_to","color"), ("T_map_camera_frame","color_optical"),
                          ("depth_frame","color_optical"), ("rgb_dtype","uint8"),
                          ("depth_dtype","uint16")):
        if m.get(key) != expected:
            raise ValueError(f"Unsupported {key}: {m.get(key)!r}")
    for key, expected in (("map_id",map_id), ("map_revision",map_revision)):
        if key in m and m[key] != expected:
            raise ValueError(f"Publisher {key} disagrees with configured SLAM session")
    w, h = m["width"], m["height"]
    if type(w) is not int or type(h) is not int or min(w,h) < 4 or w*h > MAX_PIXELS:
        raise ValueError("Invalid RGB-D dimensions")
    if m.get("depth_width") != w or m.get("depth_height") != h:
        raise ValueError("Depth dimensions differ from RGB")
    if len(parts[2]) != w*h*3 or len(parts[3]) != w*h*2:
        raise ValueError("RGB-D payload byte count mismatch")
    intr = m["color_intr"]
    if not isinstance(intr,dict):
        raise ValueError("color_intr must be a dictionary")
    if intr.get("width", w) != w or intr.get("height", h) != h:
        raise ValueError("Calibration dimensions differ from RGB")
    if not np.isfinite(intr.get("coeffs", [0])).all() or np.any(np.asarray(intr.get("coeffs", [0])) != 0):
        raise ValueError("Nonzero distortion: rectify RGB-D and calibration at publisher first")
    scale = _positive(float(m["depth_scale_m"]), "depth_scale_m")
    if type(m["frame_id"]) is not int or m["frame_id"] < 0:
        raise ValueError("frame_id must be a nonnegative integer")
    frame = RGBDFrame(
        str(m["frame_id"]), m["timestamp_ns"], "unix", map_id, map_revision,
        "color_optical", CameraIntrinsics(w,h,float(intr["fx"]),float(intr["fy"]),
                                          float(intr["ppx"]),float(intr["ppy"])),
        np.frombuffer(parts[2],np.uint8).reshape(h,w,3).copy(),
        np.frombuffer(parts[3],"<u2").reshape(h,w).astype(np.float32)*scale,
        transform(m["T_map_camera"]).copy(),
        T_map_base=transform(m["T_map_base"]).copy())
    validate_frame(frame)
    return frame


def decode_explore_map(raw, *, map_id, map_revision, timestamp_ns):
    """Slamtec explore grid: 20-byte header, optional 16-byte wrapper, cell bytes.

    Match the existing reader: 0 free, 127 unknown, other values occupied.
    The raw map starts at lower-left; planner rows start at the top.
    """
    if len(raw) < 20:
        raise ValueError("Explore map response is truncated")
    x,y,w,h,resolution = struct.unpack_from("<ffiif", raw)
    count = w*h
    if min(w,h) <= 0 or count > MAX_PIXELS or not np.isfinite([x,y,resolution]).all() or resolution <= 0:
        raise ValueError("Invalid explore map metadata")
    if len(raw) == 36+count:
        if struct.unpack_from("<I",raw,32)[0] != count:
            raise ValueError("Explore map wrapper length mismatch")
    elif len(raw) != 20+count:
        raise ValueError("Unsupported explore map length")
    cells = np.frombuffer(raw[-count:],np.uint8).reshape(h,w)
    rows = np.full((h,w), ord("#"), dtype=np.uint8)
    rows[cells==0] = ord(".")
    rows[cells==127] = ord("?")
    return GridSnapshot(map_id,map_revision,timestamp_ns,"unix",float(resolution),
                        (float(x),float(y)),
                        tuple(row.tobytes().decode("ascii") for row in rows[::-1]),
                        content_version=hashlib.sha256(raw).hexdigest())


class AnchorMapRobot:
    """Read-only adapter for the existing RGB-D publisher and Slamtec REST.

    Single-threaded lifecycle: open/use/close on the same thread. No background
    processes, camera ownership or motion commands. map_revision is supplied by
    the caller because the current publisher/REST APIs do not expose it.
    Recreate the adapter after SLAM reset, map switching or loop-closure revision.
    """

    def __init__(self, *, map_id, map_revision,
                 endpoint="ipc:///tmp/rgbd_pose.ipc", topic="rgbd.pose",
                 chassis_url="http://192.168.11.1:1448", max_frame_age_s=1.):
        from urllib.parse import urlparse
        if any(not isinstance(v,str) or not v for v in (map_id,map_revision,endpoint,topic)):
            raise ValueError("Nonempty map_id, map_revision, endpoint and topic required")
        if urlparse(chassis_url).scheme not in ("http","https") or not urlparse(chassis_url).netloc:
            raise ValueError("chassis_url must be an HTTP(S) URL")
        self.map_id, self.map_revision = map_id, map_revision
        self.endpoint, self.topic = endpoint, topic
        self.chassis_url = chassis_url.rstrip("/")
        self.max_frame_age_s = _positive(max_frame_age_s,"max_frame_age_s")
        self._socket = self._context = self._session = None
        self._owner = None

    def open(self):
        import requests
        import zmq
        if self._owner is not None:
            self._check_open()
            return
        self._owner = threading.get_ident()
        try:
            self._session = requests.Session()
            self._session.trust_env = False
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.LINGER,0)
            self._socket.setsockopt(zmq.RCVHWM,2)
            self._socket.setsockopt(zmq.MAXMSGSIZE,128*1024*1024)
            self._socket.setsockopt(zmq.SUBSCRIBE,self.topic.encode())
            self._socket.connect(self.endpoint)
        except Exception:
            self.close()
            raise

    def _check_open(self):
        if self._owner is None:
            raise RuntimeError("Call open() before reading robot data")
        if self._owner != threading.get_ident():
            raise RuntimeError("Robot adapter must be used on its opening thread")

    def capture(self, *, timeout_s=2.):
        """Wait for a fresh, valid frame; do not return a cached stale frame."""
        self._check_open()
        deadline = time.monotonic()+_positive(timeout_s,"timeout_s")
        last_error = None
        while time.monotonic() < deadline:
            remaining = deadline-time.monotonic()
            if not self._socket.poll(max(1,math.ceil(remaining*1000))):
                break
            parts = self._socket.recv_multipart()
            try:
                frame = decode_rgbd_pose_message(
                    parts,map_id=self.map_id,map_revision=self.map_revision,topic=self.topic)
                age = (time.time_ns()-frame.timestamp_ns)/1e9
                if age < -.1 or age > self.max_frame_age_s:
                    raise ValueError(f"Frame clock skew/staleness: age={age:.3f}s")
                return frame
            except (ValueError,TypeError,KeyError,OverflowError) as exc:
                last_error = exc
        raise TimeoutError(f"No fresh synchronized RGB-D frame within {timeout_s}s; last error: {last_error}")

    def _get(self, path, timeout_s):
        self._check_open()
        response = self._session.get(self.chassis_url+path,
                                     timeout=_positive(timeout_s,"timeout_s"))
        response.raise_for_status()
        return response

    def grid_snapshot(self, *, timeout_s=2.):
        """Fetch a new grid every call; no permanent map cache."""
        response = self._get("/api/core/slam/v1/maps/explore",timeout_s)
        return decode_explore_map(response.content,map_id=self.map_id,
                                  map_revision=self.map_revision,timestamp_ns=time.time_ns())

    def localization(self, *, timeout_s=2.):
        """Current REST pose, independent of camera availability.

        REST has no measurement timestamp or localization confidence. Use HTTP
        midpoint as an estimate; never replace a frame's synchronized pose.
        """
        start, mono = time.time_ns(), time.monotonic_ns()
        response = self._get("/api/core/slam/v1/localization/pose",timeout_s)
        end, rtt = time.time_ns(), time.monotonic_ns()-mono
        p = response.json()
        values = [float(p["x"]),float(p["y"]),float(p.get("z",0)),
                  float(p.get("roll",0)),float(p.get("pitch",0)),float(p["yaw"])]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite chassis localization")
        x,y,z,roll,pitch,yaw = values
        cr,sr,cp,sp,cy,sy = math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
        T = np.eye(4)
        T[:3,:3] = [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                    [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],
                    [-sp,cp*sr,cp*cr]]
        T[:3,3] = [x,y,z]
        return BasePose((start+end)//2,"unix",self.map_id,self.map_revision,
                        T,"http_midpoint",rtt/1e9)

    def close(self):
        if self._owner is not None:
            self._check_open()
        if self._socket is not None:
            self._socket.close(linger=0)
        if self._context is not None:
            self._context.term()
        if self._session is not None:
            self._session.close()
        self._socket = self._context = self._session = self._owner = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
