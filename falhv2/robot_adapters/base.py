"""Robot data and ports, independent of vendor SDKs.

SI units; map/base axes x-forward, y-left, z-up; optical axes x-right,
y-down, z-forward. Producer-owned arrays are treated as read-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Calibration of the rectified RGB image and depth aligned to that image."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RGBDFrame:
    """A synchronized frame; transform is evaluated at capture time, not inference time.

    rgb: HxWx3 uint8 RGB; depth_m: HxW float optical-Z depth in meters,
    with nonpositive/nonfinite values invalid. T_map_camera is a 4x4 rigid
    transform mapping optical coordinates to the specified map revision.
    timestamp_ns must identify a documented clock domain shared by the adapter.
    """
    frame_id: str
    timestamp_ns: int
    clock_domain: str
    map_id: str
    map_revision: str
    camera_frame: str
    intrinsics: CameraIntrinsics
    rgb: np.ndarray
    depth_m: np.ndarray
    T_map_camera: np.ndarray
    T_map_base: np.ndarray | None = None


@dataclass(frozen=True)
class GridSnapshot:
    """Axis-aligned planning grid; row 0 is top, origin is lower-left in meters.

    '.' free, '#' occupied, '?' unknown. Adapters must resample rotated grids,
    not silently discard an occupancy-map origin rotation.
    """
    map_id: str
    map_revision: str
    timestamp_ns: int
    clock_domain: str
    resolution: float
    origin: tuple[float, float]
    rows: tuple[str, ...]
    content_version: str = ""


@dataclass(frozen=True)
class BasePose:
    """Map-frame chassis pose; HTTP timestamps are estimates, not sensor time."""
    timestamp_ns: int
    clock_domain: str
    map_id: str
    map_revision: str
    T_map_base: np.ndarray
    timestamp_source: str
    rtt_s: float

    @property
    def x(self):
        return float(self.T_map_base[0, 3])

    @property
    def y(self):
        return float(self.T_map_base[1, 3])

    @property
    def yaw(self):
        import math
        return math.atan2(self.T_map_base[1, 0], self.T_map_base[0, 0])


class RobotSource(Protocol):
    """Read-only data lifecycle, independent of motion control.

    Implementations resolve capture-time transforms, synchronize RGB/depth,
    convert units and validate calibration. Localization loss, stale transforms,
    timeout or disconnection must raise an explicit error, never identity poses.
    Instantiate/open explicitly; importing a module must not connect hardware.
    """

    def open(self) -> None: ...

    def capture(self, *, timeout_s: float) -> RGBDFrame: ...

    def grid_snapshot(self, *, timeout_s: float) -> GridSnapshot:
        """Get the SLAM occupancy grid with its coordinate-revision identity."""
        ...

    def localization(self, *, timeout_s: float) -> BasePose: ...

    def close(self) -> None: ...


class Navigator(Protocol):
    """Optional motion capability, intentionally separate from RobotSource."""

    def navigate_to(self, *, map_id: str, x_m: float, y_m: float,
                    yaw_rad: float, timeout_s: float) -> None:
        """Return only on arrival; fail explicitly on timeout/rejection."""
        ...

    def cancel(self) -> None: ...


MAX_PIXELS = 4096 * 4096


def rotation(value, name="rotation"):
    import numpy as np
    r = np.asarray(value, dtype=np.float64)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(r), 1.0, atol=1e-4)):
        raise ValueError(f"{name} must be a finite proper rotation")
    return r


def transform(value):
    import numpy as np
    t = np.asarray(value, dtype=np.float64)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)):
        raise ValueError("T_map_camera must be a finite rigid 4x4 transform")
    rotation(t[:3, :3], "T_map_camera rotation")
    return t


def validate_frame(frame):
    import numpy as np
    k = frame.intrinsics
    for dimension in (k.width, k.height):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 4:
            raise ValueError("Image width/height must be integers >= 4")
    if k.width * k.height > MAX_PIXELS:
        raise ValueError("Image exceeds maximum supported pixel count")
    if not np.isfinite([k.fx, k.fy, k.cx, k.cy]).all() or min(k.fx, k.fy) <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths")
    if (not isinstance(frame.frame_id, str) or not frame.frame_id
            or not isinstance(frame.camera_frame, str) or not frame.camera_frame
            or isinstance(frame.timestamp_ns, bool)
            or not isinstance(frame.timestamp_ns, int) or frame.timestamp_ns <= 0):
        raise ValueError("Nonempty frame/camera IDs and a positive timestamp_ns are required")
    if any(not isinstance(v, str) or not v for v in (frame.map_id, frame.map_revision, frame.clock_domain)):
        raise ValueError("map_id, map_revision and clock_domain are required")
    rgb = np.asarray(frame.rgb)
    if rgb.shape != (k.height, k.width, 3) or rgb.dtype != np.uint8:
        raise ValueError("rgb must be HxWx3 uint8 RGB (not BGR)")
    depth = np.asarray(frame.depth_m)
    if depth.shape != rgb.shape[:2] or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("depth_m must be aligned HxW floating point depth in meters")
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
    if not np.isfinite(depth).all() or not np.any(depth > 0):
        raise ValueError("RGB-D inference requires at least one valid finite depth")
    transform(frame.T_map_camera)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth)
