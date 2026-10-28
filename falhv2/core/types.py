"""Boundary types for future robot mapping adapters.

SI units; map/base axes x-forward, y-left, z-up; optical axes x-right,
y-down, z-forward. Arrays are owned by the producer and treated as read-only.
These contracts do not validate runtime payloads: adapters must validate them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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


@dataclass(frozen=True)
class InstanceMask:
    """Category-free instance: HxW boolean mask in the ORIGINAL RGB pixel grid."""
    mask: np.ndarray
    confidence: float


@dataclass(frozen=True)
class SegmentationResult:
    """Echo frame_id to reject stale/mismatched asynchronous model responses."""
    frame_id: str
    model_id: str
    model_version: str
    instances: tuple[InstanceMask, ...]


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


@dataclass(frozen=True)
class ObjectTrack:
    """Persistent identity and AABB in map coordinates; no category is required.

    Bounds estimate observed geometry, not guaranteed full object extent.
    Evidence IDs must allow re-integration after SLAM pose corrections.
    """
    object_id: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    state: Literal["tentative", "confirmed", "stale", "removed"]
    first_seen_ns: int
    last_seen_ns: int
    observation_count: int
    evidence_frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class MapSnapshot:
    """Consistent grid/object snapshot, all expressed in grid.map_revision.

    revision versions the fused map; grid.map_revision identifies the SLAM
    coordinate solution. Never mix boxes from a different SLAM solution.
    """
    revision: str
    grid: GridSnapshot
    objects: tuple[ObjectTrack, ...]
