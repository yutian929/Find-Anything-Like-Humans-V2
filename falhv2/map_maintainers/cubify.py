"""Convert single-frame cuboids to map AABB observations; no temporal fusion."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..robot_adapters.base import RGBDFrame
from ..model_clients.base import CuboidDetectionResult
from ..robot_adapters.base import transform
from ..model_clients.cubify_anything import make_result


@dataclass(frozen=True)
class AABBObservation:
    frame_id: str
    timestamp_ns: int
    clock_domain: str
    map_id: str
    map_revision: str
    detection_index: int
    confidence: float
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]


def cuboids_to_map_aabbs(frame: RGBDFrame, result: CuboidDetectionResult) -> tuple[AABBObservation, ...]:
    """Transform all 8 original corners FIRST, then take map-axis min/max.

    detection_index is frame-local, never a persistent object ID. MapBuilder
    chooses its replacement policy and must handle later SLAM corrections.
    """
    if (result.frame_id, result.timestamp_ns, result.camera_frame) != (
            frame.frame_id, frame.timestamp_ns, frame.camera_frame):
        raise ValueError("Detection and RGB-D frame do not match")
    T = transform(frame.T_map_camera)
    # Validate public dataclass inputs as well as results from built-in clients.
    make_result(frame, [b.confidence for b in result.cuboids],
                          [b.corners_camera_m for b in result.cuboids],
                          result.model_version, 0.)
    observations = []
    for index, box in enumerate(result.cuboids):
        corners = np.asarray(box.corners_camera_m) @ T[:3, :3].T + T[:3, 3]
        observations.append(AABBObservation(
            frame.frame_id, frame.timestamp_ns, frame.clock_domain,
            frame.map_id, frame.map_revision, index, box.confidence,
            tuple(float(v) for v in corners.min(axis=0)),
            tuple(float(v) for v in corners.max(axis=0))))
    return tuple(observations)
