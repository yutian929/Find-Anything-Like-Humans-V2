"""Category-free model ports for local OR remote inference.

Implementations normalize results to the source RGB pixel grid, carry frame_id,
validate output shape/confidence and report errors explicitly. An unavailable
model must never be represented by a successful empty detection result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING
from ..robot_adapters.base import RGBDFrame


if TYPE_CHECKING:
    import numpy as np


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
class Cuboid3D:
    """Single-frame prediction; corners are (8,3), meters, optical camera frame."""
    confidence: float
    corners_camera_m: np.ndarray


@dataclass(frozen=True)
class CuboidDetectionResult:
    """No persistent object IDs or semantic categories are assigned here."""
    frame_id: str
    timestamp_ns: int
    camera_frame: str
    model_id: str
    model_version: str
    cuboids: tuple[Cuboid3D, ...]


class InstanceSegmenter(Protocol):
    """Local and remote implementations expose exactly the same result contract.

    A remote implementation sends RGB plus request metadata; depth, poses and
    map fusion stay local unless a future protocol explicitly needs them.
    Heavy libraries, model weights and HTTP clients are loaded only when used.
    """

    def segment(self, frame: RGBDFrame, *, timeout_s: float) -> SegmentationResult: ...

    def close(self) -> None: ...


class CuboidDetector(Protocol):
    """Category-free metric 3D cuboids; distinct from a pixel-mask segmenter.

    Local inference is synchronous; only remote clients offer HTTP timeouts.
    No temporal association, persistent IDs or map fusion happens here.
    """

    def detect(self, frame: RGBDFrame, *, score_threshold: float | None = None) -> CuboidDetectionResult: ...

    def healthz(self) -> dict: ...

    def close(self) -> None: ...
