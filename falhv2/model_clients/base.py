"""Category-free segmentation port for local OR remote inference.

Implementations normalize results to the source RGB pixel grid, carry frame_id,
validate mask shape/confidence and report errors explicitly. An unavailable
model must never be represented by a successful empty detection result.
"""
from __future__ import annotations

from typing import Protocol
from ..core.types import RGBDFrame, SegmentationResult


class InstanceSegmenter(Protocol):
    """Local and remote implementations expose exactly the same result contract.

    A remote implementation sends RGB plus request metadata; depth, poses and
    map fusion stay local unless a future protocol explicitly needs them.
    Heavy libraries, model weights and HTTP clients are loaded only when used.
    """

    def segment(self, frame: RGBDFrame, *, timeout_s: float) -> SegmentationResult: ...

    def close(self) -> None: ...
