"""Future map-building port; no SLAM, fusion or hardware backend is implemented."""
from __future__ import annotations

from typing import Protocol
from ..core.types import GridSnapshot, MapSnapshot, RGBDFrame, SegmentationResult


class MapBuilder(Protocol):
    """Fuse observations into persistent category-free object tracks.

    Require matching frame/result IDs and grid/frame map coordinates. Preserve
    evidence for loop-closure re-integration. snapshot() returns a consistent
    copy; updates must not mutate an already-published snapshot.
    """

    def set_grid(self, grid: GridSnapshot) -> None: ...

    def integrate(self, frame: RGBDFrame, result: SegmentationResult) -> None: ...

    def snapshot(self) -> MapSnapshot: ...
