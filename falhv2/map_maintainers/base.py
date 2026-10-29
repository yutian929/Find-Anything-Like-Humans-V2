"""Future map-building port; no SLAM, fusion or hardware backend is implemented."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Literal
from ..robot_adapters.base import GridSnapshot, RGBDFrame
from ..model_clients.base import SegmentationResult


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


class MapBuilder(Protocol):
    """Fuse observations into persistent category-free object tracks.

    Require matching frame/result IDs and grid/frame map coordinates. Preserve
    evidence for loop-closure re-integration. snapshot() returns a consistent
    copy; updates must not mutate an already-published snapshot.
    """

    def set_grid(self, grid: GridSnapshot) -> None: ...

    def integrate(self, frame: RGBDFrame, result: SegmentationResult) -> None: ...

    def snapshot(self) -> MapSnapshot: ...
