"""Map snapshots and maintenance port, independent of robot and model implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Literal
from ..robot_adapters.base import GridSnapshot, RGBDFrame
from ..model_clients.base import CuboidDetectionResult


@dataclass(frozen=True)
class ObjectTrack:
    """Object record and AABB in map coordinates; no category is required.

    Bounds estimate observed geometry, not guaranteed full object extent.
    Evidence IDs identify source frames; replay requires separately retained data.
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

    revision versions the maintained map; grid.map_revision identifies the SLAM
    coordinate solution. Never mix boxes from a different SLAM solution.
    """
    revision: str
    grid: GridSnapshot
    objects: tuple[ObjectTrack, ...]


class MapBuilder(Protocol):
    """Maintain category-free object records from posed cuboid observations.

    Require matching frame/result IDs and grid/frame map coordinates. Reject incompatible coordinate revisions or rebuild from retained evidence. snapshot() returns a consistent
    copy; updates must not mutate an already-published snapshot.
    """

    def set_grid(self, grid: GridSnapshot) -> None: ...

    def integrate(self, frame: RGBDFrame, result: CuboidDetectionResult) -> None: ...

    def snapshot(self) -> MapSnapshot: ...
