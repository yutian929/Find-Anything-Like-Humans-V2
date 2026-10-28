"""Robot ports. Implement vendor/ROS/replay adapters without changing planning."""
from __future__ import annotations

from typing import Protocol
from ..core.types import GridSnapshot, RGBDFrame


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

    def close(self) -> None: ...


class Navigator(Protocol):
    """Optional motion capability, intentionally separate from RobotSource."""

    def navigate_to(self, *, map_id: str, x_m: float, y_m: float,
                    yaw_rad: float, timeout_s: float) -> None:
        """Return only on arrival; fail explicitly on timeout/rejection."""
        ...

    def cancel(self) -> None: ...
