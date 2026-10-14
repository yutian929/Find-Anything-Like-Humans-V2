"""Shared finite path candidates and point observation geometry for both drivers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Station:
    index: int
    region_id: int
    cell: tuple[int, int]
    position: tuple[float, float]
    route_indices: tuple[int, ...]


@dataclass
class CandidateResult:
    station_indices: list[int]
    yaws: list[float]
    supports: dict[int, dict[int, set[str]]]
    stats: dict


def yaw_grid(step_deg: float = 30.0) -> list[float]:
    if not math.isfinite(step_deg) or not 0 < step_deg <= 360:
        raise ValueError("yaw step must be in (0, 360] degrees")
    return np.deg2rad(np.arange(0.0, 360.0, step_deg)).tolist()


def validate_objects(objects: Sequence[dict]) -> list[dict]:
    result, ids = [], set()
    for obj in objects:
        identifier = obj.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Objects require unique nonempty string IDs")
        try:
            point = tuple(float(v) for v in obj["position"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid point for object {identifier}") from exc
        if len(point) != 3 or not all(math.isfinite(v) for v in point):
            raise ValueError(f"Object {identifier} requires a finite 3D point")
        ids.add(identifier)
        result.append({"id": identifier, "position": list(point)})
    return result


def stations_from_plan(path_plan: dict) -> list[Station]:
    """First-visit path order; repeated physical locations retain all visit indices."""
    grid = path_plan["grid"]
    height, resolution = grid["height"], grid["resolution"]
    origin = grid["origin"]
    stations = []
    for region in path_plan["regions"]:
        visits = {}
        for route_index, cell in enumerate(region["route_cells"]):
            visits.setdefault(tuple(cell), []).append(route_index)
        for (row, col), indices in visits.items():
            position = (origin[0] + (col+0.5)*resolution,
                        origin[1] + (height-row-0.5)*resolution)
            stations.append(Station(len(stations), region["id"], (row, col), position, tuple(indices)))
    return stations


class PointCamera:
    """Pinhole center-point visibility with native zero-distortion intrinsics.

    Base axes: forward/left/up. Mount R = Rz(yaw) Ry(pitch) Rx(roll)
    with right-hand positive angles (positive pitch looks downward). Optical axes
    are right/down/forward. Range is Euclidean optical-center to target distance.
    Occlusion, target extent and recognition quality are not evaluated.
    """
    def __init__(self, config: dict):
        try:
            camera, mount, quality = config["camera"], config["mount"], config["observation_quality"]
            self.width, self.height = int(camera["width"]), int(camera["height"])
            if self.width != camera["width"] or self.height != camera["height"]:
                raise ValueError("image dimensions must be integers")
            self.fx, self.fy, self.cx, self.cy = (float(camera[k]) for k in ("fx", "fy", "cx", "cy"))
            self.offset = np.asarray(mount["position_in_base_m"], dtype=float)
            roll, pitch, yaw = (math.radians(float(mount[k])) for k in ("roll_deg", "pitch_deg", "yaw_deg"))
            self.min_distance, self.max_distance = (float(quality[k]) for k in ("min_distance_m", "max_distance_m"))
            coefficients = np.asarray(camera["distortion_coefficients"], dtype=float)
            if not isinstance(camera["distortion_model"], str):
                raise ValueError("distortion model is required")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Camera config requires valid intrinsics, measured mount and finite observation distances; fill every null") from exc
        values = [self.fx, self.fy, self.cx, self.cy, roll, pitch, yaw, self.min_distance, self.max_distance]
        if (self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0
                or not all(math.isfinite(v) for v in values)
                or self.offset.shape != (3,) or not np.isfinite(self.offset).all()
                or coefficients.shape != (5,) or not np.isfinite(coefficients).all()
                or not 0 <= self.min_distance < self.max_distance):
            raise ValueError("Invalid camera dimensions, mount, calibration or distance interval")
        if np.any(coefficients != 0):
            raise ValueError("This experiment requires zero-distortion/native pinhole inputs; nonzero distortion is not silently ignored")
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        optical_to_flu = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        self.base_from_optical = rz @ ry @ rx @ optical_to_flu
        self.offset_xy_norm = float(np.linalg.norm(self.offset[:2]))
        self.config = config

    def project(self, base_positions: np.ndarray, yaws: np.ndarray, point: Sequence[float]):
        positions = np.asarray(base_positions, dtype=float).reshape(-1, 2)
        yaws = np.asarray(yaws, dtype=float).reshape(-1)
        if len(positions) != len(yaws):
            raise ValueError("One yaw per base position required")
        point = np.asarray(point, dtype=float)
        dx, dy = point[0]-positions[:, 0], point[1]-positions[:, 1]
        c, s = np.cos(yaws), np.sin(yaws)
        in_base = np.column_stack((c*dx+s*dy, -s*dx+c*dy, np.full(len(yaws), point[2]))) - self.offset
        distance = np.linalg.norm(in_base, axis=1)
        optical = in_base @ self.base_from_optical
        depth = optical[:, 2]
        safe_depth = np.where(np.abs(depth) > 1e-12, depth, np.nan)
        u = self.fx*optical[:, 0]/safe_depth + self.cx
        v = self.fy*optical[:, 1]/safe_depth + self.cy
        return u, v, depth, distance

    def visible(self, base_positions: np.ndarray, yaws: np.ndarray, point: Sequence[float]) -> np.ndarray:
        u, v, depth, distance = self.project(base_positions, yaws, point)
        eps = 1e-9
        return ((depth > 1e-12) & (distance+eps >= self.min_distance) & (distance <= self.max_distance+eps)
                & (u >= -eps) & (u <= self.width-1+eps) & (v >= -eps) & (v <= self.height-1+eps))


def coverage_edges(result: CandidateResult) -> set[tuple[int, int, str]]:
    return {(i, yaw, obj) for i, directions in result.supports.items()
            for yaw, objects in directions.items() for obj in objects}
