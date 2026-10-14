"""Target-first inverse observation supports on a finite set of path stations.

Each target queries nearby stations, then applies the shared exact camera model
to every declared yaw. Spatial screening only removes impossible distances; it
does not approximate the field of view or silently discard low-vote stations.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .common import CandidateResult, PointCamera, Station, validate_objects, yaw_grid


def generate(
    stations: list[Station],
    objects: list[dict],
    camera: PointCamera,
    *,
    yaw_step_deg: float = 30.0,
    station_indices: list[int] | None = None,
) -> CandidateResult:
    """Compute supports, conservatively screening station-target distances.

    ``station_indices`` contains Station.index values, not list offsets. The
    declared finite pool and yaw grid are returned even for empty supports.
    Chunking bounds temporary point-pose arrays at roughly 65,536 rows (unless
    one yaw grid alone exceeds that size).
    """
    started = perf_counter()
    objects = validate_objects(objects)
    yaws = yaw_grid(yaw_step_deg)
    by_index = {station.index: station for station in stations}
    if len(by_index) != len(stations):
        raise ValueError("Station indices must be unique")
    selected = list(by_index) if station_indices is None else list(station_indices)
    if len(set(selected)) != len(selected):
        raise ValueError("Selected station indices must be unique")
    if any(index not in by_index for index in selected):
        raise ValueError("Selected station index is not in the station pool")
    positions = np.asarray([by_index[index].position for index in selected], dtype=float).reshape(-1, 2)
    supports: dict[int, dict[int, set[str]]] = {}
    screened_pairs = 0
    exact_tests = 0
    n_yaws = len(yaws)
    chunk_size = max(1, 65_536 // n_yaws)
    yaw_array = np.asarray(yaws, dtype=float)
    # Triangle inequality: |target.xy-base.xy| <= max_range + |mount.xy|.
    # Tiny outward slack preserves boundary candidates under float arithmetic;
    # the shared exact camera predicate still makes the final decision.
    radius = camera.max_distance + camera.offset_xy_norm + 1e-7
    if selected and objects:
        tree = cKDTree(positions)
        for obj in objects:
            point = np.asarray(obj["position"], dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("Object positions must be finite 3D points")
            nearby = sorted(tree.query_ball_point(point[:2], radius))
            screened_pairs += len(nearby)
            for start in range(0, len(nearby), chunk_size):
                local = nearby[start : start + chunk_size]
                base = np.repeat(positions[local], n_yaws, axis=0)
                headings = np.tile(yaw_array, len(local))
                visible = np.asarray(camera.visible(base, headings, point), dtype=bool).reshape(len(local), n_yaws)
                exact_tests += len(local) * n_yaws
                rows, cols = np.nonzero(visible)
                for row, col in zip(rows.tolist(), cols.tolist()):
                    index = selected[local[row]]
                    supports.setdefault(index, {}).setdefault(col, set()).add(obj["id"])
    elapsed = perf_counter() - started
    return CandidateResult(
        station_indices=selected,
        yaws=yaws,
        supports=supports,
        stats={
            "elapsed_s": elapsed,
            "generation_seconds": elapsed,
            "evaluated_point_pose_pairs": exact_tests,
            "exact_point_pose_tests": exact_tests,
            "screened_station_object_pairs": screened_pairs,
            "total_station_object_pairs": len(selected) * len(objects),
            "station_count": len(selected),
            "yaw_count": n_yaws,
            "object_count": len(objects),
            "scope": "all declared yaws at every station in the finite reference pool",
            "screening": "conservative XY radius, followed by shared exact visibility",
        },
    )
