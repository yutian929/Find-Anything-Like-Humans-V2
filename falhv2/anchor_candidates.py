"""Build the complete finite station/yaw pool and qualified observation relations."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

try:
    from . import config as settings
    from .observation_model import PointCamera, BoxObserver, validate_objects
except ImportError:
    import config as settings
    from observation_model import PointCamera, BoxObserver, validate_objects


@dataclass(frozen=True)
class Station:
    index: int
    region_id: int
    cell: tuple[int, int]
    position: tuple[float, float]


@dataclass
class CandidateResult:
    station_indices: list[int]
    yaws: list[float]
    supports: dict[int, dict[int, set[str]]]
    stats: dict
    scores: dict[int, dict[int, dict[str, float]]] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def yaw_grid(step_deg: float | None = None) -> list[float]:
    step_deg = settings.YAW_STEP_DEG if step_deg is None else step_deg
    if isinstance(step_deg, bool) or not isinstance(step_deg, (int, float)) or not math.isfinite(step_deg) or not 0 < step_deg <= 360:
        raise ValueError("yaw step must be in (0, 360] degrees")
    return np.deg2rad(np.arange(0.0, 360.0, step_deg)).tolist()


def stations_from_regions(safe_regions: dict) -> list[Station]:
    """One candidate per safe cell, ordered by region ID, row, then column.

    Indices provide deterministic selection ties; they are not a visit order.
    """
    grid = safe_regions["grid"]
    height, resolution = grid["height"], grid["resolution"]
    origin = grid["origin"]
    stations = []
    for region in sorted(safe_regions["regions"], key=lambda region: region["id"]):
        for row, col in sorted(region["cells"]):
            position = (origin[0] + (col+0.5)*resolution,
                        origin[1] + (height-row-0.5)*resolution)
            stations.append(Station(len(stations), region["id"], (row, col), position))
    return stations


def generate_candidates(
    stations: list[Station],
    objects: list[dict],
    camera: PointCamera,
    *,
    yaw_step_deg: float | None = None,
    station_indices: list[int] | None = None,
) -> CandidateResult:
    """Build weighted supports after height, mount and frustum region screening.

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
    scores: dict[int, dict[int, dict[str, float]]] = {}
    observer = BoxObserver(camera, objects) if objects and 'aabb' in objects[0] else None
    rejection_counts = Counter()
    diagnostics = {}
    skipped_occlusion = 0
    diagnostic_recomputed = 0

    def record_diagnostic(obj, metrics, base, headings, orders):
        # Deferred geometric rejections cannot compete with a qualified pose.
        # Preserve the original pose order when filling in failed diagnostics later.
        complete = [i for i, row in enumerate(metrics) if row['observation_score'] is not None]
        if not complete:
            return
        def rank(i):
            row = metrics[i]
            return (row['qualified'], row['observation_score'], row['visible_pixels'],
                    -row['center_distance_m'], -int(orders[i]))
        best = max(complete, key=rank)
        previous = diagnostics.get(obj['id'])
        if previous is None or rank(best) > previous['rank']:
            diagnostics[obj['id']] = dict(rank=rank(best), position=base[best].tolist(),
                yaw_rad=float(headings[best]), metrics=metrics[best])

    screened_pairs = 0
    exact_tests = 0
    region_pose_tests = 0
    region_rejected = 0
    n_yaws = len(yaws)
    chunk_size = max(1, 65_536 // n_yaws)
    yaw_array = np.asarray(yaws, dtype=float)
    # Triangle inequality: |target.xy-base.xy| <= max_range + |mount.xy|.
    # Tiny outward slack preserves boundary candidates under float arithmetic;
    # Range is defined to the object center in both observation models.
    radius = camera.max_distance + camera.offset_xy_norm + 1e-7
    if selected and objects:
        tree = cKDTree(positions)
        for object_index, obj in enumerate(objects):
            deferred = []
            point = np.asarray(obj["position"], dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("Object positions must be finite 3D points")
            # Tighten XY distance by the actual camera height before yaw tests.
            height = point[2]-camera.offset[2] if observer else 0.0
            object_radius = (math.sqrt(max(0, camera.max_distance**2-height**2))
                             + camera.offset_xy_norm+1e-7) if observer else radius
            nearby = (sorted(tree.query_ball_point(point[:2], object_radius))
                      if abs(height) <= camera.max_distance+1e-9 else [])
            screened_pairs += len(nearby)
            for start in range(0, len(nearby), chunk_size):
                local = nearby[start : start + chunk_size]
                base = np.repeat(positions[local], n_yaws, axis=0)
                headings = np.tile(yaw_array, len(local))
                if observer:
                    # Construct the mount/pitch/frustum-aware region BEFORE costly
                    # projection sampling and occlusion. Bounds preserve partial boxes.
                    admissible = observer.envelope(object_index, base, headings)[3]
                    region_pose_tests += len(base)
                    region_rejected += int((~admissible).sum())
                    evaluated = np.flatnonzero(admissible)
                    if not len(evaluated):
                        continue
                    metrics = observer.evaluate(object_index, base[evaluated], headings[evaluated],
                                                candidate_only=True)
                    visible = np.zeros(len(base), dtype=bool)
                    visible[evaluated] = [row['qualified'] for row in metrics]
                    visible = visible.reshape(len(local), n_yaws)
                    metrics_by_pose = dict(zip(evaluated.tolist(), metrics))
                    rejection_counts.update(row['reason'] for row in metrics)
                    orders = np.asarray(local)[evaluated // n_yaws]*n_yaws + evaluated % n_yaws
                    record_diagnostic(obj, metrics, base[evaluated], headings[evaluated], orders)
                    skipped = np.array([i for i, row in enumerate(metrics)
                                        if row['observation_score'] is None], dtype=int)
                    if len(skipped):
                        skipped_occlusion += len(skipped)
                        deferred.append((base[evaluated[skipped]], headings[evaluated[skipped]], orders[skipped]))
                else:
                    visible = np.asarray(camera.visible(base, headings, point), dtype=bool).reshape(len(local), n_yaws)
                exact_tests += len(evaluated) if observer else len(local)*n_yaws
                rows, cols = np.nonzero(visible)
                for row, col in zip(rows.tolist(), cols.tolist()):
                    index = selected[local[row]]
                    supports.setdefault(index, {}).setdefault(col, set()).add(obj["id"])
                    if observer:
                        scores.setdefault(index, {}).setdefault(col, {})[obj['id']] = metrics_by_pose[row*n_yaws+col]['observation_score']
            if observer and not diagnostics.get(obj['id'], {}).get('rank', (False,))[0]:
                # No qualified pose exists: retain the exact best-failure diagnostic,
                # including visibility, blocker IDs, refinement and deterministic ties.
                for base, headings, orders in deferred:
                    metrics = observer.evaluate(object_index, base, headings)
                    diagnostic_recomputed += len(base)
                    record_diagnostic(obj, metrics, base, headings, orders)
    elapsed = perf_counter() - started
    return CandidateResult(
        station_indices=selected,
        yaws=yaws,
        supports=supports,
        scores=scores,
        diagnostics={identifier: {key: value for key, value in row.items() if key != 'rank'} for identifier, row in diagnostics.items()},
        stats={
            "elapsed_s": elapsed,
            "generation_seconds": elapsed,
            "evaluated_object_pose_pairs": exact_tests,
            "region_pose_tests": region_pose_tests,
            "region_rejected_pose_pairs": region_rejected,
            "occlusion_skipped_pose_pairs": skipped_occlusion,
            "diagnostic_recomputed_pose_pairs": diagnostic_recomputed,
            **({"evaluated_point_pose_pairs": exact_tests, "exact_point_pose_tests": exact_tests}
               if observer is None else {}),
            "screened_station_object_pairs": screened_pairs,
            "total_station_object_pairs": len(selected) * len(objects),
            "station_count": len(selected),
            "yaw_count": n_yaws,
            "object_count": len(objects),
            "scope": "all declared yaws at every station in the finite reference pool",
            "screening": "height-aware XY query, mount-aware radial/frustum envelope, then sampled AABB" if observer else "conservative XY radius, followed by shared exact visibility",
            "observation_model": "aabb_weighted_region" if observer else "legacy_point",
            "observation_reasons": dict(rejection_counts),
        },
    )


def object_heatmaps(candidates, object_ids):
    """Sparse per-object maxima: [station, score, yaw, integer_vote_units].

    Keep full precision in exports. Aggregate voting/heatmaps round each object's
    station contribution to micro-units identically to the greedy baseline.
    """
    maps = {identifier: [] for identifier in object_ids}
    for station, directions in sorted(candidates.supports.items()):
        best = {}
        for yaw, objects in sorted(directions.items()):
            for obj in objects:
                score = candidates.scores.get(station, {}).get(yaw, {}).get(obj, 1.0)
                if obj not in best or score > best[obj][0]:
                    best[obj] = (score, yaw)
        for obj, (score, yaw) in best.items():
            maps[obj].append([station, score, yaw, round(1_000_000*score)])
    return maps
