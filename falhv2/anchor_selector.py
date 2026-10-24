"""锚点规划：安全区域 → 合格观察关系 → 最少锚点 → 最少定向照片。

默认从 data/map_aabb.json、data/robot.json 和 data/camera.d435i.json 读取输入，
输出离线 JSON / HTML / SVG，包含逐物体观察热力图及独立的旧贪心基线回放。
完整 AABB 使用距离权重乘未遮挡比例判定合格，旧点地图保留视场/距离模式。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
import math
from typing import Sequence

import numpy as np

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
import argparse
import heapq
import json
import logging

from scipy.spatial import cKDTree

try:
    from .regions_extractor import plan as find_safe_regions, diagnostic_svg, PlanningError
    from .observation_model import BoxObserver, validate_objects
    from . import config as settings
    from .config import read_input
    from .anchor_optimizer import optimize
except ImportError:
    from regions_extractor import plan as find_safe_regions, diagnostic_svg, PlanningError
    from observation_model import BoxObserver, validate_objects
    import config as settings
    from config import read_input
    from anchor_optimizer import optimize

ROOT = settings.ROOT


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
            raise ValueError("Anchor planning requires zero-distortion/native pinhole inputs; nonzero distortion is not silently ignored")
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
                    metrics = observer.evaluate(object_index, base[evaluated], headings[evaluated])
                    visible = np.zeros(len(base), dtype=bool)
                    visible[evaluated] = [row['qualified'] for row in metrics]
                    visible = visible.reshape(len(local), n_yaws)
                    metrics_by_pose = dict(zip(evaluated.tolist(), metrics))
                    rejection_counts.update(row['reason'] for row in metrics)
                    best_index = max(range(len(metrics)), key=lambda k: (metrics[k]['qualified'], metrics[k]['observation_score'], metrics[k]['visible_pixels'], -metrics[k]['center_distance_m']))
                    metric = metrics[best_index]
                    best_key = (metric['qualified'], metric['observation_score'], metric['visible_pixels'], -metric['center_distance_m'])
                    previous = diagnostics.get(obj['id'])
                    if previous is None or best_key > tuple(previous['rank']):
                        pose_index = evaluated[best_index]
                        diagnostics[obj['id']] = {'rank': best_key, 'position': base[pose_index].tolist(),
                            'yaw_rad': float(headings[pose_index]), 'metrics': metric}
                else:
                    visible = np.asarray(camera.visible(base, headings, point), dtype=bool).reshape(len(local), n_yaws)
                exact_tests += len(evaluated) if observer else len(local)*n_yaws
                rows, cols = np.nonzero(visible)
                for row, col in zip(rows.tolist(), cols.tolist()):
                    index = selected[local[row]]
                    supports.setdefault(index, {}).setdefault(col, set()).add(obj["id"])
                    if observer:
                        scores.setdefault(index, {}).setdefault(col, {})[obj['id']] = metrics_by_pose[row*n_yaws+col]['observation_score']
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

def select_anchors_greedy(candidates: CandidateResult, stations: list[Station], object_ids: list[str]) -> dict:
    targets = set(object_ids)
    station_by_id = {station.index: station for station in stations}
    station_sets = {i: set().union(*directions.values()) for i, directions in candidates.supports.items()}
    if any(not objects <= targets for objects in station_sets.values()):
        raise ValueError("Coverage contains unknown object IDs")
    if not set(station_sets) <= set(station_by_id):
        raise ValueError("Coverage contains unknown station IDs")
    def action_score(i, yaw, obj):
        return candidates.scores.get(i, {}).get(yaw, {}).get(obj, 1.0)

    # Integer micro-units keep lazy heap updates deterministic and exact.
    station_scores = {i: {obj: round(1_000_000*max(action_score(i, yaw, obj)
                            for yaw, ids in candidates.supports[i].items() if obj in ids))
                          for obj in ids} for i, ids in station_sets.items()}
    reverse = defaultdict(set)
    for i, objects in station_sets.items():
        for obj in objects:
            reverse[obj].add(i)
    remaining = set(targets)
    votes = {i: len(objects) for i, objects in station_sets.items()}
    score_votes = {i: sum(station_scores[i].values()) for i in station_sets}
    heap = [(-score_votes[i], -score, i) for i, score in votes.items() if score]
    heapq.heapify(heap)
    chosen, trace = [], []
    while heap and remaining:
        negative_score, negative, i = heapq.heappop(heap)
        if -negative != votes[i] or -negative_score != score_votes[i]:
            continue
        new = station_sets[i] & remaining
        if not new:
            continue
        chosen.append(i)
        trace.append({"station_index": i, "new_object_ids": sorted(new), "new_object_count": len(new),
                      "weighted_score": sum(station_scores[i][obj] for obj in new)/1_000_000})
        remaining.difference_update(new)
        for obj in new:
            for other in reverse[obj]:
                votes[other] -= 1
                score_votes[other] -= station_scores[other][obj]
                if votes[other] > 0:
                    heapq.heappush(heap, (-score_votes[other], -votes[other], other))
    # Keep an inclusion-minimal station cover, without moving any position.
    counts = Counter(obj for i in chosen for obj in station_sets[i])
    retained = set(chosen)
    prune_order = list(reversed(chosen))
    if candidates.scores:
        prune_order.sort(key=lambda i: sum(station_scores[i].values()))
    for i in prune_order:
        if all(counts[obj] > 1 for obj in station_sets[i]):
            retained.remove(i)
            counts.subtract(station_sets[i])
    covered = targets-remaining
    directions = {(i, yaw): objects for i in sorted(retained)
                  for yaw, objects in sorted(candidates.supports[i].items()) if objects}
    need = set(covered)
    actions = []
    while need:
        # After station selection, reduce photos first, score breaks action ties.
        key = min(directions, key=lambda key: (-len(directions[key] & need),
                  -sum(round(1_000_000*action_score(*key, obj)) for obj in directions[key] & need), key))
        if not (directions[key] & need):
            raise ValueError("Direction selection lost station coverage")
        actions.append(key)
        need.difference_update(directions[key])
    counts = Counter(obj for key in actions for obj in directions[key])
    prune_actions = list(reversed(actions))
    if candidates.scores:
        prune_actions.sort(key=lambda key: sum(round(1_000_000*action_score(*key, obj)) for obj in directions[key]))
    for key in prune_actions:
        if all(counts[obj] > 1 for obj in directions[key]):
            actions.remove(key)
            counts.subtract(directions[key])
    anchors = []
    for i in sorted(retained):
        station = station_by_id[i]
        observations = [{"yaw_index": yaw, "yaw_rad": candidates.yaws[yaw],
                         "object_ids": sorted(directions[(i, yaw)]),
                         "score_by_object": {obj: action_score(i, yaw, obj) for obj in sorted(directions[(i, yaw)])}}
                        for si, yaw in actions if si == i]
        if observations:
            anchors.append({"station_index": i, "region_id": station.region_id,
                            "position": list(station.position), "cell": list(station.cell),
                            "observations": observations})
    actual = {obj for anchor in anchors for action in anchor["observations"] for obj in action["object_ids"]}
    if actual != covered:
        raise ValueError("Final plan failed independent set-union coverage verification")
    return {"status": "empty_targets" if not targets else "complete" if not remaining else "partial",
            "anchor_count": len(anchors), "photo_count": sum(len(a["observations"]) for a in anchors),
            "coverage_fraction": len(actual)/len(targets) if targets else None,
            "covered_object_ids": sorted(actual), "uncovered_object_ids": sorted(remaining),
            "anchors": anchors, "vote_trace": trace,
            "optimality": "heuristic_not_globally_optimal",
            "navigation": "not_planned",
            "start_reachability": "not_evaluated_no_start_pose"}


def _assemble_anchors(actions, candidates, stations):
    by_id = {s.index: s for s in stations}
    anchors = []
    for i in sorted({i for i, _ in actions}):
        s = by_id[i]
        observations = []
        for si, y in actions:
            if si != i:
                continue
            ids = sorted(candidates.supports[i][y])
            observations.append(dict(yaw_index=y, yaw_rad=candidates.yaws[y], object_ids=ids,
                score_by_object={o: candidates.scores.get(i, {}).get(y, {}).get(o, 1.0) for o in ids}))
        anchors.append(dict(station_index=i, region_id=s.region_id, position=list(s.position),
                            cell=list(s.cell), observations=observations))
    return anchors


def select_anchors(candidates: CandidateResult, stations: list[Station], object_ids: list[str],
                   *, time_limit_s=None) -> dict:
    """Keep the weighted baseline; minimize station count before photo count."""
    if len({s.index for s in stations}) != len(stations):
        raise ValueError('Station IDs must be unique')
    if not set(candidates.supports) <= set(candidates.station_indices):
        raise ValueError('Support station is outside the declared candidate pool')
    if any(not isinstance(y, int) or not 0 <= y < len(candidates.yaws)
           for ds in candidates.supports.values() for y in ds):
        raise ValueError('Invalid support yaw index')
    baseline = select_anchors_greedy(candidates, stations, object_ids)
    keys = [(a['station_index'], o['yaw_index']) for a in baseline['anchors'] for o in a['observations']]
    solution = optimize(candidates.supports, object_ids, keys, time_limit_s=time_limit_s)
    selected = deepcopy(baseline)
    selected['vote_trace'] = []  # The integer solution has no greedy selection order.
    anchors = _assemble_anchors(solution['actions'], candidates, stations)
    selected.update({k: v for k, v in solution.items() if k != 'actions'})
    selected.update(anchors=anchors, anchor_count=solution['selected_cost'][0],
                    photo_count=solution['selected_cost'][1], baseline=baseline,
                    objective_order=['anchor_count', 'photo_count'],
                    optimality=('finite_pool_lexicographic_optimal'
                        if solution['optimization']['lexicographic_optimal'] else 'not_proven'))
    return selected


def _audit_observations(selected, objects, observer):
    """Audit all target/photo pairs; keep dense metrics separate from original q."""
    covered, rejected = set(), []
    started = perf_counter()
    count = 0
    for anchor in selected['anchors']:
        for action in anchor['observations']:
            metrics = {o['id']: observer.evaluate(i, [anchor['position']], [action['yaw_rad']])[0]
                       for i, o in enumerate(objects)}
            action['high_density_object_metrics'] = metrics
            visible = {oid for oid, row in metrics.items() if row['qualified']}
            covered.update(visible)
            count += len(objects)
            rejected.extend(dict(station_index=anchor['station_index'], yaw_index=action['yaw_index'],
                                 object_id=oid) for oid in sorted(set(action['object_ids']) - visible))
    missing = set(o['id'] for o in objects) - covered
    return dict(status='passed' if not missing else 'failed',
                sample_grid=observer.settings.sample_grid, object_pose_pairs=count,
                elapsed_s=perf_counter()-started, covered_object_ids=sorted(covered),
                uncovered_object_ids=sorted(missing),
                rejected_supports=rejected)


def _audit_parameters():
    grid, retries = settings.AUDIT_SAMPLE_GRID, settings.MAX_AUDIT_RETRIES
    if isinstance(grid, bool) or not isinstance(grid, int) or not 4 <= grid <= 128:
        raise ValueError('AUDIT_SAMPLE_GRID must be an integer in [4, 128]')
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError('MAX_AUDIT_RETRIES must be a nonnegative integer')
    return dict(sample_grid=grid, max_retries=retries)


def _dense_observer(observer, objects):
    grid = max(_audit_parameters()['sample_grid'], observer.settings.refinement_grid)
    return BoxObserver(observer.camera, objects, replace(observer.settings,
        sample_grid=grid, refinement_grid=grid))


def _refine_after_audit(selected, candidates, stations, objects, observer, time_limit_s=None):
    """Repair sampled false positives without changing the exported candidate q.

    Only reject relations disproved by a denser final-action audit. A repaired
    solution inherits an original-pool optimality certificate only at exactly
    the original proven cost; a refined-pool optimum alone cannot certify it.
    Bound retries and retain the audited baseline when repair is unsuccessful.
    """
    parameters = _audit_parameters()
    dense = _dense_observer(observer, objects)
    initial = _audit_observations(selected, objects, dense)
    audit = dict(initial=deepcopy(initial), final=initial, repairs=[], parameters=parameters,
                 scope='selected_actions_only_not_full_candidate_matrix')
    selected['optimization']['high_density_verification'] = audit
    if initial['status'] == 'passed':
        return selected
    logging.getLogger(__name__).info('Dense audit found uncovered targets %s; checking baseline and repairing',
                                     initial['uncovered_object_ids'])
    baseline = selected['baseline']
    baseline_audit = _audit_observations(baseline, objects, dense)
    audit['baseline'] = baseline_audit
    if selected['status'] != 'complete' or baseline_audit['status'] != 'passed':
        audit['repair_status'] = 'no_audited_complete_baseline'
        return selected
    original_cost = list(selected['selected_cost'])
    original_proven = selected['optimization']['lexicographic_optimal']
    filtered = deepcopy(candidates.supports)
    baseline_actions = [(a['station_index'], o['yaw_index'])
                        for a in baseline['anchors'] for o in a['observations']]
    elapsed = sum(selected['optimization'][stage].get('elapsed_s', 0)
                  for stage in ('stage1', 'stage2'))
    targets = [o['id'] for o in objects]
    for _ in range(parameters['max_retries']):
        for row in audit['final']['rejected_supports']:
            filtered[row['station_index']][row['yaw_index']].discard(row['object_id'])
        # Dense sampling can also add a relation absent from original q. Such a
        # relation cannot serve as the feasible incumbent of this deletion-only
        # model; retain the separately audited original baseline instead.
        if set().union(*(filtered[i][y] for i, y in baseline_actions)) != set(targets):
            break
        remaining = None if time_limit_s is None else time_limit_s-elapsed
        if remaining is not None and remaining <= 0:
            break
        solution = optimize(filtered, targets, baseline_actions, time_limit_s=remaining)
        elapsed += sum(solution['optimization'][stage].get('elapsed_s', 0)
                       for stage in ('stage1', 'stage2'))
        audit['repairs'].append(dict(selected_cost=solution['selected_cost'],
                                     optimization=solution['optimization']))
        selected['anchors'] = _assemble_anchors(solution['actions'], candidates, stations)
        selected.update(selected_cost=solution['selected_cost'],
            anchor_count=solution['selected_cost'][0], photo_count=solution['selected_cost'][1],
            selection_method='lexicographic_milp_audited_repair')
        audit['final'] = _audit_observations(selected, objects, dense)
        if audit['final']['status'] == 'passed':
            audit['repair_status'] = 'passed'
            break
    if audit['final']['status'] != 'passed':
        selected.update(deepcopy(baseline))
        selected.update(vote_trace=[], selected_cost=selected['baseline_cost'],
                        selection_method='weighted_greedy_audited_fallback')
        audit.update(final=baseline_audit, repair_status='audited_baseline_fallback')
    proven = original_proven and selected['selected_cost'] == original_cost
    selected['optimization']['lexicographic_optimal'] = proven
    audit['original_pool_certificate_preserved'] = proven
    if not proven:
        selected['optimization']['fallback_reason'] = 'audit_changed_solution_without_original_pool_certificate'
    selected['optimality'] = 'finite_pool_lexicographic_optimal' if proven else 'not_proven'
    selected['comparison_to_baseline'] = ('improved' if tuple(selected['selected_cost']) <
                                           tuple(selected['baseline_cost']) else 'equal')
    return selected


def _verify_observations(selected, objects, camera, observer):
    """Independently recompute every final action, including baseline actions."""
    actual = set()
    for anchor in selected['anchors']:
        for action in anchor['observations']:
            if observer:
                metrics = {o['id']: observer.evaluate(index, [anchor['position']], [action['yaw_rad']])[0]
                           for index, o in enumerate(objects)}
                visible = {identifier for identifier, row in metrics.items() if row['qualified']}
                action['object_metrics'] = metrics
            else:
                visible = {o['id'] for o in objects if camera.visible(
                    np.asarray([anchor['position']]), np.asarray([action['yaw_rad']]), o['position'])[0]}
            if visible != set(action['object_ids']):
                raise ValueError('Final observation failed camera geometry verification')
            actual.update(visible)
    if actual != set(selected['covered_object_ids']):
        raise ValueError('Final observations failed coverage verification')


def plan(scene, robot_config, camera_config, *, yaw_step_deg=None, selection_time_limit_s=None):
    """Select object-driven anchors directly from safe regions; no route is built.

    Targets are world-frame 3D centers; base positions are (x, y, 0).
    A station may take multiple yaw-directed photos. AABB coverage uses clipped
    framing, sampled occlusion/pixels and radial score; point maps use legacy FOV/range.
    """
    yaw_step_deg = settings.YAW_STEP_DEG if yaw_step_deg is None else yaw_step_deg
    selection_time_limit_s = settings.SELECTION_TIME_LIMIT_S if selection_time_limit_s is None else selection_time_limit_s
    audit_parameters = _audit_parameters()
    if selection_time_limit_s is not None and (isinstance(selection_time_limit_s, bool)
            or not isinstance(selection_time_limit_s, (int, float))
            or not math.isfinite(selection_time_limit_s) or selection_time_limit_s <= 0):
        raise ValueError('selection time limit must be finite and positive')
    objects = validate_objects(scene.get("objects", []))
    camera = PointCamera(camera_config)
    yaw_grid(yaw_step_deg)
    started = perf_counter()
    regions = find_safe_regions(scene, robot_config)
    stations = stations_from_regions(regions)
    preprocessing = perf_counter() - started
    candidates = generate_candidates(stations, objects, camera, yaw_step_deg=yaw_step_deg)
    started = perf_counter()
    selected = select_anchors(candidates, stations, [o['id'] for o in objects],
                              time_limit_s=selection_time_limit_s)
    observer = BoxObserver(camera, objects) if objects and 'aabb' in objects[0] else None
    if observer:
        selected = _refine_after_audit(selected, candidates, stations, objects, observer,
                                       selection_time_limit_s)
    selection = perf_counter() - started
    _verify_observations(selected['baseline'], objects, camera, observer)
    try:
        _verify_observations(selected, objects, camera, observer)
        selected['optimization']['geometry_verification'] = 'passed'
    except ValueError as exc:
        selected.update(deepcopy(selected['baseline']))
        selected.update(vote_trace=[], selection_method='weighted_greedy_fallback',
                        selected_cost=selected['baseline_cost'], comparison_to_baseline='equal',
                        optimality='not_proven')
        selected['optimization'].update(lexicographic_optimal=False,
            geometry_verification='new_plan_failed_baseline_passed', fallback_reason=str(exc))
        if observer:
            dense = _dense_observer(observer, objects)
            selected['optimization']['high_density_verification']['final'] = _audit_observations(
                selected, objects, dense)
    result = {
        'schema_version': 5, 'method': 'object_driven_lexicographic',
        'candidate_scope': 'all_safe_cell_centers_in_all_regions',
        'station_order': 'region_id_row_column',
        'scene_name': scene.get('name', 'unnamed'),
        'observation_model': 'aabb_weighted_region' if observer else 'legacy_point',
        'observation_scope': ('Solid AABBs; clipped projected framing, sampled occlusion and visible pixel area; mount-aware radial weight; recognition not evaluated'
                              if observer else '3D point centers; FOV and Euclidean distance; occlusion and appearance not evaluated'),
        'observation_settings': observer.settings.to_dict() if observer else None,
        'object_diagnostics': candidates.diagnostics,
        'selection_rule': 'min_anchor_count_then_photo_count',
        'score_definition': (f'radial_weight_{observer.settings.radial_inner_weight:g}_to_{observer.settings.radial_outer_weight:g}_times_visible_fraction'
                             if observer else 'legacy_unit_votes'),
        'region_definition': ('per_bearing_and_yaw_optical_center_distance_and_AABB_frustum_overlap_envelope'
                              if observer else 'legacy_point_no_radial_weight'),
        'uncovered_diagnosis_scope': 'No support in the evaluated discrete station/yaw pool; not continuous-space infeasibility',
        'configuration': {'robot': robot_config, 'camera': camera_config, 'yaw_step_deg': yaw_step_deg,
                          'selection_time_limit_s': selection_time_limit_s, 'audit': audit_parameters},
        'statistics': {'safe_regions_and_station_seconds': preprocessing, 'selection_seconds': selection,
                       'region_count': len(regions['regions']), 'safe_station_count': len(stations),
                       **candidates.stats},
        **selected,
    }
    return result, regions, stations, candidates


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


def render_svg(scene,regions,result,stations,candidates):
    import html, math
    svg=diagnostic_svg(scene,regions)
    grid=regions['grid']; scale=36; pad=32
    def screen(position):
        return (pad+(position[0]-grid['origin'][0])/grid['resolution']*scale,
                pad+(grid['height']-(position[1]-grid['origin'][1])/grid['resolution'])*scale)
    shapes=['<g id="anchor-observation-overlay">']
    by_id={s.index:s for s in stations}
    totals = defaultdict(float)
    for entries in object_heatmaps(candidates, [o['id'] for o in scene.get('objects', [])]).values():
        for index, score, _, units in entries:
            totals[index] += units/1_000_000
    for i,score in totals.items():
        x,y=screen(by_id[i].position)
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#ff9f1c" opacity="{min(.7,.1+.06*score):.2f}"><title>weighted score {score:.6f}</title></circle>')
    missing=set(result['uncovered_object_ids'])
    for obj in scene.get('objects',[]):
        x,y=screen(obj['position']); color='#e63946' if obj['id'] in missing else '#2a9d8f'
        name=html.escape(obj['id'])
        if 'aabb' in obj:
            box=obj['aabb']; bx,by=screen([box['min'][0],box['max'][1]])
            bw=(box['max'][0]-box['min'][0])/grid['resolution']*scale
            bh=(box['max'][1]-box['min'][1])/grid['resolution']*scale
            shapes.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bh:.2f}" fill="{color}" fill-opacity="0.2" stroke="{color}"/>')
            name+=f" z={box['min'][2]:.2f}–{box['max'][2]:.2f}m"
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="13" fill="{color}" stroke="white" stroke-width="3"/><text x="{x+15:.2f}" y="{y:.2f}" font-size="20">{name}</text>')
    for anchor in result['anchors']:
        x,y=screen(anchor['position'])
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="18" fill="none" stroke="#d000a6" stroke-width="5"/>')
        for action in anchor['observations']:
            yaw=action['yaw_rad']; length=65
            shapes.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x+length*math.cos(yaw):.2f}" y2="{y-length*math.sin(yaw):.2f}" stroke="#d000a6" stroke-width="5"/>')
    shapes.append('</g>')
    return svg.replace('</svg>','\n'.join(shapes)+'\n</svg>')


def viewer_data(scene, camera_config, result, regions, stations, candidates):
    camera = PointCamera(camera_config)
    methods = []
    def projected_actions(anchors):
        actions=[]
        for anchor_index,anchor in enumerate(anchors):
            for action in anchor['observations']:
                projected=[]
                for obj in scene.get('objects',[]):
                    u,v,z,d=camera.project(np.array([anchor['position']]),np.array([action['yaw_rad']]),obj['position'])
                    projected.append({'id':obj['id'],'u':float(u[0]) if math.isfinite(u[0]) else None,
                        'v':float(v[0]) if math.isfinite(v[0]) else None,'depth_m':float(z[0]),'distance_m':float(d[0]),
                        'qualified':obj['id'] in action['object_ids'],
                        'metrics':action.get('object_metrics', {}).get(obj['id'])})
                actions.append({'anchor_index':anchor_index,'yaw_rad':action['yaw_rad'],
                                'object_ids':action['object_ids'],'projected':projected})
        return actions
    inverse={obj['id']:[] for obj in scene.get('objects',[])}
    heatmaps = object_heatmaps(candidates, inverse)
    totals = defaultdict(int)
    for obj, entries in heatmaps.items():
        for index, score, _, units in entries:
            inverse[obj].append(index)
            totals[index] += units
    for title, selection, replay in [('最少锚点 / 最少照片', result, False),
                                      ('基线：加权贪心', result['baseline'], True)]:
        methods.append({'title': title, 'anchors': selection['anchors'],
            'actions': projected_actions(selection['anchors']), 'replay_enabled': replay,
            'candidate_count': len(candidates.station_indices),
            'votes': [[index, units/1_000_000] for index, units in sorted(totals.items())],
            'inverse': inverse, 'object_heatmaps': heatmaps,
            'vote_trace': selection['vote_trace'] if replay else [],
            'uncovered_object_ids': selection['uncovered_object_ids'],
            'generation_seconds': result['statistics']['generation_seconds']})
    return {'scene_name':scene.get('name','unnamed'),'grid':scene['grid'],'objects':scene.get('objects',[]),
    'camera':camera_config,'methods':methods,
    'optimization':result['optimization'], 'selection_method':result['selection_method'],
    'baseline_cost':result['baseline_cost'], 'selected_cost':result['selected_cost'],
    'observation_model':result['observation_model'],'observation_settings':result['observation_settings'],
    'object_diagnostics':result['object_diagnostics'],
    'yaws_rad':candidates.yaws,
    'heatmap_fields':['station_index', 'best_qualified_score', 'yaw_index', 'vote_units'],
    'stations':[[s.index,*s.position] for s in stations],
    'region_count':len(regions['regions'])}

def save_supports(destination, scene, stations, candidates):
    """Export qualified supports, continuous scores and per-object heatmaps."""
    relation = {
        'schema_version': 4, 'station_order': 'region_id_row_column',
        'scores': candidates.scores,
        'object_heatmaps': object_heatmaps(candidates, [o['id'] for o in scene.get('objects', [])]),
        'heatmap_fields': ['station_index', 'best_qualified_score', 'yaw_index', 'vote_units'],
        'object_points': scene.get('objects', []), 'yaws_rad': candidates.yaws,
        'evaluated_station_indices': candidates.station_indices,
        'positive_stations': [{'index': s.index, 'region_id': s.region_id, 'cell': list(s.cell),
                              'position': list(s.position)}
                             for s in stations if s.index in candidates.supports],
        'supports': {str(i): {str(y): sorted(ids) for y, ids in directions.items()}
                     for i, directions in candidates.supports.items()},
    }
    destination.write_text(json.dumps(relation, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=settings.MAP_PATH)
    parser.add_argument('--robot', type=Path, default=settings.ROBOT_PATH)
    parser.add_argument('--camera', type=Path, default=settings.CAMERA_PATH)
    parser.add_argument('--yaw-step-deg', type=float, default=None,
                        help=f'底盘拍摄偏航间隔，单位度，默认读取 config.py（{settings.YAW_STEP_DEG}）')
    parser.add_argument('--selection-time-limit-s', type=float, default=None,
                        help='全部 MILP 调用（含复核重选）的总时间预算；默认读取 config.py，不含观察计算与建模')
    parser.add_argument('--save-supports', action='store_true', help='另存合格支持关系、连续得分及逐物体热力图')
    parser.add_argument('--output', type=Path, default=settings.ANCHOR_PLAN_PATH)
    args = parser.parse_args()
    try:
        if args.output.suffix.lower() != '.json':
            raise ValueError('--output must have a .json suffix')
        if args.selection_time_limit_s is not None and (not math.isfinite(args.selection_time_limit_s) or args.selection_time_limit_s <= 0):
            raise ValueError('--selection-time-limit-s must be finite and positive')
        scene = read_input(args.scene)
        robot = read_input(args.robot)
        camera = read_input(args.camera)
        result, regions, stations, candidates = plan(scene, robot, camera, yaw_step_deg=args.yaw_step_deg,
                                                       selection_time_limit_s=args.selection_time_limit_s)
        payload = json.dumps(viewer_data(scene, camera, result, regions, stations, candidates),
                             ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        page = settings.ANCHOR_VIEWER_TEMPLATE.read_text(encoding='utf-8').replace('__ANCHOR_DATA__', payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        args.output.with_suffix('.svg').write_text(render_svg(scene, regions, result, stations, candidates), encoding='utf-8')
        args.output.with_suffix('.html').write_text(page, encoding='utf-8')
        if args.save_supports:
            save_supports(args.output.with_suffix('.supports.json'), scene, stations, candidates)
    except (ValueError, OSError, KeyError, TypeError, PlanningError) as exc:
        parser.exit(2, f'锚点规划失败：{exc}\n')
    print(json.dumps({k: result[k] for k in ('status', 'anchor_count', 'photo_count', 'coverage_fraction', 'uncovered_object_ids', 'baseline_cost', 'selected_cost', 'optimality')}, ensure_ascii=False, indent=2))
    if 'high_density_verification' in result['optimization']:
        audit = result['optimization']['high_density_verification']
        print(f"高密度复核：{audit['final']['status']}；重选次数：{len(audit['repairs'])}；缺口：{audit['final']['uncovered_object_ids']}")
    print(f'锚点计划：{args.output}；交互页面：{args.output.with_suffix(".html")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
