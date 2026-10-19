"""物体驱动锚点规划：安全区域 → 反向覆盖关系 → 动态投票 → 定向拍摄。

默认从 data/map.json、data/robot.json 和 data/camera.d435i.json 读取输入，
输出离线 JSON / HTML / SVG。覆盖模型仅检查三维中心点的相机视场与距离。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
import argparse
import heapq
import json

from scipy.spatial import cKDTree

try:
    from .regions_extractor import plan as find_safe_regions, diagnostic_svg, PlanningError
    from . import config as settings
    from .config import read_input
except ImportError:
    from regions_extractor import plan as find_safe_regions, diagnostic_svg, PlanningError
    import config as settings
    from config import read_input

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

def select_anchors(candidates: CandidateResult, stations: list[Station], object_ids: list[str]) -> dict:
    targets = set(object_ids)
    station_by_id = {station.index: station for station in stations}
    station_sets = {i: set().union(*directions.values()) for i, directions in candidates.supports.items()}
    if any(not objects <= targets for objects in station_sets.values()):
        raise ValueError("Coverage contains unknown object IDs")
    if not set(station_sets) <= set(station_by_id):
        raise ValueError("Coverage contains unknown station IDs")
    reverse = defaultdict(set)
    for i, objects in station_sets.items():
        for obj in objects:
            reverse[obj].add(i)
    remaining = set(targets)
    votes = {i: len(objects) for i, objects in station_sets.items()}
    heap = [(-score, i) for i, score in votes.items() if score]
    heapq.heapify(heap)
    chosen, trace = [], []
    while heap and remaining:
        negative, i = heapq.heappop(heap)
        if -negative != votes[i]:
            continue
        new = station_sets[i] & remaining
        if not new:
            continue
        chosen.append(i)
        trace.append({"station_index": i, "new_object_ids": sorted(new), "votes": len(new)})
        remaining.difference_update(new)
        for obj in new:
            for other in reverse[obj]:
                votes[other] -= 1
                if votes[other] > 0:
                    heapq.heappush(heap, (-votes[other], other))
    # Keep an inclusion-minimal station cover, without moving any position.
    counts = Counter(obj for i in chosen for obj in station_sets[i])
    retained = set(chosen)
    for i in reversed(chosen):
        if all(counts[obj] > 1 for obj in station_sets[i]):
            retained.remove(i)
            counts.subtract(station_sets[i])
    covered = targets-remaining
    directions = {(i, yaw): objects for i in sorted(retained)
                  for yaw, objects in sorted(candidates.supports[i].items()) if objects}
    need = set(covered)
    actions = []
    while need:
        # Deterministic tie-break by station index (region/row/column order) and yaw index.
        key = min(directions, key=lambda key: (-len(directions[key] & need), key))
        if not (directions[key] & need):
            raise ValueError("Direction selection lost station coverage")
        actions.append(key)
        need.difference_update(directions[key])
    counts = Counter(obj for key in actions for obj in directions[key])
    for key in list(reversed(actions)):
        if all(counts[obj] > 1 for obj in directions[key]):
            actions.remove(key)
            counts.subtract(directions[key])
    anchors = []
    for i in sorted(retained):
        station = station_by_id[i]
        observations = [{"yaw_index": yaw, "yaw_rad": candidates.yaws[yaw],
                         "object_ids": sorted(directions[(i, yaw)])}
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


def plan(scene, robot_config, camera_config, *, yaw_step_deg=30):
    """Select object-driven anchors directly from safe regions; no route is built.

    Targets are world-frame 3D centers; base positions are (x, y, 0).
    A station may take multiple yaw-directed photos. Coverage is binary FOV/range,
    with no occlusion or appearance model. Greedy selection is not an optimum proof.
    """
    objects = validate_objects(scene.get("objects", []))
    camera = PointCamera(camera_config)
    yaw_grid(yaw_step_deg)
    started = perf_counter()
    regions = find_safe_regions(scene, robot_config)
    stations = stations_from_regions(regions)
    preprocessing = perf_counter() - started
    candidates = generate_candidates(stations, objects, camera, yaw_step_deg=yaw_step_deg)
    started = perf_counter()
    selected = select_anchors(candidates, stations, [o['id'] for o in objects])
    selection = perf_counter() - started
    # Recompute every final action independently of the sparse candidate relation.
    actual = set()
    for anchor in selected['anchors']:
        for action in anchor['observations']:
            visible = {o['id'] for o in objects if camera.visible(
                np.asarray([anchor['position']]), np.asarray([action['yaw_rad']]), o['position'])[0]}
            if visible != set(action['object_ids']):
                raise ValueError('Final observation failed camera geometry verification')
            actual.update(visible)
    if actual != set(selected['covered_object_ids']):
        raise ValueError('Final observations failed coverage verification')
    result = {
        'schema_version': 2, 'method': 'object_driven',
        'candidate_scope': 'all_safe_cell_centers_in_all_regions',
        'station_order': 'region_id_row_column',
        'scene_name': scene.get('name', 'unnamed'),
        'observation_scope': '3D point centers; FOV and Euclidean distance; occlusion and appearance not evaluated',
        'uncovered_diagnosis_scope': 'No support in the evaluated discrete station/yaw pool; not continuous-space infeasibility',
        'configuration': {'robot': robot_config, 'camera': camera_config, 'yaw_step_deg': yaw_step_deg},
        'statistics': {'safe_regions_and_station_seconds': preprocessing, 'selection_seconds': selection,
                       'region_count': len(regions['regions']), 'safe_station_count': len(stations),
                       **candidates.stats},
        **selected,
    }
    return result, regions, stations, candidates


def render_svg(scene,regions,result,stations,candidates):
    import html, math
    svg=diagnostic_svg(scene,regions)
    grid=regions['grid']; scale=36; pad=32
    def screen(position):
        return (pad+(position[0]-grid['origin'][0])/grid['resolution']*scale,
                pad+(grid['height']-(position[1]-grid['origin'][1])/grid['resolution'])*scale)
    shapes=['<g id="anchor-observation-overlay">']
    by_id={s.index:s for s in stations}
    for i,directions in candidates.supports.items():
        count=len(set().union(*directions.values()))
        x,y=screen(by_id[i].position)
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#ff9f1c" opacity="{min(.7,.1+.06*count):.2f}"><title>{count} target votes</title></circle>')
    missing=set(result['uncovered_object_ids'])
    for obj in scene.get('objects',[]):
        x,y=screen(obj['position']); color='#e63946' if obj['id'] in missing else '#2a9d8f'
        name=html.escape(obj['id'])
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
    actions=[]
    for anchor_index,anchor in enumerate(result['anchors']):
        for action in anchor['observations']:
            projected=[]
            for obj in scene.get('objects',[]):
                u,v,z,d=camera.project(np.array([anchor['position']]),np.array([action['yaw_rad']]),obj['position'])
                projected.append({'id':obj['id'],'u':float(u[0]) if math.isfinite(u[0]) else None,
                    'v':float(v[0]) if math.isfinite(v[0]) else None,'depth_m':float(z[0]),'distance_m':float(d[0]),
                    'qualified':obj['id'] in action['object_ids']})
            actions.append({'anchor_index':anchor_index,'yaw_rad':action['yaw_rad'],
                            'object_ids':action['object_ids'],'projected':projected})
    inverse={obj['id']:[] for obj in scene.get('objects',[])}
    votes=[]
    for index,directions in candidates.supports.items():
        union=set().union(*directions.values())
        votes.append([index,len(union)])
        for obj in union: inverse[obj].append(index)
    methods.append({'title':'物体驱动','anchors':result['anchors'],'actions':actions,
        'candidate_count':len(candidates.station_indices),'votes':votes,'inverse':inverse,
        'uncovered_object_ids':result['uncovered_object_ids'],
        'generation_seconds':result['statistics']['generation_seconds']})
    return {'scene_name':scene.get('name','unnamed'),'grid':scene['grid'],'objects':scene.get('objects',[]),
    'camera':camera_config,'methods':methods,
    'stations':[[s.index,*s.position] for s in stations],
    'region_count':len(regions['regions'])}

def save_supports(destination, scene, stations, candidates):
    """Export the sparse binary q(station, yaw, object); missing entries are zero."""
    relation = {
        'schema_version': 2, 'station_order': 'region_id_row_column',
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=settings.MAP_PATH)
    parser.add_argument('--robot', type=Path, default=settings.ROBOT_PATH)
    parser.add_argument('--camera', type=Path, default=settings.CAMERA_PATH)
    parser.add_argument('--yaw-step-deg', type=float, default=30, help='底盘拍摄偏航间隔，单位度，默认 30')
    parser.add_argument('--save-supports', action='store_true', help='另存 A2 稀疏二值覆盖关系')
    parser.add_argument('--output', type=Path, default=settings.ANCHOR_PLAN_PATH)
    args = parser.parse_args()
    try:
        if args.output.suffix.lower() != '.json':
            raise ValueError('--output must have a .json suffix')
        scene = read_input(args.scene)
        robot = read_input(args.robot)
        camera = read_input(args.camera)
        result, regions, stations, candidates = plan(scene, robot, camera, yaw_step_deg=args.yaw_step_deg)
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
    print(json.dumps({k: result[k] for k in ('status', 'anchor_count', 'photo_count', 'coverage_fraction', 'uncovered_object_ids')}, ensure_ascii=False, indent=2))
    print(f'锚点计划：{args.output}；交互页面：{args.output.with_suffix(".html")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
