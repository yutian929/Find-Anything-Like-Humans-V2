"""Offline RGB-D visits, grouped by the planner's station and yaw indices."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from falhv2.robot_adapters.base import CameraIntrinsics, RGBDFrame, transform, validate_frame


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def view_path(station, yaw):
    return Path('anchors') / f'station_{station:04d}' / f'yaw_{yaw:03d}'


@contextmanager
def lock(root, exclusive):
    with (root / '.lock').open('a+b') as file:
        fcntl.flock(file, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def initialize(root, map_path, plan_path):
    map_bytes, plan_bytes = map_path.read_bytes(), plan_path.read_bytes()
    scene, plan = json.loads(map_bytes), json.loads(plan_bytes)
    identity = scene.get('object_map', {})
    digest = sha(map_bytes)
    if plan.get('scene_name') != scene.get('name', 'unnamed') or plan.get('map_sha256', digest) != digest:
        raise ValueError('plan does not match the supplied map')
    scene_ids = {obj['id'] for obj in scene['objects']}
    if scene_ids != set(plan['covered_object_ids']) | set(plan['uncovered_object_ids']):
        raise ValueError('map and plan have different object IDs')
    views = []
    keys = set()
    for a in plan['anchors']:
        for obs in a['observations']:
            key = a['station_index'], obs['yaw_index']
            if key in keys or any(type(i) != int or i < 0 for i in key):
                raise ValueError(f'invalid or duplicate view: {key}')
            keys.add(key)
            x, y = a['position']
            if not np.isfinite([x, y, obs['yaw_rad']]).all():
                raise ValueError('nonfinite planned pose')
            views.append(dict(station_index=key[0], yaw_index=key[1], position=[x, y],
                              yaw_rad=obs['yaw_rad'], expected_object_ids=obs['object_ids']))
    index = dict(schema_version=1, map_sha256=digest, plan_sha256=sha(plan_bytes),
                 map_id=identity.get('map_id') or scene.get('name', 'unnamed'),
                 map_revision=str(identity.get('map_revision') or digest), views=views)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.snapshot-init-', dir=root.parent) as work:
        staged = Path(work) / 'store'
        staged.mkdir()
        (staged / 'map.json').write_bytes(map_bytes)
        (staged / 'anchor_plan.json').write_bytes(plan_bytes)
        write_json(staged / 'index.json', index)
        try:
            staged.rename(root)
        except OSError:
            if not root.is_dir():
                raise
            path = root / 'index.json'
            if not path.is_file() or read_json(path) != index:
                raise ValueError('select an empty store or one initialized with this exact map and plan')
    return index


def frame_from_files(folder):
    meta = read_json(folder / 'meta.json')
    bgr = cv2.imread(str(folder / 'rgb.png'))
    if bgr is None:
        raise ValueError(f'cannot read {folder}/rgb.png')
    frame = RGBDFrame(**{k: meta[k] for k in ('frame_id', 'timestamp_ns', 'map_id',
                      'map_revision', 'camera_frame', 'clock_domain')},
                      intrinsics=CameraIntrinsics(**meta['intrinsics']),
                      rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                      depth_m=np.load(folder / 'depth_m.npy', allow_pickle=False),
                      T_map_camera=transform(meta['T_map_camera']), T_map_base=transform(meta['T_map_base']))
    return frame, meta['visit_id'], meta.get('simulated', False)


def record(root, station, yaw, visit_id, frame, *, simulated=False):
    """Caller assigns one visit ID per arrival; all frames from a dwell share it."""
    if not isinstance(visit_id, str) or not visit_id.strip() or type(simulated) != bool:
        raise ValueError('visit_id must be nonempty and simulated must be boolean')
    index = read_json(root / 'index.json')
    view = next((v for v in index['views'] if (v['station_index'], v['yaw_index']) == (station, yaw)), None)
    if view is None:
        raise ValueError('view does not exist in this plan')
    if (frame.map_id, frame.map_revision) != (index['map_id'], index['map_revision']):
        raise ValueError('frame belongs to another map revision')
    rgb, depth = validate_frame(frame)
    base = transform(frame.T_map_base)
    delta = math.atan2(base[1, 0], base[0, 0]) - view['yaw_rad']
    if (math.dist(base[:2, 3], view['position']) > 0.10 or
            abs(math.remainder(delta, 2 * math.pi)) > math.radians(10)):
        raise ValueError('measured base pose outside planned view tolerance (0.10 m / 10 deg)')
    parent = root.resolve() / view_path(station, yaw)
    with lock(root, True):
        parent.mkdir(parents=True, exist_ok=True)
        first, last = parent / 'first', parent / 'last'
        latest = last if last.exists() else first
        if latest.exists():
            previous = read_json(latest / 'meta.json')
            if previous['clock_domain'] != frame.clock_domain or previous['simulated'] != simulated:
                raise ValueError('incompatible clock domain or simulated flag')
            if visit_id in {previous['visit_id'], read_json(first / 'meta.json')['visit_id']}:
                return 'same_visit'
            if frame.timestamp_ns <= previous['timestamp_ns']:
                return 'stale_frame'
        # A relative symlink publishes the whole last bundle in one rename.
        # Reclaim leftover staging generations after an interrupted writer.
        active = last.resolve() if last.exists() else None
        for path in parent.glob('.capture-*'):
            if path != active:
                shutil.rmtree(path)
        stage = Path(tempfile.mkdtemp(prefix='.capture-', dir=parent)).resolve()
        try:
            if not cv2.imwrite(str(stage / 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError('RGB encoding failed')
            np.save(stage / 'depth_m.npy', depth, allow_pickle=False)
            meta = {k: getattr(frame, k) for k in ('frame_id', 'timestamp_ns', 'clock_domain',
                    'map_id', 'map_revision', 'camera_frame')}
            meta.update(visit_id=visit_id, simulated=simulated, station_index=station, yaw_index=yaw,
                        intrinsics=asdict(frame.intrinsics), T_map_camera=np.asarray(frame.T_map_camera).tolist(),
                        T_map_base=base.tolist())
            write_json(stage / 'meta.json', meta)
            if not first.exists():
                stage.rename(first)
                return 'first'
            link = parent / '.next'
            link.unlink(missing_ok=True)
            link.symlink_to(stage.name, target_is_directory=True)
            link.replace(last)
        finally:
            if stage.exists() and stage != last.resolve():
                shutil.rmtree(stage)
        if active is not None:
            shutil.rmtree(active)
        return 'last'


@contextmanager
def search_views(root):
    """Pin one coherent view of the store without holding a lock during inference."""
    index = read_json(root / 'index.json')
    with tempfile.TemporaryDirectory(prefix='.search-', dir=root) as work:
        current, history = [], []
        with lock(root, False):
            for v in index['views']:
                rel = view_path(v['station_index'], v['yaw_index'])
                parent = root / rel
                for slot in ('first', 'last'):
                    src = parent / slot
                    if not src.is_dir():
                        continue
                    pinned = Path(work) / rel / slot
                    pinned.mkdir(parents=True)
                    for name in ('meta.json', 'rgb.png', 'depth_m.npy'):
                        os.link(src / name, pinned / name)
                    group = history if slot == 'first' and (parent / 'last').exists() else current
                    group.append((v, slot, pinned))
        yield index, current, history


def status(root):
    index = read_json(root / 'index.json')
    summary = {'planned_views': len(index['views']), 'captured_views': 0, 'revisited_views': 0}
    planned, visited = Counter(), Counter()
    with lock(root, False):
        for v in index['views']:
            station = v['station_index']
            planned[station] += 1
            parent = root / view_path(station, v['yaw_index'])
            if (parent / 'first').is_dir():
                visited[station] += 1
                summary['captured_views'] += 1
            summary['revisited_views'] += (parent / 'last').is_dir()
    summary.update(planned_stations=len(planned), visited_stations=len(visited),
                   complete_stations=sum(visited[s] == n for s, n in planned.items()))
    return summary
