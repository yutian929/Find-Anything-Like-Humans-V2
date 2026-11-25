"""Deterministic arrivals at two actual planned stations. Geometry and depth are synthetic."""
from collections import Counter
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from falhv2.robot_adapters.base import CameraIntrinsics, RGBDFrame
from .store import initialize, record, sha, write_json


def make_demo(root, map_path, plan_path, mug, fruit, court):
    if root.exists():
        raise FileExistsError(f'demo requires a new directory: {root}')
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix='.demo-') as work:
        staged = Path(work) / 'store'
        outcomes = _populate(staged, map_path, plan_path, mug, fruit, court)
        staged.rename(root)
        return outcomes


def _populate(root, map_path, plan_path, mug, fruit, court):
    index = initialize(root, map_path, plan_path)
    first = index['views'][0]
    counts = Counter(v['station_index'] for v in index['views'])
    others = [v for v in index['views'] if v['station_index'] != first['station_index']]
    second = min(others, key=lambda v: counts[v['station_index']], default=None)
    if second is None:
        raise ValueError('demo needs two planned stations')
    outcomes = []
    for number, (view, photo, visit) in enumerate([
            (first, fruit, 'arrival-1'), (first, fruit, 'arrival-1'),
            (second, court, 'arrival-2'), (first, court, 'arrival-3'), (first, mug, 'arrival-4')], 1):
        image = cv2.imread(str(photo))
        if image is None:
            raise ValueError(f'missing demo image {photo}')
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        c, s = math.cos(view['yaw_rad']), math.sin(view['yaw_rad'])
        x, y = view['position']
        base = np.array([[c,-s,0,x],[s,c,0,y],[0,0,1,0],[0,0,0,1]])
        mount = np.array([[0,0,1,.23],[-1,0,0,0],[0,-1,0,.5],[0,0,0,1]])
        frame = RGBDFrame(f'demo-{number}', number*1_000_000_000, 'simulation', index['map_id'],
                          index['map_revision'], 'demo_optical', CameraIntrinsics(w,h,w,w,(w-1)/2,(h-1)/2),
                          rgb, np.full((h,w), 1.25, dtype=np.float32), base @ mount, base)
        slot = record(root, view['station_index'], view['yaw_index'], visit, frame, simulated=True)
        outcomes.append(dict(frame_id=frame.frame_id, visit_id=visit, station=view['station_index'],
                             yaw=view['yaw_index'], source=photo.name, source_sha256=sha(photo.read_bytes()),
                             outcome=slot))
    write_json(root / 'demo_events.json', outcomes)
    return outcomes
