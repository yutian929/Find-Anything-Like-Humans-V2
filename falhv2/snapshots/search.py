"""Search the latest observation of each visited view, then older first observations."""
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .store import read_json, search_views, sha, write_json


def estimate_point(path, box, meta):
    depth = np.load(path / 'depth_m.npy', allow_pickle=False)
    x0, y0, x1, y1 = box
    u, v = (x0+x1-1)//2, (y0+y1-1)//2
    dx, dy = (x1-x0)//8, (y1-y0)//8
    patch = depth[v-dy:v+dy+1, u-dx:u+dx+1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < max(3, patch.size / 2):
        return None
    z = float(np.median(valid))
    if np.percentile(valid, 90) - np.percentile(valid, 10) > max(0.15, z * 0.15):
        return None
    k = meta['intrinsics']
    camera = [(u-k['cx'])*z/k['fx'], (v-k['cy'])*z/k['fy'], z, 1]
    return (np.array(meta['T_map_camera']) @ camera)[:3].tolist()


def evaluate(client, target, view, slot, path, predicate_min, bbox_min):
    meta = read_json(path / 'meta.json')
    image = cv2.imread(str(path / 'rgb.png'))
    if image is None:
        raise OSError(f'cannot read {path}/rgb.png')
    h, w = image.shape[:2]
    answer = client.inspect(image, target)
    x0, y0, x1, y1 = answer['bbox']
    box = [math.floor(x0*w), math.floor(y0*h), math.ceil(x1*w), math.ceil(y1*h)]
    choice = None
    if (answer['present'] >= predicate_min and answer['bbox_confidence'] >= bbox_min
            and box[2] > box[0] and box[3] > box[1]):
        choice = client.verify(image[box[1]:box[3], box[0]:box[2]], target)
    return {**{k: meta[k] for k in ('frame_id', 'timestamp_ns', 'clock_domain', 'simulated', 'visit_id')},
            'station_index': view['station_index'], 'yaw_index': view['yaw_index'], 'slot': slot,
            **answer, 'bbox_xyxy_px': box, 'crop_choice': choice}


def find_object(root, target, client, result_dir, predicate_min=0.65, bbox_min=0.5):
    if not target.strip() or not (0 <= predicate_min <= 1 and 0 <= bbox_min <= 1):
        raise ValueError('a target and thresholds in [0,1] are required')
    result_dir = Path(result_dir)
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    if result_dir.exists():
        raise FileExistsError(f'result directory already exists: {result_dir}')
    with tempfile.TemporaryDirectory(dir=result_dir.parent, prefix='.find-') as scratch:
        staged = Path(scratch) / 'result'
        staged.mkdir()
        result = _search(root, target, client, staged, predicate_min, bbox_min)
        staged.rename(result_dir)
        return result


def _search(root, target, client, result_dir, predicate_min, bbox_min):
    audit = []
    with search_views(root) as (index, latest, history):
        captured = {(v['station_index'], v['yaw_index']) for v, _, _ in latest}
        pending = [[v['station_index'], v['yaw_index']] for v in index['views']
                   if (v['station_index'], v['yaw_index']) not in captured]
        result = dict(query=target, model=client.model, map_id=index['map_id'],
                      map_revision=index['map_revision'], plan_sha256=index['plan_sha256'],
                      planned_views=len(index['views']), captured_views=len(latest), pending_views=pending,
                      predicate_min=predicate_min, bbox_min=bbox_min, checked=audit,
                      status='not_found_in_snapshots' if latest else 'no_snapshots', match=None)
        for status, group in [('found_latest', latest), ('found_historical', history)]:
            matches = []
            for view, slot, path in group:
                row = evaluate(client, target, view, slot, path, predicate_min, bbox_min)
                row['pass'] = status.removeprefix('found_')
                audit.append(row)
                if row['crop_choice'] == 'match':
                    matches.append((row, path))
            if matches:
                row, path = max(matches, key=lambda item: (item[0]['visibility'], item[0]['present'],
                                                           item[0]['bbox_confidence']))
                meta = read_json(path / 'meta.json')
                box = row['bbox_xyxy_px']
                rgb = (path / 'rgb.png').read_bytes()
                (result_dir / 'rgb.png').write_bytes(rgb)
                marked = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.rectangle(marked, tuple(box[:2]), (box[2]-1, box[3]-1), (0, 200, 255), 3)
                if not cv2.imwrite(str(result_dir / 'bbox.png'), marked):
                    raise OSError('could not save bbox evidence')
                result.update(status=status, match={**row, 'map_xyz_m_estimate': estimate_point(path, box, meta),
                              'T_map_base': meta['T_map_base'], 'rgb_sha256': sha(rgb),
                              'rgb': 'rgb.png', 'annotation': 'bbox.png'})
                break
    write_json(result_dir / 'result.json', result)
    return result
