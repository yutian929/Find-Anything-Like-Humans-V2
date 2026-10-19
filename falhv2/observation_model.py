"""Sample projected solid AABBs to estimate framing, occlusion and visible pixels.

The target's full projected silhouette is the framing denominator. Occlusion uses
only its in-frame silhouette. Rays use the nearest target surface, so back faces
are not counted. Estimates are deterministic, not analytic visibility guarantees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math

import numpy as np


@dataclass(frozen=True)
class ObservationSettings:
    min_in_frame_fraction: float = 0.8
    min_visible_fraction: float = 0.6
    min_visible_pixels: float = 400.0
    sample_grid: int = 12
    refinement_grid: int = 36

    @classmethod
    def from_config(cls, config):
        values = config.get('observation_quality', {})
        result = cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})
        for key in ('min_in_frame_fraction', 'min_visible_fraction'):
            value = getattr(result, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{key} must be finite and in (0, 1]')
        if (isinstance(result.min_visible_pixels, bool) or not isinstance(result.min_visible_pixels, (float, int))
                or not math.isfinite(result.min_visible_pixels) or result.min_visible_pixels <= 0):
            raise ValueError('min_visible_pixels must be finite and positive')
        for key in ('sample_grid', 'refinement_grid'):
            value = getattr(result, key)
            if isinstance(value, bool) or not isinstance(value, int) or not 4 <= value <= 128:
                raise ValueError(f'{key} must be an integer in [4, 128]')
        if result.refinement_grid < result.sample_grid:
            raise ValueError('refinement_grid must be at least sample_grid')
        return result

    def to_dict(self):
        return asdict(self)


def validate_objects(objects):
    """Keep legacy points; validate optional world-frame AABBs without inventing size."""
    result, ids = [], set()
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError('Objects must be dictionaries')
        identifier = obj.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('Objects require unique nonempty string IDs')
        try:
            point = np.asarray(obj['position'], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid point for object {identifier}') from exc
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f'Object {identifier} requires a finite 3D point')
        record = {'id': identifier, 'position': point.tolist()}
        if 'aabb' in obj:
            try:
                lower = np.asarray(obj['aabb']['min'], dtype=float)
                upper = np.asarray(obj['aabb']['max'], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Object {identifier} requires aabb.min/max') from exc
            if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all()
                    or not np.isfinite(upper).all() or np.any(lower >= upper) or lower[2] < -1e-9
                    or not np.allclose(point, (lower+upper)/2, rtol=0, atol=1e-6)):
                raise ValueError(f'Object {identifier}: AABB must have positive sizes, nonnegative bottom and position at its center')
            record['aabb'] = {'min': lower.tolist(), 'max': upper.tolist()}
        ids.add(identifier)
        result.append(record)
    if any('aabb' in obj for obj in result) and not all('aabb' in obj for obj in result):
        raise ValueError('Mixed point/AABB map: 请先在地图编辑器中为所有物体补全 AABB，避免遗漏遮挡体')
    return result


def ray_box_entry(origins, directions, lower, upper):
    """Broadcasted slab intersection, finite nonnegative entry or +inf for a miss.

    Parallel rays on a face are handled without 0*inf. An origin in a box has
    entry zero; callers conservatively reject cameras inside solid geometry.
    """
    origins, directions = np.asarray(origins), np.asarray(directions)
    parallel = np.abs(directions) < 1e-12
    denominator = np.where(parallel, 1.0, directions)
    a = (np.asarray(lower)-origins)/denominator
    b = (np.asarray(upper)-origins)/denominator
    near = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=-1)
    far = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=-1)
    outside = np.any(parallel & ((origins < lower) | (origins > upper)), axis=-1)
    entry = np.maximum(near, 0.0)
    return np.where(~outside & (far >= entry), entry, np.inf)


class BoxObserver:
    def __init__(self, camera, objects, settings=None):
        self.camera = camera
        self.objects = validate_objects(objects)
        if any('aabb' not in obj for obj in self.objects):
            raise ValueError('BoxObserver requires an AABB for every object')
        self.settings = settings or ObservationSettings.from_config(camera.config)
        self.lower = np.array([o['aabb']['min'] for o in self.objects], dtype=float).reshape(-1, 3)
        self.upper = np.array([o['aabb']['max'] for o in self.objects], dtype=float).reshape(-1, 3)
        self.ids = [o['id'] for o in self.objects]
        self.corners = np.array(list(product((0, 1), repeat=3)), dtype=float)

    def poses(self, positions, yaws):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        yaws = np.asarray(yaws, dtype=float).reshape(-1)
        if len(positions) != len(yaws):
            raise ValueError('One yaw per base position required')
        c, s = np.cos(yaws), np.sin(yaws)
        rotation = np.zeros((len(yaws), 3, 3))
        rotation[:, 0, 0] = rotation[:, 1, 1] = c
        rotation[:, 1, 0] = s
        rotation[:, 0, 1] = -s
        rotation[:, 2, 2] = 1
        origins = np.column_stack((positions, np.zeros(len(yaws)))) + np.einsum('bij,j->bi', rotation, self.camera.offset)
        return origins, rotation @ self.camera.base_from_optical

    def evaluate(self, target_index, positions, yaws):
        """Return one metric record per pose, refining near qualification thresholds."""
        origins, rotation = self.poses(positions, yaws)
        records = []
        # Bound temporary arrays even for fine user-specified sampling.
        for start in range(0, len(origins), 32):
            o, r = origins[start:start+32], rotation[start:start+32]
            batch = self._evaluate(target_index, o, r, self.settings.sample_grid)
            if self.settings.refinement_grid > self.settings.sample_grid:
                selected = [i for i, row in enumerate(batch) if row['projectable'] and (
                    abs(row['in_frame_fraction']-self.settings.min_in_frame_fraction) <= .08 or
                    abs(row['visible_fraction']-self.settings.min_visible_fraction) <= .08 or
                    abs(row['visible_pixels']-self.settings.min_visible_pixels) <= .2*self.settings.min_visible_pixels)]
                if selected:
                    refined = self._evaluate(target_index, o[selected], r[selected], self.settings.refinement_grid)
                    for i, row in zip(selected, refined):
                        batch[i] = row
            records.extend(batch)
        return records

    def _evaluate(self, target_index, origins, rotation, grid_size):
        cam, settings = self.camera, self.settings
        low, high = self.lower[target_index], self.upper[target_index]
        corners = low + self.corners*(high-low)
        optical = np.einsum('bki,bij->bkj', corners[None, :, :]-origins[:, None, :], rotation)
        distance = np.linalg.norm((low+high)/2-origins, axis=1)
        camera_inside = np.zeros(len(origins), dtype=bool)
        for lo, hi in zip(self.lower, self.upper):
            camera_inside |= np.all((origins >= lo) & (origins <= hi), axis=1)
        front = np.all(optical[:, :, 2] > 1e-6, axis=1)
        range_ok = (distance >= cam.min_distance-1e-9) & (distance <= cam.max_distance+1e-9)
        records = [dict(qualified=False, projectable=False, in_frame_fraction=0.0,
                        visible_fraction=0.0, visible_pixels=0.0, projected_pixels=0.0,
                        quality=0.0, center_distance_m=float(d), occluder_ids=[],
                        sample_grid=grid_size, reason=('camera_inside_geometry' if inside else
                        'behind_or_crossing_camera_plane' if not ahead else 'distance' if not valid else 'out_of_frame'))
                   for d, inside, ahead, valid in zip(distance, camera_inside, front, range_ok)]
        active = np.flatnonzero(front & range_ok & ~camera_inside)
        if not len(active):
            return records
        xyz = optical[active]
        u = cam.fx*xyz[:, :, 0]/xyz[:, :, 2]+cam.cx
        v = cam.fy*xyz[:, :, 1]/xyz[:, :, 2]+cam.cy
        bounds = np.column_stack((u.min(axis=1), v.min(axis=1), u.max(axis=1), v.max(axis=1)))
        overlaps = (bounds[:, 2] > 0) & (bounds[:, 0] < cam.width) & (bounds[:, 3] > 0) & (bounds[:, 1] < cam.height)
        active, bounds = active[overlaps], bounds[overlaps]
        if not len(active):
            return records
        grid = (np.arange(grid_size)+.5)/grid_size
        gx, gy = np.meshgrid(grid, grid)
        uu = bounds[:, :1] + (bounds[:, 2:3]-bounds[:, :1])*gx.ravel()
        vv = bounds[:, 1:2] + (bounds[:, 3:4]-bounds[:, 1:2])*gy.ravel()
        rays = np.stack(((uu-cam.cx)/cam.fx, (vv-cam.cy)/cam.fy, np.ones_like(uu)), axis=-1)
        rays = np.einsum('bsi,bji->bsj', rays, rotation[active])
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        start = origins[active, None, :]
        target_entry = ray_box_entry(start, rays, low, high)
        silhouette = np.isfinite(target_entry)
        in_frame = silhouette & (uu >= 0) & (uu < cam.width) & (vv >= 0) & (vv < cam.height)
        nearest = target_entry.copy()
        blocker = np.full(target_entry.shape, -1, dtype=int)
        for other, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            if other == target_index:
                continue
            # Every camera-to-target segment lies inside this world-space box.
            # Reject unrelated blockers before doing per-ray slab intersections.
            segment_min = np.minimum(origins[active], low)
            segment_max = np.maximum(origins[active], high)
            relevant = np.flatnonzero(np.all((hi >= segment_min) & (lo <= segment_max), axis=1))
            if not len(relevant):
                continue
            entry = ray_box_entry(start[relevant], rays[relevant], lo, hi)
            closer = in_frame[relevant] & (entry < nearest[relevant]-1e-6)
            nearest[relevant] = np.where(closer, entry, nearest[relevant])
            blocker[relevant] = np.where(closer, other, blocker[relevant])
        visible = in_frame & (blocker == -1)
        for row, original in enumerate(active):
            total = int(silhouette[row].sum())
            framed = int(in_frame[row].sum())
            exposed = int(visible[row].sum())
            cell_pixels = float((bounds[row, 2]-bounds[row, 0])*(bounds[row, 3]-bounds[row, 1])/(grid_size**2))
            framing = framed/total if total else 0.0
            visibility = exposed/framed if framed else 0.0
            pixels = exposed*cell_pixels
            reason = ('in_frame_fraction' if framing+1e-9 < settings.min_in_frame_fraction else
                      'visible_fraction' if visibility+1e-9 < settings.min_visible_fraction else
                      'visible_pixels' if pixels+1e-9 < settings.min_visible_pixels else 'qualified')
            records[original].update(qualified=reason == 'qualified', projectable=True,
                in_frame_fraction=framing, visible_fraction=visibility, visible_pixels=pixels,
                projected_pixels=total*cell_pixels, quality=framing*visibility,
                occluder_ids=[self.ids[i] for i in sorted(set(blocker[row][blocker[row] >= 0].tolist()))],
                projection_bounds=bounds[row].tolist(), reason=reason)
        return records
