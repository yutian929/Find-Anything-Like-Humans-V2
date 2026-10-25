"""Clip projected solid AABBs for framing; sample occlusion and visible pixels.

The target's full projected silhouette is the framing denominator. Occlusion uses
only its in-frame silhouette. Rays use the nearest target surface, so back faces
are not counted. Occlusion estimates are deterministic, not analytic guarantees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
import math
from typing import Sequence

import numpy as np
import cv2

try:
    from . import config as project_config
    from .observation_region import camera_poses, radial_envelope, radial_weight
except ImportError:
    import config as project_config
    from observation_region import camera_poses, radial_envelope, radial_weight


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


@dataclass(frozen=True)
class ObservationSettings:
    min_in_frame_fraction: float = field(default_factory=lambda: project_config.OBSERVATION_DEFAULTS['min_in_frame_fraction'])
    min_observation_score: float = field(default_factory=lambda: project_config.OBSERVATION_DEFAULTS['min_observation_score'])
    min_visible_pixels: float = field(default_factory=lambda: project_config.OBSERVATION_DEFAULTS['min_visible_pixels'])
    sample_grid: int = field(default_factory=lambda: project_config.OBSERVATION_DEFAULTS['sample_grid'])
    refinement_grid: int = field(default_factory=lambda: project_config.OBSERVATION_DEFAULTS['refinement_grid'])
    radial_inner_weight: float = field(default_factory=lambda: project_config.OBSERVATION_DEFAULTS['radial_inner_weight'])
    radial_outer_weight: float = field(default_factory=lambda: project_config.OBSERVATION_DEFAULTS['radial_outer_weight'])

    @classmethod
    def from_config(cls, config):
        values = config.get('observation_quality', {})
        if 'min_visible_fraction' in values:
            raise ValueError('min_visible_fraction 已由 min_observation_score 替代；请删除旧字段')
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})

    def __post_init__(self):
        for key in ('min_in_frame_fraction', 'min_observation_score', 'radial_inner_weight', 'radial_outer_weight'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{key} must be finite and in (0, 1]')
        if self.radial_outer_weight > self.radial_inner_weight:
            raise ValueError('radial_outer_weight must not exceed radial_inner_weight')
        if (isinstance(self.min_visible_pixels, bool) or not isinstance(self.min_visible_pixels, (float, int))
                or not math.isfinite(self.min_visible_pixels) or self.min_visible_pixels <= 0):
            raise ValueError('min_visible_pixels must be finite and positive')
        for key in ('sample_grid', 'refinement_grid'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not 4 <= value <= 128:
                raise ValueError(f'{key} must be an integer in [4, 128]')
        if self.refinement_grid < self.sample_grid:
            raise ValueError('refinement_grid must be at least sample_grid')

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


def projection_areas(points, width, height):
    """Convex box silhouette and its image-clipped area, in pixel squared.

    OpenCV supplies only hull indices; all clipping/area arithmetic stays float64
    to avoid float32 image-edge jitter at the framing threshold.
    """
    indices = cv2.convexHull(points.astype(np.float32), returnPoints=False).ravel()
    polygon = [tuple(point) for point in points[indices]]

    def area(poly):
        if len(poly) < 3:
            return 0.0
        x0, y0 = poly[0]
        return abs(sum((a[0]-x0)*(b[1]-y0)-(a[1]-y0)*(b[0]-x0)
                       for a, b in zip(poly[1:-1], poly[2:])))/2

    full = area(polygon)
    for axis, bound, sign in ((0, 0, 1), (0, width, -1), (1, 0, 1), (1, height, -1)):
        if not polygon:
            break
        clipped = []
        previous = polygon[-1]
        prev_inside = sign*(previous[axis]-bound) >= 0
        for point in polygon:
            inside = sign*(point[axis]-bound) >= 0
            if inside != prev_inside:
                t = (bound-previous[axis])/(point[axis]-previous[axis])
                clipped.append(tuple(previous[k]+t*(point[k]-previous[k]) for k in (0, 1)))
            if inside:
                clipped.append(point)
            previous, prev_inside = point, inside
        polygon = clipped
    return full, min(full, area(polygon))


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
        return camera_poses(self.camera, positions, yaws)

    def envelope(self, target_index, positions, yaws):
        origins, rotation = self.poses(positions, yaws)
        low, high = self.lower[target_index], self.upper[target_index]
        return radial_envelope(self.camera, (low+high)/2,
                               low+self.corners*(high-low), origins, rotation)

    def evaluate(self, target_index, positions, yaws):
        """Return one metric record per pose, refining near qualification thresholds."""
        origins, rotation = self.poses(positions, yaws)
        low, high = self.lower[target_index], self.upper[target_index]
        envelopes = radial_envelope(self.camera, (low+high)/2,
                                    low+self.corners*(high-low), origins, rotation)
        records = []
        # Bound temporary arrays even for fine user-specified sampling.
        for start in range(0, len(origins), 32):
            o, r = origins[start:start+32], rotation[start:start+32]
            envelope = tuple(values[start:start+32] for values in envelopes)
            batch = self._evaluate(target_index, o, r, self.settings.sample_grid, envelope)
            if self.settings.refinement_grid > self.settings.sample_grid:
                selected = [i for i, row in enumerate(batch) if row['projectable'] and (
                    abs(row['in_frame_fraction']-self.settings.min_in_frame_fraction) <= .08 or
                    abs(row['observation_score']-self.settings.min_observation_score) <= .08 or
                    abs(row['visible_pixels']-self.settings.min_visible_pixels) <= .2*self.settings.min_visible_pixels)]
                if selected:
                    refined = self._evaluate(target_index, o[selected], r[selected], self.settings.refinement_grid,
                                             tuple(values[selected] for values in envelope))
                    for i, row in zip(selected, refined):
                        batch[i] = row
            records.extend(batch)
        return records

    def _evaluate(self, target_index, origins, rotation, grid_size, envelope=None):
        cam, settings = self.camera, self.settings
        low, high = self.lower[target_index], self.upper[target_index]
        corners = low + self.corners*(high-low)
        optical = np.einsum('bki,bij->bkj', corners[None, :, :]-origins[:, None, :], rotation)
        distance = np.linalg.norm((low+high)/2-origins, axis=1)
        radius, near, far, region_ok = (radial_envelope(cam, (low+high)/2, corners, origins, rotation)
                                      if envelope is None else envelope)
        camera_inside = np.zeros(len(origins), dtype=bool)
        for lo, hi in zip(self.lower, self.upper):
            camera_inside |= np.all((origins >= lo) & (origins <= hi), axis=1)
        front = np.all(optical[:, :, 2] > 1e-6, axis=1)
        range_ok = (distance >= cam.min_distance-1e-9) & (distance <= cam.max_distance+1e-9)
        records = [dict(qualified=False, projectable=False, in_frame_fraction=0.0,
                        visible_fraction=0.0, visible_pixels=0.0, projected_pixels=0.0,
                        quality=0.0, observation_score=0.0, distance_weight=0.0,
                        horizontal_distance_m=float(radius[i]),
                        radial_near_m=float(near[i]) if region_ok[i] else None,
                        radial_far_m=float(far[i]) if region_ok[i] else None,
                        center_distance_m=float(d), occluder_ids=[],
                        sample_grid=grid_size, reason=('camera_inside_geometry' if inside else
                        'behind_or_crossing_camera_plane' if not ahead else 'distance' if not valid else 'out_of_frame'))
                   for i, (d, inside, ahead, valid) in enumerate(zip(distance, camera_inside, front, range_ok))]
        active = np.flatnonzero(front & range_ok & ~camera_inside & region_ok)
        if not len(active):
            return records
        xyz = optical[active]
        u = cam.fx*xyz[:, :, 0]/xyz[:, :, 2]+cam.cx
        v = cam.fy*xyz[:, :, 1]/xyz[:, :, 2]+cam.cy
        bounds = np.column_stack((u.min(axis=1), v.min(axis=1), u.max(axis=1), v.max(axis=1)))
        overlaps = (bounds[:, 2] > 0) & (bounds[:, 0] < cam.width) & (bounds[:, 3] > 0) & (bounds[:, 1] < cam.height)
        polygons = np.stack((u, v), axis=-1)[overlaps]
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
            framed = int(in_frame[row].sum())
            exposed = int(visible[row].sum())
            # With all corners in front, the solid box silhouette is their
            # convex hull. Clip it geometrically: sampled framing oscillates at
            # the 80% gate and can falsely certify a cropped final observation.
            projected_area, framed_area = projection_areas(polygons[row], cam.width, cam.height)
            framing = framed_area/projected_area if projected_area else 0.0
            visibility = exposed/framed if framed else 0.0
            pixels = framed_area*visibility
            weight = float(radial_weight(radius[original], near[original], far[original],
                                        settings.radial_inner_weight, settings.radial_outer_weight))
            score = weight*visibility
            reason = ('in_frame_fraction' if framing+1e-9 < settings.min_in_frame_fraction else
                      'visible_pixels' if pixels+1e-9 < settings.min_visible_pixels else
                      'observation_score' if score+1e-9 < settings.min_observation_score else 'qualified')
            records[original].update(qualified=reason == 'qualified', projectable=True,
                in_frame_fraction=framing, visible_fraction=visibility, visible_pixels=pixels,
                projected_pixels=projected_area, quality=framing*visibility,
                observation_score=score, distance_weight=weight,
                occluder_ids=[self.ids[i] for i in sorted(set(blocker[row][blocker[row] >= 0].tolist()))],
                projection_bounds=bounds[row].tolist(), reason=reason)
        return records
