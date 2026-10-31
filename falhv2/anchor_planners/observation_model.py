"""Clip projected solid AABBs for framing; sample occlusion and visible pixels.

The target's full projected silhouette is the framing denominator. Occlusion uses
only its in-frame silhouette. Rays use the nearest target surface, so back faces
are not counted. Occlusion estimates are deterministic, not analytic guarantees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from functools import lru_cache
import math
from typing import Sequence

import numpy as np
import shapely

from .observation_region import camera_poses, radial_envelope, radial_weight

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
    min_in_frame_fraction: float = .8
    min_observation_score: float = .5
    min_visible_pixels: float = 400.0
    sample_grid: int = 12
    refinement_grid: int = 36
    radial_inner_weight: float = 1.0
    radial_outer_weight: float = .5

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
    """Validate world-frame boxes without assuming their bottom lies above map Z=0.

    Signed heights are valid for sensed geometry; do not clip predictions or
    require the SLAM height origin to coincide exactly with the ground.
    """
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
                    or not np.isfinite(upper).all() or np.any(lower >= upper)
                    or not np.allclose(point, (lower+upper)/2, rtol=0, atol=1e-6)):
                raise ValueError(f'Object {identifier}: AABB must have positive sizes and position at its center')
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
    lower, upper = np.asarray(lower), np.asarray(upper)
    # Work one axis at a time. This preserves the slab arithmetic and comparison
    # tolerances while avoiding several (..., rays, 3) temporaries and reductions.
    near, far, outside = -np.inf, np.inf, False
    for axis in range(3):
        origin, direction = origins[..., axis], directions[..., axis]
        lo, hi = lower[..., axis], upper[..., axis]
        parallel = np.abs(direction) < 1e-12
        denominator = np.where(parallel, 1.0, direction)
        a, b = (lo-origin)/denominator, (hi-origin)/denominator
        near = np.maximum(near, np.where(parallel, -np.inf, np.minimum(a, b)))
        far = np.minimum(far, np.where(parallel, np.inf, np.maximum(a, b)))
        outside = outside | (parallel & ((origin < lo) | (origin > hi)))
    entry = np.maximum(near, 0.0)
    return np.where(~outside & (far >= entry), entry, np.inf)


def projection_areas_batch(points, width, height):
    """Float64 GEOS convex hulls and image intersections for a batch of boxes.

    Full silhouettes inside the image need no overlay. No Float32 conversion or
    precision-grid snapping is used; threshold comparisons retain double precision.
    """
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return np.empty((0, 2))
    hulls = shapely.convex_hull(shapely.multipoints(points))
    full = shapely.area(hulls)
    framed = full.copy()
    inside = np.all((points >= 0) & (points <= [width, height]), axis=(1, 2))
    if np.any(~inside):
        clipped = shapely.intersection(hulls[~inside], _image_rectangle(width, height), grid_size=0)
        framed[~inside] = shapely.area(clipped)
    return np.column_stack((full, np.minimum(full, framed)))


@lru_cache(maxsize=16)
def _image_rectangle(width, height):
    return shapely.box(0, 0, width, height)


def projection_areas(points, width, height):
    """Scalar compatibility wrapper around the batch geometry library."""
    return tuple(projection_areas_batch(np.asarray(points)[None], width, height)[0])


@lru_cache(maxsize=16)
def _sampling_grid(size):
    """Immutable normalized pixel offsets, shared by coarse and refined batches."""
    grid = (np.arange(size)+.5)/size
    gx, gy = np.meshgrid(grid, grid)
    gx, gy = gx.ravel(), gy.ravel()
    gx.setflags(write=False)
    gy.setflags(write=False)
    return gx, gy


REASONS = ('out_of_frame', 'camera_inside_geometry', 'behind_or_crossing_camera_plane',
           'distance', 'in_frame_fraction', 'visible_pixels', 'observation_score', 'qualified')
_FLOAT_FIELDS = ('in_frame_fraction', 'visible_fraction', 'visible_pixels', 'projected_pixels',
                 'quality', 'observation_score', 'distance_weight', 'horizontal_distance_m',
                 'radial_near_m', 'radial_far_m', 'center_distance_m')


@dataclass
class ObservationBatch:
    """Columnar observation results; dictionaries are materialized only on demand.

    NaN denotes a deliberately unsampled metric or an invalid radial bound, never
    a score of zero. The legacy record interface converts these values to None.
    """
    columns: dict
    bounds: np.ndarray
    blockers: np.ndarray
    object_ids: list

    def __len__(self):
        return len(self.columns['qualified'])

    def __getitem__(self, key):
        return self.columns[key]

    @classmethod
    def empty(cls, size, object_ids, grid):
        columns = {key: np.zeros(size) for key in _FLOAT_FIELDS}
        columns.update(qualified=np.zeros(size, dtype=bool), projectable=np.zeros(size, dtype=bool),
                       sample_grid=np.full(size, grid, dtype=np.int16), reason=np.zeros(size, dtype=np.int8))
        return cls(columns, np.zeros((size, 4)), np.zeros((size, len(object_ids)), dtype=bool), object_ids)

    def view(self, start, stop):
        return ObservationBatch({key: value[start:stop] for key, value in self.columns.items()},
                                self.bounds[start:stop], self.blockers[start:stop], self.object_ids)

    def record(self, index):
        row = {key: (None if np.isnan(self[key][index]) else float(self[key][index]))
               for key in _FLOAT_FIELDS}
        row.update(qualified=bool(self['qualified'][index]), projectable=bool(self['projectable'][index]),
                   sample_grid=int(self['sample_grid'][index]), reason=REASONS[self['reason'][index]],
                   occluder_ids=([self.object_ids[i] for i in np.flatnonzero(self.blockers[index])]
                                 if not np.isnan(self['observation_score'][index]) else None))
        if row['projectable']:
            row['projection_bounds'] = self.bounds[index].tolist()
        return row

    def records(self):
        return [self.record(i) for i in range(len(self))]


class Open3DRaycaster:
    """Cached native scene queries; Float32 is not equivalent to the Float64 reference."""
    def __init__(self, lower, upper, *, nthreads=0):
        # Keep the heavy import out of normal NumPy planning processes.
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ValueError('Open3D backend requires open3d==0.19.0 in the active environment') from exc
        self.o3d = o3d
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.nthreads = nthreads
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=nthreads)
        self.targets = []
        self.object_by_geometry = {}
        for index, (lo, hi) in enumerate(zip(lower, upper)):
            mesh = o3d.geometry.TriangleMesh.create_box(*map(float, hi-lo))
            mesh.translate(lo)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            geometry_id = self.scene.add_triangles(mesh)
            self.object_by_geometry[geometry_id] = index
            target = o3d.t.geometry.RaycastingScene(nthreads=nthreads)
            target.add_triangles(mesh)
            self.targets.append(target)
        self.lookup = np.full(max(self.object_by_geometry, default=-1)+1, -1, dtype=int)
        for geometry_id, index in self.object_by_geometry.items():
            self.lookup[geometry_id] = index

    def trace(self, target_index, origins, directions, in_image, *, target_entry=None):
        rays = np.empty((*directions.shape[:-1], 6), dtype=np.float32)
        rays[..., :3] = origins
        rays[..., 3:] = directions
        if target_entry is None:
            target = self.targets[target_index].cast_rays(self.o3d.core.Tensor(rays), nthreads=self.nthreads)
            target_entry = target['t_hit'].numpy().astype(np.float64)
        in_frame = np.isfinite(target_entry) & in_image
        blocker = np.full(in_frame.shape, -1, dtype=int)
        if np.any(in_frame):
            hits = self.scene.cast_rays(self.o3d.core.Tensor(rays[in_frame]), nthreads=self.nthreads)
            distance = hits['t_hit'].numpy().astype(np.float64)
            ids = hits['geometry_ids'].numpy()
            closer = (distance < target_entry[in_frame]-1e-6) & (ids < len(self.lookup))
            objects = np.full(len(ids), -1, dtype=int)
            objects[closer] = self.lookup[ids[closer]]
            objects[objects == target_index] = -1
            blocker[in_frame] = objects
        return in_frame, blocker


class BoxObserver:
    # Bounded batches keep ray temporaries independent of the candidate pool size.
    batch_size = 128

    def __init__(self, camera, objects, settings=None, *, ray_backend='numpy', ray_threads=0):
        self.camera = camera
        self.objects = validate_objects(objects)
        if any('aabb' not in obj for obj in self.objects):
            raise ValueError('BoxObserver requires an AABB for every object')
        self.settings = settings or ObservationSettings.from_config(camera.config)
        self.lower = np.array([o['aabb']['min'] for o in self.objects], dtype=float).reshape(-1, 3)
        self.upper = np.array([o['aabb']['max'] for o in self.objects], dtype=float).reshape(-1, 3)
        self.ids = [o['id'] for o in self.objects]
        self.corners = np.array(list(product((0, 1), repeat=3)), dtype=float)
        self.box_corners = self.lower[:, None, :] + self.corners * (self.upper-self.lower)[:, None, :]
        if ray_backend not in ('numpy', 'open3d'):
            raise ValueError('ray_backend must be numpy or open3d')
        self.ray_backend = ray_backend
        self.ray_threads = ray_threads
        self._raycaster = None

    def poses(self, positions, yaws):
        return camera_poses(self.camera, positions, yaws)

    def envelope(self, target_index, positions, yaws):
        origins, rotation = self.poses(positions, yaws)
        low, high = self.lower[target_index], self.upper[target_index]
        return radial_envelope(self.camera, (low+high)/2,
                               self.box_corners[target_index], origins, rotation)

    def evaluate(self, target_index, positions, yaws, *, candidate_only=False):
        """Compatibility interface for audits and callers needing full records."""
        return self.evaluate_arrays(target_index, positions, yaws, candidate_only=candidate_only).records()

    def evaluate_arrays(self, target_index, positions, yaws, *, candidate_only=False):
        """Batch numeric interface, retaining the original sampling/refinement gates."""
        origins, rotation = self.poses(positions, yaws)
        low, high = self.lower[target_index], self.upper[target_index]
        envelopes = radial_envelope(self.camera, (low+high)/2,
                                    self.box_corners[target_index], origins, rotation)
        result = ObservationBatch.empty(len(origins), self.ids, self.settings.sample_grid)
        for start in range(0, len(origins), self.batch_size):
            stop = start+self.batch_size
            o, r = origins[start:stop], rotation[start:stop]
            envelope = tuple(values[start:stop] for values in envelopes)
            batch = result.view(start, stop)
            geometry = self._prepare_geometry(target_index, o, r, envelope, batch)
            active = geometry['active']
            if candidate_only:
                frame_fail = batch['in_frame_fraction'][active]+1e-9 < self.settings.min_in_frame_fraction
                pixel_fail = geometry['areas'][active, 1]+1e-9 < self.settings.min_visible_pixels
                skipped = active[frame_fail | pixel_fail]
                batch['reason'][skipped] = np.where(frame_fail[frame_fail | pixel_fail], 4, 5)
                for key in ('visible_fraction', 'visible_pixels', 'quality', 'observation_score'):
                    batch[key][skipped] = np.nan
                active = active[~(frame_fail | pixel_fail)]
            self._sample_occlusion(target_index, geometry, active, self.settings.sample_grid)
            if self.settings.refinement_grid > self.settings.sample_grid:
                refine = ((np.abs(batch['in_frame_fraction'][active]-self.settings.min_in_frame_fraction) <= .08)
                          | (np.abs(batch['observation_score'][active]-self.settings.min_observation_score) <= .08)
                          | (np.abs(batch['visible_pixels'][active]-self.settings.min_visible_pixels)
                             <= .2*self.settings.min_visible_pixels))
                self._sample_occlusion(target_index, geometry, active[refine], self.settings.refinement_grid)
        return result

    def _prepare_geometry(self, target_index, origins, rotation, envelope, batch):
        """Prepare numeric projection/framing arrays once for both sample densities."""
        cam, settings = self.camera, self.settings
        low, high = self.lower[target_index], self.upper[target_index]
        corners = self.box_corners[target_index]
        optical = np.matmul(corners[None, :, :]-origins[:, None, :], rotation)
        distance = np.linalg.norm((low+high)/2-origins, axis=1)
        radius, near, far, region_ok = envelope
        camera_inside = np.any(np.all((origins[:, None, :] >= self.lower)
                                      & (origins[:, None, :] <= self.upper), axis=2), axis=1)
        front = np.all(optical[:, :, 2] > 1e-6, axis=1)
        range_ok = (distance >= cam.min_distance-1e-9) & (distance <= cam.max_distance+1e-9)
        batch['horizontal_distance_m'][:] = radius
        batch['radial_near_m'][:] = np.where(region_ok, near, np.nan)
        batch['radial_far_m'][:] = np.where(region_ok, far, np.nan)
        batch['center_distance_m'][:] = distance
        batch['reason'][:] = np.select([camera_inside, ~front, ~range_ok], [1, 2, 3], default=0)
        active = np.flatnonzero(front & range_ok & ~camera_inside & region_ok)
        bounds = batch.bounds
        areas = np.zeros((len(origins), 2))
        if len(active):
            xyz = optical[active]
            u = cam.fx*xyz[:, :, 0]/xyz[:, :, 2]+cam.cx
            v = cam.fy*xyz[:, :, 1]/xyz[:, :, 2]+cam.cy
            projected_bounds = np.column_stack((u.min(axis=1), v.min(axis=1), u.max(axis=1), v.max(axis=1)))
            overlaps = ((projected_bounds[:, 2] > 0) & (projected_bounds[:, 0] < cam.width)
                        & (projected_bounds[:, 3] > 0) & (projected_bounds[:, 1] < cam.height))
            polygons = np.stack((u, v), axis=-1)[overlaps]
            active = active[overlaps]
            bounds[active] = projected_bounds[overlaps]
            weights = radial_weight(radius[active], near[active], far[active],
                                    settings.radial_inner_weight, settings.radial_outer_weight)
            if len(active):
                areas[active] = projection_areas_batch(polygons, cam.width, cam.height)
            batch['projectable'][active] = True
            batch['projected_pixels'][active] = areas[active, 0]
            batch['in_frame_fraction'][active] = np.divide(areas[active, 1], areas[active, 0],
                                                          out=np.zeros(len(active)), where=areas[active, 0] > 0)
            batch['distance_weight'][active] = weights
        segment_min = np.minimum(origins, low)
        segment_max = np.maximum(origins, high)
        relevant = np.all((self.upper[:, None, :] >= segment_min)
                          & (self.lower[:, None, :] <= segment_max), axis=2)
        return dict(origins=origins, rotation=rotation, metrics=batch, active=active,
                    bounds=bounds, areas=areas, relevant=relevant)

    def _sample_occlusion(self, target_index, geometry, active, grid_size):
        """Sample rays, then use NumPy reductions for all poses in the batch."""
        if not len(active):
            return
        cam, settings = self.camera, self.settings
        origins, rotation = geometry['origins'], geometry['rotation']
        bounds = geometry['bounds'][active]
        gx, gy = _sampling_grid(grid_size)
        uu = bounds[:, :1] + (bounds[:, 2:3]-bounds[:, :1])*gx
        vv = bounds[:, 1:2] + (bounds[:, 3:4]-bounds[:, 1:2])*gy
        rays = np.stack(((uu-cam.cx)/cam.fx, (vv-cam.cy)/cam.fy, np.ones_like(uu)), axis=-1)
        rays = np.matmul(rays, rotation[active].swapaxes(-1, -2))
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        start = origins[active, None, :]
        in_image = (uu >= 0) & (uu < cam.width) & (vv >= 0) & (vv < cam.height)
        if self.ray_backend == 'open3d':
            if self._raycaster is None:
                self._raycaster = Open3DRaycaster(self.lower, self.upper, nthreads=self.ray_threads)
            in_frame, blocker = self._raycaster.trace(target_index, start, rays, in_image)
        else:
            in_frame, blocker = self._trace_numpy(target_index, geometry, active, start, rays, in_image)
        self._summarize_samples(geometry, active, in_frame, blocker, grid_size)

    def _trace_numpy(self, target_index, geometry, active, start, rays, in_image):
        target_entry = ray_box_entry(start, rays, self.lower[target_index], self.upper[target_index])
        in_frame = np.isfinite(target_entry) & in_image
        nearest = target_entry.copy()
        blocker = np.full(target_entry.shape, -1, dtype=int)
        for other, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            if other == target_index:
                continue
            relevant = np.flatnonzero(geometry['relevant'][other, active])
            if not len(relevant):
                continue
            entry = ray_box_entry(start[relevant], rays[relevant], lo, hi)
            closer = in_frame[relevant] & (entry < nearest[relevant]-1e-6)
            nearest[relevant] = np.where(closer, entry, nearest[relevant])
            blocker[relevant] = np.where(closer, other, blocker[relevant])
        return in_frame, blocker

    def _summarize_samples(self, geometry, active, in_frame, blocker, grid_size):
        settings = self.settings
        visible = in_frame & (blocker == -1)
        framed = np.count_nonzero(in_frame, axis=1)
        exposed = np.count_nonzero(visible, axis=1)
        visibility = np.divide(exposed, framed, out=np.zeros(len(active)), where=framed != 0)
        batch = geometry['metrics']
        framing = batch['in_frame_fraction'][active]
        pixels = geometry['areas'][active, 1]*visibility
        score = batch['distance_weight'][active]*visibility
        reason = np.select([framing+1e-9 < settings.min_in_frame_fraction,
                            pixels+1e-9 < settings.min_visible_pixels,
                            score+1e-9 < settings.min_observation_score], [4, 5, 6], default=7)
        batch['reason'][active] = reason
        batch['qualified'][active] = reason == 7
        batch['visible_fraction'][active] = visibility
        batch['visible_pixels'][active] = pixels
        batch['quality'][active] = framing*visibility
        batch['observation_score'][active] = score
        batch['sample_grid'][active] = grid_size
        batch.blockers[active] = False
        rows, cols = np.nonzero(blocker >= 0)
        batch.blockers[active[rows], blocker[rows, cols]] = True
