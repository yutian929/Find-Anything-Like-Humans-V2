"""Depth-bounded replacement of category-free AABBs; no association or fusion."""
from __future__ import annotations

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import numpy as np

from ..robot_adapters.base import GridSnapshot, validate_frame
from .base import MapSnapshot, ObjectTrack
from .cubify import cuboids_to_map_aabbs


def corners(lower, upper):
    return np.array(list(product(*zip(lower, upper))), dtype=float)


def visibility(frame, lower, upper, *, sample_grid=32, depth_tolerance_m=.08,
               min_depth_m=.15, max_depth_m=8.):
    """Sample the FULL projected silhouette, including portions outside the image.

    Ray parameter is optical Z, not Euclidean distance. Intersections are with
    the original map-axis box, not a re-axis-aligned camera box. Near-plane
    crossing boxes are conservatively retained in this initial implementation.
    """
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
        raise ValueError("Invalid map AABB")
    T, k = frame.T_map_camera, frame.intrinsics
    camera = (corners(lo, hi) - T[:3, 3]) @ T[:3, :3]
    empty = dict(visible_fraction=0., samples=0, visible_samples=0,
                 valid_depth_samples=0, reason="outside_or_near_plane")
    if camera[:, 2].min() <= min_depth_m:
        return empty
    uv = camera[:, :2] / camera[:, 2, None]
    uv = uv * [k.fx, k.fy] + [k.cx, k.cy]
    low, high = uv.min(axis=0), uv.max(axis=0)
    if high[0] < -.5 or high[1] < -.5 or low[0] >= k.width-.5 or low[1] >= k.height-.5:
        return empty
    a = (np.arange(sample_grid) + .5) / sample_grid
    u, v = np.meshgrid(low[0]+a*(high[0]-low[0]), low[1]+a*(high[1]-low[1]))
    u, v = u.ravel(), v.ravel()
    directions = np.stack(((u-k.cx)/k.fx, (v-k.cy)/k.fy, np.ones_like(u)), axis=1) @ T[:3, :3].T
    origin = T[:3, 3]
    near, far = np.full(len(u), -np.inf), np.full(len(u), np.inf)
    hit = np.ones(len(u), bool)
    for axis in range(3):
        parallel = np.abs(directions[:, axis]) < 1e-12
        hit &= ~parallel | ((origin[axis] >= lo[axis]) & (origin[axis] <= hi[axis]))
        d = np.where(parallel, 1., directions[:, axis])
        t0, t1 = (lo[axis]-origin[axis])/d, (hi[axis]-origin[axis])/d
        near = np.maximum(near, np.where(parallel, -np.inf, np.minimum(t0, t1)))
        far = np.minimum(far, np.where(parallel, np.inf, np.maximum(t0, t1)))
    hit &= (far >= near) & (far > 0)
    inside = (u >= -.5) & (u < k.width-.5) & (v >= -.5) & (v < k.height-.5)
    # Clamp ONLY indices, never the coverage denominator or sampled rays.
    ix = np.clip(np.floor(u+.5).astype(int), 0, k.width-1)
    iy = np.clip(np.floor(v+.5).astype(int), 0, k.height-1)
    depth = frame.depth_m[iy, ix]
    valid = hit & inside & np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    visible = valid & (near >= min_depth_m) & (near <= max_depth_m) & (near <= depth+depth_tolerance_m)
    count = int(hit.sum())
    return dict(visible_fraction=float(visible.sum()/count) if count else 0.,
                samples=count, visible_samples=int(visible.sum()),
                valid_depth_samples=int(valid.sum()), reason="depth_test")


def validate_grid(grid):
    if (not grid.map_id or not grid.map_revision or not grid.clock_domain
            or not np.isfinite([grid.resolution, *grid.origin]).all()
            or grid.resolution <= 0 or len(grid.origin) != 2 or not grid.rows
            or not grid.rows[0] or any(len(r) != len(grid.rows[0]) or set(r)-set(".#?")
                                      for r in grid.rows)):
        raise ValueError("Invalid grid snapshot")


class DepthReplacementMap:
    """Keep only current records. IDs identify observations, not tracked objects."""

    def __init__(self, *, visible_fraction=.5, sample_grid=32, depth_tolerance_m=.08,
                 min_depth_m=.15, max_depth_m=8.):
        if (not np.isfinite([visible_fraction, depth_tolerance_m, min_depth_m, max_depth_m]).all()
                or not 0 < visible_fraction <= 1 or depth_tolerance_m < 0
                or not 0 < min_depth_m < max_depth_m
                or isinstance(sample_grid, bool) or not isinstance(sample_grid, int)
                or not 4 <= sample_grid <= 128):
            raise ValueError("Invalid depth replacement parameters")
        self.options = dict(sample_grid=sample_grid, depth_tolerance_m=depth_tolerance_m,
                            min_depth_m=min_depth_m, max_depth_m=max_depth_m)
        self.visible_fraction = visible_fraction
        self.grid = None
        self.objects = ()
        self.revision = 0
        self.last_timestamp_ns = 0
        self.last_frame_id = None
        self.last_update = {}

    def set_grid(self, grid):
        validate_grid(grid)
        if self.grid is not None and (grid.map_id, grid.map_revision, grid.clock_domain) != (
                self.grid.map_id, self.grid.map_revision, self.grid.clock_domain):
            raise ValueError("Map coordinates changed; start a new object map")
        self.grid = grid

    def integrate(self, frame, result):
        if self.grid is None:
            raise ValueError("Set grid before integrating")
        if (frame.map_id, frame.map_revision, frame.clock_domain) != (
                self.grid.map_id, self.grid.map_revision, self.grid.clock_domain):
            raise ValueError("Frame and grid coordinates/clock do not match")
        if frame.timestamp_ns <= self.last_timestamp_ns or frame.frame_id == self.last_frame_id:
            raise ValueError("Repeated or out-of-order frame")
        validate_frame(frame)
        # Validate all predictions before deleting anything, including empty result.
        observations = cuboids_to_map_aabbs(frame, result)
        retained, decisions, removed = [], [], []
        for obj in self.objects:
            check = visibility(frame, obj.lower, obj.upper, **self.options)
            delete = check["samples"] > 0 and check["visible_fraction"] >= self.visible_fraction
            decisions.append(dict(object_id=obj.object_id, deleted=delete, **check))
            if delete:
                removed.append(obj.object_id)
            else:
                retained.append(obj)
        revision = self.revision + 1
        added = tuple(ObjectTrack(
            f"observation-{revision}-{obs.detection_index}", obs.lower, obs.upper,
            "confirmed", frame.timestamp_ns, frame.timestamp_ns, 1, (frame.frame_id,))
            for obs in observations)
        # Publish atomically after every geometry/validation step succeeded.
        self.objects = tuple(retained) + added
        self.revision = revision
        self.last_timestamp_ns, self.last_frame_id = frame.timestamp_ns, frame.frame_id
        self.last_update = dict(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns,
                                removed_ids=removed, added_ids=[o.object_id for o in added],
                                decisions=decisions, model_version=result.model_version)
        return self.last_update

    def snapshot(self):
        if self.grid is None:
            raise ValueError("No grid available")
        return MapSnapshot(str(self.revision), self.grid, self.objects)

    def to_scene(self):
        """Planner-compatible scene AND resume state in a single JSON document."""
        snap = self.snapshot()
        return dict(name="Depth replacement object map",
                    grid=dict(rows=list(snap.grid.rows), origin=list(snap.grid.origin),
                              resolution=snap.grid.resolution),
                    objects=[dict(id=o.object_id, position=((np.array(o.lower)+o.upper)/2).tolist(),
                                  aabb=dict(min=list(o.lower), max=list(o.upper)))
                             for o in snap.objects],
                    mapping=dict(schema_version=1, policy="depth_replacement",
                                 grid_metadata={k:v for k,v in asdict(snap.grid).items()
                                                if k not in ("rows", "origin", "resolution")},
                                 revision=self.revision, last_timestamp_ns=self.last_timestamp_ns,
                                 last_frame_id=self.last_frame_id,
                                 options=dict(visible_fraction=self.visible_fraction, **self.options),
                                 records=[asdict(o) for o in self.objects], last_update=self.last_update))

    def save(self, path):
        import os
        import tempfile
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_scene(), ensure_ascii=False, indent=2, allow_nan=False)
        fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, **options):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = data["mapping"]
        if meta["schema_version"] != 1 or meta["policy"] != "depth_replacement":
            raise ValueError("Unsupported map state")
        builder = cls(**(options or meta["options"]))
        grid = data["grid"]
        builder.set_grid(GridSnapshot(**meta["grid_metadata"], rows=tuple(grid["rows"]),
                                     origin=tuple(grid["origin"]), resolution=grid["resolution"]))
        records = []
        for raw in meta["records"]:
            item = dict(raw)
            for key in ("lower", "upper", "evidence_frame_ids"):
                item[key] = tuple(item[key])
            obj = ObjectTrack(**item)
            lo, hi = np.asarray(obj.lower), np.asarray(obj.upper)
            if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all()
                    or np.any(lo >= hi)):
                raise ValueError("Invalid saved AABB")
            records.append(obj)
        builder.objects = tuple(records)
        builder.revision = int(meta["revision"])
        builder.last_timestamp_ns = int(meta["last_timestamp_ns"])
        builder.last_frame_id = meta["last_frame_id"]
        builder.last_update = meta["last_update"]
        if builder.revision < 0 or builder.last_timestamp_ns < 0:
            raise ValueError("Invalid map history")
        if builder.to_scene()["objects"] != data["objects"]:
            raise ValueError("Scene objects differ from resume records")
        return builder


def depth_coverage(frame, grid, *, pixel_stride=8, min_depth_m=.15, max_depth_m=8.):
    """Sparse XY raster of observed ray segments, for display ONLY.

    No convex hull, interpolation across invalid depth, or modification of SLAM
    occupancy. Height is collapsed, so this MUST NOT be used for deleting boxes.
    """
    validate_grid(grid)
    validate_frame(frame)
    if (frame.map_id, frame.map_revision) != (grid.map_id, grid.map_revision):
        raise ValueError("Frame/grid coordinate mismatch")
    if not isinstance(pixel_stride, int) or pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    k, T = frame.intrinsics, frame.T_map_camera
    height, width = len(grid.rows), len(grid.rows[0])
    mask = np.zeros((height, width), bool)
    v, u = np.mgrid[0:k.height:pixel_stride, 0:k.width:pixel_stride]
    z = frame.depth_m[::pixel_stride, ::pixel_stride].ravel()
    rays = np.stack(((u.ravel()-k.cx)/k.fx, (v.ravel()-k.cy)/k.fy,
                     np.ones(u.size)), axis=1) @ T[:3, :3].T
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    # Chunk rays to bound temporary memory, step at half a grid cell in XY.
    for start in range(0, len(z), 256):
        d, end = rays[start:start+256][valid[start:start+256]], z[start:start+256][valid[start:start+256]]
        if not len(end):
            continue
        lengths = (end-min_depth_m)*np.linalg.norm(d[:, :2], axis=1)
        steps = max(2, int(np.ceil(lengths.max()/(grid.resolution*.5)))+1)
        t = min_depth_m+(end[:, None]-min_depth_m)*np.linspace(0, 1, steps)
        xy = T[:2, 3]+d[:, None, :2]*t[:, :, None]
        col = np.floor((xy[:, :, 0]-grid.origin[0])/grid.resolution).astype(int)
        row = height-1-np.floor((xy[:, :, 1]-grid.origin[1])/grid.resolution).astype(int)
        inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        mask[row[inside], col[inside]] = True
    return mask
