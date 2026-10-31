"""Depth-bounded replacement of category-free AABBs; no association or fusion."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from ..robot_adapters.base import GridSnapshot, validate_frame
from .base import MapSnapshot, ObjectTrack
from .cubify import cuboids_to_map_aabbs


def intersects_coverage(lower, upper, grid, coverage):
    """XY AABB and observed region share at least one map cell."""
    import math
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
        raise ValueError("Invalid map AABB")
    h, w = coverage.shape
    x0, y0 = grid.origin
    r = grid.resolution
    if hi[0] < x0 or hi[1] < y0 or lo[0] >= x0+w*r or lo[1] >= y0+h*r:
        return False
    c0 = max(0, math.floor((lo[0]-x0)/r))
    c1 = min(w-1, math.floor((hi[0]-x0)/r))
    bottom = max(0, math.floor((lo[1]-y0)/r))
    top = min(h-1, math.floor((hi[1]-y0)/r))
    return bool(coverage[h-1-top:h-bottom, c0:c1+1].any())


def validate_grid(grid):
    if (not grid.map_id or not grid.map_revision or not grid.clock_domain
            or not np.isfinite([grid.resolution, *grid.origin]).all()
            or grid.resolution <= 0 or len(grid.origin) != 2 or not grid.rows
            or not grid.rows[0] or any(len(r) != len(grid.rows[0]) or set(r)-set(".#?")
                                      for r in grid.rows)):
        raise ValueError("Invalid grid snapshot")


class DepthReplacementMap:
    """Keep only current records. IDs identify observations, not tracked objects."""

    def __init__(self, *, pixel_stride=4, min_depth_m=.15, max_depth_m=8.,
                 min_object_height_m=.10):
        if (not np.isfinite([min_depth_m, max_depth_m]).all()
                or not 0 < min_depth_m < max_depth_m
                or isinstance(pixel_stride, bool) or not isinstance(pixel_stride, int)
                or pixel_stride < 1):
            raise ValueError("Invalid depth replacement parameters")
        if (isinstance(min_object_height_m, bool)
                or not isinstance(min_object_height_m, (int, float))
                or not np.isfinite(min_object_height_m) or min_object_height_m < 0):
            raise ValueError("min_object_height_m must be finite and nonnegative")
        self.min_object_height_m = float(min_object_height_m)
        self.options = dict(pixel_stride=pixel_stride, min_depth_m=min_depth_m,
                            max_depth_m=max_depth_m)
        self.coverage = None
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
        coverage = depth_coverage(frame, self.grid, **self.options)
        retained, decisions, removed = [], [], []
        for obj in self.objects:
            delete = intersects_coverage(obj.lower, obj.upper, self.grid, coverage)
            decisions.append(dict(object_id=obj.object_id, deleted=delete,
                                  intersects_coverage=delete, reason="xy_grid_intersection"))
            if delete:
                removed.append(obj.object_id)
            else:
                retained.append(obj)
        # Map Z=0 is the calibrated floor; use top height, not box thickness.
        ground = [obs.detection_index for obs in observations
                  if obs.upper[2] <= self.min_object_height_m]
        accepted = [obs for obs in observations
                    if obs.upper[2] > self.min_object_height_m
                    and intersects_coverage(obs.lower, obs.upper, self.grid, coverage)]
        skipped = [obs.detection_index for obs in observations if obs not in accepted]
        revision = self.revision + 1
        added = tuple(ObjectTrack(
            f"observation-{revision}-{obs.detection_index}", obs.lower, obs.upper,
            "confirmed", frame.timestamp_ns, frame.timestamp_ns, 1, (frame.frame_id,))
            for obs in accepted)
        # Publish atomically after every geometry/validation step succeeded.
        self.coverage = coverage
        self.objects = tuple(retained) + added
        self.revision = revision
        self.last_timestamp_ns, self.last_frame_id = frame.timestamp_ns, frame.frame_id
        self.last_update = dict(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns,
                                removed_ids=removed, added_ids=[o.object_id for o in added],
                                decisions=decisions, model_version=result.model_version,
                                predictions=len(observations), skipped_detection_indices=skipped,
                                ground_filtered_indices=ground,
                                coverage_cells=int(coverage.sum()))
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
                    mapping=dict(schema_version=2, policy="xy_depth_replacement",
                                 grid_metadata={k:v for k,v in asdict(snap.grid).items()
                                                if k not in ("rows", "origin", "resolution")},
                                 revision=self.revision, last_timestamp_ns=self.last_timestamp_ns,
                                 last_frame_id=self.last_frame_id,
                                 options=dict(self.options, min_object_height_m=self.min_object_height_m),
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
        if meta["schema_version"] != 2 or meta["policy"] != "xy_depth_replacement":
            raise ValueError("Unsupported map state; start a fresh map for XY replacement")
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


def depth_coverage(frame, grid, *, pixel_stride=4, min_depth_m=.15, max_depth_m=8.):
    """XY union of sampled depth-truncated rays: the actual replacement mask.

    No convex hull, filling invalid depth, or modification of SLAM occupancy.
    Heights collapse deliberately. Both deletion and insertion use this mask.
    """
    validate_grid(grid)
    validate_frame(frame)
    if (frame.map_id, frame.map_revision) != (grid.map_id, grid.map_revision):
        raise ValueError("Frame/grid coordinate mismatch")
    if isinstance(pixel_stride, bool) or not isinstance(pixel_stride, int) or pixel_stride < 1:
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
        lengths = end*np.linalg.norm(d[:, :2], axis=1)
        steps = max(2, int(np.ceil(lengths.max()/(grid.resolution*.5)))+1)
        t = end[:, None]*np.linspace(0, 1, steps)
        xy = T[:2, 3]+d[:, None, :2]*t[:, :, None]
        col = np.floor((xy[:, :, 0]-grid.origin[0])/grid.resolution).astype(int)
        row = height-1-np.floor((xy[:, :, 1]-grid.origin[1])/grid.resolution).astype(int)
        inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        mask[row[inside], col[inside]] = True
    return mask
