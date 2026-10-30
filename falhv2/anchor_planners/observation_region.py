"""Mount-aware radial envelopes for inverse observation regions.

For each camera pose, fix its rotation and camera-to-target XY bearing, then
solve the allowed horizontal radii analytically. The bounds intersect center
distance with positive corner depths and AABB projected-bounds/frustum overlap.
They are conservative geometric envelopes, NOT guarantees of 80% framing,
pixel area or absence of occlusion. Those gates are evaluated by BoxObserver.
"""
from __future__ import annotations

import numpy as np


def camera_poses(camera, positions, yaws):
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    yaws = np.asarray(yaws, dtype=float).reshape(-1)
    if len(positions) != len(yaws):
        raise ValueError('One yaw per base position required')
    c, s = np.cos(yaws), np.sin(yaws)
    rotation = np.zeros((len(yaws), 3, 3))
    rotation[:, 0, 0] = rotation[:, 1, 1] = c
    rotation[:, 1, 0], rotation[:, 0, 1], rotation[:, 2, 2] = s, -s, 1
    origins = np.column_stack((positions, np.zeros(len(yaws)))) + rotation @ camera.offset
    return origins, rotation @ camera.base_from_optical


def radial_envelope(camera, center, corners, origins, rotation):
    """Return (radius, near, far, admissible) arrays for actual optical centers.

    Every corner in optical coordinates is a*rho+b on a fixed bearing. Testing
    the maximum signed corner distance to each frustum side gives linear
    necessary conditions for projected bounding-box overlap (not center FOV).
    Thus partially framed boxes with off-screen centers are not dropped.
    """
    center, corners = np.asarray(center, dtype=float), np.asarray(corners, dtype=float)
    delta = center[:2]-origins[:, :2]
    radius = np.linalg.norm(delta, axis=1)
    bearing = np.zeros_like(origins)
    bearing[:, :2] = delta / np.maximum(radius[:, None], 1e-12)
    # At rho=0 the bearing is undefined; use the camera's horizontal forward.
    zero = radius < 1e-12
    fallback = rotation[zero, :2, 2]
    norm = np.linalg.norm(fallback, axis=1)
    fallback = fallback / np.maximum(norm[:, None], 1e-12)
    fallback[norm < 1e-12] = [1, 0]
    bearing[zero, :2] = fallback
    slope = np.matmul(bearing[:, None, :], rotation)[:, 0, :]
    relative = np.broadcast_to(corners[None, :, :], (len(origins), len(corners), 3)).copy()
    relative[:, :, :2] -= center[:2]
    relative[:, :, 2] -= origins[:, None, 2]
    intercept = np.matmul(relative, rotation)
    height = center[2]-origins[:, 2]
    near = np.sqrt(np.maximum(0, camera.min_distance**2-height**2))
    far = np.sqrt(np.maximum(0, camera.max_distance**2-height**2))
    valid = np.abs(height) <= camera.max_distance+1e-9

    def constrain(a, b):
        # a*rho+b >= 0, with a small outward tolerance for conservative screening.
        nonlocal near, far, valid
        positive, negative = a > 1e-12, a < -1e-12
        bound = -b / np.where(positive | negative, a, 1)
        near = np.where(positive, np.maximum(near, bound), near)
        far = np.where(negative, np.minimum(far, bound), far)
        valid &= positive | negative | (b >= -1e-9)

    constrain(slope[:, 2], intercept[:, :, 2].min(axis=1)-1e-6)
    planes = np.array([[camera.fx, 0, camera.cx],
                       [-camera.fx, 0, camera.width-camera.cx],
                       [0, camera.fy, camera.cy],
                       [0, -camera.fy, camera.height-camera.cy]])
    for plane in planes:
        constrain(slope @ plane, (intercept @ plane).max(axis=1))
    valid &= (near <= far+1e-9) & (radius >= near-1e-9) & (radius <= far+1e-9)
    return radius, near, far, valid


def radial_weight(radius, near, far, inner_weight=None, outer_weight=None):
    """Linear endpoint weights; a collapsed band gets the inner weight.

    Only call for a valid envelope. Clipping absorbs numerical boundary slack;
    it never makes an out-of-region pose eligible.
    """
    inner_weight = 1.0 if inner_weight is None else inner_weight
    outer_weight = .5 if outer_weight is None else outer_weight
    span = np.asarray(far)-near
    fraction = np.divide(np.asarray(radius)-near, span,
                         out=np.zeros_like(np.asarray(radius), dtype=float), where=span > 1e-9)
    return inner_weight-(inner_weight-outer_weight)*np.clip(fraction, 0, 1)
