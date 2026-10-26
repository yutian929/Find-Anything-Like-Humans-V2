"""Native backend geometry checks, including its documented precision boundary."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observation_model import BoxObserver, PointCamera, ray_box_entry
from observation_model import Open3DRaycaster
from test_observation_model import box, camera_config


@unittest.skipUnless(importlib.util.find_spec('open3d'), 'optional Open3D is not installed')
class NativeRaycastingTests(unittest.TestCase):
    def test_clear_half_occluded_and_behind_boxes(self):
        target = box('target', [3, -.5, .5], [4, .5, 1.5])
        for other in ([], [box('half', [1, 0, 0], [2, 1, 2])],
                      [box('behind', [5, -1, 0], [6, 1, 2])]):
            camera = PointCamera(camera_config())
            objects = [target, *other]
            reference = BoxObserver(camera, objects).evaluate(0, [[0, 0]], [0])
            native = BoxObserver(camera, objects, ray_backend='open3d', ray_threads=1)
            self.assertEqual(native.evaluate(0, [[0, 0]], [0]), reference)

    def test_float32_can_include_ray_just_outside_target(self):
        low, high = np.array([[3, -.5, .5]]), np.array([[4, .5, 1.5]])
        caster = Open3DRaycaster(low, high, nthreads=1)
        origin = np.array([[[0., 0., 1.]]])
        directions = np.array([[[1, (.5+1e-8)/3, 0]]])
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        self.assertFalse(np.isfinite(ray_box_entry(origin, directions, low[0], high[0]))[0, 0])
        native, _ = caster.trace(0, origin, directions, np.ones((1, 1), bool))
        self.assertTrue(native[0, 0])
        # This is why selecting native candidates never switches the final verifier.

    def test_origin_inside_box_is_rejected_before_native_sampling(self):
        target = box('target', [0, -.5, .5], [4, .5, 1.5])
        observer = BoxObserver(PointCamera(camera_config()), [target], ray_backend='open3d')
        row = observer.evaluate(0, [[1, 0]], [0])[0]
        self.assertEqual(row['reason'], 'camera_inside_geometry')
        self.assertIsNone(observer._raycaster)


if __name__ == '__main__':
    unittest.main()
