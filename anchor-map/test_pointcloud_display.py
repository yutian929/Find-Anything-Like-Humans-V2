"""地图系点云的前端数据契约。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import perception as P
from server import create_app


class PointcloudDisplayTest(unittest.TestCase):
    def test_world_coordinates_and_colors_survive_display_conversion(self):
        camera_points = np.array([[0, 0, 1, 255, 10, 20],
                                  [1, 0, 1, 30, 255, 40],
                                  [0, 1, 1, 50, 60, 255]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [2, -3, 0.5]
        world, _ = P.pointcloud_cam_to_world(camera_points, pose)
        shown = P.pointcloud_for_display(world)
        self.assertEqual(shown["points"][0], [2.0, -3.0, 1.5, 255.0, 10.0, 20.0])
        self.assertEqual(shown["points"][1][:3], [3.0, -3.0, 1.5])
        self.assertTrue(shown["colored"])

    def test_downsample_and_filter_invalid_points(self):
        points = np.column_stack([np.arange(10), np.zeros(10), np.ones(10)])
        points[4, 0] = np.nan
        shown = P.pointcloud_for_display(points, max_points=3)
        self.assertEqual(len(shown["points"]), 3)
        self.assertEqual(shown["points"][0][0], 0)
        self.assertEqual(shown["points"][-1][0], 9)

    def test_saved_query_pointcloud_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bottle" / "pointcloud.npy"
            path.parent.mkdir()
            np.save(path, np.array([[2, 3, 1, 255, 0, 0],
                                    [2.1, 3, 1, 0, 255, 0],
                                    [2, 3.1, 1, 0, 0, 255]], dtype=np.float32))
            app = create_app(SimpleNamespace(results_dir=Path(tmp)))
            route = next(route for route in app.routes
                         if route.path == "/api/results/{object_name}/pointcloud")
            response = route.endpoint("bottle")
            self.assertTrue(response["success"])
            self.assertTrue(response["colored"])
            self.assertEqual(response["points"][0][:3], [2.0, 3.0, 1.0])


if __name__ == "__main__":
    unittest.main()
