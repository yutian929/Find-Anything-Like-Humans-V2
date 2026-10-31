import contextlib
import io
import itertools
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from falhv2.robot_adapters.base import CameraIntrinsics, RGBDFrame, GridSnapshot
from falhv2.model_clients.cubify_anything import make_result
from falhv2.map_maintainers.depth_replace import DepthReplacementMap, visibility, depth_coverage
from falhv2.map_maintainers.cubify import cuboids_to_map_aabbs


def frame(stamp=1, depth=6.):
    T = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                  [0., -1., 0., 1.], [0., 0., 0., 1.]])
    return RGBDFrame(str(stamp), stamp, "unix", "room", "session-1", "optical",
                     CameraIntrinsics(64, 48, 40., 40., 31.5, 23.5),
                     np.zeros((48, 64, 3), np.uint8),
                     np.full((48, 64), depth, np.float32), T)


def grid():
    return GridSnapshot("room", "session-1", 1, "unix", .1, (-1., -4.),
                        tuple(["."*100]*80))


def prediction(f, boxes=()):
    points = [np.array(list(itertools.product(*zip(lo, hi))), float) for lo, hi in boxes]
    return make_result(f, [.9]*len(points), points, "test-weights", 0.)


FRONT = ((-.4, -.4, 2.), (.4, .4, 2.8))
BEHIND = ((-.4, -.4, 4.), (.4, .4, 4.8))


class DepthReplacementTests(unittest.TestCase):
    def build(self, boxes=(FRONT, BEHIND)):
        b = DepthReplacementMap()
        f = frame()
        b.set_grid(grid())
        b.integrate(f, prediction(f, boxes))
        return b

    def test_wall_keeps_hidden_box_replaces_visible_without_association(self):
        b = self.build()
        old = b.snapshot()
        f = frame(2, depth=3.)
        update = b.integrate(f, prediction(f, (FRONT,)))
        self.assertEqual(update["removed_ids"], [old.objects[0].object_id])
        self.assertEqual([o.object_id for o in b.objects],
                         [old.objects[1].object_id, "observation-2-0"])
        self.assertEqual(len(old.objects), 2)  # old snapshot remains unchanged
        self.assertEqual(b.objects[1].observation_count, 1)

    def test_successful_empty_detection_still_removes_visible_old_boxes(self):
        b = self.build()
        f = frame(2)
        b.integrate(f, prediction(f))
        self.assertEqual(b.objects, ())

    def test_invalid_depth_holes_do_not_delete(self):
        b = self.build((FRONT,))
        f = frame(2)
        f.depth_m[:] = np.nan
        f.depth_m[0, 0] = 6.  # valid frame, but no usable depth on old box
        b.integrate(f, prediction(f))
        self.assertEqual(len(b.objects), 1)
        self.assertEqual(b.last_update["decisions"][0]["visible_fraction"], 0.)

    def test_entire_invalid_frame_or_bad_result_is_atomic(self):
        for invalid in ("depth", "result", "coordinates", "clock", "replay"):
            b = self.build()
            previous = b.to_scene()
            f = frame(2)
            r = prediction(f)
            if invalid == "depth":
                f.depth_m[:] = 0
            elif invalid == "result":
                r = replace(r, frame_id="wrong")
            elif invalid == "coordinates":
                f = replace(f, map_revision="different")
            elif invalid == "clock":
                f = replace(f, clock_domain="other")
            else:
                f = frame(1)
                r = prediction(f)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                b.integrate(f, r)
            self.assertEqual(b.to_scene(), previous)

    def test_partial_occlusion_coverage_and_tolerance(self):
        b = self.build((FRONT,))
        f = frame(2)
        obj = b.objects[0]
        f.depth_m[:, :32] = 1.
        fraction = visibility(f, obj.lower, obj.upper)["visible_fraction"]
        self.assertAlmostEqual(fraction, .5)
        b.visible_fraction = .51
        b.integrate(f, prediction(f))
        self.assertEqual(len(b.objects), 1)
        f = frame(3, depth=1.95)
        self.assertEqual(visibility(f, obj.lower, obj.upper, depth_tolerance_m=0.)["visible_fraction"], 0.)
        self.assertGreater(visibility(f, obj.lower, obj.upper, depth_tolerance_m=.08)["visible_fraction"], .9)

    def test_outside_frame_near_plane_and_depth_range_retain(self):
        b = self.build()
        for obj in b.objects:
            f = frame(2)
            T = f.T_map_camera.copy()
            T[:3, :2] *= -1  # rotate camera around optical axis; move camera far away
            T[:3, 3] = [0, 100, 1]
            self.assertEqual(visibility(replace(f, T_map_camera=T), obj.lower, obj.upper)["visible_fraction"], 0.)
            self.assertEqual(visibility(frame(2, depth=20), obj.lower, obj.upper)["visible_fraction"], 0.)
        self.assertEqual(visibility(frame(), (-1, -1, 0), (1, 1, 2))["visible_fraction"], 0.)

    def test_offscreen_projection_counts_in_denominator(self):
        f = frame()
        # Optical x ~ 3 at z 2 is mostly outside this image's horizontal FOV.
        obs = cuboids_to_map_aabbs(f, prediction(f, (((1.4, -.3, 2.), (3., .3, 3.)),)))[0]
        check = visibility(f, obs.lower, obs.upper, sample_grid=64)
        self.assertGreater(check["visible_fraction"], 0.)
        self.assertLess(check["visible_fraction"], .5)

    def test_world_rigid_transform_does_not_change_visibility(self):
        b = self.build((FRONT,))
        obj = b.objects[0]
        f = frame(2)
        # Map-axis-preserving quarter turn plus translation; exercises ray transforms.
        R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        G = np.eye(4)
        G[:3, :3], G[:3, 3] = R, [8., -3., 2.]
        pts = np.array(list(itertools.product(*zip(obj.lower, obj.upper)))) @ R.T + G[:3, 3]
        changed = replace(f, T_map_camera=G @ f.T_map_camera)
        self.assertEqual(visibility(f, obj.lower, obj.upper),
                         visibility(changed, pts.min(0), pts.max(0)))

    def test_display_projection_stops_at_wall_and_preserves_grid(self):
        f = frame(depth=3.)
        g = grid()
        mask = depth_coverage(f, g, pixel_stride=4)
        xs = g.origin[0] + (np.arange(mask.shape[1])+.5)*g.resolution
        self.assertTrue(mask[:, xs < 3.].any())
        self.assertFalse(mask[:, xs > 3.15].any())
        self.assertEqual(g.rows, grid().rows)
        # A depth hole contributes no extrapolated triangle to the visualization.
        f.depth_m[:] = np.nan
        f.depth_m[24, 32] = 2.
        sparse = depth_coverage(f, g, pixel_stride=4)
        self.assertLess(sparse.sum(), mask.sum()/10)

    def test_resume_scene_and_coordinate_rejection(self):
        b = self.build()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"map.json"
            b.save(p)
            restored = DepthReplacementMap.load(p)
            self.assertEqual(restored.to_scene(), b.to_scene())
            f = frame(2, depth=3.)
            restored.integrate(f, prediction(f))
            self.assertEqual(len(restored.objects), 1)
            with self.assertRaises(ValueError):
                restored.set_grid(replace(grid(), map_revision="session-2"))
            changed_grid = replace(grid(), origin=(-2., -4.), content_version="new-grid")
            restored.set_grid(changed_grid)
            self.assertEqual(restored.grid, changed_grid)
            p.write_text(p.read_text().replace('"observation-1-0"', '"tampered"', 1))
            with self.assertRaises(ValueError):
                DepthReplacementMap.load(p)

    def test_cli_two_frames_save_resume_and_cleanup(self):
        from falhv2 import main
        from falhv2.tools import build_map
        from falhv2.anchor_planners.observation_model import validate_objects
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"map.json"
            cfg = Path(folder)/"robot.py"
            cfg.write_text("ROBOT_SOURCE = {'map_id': 'room', 'map_revision': 'session-1'}\n")
            robot = MagicMock()
            robot.__enter__.return_value = robot
            robot.grid_snapshot.return_value = grid()
            f1, f2 = frame(), frame(2, depth=3.)
            robot.capture.side_effect = [f1, f2]
            client = MagicMock()
            client.detect.side_effect = [prediction(f1, (FRONT, BEHIND)), prediction(f2, (FRONT,))]
            with patch.object(build_map, "AnchorMapRobot", return_value=robot), patch.object(
                    build_map, "build_cubify_anything", return_value=client), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(["--config", str(cfg), "--task", "build-map",
                                            "--frames", "2", "--output", str(p)]), 0)
            client.close.assert_called_once()
            robot.__exit__.assert_called_once()
            data = json.loads(p.read_text())
            self.assertEqual(len(validate_objects(data["objects"])), 2)
            self.assertEqual(len(data["mapping"]["last_update"]["removed_ids"]), 1)
            self.assertTrue(p.with_suffix(".svg").exists())
            # Same CLI resumes persistent history and uses only explicit configuration.
            robot.capture.side_effect = [frame(3)]
            client.detect.side_effect = [prediction(frame(3))]
            with patch.object(build_map, "AnchorMapRobot", return_value=robot), patch.object(
                    build_map, "build_cubify_anything", return_value=client), contextlib.redirect_stdout(io.StringIO()):
                main.main(["--config", str(cfg), "--task", "build-map", "--resume", str(p)])
            self.assertEqual(json.loads(p.read_text())["objects"], [])

    def test_same_xy_different_heights_have_different_visibility(self):
        f = frame(2)
        f.depth_m[24:, :] = 1.
        upper = visibility(f, (2., -.2, 1.2), (3., .2, 1.6))
        lower = visibility(f, (2., -.2, .4), (3., .2, .8))
        self.assertEqual(upper["visible_fraction"], 1.)
        self.assertEqual(lower["visible_fraction"], 0.)

    def test_off_axis_depth_is_optical_z_not_range(self):
        f = frame(2, depth=3.)
        obs = cuboids_to_map_aabbs(f, prediction(f, (((1.3, -.2, 2.9), (1.6, .2, 2.98)),)))[0]
        self.assertGreater(visibility(f, obs.lower, obs.upper, depth_tolerance_m=0.)["visible_fraction"], .9)

    def test_parameters_are_validated(self):
        for options in (dict(visible_fraction=0), dict(sample_grid=True),
                        dict(depth_tolerance_m=-1), dict(min_depth_m=9, max_depth_m=8),
                        dict(visible_fraction=float("nan"))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DepthReplacementMap(**options)


if __name__ == "__main__":
    unittest.main()
