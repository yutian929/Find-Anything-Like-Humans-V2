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
from falhv2.map_maintainers.depth_replace import DepthReplacementMap, depth_coverage
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
    def test_ground_filter_uses_world_top_and_preserves_depth_coverage(self):
        f = frame()
        b = DepthReplacementMap(min_object_height_m=.125)
        b.set_grid(grid())
        # Optical Y points down: world Z = 1 - optical Y.
        floor = ((-.4, .875, 2.), (.4, 1.05, 2.8))
        standing = ((-.4, -.5, 2.), (.4, 1., 2.8))
        suspended_thin = ((-.4, .4, 2.), (.4, .45, 2.8))
        update = b.integrate(f, prediction(f, (floor, standing, suspended_thin)))
        self.assertEqual(update['ground_filtered_indices'], [0])
        self.assertEqual(update['skipped_detection_indices'], [0])
        self.assertEqual(len(b.objects), 2)
        np.testing.assert_array_equal(b.coverage, depth_coverage(f, grid()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'map.json'
            b.save(path)
            restored = DepthReplacementMap.load(path)
            self.assertEqual(restored.min_object_height_m, .125)
        f = frame(2)
        update = b.integrate(f, prediction(f, (floor,)))
        self.assertEqual(len(update['removed_ids']), 2)
        self.assertEqual(b.objects, ())
        for invalid in (-1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                DepthReplacementMap(min_object_height_m=invalid)

    def build(self):
        b = DepthReplacementMap()
        f = frame()
        b.set_grid(grid())
        b.integrate(f, prediction(f, (FRONT, BEHIND)))
        return b

    def test_wall_retains_old_box_and_excludes_new_box_behind_wall(self):
        b = self.build()
        previous = b.snapshot()
        f = frame(2, depth=3.)
        update = b.integrate(f, prediction(f, (FRONT, BEHIND)))
        self.assertEqual(update["removed_ids"], [previous.objects[0].object_id])
        self.assertEqual(update["skipped_detection_indices"], [1])
        self.assertEqual([o.object_id for o in b.objects],
                         [previous.objects[1].object_id, "observation-2-0"])
        self.assertEqual(len(previous.objects), 2)

    def test_stationary_frames_and_small_prediction_jitter_do_not_accumulate(self):
        b = self.build()
        for stamp in range(2, 32):
            f = frame(stamp)
            shift = .02 if stamp % 2 else -.02
            changed = ((FRONT[0][0]+shift, *FRONT[0][1:]),
                       (FRONT[1][0]+shift, *FRONT[1][1:]))
            b.integrate(f, prediction(f, (changed, BEHIND)))
            self.assertEqual(len(b.objects), 2)
            self.assertEqual(len(b.last_update["removed_ids"]), 2)
        self.assertEqual(b.revision, 31)

    def test_successful_empty_detection_removes_intersecting_boxes(self):
        b = self.build()
        f = frame(2)
        b.integrate(f, prediction(f))
        self.assertEqual(b.objects, ())

    def test_single_shared_cell_and_height_collapse(self):
        from falhv2.map_maintainers.depth_replace import intersects_coverage
        g = replace(grid(), resolution=1., origin=(0., 0.), rows=("....",)*4)
        mask = np.zeros((4,4), bool)
        mask[2,1] = True  # x in [1,2), y in [1,2)
        self.assertTrue(intersects_coverage((1.99,1.99,-5), (3,3,-4), g, mask))
        self.assertTrue(intersects_coverage((1.99,1.99,8), (3,3,9), g, mask))
        self.assertFalse(intersects_coverage((2.01,2.01,0), (3,3,1), g, mask))
        self.assertFalse(intersects_coverage((-3,-3,0), (-1,-1,1), g, mask))

    def test_actual_mask_stops_at_depth_and_does_not_modify_occupancy(self):
        f, g = frame(depth=3.), grid()
        mask = depth_coverage(f, g, pixel_stride=4)
        xs = g.origin[0] + (np.arange(mask.shape[1])+.5)*g.resolution
        self.assertTrue(mask[:, xs < 3.].any())
        self.assertFalse(mask[:, xs > 3.15].any())
        self.assertEqual(g.rows, grid().rows)
        f.depth_m[:] = np.nan
        f.depth_m[24,32] = 2.
        sparse = depth_coverage(f, g, pixel_stride=4)
        self.assertLess(sparse.sum(), mask.sum()/10)

    def test_no_valid_sampled_rays_produce_no_additions_or_deletions(self):
        b = self.build()
        f = frame(2)
        f.depth_m[:] = np.nan
        f.depth_m[1,1] = 2.  # valid frame; not on configured sampling lattice
        before = b.objects
        update = b.integrate(f, prediction(f, (FRONT,)))
        self.assertEqual(b.objects, before)
        self.assertEqual(update["added_ids"], [])
        self.assertEqual(update["removed_ids"], [])
        self.assertEqual(update["skipped_detection_indices"], [0])
        self.assertFalse(b.coverage.any())

    def test_invalid_inputs_leave_state_unchanged(self):
        for invalid in ("depth", "result", "coordinates", "clock", "replay"):
            b = self.build()
            previous = b.to_scene()
            mask = b.coverage.copy()
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
            np.testing.assert_array_equal(b.coverage, mask)

    def test_scene_resume_and_old_policy_rejection(self):
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
            old = b.to_scene()
            old["mapping"].update(schema_version=1, policy="depth_replacement")
            p.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "fresh map"):
                DepthReplacementMap.load(p)

    def test_cli_uses_the_same_mask_and_planner_compatible_scene(self):
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
            client.detect.side_effect = [prediction(f1, (FRONT, BEHIND)), prediction(f2, (FRONT, BEHIND))]
            with patch.object(build_map, "AnchorMapRobot", return_value=robot), patch.object(
                    build_map, "build_cubify_anything", return_value=client), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(["--config", str(cfg), "--task", "build-map",
                                            "--frames", "2", "--output", str(p)]), 0)
            client.close.assert_called_once()
            robot.__exit__.assert_called_once()
            data = json.loads(p.read_text())
            self.assertEqual(len(validate_objects(data["objects"])), 2)
            self.assertEqual(data["mapping"]["last_update"]["skipped_detection_indices"], [1])
            self.assertTrue(p.with_suffix(".svg").exists())

    def test_parameters_are_validated(self):
        for options in (dict(pixel_stride=0), dict(pixel_stride=True),
                        dict(min_depth_m=9, max_depth_m=8), dict(max_depth_m=float("nan"))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DepthReplacementMap(**options)


if __name__ == "__main__":
    unittest.main()
