"""Fast observation paths preserve sampling, coverage and failure diagnostics."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import observation_model as model
from anchor_candidates import Station, generate_candidates
from test_observation_model import box, camera_config as base_camera_config


def camera_config(**quality):
    config = base_camera_config()
    config['observation_quality'].update(quality)
    return config


class ObservationAccelerationTests(unittest.TestCase):
    def setUp(self):
        self.target = box('target', [3, -.5, .5], [4, .5, 1.5])

    def test_refinement_reuses_projection_but_keeps_original_sample_density(self):
        cfg = camera_config(sample_grid=12, refinement_grid=36, min_in_frame_fraction=1.0)
        observer = model.BoxObserver(model.PointCamera(cfg), [self.target])
        with patch.object(model, 'projection_areas_batch', wraps=model.projection_areas_batch) as project:
            rows = observer.evaluate(0, [[0, 0], [0, .1]], [0, 0])
        self.assertEqual(project.call_count, 1)
        self.assertEqual([r['sample_grid'] for r in rows], [36, 36])
        self.assertTrue(all(r['qualified'] for r in rows))

    def test_geometric_rejections_skip_rays_without_fabricating_metrics(self):
        for cropped in (True, False):
            cfg = camera_config(min_visible_pixels=10 if cropped else 2000)
            if cropped:
                cfg['camera']['cx'] = 0
            observer = model.BoxObserver(model.PointCamera(cfg), [self.target])
            with patch.object(model, 'ray_box_entry', side_effect=AssertionError('unneeded ray')):
                fast = observer.evaluate(0, [[0, 0]], [0], candidate_only=True)[0]
            full = observer.evaluate(0, [[0, 0]], [0])[0]
            self.assertFalse(fast['qualified'])
            self.assertEqual(fast['reason'], full['reason'])
            self.assertEqual(fast['in_frame_fraction'], full['in_frame_fraction'])
            self.assertIsNone(fast['observation_score'])
            self.assertIsNone(fast['visible_pixels'])
            self.assertEqual(full['visible_fraction'], 1)

    def test_fast_and_full_agree_on_qualified_scores_with_occlusion_and_refinement(self):
        objects = [self.target, box('blocker', [1, 0, 0], [2, 1, 2])]
        cfg = camera_config(sample_grid=12, refinement_grid=36)
        observer = model.BoxObserver(model.PointCamera(cfg), objects)
        positions = np.array([[x, y] for x in [-1, 0, 2, 4] for y in np.linspace(-3, 3, 9)])
        for index in range(len(objects)):
            for yaw in (0, .3, np.pi):
                fast = observer.evaluate(index, positions, [yaw]*len(positions), candidate_only=True)
                full = observer.evaluate(index, positions, [yaw]*len(positions))
                self.assertEqual([r['qualified'] for r in fast], [r['qualified'] for r in full])
                self.assertEqual([r['reason'] for r in fast], [r['reason'] for r in full])
                for a, b in zip(fast, full):
                    if a['observation_score'] is not None:
                        self.assertEqual(a, b)

    def test_uncovered_target_gets_exact_best_failure_including_tied_pose_order(self):
        camera = model.PointCamera(camera_config(min_visible_pixels=1e8))
        # Duplicate coordinates and descending IDs exercise the original traversal tie.
        stations = [Station(90-i, 1, (0, i), p)
                    for i, p in enumerate([(0, 0), (0, 0), (0, 2), (1, -2)])]
        candidates = generate_candidates(stations, [self.target], camera, yaw_step_deg=30)
        observer = model.BoxObserver(camera, [self.target])
        checked = []
        for station in stations:
            for yaw in candidates.yaws:
                if observer.envelope(0, [station.position], [yaw])[3][0]:
                    metric = observer.evaluate(0, [station.position], [yaw])[0]
                    checked.append(dict(position=list(station.position), yaw_rad=yaw, metrics=metric))
        best = max(checked, key=lambda r: (r['metrics']['qualified'], r['metrics']['observation_score'],
                   r['metrics']['visible_pixels'], -r['metrics']['center_distance_m']))
        self.assertEqual(candidates.supports, {})
        self.assertEqual(candidates.diagnostics['target'], best)
        self.assertGreater(candidates.stats['diagnostic_recomputed_pose_pairs'], 0)

    def test_axiswise_slab_matches_independent_scalar_intersections(self):
        rng = np.random.default_rng(10425)
        origins = rng.uniform(-2, 2, (4, 1, 3))
        directions = rng.normal(size=(4, 101, 3))
        directions[:, :4, 0] = 0
        directions[:, 4:8, 1] = 1e-14
        low, high = np.array([-.7, -.4, -.2]), np.array([.5, .6, .9])
        expected = np.full((4, 101), np.inf)
        for batch in range(4):
            for ray in range(101):
                near, far, outside = -np.inf, np.inf, False
                for axis in range(3):
                    start, direction = origins[batch, 0, axis], directions[batch, ray, axis]
                    if abs(direction) < 1e-12:
                        outside |= start < low[axis] or start > high[axis]
                    else:
                        a, b = (low[axis]-start)/direction, (high[axis]-start)/direction
                        near, far = max(near, min(a, b)), min(far, max(a, b))
                if not outside and far >= max(near, 0):
                    expected[batch, ray] = max(near, 0)
        np.testing.assert_array_equal(model.ray_box_entry(origins, directions, low, high), expected)

    def test_array_batches_preserve_pose_order_and_individual_results(self):
        observer = model.BoxObserver(model.PointCamera(camera_config()),
            [self.target, box('blocker', [1, 0, 0], [2, 1, 2])])
        size = observer.batch_size*2+3
        positions = np.column_stack((np.zeros(size), np.linspace(-3, 3, size)))
        yaws = np.linspace(-.3, .3, size)
        arrays = observer.evaluate_arrays(0, positions, yaws, candidate_only=True)
        for i in range(len(positions)):
            single = observer.evaluate(0, positions[i:i+1], yaws[i:i+1], candidate_only=True)[0]
            self.assertEqual(arrays.record(i), single)
        empty = observer.evaluate_arrays(0, np.empty((0, 2)), [])
        self.assertEqual(empty.records(), [])
        self.assertEqual(empty.blockers.shape, (0, 2))

    def test_double_precision_geometry_preserves_framing_boundary(self):
        # Float32 rounds this rejected rectangle to an exactly 80% framing pass.
        below = np.array([[-20-2e-7, 100], [80-2e-7, 100],
                          [80-2e-7, 200], [-20-2e-7, 200]])
        exact = below + [2e-7, 0]
        areas = model.projection_areas_batch(np.stack([below, exact]), 1280, 720)
        np.testing.assert_allclose(areas[:, 0], [10000, 10000], rtol=0, atol=1e-10)
        self.assertLess(areas[0, 1]/areas[0, 0]+1e-9, .8)
        self.assertGreaterEqual(areas[1, 1]/areas[1, 0]+1e-9, .8)

    def test_geometry_handles_clipping_outside_and_degenerate_hulls(self):
        points = np.array([[[0, 0], [20, 0], [20, 20], [0, 20]],
                           [[-10, -10], [20, -10], [20, 20], [-10, 20]],
                           [[-20, -20], [-10, -20], [-10, -10], [-20, -10]],
                           [[1, 1], [2, 2], [3, 3], [4, 4]]], dtype=float)
        np.testing.assert_array_equal(model.projection_areas_batch(points, 10, 10),
                                      [[400, 100], [900, 100], [100, 0], [0, 0]])


if __name__ == '__main__':
    unittest.main()
