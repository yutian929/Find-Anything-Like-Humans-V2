"""Analytic geometry, independent dense checks, and weighted greedy contracts."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_candidates import CandidateResult, Station, generate_candidates, object_heatmaps
from observation_model import PointCamera
from anchor_selector import select_anchors_greedy as select_anchors
from anchor_export import viewer_data
from main import plan
from observation_model import BoxObserver, ObservationSettings
from observation_region import camera_poses, radial_envelope, radial_weight
from test_anchor_common import config
from test_observation_model import box, camera_config


class WeightedRegionTests(unittest.TestCase):
    def test_radial_endpoints_midpoint_and_collapsed_band(self):
        np.testing.assert_allclose(radial_weight(np.array([1, 2, 3]), 1, 3), [1, .75, .5])
        self.assertEqual(radial_weight(2, 2, 2), 1)

    def test_pitch_shrinks_ground_target_region_with_analytic_bounds(self):
        bounds = []
        for pitch in (35, 55):
            cfg = config()
            cfg['mount'].update(position_in_base_m=[0, 0, 1], pitch_deg=pitch)
            cfg['observation_quality']['max_distance_m'] = 20
            camera = PointCamera(cfg)
            origins, rotation = camera_poses(camera, [[-1, 0]], [0])
            _, near, far, valid = radial_envelope(camera, [0, 0, 0], [[0, 0, 0]], origins, rotation)
            p = math.radians(pitch)
            self.assertAlmostEqual(near[0], 1/math.tan(p+math.atan(.51)), places=9)
            self.assertAlmostEqual(far[0], 1/math.tan(p-math.atan(.5)), places=9)
            self.assertTrue(valid[0])
            bounds.append((near[0], far[0]))
        self.assertLess(bounds[1][0], bounds[0][0])
        self.assertLess(bounds[1][1], bounds[0][1])

    def test_height_distance_disk_and_impossible_height(self):
        cfg = config(); cfg['mount']['position_in_base_m'] = [0, 0, .5]
        camera = PointCamera(cfg)
        origins, rotation = camera_poses(camera, [[0, 0]], [0])
        _, near, far, valid = radial_envelope(camera, [2, 0, 1], [[2, 0, 1]], origins, rotation)
        self.assertAlmostEqual(far[0], math.sqrt(9-.25))
        # The lower distance bound is zero, but the vertical FOV creates an inner bound.
        self.assertAlmostEqual(near[0], 1.0)
        self.assertTrue(valid[0])
        self.assertFalse(radial_envelope(camera, [2, 0, 4], [[2, 0, 4]], origins, rotation)[3][0])

    def test_offset_changes_base_region_but_not_same_optical_pose(self):
        cfg = camera_config(); target = box('a', [2.9, -.5, .5], [3.1, .5, 1.5])
        c0 = PointCamera(cfg); o0 = BoxObserver(c0, [target])
        cfg2 = copy.deepcopy(cfg); cfg2['mount']['position_in_base_m'] = [.7, -.2, 1]
        o1 = BoxObserver(PointCamera(cfg2), [target])
        a = o0.evaluate(0, [[0, 0]], [0])[0]
        b = o1.evaluate(0, [[-.7, .2]], [0])[0]
        for field in ('distance_weight', 'observation_score', 'horizontal_distance_m', 'radial_near_m', 'radial_far_m'):
            self.assertAlmostEqual(a[field], b[field])
        self.assertNotEqual(a['center_distance_m'], o1.evaluate(0, [[0, 0]], [0])[0]['center_distance_m'])

    def test_region_mask_matches_independent_corner_projection(self):
        cfg = camera_config(); cfg['mount'].update(position_in_base_m=[.23, -.17, 1.2],
                                                  roll_deg=12, pitch_deg=27, yaw_deg=-18)
        camera = PointCamera(cfg)
        obj = box('target', [2, -1, .2], [3, 1, 1.8])
        observer = BoxObserver(camera, [obj])
        rng = np.random.default_rng(73)
        positions = rng.uniform(-5, 5, (500, 2)); yaws = rng.uniform(-math.pi, math.pi, 500)
        _, _, _, actual = observer.envelope(0, positions, yaws)
        corners = observer.lower[0]+observer.corners*(observer.upper[0]-observer.lower[0])
        projections = [camera.project(positions, yaws, corner) for corner in corners]
        u = np.array([p[0] for p in projections]); v = np.array([p[1] for p in projections])
        z = np.array([p[2] for p in projections])
        distance = camera.project(positions, yaws, obj['position'])[3]
        expected = ((z.min(axis=0)>1e-6) & (distance>=camera.min_distance) & (distance<=camera.max_distance)
                    & (u.max(axis=0)>0) & (u.min(axis=0)<camera.width)
                    & (v.max(axis=0)>0) & (v.min(axis=0)<camera.height))
        np.testing.assert_array_equal(actual, expected)
        # This explicitly retains some boxes whose center is outside the image.
        self.assertTrue(np.any(expected & ~camera.visible(positions, yaws, obj['position'])))

    def test_half_occlusion_at_inner_and_outer_score_boundaries(self):
        target = box('target', [2.9, -.5, .5], [3.1, .5, 1.5])
        blocker = box('half', [1, 0, 0], [2, 1, 2])
        cfg = camera_config();cfg['observation_quality'].update(min_distance_m=3, max_distance_m=5)
        observer = BoxObserver(PointCamera(cfg), [target, blocker])
        near = observer.evaluate(0, [[0, 0]], [0])[0]
        self.assertAlmostEqual(near['distance_weight'], 1)
        self.assertAlmostEqual(near['visible_fraction'], .5)
        self.assertAlmostEqual(near['observation_score'], .5)
        self.assertTrue(near['qualified'])  # The obsolete .6 visibility gate must not apply.
        far = observer.evaluate(0, [[-2, 0]], [0])[0]
        self.assertAlmostEqual(far['distance_weight'], .5)
        self.assertLess(far['observation_score'], .5)
        self.assertFalse(far['qualified'])
        clear = BoxObserver(PointCamera(cfg), [target]).evaluate(0, [[-2, 0]], [0])[0]
        self.assertAlmostEqual(clear['observation_score'], .5)
        self.assertTrue(clear['qualified'])
        with self.assertRaisesRegex(ValueError, 'min_visible_fraction'):
            ObservationSettings.from_config(camera_config(min_visible_fraction=.6))

    def test_framing_gate_uses_clipped_polygon_not_sample_count(self):
        target = box('target', [3, -.5, .5], [4, .5, 1.5])
        cfg = camera_config(); cfg['camera'].update(width=100, cx=90.5)
        # Front face spans u=[73.8333, 107.1667], so exactly 78.5% is in frame.
        for n in (12, 36, 96):
            row = BoxObserver(PointCamera(cfg), [target],
                              ObservationSettings(sample_grid=n, refinement_grid=n)).evaluate(0, [[0, 0]], [0])[0]
            self.assertAlmostEqual(row['in_frame_fraction'], .785, places=10)
            self.assertFalse(row['qualified'])
            self.assertEqual(row['reason'], 'in_frame_fraction')

    def test_weighted_votes_can_prefer_fewer_objects_and_update_dynamically(self):
        stations = [Station(i, 1, (0, i), (i, 0)) for i in range(3)]
        c = CandidateResult([0, 1, 2], [0, 1],
            {0: {0: {'a', 'b', 'c'}}, 1: {0: {'a', 'b'}, 1: {'a'}}, 2: {0: {'b', 'c'}}}, {},
            {0: {0: {'a': .5, 'b': .5, 'c': .5}},
             1: {0: {'a': .9, 'b': .9}, 1: {'a': .8}}, 2: {0: {'b': .9, 'c': .8}}})
        result = select_anchors(c, stations, ['a', 'b', 'c'])
        self.assertEqual([row['station_index'] for row in result['vote_trace']], [1, 2])
        self.assertEqual([row['weighted_score'] for row in result['vote_trace']], [1.8, .8])
        self.assertEqual(result['vote_trace'][1]['new_object_ids'], ['c'])
        heat = object_heatmaps(c, ['a', 'b', 'c', 'missing'])
        self.assertEqual(heat['a'][1], [1, .9, 0, 900000])
        self.assertEqual(heat['missing'], [])

    def test_heap_and_exported_heatmap_match_independent_weighted_greedy(self):
        rng = np.random.default_rng(27)
        stations = [Station(i, 1, (0, i), (i, 0)) for i in range(9)]
        targets = ['a', 'b', 'c', 'd', 'e']
        for _ in range(20):
            supports, scores = {}, {}
            for i in range(9):
                for yaw in range(3):
                    ids = {obj for obj in targets if rng.random() < .4}
                    if ids:
                        supports.setdefault(i, {})[yaw] = ids
                        scores.setdefault(i, {})[yaw] = {obj: float(rng.uniform(.5, 1)) for obj in sorted(ids)}
            c = CandidateResult(list(range(9)), [0, 1, 2], supports, {}, scores)
            matrix = {i: {} for i in range(9)}
            for obj, entries in object_heatmaps(c, targets).items():
                for i, score, yaw, units in entries:
                    self.assertEqual(score, max(scores[i][y].get(obj, 0) for y in scores[i]))
                    matrix[i][obj] = units
            remaining = set(targets); trace = []
            while remaining:
                i = min(matrix, key=lambda i: (-sum(s for o, s in matrix[i].items() if o in remaining),
                                               -len(remaining & matrix[i].keys()), i))
                new = remaining & matrix[i].keys()
                if not new: break
                trace.append(i); remaining -= new
            result = select_anchors(c, stations, targets)
            self.assertEqual([row['station_index'] for row in result['vote_trace']], trace)
            self.assertEqual(set(result['uncovered_object_ids']), remaining)

    def test_exported_heatmaps_and_final_actions_use_identical_scores(self):
        scene = {'grid': {'rows': ['......']*6, 'origin': [0, -3], 'resolution': 1},
                 'objects': [box('target', [2.9, -.5, .5], [3.1, .5, 1.5]),
                             box('far', [99, 99, 0], [100, 100, 1])]}
        robot = {'robot': {'footprint_width_m': .2, 'footprint_length_m': .2, 'safety_margin_m': 0}}
        camera = camera_config()
        result, regions, stations, candidates = plan(scene, robot, camera)
        data = viewer_data(scene, camera, result, regions, stations, candidates)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(data['methods'][0]['object_heatmaps']['far'], [])
        for a in result['anchors']:
            for action in a['observations']:
                for obj in action['object_ids']:
                    metric = action['object_metrics'][obj]
                    self.assertGreaterEqual(metric['observation_score']+1e-9, .5)
                    self.assertEqual(action['score_by_object'][obj], metric['observation_score'])
        self.assertGreater(candidates.stats['region_rejected_pose_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
