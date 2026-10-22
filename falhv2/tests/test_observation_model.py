import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_selector import PointCamera, Station, CandidateResult, generate_candidates, select_anchors, plan
from observation_model import BoxObserver, ObservationSettings, ray_box_entry, validate_objects
from test_anchor_common import config


def box(identifier, low, high):
    return {'id': identifier, 'position': ((np.array(low)+high)/2).tolist(),
            'aabb': {'min': low, 'max': high}}


def camera_config(**quality):
    cfg = config()
    cfg['mount']['position_in_base_m'] = [0, 0, 1]
    cfg['observation_quality'].update(max_distance_m=10, sample_grid=24, refinement_grid=48,
                                      min_visible_pixels=10, **quality)
    return cfg


class BoxObservationTests(unittest.TestCase):
    def setUp(self):
        self.target = box('target', [3, -.5, .5], [4, .5, 1.5])

    def metric(self, others=(), cfg=None, position=(0, 0), yaw=0):
        observer = BoxObserver(PointCamera(cfg or camera_config()), [self.target, *others])
        return observer.evaluate(0, [position], [yaw])[0]

    def test_analytic_unoccluded_projection_and_pixel_area(self):
        row = self.metric()
        self.assertTrue(row['qualified'])
        self.assertEqual(row['in_frame_fraction'], 1)
        self.assertEqual(row['visible_fraction'], 1)
        self.assertAlmostEqual(row['visible_pixels'], (100/3)**2, places=7)
        self.assertEqual(row['occluder_ids'], [])

    def test_full_half_and_behind_occluders(self):
        full = self.metric([box('front', [1, -1, 0], [2, 1, 2])])
        self.assertEqual(full['visible_fraction'], 0)
        self.assertFalse(full['qualified'])
        self.assertEqual(full['occluder_ids'], ['front'])
        half = self.metric([box('half', [1, 0, 0], [2, 1, 2])])
        self.assertAlmostEqual(half['visible_fraction'], .5)
        self.assertAlmostEqual(half['quality'], .5)
        self.assertFalse(half['qualified'])
        permissive = self.metric([box('half', [1, 0, 0], [2, 1, 2])], camera_config(min_observation_score=.4))
        self.assertTrue(permissive['qualified'])
        behind = self.metric([box('behind', [5, -1, 0], [6, 1, 2])])
        self.assertEqual(behind['visible_fraction'], 1)
        self.assertEqual(behind['occluder_ids'], [])

    def test_only_nearest_occluder_is_reported(self):
        row = self.metric([box('farther', [2, -1, 0], [2.5, 1, 2]),
                           box('nearer', [1, -1, 0], [1.5, 1, 2])])
        self.assertEqual(row['occluder_ids'], ['nearer'])

    def test_in_frame_fraction_is_separate_from_visibility(self):
        cfg = camera_config()
        cfg['camera']['cx'] = 0
        row = self.metric(cfg=cfg)
        self.assertAlmostEqual(row['in_frame_fraction'], .5)
        self.assertEqual(row['visible_fraction'], 1)
        self.assertFalse(row['qualified'])
        self.assertEqual(row['reason'], 'in_frame_fraction')

    def test_small_visible_area_fails_even_without_occlusion(self):
        cfg = camera_config()
        cfg['observation_quality']['min_visible_pixels'] = 2000
        row = self.metric(cfg=cfg)
        self.assertEqual(row['visible_fraction'], 1)
        self.assertFalse(row['qualified'])
        self.assertEqual(row['reason'], 'visible_pixels')

    def test_rays_parallel_tangent_inside_and_behind(self):
        origins = np.array([[-1, .5, .5], [-1, 2, .5], [-1, 0, .5], [.5, .5, .5], [2, .5, .5]])
        directions = np.tile([1., 0, 0], (5, 1))
        result = ray_box_entry(origins, directions, np.zeros(3), np.ones(3))
        np.testing.assert_array_equal(result, [1, np.inf, 1, 0, np.inf])

    def test_invalid_geometry_mixed_maps_and_thresholds(self):
        for field in ('min', 'max'):
            bad = copy.deepcopy(self.target)
            bad['aabb'][field][0] = float('nan')
            with self.assertRaises(ValueError): validate_objects([bad])
        bad = copy.deepcopy(self.target);bad['position'][0] += .1
        with self.assertRaises(ValueError): validate_objects([bad])
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            validate_objects([self.target, {'id': 'point', 'position': [0, 0, 0]}])
        for key, value in [('min_observation_score', 0), ('sample_grid', 1),
                           ('min_visible_pixels', float('nan')), ('refinement_grid', 4)]:
            cfg = camera_config();cfg['observation_quality'][key] = value
            with self.assertRaises(ValueError): ObservationSettings.from_config(cfg)

    def test_camera_inside_geometry_and_near_plane_are_explicit(self):
        row = self.metric(position=(3.5, 0))
        self.assertFalse(row['qualified']);self.assertEqual(row['reason'], 'camera_inside_geometry')
        # Outside the target in Y, with the box straddling the optical Z=0 plane.
        row = self.metric(position=(3.5, 2))
        self.assertFalse(row['qualified']);self.assertEqual(row['reason'], 'behind_or_crossing_camera_plane')

    def test_pose_transform_matches_independent_point_projection(self):
        cfg = camera_config()
        cfg['mount'] = {'position_in_base_m': [.23, -.1, 1], 'roll_deg': 12, 'pitch_deg': 18, 'yaw_deg': -15}
        camera = PointCamera(cfg)
        observer = BoxObserver(camera, [self.target])
        bases = np.array([[0, 0], [1, -1]])
        yaws = np.array([.2, -.4])
        origins, rotations = observer.poses(bases, yaws)
        point = np.array([4, 1, 1.2])
        optical = np.einsum('bi,bij->bj', point-origins, rotations)
        u, v, depth, _ = camera.project(bases, yaws, point)
        np.testing.assert_allclose(u, camera.fx*optical[:, 0]/optical[:, 2]+camera.cx)
        np.testing.assert_allclose(v, camera.fy*optical[:, 1]/optical[:, 2]+camera.cy)
        np.testing.assert_allclose(depth, optical[:, 2])

    def test_candidate_screening_matches_all_pose_evaluation(self):
        camera = PointCamera(camera_config())
        objects = [self.target, box('half', [1, 0, 0], [2, 1, 2])]
        stations = [Station(11+3*i, 1, (0, i), p) for i, p in enumerate([(0, 0), (0, -2), (5, 0), (100, 100)])]
        candidate = generate_candidates(stations, objects, camera, yaw_step_deg=90)
        observer = BoxObserver(camera, objects)
        expected = {}
        for st in stations:
            for yi, yaw in enumerate(candidate.yaws):
                for oi, obj in enumerate(objects):
                    if observer.evaluate(oi, [st.position], [yaw])[0]['qualified']:
                        expected.setdefault(st.index, {}).setdefault(yi, set()).add(obj['id'])
        self.assertEqual(candidate.supports, expected)
        self.assertNotIn('target', candidate.supports.get(11, {}).get(0, set()))

    def test_weighted_votes_deduplicate_directions_and_choose_best_photo(self):
        stations = [Station(i, 1, (0, i), (i, 0)) for i in range(3)]
        candidates = CandidateResult([0, 1, 2], [0, 1],
            {0: {0: {'a'}, 1: {'a'}}, 1: {0: {'a'}}, 2: {0: {'b', 'c'}}}, {},
            {0: {0: {'a': .6}, 1: {'a': .6}}, 1: {0: {'a': .9}}, 2: {0: {'b': .5, 'c': .5}}})
        result = select_anchors(candidates, stations, ['a', 'b', 'c'])
        self.assertEqual([r['station_index'] for r in result['vote_trace']], [2, 1])
        self.assertEqual(result['photo_count'], 2)
        # A single station's tied directions also prefer the better observation.
        one = CandidateResult([0], [0, 1], {0: {0: {'a'}, 1: {'a'}}}, {},
                              {0: {0: {'a': .6}, 1: {'a': .9}}})
        self.assertEqual(select_anchors(one, stations, ['a'])['anchors'][0]['observations'][0]['yaw_index'], 1)

    def test_aabb_cli_exports_metrics_without_touching_inputs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            scene = {'grid': {'rows': ['....']*4, 'origin': [0, -2], 'resolution': 1},
                     'objects': [self.target]}
            robot = {'robot': {'footprint_width_m': .2, 'footprint_length_m': .2, 'safety_margin_m': 0}}
            for name, value in [('map', scene), ('robot', robot), ('camera', camera_config())]:
                (folder/f'{name}.json').write_text(json.dumps(value))
            before = (folder/'map.json').read_bytes()
            output = folder/'plan.json'
            run = subprocess.run([sys.executable, '-B', str(root/'anchor_selector.py'), '--scene', str(folder/'map.json'),
                '--robot', str(folder/'robot.json'), '--camera', str(folder/'camera.json'), '--output', str(output),
                '--save-supports'], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result['observation_model'], 'aabb_weighted_region')
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['anchors'][0]['observations'][0]['object_metrics']['target']['qualified'])
            self.assertEqual((folder/'map.json').read_bytes(), before)
            self.assertTrue(json.loads(output.with_suffix('.supports.json').read_text())['scores'])


if __name__ == '__main__':
    unittest.main()
