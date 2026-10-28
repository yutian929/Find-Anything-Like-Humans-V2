"""Object workers retain every occluder, deterministic ties and parent settings."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from falhv2.anchor_planners import anchor_candidates as candidates
from falhv2.anchor_planners import observation_model as model
from test_observation_model import box, camera_config


class ParallelCandidateTests(unittest.TestCase):
    def setUp(self):
        self.stations = [candidates.Station(90-i*3, 1, (0, i), position)
                         for i, position in enumerate([(0, 0), (0, 0), (0, 3), (1, -3)])]
        self.objects = [box('target', [3, -.5, .5], [4, .5, 1.5]),
                        box('blocker', [1, -1, 0], [2, 1, 2])]
        self.camera = model.PointCamera(camera_config())

    def compare(self, expected, actual):
        for field in ('station_indices', 'yaws', 'supports', 'scores', 'diagnostics'):
            self.assertEqual(getattr(expected, field), getattr(actual, field), field)
        volatile = {'elapsed_s', 'generation_seconds', 'candidate_workers', 'candidate_executor'}
        self.assertEqual({k: v for k, v in expected.stats.items() if k not in volatile},
                         {k: v for k, v in actual.stats.items() if k not in volatile})
        self.assertEqual(actual.stats['elapsed_s'], actual.stats['generation_seconds'])

    def test_threads_and_spawn_preserve_occlusion_and_selected_station_order(self):
        kwargs = dict(yaw_step_deg=30, station_indices=[81, 90, 87, 84])
        serial = candidates.generate_candidates(self.stations, self.objects, self.camera, **kwargs)
        self.assertNotIn('target', serial.supports.get(90, {}).get(0, set()))
        for executor in ('threads', 'processes'):
            with self.subTest(executor=executor):
                parallel = candidates.generate_candidates(self.stations, self.objects, self.camera,
                                                          workers=2, executor=executor, **kwargs)
                self.compare(serial, parallel)
                self.assertEqual(parallel.stats['candidate_workers'], 2)
                self.assertEqual(parallel.stats['candidate_executor'], executor)

    def test_spawn_retains_parent_defaults_and_best_failure_diagnostics(self):
        config = camera_config()
        config['observation_quality'].pop('min_visible_pixels', None)
        config['observation_quality'].pop('sample_grid', None)
        config['observation_quality'].pop('refinement_grid', None)
        camera = model.PointCamera(config)
        with patch.dict(model.project_config.OBSERVATION_DEFAULTS,
                        min_visible_pixels=1e8, sample_grid=8, refinement_grid=16):
            serial = candidates.generate_candidates(self.stations, self.objects, camera)
            parallel = candidates.generate_candidates(self.stations, self.objects, camera, workers=2)
        self.assertEqual(serial.supports, {})
        self.assertGreater(serial.stats['diagnostic_recomputed_pose_pairs'], 0)
        self.compare(serial, parallel)

    def test_parallel_point_map_keeps_legacy_statistics(self):
        objects = [{'id': obj['id'], 'position': obj['position']} for obj in self.objects]
        serial = candidates.generate_candidates(self.stations, objects, self.camera)
        parallel = candidates.generate_candidates(self.stations, objects, self.camera, workers=2)
        self.compare(serial, parallel)
        self.assertIn('exact_point_pose_tests', parallel.stats)

    def test_empty_inputs_and_one_target_do_not_start_pool(self):
        with patch.object(candidates, 'ProcessPoolExecutor', side_effect=AssertionError('unneeded pool')):
            for stations, objects, indices in [(self.stations, [], None), ([], self.objects, None),
                                               (self.stations, self.objects, []),
                                               (self.stations, self.objects[:1], None)]:
                result = candidates.generate_candidates(stations, objects, self.camera,
                                                        workers=4, station_indices=indices)
                self.assertEqual(result.stats['candidate_executor'], 'serial')

    def test_invalid_arguments_fail_before_spawning(self):
        with patch.object(candidates, 'ProcessPoolExecutor', side_effect=AssertionError('unneeded pool')):
            for value in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    candidates.generate_candidates(self.stations, self.objects, self.camera, workers=value)
            with self.assertRaises(ValueError):
                candidates.generate_candidates(self.stations, self.objects, self.camera, executor='invalid')
            with self.assertRaises(ValueError):
                candidates.generate_candidates(self.stations*2, self.objects, self.camera, workers=2)

    def test_worker_failure_is_propagated_instead_of_returning_partial_coverage(self):
        original = candidates._generate_candidates_serial
        def fail(*args, **kwargs):
            if kwargs.get('_target_indices') == (1,):
                raise ValueError('worker failure')
            return original(*args, **kwargs)
        with patch.object(candidates, '_generate_candidates_serial', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'worker failure'):
                candidates.generate_candidates(self.stations, self.objects, self.camera,
                                                workers=2, executor='threads')


if __name__ == '__main__':
    unittest.main()
