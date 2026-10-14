"""Sampling guarantees independent from the downstream station selector."""

import unittest

from falhv2.anchor_experiments.common import PointCamera, Station, coverage_edges
from falhv2.anchor_experiments.position_driver import _uniform_order, generate


def camera():
    return PointCamera({
        "camera": {"width": 101, "height": 101, "fx": 50, "fy": 50,
                   "cx": 50, "cy": 50, "distortion_model": "none",
                   "distortion_coefficients": [0] * 5},
        "mount": {"position_in_base_m": [0, 0, 1],
                  "roll_deg": 0, "pitch_deg": 0, "yaw_deg": 0},
        "observation_quality": {"min_distance_m": 0.4, "max_distance_m": 0.75},
    })


class PositionDriverTests(unittest.TestCase):
    def setUp(self):
        self.stations = [Station(i * 2, 1, (0, i), (float(i), 0.0), (i,)) for i in range(11)]
        self.target = [{"id": "target", "position": [8.5, 0, 1]}]

    def test_random_reproducible_and_refinement_nested(self):
        small = generate(self.stations, self.target, camera(), strategy="random",
                         initial_count=2, seed=17, refine=False, yaw_step_deg=90)
        large = generate(self.stations, self.target, camera(), strategy="random",
                         initial_count=2, seed=17, refine=True, yaw_step_deg=90)
        repeat = generate(self.stations, self.target, camera(), strategy="random",
                          initial_count=2, seed=17, refine=True, yaw_step_deg=90)
        self.assertEqual(small.station_indices, large.station_indices[:len(small.station_indices)])
        self.assertEqual(large.station_indices, repeat.station_indices)
        self.assertEqual(coverage_edges(large), coverage_edges(repeat))
        self.assertEqual(len(large.station_indices), len(set(large.station_indices)))

    def test_uniform_refinement_repairs_missing_target(self):
        small = generate(self.stations, self.target, camera(), initial_count=3,
                         refine=False, yaw_step_deg=90)
        self.assertEqual(small.station_indices, [0, 20, 10])
        self.assertEqual(small.stats["covered_object_count"], 0)
        refined = generate(self.stations, self.target, camera(), initial_count=3, yaw_step_deg=90)
        self.assertEqual(refined.stats["covered_object_count"], 1)
        self.assertEqual(refined.stats["stop_reason"], "all_objects_supported")
        self.assertEqual(small.station_indices, refined.station_indices[:3])
        self.assertEqual(refined.stats["evaluated_point_pose_pairs"], 4 * len(refined.station_indices))

    def test_unobservable_object_exhausts_finite_pool(self):
        targets = [{"id": "too_high", "position": [8.5, 0, 100]}]
        result = generate(self.stations, targets, camera(), initial_count=2)
        self.assertEqual(set(result.station_indices), {s.index for s in self.stations})
        self.assertEqual(result.supports, {})
        self.assertEqual(result.stats["stop_reason"], "candidate_pool_exhausted")
        limited = generate(self.stations, targets, camera(), initial_count=2, max_stations=4)
        self.assertEqual(limited.stats["final_station_count"], 4)
        self.assertEqual(limited.stats["stop_reason"], "station_budget_exhausted")

    def test_fixed_pool_evaluated_exactly_and_empty_objects_skip_work(self):
        fixed = generate(self.stations, self.target, camera(), station_indices=[18, 18, 0],
                         max_stations=1, initial_count=1, yaw_step_deg=90)
        self.assertEqual(fixed.station_indices, [18, 0])
        self.assertEqual(fixed.stats["stop_reason"], "fixed_candidate_pool_evaluated")
        self.assertEqual(fixed.stats["covered_object_count"], 1)
        empty = generate(self.stations, [], camera())
        self.assertEqual(empty.station_indices, [])
        self.assertEqual(empty.stats["evaluated_point_pose_pairs"], 0)

    def test_exhaustive_matches_manual_visibility(self):
        sensor = camera()
        result = generate(self.stations, self.target, sensor, strategy="exhaustive", yaw_step_deg=90)
        import numpy as np
        expected = set()
        for station in self.stations:
            for yaw_index, yaw in enumerate(result.yaws):
                if sensor.visible(np.asarray([station.position]), np.asarray([yaw]), self.target[0]["position"])[0]:
                    expected.add((station.index, yaw_index, "target"))
        self.assertEqual(coverage_edges(result), expected)

    def test_rejects_duplicate_ids_and_nonfinite_object_positions(self):
        for objects in ([self.target[0], self.target[0]],
                        [{"id": "invalid", "position": [0, float("nan"), 1]}],
                        [{"id": "invalid", "position": [float("inf"), 0, 1]}]):
            with self.subTest(objects=objects), self.assertRaises(ValueError):
                generate(self.stations, objects, camera())

    def test_uniform_keeps_each_region_endpoint_and_reports_budget_tradeoff(self):
        stations = [Station(i, i // 3, (0, i), (float(i), 0.0), (i,)) for i in range(9)]
        objects = [{"id": "unreachable", "position": [0, 0, 100]}]
        initial = generate(stations, objects, camera(), initial_count=2, refine=False)
        endpoints = {0, 2, 3, 5, 6, 8}
        self.assertEqual(set(initial.station_indices), endpoints)
        self.assertEqual(initial.stats["initial_station_count"], 6)
        self.assertEqual(initial.stats["evaluated_region_endpoint_count"], 6)
        self.assertFalse(initial.stats["endpoint_budget_limited"])
        limited = generate(stations, objects, camera(), initial_count=2, refine=False, max_stations=4)
        self.assertEqual(limited.stats["initial_station_count"], 4)
        self.assertEqual(limited.stats["evaluated_region_endpoint_count"], 4)
        self.assertTrue(limited.stats["endpoint_budget_limited"])

    def test_lazy_uniform_prefix_preserves_previous_eager_order(self):
        from itertools import islice
        # Prefixes recorded before replacing the eager implementation.
        cases = [
            (0, 2, [], []),
            (1, 1, [0], [0]),
            (7, 3, [0, 6], [0, 6, 3, 1, 4, 2, 5]),
            (17, 4, [0, 4, 5, 16], [0, 4, 5, 16, 11, 8, 13, 2, 6, 9, 14, 1, 3, 7, 10, 12, 15]),
            (100, 5, [0, 9, 10, 99], [0, 9, 10, 99, 25, 50, 74, 37, 86, 62, 17, 43, 92, 31, 56, 68, 80, 4, 21, 13]),
            (64000, 8, [0, 31999, 32000, 63999], [0, 31999, 32000, 63999, 9143, 18285, 27428, 36571, 45714, 54856, 4571, 22856, 41142, 59427, 13714, 50285, 6857, 25142, 43428, 61713]),
        ]
        for count, initial, endpoints, expected in cases:
            with self.subTest(count=count):
                self.assertEqual(list(islice(_uniform_order(count, initial, endpoints), 20)), expected)

    def test_uniform_refinement_deterministic(self):
        first = generate(self.stations, self.target, camera(), initial_count=3, yaw_step_deg=90)
        second = generate(self.stations, self.target, camera(), initial_count=3, yaw_step_deg=90)
        self.assertEqual(first.station_indices, second.station_indices)
        self.assertEqual(coverage_edges(first), coverage_edges(second))


if __name__ == "__main__":
    unittest.main()
