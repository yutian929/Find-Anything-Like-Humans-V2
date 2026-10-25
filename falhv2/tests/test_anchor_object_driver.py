"""Target-first spatial screening must preserve exact finite-pool supports."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observation_model import PointCamera
from anchor_candidates import Station
from anchor_candidates import generate_candidates as generate


class OffsetCamera:
    """Independent test predicate: yaw-rotating translated camera, range + FOV."""

    max_distance = 2.0
    min_distance = 0.3
    offset_xy_norm = 0.7

    def visible(self, base_positions, yaws, point):
        heading = yaws + 0.4
        # Nonzero forward camera offset rotates with the base.
        camera_xy = base_positions + self.offset_xy_norm * np.column_stack((np.cos(yaws), np.sin(yaws)))
        delta = np.asarray(point)[:2] - camera_xy
        z = np.sum(delta * np.column_stack((np.cos(heading), np.sin(heading))), axis=1)
        side = np.sum(delta * np.column_stack((-np.sin(heading), np.cos(heading))), axis=1)
        distance = np.sqrt(np.sum(delta**2, axis=1) + (point[2] - 0.8)**2)
        return (z > 0) & (np.abs(side) <= z * 0.6) & (distance <= self.max_distance) & (distance >= self.min_distance)


def brute_supports(stations, objects, camera, yaws):
    supports = {}
    for station in stations:
        for j, yaw in enumerate(yaws):
            for obj in objects:
                if camera.visible(np.array([station.position]), np.array([yaw]), obj["position"])[0]:
                    supports.setdefault(station.index, {}).setdefault(j, set()).add(obj["id"])
    return supports


class ObjectDriverTests(unittest.TestCase):
    def test_screening_matches_dense_with_offset_camera_and_noncontiguous_indices(self):
        rng = np.random.default_rng(42)
        stations = [Station(11 + 3 * i, 1, (i, 0), tuple(p)) for i, p in enumerate(rng.uniform(-8, 8, (80, 2)))]
        objects = [{"id": f"p{i}", "position": p.tolist()} for i, p in enumerate(rng.uniform([-6, -6, 0], [6, 6, 2], (15, 3)))]
        camera = OffsetCamera()
        result = generate(stations, objects, camera, yaw_step_deg=45)
        self.assertEqual(result.supports, brute_supports(stations, objects, camera, result.yaws))
        self.assertLess(result.stats["exact_point_pose_tests"], len(stations) * len(objects) * len(result.yaws))
        selected = [stations[17].index, stations[2].index]
        partial = generate(stations, objects, camera, yaw_step_deg=45, station_indices=selected)
        self.assertEqual(partial.station_indices, selected)
        self.assertEqual(partial.supports, {key: value for key, value in result.supports.items() if key in selected})

    def test_screening_preserves_maximum_range_boundary(self):
        station = Station(19, 1, (0, 0), (0.0, 0.0))
        # Base-to-object distance exceeds camera max range due to mount offset.
        objects = [{"id": "edge", "position": [2.7 - 1e-12, 0.0, 0.8]}]
        result = generate([station], objects, OffsetCamera(), yaw_step_deg=90)
        self.assertEqual(result.supports, brute_supports([station], objects, OffsetCamera(), result.yaws))
        self.assertEqual(result.stats["screened_station_object_pairs"], 1)
        self.assertIn("edge", result.supports[19][0])

    def test_shared_camera_with_rotated_translated_mount_matches_dense(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "data/camera.d435i.json").read_text())
        config["mount"] = {"position_in_base_m": [0.6, -0.35, 1.1], "roll_deg": 12.0, "pitch_deg": 22.0, "yaw_deg": -31.0}
        config["observation_quality"] = {"min_distance_m": 0.2, "max_distance_m": 3.0}
        camera = PointCamera(config)
        rng = np.random.default_rng(893)
        stations = [Station(100 + i * 7, 1, (i, 0), tuple(p)) for i, p in enumerate(rng.uniform(-5, 5, (45, 2)))]
        objects = [{"id": str(i), "position": p.tolist()} for i, p in enumerate(rng.uniform([-4, -4, 0], [4, 4, 2.5], (18, 3)))]
        result = generate(stations, objects, camera, yaw_step_deg=30)
        self.assertTrue(result.supports)
        self.assertEqual(result.supports, brute_supports(stations, objects, camera, result.yaws))

    def test_empty_inputs_and_missing_reference_index(self):
        camera = OffsetCamera()
        self.assertEqual(generate([], [], camera).supports, {})
        with self.assertRaises(ValueError):
            generate([], [], camera, station_indices=[99])


if __name__ == "__main__":
    unittest.main()
