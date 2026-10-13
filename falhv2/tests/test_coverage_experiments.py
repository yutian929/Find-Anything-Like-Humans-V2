import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from benchmark_preprocessing import distance_field, verify_small_maps
from coverage_candidates import greedy, tree_walk, scanline_sweep, open_route
from compare_coverage import metrics
from path_planner import GridMap, PlanningError
from plan_candidate import plan_candidate


class CoverageExperimentTests(unittest.TestCase):
    def test_distance_matches_square_geometry_and_outer_boundary(self):
        result = verify_small_maps()
        self.assertEqual(result["queries"], 700)
        self.assertLess(result["max_error_m"], 1e-10)

    def test_all_candidates_cover_random_components_without_jumps(self):
        rng = np.random.default_rng(132)
        for _ in range(20):
            occupied = rng.random((9, 11)) < 0.3
            grid = GridMap(tuple("".join("#" if v else "." for v in row) for row in occupied), 0.5, (0, 0))
            _, dense = distance_field(grid, full=True)
            safe = (~occupied) & (dense[1::2,1::2] >= 0.2)
            labels, count = label(safe)
            regions = [set(map(tuple, np.argwhere(labels == i))) for i in range(1, count+1)]
            for fn in (greedy, tree_walk, scanline_sweep):
                with self.subTest(method=fn.__name__):
                    paths = [open_route(fn(cells)) for cells in regions]
                    result = metrics(paths, regions, 0.5, dense, 0.2)
                    self.assertEqual(result["coverage_fraction"], 1)
                    self.assertTrue(result["collision_free"])

    def test_single_cell_and_one_cell_corridor(self):
        for cells in ({(0,0)}, {(0,c) for c in range(11)}, {(0,0),(1,0),(2,0),(2,1)}):
            for fn in (greedy, tree_walk, scanline_sweep):
                route = open_route(fn(cells))
                self.assertEqual(set(route), cells)
                self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(route,route[1:])))

    def test_candidate_exports_separate_routes_and_world_coordinates(self):
        scene = {"grid": {"rows": ["..#..", "..#.."], "resolution": 0.5, "origin": [2, -3]}}
        config = {"robot": {"footprint_width_m": 0.28, "footprint_length_m": 0.28, "safety_margin_m": 0}}
        result = plan_candidate(scene, config)
        self.assertEqual(result["statistics"]["region_count"], 2)
        self.assertEqual(result["statistics"]["traversable_cells"], 8)
        grid = GridMap.from_scene(scene)
        for region in result["regions"]:
            for cell, waypoint in zip(region["route_cells"], region["waypoints"]):
                self.assertEqual(grid.cell_to_world(cell), (waypoint["x_m"], waypoint["y_m"]))
        scene["grid"]["rows"] = ["#####", "#####"]
        with self.assertRaises(PlanningError) as caught:
            plan_candidate(scene, config)
        self.assertEqual(caught.exception.code, "no_traversable_cells")

    def test_validation_rejects_missing_region(self):
        with self.assertRaisesRegex(ValueError, "one route per"):
            metrics([], [{(0, 0)}], 0.5, np.ones((3, 3)), 0.2)


if __name__ == "__main__":
    unittest.main()
