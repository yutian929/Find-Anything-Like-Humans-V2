import json
import math
import sys
import unittest
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_planner import GridMap, PlanningError, diagnostic_svg, plan


def scene(rows, resolution=0.5, origin=(0, 0)):
    return {"name": "test", "grid": {"rows": rows, "resolution": resolution,
                                     "origin": list(origin)}, "objects": []}


def robot(width=0.28, length=0.28, margin=0.0):
    return {"robot": {"footprint_width_m": width,
                      "footprint_length_m": length, "safety_margin_m": margin}}


class PathPlannerTests(unittest.TestCase):
    def test_coordinate_conversion_uses_bottom_left_origin_and_top_first_rows(self):
        grid = GridMap.from_scene(scene([".#?", "..."], origin=(10, -2)))
        self.assertEqual(grid.cell_to_world((0, 0)), (10.25, -1.25))
        self.assertEqual(grid.cell_to_world((1, 2)), (11.25, -1.75))
        self.assertEqual(grid.world_to_cell(10.25, -1.25), (0, 0))
        self.assertEqual(grid.world_to_cell(11.25, -1.75), (1, 2))

    def test_each_region_path_is_continuous_safe_and_complete(self):
        data = scene(["..?...", "..?..."])
        result = plan(data, robot())
        grid = GridMap.from_scene(data)
        self.assertEqual(result["statistics"]["region_count"], 2)
        self.assertEqual(result["statistics"]["traversable_cells"], 10)
        self.assertEqual(result["statistics"]["visited_traversable_cells"], 10)
        self.assertEqual([len(region["cells"]) for region in result["regions"]], [4, 6])
        all_cells = set()
        for region in result["regions"]:
            cells = [tuple(cell) for cell in region["route_cells"]]
            self.assertEqual(set(cells), {tuple(cell) for cell in region["cells"]})
            self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                                for a, b in zip(cells, cells[1:])))
            all_cells.update(cells)
            self.assertAlmostEqual(region["path_length_m"], 0.5 * (len(cells) - 1))
            for a, b in zip(region["waypoints"], region["waypoints"][1:]):
                midpoint = ((a["x_m"] + b["x_m"]) / 2, (a["y_m"] + b["y_m"]) / 2)
                self.assertGreaterEqual(grid.clearance_at(*midpoint) + 1e-10,
                                        math.hypot(0.28, 0.28) / 2)
        self.assertTrue(all(grid.rows[row][col] == "." for row, col in all_cells))
        self.assertEqual(result["visual_object_coverage"], "not_evaluated")

    def test_footprint_clearance_excludes_narrow_cells(self):
        data = scene(["###", "#.#", "###"])
        with self.assertRaises(PlanningError) as caught:
            plan(data, robot(width=0.5, length=0.5))
        self.assertEqual(caught.exception.code, "no_traversable_cells")

    def test_all_blocked_is_explicit_failure(self):
        with self.assertRaises(PlanningError) as caught:
            plan(scene(["##", "??"]), robot())
        self.assertEqual(caught.exception.code, "no_traversable_cells")

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(PlanningError) as caught:
            plan(scene(["..", ".."]), robot(width=-0.1))
        self.assertEqual(caught.exception.code, "invalid_robot")

    def test_example_map_plan_and_svg(self):
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "examples/map.json").read_text())
        data.pop("objects")  # A1 needs only the grid, even for an editor export.
        config = json.loads((root / "config/robot.example.json").read_text())
        result = plan(data, config)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["statistics"]["visited_traversable_cells"],
                         result["statistics"]["traversable_cells"])
        grid = GridMap.from_scene(data)
        clearance = math.hypot(config["robot"]["footprint_width_m"],
                               config["robot"]["footprint_length_m"]) / 2 + config["robot"]["safety_margin_m"]
        for region in result["regions"]:
            route = [tuple(cell) for cell in region["route_cells"]]
            for a, b in zip(route, route[1:]):
                self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
                ax, ay = grid.cell_to_world(a)
                bx, by = grid.cell_to_world(b)
                self.assertGreaterEqual(grid.clearance_at((ax + bx) / 2, (ay + by) / 2)
                                        + 1e-10, clearance)
        graphic = diagnostic_svg(data, result)
        self.assertIn("<polyline", graphic)
        ElementTree.fromstring(graphic)


if __name__ == "__main__":
    unittest.main()
