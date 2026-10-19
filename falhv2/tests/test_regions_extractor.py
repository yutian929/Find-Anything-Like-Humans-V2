import math
import sys
import unittest
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regions_extractor import GridMap, PlanningError, diagnostic_svg, plan


def scene(rows, resolution=0.5, origin=(0, 0)):
    return {"name": "test", "grid": {"rows": rows, "resolution": resolution,
                                     "origin": list(origin)}, "objects": []}


def robot(width=0.28, length=0.28, margin=0.0):
    return {"robot": {"footprint_width_m": width,
                      "footprint_length_m": length, "safety_margin_m": margin}}


class SafeRegionTests(unittest.TestCase):
    def test_coordinate_conversion_uses_bottom_left_origin_and_top_first_rows(self):
        grid = GridMap.from_scene(scene([".#?", "..."], origin=(10, -2)))
        self.assertEqual(grid.cell_to_world((0, 0)), (10.25, -1.25))
        self.assertEqual(grid.cell_to_world((1, 2)), (11.25, -1.75))
        self.assertEqual(grid.world_to_cell(10.25, -1.25), (0, 0))
        self.assertEqual(grid.world_to_cell(11.25, -1.75), (1, 2))

    def test_regions_partition_exactly_the_safe_free_cells(self):
        data = scene(["..?...", "..?..."])
        result = plan(data, robot())
        grid = GridMap.from_scene(data)
        self.assertEqual(result["statistics"]["region_count"], 2)
        self.assertEqual(result["statistics"]["traversable_cells"], 10)
        self.assertEqual([len(region["cells"]) for region in result["regions"]], [4, 6])
        all_cells = set()
        for region in result['regions']:
            cells = set(map(tuple, region['cells']))
            self.assertFalse(all_cells & cells)
            all_cells.update(cells)
            reached = {min(cells)}
            pending = list(reached)
            while pending:
                r, c = pending.pop()
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    neighbor = (r+dr, c+dc)
                    if neighbor in cells and neighbor not in reached:
                        reached.add(neighbor)
                        pending.append(neighbor)
            self.assertEqual(reached, cells)
            self.assertNotIn('route_cells', region)
            self.assertNotIn('waypoints', region)
        expected = {(r, c) for r in range(grid.height) for c in range(grid.width)
                    if grid.rows[r][c] == '.' and
                    grid.clearance_at(*grid.cell_to_world((r, c))) + 1e-10 >= math.hypot(.28, .28)/2}
        self.assertEqual(all_cells, expected)
        self.assertEqual(result["visual_object_coverage"], "not_evaluated")

    def test_diagonal_free_cells_are_separate_regions(self):
        result = plan(scene(['.#', '#.']), robot())
        self.assertEqual([r['cells'] for r in result['regions']], [[[0, 0]], [[1, 1]]])

    def test_robot_clearance_splits_a_connected_white_area(self):
        data = scene(['....#....', '....#....', '.........', '....#....', '....#....'], resolution=1)
        small = plan(data, robot(width=.2, length=.2))
        large = plan(data, robot(width=.8, length=.8))
        self.assertEqual(len(small['regions']), 1)
        self.assertEqual(len(large['regions']), 2)
        self.assertTrue(all([2, 4] not in r['cells'] for r in large['regions']))

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

    def test_map_plan_and_svg(self):
        # Fixed test input stays independent of user maps and robot settings.
        data = scene(["............", "............", "....##......",
                      "....##......", "............", "............"])
        data.pop("objects")  # A1 needs only the grid, even for an editor export.
        config = robot()
        result = plan(data, config)
        self.assertEqual(result["status"], "complete")
        grid = GridMap.from_scene(data)
        clearance = math.hypot(config["robot"]["footprint_width_m"],
                               config["robot"]["footprint_length_m"]) / 2 + config["robot"]["safety_margin_m"]
        expected = {(r, c) for r in range(grid.height) for c in range(grid.width)
                    if grid.rows[r][c] == '.' and
                    grid.clearance_at(*grid.cell_to_world((r, c))) + 1e-10 >= clearance}
        actual = {tuple(cell) for region in result['regions'] for cell in region['cells']}
        self.assertEqual(actual, expected)
        graphic = diagnostic_svg(data, result)
        self.assertNotIn("<polyline", graphic)
        ElementTree.fromstring(graphic)


if __name__ == "__main__":
    unittest.main()
