"""Compare A1 preprocessing with a distance-field prototype (not the default planner).

Run in conda test from the repository root. Requires NumPy and SciPy.
The coverage routing algorithm is kept unchanged to isolate preprocessing cost.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt, label

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from path_planner import GridMap, Robot, _region_route


def distance_field(grid, full=False):
    """Cell-center distances to closed obstacle squares and the outer boundary.

    A grid with half the original spacing contains obstacle-square corners,
    edge midpoints and centers. For queries at original cell centers, the
    nearest point on any axis-aligned obstacle square is on this finer grid.
    Thus EDT here measures distance to squares, not just obstacle centers.
    Outer boundary samples are included even on an otherwise empty map.
    """
    blocked = np.array([[v != "." for v in row] for row in grid.rows])
    h, w = blocked.shape
    occupied = np.zeros((2 * h + 1, 2 * w + 1), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            occupied[dr:dr + 2 * h:2, dc:dc + 2 * w:2] |= blocked
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    distances = distance_transform_edt(~occupied, sampling=grid.resolution / 2)
    return blocked, distances if full else distances[1::2, 1::2]


def verify_small_maps():
    rng = np.random.default_rng(20261013)
    queries = 0
    max_error = 0.0
    for index in range(20):
        cells = rng.choice([".", "#", "?"], (5, 7), p=[0.7, 0.2, 0.1])
        if index == 0:
            cells[:] = "."
        if index == 1:
            cells[:] = "#"
        grid = GridMap(tuple("".join(row) for row in cells),
                       0.05 if index % 2 else 0.5, (2.3, -1.7))
        _, distances = distance_field(grid)
        for row in range(grid.height):
            for col in range(grid.width):
                exact = grid.clearance_at(*grid.cell_to_world((row, col)))
                max_error = max(max_error, abs(exact - distances[row, col]))
                queries += 1
    assert max_error < 1e-10, max_error
    return {"maps": 20, "queries": queries, "max_error_m": max_error}


def benchmark(name, scene, config, sample_count):
    grid = GridMap.from_scene(scene)
    robot = Robot.from_config(config)
    started = perf_counter()
    blocked, distances = distance_field(grid)
    safe = (~blocked) & (distances + 1e-10 >= robot.clearance_m)
    preprocessing_seconds = perf_counter() - started
    free = np.argwhere(~blocked)
    count = min(sample_count, len(free))
    sample = free[np.linspace(0, len(free) - 1, count, dtype=int)] if count else []
    started = perf_counter()
    reference = [grid.clearance_at(*grid.cell_to_world(tuple(cell))) for cell in sample]
    reference_seconds = perf_counter() - started
    error = max((abs(value - distances[tuple(cell)]) for cell, value in zip(sample, reference)), default=0)
    assert error < 1e-10, error

    started = perf_counter()
    labels, region_count = label(safe, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    regions = [set(map(tuple, np.argwhere(labels == i))) for i in range(1, region_count + 1)]
    component_seconds = perf_counter() - started
    started = perf_counter()
    plans = [_region_route(grid, cells, i + 1) for i, cells in enumerate(regions)]
    routing_seconds = perf_counter() - started

    for cells, plan in zip(regions, plans):
        route = list(map(tuple, plan["route_cells"]))
        assert set(route) == cells
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(route, route[1:]))
    return {
        "case": name,
        "grid_sha256": hashlib.sha256(json.dumps(scene["grid"], sort_keys=True).encode()).hexdigest(),
        "width": grid.width, "height": grid.height, "resolution_m": grid.resolution,
        "free_cells": len(free), "safe_cells": int(safe.sum()),
        "region_sizes": list(map(len, regions)),
        "robot": config["robot"], "clearance_radius_m": robot.clearance_m,
        "old_clearance_sample_count": count,
        "old_clearance_sample_seconds": reference_seconds,
        "old_full_clearance_estimated_seconds": reference_seconds / count * len(free) if count else None,
        "max_sample_clearance_error_m": float(error),
        "prototype_preprocessing_seconds": preprocessing_seconds,
        "component_extraction_seconds": component_seconds,
        "unchanged_routing_seconds": routing_seconds,
        "path_length_m": sum(p["path_length_m"] for p in plans),
        "waypoints": sum(len(p["waypoints"]) for p in plans),
        "all_safe_centers_visited": True, "all_route_edges_four_connected": True,
    }


def synthetic_scene(size):
    cells = np.full((size, size), ".", dtype="<U1")
    cells[[0, -1], :] = "#"
    cells[:, [0, -1]] = "#"
    cells[size // 4:size // 2, size // 4:size // 3] = "#"
    cells[size // 2:3 * size // 4, 2 * size // 3:3 * size // 4] = "#"
    return {"grid": {"rows": ["".join(row) for row in cells], "resolution": 0.05, "origin": [0, 0]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=MODULE / "outputs/map.json")
    parser.add_argument("--robot", type=Path, default=MODULE / "config/robot.example.json")
    parser.add_argument("--output", type=Path, default=MODULE / "outputs/preprocessing_benchmark.json")
    args = parser.parse_args()
    config = json.loads(args.robot.read_text())
    result = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "scope": "single_run_preprocessing_prototype_and_existing_route_not_external_planner_comparison",
        "timing_excludes": ["imports", "file_io", "diagnostic_rendering", "validation"],
        "small_map_distance_validation": verify_small_maps(),
        "cases": [benchmark("current_map", json.loads(args.scene.read_text()), config, 60)],
    }
    for size in (300, 512):
        result["cases"].append(benchmark(f"synthetic_{size}", synthetic_scene(size), config, 20))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
