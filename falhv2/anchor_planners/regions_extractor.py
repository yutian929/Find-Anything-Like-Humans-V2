"""A1: extract all safe grid regions without generating a coverage route.

Uses a circumscribed robot disk, half-grid EDT clearance and four-connected
components. No start pose is supplied: regions are not filtered by reachability
from a particular robot position. Navigation between selected anchors is separate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


Cell = tuple[int, int]  # (row from top, column from left)


class PlanningError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GridMap:
    rows: tuple[str, ...]
    resolution: float
    origin: tuple[float, float]

    @classmethod
    def from_scene(cls, scene: dict[str, Any]) -> GridMap:
        try:
            grid = scene["grid"]
            rows = tuple(grid["rows"])
            resolution = float(grid["resolution"])
            origin = tuple(float(v) for v in grid["origin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanningError("invalid_map", "地图缺少有效的 grid.rows、resolution 或 origin") from exc
        if (not rows or not isinstance(rows[0], str) or not rows[0]
                or any(not isinstance(row, str) or len(row) != len(rows[0])
                       or set(row) - {".", "#", "?"} for row in rows)
                or not math.isfinite(resolution) or resolution <= 0
                or len(origin) != 2 or not all(math.isfinite(v) for v in origin)):
            raise PlanningError("invalid_map", "栅格必须为等宽非空行，且只含 . # ?；分辨率须为正数")
        return cls(rows, resolution, origin)

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])

    def cell_to_world(self, cell: Cell) -> tuple[float, float]:
        row, col = cell
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise PlanningError("invalid_cell", f"栅格索引越界：{cell}")
        return (self.origin[0] + (col + 0.5) * self.resolution,
                self.origin[1] + (self.height - row - 0.5) * self.resolution)

    def world_to_cell(self, x: float, y: float) -> Cell:
        if not math.isfinite(x) or not math.isfinite(y):
            raise PlanningError("invalid_start", "起点坐标必须是有限数")
        col = math.floor((x - self.origin[0]) / self.resolution)
        row = self.height - 1 - math.floor((y - self.origin[1]) / self.resolution)
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise PlanningError("invalid_start", "起点在地图范围外")
        return row, col

    def clearance_at(self, x: float, y: float) -> float:
        """Distance to occupied/unknown square or outer map boundary."""
        x0, y0 = self.origin
        x1 = x0 + self.width * self.resolution
        y1 = y0 + self.height * self.resolution
        distance = min(x - x0, x1 - x, y - y0, y1 - y)
        for row, line in enumerate(self.rows):
            for col, symbol in enumerate(line):
                if symbol == ".":
                    continue
                left = x0 + col * self.resolution
                right = left + self.resolution
                bottom = y0 + (self.height - row - 1) * self.resolution
                top = bottom + self.resolution
                dx = max(left - x, 0.0, x - right)
                dy = max(bottom - y, 0.0, y - top)
                distance = min(distance, math.hypot(dx, dy))
        return distance


@dataclass(frozen=True)
class Robot:
    footprint_width_m: float
    footprint_length_m: float
    safety_margin_m: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Robot:
        try:
            robot = config["robot"]
            width = float(robot["footprint_width_m"])
            length = float(robot["footprint_length_m"])
            margin = float(robot["safety_margin_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanningError("invalid_robot", "机器人需要 footprint_width_m、footprint_length_m 和 safety_margin_m") from exc
        if (not all(math.isfinite(v) for v in (width, length, margin))
                or width <= 0 or length <= 0 or margin < 0):
            raise PlanningError("invalid_robot", "机器人长宽须为正数，安全距离不得为负")
        return cls(width, length, margin)

    @property
    def clearance_m(self) -> float:
        # A circumscribed disk is safe at any yaw for an in-place-turning base.
        return math.hypot(self.footprint_width_m, self.footprint_length_m) / 2 + self.safety_margin_m


def distance_field(grid: GridMap, full: bool = False) -> tuple[np.ndarray, np.ndarray]:
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


def _connected_regions(safe: np.ndarray) -> list[set[Cell]]:
    """Label four-connected regions and preserve topmost-leftmost ID order."""
    count, labels = cv2.connectedComponents(safe.astype(np.uint8), connectivity=4)
    components: list[set[Cell]] = [set() for _ in range(count - 1)]
    rows, cols = np.nonzero(safe)
    # Group in one pass, avoiding a full-image scan for every component.
    for row, col, region_id in zip(rows, cols, labels[rows, cols]):
        components[int(region_id) - 1].add((int(row), int(col)))
    return sorted(components, key=min)


def plan(scene: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    grid = GridMap.from_scene(scene)
    robot = Robot.from_config(config)
    blocked, dense = distance_field(grid, full=True)
    safe = (~blocked) & (dense[1::2, 1::2] + 1e-10 >= robot.clearance_m)
    components = _connected_regions(safe)
    if not components:
        raise PlanningError("no_traversable_cells", "机器人足迹及安全距离下，没有可通行的空闲格中心")
    regions = [{"id": index + 1, "cells": [list(cell) for cell in sorted(cells)]}
               for index, cells in enumerate(components)]

    return {
        "status": "complete",
        "scope": "all_traversable_cell_centers_by_connected_region",
        "scene_name": scene.get("name", "unnamed"),
        "robot": {"footprint_width_m": robot.footprint_width_m,
                  "footprint_length_m": robot.footprint_length_m,
                  "safety_margin_m": robot.safety_margin_m},
        "internal_config": {"connectivity": 4,
                            "clearance_method": "half_grid_edt",
                            "component_method": "opencv_connected_components",
                            "unknown_cells_passable": False,
                            "clearance_model": "circumscribed_circle",
                            "assumes_in_place_rotation": True,
                            "region_seed": "topmost_leftmost_traversable_cell",
                            "start_reachability": "not_evaluated_no_start_pose"},
        "grid": {"resolution": grid.resolution, "origin": list(grid.origin),
                 "width": grid.width, "height": grid.height},
        "statistics": {"free_cells": int((~blocked).sum()), "traversable_cells": int(safe.sum()),
                       "region_count": len(regions)},
        "regions": regions,
        "visual_object_coverage": "not_evaluated",
    }


def diagnostic_svg(scene: dict[str, Any], result: dict[str, Any]) -> str:
    """Render safe regions without a navigation route."""
    grid = GridMap.from_scene(scene)
    scale = 36
    pad = 32
    width = grid.width * scale + 2 * pad
    height = grid.height * scale + 2 * pad
    region_for_cell = {tuple(cell): region["id"] for region in result["regions"]
                       for cell in region["cells"]}
    palette = ("#e2f4ed", "#e4edff", "#fff0da", "#f2e8ff")
    display_scale = min(1.0, 1200 / max(width, height))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * display_scale:.0f}" height="{height * display_scale:.0f}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>']
    colors = {"#": "#263447", "?": "#aeb9c7", ".": "#ffffff"}
    for row, line in enumerate(grid.rows):
        fills = []
        for col, symbol in enumerate(line):
            region_id = region_for_cell.get((row, col))
            fills.append(palette[(region_id - 1) % len(palette)] if region_id else colors[symbol])
        # Merge same-color runs on large maps instead of emitting one SVG node per cell.
        left = 0
        for right in range(1, grid.width + 1):
            if right == grid.width or fills[right] != fills[left] or max(grid.width, grid.height) <= 40:
                stroke = 'stroke="#d5dde5" stroke-width="0.7"' if max(grid.width, grid.height) <= 40 else ''
                parts.append(f'<rect x="{pad + left * scale}" y="{pad + row * scale}" '
                             f'width="{(right - left) * scale}" height="{scale}" fill="{fills[left]}" {stroke}/>')
                left = right

    def screen(x: float, y: float) -> tuple[float, float]:
        return (pad + (x - grid.origin[0]) / grid.resolution * scale,
                pad + (grid.height - (y - grid.origin[1]) / grid.resolution) * scale)

    for region in result["regions"]:
        sx, sy = screen(*grid.cell_to_world(tuple(region["cells"][0])))
        parts.append(f'<text x="{sx:.2f}" y="{sy:.2f}" font-size="11" '
                     f'fill="#233044">R{region["id"]}</text>')
    parts.append('<text x="12" y="20" font-size="12" fill="#233044">'
                 'Colored areas: safe connected regions (no start pose supplied)</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main(argv=None, *, scene_path=None, robot_path=None, output_path=None, svg_path=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=scene_path, required=scene_path is None)
    parser.add_argument('--robot', type=Path, default=robot_path, required=robot_path is None)
    parser.add_argument("--output", type=Path, default=output_path, required=output_path is None)
    parser.add_argument("--svg", type=Path, default=svg_path, required=svg_path is None)
    args = parser.parse_args(argv)
    try:
        scene = json.loads(args.scene.read_text(encoding='utf-8'))
        config = json.loads(args.robot.read_text(encoding='utf-8'))
        result = plan(scene, config)
    except (OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, PlanningError) else "invalid_input"
        parser.exit(2, f"规划失败 [{code}]: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.svg.write_text(diagnostic_svg(scene, result) + "\n", encoding="utf-8")
    stats = result["statistics"]
    print(f"安全区域完成：{stats['region_count']} 个连通区域、{stats['traversable_cells']} 个可通行格中心，"
          f"结果 {args.output}，诊断图 {args.svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
