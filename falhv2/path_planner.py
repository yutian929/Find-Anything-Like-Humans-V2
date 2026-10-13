"""A1: find all safe grid regions and plan one coverage route per region.

This covers traversable navigation cell centers, not visual objects. All movement
is four-connected, so consecutive cell centers are joined by a free-cell edge.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


Cell = tuple[int, int]  # (row from top, column from left)
NEIGHBORS = ((-1, 0), (0, 1), (1, 0), (0, -1))


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


def _reachable(start: Cell, allowed: set[Cell]) -> set[Cell]:
    reached = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for dr, dc in NEIGHBORS:
            next_cell = (row + dr, col + dc)
            if next_cell in allowed and next_cell not in reached:
                reached.add(next_cell)
                queue.append(next_cell)
    return reached


def _route_to_unvisited(current: Cell, unvisited: set[Cell], allowed: set[Cell]) -> list[Cell]:
    queue = deque([current])
    parent: dict[Cell, Cell | None] = {current: None}
    target = None
    while queue:
        cell = queue.popleft()
        if cell in unvisited:
            target = cell
            break
        for dr, dc in NEIGHBORS:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if neighbor in allowed and neighbor not in parent:
                parent[neighbor] = cell
                queue.append(neighbor)
    if target is None:
        raise PlanningError("disconnected", "可达区域内存在无法连接的格子")
    route = []
    while target != current:
        route.append(target)
        target = parent[target]
    return list(reversed(route))


def _region_route(grid: GridMap, cells: set[Cell], region_id: int) -> dict[str, Any]:
    start = min(cells)
    unvisited = cells - {start}
    route_cells = [start]
    current = start
    while unvisited:
        segment = _route_to_unvisited(current, unvisited, cells)
        route_cells.extend(segment)
        unvisited.difference_update(segment)
        current = route_cells[-1]

    positions = [grid.cell_to_world(cell) for cell in route_cells]
    length = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
    waypoints = []
    for index, (px, py) in enumerate(positions):
        if index + 1 < len(positions):
            nx, ny = positions[index + 1]
            heading = math.atan2(ny - py, nx - px)
        elif index:
            heading = waypoints[-1]["yaw_rad"]
        else:
            heading = 0.0
        waypoints.append({"x_m": px, "y_m": py, "yaw_rad": heading})
    return {"id": region_id, "cells": [list(cell) for cell in sorted(cells)],
            "route_cells": [list(cell) for cell in route_cells],
            "waypoints": waypoints, "path_length_m": length}


def plan(scene: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    grid = GridMap.from_scene(scene)
    robot = Robot.from_config(config)
    free = {(row, col) for row, line in enumerate(grid.rows)
            for col, symbol in enumerate(line) if symbol == "."}
    allowed = {cell for cell in free
               if grid.clearance_at(*grid.cell_to_world(cell)) + 1e-10 >= robot.clearance_m}
    if not allowed:
        raise PlanningError("no_traversable_cells", "机器人足迹及安全距离下，没有可通行的空闲格中心")
    remaining = set(allowed)
    regions = []
    while remaining:
        cells = _reachable(min(remaining), remaining)
        regions.append(_region_route(grid, cells, len(regions) + 1))
        remaining.difference_update(cells)

    return {
        "status": "complete",
        "scope": "all_traversable_cell_centers_by_connected_region",
        "scene_name": scene.get("name", "unnamed"),
        "robot": {"footprint_width_m": robot.footprint_width_m,
                  "footprint_length_m": robot.footprint_length_m,
                  "safety_margin_m": robot.safety_margin_m},
        "internal_config": {"connectivity": 4, "path_method": "nearest_unvisited_shortest_grid_route",
                            "unknown_cells_passable": False, "waypoint_spacing_m": grid.resolution,
                            "clearance_model": "circumscribed_circle",
                            "assumes_in_place_rotation": True,
                            "region_seed": "topmost_leftmost_traversable_cell"},
        "grid": {"resolution": grid.resolution, "origin": list(grid.origin),
                 "width": grid.width, "height": grid.height},
        "statistics": {"free_cells": len(free), "traversable_cells": len(allowed),
                       "region_count": len(regions),
                       "visited_traversable_cells": sum(len(set(map(tuple, region["route_cells"])))
                                                        for region in regions),
                       "total_path_length_m": sum(region["path_length_m"] for region in regions),
                       "total_waypoint_count": sum(len(region["waypoints"]) for region in regions)},
        "regions": regions,
        "visual_object_coverage": "not_evaluated",
    }


def diagnostic_svg(scene: dict[str, Any], result: dict[str, Any]) -> str:
    """Render all traversable regions and their routes without dependencies."""
    grid = GridMap.from_scene(scene)
    scale = 36
    pad = 32
    width = grid.width * scale + 2 * pad
    height = grid.height * scale + 2 * pad
    region_for_cell = {tuple(cell): region["id"] for region in result["regions"]
                       for cell in region["cells"]}
    palette = ("#e2f4ed", "#e4edff", "#fff0da", "#f2e8ff")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>']
    colors = {"#": "#263447", "?": "#aeb9c7", ".": "#ffffff"}
    for row, line in enumerate(grid.rows):
        for col, symbol in enumerate(line):
            region_id = region_for_cell.get((row, col))
            fill = palette[(region_id - 1) % len(palette)] if region_id else colors[symbol]
            parts.append(f'<rect x="{pad + col * scale}" y="{pad + row * scale}" '
                         f'width="{scale}" height="{scale}" fill="{fill}" '
                         'stroke="#d5dde5" stroke-width="0.7"/>')

    def screen(x: float, y: float) -> tuple[float, float]:
        return (pad + (x - grid.origin[0]) / grid.resolution * scale,
                pad + (grid.height - (y - grid.origin[1]) / grid.resolution) * scale)

    line_palette = ("#147d9d", "#5458b4", "#bd7424", "#8b48a0")
    for region in result["regions"]:
        points = [screen(p["x_m"], p["y_m"]) for p in region["waypoints"]]
        color = line_palette[(region["id"] - 1) % len(line_palette)]
        if len(points) > 1:
            coords = " ".join(f"{sx:.2f},{sy:.2f}" for sx, sy in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                         'stroke-width="2.5" stroke-linejoin="round" opacity="0.85"/>')
        sx, sy = points[0]
        parts.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="5" fill="{color}" '
                     'stroke="white" stroke-width="2"/>')
        parts.append(f'<text x="{sx + 6:.2f}" y="{sy - 6:.2f}" font-size="11" '
                     f'fill="{color}">R{region["id"]}</text>')
    parts.append('<text x="12" y="20" font-size="12" fill="#233044">'
                 'Colored area: traversable region  Line: region coverage route  Dot: algorithmic first cell</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path(__file__).parent / "outputs/map.json")
    parser.add_argument("--robot", type=Path,
                        default=Path(__file__).parent / "config/robot.example.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs/path_plan.json")
    parser.add_argument("--svg", type=Path, default=Path(__file__).parent / "outputs/path_diagnostic.svg")
    args = parser.parse_args()
    try:
        scene = json.loads(args.scene.read_text(encoding="utf-8"))
        config = json.loads(args.robot.read_text(encoding="utf-8"))
        result = plan(scene, config)
    except (OSError, json.JSONDecodeError, PlanningError) as exc:
        code = exc.code if isinstance(exc, PlanningError) else "invalid_input"
        parser.exit(2, f"规划失败 [{code}]: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.svg.write_text(diagnostic_svg(scene, result) + "\n", encoding="utf-8")
    stats = result["statistics"]
    print(f"路径完成：{stats['region_count']} 个连通区域、{stats['traversable_cells']} 个可通行格中心，"
          f"路线总长 {stats['total_path_length_m']:.2f} m；"
          f"结果 {args.output}，诊断图 {args.svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
