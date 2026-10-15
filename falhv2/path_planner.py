"""A1: find all safe grid regions and plan one coverage route per region.

This covers traversable navigation cell centers, not visual objects. All movement
is four-connected, so consecutive cell centers are joined by a free-cell edge.
Uses NumPy/SciPy EDT for obstacle inflation, interval decomposition and zigzag
sweeps with A* connectors, and OpenCV for four-connected components.
No prescribed start/end or experiment modules needed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    from . import config as settings
    from .config import read_input
except ImportError:
    import config as settings
    from config import read_input


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


def astar(start: Cell, goal: Cell, cells: set[Cell]) -> list[Cell]:
    if start == goal:
        return [start]
    distance = {start: 0}
    parent = {}
    heap = [(abs(start[0]-goal[0]) + abs(start[1]-goal[1]), 0, start)]
    while heap:
        _, cost, cell = heapq.heappop(heap)
        if cost != distance[cell]:
            continue
        if cell == goal:
            path = [cell]
            while cell != start:
                cell = parent[cell]
                path.append(cell)
            return path[::-1]
        for dr, dc in NEIGHBORS:
            other = (cell[0] + dr, cell[1] + dc)
            if other in cells and cost + 1 < distance.get(other, float("inf")):
                distance[other] = cost + 1
                parent[other] = cell
                h = abs(other[0]-goal[0]) + abs(other[1]-goal[1])
                heapq.heappush(heap, (cost+1+h, cost+1, other))
    raise PlanningError("disconnected", "分区之间无法在安全区域内连接")


def decompose(cells: set[Cell]) -> list[list[tuple[int, int, int]]]:
    """Continue a cell only across one-to-one overlapping row intervals."""
    by_row = defaultdict(list)
    for r, c in sorted(cells):
        by_row[r].append(c)
    groups = []
    previous = []
    previous_row = None
    for row, columns in sorted(by_row.items()):
        runs = []
        left = right = columns[0]
        for col in columns[1:]:
            if col == right + 1:
                right = col
            else:
                runs.append((row, left, right))
                left = right = col
        runs.append((row, left, right))
        if previous_row != row - 1:
            previous = []
        links = [[i for i, (_, lo, hi, _) in enumerate(previous)
                  if max(lo, run[1]) <= min(hi, run[2])] for run in runs]
        degrees = defaultdict(int)
        for matches in links:
            for i in matches:
                degrees[i] += 1
        current = []
        for run, matches in zip(runs, links):
            if len(matches) == 1 and degrees[matches[0]] == 1:
                group = previous[matches[0]][3]
            else:
                group = len(groups)
                groups.append([])
            groups[group].append(run)
            current.append((*run, group))
        previous, previous_row = current, row
    return groups


def row_sweep(runs, reverse_rows, first_right):
    runs = runs[::-1] if reverse_rows else runs
    path = []
    for index, (row, left, right) in enumerate(runs):
        go_right = first_right if index % 2 == 0 else not first_right
        start, stop, step = (left, right, 1) if go_right else (right, left, -1)
        if path:
            prev_row, prev_col = path[-1]
            prev_left, prev_right = runs[index-1][1:]
            crossing = max(max(left, prev_left), min(min(right, prev_right), prev_col))
            move = 1 if crossing >= prev_col else -1
            path.extend((prev_row, c) for c in range(prev_col+move, crossing+move, move))
            path.append((row, crossing))
            move = 1 if start >= crossing else -1
            path.extend((row, c) for c in range(crossing+move, start+move, move))
            path.extend((row, c) for c in range(start+step, stop+step, step))
        else:
            path.extend((row, c) for c in range(start, stop+step, step))
    return path


def scanline_one_axis(cells: set[Cell]) -> list[Cell]:
    groups = decompose(cells)
    candidates = {i: [row_sweep(runs, reverse, direction)
                      for reverse in (False, True) for direction in (True, False)]
                  for i, runs in enumerate(groups)}
    path = [min(cells)]
    while candidates:
        current = path[-1]
        _, group, variant = min((abs(paths[j][0][0]-current[0]) + abs(paths[j][0][1]-current[1]), i, j)
                               for i, paths in candidates.items() for j in range(4))
        sweep = candidates.pop(group)[variant]
        path.extend(astar(current, sweep[0], cells)[1:])
        path.extend(sweep[1:])
    return path


def turns(path):
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    events = quarter_turns = 0
    for a, b in zip(directions, directions[1:]):
        if a != b:
            events += 1
            quarter_turns += 2 if a == (-b[0], -b[1]) else 1
    return events, quarter_turns


def scanline_sweep(cells: set[Cell]) -> list[Cell]:
    """Compare complete horizontal/vertical sweeps by length, then turns."""
    horizontal = scanline_one_axis(cells)
    vertical = [(c, r) for r, c in scanline_one_axis({(c, r) for r, c in cells})]
    return min([horizontal, vertical], key=lambda path: (len(path), turns(path)[1]))


def _region_route(grid: GridMap, cells: set[Cell], region_id: int) -> dict[str, Any]:
    route_cells = scanline_sweep(cells)
    if len(route_cells) > 1 and route_cells[0] == route_cells[-1]:
        route_cells = route_cells[:-1]

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
    regions = []
    for cells in components:
        region = _region_route(grid, cells, len(regions) + 1)
        route = np.asarray(region["route_cells"], dtype=int)
        if set(map(tuple, route)) != cells or (len(route) > 1 and
                not np.all(np.abs(np.diff(route, axis=0)).sum(axis=1) == 1)):
            raise PlanningError("invalid_route", "覆盖路线不完整或存在不连续移动")
        midpoints = route[:-1] + route[1:] + 1
        if not np.all(dense[midpoints[:, 0], midpoints[:, 1]] + 1e-10 >= robot.clearance_m):
            raise PlanningError("unsafe_route", "覆盖路线不满足安全距离")
        regions.append(region)

    return {
        "status": "complete",
        "scope": "all_traversable_cell_centers_by_connected_region",
        "scene_name": scene.get("name", "unnamed"),
        "robot": {"footprint_width_m": robot.footprint_width_m,
                  "footprint_length_m": robot.footprint_length_m,
                  "safety_margin_m": robot.safety_margin_m},
        "internal_config": {"connectivity": 4, "path_method": "scanline_sweep",
                            "clearance_method": "half_grid_edt",
                            "component_method": "opencv_connected_components",
                            "unknown_cells_passable": False, "waypoint_spacing_m": grid.resolution,
                            "clearance_model": "circumscribed_circle",
                            "assumes_in_place_rotation": True,
                            "region_seed": "topmost_leftmost_traversable_cell"},
        "grid": {"resolution": grid.resolution, "origin": list(grid.origin),
                 "width": grid.width, "height": grid.height},
        "statistics": {"free_cells": int((~blocked).sum()), "traversable_cells": int(safe.sum()),
                       "region_count": len(regions),
                       "visited_traversable_cells": sum(len(set(map(tuple, region["route_cells"])))
                                                        for region in regions),
                       "total_path_length_m": sum(region["path_length_m"] for region in regions),
                       "total_waypoint_count": sum(len(region["waypoints"]) for region in regions),
                       "quarter_turns": sum(turns(region["route_cells"])[1] for region in regions)},
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

    line_palette = ("#147d9d", "#5458b4", "#bd7424", "#8b48a0")
    for region in result["regions"]:
        # SVG needs only corners; keep all original execution waypoints in JSON.
        route = region["route_cells"]
        indices = [0]
        for i in range(1, len(route) - 1):
            before = (route[i][0] - route[i-1][0], route[i][1] - route[i-1][1])
            after = (route[i+1][0] - route[i][0], route[i+1][1] - route[i][1])
            if before != after:
                indices.append(i)
        if len(route) > 1:
            indices.append(len(route) - 1)
        points = [screen(region["waypoints"][i]["x_m"], region["waypoints"][i]["y_m"]) for i in indices]
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
    parser.add_argument('--scene', type=Path, default=settings.MAP_PATH)
    parser.add_argument('--robot', type=Path, default=settings.ROBOT_PATH)
    parser.add_argument("--output", type=Path, default=settings.PATH_PLAN_PATH)
    parser.add_argument("--svg", type=Path, default=settings.PATH_SVG_PATH)
    args = parser.parse_args()
    try:
        scene = read_input(args.scene)
        config = read_input(args.robot)
        result = plan(scene, config)
    except (OSError, ValueError) as exc:
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
