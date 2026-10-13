"""Comparable full-cell coverage candidates; no robot start/end is an input.

scanline_sweep is a local row-interval decomposition prototype, NOT the IPA
implementation. estc_upstream calls the unmodified LS-MCPP source checkout.
"""

from collections import defaultdict
import heapq
import importlib
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_planner import NEIGHBORS, _route_to_unvisited


def greedy(cells):
    current = min(cells)
    unvisited = cells - {current}
    path = [current]
    while unvisited:
        segment = _route_to_unvisited(current, unvisited, cells)
        path.extend(segment)
        unvisited.difference_update(segment)
        current = path[-1]
    return path


def tree_walk(cells):
    """Iterative DFS with backtracking; linear traversal, not classical STC."""
    root = min(cells)
    seen = {root}
    stack = [(root, iter(NEIGHBORS))]
    path = [root]
    while stack:
        cell, directions = stack[-1]
        for dr, dc in directions:
            other = (cell[0] + dr, cell[1] + dc)
            if other in cells and other not in seen:
                seen.add(other)
                path.append(other)
                stack.append((other, iter(NEIGHBORS)))
                break
        else:
            stack.pop()
            if stack:
                path.append(stack[-1][0])
    return path


def astar(start, goal, cells):
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
    raise ValueError("Disconnected route request")


def decompose(cells):
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


def scanline_one_axis(cells):
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


def scanline_sweep(cells):
    horizontal = scanline_one_axis(cells)
    vertical = [(c, r) for r, c in scanline_one_axis({(c, r) for r, c in cells})]
    return min([horizontal, vertical], key=lambda path: (len(path), turns(path)[1]))


def load_estc(checkout):
    """Map the upstream setup.py package layout without installing its solvers."""
    checkout = Path(checkout)
    for name, directory in [("lsmcpp", checkout / "mcpp"), ("lsmcpp.benchmark", checkout / "benchmark")]:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        sys.modules[name] = module
    graph = importlib.import_module("lsmcpp.graph")
    planner = importlib.import_module("lsmcpp.estc")
    import networkx as nx

    def run(cells):
        if len(cells) == 1:
            return list(cells)
        # Quarter-grid coordinates required by upstream contract/decomp.
        positions = {cell: ((cell[1] - 0.5) / 2, (cell[0] - 0.5) / 2) for cell in sorted(cells)}
        reverse = {p: cell for cell, p in positions.items()}
        network = nx.Graph()
        for cell, p in positions.items():
            network.add_node(p, terrain_weight=0)
            for dr, dc in ((-1, 0), (0, -1)):
                other = (cell[0]+dr, cell[1]+dc)
                if other in cells:
                    network.add_edge(p, positions[other], weight=1)
        dec = graph.contract(network)
        root = dec.undecomp(positions[min(cells)])
        route = planner.ExtSTCPlanner.plan(root, dec)
        return [reverse[p] for p in route]
    return run


def open_route(route):
    """Remove a closing edge when unnecessary; endpoints are unconstrained."""
    if len(route) > 1 and route[0] == route[-1]:
        route = route[:-1]
    return route
