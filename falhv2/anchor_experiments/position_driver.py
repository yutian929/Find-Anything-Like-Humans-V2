"""Path-constrained station sampling followed by exact point visibility tests.

Uniform and random sampling grow a nested pool. Missing coverage may trigger
more samples, up to the finite A1 station pool; sampled coordinates never move.
"""

from __future__ import annotations

import heapq
from itertools import islice
from time import perf_counter
from typing import Iterator

import numpy as np

from .common import CandidateResult, PointCamera, Station, validate_objects, yaw_grid


def _uniform_order(count: int, initial_count: int, endpoints: list[int]) -> Iterator[int]:
    """Spread the initial samples, then fill the largest remaining index gaps."""
    if count == 0:
        return
    initial = min(initial_count, count)
    # Include both ends whenever at least two initial samples are requested.
    first = list(dict.fromkeys([
        *endpoints,
        *np.linspace(0, count - 1, initial).round().astype(int).tolist(),
    ]))
    selected = sorted(set(first))
    yield from first
    gaps: list[tuple[int, int, int]] = []

    def add_gap(left: int, right: int) -> None:
        if right - left > 1:
            heapq.heappush(gaps, (-(right - left - 1), left, right))

    boundaries = [-1, *selected, count]
    for left, right in zip(boundaries, boundaries[1:]):
        add_gap(left, right)
    while gaps:
        _, left, right = heapq.heappop(gaps)
        middle = (left + right) // 2
        yield middle
        add_gap(left, middle)
        add_gap(middle, right)


def generate(
    stations: list[Station],
    objects,
    camera: PointCamera,
    *,
    yaw_step_deg: float = 30.0,
    strategy: str = "uniform",
    initial_count: int = 64,
    seed: int = 0,
    refine: bool = True,
    max_stations: int | None = None,
    station_indices: list[int] | None = None,
) -> CandidateResult:
    """Generate support by evaluating sampled station/yaw pairs against objects.

    ``objects`` contains dictionaries with ``id`` and world xyz ``position``.
    ``station_indices``
    denotes exact Station.index values and bypasses sampling/refinement. Full
    exhaustive coverage still means coverage under the discrete yaw grid only.
    Uniform sampling starts with each region's first/last supplied station;
    this may exceed initial_count. A hard max_stations cap takes precedence,
    and any omitted endpoints are explicitly reported in stats.
    """
    started = perf_counter()
    objects = validate_objects(objects)
    if strategy not in {"uniform", "random", "exhaustive"}:
        raise ValueError("strategy must be uniform, random, or exhaustive")
    if initial_count < 1:
        raise ValueError("initial_count must be positive")
    if max_stations is not None and max_stations < 0:
        raise ValueError("max_stations must be nonnegative")
    by_id = {station.index: station for station in stations}
    if len(by_id) != len(stations):
        raise ValueError("station indices must be unique")
    yaws = yaw_grid(yaw_step_deg)
    fixed = station_indices is not None
    endpoints: list[int] = []
    if fixed:
        order = list(dict.fromkeys(station_indices))
        if any(index not in by_id for index in order):
            raise ValueError("station_indices contains an unknown station index")
    elif strategy == "random":
        order = [stations[i].index for i in np.random.default_rng(seed).permutation(len(stations))]
    elif strategy == "exhaustive":
        order = [station.index for station in stations]
    else:
        region_positions: dict[int, tuple[int, int]] = {}
        for position, station in enumerate(stations):
            first, _ = region_positions.get(station.region_id, (position, position))
            region_positions[station.region_id] = (first, position)
        endpoints = list(dict.fromkeys(
            endpoint for positions in region_positions.values()
            for endpoint in (positions[0], positions[-1])
        ))
        order = (stations[i].index for i in _uniform_order(len(stations), initial_count, endpoints))
    pool_count = len(order) if fixed else len(stations)
    if not fixed and max_stations is not None:
        pool_count = min(pool_count, max_stations)
    order = iter(order)

    supports: dict[int, dict[int, set[str]]] = {}
    evaluated: list[int] = []
    covered: set[str] = set()
    pairs = 0
    initial_actual = 0

    def evaluate(batch: list[int]) -> None:
        nonlocal pairs
        # Bounded chunks avoid a full N_stations x N_yaws x N_objects tensor.
        for start in range(0, len(batch), 256):
            indices = batch[start:start + 256]
            positions = np.repeat(
                np.asarray([by_id[index].position for index in indices], dtype=float),
                len(yaws), axis=0,
            )
            angles = np.tile(np.asarray(yaws, dtype=float), len(indices))
            for obj in objects:
                object_id = obj["id"]
                mask = np.asarray(camera.visible(positions, angles, obj["position"]), dtype=bool)
                pairs += len(positions)
                for offset in np.flatnonzero(mask):
                    index = indices[int(offset) // len(yaws)]
                    yaw_index = int(offset) % len(yaws)
                    supports.setdefault(index, {}).setdefault(yaw_index, set()).add(object_id)
                    covered.add(object_id)
            evaluated.extend(indices)

    if not objects:
        reason = "no_objects"
    elif not pool_count:
        reason = "empty_candidate_pool" if not stations or fixed else "station_budget_exhausted"
    else:
        target = pool_count if fixed or strategy == "exhaustive" else min(max(initial_count, len(endpoints)), pool_count)
        initial_actual = target
        evaluate(list(islice(order, target)))
        while not fixed and strategy != "exhaustive" and refine and len(covered) < len(objects) and target < pool_count:
            next_target = min(pool_count, max(target + 1, target * 2))
            evaluate(list(islice(order, next_target - target)))
            target = next_target
        if fixed:
            reason = "fixed_candidate_pool_evaluated"
        elif len(covered) == len(objects):
            reason = "all_objects_supported"
        elif not refine and strategy != "exhaustive":
            reason = "refinement_disabled"
        elif pool_count < len(stations):
            reason = "station_budget_exhausted"
        else:
            reason = "candidate_pool_exhausted"

    return CandidateResult(
        station_indices=evaluated,
        yaws=yaws,
        supports=supports,
        stats={
            "method": "position_driven",
            "strategy": strategy,
            "seed": seed,
            "refine": refine,
            "fixed_candidate_pool": fixed,
            "available_station_count": len(stations),
            "requested_initial_station_count": initial_count,
            "initial_station_count": initial_actual,
            "final_station_count": len(evaluated),
            "yaw_count": len(yaws),
            "evaluated_point_pose_pairs": pairs,
            "covered_object_count": len(covered),
            "total_object_count": len(objects),
            "region_endpoint_count": len(endpoints),
            "evaluated_region_endpoint_count": len({stations[i].index for i in endpoints} & set(evaluated)),
            "endpoint_budget_limited": bool(endpoints and max_stations is not None and max_stations < len(endpoints)),
            "stop_reason": reason,
            "generation_seconds": perf_counter() - started,
        },
    )
