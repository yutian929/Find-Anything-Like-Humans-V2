"""Lexicographic finite-pool cover: minimize stations, then directional photos.

Geometry and scores are deliberately outside this module. Only already qualified
supports enter the integer models. A validated baseline remains available if a
solver times out, fails, or returns an invalid incumbent.
"""
from __future__ import annotations

import math
import logging
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

LOGGER = logging.getLogger(__name__)


def cost(actions):
    return len({i for i, _ in actions}), len(actions)


def validate_actions(actions, supports, targets):
    keys = list(actions)
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate selected actions')
    covered = set()
    for station, yaw in keys:
        ids = supports.get(station, {}).get(yaw)
        if not ids:
            raise ValueError('Selected action has no qualified support')
        covered.update(ids)
    if covered != set(targets):
        raise ValueError('Selected actions do not cover the declared targets')
    return sorted(keys)


def _photo_cover(station_ids, supports, targets):
    directions = {(i, y): ids for i in sorted(station_ids)
                  for y, ids in sorted(supports[i].items()) if ids}
    remaining, actions = set(targets), []
    while remaining:
        key = min(directions, key=lambda k: (-len(directions[k] & remaining), k))
        if not directions[key] & remaining:
            raise ValueError('Station incumbent lost coverage')
        actions.append(key)
        remaining -= directions[key]
    return validate_actions(actions, supports, targets)


def _finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _binary(result, size):
    if result.x is None:
        return None
    x = np.asarray(result.x, dtype=float)
    if (x.shape != (size,) or not np.isfinite(x).all()
            or np.any(np.abs(x - np.rint(x)) > 1e-6)
            or np.any(x < -1e-6) or np.any(x > 1 + 1e-6)):
        raise ValueError('Solver incumbent is not binary')
    return np.rint(x).astype(bool)


def _equivalent_cover_model(supports):
    """Exact reduction for unit photo/station costs, with original action mapping.

    At one station a photo whose coverage is a subset of another is unnecessary.
    Stations with identical families of maximal photo covers are interchangeable
    (not merely stations with equal coverage unions!). Multiple equivalent
    stations can be consolidated at one, never increasing either resource cost.
    This reduction is invalid for objectives involving yaw, score or travel cost.
    The original supports/heatmaps remain untouched.
    """
    representatives, reduced, action_map = {}, {}, {}
    for i, ds in sorted(supports.items()):
        by_cover = {}
        for y, ids in sorted(ds.items()):
            if ids:
                by_cover.setdefault(frozenset(ids), y)
        maximal = sorted((ids for ids in by_cover if not any(ids < other for other in by_cover)),
                         key=lambda ids: tuple(sorted(ids)))
        if not maximal:
            continue
        signature = tuple(tuple(sorted(ids)) for ids in maximal)
        representative = representatives.setdefault(signature, i)
        if representative == i:
            reduced[i] = {by_cover[ids]: set(ids) for ids in maximal}
        for y, ids in ds.items():
            if ids:
                ry = min(ry for ry, cover in reduced[representative].items() if set(ids) <= cover)
                action_map[(i, y)] = (representative, ry)
    return reduced, action_map


def optimize(supports, object_ids, baseline_actions, *, time_limit_s=None):
    """Return action keys plus auditable solver metadata; never worsen baseline.

    time_limit_s budgets the sum of the two milp calls, excluding model assembly.
    Uncovered targets retain baseline partial coverage and skip optimization.
    """
    if time_limit_s is not None and (not math.isfinite(time_limit_s) or time_limit_s <= 0):
        raise ValueError('selection time limit must be finite and positive')
    targets = set(object_ids)
    if len(targets) != len(object_ids):
        raise ValueError('Target IDs must be unique')
    all_ids = set().union(*(ids for ds in supports.values() for ids in ds.values()))
    if not all_ids <= targets:
        raise ValueError('Supports contain unknown target IDs')
    best = validate_actions(baseline_actions, supports, all_ids)
    baseline_cost = cost(best)
    metadata = dict(scope='finite_station_yaw_pool',
                    stage1={'status': 'not_run', 'proven_optimal': False},
                    stage2={'status': 'not_run', 'proven_optimal': False},
                    lexicographic_optimal=False, fallback_reason=None,
                    time_limit_s=time_limit_s)
    source = 'weighted_greedy_fallback'

    def finish():
        return dict(actions=best, baseline_cost=list(baseline_cost),
                    selected_cost=list(cost(best)), selection_method=source,
                    comparison_to_baseline='improved' if cost(best) < baseline_cost else 'equal',
                    optimization=metadata)

    if not targets or all_ids != targets:
        metadata['fallback_reason'] = 'empty_targets' if not targets else 'uncovered_targets'
        return finish()

    original_station_count = sum(any(ds.values()) for ds in supports.values())
    original_action_count = sum(bool(ids) for ds in supports.values() for ids in ds.values())
    supports, action_map = _equivalent_cover_model(supports)
    mapped = validate_actions(sorted({action_map[key] for key in best}), supports, targets)
    if cost(mapped) < cost(best):
        best, source = mapped, 'lexicographic_milp'
    stations = sorted(i for i, ds in supports.items() if any(ds.values()))
    station_index = {i: j for j, i in enumerate(stations)}
    objects = sorted(targets)
    object_index = {obj: j for j, obj in enumerate(objects)}
    station_sets = {i: set().union(*supports[i].values()) for i in stations}
    actions = sorted((i, y) for i in stations for y, ids in supports[i].items() if ids)
    ns, na, no = len(stations), len(actions), len(objects)
    metadata['exact_reduction'] = dict(rule='maximal_photo_covers_and_equal_cover_families',
        original_stations=original_station_count, original_actions=original_action_count,
        model_stations=ns, model_actions=na,
        guarantee='preserves_global_lexicographic_resource_optimum')
    LOGGER.info('Exact cover model: %d/%d stations, %d/%d actions',
                ns, original_station_count, na, original_action_count)
    elapsed = 0.0

    def solve(stage, c, rows, columns, values, lower, upper):
        nonlocal elapsed
        entry = metadata[stage]
        remaining = None if time_limit_s is None else time_limit_s - elapsed
        if remaining is not None and remaining <= 0:
            entry.update(status='time_limit', message='Shared MILP budget exhausted')
            return None
        matrix = coo_matrix((values, (rows, columns)), shape=(len(lower), len(c))).tocsc()
        options = {'mip_rel_gap': 0.0}
        if remaining is not None:
            options['time_limit'] = remaining
        start = perf_counter()
        LOGGER.info('%s: solving %d variables, %d constraints', stage, len(c), len(lower))
        try:
            result = milp(c=np.asarray(c, dtype=float), integrality=np.ones(len(c)),
                          bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix, lower, upper), options=options)
        finally:
            duration = perf_counter() - start
            elapsed += duration
            entry['elapsed_s'] = duration
        entry.update(status={0: 'optimal', 1: 'limit_reached', 2: 'infeasible',
                             3: 'unbounded', 4: 'solver_error'}.get(result.status, 'solver_error'),
                     solver_status=int(result.status), message=str(result.message),
                     objective=_finite(result.fun), lower_bound=_finite(getattr(result, 'mip_dual_bound', None)),
                     gap=_finite(getattr(result, 'mip_gap', None)))
        LOGGER.info('%s: %s, objective=%s, lower_bound=%s, %.3fs',
                    stage, entry['status'], entry['objective'], entry['lower_bound'], duration)
        if result.status not in (0, 1):
            raise ValueError('Solver reported failure on a baseline-feasible model')
        return result

    def certify(entry, objective):
        if entry['objective'] is None or abs(entry['objective'] - objective) > 1e-6:
            raise ValueError('Solver objective does not match decoded actions')
        entry['proven_optimal'] = bool(entry['status'] == 'optimal'
            and entry['gap'] is not None and entry['gap'] <= 1e-9
            and entry['lower_bound'] is not None
            and abs(entry['lower_bound'] - objective) <= 1e-6)

    stage = 'stage1'
    try:
        rr, cc, vv = [], [], []
        for i in stations:
            for obj in sorted(station_sets[i]):
                rr.append(object_index[obj]); cc.append(station_index[i]); vv.append(1)
            rr.append(no); cc.append(station_index[i]); vv.append(1)
        result = solve(stage, np.ones(ns), rr, cc, vv,
                       [1] * no + [0], [np.inf] * no + [baseline_cost[0]])
        if result is not None:
            bits = _binary(result, ns)
            if bits is not None:
                selected = [stations[j] for j in np.flatnonzero(bits)]
                if not selected or len(selected) > baseline_cost[0]:
                    raise ValueError('Invalid station count')
                candidate = _photo_cover(selected, supports, targets)
                certify(metadata[stage], len(selected))
                if metadata[stage]['proven_optimal'] and cost(candidate)[0] != len(selected):
                    raise ValueError('Proven station count contradicts feasible photo cover')
                if cost(candidate) < cost(best):
                    best, source = candidate, 'lexicographic_milp'
        # Even without a station certificate, improve photos at the best known K.
        k = cost(best)[0]
        stage = 'stage2'
        metadata[stage]['fixed_anchor_count'] = k
        rr, cc, vv = [], [], []
        lower, upper = [1] * no, [np.inf] * no
        station_actions = {i: [] for i in stations}
        for j, (i, y) in enumerate(actions):
            a = ns + j
            station_actions[i].append(a)
            for obj in sorted(supports[i][y]):
                rr.append(object_index[obj]); cc.append(a); vv.append(1)
            row = len(lower)
            rr.extend([row, row]); cc.extend([a, station_index[i]]); vv.extend([1, -1])
            lower.append(-np.inf); upper.append(0)
        for i in stations:
            row = len(lower)
            for col, value in [(station_index[i], 1), *[(a, -1) for a in station_actions[i]]]:
                rr.append(row); cc.append(col); vv.append(value)
            lower.append(-np.inf); upper.append(0)
        # These station-cover rows are implied for integer solutions, but are
        # stronger than individual y<=x links in the LP relaxation: multiple
        # fractional photos at one station must not multiply its coverage.
        for obj in objects:
            row = len(lower)
            for i in stations:
                if obj in station_sets[i]:
                    rr.append(row); cc.append(station_index[i]); vv.append(1)
            lower.append(1); upper.append(np.inf)
        row = len(lower)
        rr.extend([row] * ns); cc.extend(range(ns)); vv.extend([1] * ns)
        lower.append(k); upper.append(k)
        row = len(lower)
        rr.extend([row] * na); cc.extend(range(ns, ns + na)); vv.extend([1] * na)
        lower.append(0); upper.append(cost(best)[1])
        result = solve(stage, [0] * ns + [1] * na, rr, cc, vv, lower, upper)
        if result is not None:
            bits = _binary(result, ns + na)
            if bits is not None:
                candidate = validate_actions([actions[j] for j in np.flatnonzero(bits[ns:])], supports, targets)
                selected = {stations[j] for j in np.flatnonzero(bits[:ns])}
                if selected != {i for i, _ in candidate} or len(selected) != k or len(candidate) > len(best):
                    raise ValueError('Invalid station/photo association or objective bound')
                certify(metadata[stage], len(candidate))
                if cost(candidate) < cost(best):
                    best = candidate
                source = 'lexicographic_milp'
        metadata['lexicographic_optimal'] = bool(
            metadata['stage1']['proven_optimal'] and metadata['stage2']['proven_optimal'])
        if not metadata['lexicographic_optimal']:
            metadata['fallback_reason'] = 'optimization_not_proven_complete'
    except (ValueError, RuntimeError, TypeError, AttributeError, OverflowError) as exc:
        metadata[stage].update(status='invalid_or_failed', proven_optimal=False, error=str(exc))
        metadata['fallback_reason'] = f'{stage}: {exc}'
        metadata['lexicographic_optimal'] = False
    return finish()
