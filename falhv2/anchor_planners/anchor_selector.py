"""Select anchors from qualified candidates; retain the weighted greedy baseline.

Geometry, pipeline orchestration, verification and exports live in separate modules.
The former public imports and script entry point remain available for compatibility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import heapq

from .anchor_candidates import (Station, CandidateResult, generate_candidates,
                                stations_from_regions, yaw_grid, object_heatmaps)
from .observation_model import PointCamera, validate_objects
from .anchor_optimizer import optimize
from .anchor_export import render_svg, viewer_data, save_supports

def select_anchors_greedy(candidates: CandidateResult, stations: list[Station], object_ids: list[str]) -> dict:
    targets = set(object_ids)
    station_by_id = {station.index: station for station in stations}
    station_sets = {i: set().union(*directions.values()) for i, directions in candidates.supports.items()}
    if any(not objects <= targets for objects in station_sets.values()):
        raise ValueError("Coverage contains unknown object IDs")
    if not set(station_sets) <= set(station_by_id):
        raise ValueError("Coverage contains unknown station IDs")
    def action_score(i, yaw, obj):
        return candidates.scores.get(i, {}).get(yaw, {}).get(obj, 1.0)

    # Integer micro-units keep lazy heap updates deterministic and exact.
    station_scores = {i: {obj: round(1_000_000*max(action_score(i, yaw, obj)
                            for yaw, ids in candidates.supports[i].items() if obj in ids))
                          for obj in ids} for i, ids in station_sets.items()}
    reverse = defaultdict(set)
    for i, objects in station_sets.items():
        for obj in objects:
            reverse[obj].add(i)
    remaining = set(targets)
    votes = {i: len(objects) for i, objects in station_sets.items()}
    score_votes = {i: sum(station_scores[i].values()) for i in station_sets}
    heap = [(-score_votes[i], -score, i) for i, score in votes.items() if score]
    heapq.heapify(heap)
    chosen, trace = [], []
    while heap and remaining:
        negative_score, negative, i = heapq.heappop(heap)
        if -negative != votes[i] or -negative_score != score_votes[i]:
            continue
        new = station_sets[i] & remaining
        if not new:
            continue
        chosen.append(i)
        trace.append({"station_index": i, "new_object_ids": sorted(new), "new_object_count": len(new),
                      "weighted_score": sum(station_scores[i][obj] for obj in new)/1_000_000})
        remaining.difference_update(new)
        for obj in new:
            for other in reverse[obj]:
                votes[other] -= 1
                score_votes[other] -= station_scores[other][obj]
                if votes[other] > 0:
                    heapq.heappush(heap, (-score_votes[other], -votes[other], other))
    # Keep an inclusion-minimal station cover, without moving any position.
    counts = Counter(obj for i in chosen for obj in station_sets[i])
    retained = set(chosen)
    prune_order = list(reversed(chosen))
    if candidates.scores:
        prune_order.sort(key=lambda i: sum(station_scores[i].values()))
    for i in prune_order:
        if all(counts[obj] > 1 for obj in station_sets[i]):
            retained.remove(i)
            counts.subtract(station_sets[i])
    covered = targets-remaining
    directions = {(i, yaw): objects for i in sorted(retained)
                  for yaw, objects in sorted(candidates.supports[i].items()) if objects}
    need = set(covered)
    actions = []
    while need:
        # After station selection, reduce photos first, score breaks action ties.
        key = min(directions, key=lambda key: (-len(directions[key] & need),
                  -sum(round(1_000_000*action_score(*key, obj)) for obj in directions[key] & need), key))
        if not (directions[key] & need):
            raise ValueError("Direction selection lost station coverage")
        actions.append(key)
        need.difference_update(directions[key])
    counts = Counter(obj for key in actions for obj in directions[key])
    prune_actions = list(reversed(actions))
    if candidates.scores:
        prune_actions.sort(key=lambda key: sum(round(1_000_000*action_score(*key, obj)) for obj in directions[key]))
    for key in prune_actions:
        if all(counts[obj] > 1 for obj in directions[key]):
            actions.remove(key)
            counts.subtract(directions[key])
    anchors = []
    for i in sorted(retained):
        station = station_by_id[i]
        observations = [{"yaw_index": yaw, "yaw_rad": candidates.yaws[yaw],
                         "object_ids": sorted(directions[(i, yaw)]),
                         "score_by_object": {obj: action_score(i, yaw, obj) for obj in sorted(directions[(i, yaw)])}}
                        for si, yaw in actions if si == i]
        if observations:
            anchors.append({"station_index": i, "region_id": station.region_id,
                            "position": list(station.position), "cell": list(station.cell),
                            "observations": observations})
    actual = {obj for anchor in anchors for action in anchor["observations"] for obj in action["object_ids"]}
    if actual != covered:
        raise ValueError("Final plan failed independent set-union coverage verification")
    return {"status": "empty_targets" if not targets else "complete" if not remaining else "partial",
            "anchor_count": len(anchors), "photo_count": sum(len(a["observations"]) for a in anchors),
            "coverage_fraction": len(actual)/len(targets) if targets else None,
            "covered_object_ids": sorted(actual), "uncovered_object_ids": sorted(remaining),
            "anchors": anchors, "vote_trace": trace,
            "optimality": "heuristic_not_globally_optimal",
            "navigation": "not_planned",
            "start_reachability": "not_evaluated_no_start_pose"}


def assemble_anchors(actions, candidates, stations):
    by_id = {s.index: s for s in stations}
    anchors = []
    for i in sorted({i for i, _ in actions}):
        s = by_id[i]
        observations = []
        for si, y in actions:
            if si != i:
                continue
            ids = sorted(candidates.supports[i][y])
            observations.append(dict(yaw_index=y, yaw_rad=candidates.yaws[y], object_ids=ids,
                score_by_object={o: candidates.scores.get(i, {}).get(y, {}).get(o, 1.0) for o in ids}))
        anchors.append(dict(station_index=i, region_id=s.region_id, position=list(s.position),
                            cell=list(s.cell), observations=observations))
    return anchors


def select_anchors(candidates: CandidateResult, stations: list[Station], object_ids: list[str],
                   *, time_limit_s=None) -> dict:
    """Keep the weighted baseline; minimize station count before photo count."""
    if len({s.index for s in stations}) != len(stations):
        raise ValueError('Station IDs must be unique')
    if not set(candidates.supports) <= set(candidates.station_indices):
        raise ValueError('Support station is outside the declared candidate pool')
    if any(not isinstance(y, int) or not 0 <= y < len(candidates.yaws)
           for ds in candidates.supports.values() for y in ds):
        raise ValueError('Invalid support yaw index')
    baseline = select_anchors_greedy(candidates, stations, object_ids)
    keys = [(a['station_index'], o['yaw_index']) for a in baseline['anchors'] for o in a['observations']]
    solution = optimize(candidates.supports, object_ids, keys, time_limit_s=time_limit_s)
    selected = deepcopy(baseline)
    selected['vote_trace'] = []  # The integer solution has no greedy selection order.
    anchors = assemble_anchors(solution['actions'], candidates, stations)
    selected.update({k: v for k, v in solution.items() if k != 'actions'})
    selected.update(anchors=anchors, anchor_count=solution['selected_cost'][0],
                    photo_count=solution['selected_cost'][1], baseline=baseline,
                    objective_order=['anchor_count', 'photo_count'],
                    optimality=('finite_pool_lexicographic_optimal'
                        if solution['optimization']['lexicographic_optimal'] else 'not_proven'))
    return selected


def plan(scene, robot_config, camera_config, *, yaw_step_deg=None, selection_time_limit_s=None):
    """Compatibility entry point; the pipeline is implemented in pipeline.plan."""
    from .pipeline import plan as run_plan
    return run_plan(scene, robot_config, camera_config, yaw_step_deg=yaw_step_deg,
                    selection_time_limit_s=selection_time_limit_s)


def main():
    """Compatibility CLI; prefer python falhv2/main.py."""
    from .pipeline import main as run
    return run()


if __name__ == '__main__':
    raise SystemExit(main())
