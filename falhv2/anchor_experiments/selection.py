"""Shared dynamic station voting, redundancy deletion and direction selection."""
from collections import Counter, defaultdict
import heapq

from .common import CandidateResult, Station


def select_anchors(candidates: CandidateResult, stations: list[Station], object_ids: list[str]) -> dict:
    targets = set(object_ids)
    station_by_id = {station.index: station for station in stations}
    station_sets = {i: set().union(*directions.values()) for i, directions in candidates.supports.items()}
    if any(not objects <= targets for objects in station_sets.values()):
        raise ValueError("Coverage contains unknown object IDs")
    if not set(station_sets) <= set(station_by_id):
        raise ValueError("Coverage contains unknown station IDs")
    reverse = defaultdict(set)
    for i, objects in station_sets.items():
        for obj in objects:
            reverse[obj].add(i)
    remaining = set(targets)
    votes = {i: len(objects) for i, objects in station_sets.items()}
    heap = [(-score, i) for i, score in votes.items() if score]
    heapq.heapify(heap)
    chosen, trace = [], []
    while heap and remaining:
        negative, i = heapq.heappop(heap)
        if -negative != votes[i]:
            continue
        new = station_sets[i] & remaining
        if not new:
            continue
        chosen.append(i)
        trace.append({"station_index": i, "new_object_ids": sorted(new), "votes": len(new)})
        remaining.difference_update(new)
        for obj in new:
            for other in reverse[obj]:
                votes[other] -= 1
                if votes[other] > 0:
                    heapq.heappush(heap, (-votes[other], other))
    # Keep an inclusion-minimal station cover, without moving any position.
    counts = Counter(obj for i in chosen for obj in station_sets[i])
    retained = set(chosen)
    for i in reversed(chosen):
        if all(counts[obj] > 1 for obj in station_sets[i]):
            retained.remove(i)
            counts.subtract(station_sets[i])
    covered = targets-remaining
    directions = {(i, yaw): objects for i in sorted(retained)
                  for yaw, objects in sorted(candidates.supports[i].items()) if objects}
    need = set(covered)
    actions = []
    while need:
        # Deterministic tie-break by station first-visit order and yaw index.
        key = min(directions, key=lambda key: (-len(directions[key] & need), key))
        if not (directions[key] & need):
            raise ValueError("Direction selection lost station coverage")
        actions.append(key)
        need.difference_update(directions[key])
    counts = Counter(obj for key in actions for obj in directions[key])
    for key in list(reversed(actions)):
        if all(counts[obj] > 1 for obj in directions[key]):
            actions.remove(key)
            counts.subtract(directions[key])
    anchors = []
    for i in sorted(retained):
        station = station_by_id[i]
        observations = [{"yaw_index": yaw, "yaw_rad": candidates.yaws[yaw],
                         "object_ids": sorted(directions[(i, yaw)])}
                        for si, yaw in actions if si == i]
        if observations:
            anchors.append({"station_index": i, "region_id": station.region_id,
                            "position": list(station.position), "cell": list(station.cell),
                            "first_route_index": station.route_indices[0],
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
            "region_transfer": "separate_region_routes_no_inter_region_connection"}


def exact_minimum_station_count(candidates: CandidateResult, object_ids: list[str]) -> int | None:
    """Small-case discrete optimum via bitmask DP; returns None if infeasible."""
    if len(object_ids) > 12:
        raise ValueError("Exact audit is restricted to <=12 targets")
    bits = {obj: 1 << i for i, obj in enumerate(object_ids)}
    masks = {sum(bits[obj] for obj in set().union(*directions.values()))
             for directions in candidates.supports.values()}
    dp = {0: 0}
    for mask in sorted(masks):
        for covered, count in list(dp.items()):
            other = covered | mask
            dp[other] = min(dp.get(other, 10**9), count+1)
    return dp.get((1 << len(bits))-1)


def audit_small_station_bound(candidates: CandidateResult, object_ids: list[str], feasible_count: int) -> dict:
    """Exclude all one/two-station covers; certify a supplied three-station cover.

    Identical station unions can be collapsed for this station-count audit,
    since all declared directions at a station are allowed. No photo optimum
    or continuous-space guarantee follows from this certificate.
    """
    bits = {obj: 1 << i for i, obj in enumerate(object_ids)}
    full = (1 << len(bits))-1
    masks = sorted({sum(bits[obj] for obj in set().union(*d.values())) for d in candidates.supports.values()})
    one = full in masks
    two = one or any(a | b == full for i,a in enumerate(masks) for b in masks[i+1:])
    optimum = 0 if not bits else 1 if one else 2 if two else 3 if feasible_count == 3 else None
    return {'unique_station_coverage_patterns':len(masks), 'one_station_feasible':one,
            'two_stations_feasible':two, 'certified_minimum_station_count':optimum,
            'scope':'finite station/yaw pool; station count only, assuming supplied cover is verified feasible'}
