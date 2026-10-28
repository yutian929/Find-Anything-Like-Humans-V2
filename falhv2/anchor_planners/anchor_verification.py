"""Audit selected observations, repair sampled coverage gaps and verify fallback plans."""
from copy import deepcopy
from dataclasses import replace
import logging
import math
from time import perf_counter

import numpy as np

from ..config import defaults as settings
from .observation_model import BoxObserver
from .anchor_optimizer import optimize, find_cover_at_cost
from .anchor_selector import assemble_anchors

def audit_observations(selected, objects, observer):
    """Audit all target/photo pairs; keep dense metrics separate from original q."""
    covered, rejected = set(), []
    started = perf_counter()
    count = 0
    for anchor in selected['anchors']:
        for action in anchor['observations']:
            metrics = {o['id']: observer.evaluate(i, [anchor['position']], [action['yaw_rad']])[0]
                       for i, o in enumerate(objects)}
            action['high_density_object_metrics'] = metrics
            visible = {oid for oid, row in metrics.items() if row['qualified']}
            covered.update(visible)
            count += len(objects)
            rejected.extend(dict(station_index=anchor['station_index'], yaw_index=action['yaw_index'],
                                 object_id=oid) for oid in sorted(set(action['object_ids']) - visible))
    missing = set(o['id'] for o in objects) - covered
    return dict(status='passed' if not missing else 'failed',
                sample_grid=observer.settings.sample_grid, object_pose_pairs=count,
                elapsed_s=perf_counter()-started, covered_object_ids=sorted(covered),
                uncovered_object_ids=sorted(missing),
                rejected_supports=rejected)


def audit_parameters():
    grid, retries = settings.AUDIT_SAMPLE_GRID, settings.MAX_AUDIT_RETRIES
    same_cost_limit = settings.SAME_COST_REPAIR_TIME_LIMIT_S
    if isinstance(grid, bool) or not isinstance(grid, int) or not 4 <= grid <= 128:
        raise ValueError('AUDIT_SAMPLE_GRID must be an integer in [4, 128]')
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError('MAX_AUDIT_RETRIES must be a nonnegative integer')
    if (isinstance(same_cost_limit, bool) or not isinstance(same_cost_limit, (int, float))
            or not math.isfinite(same_cost_limit) or same_cost_limit < 0):
        raise ValueError('SAME_COST_REPAIR_TIME_LIMIT_S must be finite and nonnegative')
    return dict(sample_grid=grid, max_retries=retries, same_cost_time_limit_s=same_cost_limit)


def dense_observer(observer, objects):
    grid = max(audit_parameters()['sample_grid'], observer.settings.refinement_grid)
    return BoxObserver(observer.camera, objects, replace(observer.settings,
        sample_grid=grid, refinement_grid=grid))


def refine_after_audit(selected, candidates, stations, objects, observer, time_limit_s=None):
    """Repair sampled false positives without changing the exported candidate q.

    Only reject relations disproved by a denser final-action audit. A repaired
    solution inherits an original-pool optimality certificate only at exactly
    the original proven cost; a refined-pool optimum alone cannot certify it.
    Bound retries and retain the audited baseline when repair is unsuccessful.
    """
    parameters = audit_parameters()
    dense = dense_observer(observer, objects)
    initial = audit_observations(selected, objects, dense)
    audit = dict(initial=deepcopy(initial), final=initial, repairs=[], same_cost_searches=[], parameters=parameters,
                 scope='selected_actions_only_not_full_candidate_matrix')
    selected['optimization']['high_density_verification'] = audit
    if initial['status'] == 'passed':
        return selected
    logging.getLogger(__name__).info('Dense audit found uncovered targets %s; checking baseline and repairing',
                                     initial['uncovered_object_ids'])
    baseline = selected['baseline']
    baseline_audit = audit_observations(baseline, objects, dense)
    audit['baseline'] = baseline_audit
    if selected['status'] != 'complete' or baseline_audit['status'] != 'passed':
        audit['repair_status'] = 'no_audited_complete_baseline'
        return selected
    original_cost = list(selected['selected_cost'])
    original_proven = selected['optimization']['lexicographic_optimal']
    filtered = deepcopy(candidates.supports)
    baseline_actions = [(a['station_index'], o['yaw_index'])
                        for a in baseline['anchors'] for o in a['observations']]
    elapsed = sum(selected['optimization'][stage].get('elapsed_s', 0)
                  for stage in ('stage1', 'stage2'))
    targets = [o['id'] for o in objects]
    for _ in range(parameters['max_retries']):
        for row in audit['final']['rejected_supports']:
            filtered[row['station_index']][row['yaw_index']].discard(row['object_id'])
        # Dense sampling can also add a relation absent from original q. Such a
        # relation cannot serve as the feasible incumbent of this deletion-only
        # model; retain the separately audited original baseline instead.
        if set().union(*(filtered[i][y] for i, y in baseline_actions)) != set(targets):
            break
        remaining = None if time_limit_s is None else time_limit_s-elapsed
        if remaining is not None and remaining <= 0:
            break
        solution = None
        if original_proven and parameters['same_cost_time_limit_s'] > 0:
            limit = parameters['same_cost_time_limit_s']
            search = find_cover_at_cost(filtered, targets, original_cost,
                                       time_limit_s=min(limit, remaining) if remaining is not None else limit)
            elapsed += search['elapsed_s']
            audit['same_cost_searches'].append({k: v for k, v in search.items() if k != 'actions'})
            if search['actions'] is not None:
                solution = dict(actions=search['actions'], selected_cost=original_cost,
                                optimization={'fixed_cost_feasibility': audit['same_cost_searches'][-1]})
        if solution is None:
            remaining = None if time_limit_s is None else time_limit_s-elapsed
            if remaining is not None and remaining <= 0:
                break
            solution = optimize(filtered, targets, baseline_actions, time_limit_s=remaining)
            elapsed += sum(solution['optimization'][stage].get('elapsed_s', 0)
                           for stage in ('stage1', 'stage2'))
        audit['repairs'].append(dict(selected_cost=solution['selected_cost'],
                                     optimization=solution['optimization']))
        selected['anchors'] = assemble_anchors(solution['actions'], candidates, stations)
        selected.update(selected_cost=solution['selected_cost'],
            anchor_count=solution['selected_cost'][0], photo_count=solution['selected_cost'][1],
            selection_method='lexicographic_milp_audited_repair')
        audit['final'] = audit_observations(selected, objects, dense)
        if audit['final']['status'] == 'passed':
            audit['repair_status'] = 'passed'
            break
    if audit['final']['status'] != 'passed':
        selected.update(deepcopy(baseline))
        selected.update(vote_trace=[], selected_cost=selected['baseline_cost'],
                        selection_method='weighted_greedy_audited_fallback')
        audit.update(final=baseline_audit, repair_status='audited_baseline_fallback')
    proven = original_proven and selected['selected_cost'] == original_cost
    selected['optimization']['lexicographic_optimal'] = proven
    audit['original_pool_certificate_preserved'] = proven
    if not proven:
        selected['optimization']['fallback_reason'] = 'audit_changed_solution_without_original_pool_certificate'
    selected['optimality'] = 'finite_pool_lexicographic_optimal' if proven else 'not_proven'
    selected['comparison_to_baseline'] = ('improved' if tuple(selected['selected_cost']) <
                                           tuple(selected['baseline_cost']) else 'equal')
    return selected


def verify_observations(selected, objects, camera, observer):
    """Independently recompute every final action, including baseline actions."""
    actual = set()
    for anchor in selected['anchors']:
        for action in anchor['observations']:
            if observer:
                metrics = {o['id']: observer.evaluate(index, [anchor['position']], [action['yaw_rad']])[0]
                           for index, o in enumerate(objects)}
                visible = {identifier for identifier, row in metrics.items() if row['qualified']}
                action['object_metrics'] = metrics
            else:
                visible = {o['id'] for o in objects if camera.visible(
                    np.asarray([anchor['position']]), np.asarray([action['yaw_rad']]), o['position'])[0]}
            if visible != set(action['object_ids']):
                raise ValueError('Final observation failed camera geometry verification')
            actual.update(visible)
    if actual != set(selected['covered_object_ids']):
        raise ValueError('Final observations failed coverage verification')


def verify_plan(selected, objects, camera, observer):
    """Verify both plans under the original model; fall back if the new plan fails."""
    verify_observations(selected['baseline'], objects, camera, observer)
    try:
        verify_observations(selected, objects, camera, observer)
        selected['optimization']['geometry_verification'] = 'passed'
    except ValueError as exc:
        selected.update(deepcopy(selected['baseline']))
        selected.update(vote_trace=[], selection_method='weighted_greedy_fallback',
                        selected_cost=selected['baseline_cost'], comparison_to_baseline='equal',
                        optimality='not_proven')
        selected['optimization'].update(lexicographic_optimal=False,
            geometry_verification='new_plan_failed_baseline_passed', fallback_reason=str(exc))
        if observer:
            dense = dense_observer(observer, objects)
            selected['optimization']['high_density_verification']['final'] = audit_observations(
                selected, objects, dense)
    return selected
