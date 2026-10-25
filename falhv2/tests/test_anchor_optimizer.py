"""Independent action-subset oracle and resource-priority/failure contracts."""
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anchor_optimizer as solver
import anchor_selector as selector
import anchor_verification as verification
import main as pipeline


def instance(supports, scores=None, targets=None):
    stations = [selector.Station(i, 1, (0, j), (j, 0)) for j, i in enumerate(sorted(supports))]
    n = max((y for ds in supports.values() for y in ds), default=0) + 1
    candidates = selector.CandidateResult([s.index for s in stations], list(range(n)), supports, {}, scores or {})
    targets = sorted(set().union(*(ids for ds in supports.values() for ids in ds.values()))) if targets is None else targets
    return candidates, stations, targets


def exhaustive(supports, targets):
    keys = [(i, y) for i, ds in supports.items() for y, ids in ds.items() if ids]
    best = None
    for n in range(len(keys) + 1):
        for chosen in itertools.combinations(keys, n):
            covered = set().union(*(supports[i][y] for i, y in chosen))
            if covered == set(targets):
                value = (len({i for i, _ in chosen}), n)
                best = value if best is None else min(best, value)
    return best


class OptimizerTests(unittest.TestCase):
    def solve(self, supports, scores=None, targets=None, **kwargs):
        return selector.select_anchors(*instance(supports, scores, targets), **kwargs)

    def test_counterexample_improves_three_stations_to_two(self):
        sets = [set('1234'), set('125'), set('346'), {'5'}, {'6'}]
        weights = [1, .9, .9, 1, 1]
        r = self.solve({i: {0: ids} for i, ids in enumerate(sets)},
                       {i: {0: {o: weights[i] for o in ids}} for i, ids in enumerate(sets)})
        self.assertEqual(r['baseline_cost'], [3, 3])
        self.assertEqual(r['selected_cost'], [2, 2])
        self.assertTrue(r['optimization']['lexicographic_optimal'])
        self.assertEqual(r['vote_trace'], [])

    def test_photo_stage_can_reselect_stations(self):
        real_milp = solver.milp
        calls = []
        def solve_stage(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                # Deliberately return the worse photo arrangement at the same K.
                return SimpleNamespace(status=0, message='station optimum', x=[1, 0],
                                       fun=1, mip_gap=0, mip_dual_bound=1)
            return real_milp(**kwargs)
        with patch.object(solver, 'milp', side_effect=solve_stage):
            r = self.solve({7: {0: {'a'}, 1: {'b'}, 2: {'c'}}, 93: {0: set('abc')}})
        self.assertEqual(r['baseline_cost'], [1, 3])
        self.assertEqual(r['selected_cost'], [1, 1])
        self.assertEqual(r['anchors'][0]['station_index'], 93)
        self.assertTrue(r['optimization']['lexicographic_optimal'])

    def test_stations_have_strict_priority_over_photos(self):
        r = self.solve({0: {0: {'a'}, 1: {'b'}, 2: {'c'}},
                        1: {0: set('ab')}, 2: {0: {'c'}}})
        self.assertEqual(r['selected_cost'], [1, 3])

    def test_scores_cannot_override_lower_resource_cost(self):
        r = self.solve({0: {0: set('ab')}, 1: {0: {'a'}}, 2: {0: {'b'}}},
                       {0: {0: {'a': .5, 'b': .5}}, 1: {0: {'a': 1}}, 2: {0: {'b': 1}}})
        self.assertEqual(r['selected_cost'], [1, 1])

    def test_random_instances_match_independent_action_subset_oracle(self):
        rng = np.random.default_rng(830)
        for _ in range(25):
            supports = {i: {y: {o for o in 'abcd' if rng.random() < .5} for y in range(3)}
                        for i in [2, 15, 71]}
            c, stations, targets = instance(supports)
            r = selector.select_anchors(c, stations, targets)
            self.assertEqual(tuple(r['selected_cost']), exhaustive(supports, targets))
            self.assertLessEqual(tuple(r['selected_cost']), tuple(r['baseline_cost']))
            self.assertTrue(r['optimization']['lexicographic_optimal'])

    def test_exact_reduction_preserves_action_families_not_just_unions(self):
        supports = {10: {0: {'a'}, 1: {'a', 'b'}, 2: {'c'}},
                    20: {4: {'a', 'b'}, 7: {'c'}},
                    30: {0: {'a'}, 1: {'b'}, 2: {'c'}},
                    40: {0: set('abc')}}
        before = {i: {y: set(ids) for y, ids in ds.items()} for i, ds in supports.items()}
        reduced, mapping = solver._equivalent_cover_model(supports)
        self.assertEqual(set(reduced), {10, 30, 40})
        self.assertEqual(set(reduced[10]), {1, 2})
        self.assertEqual(mapping[(20, 4)], (10, 1))
        self.assertEqual(mapping[(10, 0)], (10, 1))
        self.assertEqual(supports, before)
        self.assertEqual(exhaustive(supports, set('abc')), exhaustive(reduced, set('abc')))

    def test_many_equivalent_stations_keep_a_feasible_baseline_mapping(self):
        supports = {i: {i % 3: {'a'}, (i % 3) + 3: {'b'}} for i in range(5)}
        r = solver.optimize(supports, ['a', 'b'], [(0, 0), (1, 4)])
        self.assertEqual(r['selected_cost'], [1, 2])
        self.assertTrue(r['optimization']['lexicographic_optimal'])
        self.assertEqual(r['optimization']['exact_reduction']['model_stations'], 1)

    def test_empty_and_partial_do_not_claim_full_cover_optimality(self):
        for supports, targets, status in [({}, [], 'empty_targets'),
                                         ({1: {0: {'a'}}}, ['a', 'missing'], 'partial')]:
            r = self.solve(supports, targets=targets)
            self.assertEqual(r['status'], status)
            self.assertFalse(r['optimization']['lexicographic_optimal'])
            self.assertEqual(r['optimization']['stage1']['status'], 'not_run')

    def test_solver_exception_preserves_baseline(self):
        with patch.object(solver, 'milp', side_effect=RuntimeError('test failure')):
            r = self.solve({1: {0: {'a'}}})
        self.assertEqual(r['selected_cost'], r['baseline_cost'])
        self.assertIn('test failure', r['optimization']['fallback_reason'])
        self.assertEqual(r['selection_method'], 'weighted_greedy_fallback')

    def test_shared_budget_exhaustion_and_no_incumbent(self):
        response = SimpleNamespace(status=1, message='limit', x=None, fun=None)
        with patch.object(solver, 'milp', return_value=response) as call, \
             patch.object(solver, 'perf_counter', side_effect=[0, 2]):
            r = self.solve({1: {0: {'a'}}}, time_limit_s=1)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(r['selected_cost'], [1, 1])
        self.assertEqual(r['optimization']['stage2']['status'], 'time_limit')
        self.assertFalse(r['optimization']['lexicographic_optimal'])

    def test_invalid_integer_incumbent_does_not_replace_baseline(self):
        for x in ([.5], [0], [2], [float('nan')]):
            response = SimpleNamespace(status=0, message='mock', x=x, fun=1, mip_gap=0, mip_dual_bound=1)
            with patch.object(solver, 'milp', return_value=response):
                r = self.solve({1: {0: {'a'}}})
            self.assertEqual(r['selected_cost'], [1, 1])
            self.assertEqual(r['optimization']['stage1']['status'], 'invalid_or_failed')

    def test_photo_station_link_is_independently_checked(self):
        # Stage 1 is valid; stage 2 falsely claims a selected station but no photo.
        responses = [SimpleNamespace(status=0, message='mock', x=[1], fun=1, mip_gap=0, mip_dual_bound=1),
                     SimpleNamespace(status=0, message='mock', x=[1, 0], fun=0, mip_gap=0, mip_dual_bound=0)]
        with patch.object(solver, 'milp', side_effect=responses):
            r = self.solve({1: {0: {'a'}}})
        self.assertEqual(r['selected_cost'], [1, 1])
        self.assertEqual(r['optimization']['stage2']['status'], 'invalid_or_failed')
        self.assertFalse(r['optimization']['lexicographic_optimal'])

    def test_invalid_input_is_rejected_before_solver(self):
        for limit in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.solve({1: {0: {'a'}}}, time_limit_s=limit)
        with self.assertRaises(ValueError):
            solver.optimize({1: {0: {'unknown'}}}, ['a'], [(1, 0)])

    def test_stage_one_improvement_survives_photo_solver_failure(self):
        sets = [set('1234'), set('125'), set('346'), {'5'}, {'6'}]
        weights = [1, .9, .9, 1, 1]
        response = SimpleNamespace(status=0, message='mock', x=[0, 1, 1, 0, 0],
                                   fun=2, mip_gap=0, mip_dual_bound=2)
        with patch.object(solver, 'milp', side_effect=[response, RuntimeError('photo failure')]):
            r = self.solve({i: {0: ids} for i, ids in enumerate(sets)},
                           {i: {0: {o: weights[i] for o in ids}} for i, ids in enumerate(sets)})
        self.assertEqual(r['baseline_cost'], [3, 3])
        self.assertEqual(r['selected_cost'], [2, 2])
        self.assertTrue(r['optimization']['stage1']['proven_optimal'])
        self.assertFalse(r['optimization']['lexicographic_optimal'])

    def test_conditional_photo_optimum_is_not_global_station_certificate(self):
        responses = [SimpleNamespace(status=1, message='limit', x=[1], fun=1, mip_gap=.5, mip_dual_bound=.5),
                     SimpleNamespace(status=0, message='optimal', x=[1, 1], fun=1, mip_gap=0, mip_dual_bound=1)]
        with patch.object(solver, 'milp', side_effect=responses):
            r = self.solve({1: {0: {'a'}}})
        self.assertFalse(r['optimization']['stage1']['proven_optimal'])
        self.assertTrue(r['optimization']['stage2']['proven_optimal'])
        self.assertFalse(r['optimization']['lexicographic_optimal'])

    def test_geometry_failure_falls_back_to_verified_baseline(self):
        scene = {'grid': {'rows': ['....'] * 4, 'resolution': 1, 'origin': [0, 0]},
                 'objects': [{'id': 'a', 'position': [2, 2, 0]}]}
        from test_anchor_common import config
        robot = {'robot': {'footprint_width_m': .2, 'footprint_length_m': .2, 'safety_margin_m': 0}}
        with patch.object(verification, 'verify_observations', side_effect=[None, ValueError('geometry mismatch')]):
            r, *_ = pipeline.plan(scene, robot, config())
        self.assertEqual(r['selected_cost'], r['baseline_cost'])
        self.assertEqual(r['anchors'], r['baseline']['anchors'])
        self.assertFalse(r['optimization']['lexicographic_optimal'])

    def audit_case(self, alternate=True, time_limit_s=None):
        supports = {0: {0: set('1234')}, 1: {0: set('125')},
                    2: {0: set('346')}, 3: {0: {'5'}}, 4: {0: {'6'}}}
        if alternate:
            supports[5] = {0: set('125')}
        scores = {i: {0: {o: (.9 if i in (1, 2, 5) else 1) for o in ds[0]}}
                  for i, ds in supports.items()}
        c, stations, targets = instance(supports, scores)
        selected = selector.select_anchors(c, stations, targets)
        self.assertEqual(selected['selected_cost'], [2, 2])
        from observation_model import ObservationSettings
        settings = ObservationSettings()
        def evaluate(index, positions, yaws):
            station = int(positions[0][0])
            oid = targets[index]
            return [{'qualified': oid in supports[station][0] and not (station == 1 and oid == '5')}]
        dense = SimpleNamespace(settings=SimpleNamespace(sample_grid=96), evaluate=evaluate)
        with patch.object(verification, 'BoxObserver', return_value=dense):
            result = verification.refine_after_audit(selected, c, stations,
                [{'id': oid} for oid in targets], SimpleNamespace(camera=None, settings=settings), time_limit_s)
        self.assertEqual(c.supports[1][0], set('125'))  # Original heatmap/supports stay intact.
        return result

    def test_dense_audit_repairs_at_original_proven_cost(self):
        r = self.audit_case()
        audit = r['optimization']['high_density_verification']
        self.assertEqual(audit['initial']['status'], 'failed')
        self.assertEqual(audit['final']['status'], 'passed')
        self.assertEqual(len(audit['repairs']), 1)
        self.assertEqual(r['selected_cost'], [2, 2])
        self.assertTrue(r['optimization']['lexicographic_optimal'])
        self.assertTrue(audit['original_pool_certificate_preserved'])
        self.assertEqual({a['station_index'] for a in r['anchors']}, {2, 5})

    def test_dense_repair_optimum_does_not_prove_original_optimum(self):
        r = self.audit_case(alternate=False)
        self.assertEqual(r['selected_cost'], [3, 3])
        self.assertEqual(r['optimization']['high_density_verification']['final']['status'], 'passed')
        self.assertFalse(r['optimization']['lexicographic_optimal'])
        self.assertEqual(r['optimality'], 'not_proven')

    def test_dense_audit_budget_exhaustion_uses_audited_baseline(self):
        r = self.audit_case(time_limit_s=1e-12)
        audit = r['optimization']['high_density_verification']
        self.assertEqual(audit['repairs'], [])
        self.assertEqual(audit['repair_status'], 'audited_baseline_fallback')
        self.assertEqual(r['selected_cost'], r['baseline_cost'])
        self.assertFalse(r['optimization']['lexicographic_optimal'])

    def test_dense_audit_does_not_ignore_uncovered_original_targets(self):
        dense = SimpleNamespace(settings=SimpleNamespace(sample_grid=96))
        audit = verification.audit_observations({'anchors': [], 'covered_object_ids': []},
                                             [{'id': 'missing'}], dense)
        self.assertEqual(audit['status'], 'failed')
        self.assertEqual(audit['uncovered_object_ids'], ['missing'])

    def test_dense_repair_retries_are_bounded(self):
        failed = dict(status='failed', rejected_supports=[
            dict(station_index=1, yaw_index=0, object_id='5')], uncovered_object_ids=['5'])
        passed = dict(status='passed', rejected_supports=[], uncovered_object_ids=[])
        with patch.object(verification, 'audit_observations',
                          side_effect=[failed, passed, failed, failed, failed]):
            r = self.audit_case()
        audit = r['optimization']['high_density_verification']
        self.assertEqual(len(audit['repairs']), 3)
        self.assertEqual(audit['repair_status'], 'audited_baseline_fallback')
        self.assertEqual(r['selected_cost'], r['baseline_cost'])

    def test_configured_zero_retries_uses_audited_baseline(self):
        with patch.object(verification.settings, 'MAX_AUDIT_RETRIES', 0):
            r = self.audit_case()
        audit = r['optimization']['high_density_verification']
        self.assertEqual(audit['repairs'], [])
        self.assertEqual(audit['repair_status'], 'audited_baseline_fallback')
        self.assertEqual(r['selected_cost'], r['baseline_cost'])

    def test_dense_audit_does_not_claim_unverified_baseline_is_safe(self):
        failed = dict(status='failed', rejected_supports=[], uncovered_object_ids=['5'])
        with patch.object(verification, 'audit_observations', side_effect=[failed, failed]):
            r = self.audit_case()
        audit = r['optimization']['high_density_verification']
        self.assertEqual(audit['repair_status'], 'no_audited_complete_baseline')
        self.assertEqual(audit['final']['status'], 'failed')
        self.assertEqual(audit['repairs'], [])


if __name__ == '__main__':
    unittest.main()
