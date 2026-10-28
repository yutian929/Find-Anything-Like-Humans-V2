import copy
import itertools
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from falhv2.anchor_planners.observation_model import PointCamera, validate_objects
from falhv2.anchor_planners.anchor_candidates import Station, CandidateResult
from falhv2.anchor_planners.anchor_selector import select_anchors_greedy as select_anchors


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



def config():
    return {"camera": {"width": 201, "height": 101, "fx": 100, "fy": 100, "cx": 100, "cy": 50,
                       "distortion_model": "none", "distortion_coefficients": [0]*5},
            "mount": {"position_in_base_m": [0, 0, 0], "roll_deg": 0, "pitch_deg": 0, "yaw_deg": 0},
            "observation_quality": {"min_distance_m": .2, "max_distance_m": 3}}


class AnchorCommonTests(unittest.TestCase):
    def test_known_projection_fov_range_and_height(self):
        camera = PointCamera(config())
        p = np.array([[0., 0.]])
        u, v, z, distance = camera.project(p, np.array([0.]), [2, .4, .2])
        self.assertAlmostEqual(u[0], 80)
        self.assertAlmostEqual(v[0], 40)
        self.assertAlmostEqual(z[0], 2)
        self.assertAlmostEqual(distance[0], math.sqrt(4.2))
        for point, expected in (([1, 1, 0], True), ([1, 1.001, 0], False),
                                ([1, 0, .5], True), ([1, 0, .501], False),
                                ([-1, 0, 0], False), ([3, 0, 0], True),
                                ([3.001, 0, 0], False), ([.2, 0, 0], True), ([.19, 0, 0], False)):
            self.assertEqual(bool(camera.visible(p, np.zeros(1), point)[0]), expected, point)

    def test_independent_homogeneous_transform_with_all_mount_axes(self):
        cfg = config()
        cfg['mount'] = {'position_in_base_m': [.23, -.12, .5], 'roll_deg': 12, 'pitch_deg': 18, 'yaw_deg': -15}
        camera = PointCamera(cfg)
        roll, pitch, yaw = np.deg2rad([12, 18, -15])
        def rx(a): return np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
        def ry(a): return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
        def rz(a): return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
        base_yaw = 1.1
        world_from_optical = rz(base_yaw) @ rz(yaw) @ ry(pitch) @ rx(roll) @ np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        origin = np.array([2, -3, 0]) + rz(base_yaw) @ np.array(cfg['mount']['position_in_base_m'])
        target = origin + world_from_optical @ np.array([.2, -.1, 2.])
        u,v,z,d = camera.project(np.array([[2.,-3.]]), np.array([base_yaw]), target)
        self.assertAlmostEqual(u[0], 110)
        self.assertAlmostEqual(v[0], 45)
        self.assertAlmostEqual(z[0], 2)
        self.assertAlmostEqual(d[0], math.sqrt(4.05))
        self.assertTrue(camera.visible(np.array([[2.,-3.]]), np.array([base_yaw]), target)[0])

    def test_invalid_or_unknown_model_conditions_are_rejected(self):
        for field, value in [('pitch_deg', None), ('yaw_deg', float('nan'))]:
            cfg=config(); cfg['mount'][field]=value
            with self.assertRaises(ValueError): PointCamera(cfg)
        cfg=config(); cfg['camera']['distortion_coefficients'][0]=.1
        with self.assertRaises(ValueError): PointCamera(cfg)
        with self.assertRaises(ValueError): validate_objects([{'id':'x','position':[0,0,0]}]*2)

    def test_dynamic_votes_deduplicate_objects_and_preserve_all_targets(self):
        stations=[Station(i,1,(0,i),(i,0)) for i in range(3)]
        candidate=CandidateResult([0,1,2],[0.,math.pi],{
            0:{0:{'a','b'},1:{'a'}}, 1:{0:{'a','b','c'}}, 2:{0:{'c','d'}}}, {})
        selected=select_anchors(candidate,stations,['a','b','c','d'])
        self.assertEqual(selected['anchor_count'],2)
        self.assertEqual(selected['photo_count'],2)
        self.assertEqual(selected['vote_trace'][0]['station_index'],1)
        self.assertEqual(selected['coverage_fraction'],1)
        self.assertEqual(exact_minimum_station_count(candidate,['a','b','c','d']),2)
        partial=select_anchors(candidate,stations,['a','b','c','d','missing'])
        self.assertEqual(partial['uncovered_object_ids'],['missing'])
        self.assertEqual(partial['status'],'partial')
        empty=select_anchors(CandidateResult([],[],{},{}),stations,[])
        self.assertEqual(empty['anchor_count'],0)
        self.assertIsNone(empty['coverage_fraction'])

    def test_small_exact_audit_and_dynamic_votes_against_brute_force(self):
        rng=np.random.default_rng(14)
        targets=['a','b','c','d','e']
        stations=[Station(i,1,(0,i),(i,0)) for i in range(6)]
        for _ in range(30):
            unions={i:{o for o in targets if rng.random()<.5} for i in range(6)}
            candidate=CandidateResult(list(range(6)),[0.],{i:{0:s} for i,s in unions.items() if s},{})
            expected=None
            for n in range(7):
                if any(set().union(*(unions[i] for i in subset))==set(targets)
                       for subset in itertools.combinations(range(6),n)):
                    expected=n
                    break
            self.assertEqual(exact_minimum_station_count(candidate,targets),expected)
            remaining=set(targets); expected_trace=[]
            while remaining:
                i=min(unions,key=lambda i:(-len(unions[i]&remaining),i))
                new=unions[i]&remaining
                if not new: break
                expected_trace.append(i); remaining-=new
            chosen=select_anchors(candidate,stations,targets)
            self.assertEqual([row['station_index'] for row in chosen['vote_trace']],expected_trace)
            self.assertEqual(set(chosen['uncovered_object_ids']),remaining)


if __name__=='__main__': unittest.main()
