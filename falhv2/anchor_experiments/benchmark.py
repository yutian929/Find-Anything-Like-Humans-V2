"""Reproducible comparison: full-pool equivalence, sampling, refinement, voting."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import platform
import statistics
import sys
from time import perf_counter

import cv2
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from path_planner import plan as plan_path
from anchor_experiments.common import PointCamera, stations_from_plan, validate_objects, coverage_edges
from anchor_experiments import object_driver, position_driver
from anchor_experiments.fixtures import fixtures
from anchor_experiments.selection import select_anchors, exact_minimum_station_count, audit_small_station_bound
from anchor_experiments.run import render_svg


def fingerprint(candidate):
    return hashlib.sha256(repr(sorted(coverage_edges(candidate))).encode()).hexdigest()


def audit_selection(selected,candidate,stations,objects,camera):
    by_id={s.index:s for s in stations}; by_obj={o['id']:o['position'] for o in objects}
    actual=set()
    for anchor in selected['anchors']:
        station=by_id[anchor['station_index']]
        assert tuple(anchor['position'])==station.position
        for action in anchor['observations']:
            assert action['yaw_rad']==candidate.yaws[action['yaw_index']]
            for obj in action['object_ids']:
                assert camera.visible(np.array([station.position]),np.array([action['yaw_rad']]),by_obj[obj])[0]
                actual.add(obj)
    assert actual==set(selected['covered_object_ids'])
    assert actual=={obj for _,_,obj in coverage_edges(candidate)}
    assert set(selected['uncovered_object_ids'])==set(by_obj)-actual


def measure(fn,stations,objects,camera,repeats):
    times=[]; selection_times=[]; hashes=[]
    for _ in range(repeats):
        candidate=fn(); times.append(candidate.stats['generation_seconds']); hashes.append(fingerprint(candidate))
        started=perf_counter(); selected=select_anchors(candidate,stations,[o['id'] for o in objects]); selection_times.append(perf_counter()-started)
        audit_selection(selected,candidate,stations,objects,camera)
    assert len(set(hashes))==1,'Output changed across deterministic repeats'
    return {'generation_seconds':times,'median_generation_seconds':statistics.median(times),
            'selection_seconds':selection_times,'median_selection_seconds':statistics.median(selection_times),
            'station_count':len(candidate.station_indices),'positive_station_count':len(candidate.supports),
            'point_pose_tests':candidate.stats['evaluated_point_pose_pairs'],
            'support_edges':len(coverage_edges(candidate)), 'coverage_hash':hashes[0],
            'deterministic_repeats':True,'driver_stats':candidate.stats,
            **{key:selected[key] for key in ('status','anchor_count','photo_count','coverage_fraction','covered_object_ids','uncovered_object_ids')}},candidate,selected


def diagnose(point,stations,camera,yaws):
    counts=Counter()
    positions=np.array([s.position for s in stations]); angles=np.array(yaws)
    for start in range(0,len(stations),1024):
        p=np.repeat(positions[start:start+1024],len(angles),axis=0)
        theta=np.tile(angles,len(positions[start:start+1024]))
        u,v,z,d=camera.project(p,theta,point)
        reasons=np.full(len(d),'supported',dtype='<U20')
        reasons[(v < -1e-9)|(v > camera.height-1+1e-9)]='vertical_fov'
        reasons[(u < -1e-9)|(u > camera.width-1+1e-9)]='horizontal_fov'
        reasons[z <= 1e-12]='behind_camera'
        reasons[(d+1e-9 < camera.min_distance)|(d > camera.max_distance+1e-9)]='distance'
        labels,n=np.unique(reasons,return_counts=True)
        counts.update(dict(zip(labels.tolist(),n.tolist())))
    return dict(counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/anchor_experiment/benchmark')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--random-seeds',type=int,default=10)
    parser.add_argument('--fixtures',type=Path)
    parser.add_argument('--cases',nargs='*')
    args=parser.parse_args()
    if args.repeats<1 or args.random_seeds<1: parser.error('repeats and random-seeds must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    specs=json.loads(args.fixtures.read_text()) if args.fixtures else list(fixtures())
    if args.cases: specs=[row for row in specs if row[0] in args.cases]
    (args.output/'fixtures.json').write_text(json.dumps(specs,ensure_ascii=False)+'\n')
    results={'base_commit':'04c79bb','environment':{'python':platform.python_version(),'numpy':np.__version__,
             'scipy':scipy.__version__,'opencv':cv2.__version__,'platform':platform.platform()},
             'repeats':args.repeats,'random_seeds':list(range(args.random_seeds)),
             'contract':'all input point IDs; shared A1 path; 30deg base yaw; no occlusion or target extent; zero native distortion',
             'timing_scope':'candidate generation includes spatial tree or sampling and coverage construction; excludes A1, selection, validation, serialization; selection timed separately',
             'cases':[]}
    for name,scene,robot_config,camera_config in specs:
        print('START',name,flush=True)
        objects=validate_objects(scene.get('objects',[])); camera=PointCamera(camera_config)
        started=perf_counter(); path=plan_path(scene,robot_config); stations=stations_from_plan(path); a1=perf_counter()-started
        case={'name':name,'shape':[len(scene['grid']['rows']),len(scene['grid']['rows'][0])],
              'input_sha256':hashlib.sha256(json.dumps([scene,robot_config,camera_config],sort_keys=True).encode()).hexdigest(),
              'station_count':len(stations),'object_count':len(objects),'region_count':len(path['regions']),
              'path_length_m':path['statistics']['total_path_length_m'],'a1_and_stations_seconds':a1,'methods':{}}
        methods=case['methods']
        exhaustive,ref,ref_selected=measure(lambda:position_driver.generate(stations,objects,camera,strategy='exhaustive'),stations,objects,camera,args.repeats)
        methods['position_exhaustive']=exhaustive
        measured,full,selected=measure(lambda:object_driver.generate(stations,objects,camera),stations,objects,camera,args.repeats)
        methods['object_all']=measured
        assert coverage_edges(full)==coverage_edges(ref),'Inverse screening lost or added support'
        case['full_pool_equivalence']=True
        case['full_pool_uncovered_object_ids']=selected['uncovered_object_ids']
        case['uncovered_failure_counts']={o['id']:diagnose(o['position'],stations,camera,ref.yaws)
                for o in objects if o['id'] in selected['uncovered_object_ids']}
        if len(objects)<=12:
            case['exact_minimum_anchors_full_targets']=exact_minimum_station_count(ref,[o['id'] for o in objects])
            case['exact_minimum_anchors_coverable_targets']=exact_minimum_station_count(ref,selected['covered_object_ids'])
        if name=='user_map' and not selected['uncovered_object_ids']:
            case['station_count_certificate']=audit_small_station_bound(ref,[o['id'] for o in objects],selected['anchor_count'])
        for refine in (False,True):
            label='uniform_refined' if refine else 'uniform_64'
            row,candidate,plan=measure(lambda:position_driver.generate(stations,objects,camera,strategy='uniform',refine=refine),stations,objects,camera,args.repeats)
            check=object_driver.generate(stations,objects,camera,station_indices=candidate.station_indices)
            assert coverage_edges(check)==coverage_edges(candidate)
            row['fixed_pool_equivalence']=True
            row['missed_but_full_pool_coverable']=sorted(set(selected['covered_object_ids'])-set(plan['covered_object_ids']))
            methods[label]=row
        for refine in (False,True):
            rows=[]
            for seed in range(args.random_seeds):
                row,candidate,plan=measure(lambda:position_driver.generate(stations,objects,camera,strategy='random',seed=seed,refine=refine),stations,objects,camera,args.repeats)
                check=object_driver.generate(stations,objects,camera,station_indices=candidate.station_indices)
                assert coverage_edges(check)==coverage_edges(candidate)
                row['fixed_pool_equivalence']=True; row['seed']=seed
                row['missed_but_full_pool_coverable']=sorted(set(selected['covered_object_ids'])-set(plan['covered_object_ids']))
                rows.append(row)
            methods['random_refined' if refine else 'random_64']={'runs':rows}
        # Isolate discretization: same full station pool, finer yaw, not a new method.
        fine=object_driver.generate(stations,objects,camera,yaw_step_deg=15)
        fine_selected=select_anchors(fine,stations,[o['id'] for o in objects])
        audit_selection(fine_selected,fine,stations,objects,camera)
        case['yaw_15deg_sensitivity']={'anchor_count':fine_selected['anchor_count'],'photo_count':fine_selected['photo_count'],
            'uncovered_object_ids':fine_selected['uncovered_object_ids'],'generation_seconds':fine.stats['generation_seconds']}
        if name=='user_map':
            example={'experimental':True, 'method':'object_all', 'observation_scope':results['contract'],
                     'configuration':{'camera':camera_config,'robot':robot_config}, **selected}
            (args.output/'user_object_plan.json').write_text(json.dumps(example,ensure_ascii=False,indent=2)+'\n')
            (args.output/'user_object_plan.svg').write_text(render_svg(scene,path,example,stations,full))
        results['cases'].append(case)
        (args.output/'comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print('DONE',name,'stations',len(stations),'objects',len(objects),'anchors',selected['anchor_count'],
              'photos',selected['photo_count'],'gaps',selected['uncovered_object_ids'],flush=True)


if __name__=='__main__': main()
