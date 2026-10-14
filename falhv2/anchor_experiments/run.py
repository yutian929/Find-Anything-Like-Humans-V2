"""Experimental CLI: shared A1 -> point supports -> station voting -> scan actions."""
from pathlib import Path
import argparse
import json
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from path_planner import plan as plan_path, diagnostic_svg
from anchor_experiments.common import PointCamera, validate_objects, stations_from_plan
from anchor_experiments import object_driver, position_driver
from anchor_experiments.selection import select_anchors


def run_scene(scene,robot_config,camera_config,*,method='object',strategy='uniform',initial_count=64,
              seed=0,yaw_step_deg=30,refine=True,max_stations=None):
    objects=validate_objects(scene.get('objects',[]))
    camera=PointCamera(camera_config)
    started=perf_counter(); path=plan_path(scene,robot_config); stations=stations_from_plan(path)
    preprocessing=perf_counter()-started
    if method=='object':
        candidates=object_driver.generate(stations,objects,camera,yaw_step_deg=yaw_step_deg)
    elif method=='position':
        candidates=position_driver.generate(stations,objects,camera,yaw_step_deg=yaw_step_deg,
            strategy=strategy,initial_count=initial_count,seed=seed,refine=refine,max_stations=max_stations)
    else: raise ValueError('method must be object or position')
    started=perf_counter(); selected=select_anchors(candidates,stations,[o['id'] for o in objects])
    selection=perf_counter()-started
    return {"experimental":True,"method":method,"scene_name":scene.get('name','unnamed'),
            "observation_scope":"3D point centers; FOV and Euclidean distance; occlusion and appearance not evaluated",
            "configuration":{"robot":robot_config,"camera":camera_config,"yaw_step_deg":yaw_step_deg},
            "statistics":{"a1_and_station_seconds":preprocessing,"selection_seconds":selection,
                          "path_length_m":path['statistics']['total_path_length_m'],
                          "region_count":len(path['regions']),"path_station_count":len(stations),**candidates.stats},
            **selected},path,stations,candidates


def render_svg(scene,path,result,stations,candidates):
    import html, math
    svg=diagnostic_svg(scene,path)
    grid=path['grid']; scale=36; pad=32
    def screen(position):
        return (pad+(position[0]-grid['origin'][0])/grid['resolution']*scale,
                pad+(grid['height']-(position[1]-grid['origin'][1])/grid['resolution'])*scale)
    shapes=['<g id="anchor-observation-overlay">']
    by_id={s.index:s for s in stations}
    for i,directions in candidates.supports.items():
        count=len(set().union(*directions.values()))
        x,y=screen(by_id[i].position)
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#ff9f1c" opacity="{min(.7,.1+.06*count):.2f}"><title>{count} target votes</title></circle>')
    missing=set(result['uncovered_object_ids'])
    for obj in scene.get('objects',[]):
        x,y=screen(obj['position']); color='#e63946' if obj['id'] in missing else '#2a9d8f'
        name=html.escape(obj['id'])
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="13" fill="{color}" stroke="white" stroke-width="3"/><text x="{x+15:.2f}" y="{y:.2f}" font-size="20">{name}</text>')
    for anchor in result['anchors']:
        x,y=screen(anchor['position'])
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="18" fill="none" stroke="#d000a6" stroke-width="5"/>')
        for action in anchor['observations']:
            yaw=action['yaw_rad']; length=65
            shapes.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x+length*math.cos(yaw):.2f}" y2="{y-length*math.sin(yaw):.2f}" stroke="#d000a6" stroke-width="5"/>')
    shapes.append('</g>')
    return svg.replace('</svg>','\n'.join(shapes)+'\n</svg>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,default=Path(__file__).with_name('fixtures')/'user_map_20261014.json')
    parser.add_argument('--robot',type=Path,default=ROOT/'config/robot.example.json')
    parser.add_argument('--camera',type=Path,default=ROOT/'config/camera.d435i.json')
    parser.add_argument('--method',choices=('object','position'),default='object')
    parser.add_argument('--strategy',choices=('uniform','random','exhaustive'),default='uniform')
    parser.add_argument('--initial-count',type=int,default=64)
    parser.add_argument('--max-stations',type=int)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--yaw-step-deg',type=float,default=30)
    parser.add_argument('--no-refine',action='store_true')
    parser.add_argument('--save-supports',action='store_true',help='另存完整稀疏方向级覆盖关系')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/anchor_experiment/anchor_plan.json')
    args=parser.parse_args()
    try:
        scene=json.loads(args.scene.read_text())
        result,path,stations,candidates=run_scene(scene,json.loads(args.robot.read_text()),json.loads(args.camera.read_text()),
            method=args.method,strategy=args.strategy,initial_count=args.initial_count,seed=args.seed,
            yaw_step_deg=args.yaw_step_deg,refine=not args.no_refine,max_stations=args.max_stations)
        result['uncovered_diagnosis_scope']='No support among evaluated discrete station/yaw candidates; not continuous-space infeasibility'
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        args.output.with_suffix('.svg').write_text(render_svg(scene,path,result,stations,candidates))
        if args.save_supports:
            # Full A2 relation for either driver, independent of A3 selection.
            support_result={'object_points':scene.get('objects',[]),'yaws_rad':candidates.yaws,
                'evaluated_station_indices':candidates.station_indices,
                'positive_stations':[{'index':s.index,'region_id':s.region_id,'cell':list(s.cell),
                    'position':list(s.position),'route_indices':list(s.route_indices)}
                    for s in stations if s.index in candidates.supports],
                'supports':{str(i):{str(yaw):sorted(objects) for yaw,objects in directions.items()}
                            for i,directions in candidates.supports.items()}}
            args.output.with_suffix('.supports.json').write_text(json.dumps(support_result,ensure_ascii=False)+'\n')
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(2,f'实验规划失败：{exc}\n')
    print(json.dumps({k:result[k] for k in ('status','anchor_count','photo_count','coverage_fraction','uncovered_object_ids')},ensure_ascii=False,indent=2))
    print(args.output)


if __name__=='__main__': main()
