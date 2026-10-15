"""Generate an offline interactive HTML and side-by-side PNG for anchor methods."""
from pathlib import Path
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from anchor_experiments.run import run_scene
from anchor_experiments.common import PointCamera


def build_data(scene,robot_config,camera_config,seed=0):
    camera=PointCamera(camera_config)
    methods=[]
    for method,strategy,title in [('object','uniform','物体驱动'),('position','uniform','均匀采样＋加密'),('position','random',f'随机采样＋加密 · seed {seed}')]:
        result,path,stations,candidates=run_scene(scene,robot_config,camera_config,method=method,strategy=strategy,seed=seed)
        actions=[]
        for anchor_index,anchor in enumerate(result['anchors']):
            for action in anchor['observations']:
                projected=[]
                for obj in scene.get('objects',[]):
                    u,v,z,d=camera.project(np.array([anchor['position']]),np.array([action['yaw_rad']]),obj['position'])
                    projected.append({'id':obj['id'],'u':float(u[0]) if math.isfinite(u[0]) else None,
                        'v':float(v[0]) if math.isfinite(v[0]) else None,'depth_m':float(z[0]),'distance_m':float(d[0]),
                        'qualified':obj['id'] in action['object_ids']})
                actions.append({'anchor_index':anchor_index,'yaw_rad':action['yaw_rad'],
                                'object_ids':action['object_ids'],'projected':projected})
        inverse={obj['id']:[] for obj in scene.get('objects',[])}
        votes=[]
        for index,directions in candidates.supports.items():
            union=set().union(*directions.values())
            votes.append([index,len(union)])
            for obj in union: inverse[obj].append(index)
        methods.append({'title':title,'anchors':result['anchors'],'actions':actions,
            'candidate_count':len(candidates.station_indices),'votes':votes,'inverse':inverse,
            'uncovered_object_ids':result['uncovered_object_ids'],
            'generation_seconds':result['statistics']['generation_seconds']})
    return {'scene_name':scene.get('name','unnamed'),'grid':scene['grid'],'objects':scene.get('objects',[]),
        'camera':camera_config,'methods':methods,
        'stations':[[s.index,*s.position] for s in stations],
        'regions':[{'id':region['id'],'route':[[p['x_m'],p['y_m']] for p in region['waypoints']]} for region in path['regions']],
        'path_length_m':path['statistics']['total_path_length_m']}


def render_png(data,destination):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/falhv2-anchor-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    grid=data['grid']; rows=grid['rows']; resolution=grid['resolution']; x0,y0=grid['origin']
    image=np.array([[0 if c=='#' else 1 if c=='?' else 2 for c in row] for row in rows])
    targets=np.array([obj['position'] for obj in data['objects']]).reshape(-1,3)
    labels=['Object-driven','Uniform + refinement','Random + refinement']
    fig,axes=plt.subplots(1,3,figsize=(15,5.4),layout='constrained')
    for ax,method,label in zip(axes,data['methods'],labels):
        ax.imshow(image,cmap=ListedColormap(['#29394e','#b3bfcc','#f4f7f9']),vmin=0,vmax=2,
                  extent=(x0,x0+len(rows[0])*resolution,y0,y0+len(rows)*resolution))
        if len(targets):ax.scatter(targets[:,0],targets[:,1],s=22,c='#11896a',edgecolors='white',linewidths=.6,zorder=3)
        for i,anchor in enumerate(method['anchors'],1):
            x,y=anchor['position']; ax.scatter([x],[y],s=70,marker='D',c='#c12798',edgecolors='white',zorder=5)
            ax.text(x+.07,y+.09,f'A{i}',color='#a60075',weight='bold',fontsize=9)
            for observation in anchor['observations']:
                yaw=observation['yaw_rad']; ax.arrow(x,y,.4*math.cos(yaw),.4*math.sin(yaw),width=.012,
                    head_width=.08,length_includes_head=True,color='#c12798',zorder=4)
        ax.set_title(f'{label}\n{len(method["anchors"])} stations / {len(method["actions"])} photos',fontsize=11)
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.suptitle('Same path and 3D point model | FOV + range only; occlusion not evaluated',fontsize=11)
    fig.savefig(destination,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,default=Path(__file__).with_name('fixtures')/'user_map_20261014.json')
    parser.add_argument('--robot',type=Path,default=ROOT/'config/robot.example.json')
    parser.add_argument('--camera',type=Path,default=ROOT/'config/camera.d435i.json')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/anchor_experiment/visualization.html')
    args=parser.parse_args()
    try:
        data=build_data(json.loads(args.scene.read_text()),json.loads(args.robot.read_text()),json.loads(args.camera.read_text()),args.seed)
        template=Path(__file__).with_name('viewer.html').read_text()
        # Escape HTML parser delimiters while retaining arbitrary object IDs as text.
        payload=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(template.replace('__ANCHOR_DATA__',payload))
        render_png(data,args.output.with_suffix('.png'))
    except (ValueError,OSError,KeyError) as exc:parser.exit(2,f'可视化生成失败：{exc}\n')
    print(args.output);print(args.output.with_suffix('.png'))


if __name__=='__main__':main()
