"""Build compact versionable results and report figures from local raw benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR','/tmp/falhv2-anchor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

REPO=Path(__file__).resolve().parents[2]
DEFAULT=REPO/'falhv2/outputs/anchor_experiment/benchmark'


def aggregate(rows):
    keys=('anchor_count','photo_count','station_count','point_pose_tests')
    result={key:{'min':min(row[key] for row in rows),'median':statistics.median(row[key] for row in rows),
                 'max':max(row[key] for row in rows)} for key in keys}
    result['median_generation_seconds']=statistics.median(row['median_generation_seconds'] for row in rows)
    result['median_selection_seconds']=statistics.median(row['median_selection_seconds'] for row in rows)
    result['seeds']=len(rows)
    result['missed_coverable_runs']=sum(bool(row.get('missed_but_full_pool_coverable',[])) for row in rows)
    result['uncovered_object_counts']=[len(row['uncovered_object_ids']) for row in rows]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=REPO/'docs/falhv2/anchor_driver_summary_20261014.json')
    args=parser.parse_args()
    raw=json.loads((args.input/'comparison.json').read_text())
    summary={key:value for key,value in raw.items() if key!='cases'}
    summary['raw_results_sha256']=hashlib.sha256((args.input/'comparison.json').read_bytes()).hexdigest()
    summary['cases']=[]
    for case in raw['cases']:
        current={key:value for key,value in case.items() if key!='methods'}
        current['methods']={}
        for name,values in case['methods'].items():
            if 'runs' in values: current['methods'][name]=aggregate(values['runs'])
            else: current['methods'][name]={key:value for key,value in values.items()
                         if key not in ('driver_stats','generation_seconds','selection_seconds')}
        summary['cases'].append(current)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    assets=args.output.parent/'assets'; assets.mkdir(exist_ok=True)
    cases={case['name']:case for case in summary['cases']}
    names=['object_all','position_exhaustive','uniform_refined','random_refined']
    labels=['Object / all','Position / all','Uniform + refine','Random + refine']
    colors=['#087f8c','#8d99ae','#e59a24','#a25cad']
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,case_name in enumerate(('user_map','large_512')):
        case=cases[case_name]; data=case['methods']
        times=[data[name]['median_generation_seconds']*1000 for name in names]
        axes[row,0].barh(labels,times,color=colors)
        axes[row,0].set_xscale('log');axes[row,0].set_xlabel('Candidate generation (ms, log scale)')
        axes[row,0].set_xlim(max(.1,min(times)/1.5), max(times)*2)
        axes[row,0].set_title(f'{case_name}: {case["station_count"]:,} path stations / {case["object_count"]} targets')
        for y,time in enumerate(times): axes[row,0].text(time*1.08,y,f'{time:.1f}',va='center',fontsize=9)
        counts=[];lo=[];hi=[]
        for name in names:
            count=data[name]['anchor_count']
            if isinstance(count,dict): counts.append(count['median']);lo.append(count['median']-count['min']);hi.append(count['max']-count['median'])
            else: counts.append(count);lo.append(0);hi.append(0)
        axes[row,1].barh(labels,counts,xerr=[lo,hi],color=colors,capsize=4)
        axes[row,1].set_xlabel('Selected stations (random: median and seed range)')
        axes[row,1].set_xlim(0,max(counts)+max(hi)+2)
        axes[row,1].set_title('Selected stations; same observation model')
    fig.savefig(assets/'anchor_driver_comparison_20261014.png',dpi=150)
    plt.close(fig)
    specs=json.loads((args.input/'fixtures.json').read_text())
    _,scene,robot,camera=next(row for row in specs if row[0]=='user_map')
    plan=json.loads((args.input/'user_object_plan.json').read_text())
    grid=scene['grid'];cells=np.array([[0 if c=='#' else 1 if c=='?' else 2 for c in row] for row in grid['rows']])
    h,w=cells.shape;resolution=grid['resolution'];x0,y0=grid['origin']
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    ax.imshow(cells,cmap=ListedColormap(['#27364a','#aeb9c7','#f3f7f7']),vmin=0,vmax=2,
              extent=(x0,x0+w*resolution,y0,y0+h*resolution))
    points=np.array([o['position'] for o in scene['objects']])
    ax.scatter(points[:,0],points[:,1],s=35,c='#178f6c',edgecolors='white',linewidths=.7,label='Target center')
    for obj in scene['objects']: ax.text(obj['position'][0]+.03,obj['position'][1]+.03,obj['id'],fontsize=7,color='#174c39',
                                        bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':.5})
    for i,anchor in enumerate(plan['anchors'],1):
        x,y=anchor['position'];ax.scatter([x],[y],marker='D',s=90,c='#c92397',edgecolors='white')
        ax.text(x+.08,y+.08,f'A{i} / R{anchor["region_id"]}',fontsize=10,weight='bold',color='#a60077')
        for action in anchor['observations']:
            theta=action['yaw_rad'];ax.arrow(x,y,.55*np.cos(theta),.55*np.sin(theta),width=.012,
                head_width=.1,length_includes_head=True,color='#c92397',alpha=.8)
    ax.scatter([],[],marker='D',s=90,c='#c92397',label='Selected station / base yaw')
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.legend(loc='lower left',fontsize=8)
    ax.set_title(f'User map: {plan["anchor_count"]} stations / {plan["photo_count"]} photos / 28 point targets\nFOV + range only; occlusion not evaluated',fontsize=11)
    fig.savefig(assets/'anchor_user_plan_20261014.png',dpi=150);plt.close(fig)
    print(args.output)


if __name__=='__main__': main()
