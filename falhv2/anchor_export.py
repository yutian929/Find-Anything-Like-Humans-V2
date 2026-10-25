"""Serialize planning results and build the existing offline HTML/SVG views."""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

try:
    from . import config as settings
    from .regions_extractor import diagnostic_svg
    from .observation_model import PointCamera
    from .anchor_candidates import object_heatmaps
except ImportError:
    import config as settings
    from regions_extractor import diagnostic_svg
    from observation_model import PointCamera
    from anchor_candidates import object_heatmaps


def render_svg(scene,regions,result,stations,candidates):
    import html, math
    svg=diagnostic_svg(scene,regions)
    grid=regions['grid']; scale=36; pad=32
    def screen(position):
        return (pad+(position[0]-grid['origin'][0])/grid['resolution']*scale,
                pad+(grid['height']-(position[1]-grid['origin'][1])/grid['resolution'])*scale)
    shapes=['<g id="anchor-observation-overlay">']
    by_id={s.index:s for s in stations}
    totals = defaultdict(float)
    for entries in object_heatmaps(candidates, [o['id'] for o in scene.get('objects', [])]).values():
        for index, score, _, units in entries:
            totals[index] += units/1_000_000
    for i,score in totals.items():
        x,y=screen(by_id[i].position)
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#ff9f1c" opacity="{min(.7,.1+.06*score):.2f}"><title>weighted score {score:.6f}</title></circle>')
    missing=set(result['uncovered_object_ids'])
    for obj in scene.get('objects',[]):
        x,y=screen(obj['position']); color='#e63946' if obj['id'] in missing else '#2a9d8f'
        name=html.escape(obj['id'])
        if 'aabb' in obj:
            box=obj['aabb']; bx,by=screen([box['min'][0],box['max'][1]])
            bw=(box['max'][0]-box['min'][0])/grid['resolution']*scale
            bh=(box['max'][1]-box['min'][1])/grid['resolution']*scale
            shapes.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bh:.2f}" fill="{color}" fill-opacity="0.2" stroke="{color}"/>')
            name+=f" z={box['min'][2]:.2f}–{box['max'][2]:.2f}m"
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="13" fill="{color}" stroke="white" stroke-width="3"/><text x="{x+15:.2f}" y="{y:.2f}" font-size="20">{name}</text>')
    for anchor in result['anchors']:
        x,y=screen(anchor['position'])
        shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="18" fill="none" stroke="#d000a6" stroke-width="5"/>')
        for action in anchor['observations']:
            yaw=action['yaw_rad']; length=65
            shapes.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x+length*math.cos(yaw):.2f}" y2="{y-length*math.sin(yaw):.2f}" stroke="#d000a6" stroke-width="5"/>')
    shapes.append('</g>')
    return svg.replace('</svg>','\n'.join(shapes)+'\n</svg>')


def viewer_data(scene, camera_config, result, regions, stations, candidates):
    camera = PointCamera(camera_config)
    methods = []
    def projected_actions(anchors):
        actions=[]
        for anchor_index,anchor in enumerate(anchors):
            for action in anchor['observations']:
                projected=[]
                for obj in scene.get('objects',[]):
                    u,v,z,d=camera.project(np.array([anchor['position']]),np.array([action['yaw_rad']]),obj['position'])
                    projected.append({'id':obj['id'],'u':float(u[0]) if math.isfinite(u[0]) else None,
                        'v':float(v[0]) if math.isfinite(v[0]) else None,'depth_m':float(z[0]),'distance_m':float(d[0]),
                        'qualified':obj['id'] in action['object_ids'],
                        'metrics':action.get('object_metrics', {}).get(obj['id'])})
                actions.append({'anchor_index':anchor_index,'yaw_rad':action['yaw_rad'],
                                'object_ids':action['object_ids'],'projected':projected})
        return actions
    inverse={obj['id']:[] for obj in scene.get('objects',[])}
    heatmaps = object_heatmaps(candidates, inverse)
    totals = defaultdict(int)
    for obj, entries in heatmaps.items():
        for index, score, _, units in entries:
            inverse[obj].append(index)
            totals[index] += units
    for title, selection, replay in [('最少锚点 / 最少照片', result, False),
                                      ('基线：加权贪心', result['baseline'], True)]:
        methods.append({'title': title, 'anchors': selection['anchors'],
            'actions': projected_actions(selection['anchors']), 'replay_enabled': replay,
            'candidate_count': len(candidates.station_indices),
            'votes': [[index, units/1_000_000] for index, units in sorted(totals.items())],
            'inverse': inverse, 'object_heatmaps': heatmaps,
            'vote_trace': selection['vote_trace'] if replay else [],
            'uncovered_object_ids': selection['uncovered_object_ids'],
            'generation_seconds': result['statistics']['generation_seconds']})
    return {'scene_name':scene.get('name','unnamed'),'grid':scene['grid'],'objects':scene.get('objects',[]),
    'camera':camera_config,'methods':methods,
    'optimization':result['optimization'], 'selection_method':result['selection_method'],
    'baseline_cost':result['baseline_cost'], 'selected_cost':result['selected_cost'],
    'observation_model':result['observation_model'],'observation_settings':result['observation_settings'],
    'object_diagnostics':result['object_diagnostics'],
    'yaws_rad':candidates.yaws,
    'heatmap_fields':['station_index', 'best_qualified_score', 'yaw_index', 'vote_units'],
    'stations':[[s.index,*s.position] for s in stations],
    'region_count':len(regions['regions'])}

def save_supports(destination, scene, stations, candidates):
    """Export qualified supports, continuous scores and per-object heatmaps."""
    relation = {
        'schema_version': 4, 'station_order': 'region_id_row_column',
        'scores': candidates.scores,
        'object_heatmaps': object_heatmaps(candidates, [o['id'] for o in scene.get('objects', [])]),
        'heatmap_fields': ['station_index', 'best_qualified_score', 'yaw_index', 'vote_units'],
        'object_points': scene.get('objects', []), 'yaws_rad': candidates.yaws,
        'evaluated_station_indices': candidates.station_indices,
        'positive_stations': [{'index': s.index, 'region_id': s.region_id, 'cell': list(s.cell),
                              'position': list(s.position)}
                             for s in stations if s.index in candidates.supports],
        'supports': {str(i): {str(y): sorted(ids) for y, ids in directions.items()}
                     for i, directions in candidates.supports.items()},
    }
    destination.write_text(json.dumps(relation, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def export_plan(output, scene, camera_config, result, regions, stations, candidates, *, include_supports=True):
    """Write the plan and views without rerunning observation or selection."""
    output = Path(output)
    if output.suffix.lower() != ".json":
        raise ValueError("--output must have a .json suffix")
    payload = json.dumps(viewer_data(scene, camera_config, result, regions, stations, candidates),
                         ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = settings.ANCHOR_VIEWER_TEMPLATE.read_text(encoding='utf-8').replace('__ANCHOR_DATA__', payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    output.with_suffix('.svg').write_text(render_svg(scene, regions, result, stations, candidates), encoding='utf-8')
    output.with_suffix('.html').write_text(page, encoding='utf-8')
    if include_supports:
        save_supports(output.with_suffix('.supports.json'), scene, stations, candidates)


def export_safe_regions(scene, regions, output, svg):
    """Export the already computed A1 result, with the standalone A1 format."""
    output, svg = Path(output), Path(svg)
    output.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(regions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    svg.write_text(diagnostic_svg(scene, regions) + '\n', encoding='utf-8')
