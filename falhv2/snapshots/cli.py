import argparse
from collections import Counter
import json
from pathlib import Path
from uuid import uuid4

from falhv2.model_clients.valen import Valen
from .search import find_object
from .store import frame_from_files, initialize, lock, read_json, record, view_path


def snapshots(argv, *, map_path, plan_path, store):
    parser = argparse.ArgumentParser(description='离线快照库，以到访事件采集 first / last')
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('init', 'record', 'status', 'demo'):
        p = sub.add_parser(name)
        p.add_argument('--store', type=Path, default=store, required=(name == 'demo'))
        if name in ('init', 'demo'):
            p.add_argument('--map', type=Path, default=map_path)
            p.add_argument('--plan', type=Path, default=plan_path)
        if name == 'record':
            p.add_argument('--station', type=int, required=True)
            p.add_argument('--yaw', type=int, required=True)
            p.add_argument('--frame', type=Path, required=True, help='包含 rgb.png、depth_m.npy、meta.json 的目录')
        if name == 'demo':
            for image in ('mug', 'fruit', 'court'):
                p.add_argument('--' + image, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == 'init':
            initialize(args.store, args.map, args.plan)
        if args.action == 'record':
            frame, visit, simulated = frame_from_files(args.frame)
            print(record(args.store, args.station, args.yaw, visit, frame, simulated=simulated))
        if args.action == 'demo':
            from .demo import make_demo
            make_demo(args.store, args.map, args.plan, args.mug, args.fruit, args.court)
        index = read_json(args.store / 'index.json')
        summary = {'planned_views': len(index['views']), 'captured_views': 0, 'revisited_views': 0}
        planned, visited = Counter(), Counter()
        with lock(args.store, False):
            for v in index['views']:
                station = v['station_index']
                planned[station] += 1
                parent = args.store / view_path(station, v['yaw_index'])
                if (parent / 'first').is_dir():
                    visited[station] += 1
                    summary['captured_views'] += 1
                summary['revisited_views'] += (parent / 'last').is_dir()
        summary.update(planned_stations=len(planned), visited_stations=len(visited),
                       complete_stations=sum(visited[s] == n for s, n in planned.items()))
        print(json.dumps(summary, ensure_ascii=False))
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        parser.exit(2, f'快照操作失败：{exc}\n')
    return 0


def search(argv, *, store, endpoint, model, output):
    parser = argparse.ArgumentParser(description='使用 Valen 在已访问视角查找目标')
    parser.add_argument('--query', required=True, help='待查找物品的文字描述')
    parser.add_argument('--store', type=Path, default=store)
    parser.add_argument('--endpoint', default=endpoint, help='完整 /v1/decisions 地址')
    parser.add_argument('--model', default=model)
    parser.add_argument('--output', type=Path, help='本次结果的新目录')
    parser.add_argument('--predicate-min', default=.65, type=float)
    parser.add_argument('--bbox-min', default=.5, type=float)
    args = parser.parse_args(argv)
    dest = args.output or output / uuid4().hex
    try:
        result = find_object(args.store, args.query, Valen(args.endpoint, args.model), dest,
                             args.predicate_min, args.bbox_min)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        parser.exit(2, f'查找失败：{exc}\n')
    print(json.dumps({k: result[k] for k in ('status', 'captured_views', 'planned_views', 'match')},
                     ensure_ascii=False, indent=2))
    print(f'结果：{dest / "result.json"}')
    return 0
