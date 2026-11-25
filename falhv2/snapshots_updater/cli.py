import argparse
import json
from pathlib import Path

from .store import frame_from_files, initialize, record, status


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
        print(json.dumps(status(args.store), ensure_ascii=False))
    except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
        parser.exit(2, f'快照操作失败：{exc}\n')
    return 0
