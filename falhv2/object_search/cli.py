import argparse
import json
from pathlib import Path
from uuid import uuid4

from falhv2.model_clients.valen import Valen
from .search import find_object


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
