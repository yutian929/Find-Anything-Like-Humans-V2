"""Read a scene, plan anchor views and export results."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

from .anchor_export import export_plan, export_safe_regions
from .planning import plan
from .regions_extractor import PlanningError

LOGGER = logging.getLogger(__name__)


def read_input_bytes(path):
    try:
        return Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f'Input file missing: {path}; check --config or explicit input paths') from exc


def read_input(path):
    return json.loads(read_input_bytes(path))


def main(argv, config):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=config.MAP_PATH)
    parser.add_argument('--robot', type=Path, default=config.ROBOT_PATH)
    parser.add_argument('--camera', type=Path, default=config.CAMERA_PATH)
    parser.add_argument('--yaw-step-deg', type=float, default=None,
                        help=f'底盘拍摄偏航间隔，单位度，默认读取 config/defaults.py（{config.YAW_STEP_DEG}）')
    parser.add_argument('--selection-time-limit-s', type=float, default=None,
                        help='全部 MILP 调用（含复核重选）的总时间预算；默认读取 config/defaults.py，不含观察计算与建模')
    parser.add_argument('--save-supports', action=argparse.BooleanOptionalAction, default=True,
                        help='导出合格支持关系、连续得分及热力图（默认开启）')
    parser.add_argument('--ray-backend', choices=('numpy', 'open3d'), default='numpy',
                        help='候选射线后端；Open3D 为 Float32 实验后端，最终复核始终使用 NumPy Float64')
    parser.add_argument('--observation-workers', type=int, default=None,
                        help='逐物体候选计算的 worker 数；默认读取 config/defaults.py；1 恢复串行')
    parser.add_argument('--observation-executor', choices=('threads', 'processes'), default=None,
                        help='候选并行方式；默认读取 config/defaults.py')
    parser.add_argument('--output', type=Path, default=config.ANCHOR_PLAN_PATH)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix.lower() != '.json':
            raise ValueError('--output must have a .json suffix')
        scene_bytes = read_input_bytes(args.scene)
        scene = json.loads(scene_bytes)
        robot = read_input(args.robot)
        camera = read_input(args.camera)
        result, regions, stations, candidates = plan(scene, robot, camera, yaw_step_deg=args.yaw_step_deg,
                                                       selection_time_limit_s=args.selection_time_limit_s,
                                                       ray_backend=args.ray_backend,
                                                       observation_workers=args.observation_workers,
                                                       observation_executor=args.observation_executor,
                                                       project_config=config)
        result['map_sha256'] = hashlib.sha256(scene_bytes).hexdigest()
        LOGGER.info('5/5 导出安全区域、计划与离线可视化')
        export_plan(args.output, scene, camera, result, regions, stations, candidates,
                    include_supports=args.save_supports, template_path=config.ANCHOR_VIEWER_TEMPLATE)
        # Respect configured A1 paths for the default run; keep custom runs together.
        if args.output.resolve() == config.ANCHOR_PLAN_PATH.resolve():
            regions_output, regions_svg = config.SAFE_REGIONS_PATH, config.SAFE_REGIONS_SVG_PATH
        else:
            regions_output = args.output.with_suffix('.safe_regions.json')
            regions_svg = args.output.with_suffix('.safe_regions.svg')
        export_safe_regions(scene, regions, regions_output, regions_svg)
    except (ValueError, OSError, KeyError, TypeError, PlanningError) as exc:
        parser.exit(2, f'锚点规划失败：{exc}\n')
    print(json.dumps({k: result[k] for k in ('status', 'anchor_count', 'photo_count', 'coverage_fraction', 'uncovered_object_ids', 'baseline_cost', 'selected_cost', 'optimality')}, ensure_ascii=False, indent=2))
    if 'high_density_verification' in result['optimization']:
        audit = result['optimization']['high_density_verification']
        print(f"高密度复核：{audit['final']['status']}；重选次数：{len(audit['repairs'])}；缺口：{audit['final']['uncovered_object_ids']}")
    print(f'锚点计划：{args.output}；交互页面：{args.output.with_suffix(".html")}')
    return 0
