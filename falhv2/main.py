"""FALHv2 离线任务入口。使用 --task TASK --help 查看任务参数。"""
import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falhv2.config import load_config

TASKS = ("anchors-plan", "safe-regions", "map-editor", "camera-parameters", "find-object")


def plan(scene, robot_config, camera_config, **options):
    """Python entry point for callers that rely on the project defaults."""
    from falhv2.anchor_planners.planning import plan as plan_scene
    if options.get("project_config") is None:
        options["project_config"] = load_config()
    return plan_scene(scene, robot_config, camera_config, **options)


def run_task(task, argv, config):
    """Pass only the parameters a tool needs; tools never load project config."""
    if task == 'anchors-plan':
        from falhv2.anchor_planners.cli import main as run
        return run(argv, config)
    if task == 'find-object':
        from falhv2.object_search.cli import search
        return search(argv, store=config.SNAPSHOT_STORE, endpoint=config.VALEN_ENDPOINT,
                      model=config.VALEN_MODEL, output=config.OUTPUT_DIR / 'searches')
    if task == 'safe-regions':
        from falhv2.anchor_planners.regions_extractor import main as run
        return run(argv, scene_path=config.MAP_PATH, robot_path=config.ROBOT_PATH,
                   output_path=config.SAFE_REGIONS_PATH, svg_path=config.SAFE_REGIONS_SVG_PATH)
    if task == 'map-editor':
        from falhv2.tools.map_editor import main as run
        return run(argv, page_path=config.EDITOR_PAGE, map_path=config.MAP_PATH)
    if task == 'camera-parameters':
        from falhv2.tools.camera_parameters import main as run
        return run(argv, output_path=config.CAMERA_EXPORT_PATH, camera_path=config.CAMERA_PATH)
    raise ValueError(f'Unknown task: {task}')


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--config', type=Path, help='项目配置文件')
    parser.add_argument('--task', choices=TASKS, help='选择任务，默认 anchors-plan')
    options, remaining = parser.parse_known_args(argv)
    if options.task is None and remaining in (['--help'], ['-h']):
        parser.print_help()
        return 0
    try:
        config = load_config(options.config)
    except (OSError, ValueError, SyntaxError) as exc:
        parser.error(str(exc))
    return run_task(options.task or 'anchors-plan', remaining, config)


if __name__ == '__main__':
    raise SystemExit(main())
