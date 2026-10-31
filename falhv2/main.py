"""统一入口：持续更新地图，按用户指令重规划锚点。

运行 python -B falhv2/main.py 启动实时系统；--task anchors 保留离线选点。
地图编辑与真机相机标定是独立的输入准备步骤。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from time import perf_counter

# Keep script and package execution on the same canonical package imports.
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falhv2.config import defaults as settings
from falhv2.anchor_planners import anchor_verification as verification
from falhv2.anchor_planners.regions_extractor import plan as find_safe_regions, PlanningError
from falhv2.anchor_planners.observation_model import PointCamera, BoxObserver, validate_objects
from falhv2.anchor_planners.anchor_candidates import stations_from_regions, yaw_grid, generate_candidates
from falhv2.anchor_planners.anchor_selector import select_anchors
from falhv2.anchor_planners.anchor_export import export_plan, export_safe_regions

LOGGER = logging.getLogger(__name__)


def load_config(path=None):
    """Select a Python config in the entry point; omitted fields inherit defaults."""
    if path is None:
        return settings
    import runpy
    from types import SimpleNamespace
    path = Path(path).expanduser().resolve()
    if path.suffix != '.py':
        raise ValueError('--config must name a Python configuration file')
    selected = runpy.run_path(str(path))
    values = {key: value for key, value in vars(settings).items() if key.isupper()}
    for key, value in selected.items():
        if not key.isupper():
            continue
        if isinstance(values.get(key), Path) and isinstance(value, (str, Path)):
            value = Path(value).expanduser()
            value = value if value.is_absolute() else path.parent / value
        if isinstance(value, dict) and isinstance(values.get(key), dict):
            value = {**values[key], **value}
        values[key] = value
    return SimpleNamespace(**values)


def read_input(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ValueError(f'Input file missing: {path}; check --config or explicit input paths') from exc


def configured_camera(camera_config, config=None):
    config = settings if config is None else config
    return {**camera_config, 'observation_quality': {
        **config.OBSERVATION_DEFAULTS, **camera_config.get('observation_quality', {})}}


def planning_audit(config=None):
    config = settings if config is None else config
    return verification.audit_parameters(
        sample_grid=config.AUDIT_SAMPLE_GRID, max_retries=config.MAX_AUDIT_RETRIES,
        same_cost_time_limit_s=config.SAME_COST_REPAIR_TIME_LIMIT_S)


def run_task(task, argv, config):
    """Pass only the parameters a tool needs; tools never load project config."""
    if task == 'live':
        return run_live(argv, config)
    if task == 'cubify-preview':
        from falhv2.tools.cubify_preview import main as run
        return run(argv, model_config=dict(config.CUBIFY), output_dir=config.OUTPUT_DIR)
    if task == 'build-map':
        from falhv2.tools.build_map import main as run
        return run(argv, model_config=dict(config.CUBIFY), robot_config=dict(config.ROBOT_SOURCE),
                   mapping_config=dict(config.MAP_UPDATE), output_dir=config.OUTPUT_DIR)
    if task == 'cubify-server':
        from falhv2.tools.cubify_server import main as run
        return run(argv, model_config=dict(config.CUBIFY))
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


def plan(scene, robot_config, camera_config, *, yaw_step_deg=None, selection_time_limit_s=None,
         ray_backend='numpy', observation_workers=None, observation_executor=None, project_config=None):
    """Select object-driven anchors directly from safe regions; no route is built.

    Targets are world-frame 3D centers; base positions are (x, y, 0).
    A station may take multiple yaw-directed photos. AABB coverage uses clipped
    framing, sampled occlusion/pixels and radial score; point maps use legacy FOV/range.
    """
    cfg = settings if project_config is None else project_config
    yaw_step_deg = cfg.YAW_STEP_DEG if yaw_step_deg is None else yaw_step_deg
    selection_time_limit_s = cfg.SELECTION_TIME_LIMIT_S if selection_time_limit_s is None else selection_time_limit_s
    audit_parameters = planning_audit(cfg)
    observation_workers = cfg.OBSERVATION_WORKERS if observation_workers is None else observation_workers
    observation_executor = cfg.OBSERVATION_EXECUTOR if observation_executor is None else observation_executor
    if isinstance(observation_workers, bool) or not isinstance(observation_workers, int) or observation_workers < 1:
        raise ValueError('observation workers must be a positive integer')
    if observation_executor not in ('threads', 'processes'):
        raise ValueError('observation executor must be threads or processes')
    if ray_backend not in ('numpy', 'open3d'):
        raise ValueError('ray_backend must be numpy or open3d')
    if selection_time_limit_s is not None and (isinstance(selection_time_limit_s, bool)
            or not isinstance(selection_time_limit_s, (int, float))
            or not math.isfinite(selection_time_limit_s) or selection_time_limit_s <= 0):
        raise ValueError('selection time limit must be finite and positive')
    objects = validate_objects(scene.get("objects", []))
    camera = PointCamera(configured_camera(camera_config, cfg))
    yaw_grid(yaw_step_deg)
    started = perf_counter()
    LOGGER.info('1/5 提取安全区域与候选站位')
    regions = find_safe_regions(scene, robot_config)
    stations = stations_from_regions(regions)
    preprocessing = perf_counter() - started
    LOGGER.info('2/5 计算观察候选：%d 个站位、%d 个目标', len(stations), len(objects))
    candidates = generate_candidates(stations, objects, camera, yaw_step_deg=yaw_step_deg,
                                     ray_backend=ray_backend, workers=observation_workers,
                                     executor=observation_executor)
    started = perf_counter()
    LOGGER.info('3/5 加权基线与两阶段整数优化')
    selected = select_anchors(candidates, stations, [o['id'] for o in objects],
                              time_limit_s=selection_time_limit_s)
    LOGGER.info('4/5 复核最终观察，必要时重选')
    # Native Float32 candidates must still pass independent Float64 verification.
    observer = BoxObserver(camera, objects, ray_backend='numpy') if objects and 'aabb' in objects[0] else None
    if observer:
        selected = verification.refine_after_audit(selected, candidates, stations, objects, observer,
                                                   selection_time_limit_s, parameters=audit_parameters)
    selection = perf_counter() - started
    selected = verification.verify_plan(selected, objects, camera, observer, sample_grid=audit_parameters["sample_grid"])
    result = {
        'schema_version': 5, 'method': 'object_driven_lexicographic',
        'candidate_scope': 'all_safe_cell_centers_in_all_regions',
        'station_order': 'region_id_row_column',
        'scene_name': scene.get('name', 'unnamed'),
        'observation_model': 'aabb_weighted_region' if observer else 'legacy_point',
        'observation_scope': ('Solid AABBs; clipped projected framing, sampled occlusion and visible pixel area; mount-aware radial weight; recognition not evaluated'
                              if observer else '3D point centers; FOV and Euclidean distance; occlusion and appearance not evaluated'),
        'observation_settings': observer.settings.to_dict() if observer else None,
        'object_diagnostics': candidates.diagnostics,
        'selection_rule': 'min_anchor_count_then_photo_count',
        'score_definition': (f'radial_weight_{observer.settings.radial_inner_weight:g}_to_{observer.settings.radial_outer_weight:g}_times_visible_fraction'
                             if observer else 'legacy_unit_votes'),
        'region_definition': ('per_bearing_and_yaw_optical_center_distance_and_AABB_frustum_overlap_envelope'
                              if observer else 'legacy_point_no_radial_weight'),
        'uncovered_diagnosis_scope': 'No support in the evaluated discrete station/yaw pool; not continuous-space infeasibility',
        'configuration': {'robot': robot_config, 'camera': camera_config, 'yaw_step_deg': yaw_step_deg,
                          'selection_time_limit_s': selection_time_limit_s, 'audit': audit_parameters,
                          'candidate_ray_backend': ray_backend if observer else None,
                          'verification_ray_backend': 'numpy' if observer else None,
                          'observation_workers': observation_workers,
                          'observation_executor': observation_executor},
        'statistics': {'safe_regions_and_station_seconds': preprocessing, 'selection_seconds': selection,
                       'region_count': len(regions['regions']), 'safe_station_count': len(stations),
                       **candidates.stats},
        **selected,
    }
    return result, regions, stations, candidates


def write_json_atomic(path, payload):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class LiveSystem:
    """Fast map updates and serialized slow planning, sharing immutable snapshots."""

    def __init__(self, config, output_dir, robot_options, *, timeout_s=5., update_interval_s=.2):
        import threading
        import queue
        self.config = config
        self.output_dir = Path(output_dir)
        self.robot_options = dict(robot_options)
        self.timeout_s = timeout_s
        self.update_interval_s = update_interval_s
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.requests = queue.Queue()
        self.threads = []
        self.serial = 0
        self.scene = None
        self.view = None
        self.fast = dict(status='starting', error=None, updated_at=None)
        self.slow = dict(status='idle', error=None, source_revision=None, last_result=None)

    def start(self):
        import threading
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, target in [('map-updates', self._fast_loop), ('anchor-replanning', self._slow_loop)]:
            thread = threading.Thread(name=name, target=target)
            self.threads.append(thread)
            thread.start()

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join()

    def state(self):
        import copy
        with self.lock:
            return copy.deepcopy(dict(fast=self.fast, slow=self.slow, view=self.view))

    def request_replan(self):
        import copy
        with self.lock:
            if self.stop.is_set():
                return False, '系统正在退出'
            if self.scene is None or self.fast['status'] != 'running':
                return False, '还没有可用于重规划的最新地图，请等待采集恢复'
            if self.slow['status'] in ('queued', 'running'):
                return False, '已有重规划正在执行，请等待完成'
            self.serial += 1
            scene = copy.deepcopy(self.scene)
            revision = scene['mapping']['revision']
            self.slow.update(status='queued', error=None, source_revision=revision,
                             request_id=self.serial)
            self.requests.put((self.serial, scene))
            return True, f'已冻结地图版本 {revision}，开始重规划'

    def _fast_loop(self):
        import time
        import base64
        client = None
        try:
            import cv2
            import numpy as np
            from falhv2.model_clients import build_cubify_anything
            from falhv2.robot_adapters.anchor_map import AnchorMapRobot
            from falhv2.map_maintainers.depth_replace import DepthReplacementMap
            from falhv2.tools.cubify_preview import draw_cuboid, color
            builder = DepthReplacementMap(**self.config.MAP_UPDATE)
            client = build_cubify_anything(dict(self.config.CUBIFY))
            with AnchorMapRobot(**self.robot_options) as robot:
                while not self.stop.is_set():
                    started = time.monotonic()
                    try:
                        builder.set_grid(robot.grid_snapshot(timeout_s=self.timeout_s))
                        frame = robot.capture(timeout_s=self.timeout_s)
                        result = client.detect(frame)
                        update = builder.integrate(frame, result)
                        mask = builder.coverage
                        spans = []
                        for row, values in enumerate(mask):
                            edges = np.flatnonzero(np.diff(np.r_[False, values, False]))
                            spans.extend([row, int(a), int(b)] for a, b in edges.reshape(-1, 2))
                        rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                        for index, box in enumerate(result.cuboids):
                            draw_cuboid(rgb, box.corners_camera_m, frame.intrinsics, color(index),
                                        f'#{index+1} {box.confidence:.2f}')
                        ok, image = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if not ok:
                            raise ValueError('RGB preview encoding failed')
                        scene = builder.to_scene()
                        builder.save(self.output_dir/'map.json')
                        pose = frame.T_map_base if frame.T_map_base is not None else frame.T_map_camera
                        view = dict(grid=scene['grid'], objects=scene['objects'], coverage=spans,
                                    revision=builder.revision, frame_id=frame.frame_id,
                                    timestamp_ns=frame.timestamp_ns,
                                    map_id=builder.grid.map_id, map_revision=builder.grid.map_revision,
                                    robot_position=pose[:2, 3].tolist(),
                                    camera_position=frame.T_map_camera[:2, 3].tolist(),
                                    camera_forward=frame.T_map_camera[:2, 2].tolist(),
                                    added_ids=update['added_ids'], removed_ids=update['removed_ids'],
                                    predictions=update['predictions'],
                                    skipped_predictions=len(update['skipped_detection_indices']),
                                    rgb='data:image/jpeg;base64,'+base64.b64encode(image).decode())
                        with self.lock:
                            self.scene, self.view = scene, view
                            self.fast.update(status='running', error=None, updated_at=time.time(),
                                             processing_s=time.monotonic()-started)
                    except Exception as exc:
                        LOGGER.warning('地图更新失败，保留上一份成功快照：%s', exc)
                        with self.lock:
                            self.fast.update(status='error', error=str(exc))
                        if self.stop.wait(1.):
                            break
                    self.stop.wait(self.update_interval_s)
        except Exception as exc:
            LOGGER.exception('快系统启动失败')
            with self.lock:
                self.fast.update(status='error', error=str(exc))
        finally:
            if client is not None:
                client.close()
            with self.lock:
                if self.fast['status'] != 'error':
                    self.fast['status'] = 'stopped'

    def _plan_snapshot(self, request_id, scene):
        folder = self.output_dir/'plans'/f'{request_id:04d}'
        write_json_atomic(folder/'scene.json', scene)
        robot = read_input(self.config.ROBOT_PATH)
        camera = read_input(self.config.CAMERA_PATH)
        result, regions, stations, candidates = plan(
            scene, robot, camera, project_config=self.config)
        source = dict(map_id=scene['mapping']['grid_metadata']['map_id'],
                      map_revision=scene['mapping']['grid_metadata']['map_revision'],
                      revision=scene['mapping']['revision'],
                      frame_id=scene['mapping']['last_frame_id'],
                      timestamp_ns=scene['mapping']['last_timestamp_ns'])
        result['source_map'] = source
        export_plan(folder/'anchor_plan.json', scene, camera, result, regions, stations,
                    candidates, template_path=self.config.ANCHOR_VIEWER_TEMPLATE)
        export_safe_regions(scene, regions, folder/'safe_regions.json', folder/'safe_regions.svg')
        return dict(request_id=request_id, source_map=source, status=result['status'],
                    anchor_count=result['anchor_count'], photo_count=result['photo_count'],
                    anchors=result['anchors'], coverage_fraction=result['coverage_fraction'],
                    url=f'/plans/{request_id:04d}/anchor_plan.html')

    def _slow_loop(self):
        import queue
        while not self.stop.is_set():
            try:
                request_id, scene = self.requests.get(timeout=.2)
            except queue.Empty:
                continue
            with self.lock:
                self.slow.update(status='running')
            try:
                result = self._plan_snapshot(request_id, scene)
                with self.lock:
                    self.slow.update(status='complete', error=None, last_result=result)
                LOGGER.info('重规划完成：地图版本 %s，%s 个锚点，%s 张照片',
                            result['source_map']['revision'], result['anchor_count'],
                            result['photo_count'])
            except Exception as exc:
                LOGGER.exception('重规划失败；快系统继续运行')
                with self.lock:
                    self.slow.update(status='error', error=str(exc))


def make_live_server(runtime, host, port, page_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import unquote, urlsplit
    import mimetypes

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, content, content_type):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

        def json(self, code, value):
            self.send(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                      'application/json; charset=utf-8')

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == '/api/state':
                return self.json(200, runtime.state())
            if path == '/':
                return self.send(200, Path(page_path).read_bytes(), 'text/html; charset=utf-8')
            if path.startswith('/plans/'):
                root = (runtime.output_dir/'plans').resolve()
                target = (runtime.output_dir/path.lstrip('/')).resolve()
                if root in target.parents and target.is_file():
                    mime = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
                    return self.send(200, target.read_bytes(), mime)
            self.send(404, b'Not found', 'text/plain')

        def do_POST(self):
            # Same-origin browser commands only. CLI input remains local.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://'+self.headers.get('Host', ''):
                return self.json(403, dict(error='Origin mismatch'))
            if urlsplit(self.path).path != '/api/replan':
                return self.json(404, dict(error='Unknown command'))
            accepted, message = runtime.request_replan()
            self.json(202 if accepted else 409, dict(accepted=accepted, message=message))

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def run_live(argv, config):
    import threading
    import uuid
    from datetime import datetime
    options = config.LIVE
    parser = argparse.ArgumentParser(description='快系统持续更新地图；输入 replan / 重规划 触发锚点选择')
    parser.add_argument('--host', default=options['host'])
    parser.add_argument('--port', type=int, default=options['port'])
    parser.add_argument('--map-id', default=config.ROBOT_SOURCE.get('map_id'))
    parser.add_argument('--map-revision', default=config.ROBOT_SOURCE.get('map_revision'))
    parser.add_argument('--endpoint', default=config.ROBOT_SOURCE['endpoint'])
    parser.add_argument('--chassis-url', default=config.ROBOT_SOURCE['chassis_url'])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('port must be between 0 and 65535')
    if (not math.isfinite(options['timeout_s']) or options['timeout_s'] <= 0
            or not math.isfinite(options['update_interval_s']) or options['update_interval_s'] < 0
):
        parser.error('Invalid LIVE timing')
    # Fresh application session labels; not a claim that the vendor provides SLAM revisions.
    session = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    robot_options = dict(map_id=args.map_id or 'live-'+session,
                         map_revision=args.map_revision or 'session-'+session,
                         endpoint=args.endpoint, chassis_url=args.chassis_url)
    output = args.output_dir or config.OUTPUT_DIR/('live_'+session)
    if output.exists() and any(output.iterdir()):
        parser.error('Live output directory must be new or empty')
    runtime = LiveSystem(config, output, robot_options, timeout_s=options['timeout_s'],
                         update_interval_s=options['update_interval_s'])
    server = make_live_server(runtime, args.host, args.port, config.LIVE_PAGE)
    browser_thread = threading.Thread(target=server.serve_forever, name='live-http', daemon=True)

    def console():
        while not runtime.stop.is_set():
            try:
                command = input().strip().lower()
            except EOFError:
                return  # Browser remains usable with redirected or closed stdin.
            if command in ('replan', '重规划', '重新规划'):
                print(runtime.request_replan()[1], flush=True)
            elif command in ('quit', 'exit', '退出'):
                runtime.stop.set()
            elif command:
                print('可用指令：replan / 重规划；quit / 退出。', flush=True)

    try:
        browser_thread.start()
        runtime.start()
        threading.Thread(target=console, name='live-console', daemon=True).start()
        address = '127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host
        print(f'实时地图：http://{address}:{server.server_port}\n输出目录：{output}\n'
              '输入 replan / 重规划，或点击网页按钮；输入 quit / 退出，或 Ctrl+C 停止。\n'
              '地图切换、重定位或回环坐标修正后，请重启本次会话。', flush=True)
        while not runtime.stop.wait(.25):
            pass
    except KeyboardInterrupt:
        runtime.stop.set()
    finally:
        print('正在退出，等待当前采集/推理或重规划完成……', flush=True)
        runtime.close()
        server.shutdown()
        server.server_close()
        browser_thread.join()
    return 0



def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', type=Path, help='Python config file; defaults to config/defaults.py')
    common.add_argument('--task', default='live',
                        choices=('live', 'anchors', 'cubify-preview', 'cubify-server', 'build-map',
                                 'safe-regions', 'map-editor', 'camera-parameters'))
    options, remaining = common.parse_known_args(argv)
    try:
        config = load_config(options.config)
    except (OSError, ValueError, SyntaxError) as exc:
        common.error(str(exc))
    if options.task != 'anchors':
        return run_task(options.task, remaining, config)
    parser = argparse.ArgumentParser(description=__doc__, parents=[common])
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
    args = parser.parse_args(remaining)
    try:
        if args.output.suffix.lower() != '.json':
            raise ValueError('--output must have a .json suffix')
        if args.selection_time_limit_s is not None and (not math.isfinite(args.selection_time_limit_s) or args.selection_time_limit_s <= 0):
            raise ValueError('--selection-time-limit-s must be finite and positive')
        scene = read_input(args.scene)
        robot = read_input(args.robot)
        camera = read_input(args.camera)
        result, regions, stations, candidates = plan(scene, robot, camera, yaw_step_deg=args.yaw_step_deg,
                                                       selection_time_limit_s=args.selection_time_limit_s,
                                                       ray_backend=args.ray_backend,
                                                       observation_workers=args.observation_workers,
                                                       observation_executor=args.observation_executor,
                                                       project_config=config)
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


if __name__ == '__main__':
    raise SystemExit(main())
