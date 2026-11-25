"""Offline anchor selection and verification for one map snapshot."""
import logging
import math
from time import perf_counter

from falhv2.anchor_planners import anchor_verification as verification
from falhv2.anchor_planners.regions_extractor import plan as find_safe_regions
from falhv2.anchor_planners.observation_model import PointCamera, BoxObserver, validate_objects
from falhv2.anchor_planners.anchor_candidates import stations_from_regions, yaw_grid, generate_candidates
from falhv2.anchor_planners.anchor_selector import select_anchors

LOGGER = logging.getLogger(__name__)


def configured_camera(camera_config, config):
    return {**camera_config, 'observation_quality': {
        **config.OBSERVATION_DEFAULTS, **camera_config.get('observation_quality', {})}}


def planning_audit(config):
    return verification.audit_parameters(
        sample_grid=config.AUDIT_SAMPLE_GRID, max_retries=config.MAX_AUDIT_RETRIES,
        same_cost_time_limit_s=config.SAME_COST_REPAIR_TIME_LIMIT_S)


def plan(scene, robot_config, camera_config, *, yaw_step_deg=None, selection_time_limit_s=None,
         ray_backend='numpy', observation_workers=None, observation_executor=None, project_config):
    """Select object-driven anchors directly from safe regions; no route is built.

    Targets are world-frame 3D centers; base positions are (x, y, 0).
    A station may take multiple yaw-directed photos. AABB coverage uses clipped
    framing, sampled occlusion/pixels and radial score; point maps use legacy FOV/range.
    """
    cfg = project_config
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
