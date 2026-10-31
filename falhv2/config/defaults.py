"""项目路径与实验默认参数。物理标定、地图和机器人参数仍读取 data/ JSON。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'

# 持久输入：三个文件集中放在 data/。
MAP_PATH = DATA_DIR / 'map_aabb.json'
ROBOT_PATH = DATA_DIR / 'robot.json'
CAMERA_PATH = DATA_DIR / 'camera.d435i.json'

# 可重新生成的结果与相机读取草稿。
SAFE_REGIONS_PATH = OUTPUT_DIR / 'safe_regions.json'
SAFE_REGIONS_SVG_PATH = OUTPUT_DIR / 'safe_regions.svg'
ANCHOR_PLAN_PATH = OUTPUT_DIR / 'anchor_plan.json'
CAMERA_EXPORT_PATH = OUTPUT_DIR / 'camera.d435i.json'

EDITOR_PAGE = ROOT / 'assets' / 'map_editor.html'
ANCHOR_VIEWER_TEMPLATE = ROOT / 'assets' / 'anchor_selector.html'

# 观察实验默认值。相机 JSON 的 observation_quality 同名字段优先于这里。
# 得分 = 距离权重 × 画面内未遮挡比例；入镜、像素面积是另外的硬门槛。
OBSERVATION_DEFAULTS = {
    'min_in_frame_fraction': 0.8,   # 最低入镜比例，(0, 1]
    'min_observation_score': 0.5,   # 最低观察得分，(0, 1]，包含等于门槛
    'min_visible_pixels': 400.0,    # 最低可见投影面积，px²，必须 > 0
    'radial_inner_weight': 1.0,     # 观察区域内边界权重
    'radial_outer_weight': 0.5,     # 外边界权重；0 < 外端 <= 内端 <= 1，线性下降
    'sample_grid': 12,             # 常规遮挡采样边长，12 表示 12×12
    'refinement_grid': 36,         # 临界加密边长；整数 [4,128]，不小于常规采样
}

# 规划实验默认值。方向间隔、求解预算可由同名 CLI 参数覆盖。
YAW_STEP_DEG = 30.0                # 每站方向间隔，(0,360] 度
OBSERVATION_WORKERS = 4           # 逐物体候选计算的 worker 数；1 恢复串行
OBSERVATION_EXECUTOR = 'processes' # 'threads' 或 'processes'；仅影响候选阶段
SELECTION_TIME_LIMIT_S = None      # 全部 MILP 调用共享的秒数预算；None 表示不限时
AUDIT_SAMPLE_GRID = 96            # 最终复核边长，整数 [4,128]；实际不低于 refinement_grid
MAX_AUDIT_RETRIES = 3             # 高密度复核失败后的重选次数；0 表示直接尝试基线兜底
SAME_COST_REPAIR_TIME_LIMIT_S = 2.0 # 已有最优性证明时，先限时寻找同成本修复；0 关闭


# Cubify Anything: one model, one configuration block.
CUBIFY = {
    "mode": "local",             # local | remote
    "path": str(Path.home() / "models_weights/cubifyanything/cutr_rgbd.pth"),
    "source_path": str(Path.home() / "map_test_new_v8_clean"),  # SDK root; None if installed
    "device": "auto",
    "max_side": 768,              # RGB longest side, 256..1024
    "score_threshold": 0.25,
    "url": "http://127.0.0.1:8093",  # Remote model address, including port
    "timeout": 60,                # Remote HTTP timeout in seconds
}


# Robot session identity must be supplied explicitly; the current API has no revision ID.
ROBOT_SOURCE = {
    "map_id": None,
    "map_revision": None,
    "endpoint": "ipc:///tmp/rgbd_pose.ipc",
    "chassis_url": "http://192.168.11.1:1448",
}

# Replace old boxes visible in the current posed depth image; no tracking/fusion.
MAP_UPDATE = {
    "visible_fraction": 0.5,     # fraction of the full projected old box
    "sample_grid": 32,          # samples per projected bounding-box axis
    "depth_tolerance_m": 0.08,
    "min_depth_m": 0.15,
    "max_depth_m": 8.0,
}
