"""项目文件路径。更换地图、机器人或相机时，在这里修改对应路径。"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'

# 持久输入：三个文件集中放在 data/。
MAP_PATH = DATA_DIR / 'map.json'
ROBOT_PATH = DATA_DIR / 'robot.json'
CAMERA_PATH = DATA_DIR / 'camera.d435i.json'

# 可重新生成的结果与相机读取草稿。
PATH_PLAN_PATH = OUTPUT_DIR / 'path_plan.json'
PATH_SVG_PATH = OUTPUT_DIR / 'path_diagnostic.svg'
ANCHOR_PLAN_PATH = OUTPUT_DIR / 'anchor_plan.json'
CAMERA_EXPORT_PATH = OUTPUT_DIR / 'camera.d435i.json'

EDITOR_PAGE = ROOT / 'map_editor.html'
ANCHOR_VIEWER_TEMPLATE = ROOT / 'anchor_selector.html'


def read_input(path: Path) -> dict:
    """只读取指定输入；缺失时明确报错，不自动选择其他文件。"""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ValueError(f'输入文件不存在：{path}。请检查 config.py 中的路径或用命令行指定输入。') from exc
