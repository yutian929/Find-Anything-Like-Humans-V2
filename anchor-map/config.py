"""Find-Anything-Like-Humans V2 —— anchor-map 全局配置（单进程状态机版）。

由三节点重构而来：数据源(FrameSource) + 状态机(Engine) + 锚点地图(AnchorStore)
+ 模型客户端(model_clients) + HTTP/UI(server) 全在一个进程内。

模型统一用 `mode` 字段决定走本地(local，从 models/ 加载权重)还是远程(remote，连服务器端口)，
model_clients 工厂只看 mode 建对应实现，切换只改这里一行。
"""
from __future__ import annotations

from pathlib import Path

# ===========================================================================
# 路径基准
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parent               # anchor-map/
PROJECT_ROOT = REPO_ROOT.parent                           # Find-Anything-Like-Humans-V2/
MODELS_DIR = PROJECT_ROOT / "models"                      # 本地模型权重集中存放
RUNTIME_DIR = REPO_ROOT / "runtime"                       # 运行产物（自动创建）
MAP_ROOT = str(RUNTIME_DIR / "server_map")                # anchors/ + results/
STATIC_DIR = str(REPO_ROOT / "static")                    # 前端页面

# ===========================================================================
# HTTP 服务（UI + /api/*）
# ===========================================================================
HOST, PORT = "0.0.0.0", 9200

# ===========================================================================
# 状态机 / 建图
# ===========================================================================
ENGINE_TICK_HZ = 2.0            # MAPPING 态每秒从 FrameSource 取帧并更新锚点地图的频率
XY_THRESH = 0.10                # 锚点位置匹配阈值（米）
YAW_THRESH_DEG = 10.0           # 锚点朝向匹配阈值（度）
DEFAULT_DEPTH_SCALE = 0.001     # 深度 uint16 -> 米 的缺省尺度

# ===========================================================================
# 数据源：已同步 RGB-D + pose（ZMQ）以及独立的 Slamtec 地图（REST）
# ===========================================================================
# 协议见 docs/README_rgbd_pose_protocol.md：
# [topic, msgpack metadata, RGB bytes, Depth bytes]。metadata 已包含同步后的
# T_map_base / T_map_camera，消费端不得再单独轮询 pose 后与图像拼接。
FRAME_ZMQ_ENDPOINT = "ipc:///tmp/rgbd_pose.ipc"
FRAME_ZMQ_TOPIC = "rgbd.pose"
FRAME_PROTOCOL_VERSION = 1

# Slamtec 底盘 REST 仅用于一次性加载占据栅格底图；位姿不从这里读取。
CHASSIS_URL = "http://192.168.11.1:1448"
CHASSIS_TIMEOUT_S = 2.0         # REST 请求超时（秒）
MAP_LOAD_RETRY_S = 10.0         # 启动时地图未就绪则低频重试；首次成功后不再请求

# ===========================================================================
# 兼容/离线坐标变换工具使用的静态外参（实时主链路使用帧内 T_map_camera）
# 光学系->底盘；米 + 四元数 wxyz。四元数已含光学系->机体系轴变换。
# ===========================================================================
CAM2BASE_XYZ_M = (-0.01322, 0.02376, 1.2)
CAM2BASE_QUAT_WXYZ = (0.396687081, -0.585354046, 0.585354046, -0.396687081)

# ===========================================================================
# 模型配置块（每个模型各自决定 local / remote）
# ===========================================================================
# local：从 path 加载权重（device: auto|cpu|cuda|cuda:0）。
# remote：连 url 端口，协议见 model_clients/base.py（JSON over HTTP）。
RYNNBRAIN = {
    "mode": "local",
    "path": str(MODELS_DIR / "RynnBrain-8B"),
    "device": "auto",
    "batch_size": 4,            # 召回阶段批量前向的批大小；<=0 表示一次跑完全部候选
}
SAM2 = {
    "mode": "local",
    "path": str(MODELS_DIR / "sam2.1-hiera-small"),
    "model_id": "facebook/sam2.1-hiera-small",
}
SAM3 = {
    "mode": "local",            # 跑不动端侧时改 {"mode": "remote", "url": "http://gpu-server:8003", "timeout": 30}
    "path": str(MODELS_DIR / "sam3"),
    "model_id": "facebook/sam3",
    "timeout": 30,
}

# 启动时是否预加载本地模型（RynnBrain/SAM2/本地 SAM3）。
# 默认 False：首次 query/detect 时才懒加载（秒起、不因缺权重而启动报错）。
# 权重就位后可设 True，启动即在引擎线程内预热，首次检索更快。
PRELOAD_MODELS = False
# 全离线：禁止 transformers/huggingface_hub 联网（内网 True）。
HF_OFFLINE = True

# ===========================================================================
# SAM3 仲裁参数（semantic_query 多候选时用 SAM3 作检测器给候选打分）
# ===========================================================================
SAM3_DEFAULT_THRESHOLD = 0.5          # 实例置信度阈值
SAM3_DEFAULT_MASK_THRESHOLD = 0.5     # mask 二值化阈值
SAM3_OVERLAP_THRESHOLD = 0.80         # 同图冗余实例去重阈值（交集/较小框面积）
QUERY_SAM3_OVERLAP_THRESH = 0.5       # SAM3 实例 bbox 与 RynnBrain bbox 的重合度阈值（判定确认同一物体）
QUERY_SAM3_SCORE_THRESH = 0.3         # 参与打分的 SAM3 实例最低置信度

# ===========================================================================
# 中文物体名 -> 英文 prompt 词典（原 objects_list.json 的 categories + new_queries）
# semantic_query 把中文物体名转成 SAM3 prompt 时查表；查不到且含中文时即时用 RynnBrain 翻译。
# ===========================================================================
OBJECT_PROMPTS = {
    "矿泉水瓶": "water bottle",
    "瓶装矿泉水": "bottled mineral water",
    "矿泉水": "mineral water",
    "餐盘": "dining plate",
    "不锈钢餐盘": "stainless steel plate",
    "不锈钢碗": "stainless steel bowl",
    "碗": "bowl",
    "药盒": "medicine bottle",
    "门把手": "door handle",
    "门": "door",
    "灯控开关": "lighting control",
    "写有门厅灯的灯控开关": "light switch with hall light",
    "花盆": "flowerpot",
    "浇水壶": "watering can",
    "衣柜": "wardrobe",
    "放有电视的桌子": "table with television",
    "放有花盆的桌子": "table with flowerpot",
    "桌子": "table",
    "椅子": "chair",
    "床": "bed",
    "床头柜": "bedside table",
    "冰箱": "refrigerator",
    "马桶": "toilet",
    "洗手池": "sink",
    "水池": "pool",
    "洗衣机": "washing machine",
    "电风扇": "electric fan",
    "插座": "socket",
    "座机电话": "landline phone",
    "电视": "television",
    "电视机": "tv",
    "电视机柜": "tv cabinet",
}

# ===========================================================================
# UI「预定义检索」清单
# ===========================================================================
PREDEFINED_QUERY_OBJECTS_FULL = [
    "瓶装矿泉水", "不锈钢餐盘", "不锈钢碗", "药盒", "门把手", "写有门厅灯的灯控开关",
    "花盆", "浇水壶", "衣柜", "椅子", "床", "桌子", "冰箱", "马桶", "电风扇",
    "插座", "座机电话", "电视", "洗手池", "门", "洗衣机", "电视机柜", "床头柜",
]
PREDEFINED_QUERY_OBJECTS_PARTIAL = [
    "瓶装矿泉水", "不锈钢餐盘", "不锈钢碗", "药盒", "花盆", "浇水壶", "座机电话", "椅子",
]
PREDEFINED_QUERY_OBJECTS = PREDEFINED_QUERY_OBJECTS_FULL
