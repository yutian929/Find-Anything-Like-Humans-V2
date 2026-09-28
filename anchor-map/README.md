# anchor-map（Find-Anything-Like-Humans V2）

由 `ElderlyCareRobot-TASK2/anchor-map-v2` 的**三节点架构**重构而来的**单进程状态机**版：
数据源(FrameSource) + 状态机(Engine) + 锚点地图(AnchorStore) + 模型客户端(model_clients)
+ HTTP/UI(server) 全在一个进程内。

## 目录

```
anchor-map/
├── config.py            全局配置（模型 local/remote、阈值、中英词典、预定义清单）
├── main.py              入口：装配 + 启动
├── engine.py            ★ 状态机（MAPPING↔QUERYING/DETECTING/CONTROL）
├── anchor_store.py      锚点地图：建/匹配/刷新/删单个/清空/落盘
├── frame_source.py      ★ rgbd.pose 解码/最新同步帧缓存 + 地图一次性加载
├── geometry.py          坐标变换
├── perception.py        解码/点云/mask→3D/结果图 等纯算法工具
├── model_clients/       RynnBrain / SAM2 / SAM3 —— 各含 local + remote 两实现
│   ├── base.py  __init__.py(工厂)  rynnbrain.py  sam2.py  sam3.py
├── server.py            FastAPI：控制台 + 3D 地图页 + /api/*
├── static/index.html    控制台
├── static/map.html      three.js 3D 地图
└── requirements.txt

../models/                本地模型权重（凡 mode=local 的模型放这）
├── RynnBrain-2B/  sam2.1-hiera-small/  sam3/
```

## 状态机

- **MAPPING（默认）**：每 tick 从 `FrameSource.latest()` 取同步帧 → 直接使用帧内 `T_map_camera` → `AnchorStore.update`
  （匹配则刷新，建锚脉冲置位则新建）。
- **收到请求**：从队列取出，切到对应态处理，回填结果再切回 MAPPING（单线程，无需锁）。
- 请求类型：`semantic_query`（遍历所有锚点）、`instant_detect`（当前帧），及建锚/删锚/清空控制。

## 运行

```bash
cd anchor-map
pip install -r requirements-local.txt  # 当前 config.py 的三个模型都使用本地模式
python main.py
```

仅使用远程模型时可改装 `requirements.txt`。本地 SAM3 需要
`transformers==5.8.0`；已安装旧版 `transformers` 的环境需升级后重启
`python main.py`。`open3d` 可选，缺失时点云处理会使用 numpy 兜底。

本地模型权重放在项目根目录 `models/`。运行
`../download_models.sh` 可下载配置所需的 RynnBrain-2B、SAM2.1 和 SAM3；
也可用 `../download_models.sh rynnbrain|sam2|sam3` 单独下载或续传。
默认下载 2B（约 4.9 GB）；如需 8B，可运行 `RYNNBRAIN_SIZE=8B ../download_models.sh rynnbrain`，
并将 `config.py` 的 `RYNNBRAIN["path"]` 指向 `models/RynnBrain-8B`。
RynnBrain 和 SAM2.1 使用 Hugging Face 的 `hf` 命令下载；SAM3 使用 ModelScope
的 `modelscope` 命令下载。脚本会验证配置文件和 safetensors 权重是否到位。

浏览器打开 **http://127.0.0.1:9200/**。

`frame_source.py` 按 `docs/README_rgbd_pose_protocol.md` 订阅
`ipc:///tmp/rgbd_pose.ipc` / `rgbd.pose`。发布端在一条消息中提供 RGB、Depth、
`T_map_base` 和 `T_map_camera`，本进程直接消费同步后的 `T_map_camera`，不再单独轮询底盘 pose。
使用当前发布端的协议 v2；深度图须对齐到彩色光学系。

占据栅格由独立后台任务从 Slamtec `/maps/explore` 获取。启动时失败会按
`MAP_LOAD_RETRY_S` 低频重试，首次成功后缓存到内存并停止请求；UI 的 map 接口只读该缓存。

## 模型 local / remote

每个模型在 `config.py` 用 `mode` 决定：
```python
RYNNBRAIN = {"mode": "local",  "path": ".../RynnBrain-2B", "device": "auto", "batch_size": 4}
SAM3      = {"mode": "remote", "url": "http://gpu-server:8003", "timeout": 30}
```
远程 HTTP 协议见 `model_clients/base.py`（JSON over POST；图 base64-JPEG、mask base64-PNG、bbox `[x1,y1,x2,y2]`）。

## 主要接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` `/map` | 控制台 / 3D 地图页 |
| GET | `/api/status` `/api/anchors` `/api/objects` | 引擎状态 / 锚点 / 检索历史 |
| GET | `/api/rgb` `/api/map/meta` `/api/map/image` | 当前帧 / 底图元信息 / 底图 |
| POST | `/api/query` | semantic_query（form: object, x_min/x_max/y_min/y_max 可选）|
| POST | `/api/detect` | instant_detect（form: object）|
| POST | `/api/anchor/build` | 建锚脉冲 |
| POST | `/api/anchor/{id}/delete` | 删除单个锚点 |
| POST | `/api/anchors/clear` `/api/history/clear` | 清空锚点 / 清空历史 |
| GET | `/api/results/{object}/{asset}` | 检索结果图 |
