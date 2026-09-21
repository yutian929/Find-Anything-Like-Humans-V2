# FALHV2 anchor-map 重构文档

> 来源仓库：`/home/hri/yutian/ElderlyCareRobot-TASK2/anchor-map-v2`
> 目标仓库：`/home/hri/yutian/Find-Anything-Like-Humans-V2`（本仓库）
>
> 本文档三部分：**第一部分**是来源仓库现状分析；**第二部分**是重构说明（需求原话）；
> **第三部分**是据此制定的重构计划与目标代码结构（供评审，确认后落地）。

---
---

# 第一部分 · anchor-map-v2 代码结构梳理（现状分析）

## 一、这个项目是干什么的

**一个自包含的「三节点在线语义地图」系统**：把机器人的 RGBD + 位姿数据当作实时数据流，做坐标换算 → 维护「锚点语义地图」→ 提供**语义检索 / 即时检测** → 在网页上用 three.js 做**实时 3D 语义地图可视化**。

核心能力有三块：

1. **建图**：机器人在若干位置建立「锚点」（anchor），每个锚点存一帧 RGB + 深度 + 相机位姿。
2. **检索 / 检测**：用大模型在锚点地图里找物体，返回物体的地图系 3D 坐标。
   - `/query`：在已建锚点里检索单个物体（RynnBrain 召回 + SAM3 仲裁 + SAM2 精分割）。
   - `/detect`：从机器人**当前实时帧**检测物体。
   - `/multi_query/all`：全类别多实例检索（SAM3 一次把某类别在所有锚点里的全部实例找出来）。
3. **可视化**：3D 网页地图展示占据栅格底图 + 锚点 + 命中物体 + 机器人实时位姿。

## 二、三节点架构与数据流

```
节点1 publisher (:9100)  --HTTP拉取-->  节点2 relay (:9200)  --HTTP /update-->  节点3 core (:9300)
   发布当前帧                            坐标换算+定时上报+UI                     锚点地图+模型检索
                                          ↑ UI 的 query/detect 转发到 core ↓
```

数据流：`节点1 --HTTP拉取--> 节点2 --HTTP /update--> 节点3`；UI 的 query/detect 经节点2 转发到节点3。

| 节点 | 文件 | 端口 | 职责 |
|---|---|---|---|
| **节点1** publisher | `node1_publisher_stream.py`（ZMQ 真机流）/ `node1_publisher_offline.py`（录制序列回放） | 9100 | 发布「当前帧位姿 + RGB/Depth + 占据栅格」。两版对外 HTTP 接口完全一致 |
| **节点2** relay | `node2_relay.py` | 9200 | 订阅节点1 → 底盘→相机 4×4 坐标换算 → 定时上报节点3；缓存地图；提供控制台 UI + 3D 地图页 + query/detect 代理 + 录制控制 |
| **节点3** core | `node3_core.py` | 9300 | 维护锚点语义地图，跑 **RynnBrain-8B + SAM2 + SAM3** 三个模型做检索/检测 |

## 三、每个文件的作用与代码结构

### 核心算法与配置

- **`config.py`** (251 行)：全局配置，按节点分区。端口、ZMQ 参数、相机外参、锚点匹配阈值（`XY_THRESH` / `YAW_THRESH_DEG`）、SAM3 参数、预定义检索清单（`PREDEFINED_QUERY_OBJECTS_FULL/PARTIAL`）等。模型权重来自外部 `../task2_3/model_weights`（只读）。
- **`geometry.py`** (132 行)：纯坐标变换。四元数→旋转矩阵、`cam2base`（相机→底盘外参）、`base2map`（底盘→地图）、`camera_pose_in_map`（合成相机在地图系的 4×4 位姿）、世界坐标→像素。
- **`core_models.py`** (1068 行)：**节点3 的算法库**，整个项目的算法心脏。包含：
  - 图像/深度解码、bbox 处理、内参/位姿归一化（`decode_rgb_part` / `decode_depth_part` / `normalize_intrinsics` / `normalize_camera_pose` 等）
  - 点云去噪、mask→3D 中心（`denoise_pointcloud` / `mask_to_3d_center` / `mask_to_3d_points_and_center` / `compute_bbox_3d_corners`）
  - **三个模型封装类**：
    - `RynnBrainDetector`（存在性检测 / 出 bbox / 中译英翻译）
    - `SAM2Segmenter`（精分割）
    - `SAM3MultiInstanceSegmenter`（多实例检测 `segment_batch` / `segment_categories` / `detect_prompt_batch`）
  - `AnchorStore`（锚点存储：`create_anchor` / `find_match` / `refresh_anchor` / `update` / `clear` / 落盘索引）
  - 多实例去重/3D 归并工具（`filter_nested_instances` / `bbox_overlap_smaller` / `merge_instances_3d`）

### 三个节点主程序

- **`node3_core.py`** (1165 行)：FastAPI 服务。
  - 路由：`/healthz` `/update` `/query` `/detect` `/multi_query/all` `/anchors` `/reset_anchors` `/reset_history` `/reset` `/results/*` `/results_multi/*` `/detect/pointcloud`。
  - 模型加载：`_ensure_models` / `_ensure_rynnbrain`（懒加载）、`_load_sam3_at_startup`（启动即加载 SAM3）。
  - `/query` 检索流水线：`_collect_rynnbrain_candidates`（RynnBrain 召回，遍历全部候选不早退）→ `_score_candidates_by_sam3`（SAM3 打分）→ `_arbitrate_candidates`（仲裁）→ `_run_search_pass` → SAM2 精分割 → 3D 中心 → 落盘。
  - 多实例：`_run_all_multi_query` / `_materialize_multi_instances` / `_finalize_category_result`。
- **`node2_relay.py`** (629 行)：FastAPI。
  - 后台线程：`_subscriber_loop`（按 `RELAY_SUBSCRIBE_HZ` 拉节点1 当前帧）、`_uploader_loop`（按 `UPDATE_HZ` 坐标换算后上报节点3 `/update`）、`_refresh_map_cache`（仅 map_version 变化才拉底图）。
  - `/api/*` 代理接口：`set_anchor` `query` `detect` `reset_*` `predefined_objects` `record/*` `status` `map` `rgb` `anchors` `objects` `multi_query/all` `multi_objects` `results*`。
  - `_rewrite_result_urls`：把 core 的结果图 URL 重写为同源。
  - 首页 UI：`/`（index_multi.html）、`/map_multi`（map_multi.html）。
- **`node1_publisher_stream.py`** (482 行)：接真机 ZMQ 流。
  - `StreamState`（帧状态：`ingest` / `snapshot_cursor` / `get_frame_image` / `get_map` / `health`）
  - `FrameRecorder`（录制器：`start` / `write` / `stop` / `toggle` / `status`，把原始 msgpack 零重编码落盘）
  - `_zmq_thread`（ZMQ SUB + CONFLATE 只留最新帧）
  - HTTP 接口：`/healthz` `/stream/cursor` `/frame/{name}/rgb|depth` `/map/*` `/control` `/record/*`。
- **`node1_publisher_offline.py`** (247 行)：回放录制序列。复用 stream 版的 `StreamState`，HTTP 接口完全一致。`_resolve_record_dir`（序列路径优先级：CLI > env/config > 最新 rec_*）、`_load_manifest`、`_replay_thread`（按帧 stamp 差 sleep 回放）。

### 前端（`static_relay/`）

- **`index_multi.html`** (702 行)：控制台首页（控制面板 + 状态栏 + 内嵌 `<iframe src="/map_multi">`）。含建锚/清空/录制/Query/预定义检索/即时检测/全类别检索等按钮，`postMap(...)` 向地图页发 postMessage。
- **`map_multi.html`** (651 行)：three.js 3D 语义地图页，独立向 `/api/*` 拉数据，可 iframe 嵌入。占据栅格底图 + 锚点 + 命中物体 + 多实例紫色标记 + 即时检测点 + 机器人实时位姿，周期自动刷新。

### 配套 / 运维

- **`objects_list.json`**：SAM3 多实例检索的 `categories`（中文类别→英文 prompt + 阈值）配置，以及 `new_queries`（`/query` 未知中文名的英文翻译缓存，自动维护）。
- **`scripts/robot_stream_publisher.py`** (491 行)：**机器人端**脚本（跑在机器人上，需 ROS2 环境），采集彩色/深度/位姿/内参[/地图]，时间对齐 + 压缩后经 ZMQ PUB 发出。
- **`scripts/tmp_latency_probe.py`** (176 行) / **`scripts/tmp_zmq_latency.py`** (134 行)：延迟探测临时脚本。
- **`restart_anchor_map_tmux.sh`** + **`anchor-map-v2.service`**：tmux 一键拉起 core/pub/relay 三个会话 + systemd 开机自启。
- **`nginx-dashboard.conf`**：nginx 反代配置。
- **`docs/startup-guide.html`**、**`docs/query-sam3-arbitration-plan.md`**：启动指南 + SAM3 仲裁方案文档。
- **`seq_loader.py`** (173 行)：**遗留代码**，README 明确标注「节点1 已不使用」（老的 task2_6/seq_* 离线序列读取器）。
- **`environment.yml`** / **`requirements.txt`** / **`requirements-lock.txt`**：依赖清单。

### 运行产物（`runtime/`，自动创建）

- `server_map/`：`anchors/`（锚点）+ `results/`（单物体检索结果）+ `results_multi/`（多实例结果）
- `cache/map/`：占据栅格底图缓存
- `recordings/`：录制序列 `rec_<时间戳>/`（供离线回放）

## 四、关键技术约定

- **坐标换算**（relay 侧）：`T_cam2map = T_base2map(x,y,yaw) @ T_cam2base`。优先用随帧下发的动态胸部相机外参 `chest_cam_tf`（光学系→底盘），缺失时回退 `config.py` 的静态外参 `CAM2BASE_XYZ_M` / `CAM2BASE_QUAT_WXYZ`。外参四元数已含「光学系→机体系」轴变换。
- **深度精度**：全程 PNG16 无损（`imencode('.png', uint16)` ↔ `IMREAD_UNCHANGED`），不走 JPEG/base64。ZMQ 流/录制可半分辨率传输，节点1 按内参上采样回彩色分辨率。
- **数据源可插拔**：真机 ZMQ 流 / 离线回放两版对外 HTTP 接口完全一致，relay/core/UI 零改动。
- **地图缓存**：relay 仅在 `map_version` 变化时才拉取并缓存底图。
- **锚点匹配**：core 用相机 4×4 提取 `(x,y,yaw)` 做阈值匹配（`XY_THRESH` / `YAW_THRESH_DEG`）。
- **模型加载**：`update` 纯落盘不需模型；`query`/`detect` 需模型；SAM3 为硬依赖，node3 启动即加载（需 `transformers>=5.0`，约 3.4GB 权重）；RynnBrain/SAM2 可懒加载。

## 五、文件规模一览

| 文件 | 行数 | 类型 |
|---|---|---|
| `node3_core.py` | 1165 | 节点3 主程序 |
| `core_models.py` | 1068 | 算法库 |
| `index_multi.html` | 702 | 前端控制台 |
| `map_multi.html` | 651 | 前端 3D 地图 |
| `node2_relay.py` | 629 | 节点2 主程序 |
| `robot_stream_publisher.py` | 491 | 机器人端脚本 |
| `node1_publisher_stream.py` | 482 | 节点1（真机流） |
| `config.py` | 251 | 配置 |
| `node1_publisher_offline.py` | 247 | 节点1（回放） |
| `tmp_latency_probe.py` | 176 | 临时脚本 |
| `seq_loader.py` | 173 | 遗留（未使用） |
| `tmp_zmq_latency.py` | 134 | 临时脚本 |
| `geometry.py` | 132 | 坐标变换 |

---
---

# 第二部分 · 重构说明（需求原话）

1. 现在我希望 FALHV2 的第一版代码，也就是在 `/home/hri/yutian/Find-Anything-Like-Humans-V2` 路径下，新建一个 `anchor-map` 文件夹，在这个里面放运行的代码，然后 `/home/hri/yutian/Find-Anything-Like-Humans-V2` 下面还有一个 `models` 文件夹，在这个里面存放所有的要用到的模型。

2. 在 `anchor-map` 文件夹下面，希望对来源仓库 `/home/hri/yutian/ElderlyCareRobot-TASK2/anchor-map-v2` 中的代码做重构，具体重构如下：
   - **2.1** 将 3 个节点的功能实现整合成一个状态机，现在数据的获取变成直接读取 + HTTP 了，不需要单独写一个节点来实现，可以是写成一个类，用于获得对齐帧的数据，这个类可以后续我来完成。然后状态机的流程大概就是，没事的时候一直都是在跑匹配刷新的机制，也就是更新 anchor-map，然后在有请求的时候，就去 anchor-map 里面查找，然后返回结果。这里的请求只会有两种，就是 semantic_query 和 instant_detect，前者要遍历所有的 anchor，找到合适的，后者就是只要拿到当前帧，做一个检测返回就行。
   - **2.2** 可视化，建立锚点，删除锚点（优化成可具体到某一个），也即是之前有个 UI 的功能，需要保留，你看怎么写比较合适。
   - **2.3** config.py 要保留，里面定义模型路径这些，还有一些预定义的物品清单。其中模型路径有可能是某个端口，因为可能有些模型跑不了端侧的，需要部署在服务器上，这个你要按类别写成类做初始化，然后给主要的状态机里面使用。
   - **2.4** 代码要简洁，然后各模型，比如 RynnBrain 可以单独写一个类，一个 python 脚本。

3. 先写清楚要改动的地方，列出一个改造计划，重点给我看改造后的代码结构。

---
---

# 第三部分 · 重构计划与目标代码结构

## 一、总体思路：三节点 → 单进程状态机

原系统是三个独立进程 + HTTP 串联：

```
node1 publisher(:9100) --HTTP--> node2 relay(:9200) --HTTP--> node3 core(:9300)
   拉取当前帧                     坐标换算+上报+UI               锚点地图+模型
```

重构后合并为**一个进程**，进程内是一个**状态机 Engine**：

```
                         ┌─────────────── FrameSource（对齐帧数据源，用户后续补全）
                         │  直接读取 + HTTP，产出 Frame{rgb,depth,pose2d,intrinsics,cam_tf,map}
                         ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                        Engine（状态机，单线程主循环）           │
   │                                                                │
   │   MAPPING（默认/空闲）──有请求──> QUERYING / DETECTING ──完成──┐ │
   │       ↑ 持续跑「匹配刷新」，更新 AnchorStore                   │ │
   │       └───────────────────────────────────────────────────────┘ │
   └───────────────▲────────────────────────────────┬───────────────┘
                   │ 请求(入队) + 结果(Future)        │ 读写
        ┌──────────┴──────────┐            ┌──────────▼──────────┐
        │   HTTP Server(UI)   │            │     AnchorStore     │
        │  控制台 + 3D 地图页  │            │  建/匹配/刷新/删/清   │
        └─────────────────────┘            └──────────┬──────────┘
                                                       │ 调用
                                          ┌────────────▼────────────┐
                                          │  model_clients（本地/远程）│
                                          │  RynnBrain / SAM2 / SAM3  │
                                          └──────────────────────────┘
```

**状态机行为**（对应重构说明 2.1）：

- **MAPPING（默认态）**：主循环每 tick 从 `FrameSource` 取最新对齐帧 → `geometry` 算相机在地图系 4×4 位姿 → `AnchorStore.update()`（按位姿阈值匹配已有锚点则刷新其快照；若「建锚脉冲」置位则新建锚点）。这就是「没事时一直跑匹配刷新」。
- **收到请求** → 主循环从请求队列取出，切到对应态处理，处理完把结果回填 `Future`，再切回 MAPPING。因为主循环单线程，模型与 AnchorStore 无并发，无需锁。
- **两种检索请求**：
  - `semantic_query(object, xy_range?)`：**遍历所有锚点**做 RynnBrain 召回 + SAM3 仲裁 + SAM2 精分割 → 地图系 3D 中心。
  - `instant_detect(object)`：只取**当前帧**做一次检测，返回 bbox + 地图系 3D 坐标。
- **控制类请求**（也走队列，保证串行安全）：建锚脉冲、删除单个锚点、清空锚点、清空历史。

## 二、目标目录结构

```
Find-Anything-Like-Humans-V2/
├── anchor-map/                      # ← 运行代码（单进程）
│   ├── config.py                    # 全局配置：模型路径/端口、阈值、预定义清单、外参
│   ├── main.py                      # 入口：初始化 Engine + 启动 HTTP Server
│   │
│   ├── engine.py                    # ★ 状态机核心（主循环 + 请求队列 + 状态流转）
│   ├── anchor_store.py              # 锚点地图：建/匹配/刷新/删单个/清空/落盘（原 AnchorStore）
│   ├── frame_source.py              # ★ 对齐帧数据源：Frame 数据类 + FrameSource 接口/占位实现
│   ├── geometry.py                  # 坐标变换（原样迁移）
│   ├── perception.py                # 解码/点云去噪/mask→3D（原 core_models.py 的工具函数部分）
│   │
│   ├── model_clients/               # 模型封装：本地加载 or 远程端口（按 config 二选一）
│   │   ├── __init__.py              # build_rynnbrain()/build_sam2()/build_sam3() 工厂
│   │   ├── base.py                  # 抽象接口 + 远程 HTTP 调用基类
│   │   ├── rynnbrain.py             # RynnBrain：本地(torch) + 远程(HTTP) 两实现
│   │   ├── sam2.py                  # SAM2：同上
│   │   └── sam3.py                  # SAM3：同上
│   │
│   ├── server.py                    # FastAPI：静态页 + /api/*（把请求转成 Engine 队列请求）
│   ├── static/
│   │   ├── index.html               # 控制台（原 index_multi.html 精简，单源、去 relay 代理）
│   │   └── map.html                 # three.js 3D 地图（原 map_multi.html，改为同源直连 /api/*）
│   │
│   └── requirements.txt             # 精简依赖（去 zmq/msgpack 等数据源相关）
│
├── models/                          # ← 本地模型权重集中存放（凡 mode=local 的模型在此放权重）
│   ├── RynnBrain-8B/
│   ├── sam2.1-hiera-small/
│   └── sam3/                        # 若 SAM3 保留且 mode=local 才需要；remote 则此处留空
│
└── anchor-map-refactor-plan.md      # 本文档
```

## 三、模块职责与关键接口（改造后）

### 1. `frame_source.py`（新增，★用户后续补全采集逻辑）
统一「对齐帧」数据契约，替代原 node1 全部职责（ZMQ/回放/录制都不要了）。

```python
@dataclass
class Frame:
    rgb: np.ndarray          # BGR, HxWx3
    depth: np.ndarray        # uint16, 与 rgb 同分辨率（PNG16 无损语义）
    pose2d: dict             # {x, y, yaw}  底盘在地图系
    intrinsics: dict         # {fx, fy, cx, cy, depth_scale}
    cam_tf: dict | None      # 动态相机外参(光学系→底盘)，缺失则回退 config 静态外参
    stamp: float
    occupancy_map: OccMap | None   # 占据栅格（可选，随帧或单独）

class FrameSource:
    def latest(self) -> Frame | None: ...     # 供 MAPPING/instant_detect 取当前帧
    def occupancy_map(self) -> OccMap | None:  # 供 UI 底图
    def healthz(self) -> dict: ...
```
> 骨架 + 占位（如从固定路径读/HTTP 拉），真正采集由你补全。Engine 只依赖 `latest()`。

### 2. `engine.py`（新增，★核心）
```python
class State(Enum): BOOTING, MAPPING, QUERYING, DETECTING, CONTROL

class Engine:
    def __init__(self, source, store, models, cfg): ...
    def run(self):                 # 后台线程主循环（见下）
    # 供 server 调用：入队 + 等 Future
    def semantic_query(self, obj, xy_range=None) -> dict
    def instant_detect(self, obj) -> dict
    def build_anchor(self) -> dict          # 置建锚脉冲
    def delete_anchor(self, anchor_id) -> dict
    def clear_anchors(self) / clear_history(self)
    def status(self) -> dict                # 当前状态/锚点数/上次update/当前帧信息
```
主循环骨架：
```python
while self.running:
    req = self.queue.get_nowait_or_none()
    if req:
        self.state = req.state          # QUERYING/DETECTING/CONTROL
        req.future.set_result(self._handle(req))
    else:
        self.state = State.MAPPING
        f = self.source.latest()
        if f: self.store.update(f, is_new=self._take_anchor_pulse())
    sleep(tick)
```

### 3. `anchor_store.py`（迁移 + 增强）
原 `AnchorStore` 迁移；`update()` 入参从散列改为 `Frame`；**新增单个删除**（重构说明 2.2）：
```python
def update(self, frame: Frame, is_new: bool) -> dict     # 建/匹配刷新
def find_match(self, camera_pose) -> anchor | None
def delete(self, anchor_id: str) -> dict                 # ★ 新增：删除指定锚点
def clear(self) -> dict                                  # 清空全部
def anchors(self) -> list
```

### 4. `model_clients/`（重构说明 2.3、2.4）
每个模型**独立脚本 + 独立类**，且支持**本地加载 / 远程端口**两种模式，由 config 决定：
```python
# base.py
class RynnBrainBase(ABC):
    def exists(self, image, obj) -> bool
    def bbox(self, image, obj) -> list | None
    def translate(self, zh) -> str
    # + 批量接口
# rynnbrain.py:  RynnBrainLocal(torch 加载 models/RynnBrain-8B) / RynnBrainRemote(HTTP)
# sam2.py:       SAM2Local / SAM2Remote     -> segment(image, bbox)->mask
# sam3.py:       SAM3Local / SAM3Remote     -> detect(images, prompt)->instances（仅作 semantic_query 仲裁检测器）
# __init__.py:   按 config 的 mode 建对应实现
```
> 全类别多实例检索已删除，SAM3 不再有 `segment_categories`；只保留 `detect`（对候选图按 prompt 检测、给 RynnBrain 候选打分）。

### 5. `perception.py`（从 `core_models.py` 拆出）
纯算法工具：`decode_rgb/depth`、`normalize_intrinsics/pose`、`denoise_pointcloud`、`mask_to_3d_center`、`bbox` 工具，以及 SAM3 仲裁用的同图去重 `filter_nested_instances` 与重合度 `bbox_overlap_smaller`。**不含**模型类（挪到 `model_clients/`）、**不含** AnchorStore（挪到 `anchor_store.py`）。原 `merge_instances_3d`（跨锚点 3D 归并）随全类别检索一并删除。

### 6. `server.py` + `static/`（重构说明 2.2）
单进程单源，去掉原 relay 的 `/api/*` 转发与 URL 重写。接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 控制台 `index.html` |
| GET | `/map` | 3D 地图页 `map.html` |
| GET | `/api/status` | Engine 状态 / 锚点数 / 当前帧 / 上次 update |
| GET | `/api/anchors` `/api/map` `/api/rgb` | 锚点列表 / 底图 / 当前帧图 |
| POST | `/api/anchor/build` | 建锚脉冲 |
| POST | `/api/anchor/{id}/delete` | ★ 删除单个锚点（新功能） |
| POST | `/api/anchors/clear` `/api/history/clear` | 清空锚点 / 清空历史 |
| POST | `/api/query` | semantic_query |
| POST | `/api/detect` | instant_detect |
| GET | `/api/results/{object}/{asset}` | 结果图 |

`static/` 由原 2 个 html 精简迁移：数据源从 relay `:9200/api/*` 改为同源 `/api/*`；地图页保留占据栅格底图 + 锚点 + 命中物体 + 即时检测点 + 机器人实时位姿，周期自动刷新；控制台保留建锚/删锚(可选单个)/清空/Query/预定义检索/即时检测。去掉录制卡片、全类别检索卡片。

### 8. 远程模型 HTTP 协议 v1（本文档约定）

本地类与远程类实现**同一套** `base.py` 抽象方法；远程类把调用序列化为下面的端点。约定：

- 传输：`HTTP POST`，请求/响应体均为 **JSON（UTF-8）**；`Content-Type: application/json`。
- 图像：`image_b64` = base64(JPEG bytes)；mask 回传 `mask_b64` = base64(PNG 单通道 0/255，无损)。
- bbox 统一 `[x1, y1, x2, y2]`（像素整数，左上/右下）。
- 每个服务提供 `GET /healthz` → `{"ok": true, "model": "<id>", "device": "..."}`。
- 出错：返回 HTTP 4xx/5xx + `{"error": "..."}`；客户端按 config 的 `timeout` 超时。
- 批量优先：能批处理的接口一次收多图（本地类内部也走同一 batch 签名，单图 = 长度 1 的批）。

**RynnBrain**
```
POST /exists_batch  {"images_b64":[...], "object":"矿泉水瓶"}          -> {"exists":[true,false,...]}
POST /bbox_batch    {"images_b64":[...], "object":"矿泉水瓶"}          -> {"bboxes":[[x1,y1,x2,y2]|null, ...]}
POST /translate     {"text":"矿泉水瓶"}                                 -> {"english":"water bottle"}
```
**SAM2**
```
POST /segment       {"image_b64":..., "bbox":[x1,y1,x2,y2]}            -> {"mask_b64":"<PNG>"}
```
**SAM3**（仅仲裁用）
```
POST /detect        {"images_b64":[...], "prompt":"water bottle",
                     "threshold":0.5, "mask_threshold":0.5}
                    -> {"results":[{"instances":[{"bbox":[x1,y1,x2,y2],
                                                   "score":0.87,
                                                   "mask_b64":"<PNG>"}]}, ...]}
```

### 7. `config.py`（保留 + 调整，重构说明 2.3）
- 保留：阈值（`XY_THRESH`/`YAW_THRESH_DEG`）、SAM3 仲裁参数（`SAM3_DEFAULT_THRESHOLD`/`SAM3_DEFAULT_MASK_THRESHOLD`/`QUERY_SAM3_OVERLAP_THRESH`/`QUERY_SAM3_SCORE_THRESH`/`SAM3_BATCH_SIZE`）、预定义清单 `PREDEFINED_QUERY_OBJECTS_*`、静态相机外参、`MAP_ROOT` 等落盘路径。
- **新增 `OBJECT_PROMPTS`**（原 `objects_list.json` 的 `categories`）：中文名 → 英文 prompt 词典，供 semantic_query 把中文物体名转成 SAM3 prompt 查表。因体量小且纯静态，直接进 config.py：
```python
OBJECT_PROMPTS = {"矿泉水瓶": "water bottle", "餐盘": "dining plate", ...}
```
- 模型路径改为**按类别的配置块**：每个模型都用统一的 `mode` 字段自行决定走本地(`local`，从 `models/` 加载权重)还是远程(`remote`，连服务器端口)。`model_clients/__init__.py` 工厂只看 `mode` 建对应实现，切换只改这一行、代码结构不变：
```python
MODELS_DIR = REPO_ROOT.parent / "models"
RYNNBRAIN = {"mode": "local",  "path": MODELS_DIR/"RynnBrain-8B", "device": "auto", "batch_size": 4}
SAM2      = {"mode": "local",  "path": MODELS_DIR/"sam2.1-hiera-small"}
SAM3      = {"mode": "local",  "path": MODELS_DIR/"sam3", "device": "auto"}   # 若保留 SAM3
# 任一模型都可改远程，例如： RYNNBRAIN = {"mode": "remote", "url": "http://gpu-server:8001", "timeout": 30}
```
- 去掉：节点1 ZMQ/回放/录制、relay 上游 URL（`PUBLISHER_URL`/`CORE_URL`）、`UPDATE_HZ`（改为 Engine tick）等跨进程配置；全类别相关 `MULTI_INSTANCE_*` / `MULTI_INSTANCE_OBJECTS_JSON`。
- **不再有 `objects_list.json` 与 `new_queries` 运行时翻译缓存**：semantic_query 遇到 `OBJECT_PROMPTS` 里没有的中文名时，即时用 RynnBrain 翻译一次（无落盘状态，代码最简）。

## 四、删减清单（明确不迁移）

| 原文件/功能 | 处理 |
|---|---|
| `node1_publisher_stream.py` / `node1_publisher_offline.py` | **删**，职责并入 `frame_source.py` |
| `scripts/robot_stream_publisher.py`、ZMQ/msgpack 流、录制+回放 | **删**（数据源改直接读+HTTP） |
| `node2_relay.py` 的进程分离、`_subscriber_loop`/`_uploader_loop`、`/api/*` 代理 + URL 重写 | **删/内联**进 Engine + server |
| `seq_loader.py`（遗留未用）、`scripts/tmp_*.py`（延迟探测） | **删** |
| **全类别多实例检索** `multi_query/all`、`results_multi/`、SAM3 `segment_categories`、`merge_instances_3d` | **删**（只保留 semantic_query + instant_detect 两接口） |
| `objects_list.json`（categories + new_queries） | **删**：静态 categories 收进 config 的 `OBJECT_PROMPTS`；new_queries 缓存不要，改即时翻译 |
| `restart_anchor_map_tmux.sh`、`anchor-map-v2.service`、`nginx-dashboard.conf` | **暂不迁移**（单进程，后续按需加简单启动脚本） |
| `docs/`、`environment.yml`/`requirements-lock.txt` | 按需保留（`requirements.txt` 精简重写） |

## 五、保留并重构的映射

| 原 | 新 |
|---|---|
| `config.py` | `config.py`（模型路径改配置块、去跨进程项） |
| `geometry.py` | `geometry.py`（原样） |
| `core_models.py` 工具函数 | `perception.py` |
| `core_models.py` 的 `AnchorStore` | `anchor_store.py`（+ 单删） |
| `core_models.py` 的 3 个模型类 | `model_clients/{rynnbrain,sam2,sam3}.py`（+ 远程模式；SAM3 只留 detect） |
| `node3_core.py` 的 query/detect 流水线 | `engine.py` 内 `_handle_query`（遍历锚点 + SAM3 仲裁）/ `_handle_detect`（当前帧） |
| `node2_relay.py` 的 UI 与 `/api/*` | `server.py` + `static/` |
| `index_multi.html` / `map_multi.html` | `static/index.html` / `static/map.html` |
| `objects_list.json` 的 categories | 收进 `config.py` 的 `OBJECT_PROMPTS`（new_queries 缓存删除） |

## 六、落地顺序（确认结构后执行）

1. 建目录 `anchor-map/`、`models/`。
2. 迁移 `config.py`（改模型配置块 + 新增 `OBJECT_PROMPTS`、去跨进程与全类别项）、`geometry.py`。
3. 拆 `core_models.py` → `perception.py` + `anchor_store.py`(+单删) + `model_clients/*`(+远程；SAM3 只留 detect)。
4. 写 `frame_source.py`（Frame + 接口 + 占位实现）。
5. 写 `engine.py`（状态机主循环 + semantic_query/instant_detect + 控制类）。
6. 写 `server.py` + 精简 `static/`（同源、去录制/全类别卡片）。
7. 写 `main.py`、`requirements.txt`、`README.md`，最小跑通（用占位 FrameSource）。

## 七、已确认决策

1. **命名**：目录/文件命名（`model_clients/`、`perception.py` 等）保持不变。 ✅
2. **全类别多实例检索**：删除，只做 `semantic_query` 与 `instant_detect` 两个接口；SAM3 仅作 semantic_query 的仲裁检测器保留。 ✅
3. **远程模型 HTTP 协议**：采用本文档「三·8 远程模型 HTTP 协议 v1」。 ✅
