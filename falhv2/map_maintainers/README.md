# 深度约束的删旧存新

当前版本面向锚点选择维护粗略的无类别物体分布，不做跨帧关联、融合或稳定身份追踪。
底盘提供栅格，Cubify 提供当前帧的三维框，所有几何统一到采集时刻的地图坐标系。

- `cubify.py`：变换全部八个角点后，计算地图系 AABB。
- `depth_replace.py`：检查旧 AABB 的深度可见性，替换、快照、原子存储和规划导出。
- `../tools/build_map.py`：采集、模型调用和诊断 SVG；仅由 main.py 注入配置。

## 更新规则（二维栅格相交）

1. 使用采集时刻相机位姿，把有效深度像素对应的射线段投到地图 XY 平面。
2. 射线从相机到实测表面停止，栅格化后取并集；不取凸包，不跨无效深度补洞。
3. 将旧地图 AABB 的 XY 范围投到同一栅格，只要共享至少一个栅格，就删除整个旧记录。
4. 本帧新 AABB 也必须与同一覆盖区域相交才加入，避免“能加入但永远删不掉”的累积。

三维 AABB 原样保留给锚点规划；仅维护判断降为二维。不做关联、融合或跨帧去重。
这会忽略高度：射线从物体上方或下方经过，也可能更新同一 XY 区域的框；这是本版本明确采用的近似。
被高墙完全挡住的射线会停止；如有射线越过墙顶，二维覆盖仍可能延伸到墙后。

MAP_UPDATE 只有 pixel_stride（默认每 4 像素采一条射线）、min_depth_m（0.15）、max_depth_m（8）。
射线在 XY 中按不大于半个栅格间距采样，保留采样产生的空隙，不使用假设扇形。
原 50% 可见比例、32×32 旧框投影采样和深度容差已移除。

更新掩膜每帧只计算一次，删除、加入、网页和 SVG 共用它，青色区域就是实际更新范围。
底盘占据栅格不会被修改。无有效采样射线时不删除、不新增；全无效帧和推理失败会报错，保留上一份地图。
成功的空预测仍会删除与当前覆盖区域相交的旧框。

存储策略升级为 schema_version=2 / xy_depth_replacement；拒绝按新策略直接恢复旧三维规则的累积地图。
切换本版本后重启 main.py，默认创建新会话；旧输出文件保留，不自动修改。

## 运行

在仓库根目录，使用已安装模型和机器人依赖的环境：

```bash
conda run -n depth python falhv2/main.py --task build-map \
  --map-id room-a --map-revision slam-session-1 \
  --frames 10 --output falhv2/outputs/robot_map.json

# 继续维护同一地图坐标系下的已有结果
conda run -n depth python falhv2/main.py --task build-map \
  --map-id room-a --map-revision slam-session-1 \
  --frames 10 --resume falhv2/outputs/robot_map.json

# 使用生成的地图运行现有锚点选择流程
python falhv2/main.py --task anchors --scene falhv2/outputs/robot_map.json
```

示例 room-a / slam-session-1 必须替换为实际会话标识。
ROBOT_SOURCE 字典可配置 map_id、map_revision、endpoint 和 chassis_url；
也可通过上面的 CLI 参数覆盖。CUBIFY 仍为一个字典，支持 local / remote。
只有 main.py 加载配置；可添加 `--config /path/to/robot.py` 选择另一份配置。

默认只处理一帧；--frames 为有限帧数，不自动移动底盘。模型加载一次，每次更新重新获取栅格和新鲜同步 RGB-D。
已有输出文件必须显式 --resume，避免无意覆盖。默认 main.py 现在启动持续更新与按指令重规划；--task anchors 执行离线选点。

## 输出与限制

一个 JSON 同时包含规划器的 grid + objects，以及 mapping 下的恢复状态、参数和最新一帧删除诊断。
每帧通过同目录临时文件和原子替换保存；帧处理失败时，之前成功保存的 JSON 保持不变。
同名 SVG 显示栅格、稀疏深度覆盖、删除框（橙）、保留框（紫）、新增框（绿）；SVG 是附属诊断，不和 JSON 作跨文件事务。
不保存所有历史帧，记录 ID 只代表单次观测；ObjectTrack 的 confirmed 在这里表示已纳入地图，不表示多帧确认。

当前底盘/发布器不提供可靠的坐标修正版本，map_revision 由调用者指定，无法自动发现未申报的回环修正。
地图切换、定位重置或坐标修正后，必须使用新的会话版本和输出文件；当前实现拒绝混用已声明的不同版本，不做历史重投影。
同坐标系下栅格范围/原点随探索变化可以正常更新。

已覆盖墙遮挡、一个栅格相交、不同高度、无效深度、坐标/时序错误、空预测、静止重复帧与预测抖动不累积、保存恢复、旧策略拒绝，以及两帧 CLI 更新。

### Simple ground rejection

`MAP_UPDATE["min_object_height_m"]` defaults to `0.10`. The map frame must have calibrated ground at Z=0 and Z pointing up. New cuboids whose highest map-frame corner is at or below this threshold are excluded. This tests height above ground, not box thickness or bottom height. Small low objects may also be rejected; thick floor predictions may remain. Original depth coverage and old-box XY deletion are unchanged. Update diagnostics include `ground_filtered_indices`; these also count as skipped predictions. The threshold is saved in map state. Restart live sessions after configuration changes.
