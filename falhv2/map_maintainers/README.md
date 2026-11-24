# 深度约束的删旧存新

当前版本面向锚点选择维护粗略的无类别物体分布，不做跨帧关联、融合或稳定身份追踪。
底盘提供栅格，Cubify 提供当前帧的三维框，所有几何统一到采集时刻的地图坐标系。

- `cubify.py`：变换全部八个角点后，计算地图系 AABB。
- `depth_replace.py`：检查旧 AABB 的深度可见性，替换、快照、原子存储和规划导出。

## 更新规则（XY 面积覆盖比例，默认 0.9）

1. 使用采集时刻相机位姿，把有效深度像素对应的射线段投到地图 XY 平面。
2. 射线从相机到实测表面停止，栅格化后取并集；不取凸包，不跨无效深度补洞。
3. 对每个旧 AABB，计算 `r = 面积(AABB 的 XY 投影 ∩ 青色深度覆盖区) / 完整 AABB 的 XY 投影面积`。
   只有 `r >= min_coverage_ratio` 才删除；默认阈值 **0.9**，不足时保留整个旧框。
4. 新 AABB 使用同一个比例门槛；覆盖不足暂不入图。
   新框还需满足原有的离地高度筛选。

这不是 IoU：分母是完整物体投影面积，不是并集面积。边界栅格按实际相交面积加权；
AABB 超出地图范围的部分仍计入分母，不能因裁剪而把少量重叠误算成完全覆盖。
比例比较包含 1e-12 的浮点舍入容差，因此几何上恰好 90% 也能通过。
此版本不叠加相机图像边缘过滤，不引入新旧实例匹配、融合或稳定 ID。

参数 `min_coverage_ratio` 取值 `(0,1]`，默认 `0.9`；另外保留 pixel_stride=4、
min_depth_m=0.15、max_depth_m=8、min_object_height_m=0.10。
射线在 XY 中按不大于半个栅格间距采样；深度无效和稀疏采样造成的空隙保留，
不为了达到阈值填充青色区域。青色区域本身与底盘占据栅格均不因本次改动而变化。

覆盖比例是二维深度覆盖比例，不等于物体在 RGB 图像内出现的比例；高度仍被忽略。
完全入镜的厚物体也可能因深度止于正面、覆盖不到其整个占地范围而达不到 90%。
阈值变保守可能减少新物体入图，并保留过时旧框。若大旧框未删、局部新框却通过门槛，
仍可能有重叠框；本次最小改动不解决实例去重。需用同一录制检查保留和更新效果。
无有效采样射线时不删除、不新增；推理失败保留上一份地图；成功空预测仍删除达到 90% 覆盖的旧框。

`mapping.last_update.decisions` 记录每个旧框的 `coverage_ratio`、`deleted` 和 `reason`；
`detection_coverage_ratios` 记录本帧每个新框的比例，`coverage_filtered_indices` 记录
因不足阈值而拒绝的新检测，`min_coverage_ratio` 记录本次阈值。
`skipped_detection_indices` 包含所有未入图检测，含高度过滤与覆盖过滤，两个原因可能重叠。

新恢复状态为 schema_version=3 / xy_depth_coverage_ratio，保存实际阈值。
旧相交即删除策略的 schema_version=2 地图不能通过 `DepthReplacementMap.load` 恢复更新状态。
旧输出保留，可用于结果对比和离线查看。

## 使用

应用可通过 Python API 调用 `DepthReplacementMap` 的 `set_grid`、`integrate` 和 `save`。
锚点规划直接读取外部建图系统导出的一份完整场景：

```bash
python falhv2/main.py --task anchors --scene /path/to/falhv2_scene.json
```

## 输出与限制

一个 JSON 同时包含规划器的 grid + objects，以及 mapping 下的恢复状态、参数和最新一帧删除诊断。
每帧通过同目录临时文件和原子替换保存；帧处理失败时，之前成功保存的 JSON 保持不变。
不保存所有历史帧，记录 ID 只代表单次观测；ObjectTrack 的 confirmed 在这里表示已纳入地图，不表示多帧确认。

当前底盘/发布器不提供可靠的坐标修正版本，map_revision 由调用者指定，无法自动发现未申报的回环修正。
地图切换、定位重置或坐标修正后，必须使用新的会话版本和输出文件；当前实现拒绝混用已声明的不同版本，不做历史重投影。
同坐标系下栅格范围/原点随探索变化可以正常更新。

已覆盖墙遮挡、一个栅格相交、不同高度、无效深度、坐标/时序错误、空预测、静止重复帧与预测抖动不累积、保存恢复和旧策略拒绝。

### Simple ground rejection

`min_object_height_m` defaults to `0.10`. The map frame must have calibrated ground at Z=0 and Z pointing up. New cuboids whose highest map-frame corner is at or below this threshold are excluded. This tests height above ground, not box thickness or bottom height. Small low objects may also be rejected; thick floor predictions may remain. Original depth coverage is unchanged; old-box deletion uses the coverage-ratio threshold above. Update diagnostics include `ground_filtered_indices`; these also count as skipped predictions. The threshold is saved in map state.

## 0.9 覆盖比例验收（2026-11-03）

160 项 Python 测试、实时页面 Node 交互测试通过。新增测试覆盖边界格面积、地图外分母、
80% 保留/拒绝新增、恰好 90% 更新、参数校验及新策略保存恢复。
真实恢复包在 2 倍速下完成回放：接收 1332 帧与 49 张地图，Cubify 处理 86 帧，
最终 73 个框、状态 complete；最后一帧删除诊断与 0.9 门槛逐项核对通过。
结果位于 `outputs/replay_validation_1793699937308515895/`。框数量不代表几何质量或去重结果；
相机扫过后的保留效果仍需在页面中观察。测试进程已退出。
