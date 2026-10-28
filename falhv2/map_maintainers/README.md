# 真实地图构建与维护（接口阶段）

当前只有 MapBuilder 协议，尚未实现 SLAM 接入、实例融合或地图写入。
共享输入输出结构位于 core/types.py。详细设计见
[机器人建图架构](../../docs/falhv2/robot_mapping_architecture.md)。

后续按实际功能逐步增加：
- pipeline.py：组织采集、推理、融合、发布；不接管底盘控制。
- projection.py：mask + 对齐深度反投影、滤除无效深度/离群点、变换到 map。
- association.py：多视角实例关联、稳定 ID、拆分/合并和冲突处理。
- object_map.py：证据融合、无类别 AABB、生命周期和可观测性驱动的失效。
- store.py：录制索引、版本快照、原子发布及回滚。
- export.py：显式导出与现有规划器兼容的 grid + objects JSON。

栅格地图来自机器人 SLAM；3D 物体分布来自 RGB-D 多视角融合。两者必须使用
同一个坐标修正版本。SLAM 回环后重新投影证据，不能只改栅格、不更新物体。
从来没看见过的深度区域不能当作自由空间；某帧未检出也不能直接删除旧物体。
