# 深度约束的删旧存新

当前版本面向锚点选择维护粗略的无类别物体分布，不做跨帧关联、融合或稳定身份追踪。
底盘提供栅格，Cubify 提供当前帧的三维框，所有几何统一到采集时刻的地图坐标系。

- `cubify.py`：变换全部八个角点后，计算地图系 AABB。
- `depth_replace.py`：检查旧 AABB 的深度可见性，替换、快照、原子存储和规划导出。
- `../tools/build_map.py`：采集、模型调用和诊断 SVG；仅由 main.py 注入配置。

## 更新规则

1. 将旧框八角点投影到当前 RGB 图像，采样完整投影包围矩形。
2. 将像素射线变换到地图系，与原地图 AABB 求交；未击中旧框的样本不进入分母。
3. 有效画面内深度满足 `旧框入射深度 <= 实测深度 + 容差`，计为可见。
   射线参数是光轴 Z；画面外、无效/超范围深度、被更近表面遮挡均不提供可见证据。
4. 可见样本 / 完整投影击中样本达到门槛时，删除整个旧框。其余旧框保留。
5. 加入全部本帧预测框，分配新的观测 ID，不匹配旧物体。

默认门槛 0.5、32×32 采样、0.08m 深度容差、有效深度 0.15–8m；
均由 main.py 加载的 MAP_UPDATE 字典控制。这是待现场调整的初始参数。
成功推理的空结果仍会清除可见旧框；推理失败、全无效深度、错帧、旧时间戳和坐标版本冲突不会清除旧框。
跨越相机近裁剪面的旧框保守保留。小框/薄遮挡的采样误差、深度边缘误差和定位误差尚需现场验证。

二维诊断通过采样深度射线段栅格化生成，不取凸包、不填补无效深度；允许存在采样空隙。
该图会丢失高度信息，只用于显示，不参与删除判断，不修改底盘占据栅格。

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
python falhv2/main.py --scene falhv2/outputs/robot_map.json
```

示例 room-a / slam-session-1 必须替换为实际会话标识。
ROBOT_SOURCE 字典可配置 map_id、map_revision、endpoint 和 chassis_url；
也可通过上面的 CLI 参数覆盖。CUBIFY 仍为一个字典，支持 local / remote。
只有 main.py 加载配置；可添加 `--config /path/to/robot.py` 选择另一份配置。

默认只处理一帧；--frames 为有限帧数，不自动移动底盘。模型加载一次，每次更新重新获取栅格和新鲜同步 RGB-D。
已有输出文件必须显式 --resume，避免无意覆盖。默认 main.py 仍只执行原有离线锚点选择。

## 输出与限制

一个 JSON 同时包含规划器的 grid + objects，以及 mapping 下的恢复状态、参数和最新一帧删除诊断。
每帧通过同目录临时文件和原子替换保存；帧处理失败时，之前成功保存的 JSON 保持不变。
同名 SVG 显示栅格、稀疏深度覆盖、删除框（橙）、保留框（紫）、新增框（绿）；SVG 是附属诊断，不和 JSON 作跨文件事务。
不保存所有历史帧，记录 ID 只代表单次观测；ObjectTrack 的 confirmed 在这里表示已纳入地图，不表示多帧确认。

当前底盘/发布器不提供可靠的坐标修正版本，map_revision 由调用者指定，无法自动发现未申报的回环修正。
地图切换、定位重置或坐标修正后，必须使用新的会话版本和输出文件；当前实现拒绝混用已声明的不同版本，不做历史重投影。
同坐标系下栅格范围/原点随探索变化可以正常更新。

已覆盖墙遮挡、部分入镜、部分遮挡、深度空洞、光轴深度、不同高度、坐标变换、
空预测、重复/乱序帧、版本冲突、保存恢复及两帧 CLI 更新。测试使用合成几何与模拟采集/推理；
不代表已经完成实机多帧维护验证。
