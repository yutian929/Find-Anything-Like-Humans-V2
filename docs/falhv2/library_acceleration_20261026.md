# 现成数值库与几何库优化记录（2026-10-26）

## 环境与验收口径

- 所有 Python 命令统一 `source /home/hri/miniconda3/etc/profile.d/conda.sh && conda activate test`。
- 起点：`df0a8df`；Python 3.10.19、NumPy 1.26.4、SciPy 1.15.3、OpenCV 4.10.0。
- 使用默认地图、相机、机器人与采样密度，完整导出六个产物。临时基准放在 `/tmp/falhv2-libraries-20261026/`。
- 分阶段记录完整进程耗时、候选生成耗时、支持关系/评分/热图、诊断、最终成本和高密度复核。
- 浮点几何库可能改变末位舍入；必须区别数值误差、合格关系变化与最终覆盖变化，不能只比较站位数量。

## 1. matmul

将 observation_model 和 observation_region 中四处对应批量矩阵乘法的 einsum 改为 NumPy matmul。
前期同环境完整链路单次对照：35.535 → 30.866 秒；候选生成 26.170 → 21.564 秒。
完整 supports（含 scores、heatmaps）、诊断、基线及最终 anchors 均相同。

## 2. NumPy 批量统计与数组接口

新增 ObservationBatch / evaluate_arrays：候选阶段保留列式数组，仅物体最佳诊断生成字典。
相机入盒判定、粗筛、加密判定、采样归约与评分使用 NumPy 广播、count_nonzero、divide 和 select；
遮挡 ID 使用布尔关系矩阵，诊断排名使用 lexsort 并保留原遍历顺序作为最后平局规则。
evaluate() 保留逐行字典兼容接口。

完整运行 28.573 秒，其中候选生成 19.491 秒。
完整 supports（含得分与热图）、诊断、基线及最终 anchors 与起点完全相同；4 站10张，高密度复核通过。
原 93 项测试通过，追加跨批次顺序与单姿态结果对照、空输入测试。

## 3. 双精度几何库

在 test 环境安装 Shapely 2.1.2（不升级其他依赖）。采用 multipoints → convex_hull → area，
仅对跨图像边界的凸包做 intersection(grid_size=0)，全部保留 Float64。
移除 Python 多边形裁剪与面积算子，保留 projection_areas 的标量兼容包装。

完整运行 27.434 秒，候选生成 18.288 秒（此时批次仍为 32）。
完整候选关系、得分、热图与起点完全相同；最终站位/朝向、合格判定不变。
诊断、基线及最终照片的几何数值不再逐位相同：面积最大绝对差 1.165e-10 px²，
入镜比例/quality 最大差 3.331e-16。没有通过四舍五入隐藏差异。

另取 3 种相机 roll/pitch/yaw 与随机站位，共 3,360 个物体/姿态，和冻结的 df0a8df 观察器对照：
合格、原因、可见率、评分、遮挡 ID 无差异；几何浮点差异在 rtol=1e-12、atol=1e-9 内。
超出画面的巨大投影面积绝对差可达 2.981e-8 px²，相对误差仍符合上述容差。
加入专门的 80% 入镜阈值用例（0.799999998 必须拒绝）、退化凸包、完全出框与裁剪解析值测试。

接口依据：[Shapely 批量运算](https://shapely.readthedocs.io/en/stable/)、
[双精度 intersection](https://shapely.readthedocs.io/en/2.1.2/reference/shapely.intersection.html)。

## 4. Open3D 射线求交

实现 Open3DRaycaster：每个 AABB 用库函数创建三角网格，缓存全场景和各目标的 RaycastingScene；
一次求目标轮廓，一次求场景最近表面。只对图像内命中目标的射线查询遮挡。
Open3D 0.19.0 已在 test 环境，不新增另一套环境；采用 CPU RaycastingScene。
Open3DRaycaster 工具类合并在 observation_model.py 中，保留场景缓存；Open3D 延迟导入，默认 NumPy 路径不加载它。
所有依赖统一维护在 falhv2/requirements.txt，包括 open3d==0.19.0；不再拆分独立的 Open3D 依赖文件。

初始批次 32 的 Open3D 候选耗时 21.291 秒，全流程（不含初始 Python/NumPy 导入）30.427 秒。
完整 133,660 条合格关系不变，但有 3 条评分变化：

| station / yaw / object | Float64 得分 | Open3D 得分 |
|---|---:|---:|
| 5895 / 10 / o20 | 0.5726599652587581 | 0.5726619128598486 |
| 7565 / 10 / o20 | 0.5229310421666916 | 0.5229918681247843 |
| 8196 / 4 / o7 | 0.5223752065345624 | 0.5224248241655419 |

最大差 6.083e-5，热图也有变化。随机 3,360 姿态对照没有触发这些边界，不能据此认定后端等价。
额外的贴边用例证明：距离目标轮廓外 1e-8 m 的射线会被 Float32 当作命中。
Open3D 返回最近三角面；原实现还包含按物体顺序更新的 1e-6 m 容差，近共面时遮挡 ID 不保证相同。
参考 [Open3D cast_rays 的 Float32 输入与最近命中语义](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)。

### 批次与线程对照

每项独立 Python 进程、一次完整候选生成，计入 Open3D 导入、建场和首次构建加速结构；不含 MILP 和导出。
threads=0 为 Open3D 自动线程，1 为显式单线程。

| 后端 | 批次 | threads | 候选生成（秒） | 与原完整 supports/评分/热图一致 |
|---|---:|---:|---:|---|
| NumPy | 32 | — | 18.288 | 是 |
| NumPy | 128 | — | 14.151 | 是 |
| NumPy | 512 | — | 14.116 | 是 |
| Open3D | 128 | 0 | 15.499 | 关系相同，3 条评分不同 |
| Open3D | 512 | 0 | 14.561 | 关系相同，3 条评分不同 |
| Open3D | 128 | 1 | 26.641 | 关系相同，3 条评分不同 |
| Float64 目标求交 + Open3D 场景遮挡 | 128 | 0 | 16.006 | 是 |

选择 **NumPy + 128 批次** 为默认：512 的单次收益很小，射线临时数组规模却扩大四倍。
混合后端修复了当前地图的评分差异，但没有更快，未纳入正式规划选项。
保留原 NumPy Float64 三轴广播求交作为参考，没有新增手写射线算子。
这组结果仅针对当前 28 个盒体和现有候选预筛，不能外推到大量遮挡体或长期驻留进程。

### 复现

```bash
conda activate test
# 正常完整运行：默认 NumPy，128 批次。
python -B falhv2/main.py
# 显式实验原生 Float32 候选；最终几何复核和高密度复核仍为 NumPy Float64。
python -B falhv2/main.py --ray-backend open3d --output falhv2/outputs/open3d/anchor_plan.json
```

`main.plan(..., ray_backend='open3d')` 也可选择实验后端。
输出 configuration 记录候选和验证的后端；最终验证始终独立使用 Float64。
这不会把 Float32 候选误判为与 Float64 候选池完全等价，最优性仍仅针对实际生成的有限候选关系。

## 最终验收

冻结 df0a8df 源码后，按旧1→新1→旧2→新2→旧3→新3 顺序串行运行完整 main.py，
每次均为新的 test 环境 Python 进程，包含启动、MILP、复核及六个文件导出。
所有运行输出独立目录，默认输入不变。

| 版本 | 完整运行三次（秒） | 完整中位数 | 候选生成中位数 | 选择/复核中位数 |
|---|---|---:|---:|---:|
| df0a8df | 35.373 / 35.389 / 35.171 | 35.373 | 26.020 | 7.950 |
| 最终 NumPy + GEOS + 128 批次 | 23.163 / 23.095 / 23.363 | **23.163** | **14.030** | 7.885 |

完整运行减少 **34.52%**，加速 **1.527 倍**；候选生成减少约 **46.08%**。
另跑正式 CLI 的 Open3D 128 批次版本：完整 **25.066 秒**，候选 **15.499 秒**，
仍为 4 站10张并通过 Float64 高密度复核，但慢于默认版且仍有上述 3 条评分差异。

验收结果：

- 三轮默认运行的全部 supports、scores、object_heatmaps 与 df0a8df 完全一致；133,660 条合格关系。
- 最终站位和朝向相同，4 站10张；28/28 覆盖，高密度复核通过，有限候选池字典序最优性结论保留。
- 诊断与照片内的几何面积有前述双精度末位差异，不宣称完整 JSON 逐字节相同。
- 100 项 Python 测试及 2 套 Node 编辑器/查看器测试通过。
- 新增数组跨批次/空输入、双精度阈值、原生求交及独立验证调度用例。
- 实验阶段验证过 NumPy、Open3D、混合三种基准模式；随后按目录整理要求删除独立 benchmarks 目录，保留测量结果与临时实验日志。
- 将第三轮最终默认版的六个产物更新到 falhv2/outputs；输入文件未修改。

逐次时间、输入 SHA256、环境版本、浮点差异与对照结果见
[机器可读验收记录](library_acceleration_20261026.json)。临时日志与冻结源码位于
`/tmp/falhv2-libraries-20261026/`；JSON 验收记录同时复制到 `falhv2/outputs/`。
