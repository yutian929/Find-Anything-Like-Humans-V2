# 模型客户端边界

此目录对应旧 `anchor-map/model_clients` 的职责；当前只定义
`base.InstanceSegmenter` 协议，尚未迁移模型或实现远程服务。

后续每个模型可在独立模块中提供 Local / Remote 两个实现，用配置选择
backend、mode、device/weights 或 endpoint/timeout。模式非法应明确报错，
不要静默回退到本地或触发下载。离线锚点规划不加载这些依赖。

两种实现都输出原图坐标系下的无类别实例 mask、confidence、frame_id 和模型版本。
图像缩放、padding、mask 编解码及 HTTP 协议适配由客户端承担；3D 反投影、
多帧关联与 AABB 融合由 map_maintainers 承担。远程调用须验证请求和响应帧 ID、
mask 尺寸、值域、模型版本以及超时；失败不等同于没有物体。

旧客户端不能直接复制：SAM2 需要提示框，旧 SAM3 接口需要文本提示且只返回
bbox，均不等同于无类别自动实例发现；旧 SAM3 还依赖旧项目的全局 config/perception。
接入时应选择支持所需能力的实现并显式适配，保留旧 anchor-map 功能独立运行。
