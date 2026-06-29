# Deployment Index

主要部署文档：

- `docs/DEPLOYMENT_ATLAS_ROS2.md`: Atlas200I DKA2、ROS2、CANN、OM、启动和验证。
- `docs/TRAINING_PIPELINES.md`: YOLOv5、LFNet、LaneNet + H-Net 的训练/导出/ATC 转换。
- `docs/ARCHITECTURE_ROS2.md`: 节点图、topic 合约、决策边界。

建议部署顺序：

1. 在开发机准备数据和训练模型。
2. 导出 ONNX。
3. 在 Atlas 上用 ATC 转 OM，或在与目标 CANN 版本一致的训练环境中转好后复制到 Atlas。
4. 先用 `enable_hardware:=false` 验证 ROS2 topic 闭环。
5. 接入 ESP32 串口和原厂小车源码，打开 `enable_hardware:=true`。
6. 现场前录制 rosbag 和视频，固化模型版本。

