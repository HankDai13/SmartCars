# Project Baseline

## 课程目标映射

本项目按 `1.15` 系数作为主线：

- ROS2 底层控制框架。
- topic 通信。
- 视觉感知、决策、底盘执行节点分层解耦。
- 推理、决策、交互、执行、反馈形成闭环。

`1.30` 作为增强：

- LLM Agent 作为高层决策中枢。
- Agent 输出受限于安全动作集合。
- 确定性节点保留实时控制和安全兜底。

## 原厂代码复用边界

原厂小车源码仍然是硬件适配基础，主要复用：

- 摄像头采集和共享图像经验。
- LFNet / YOLOv5 推理封装。
- `base_action.py`、`complex_actions.py` 动作库。
- `controller.py` 串口通信和 ESP32 指令下发。

本仓库新增的是 ROS2 工程层：

- 原 `lane_following.py` 的巡线推理拆成 `lane_lfnet_node`。
- 原 `helper.py` 的路标检测拆成 `sign_yolo_node`。
- 原 scene 中的混合决策逻辑收敛到 `decision_node`。
- 原 controller 和 action 库由 `chassis_node` 调用。

## 验收证据

建议每个阶段保留证据：

- `ros2 topic list`
- `ros2 node list`
- `rqt_graph`
- `ros2 topic echo /perception/lane`
- `ros2 topic echo /decision/action_command`
- `ros2 bag record -a`
- 现场视频和日志

