# SmartCars

小学期智能小车项目 base repo。当前目标是先稳定完成课程手册里的 `1.15` 系数要求，再把 `1.30` 大语言模型 Agent 作为可插拔增强层。

## 目标

`1.15` 的验收重点：

- 使用 ROS2 作为底层控制框架。
- 基于 topic 消息通信。
- 分层解耦视觉感知、决策、底盘执行节点。
- 完成推理、决策、交互、执行、反馈闭环。

`1.30` 的增强方向：

- 在 ROS2 框架基础上加入 LLM Agent 决策中枢。
- Agent 只做高层任务规划和策略裁决，不直接输出电机速度。
- 急停、巡线、避障、底盘执行继续走确定性规则/PID，保证实时性。

## 仓库结构

```text
configs/                 全局配置样例：模型、topic、底盘参数
data/                    本地数据集目录，不提交大文件
deploy/                  Atlas200I DKA2、CANN、ROS2、模型部署说明
docs/                    课程原始资料和项目设计文档
external/                第三方源码 sparse checkout 目录
ml/
  yolov5/                YOLOv5 地标/行人/障碍检测 pipeline
  lfnet/                 LFNet 巡线角度回归 pipeline
  lanenet_hnet/          LaneNet + H-Net 车道分割/拟合 pipeline
models/                  模型权重、ONNX、OM 的本地归档目录
reports/                 过程材料、验收记录、汇报素材
ros2_ws/src/             ROS2 工作空间源码
scripts/                 环境安装、上游源码拉取、构建辅助脚本
tools/                   数据处理和模型导出辅助工具
```

## 快速开始

在 Windows 开发机上准备仓库：

```powershell
.\scripts\fetch_ascend_car.ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
```

在 Atlas / Ubuntu 22.04 上准备 ROS2：

```bash
bash scripts/ros2_setup_atlas.sh
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch smartcar_bringup smartcar_1_15.launch.py
```

## 当前工程路线

1. 先接通原厂小车源码：摄像头、ESP32 串口、动作库、基础 demo。
2. 训练并部署三个视觉 pipeline：YOLOv5、LFNet、LaneNet+H-Net。
3. 把原 demo 的单体/多进程结构改为 ROS2 topic 节点图。
4. 完成现场任务闭环：巡线、路标反应、避障/行人、掉头、泊车、状态反馈。
5. 挂接 Agent 节点作为 `1.30` 增强，但保留 `1.15` 确定性决策节点作为稳定兜底。

