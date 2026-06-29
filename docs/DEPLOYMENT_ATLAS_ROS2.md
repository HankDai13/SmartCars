# Atlas200I DKA2 Deployment

## 1. 拉取原厂小车源码

```bash
bash scripts/fetch_ascend_car.sh
```

确认路径：

```text
external/ascend-devkit/src/E2E-Sample/Car
```

## 2. 安装 ROS2 Humble

在 Atlas 的 Ubuntu 22.04 环境执行：

```bash
bash scripts/ros2_setup_atlas.sh
```

如果课程镜像不是 Ubuntu 22.04，应按课程 ROS2 教程手动安装对应发行版。

## 3. 安装 CANN 并确认芯片型号

加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
```

记录实际 `soc_version`。教程中常见为 `Ascend310B4`，如果设备不同，以 `npu-smi info` 为准。

## 4. 构建 ROS2 工作空间

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 5. 启动 1.15 基础栈

先用 mock 底盘验证 topic 闭环：

```bash
ros2 launch smartcar_bringup smartcar_1_15.launch.py enable_hardware:=false
```

接入真车：

```bash
ros2 launch smartcar_bringup smartcar_1_15.launch.py \
  enable_hardware:=true \
  car_python_path:=external/ascend-devkit/src/E2E-Sample/Car/python
```

## 6. 验证

```bash
ros2 node list
ros2 topic list
ros2 topic echo /perception/lane
ros2 topic echo /decision/action_command
ros2 topic echo /chassis/status
```

建议现场前录包：

```bash
ros2 bag record -a -o reports/rosbag_site_dry_run
```

## 7. 常见问题

- 摄像头打不开：用 `v4l2-ctl --list-devices` 检查设备号，修改 `camera_index`。
- 没有模型：感知节点会发布无效 lane 或空 detection，用于先验证 ROS2 通信。
- 串口失败：确认 ESP32 USB 设备名，常见为 `/dev/ttyUSB0` 或 `/dev/ttyACM0`。
- OM 推理失败：确认 CANN 环境、`soc_version`、模型输入 shape 一致。

