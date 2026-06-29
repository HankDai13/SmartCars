# ROS2 Architecture

## Node Graph

```text
camera_node
  publishes sensor_msgs/Image
  -> /camera/image_raw

lane_lfnet_node
  subscribes /camera/image_raw
  publishes smartcar_interfaces/Lane
  -> /perception/lane

sign_yolo_node
  subscribes /camera/image_raw
  publishes smartcar_interfaces/DetectionArray
  -> /perception/detections

decision_node
  subscribes /perception/lane, /perception/detections
  publishes smartcar_interfaces/ActionCommand
  -> /decision/action_command

chassis_node
  subscribes /decision/action_command
  publishes smartcar_interfaces/ChassisStatus
  -> /chassis/status

agent_node optional
  subscribes compact state
  publishes /agent/advice
```

## Topic Contract

| Topic | Type | Producer | Consumer |
| --- | --- | --- | --- |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | camera | perception |
| `/perception/lane` | `smartcar_interfaces/msg/Lane` | lane perception | decision |
| `/perception/detections` | `smartcar_interfaces/msg/DetectionArray` | YOLO perception | decision |
| `/decision/action_command` | `smartcar_interfaces/msg/ActionCommand` | decision/agent | chassis |
| `/chassis/status` | `smartcar_interfaces/msg/ChassisStatus` | chassis | monitor/agent |

## Decision Policy

优先级从高到低：

1. 急停：检测到 `person` 或 `obstacle`，且置信度超过阈值。
2. 离散动作：`left`、`right`、`turnaround`、`park` 等路标，带冷却时间。
3. 巡线：lane 有效时按 steering angle 输出 `follow_lane`。
4. 丢线：根据配置选择慢速搜索或停车。

## Agent Boundary

Agent 只允许输出高层动作：

- `stop`
- `resume`
- `turn_left`
- `turn_right`
- `u_turn`
- `park`
- `follow_lane`
- `set_strategy`

禁止让 Agent 直接输出四轮转速、串口字节或底层 PID 参数。底层安全策略始终由 `decision_node` / `chassis_node` 执行。

