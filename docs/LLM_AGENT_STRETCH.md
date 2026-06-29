# LLM Agent Stretch for 1.30

Agent 节点的目标是做高层决策，不接管实时控制。

## 输入

- 当前任务阶段。
- 最近一次路标检测结果。
- lane 有效性、转向角、置信度。
- 底盘状态、最近动作、是否急停。
- 可选：地图编号、验收任务顺序、人工指令。

## 输出

只允许输出安全动作集合：

- `follow_lane`
- `turn_left`
- `turn_right`
- `u_turn`
- `park`
- `stop`
- `resume`

## 安全约束

- Agent 输出必须经过 `decision_node` 二次校验。
- 急停优先级高于 Agent。
- Agent 不修改底盘串口协议。
- Agent 不直接输出四轮转速。
- Agent 调用失败时自动回落到确定性状态机。

