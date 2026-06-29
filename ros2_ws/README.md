# ROS2 Workspace

Build on Atlas / Ubuntu:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch smartcar_bringup smartcar_1_15.launch.py
```

Packages:

- `smartcar_interfaces`: custom messages.
- `smartcar_camera`: camera publisher.
- `smartcar_perception`: LFNet lane and YOLO detection nodes.
- `smartcar_decision`: deterministic decision state machine.
- `smartcar_chassis`: adapter from action command to original car controller.
- `smartcar_agent`: optional 1.30 high-level agent.
- `smartcar_bringup`: launch and runtime config.

