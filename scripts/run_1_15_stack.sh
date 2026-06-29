#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch smartcar_bringup smartcar_1_15.launch.py

