#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 pkg list | grep smartcar

