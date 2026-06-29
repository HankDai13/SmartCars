#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

if ! grep -q "Ubuntu 22.04" /etc/os-release; then
  echo "This script targets Ubuntu 22.04 / ROS2 Humble. Continue manually if your Atlas image differs."
fi

$SUDO apt update
$SUDO apt install -y software-properties-common curl gnupg lsb-release
$SUDO add-apt-repository universe -y

$SUDO curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
  | $SUDO tee /etc/apt/sources.list.d/ros2.list > /dev/null

$SUDO apt update
$SUDO apt install -y \
  ros-humble-ros-base \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-vision-opencv \
  python3-colcon-common-extensions \
  python3-pip \
  python3-opencv \
  v4l-utils

python3 -m pip install --user -r requirements-runtime.txt

echo "source /opt/ros/humble/setup.bash" >> "$HOME/.bashrc"
echo "ROS2 Humble base environment installed. Open a new shell or source /opt/ros/humble/setup.bash."

