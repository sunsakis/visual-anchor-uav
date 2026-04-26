#!/bin/bash
# Wrapper for bench/anchor_hover_bench.py: sources ROS2 humble + PX4 gazebo env
# + our ros2_ws overlay so rclpy, px4_msgs, gazebo_ros_state, and gzserver
# plugins all resolve. Harness uses the SYSTEM python3 (rclpy comes from
# /opt/ros/humble; anchor_estimator itself inserts its bench venv at import
# time, so the subprocess still sees torch/cv2/numpy from bench/.venv).
#
# NOTE: no `set -u` — setup.bash references unset vars on first boot.

source /opt/ros/humble/setup.bash 2>/dev/null || true
source /home/teo/ros2_ws/install/setup.bash 2>/dev/null || true

export PX4_SRC=/home/teo/Drone/PX4-Autopilot
export PX4_BUILD=${PX4_SRC}/build/px4_sitl_default
source "$PX4_SRC/Tools/simulation/gazebo-classic/setup_gazebo.bash" \
  "$PX4_SRC" "$PX4_BUILD" >/dev/null 2>&1 || true

export GAZEBO_RESOURCE_PATH="/home/teo/Drone/gazebo_resources:/home/teo/Drone/gazebo_resources/models:/usr/share/gazebo-11${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"
export GAZEBO_MODEL_PATH="/home/teo/Drone/gazebo_resources/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"

pkill -9 -f "gzserver|gzclient|MicroXRCEAgent|anchor_estimator" 2>/dev/null || true
sleep 1

exec python3 /home/teo/Drone/bench/anchor_hover_bench.py "$@"
