#!/bin/bash
# Wrapper for sim_texture_probe.py — sources ROS2 + PX4 gazebo env so
# gzserver finds its plugins (libgazebo_ros_state.so, libgazebo_ros_camera.so)
# and OGRE finds its shader libs. Runs the probe under the bench venv's
# python3 (torch 2.11 + cv2 4.11 + numpy 1.26).
#
# NOTE: no `set -u` — ROS2's setup.bash references unset vars on first boot.

source /opt/ros/humble/setup.bash 2>/dev/null || true

export PX4_SRC=/home/teo/Drone/PX4-Autopilot
export PX4_BUILD=${PX4_SRC}/build/px4_sitl_default
source "$PX4_SRC/Tools/simulation/gazebo-classic/setup_gazebo.bash" \
  "$PX4_SRC" "$PX4_BUILD" >/dev/null 2>&1 || true

export GAZEBO_RESOURCE_PATH="/home/teo/Drone/gazebo_resources:/home/teo/Drone/gazebo_resources/models:/usr/share/gazebo-11${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"
export GAZEBO_MODEL_PATH="/home/teo/Drone/gazebo_resources/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"

pkill -9 -f "gzserver|gzclient|px4|MicroXRCEAgent" 2>/dev/null || true
sleep 1

cd /home/teo/Drone/bench
exec /home/teo/Drone/bench/.venv/bin/python3 sim_texture_probe.py "$@"
