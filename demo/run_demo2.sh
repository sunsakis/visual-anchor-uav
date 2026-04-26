#!/bin/bash
# Runner for demo2_motion_delta_aerial.py — mirrors bench/run_sim_probe.sh.
source /opt/ros/humble/setup.bash 2>/dev/null || true
export PX4_SRC=/home/teo/Drone/PX4-Autopilot
export PX4_BUILD=${PX4_SRC}/build/px4_sitl_default
source "$PX4_SRC/Tools/simulation/gazebo-classic/setup_gazebo.bash" \
  "$PX4_SRC" "$PX4_BUILD" >/dev/null 2>&1 || true
export GAZEBO_RESOURCE_PATH="/home/teo/Drone/gazebo_resources:/home/teo/Drone/gazebo_resources/models:/usr/share/gazebo-11${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"
export GAZEBO_MODEL_PATH="/home/teo/Drone/gazebo_resources/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
pkill -9 -f "gzserver|gzclient|px4|MicroXRCEAgent" >/dev/null 2>&1 || true
sleep 1
exec /home/teo/Drone/bench/.venv/bin/python3 /home/teo/Drone/demo/demo2_motion_delta_aerial.py "$@"
