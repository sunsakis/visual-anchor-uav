#!/bin/bash
# One-command launcher for a PX4 SITL hover test run.
#
# Starts: MicroXRCEAgent, gzserver (headless), gzclient (3D GUI),
# and applies EKF/MPC params. No telemetry tabs — use the gzclient
# window as the single visual source of truth and spot-check topics
# on demand from any terminal.
#
# Usage:  bash /home/teo/Drone/launch_test_panel.sh
# Then:   ros2 run offboard_commander hover_test    (in any terminal)

set -euo pipefail

echo "=== Drone sim launch ==="

echo "[1/5] Killing stale processes..."
pkill -9 -f "px4|gzserver|gzclient|MicroXRCEAgent" 2>/dev/null || true
sleep 2

echo "[2/5] Starting MicroXRCEAgent on UDP 8888..."
nohup MicroXRCEAgent udp4 -p 8888 > /tmp/xrce.log 2>&1 & disown

echo "[3/5] Starting PX4 SITL + gzserver (headless)..."
nohup bash /tmp/start_px4_sitl.sh > /tmp/px4_sitl.log 2>&1 & disown
echo "      Waiting 15 s for boot..."
sleep 15

echo "[4/5] Applying EKF/MPC params..."
python3 /tmp/set_flow_params.py || true

echo "[5/5] Starting gzclient (3D world view)..."
# gzclient resolves model:// URIs itself for rendering — without these env vars
# meshes (e.g. iris.stl body) fail silently and you see only the propellers /
# camera / lidar primitives floating without the iris fuselage.
export PX4_SRC=/home/teo/Drone/PX4-Autopilot
export PX4_BUILD=${PX4_SRC}/build/px4_sitl_default
source "$PX4_SRC/Tools/simulation/gazebo-classic/setup_gazebo.bash" "$PX4_SRC" "$PX4_BUILD" >/dev/null 2>&1 || true
export GAZEBO_RESOURCE_PATH="/home/teo/Drone/gazebo_resources:/usr/share/gazebo-11${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"
nohup gzclient > /tmp/gzclient.log 2>&1 & disown

echo ""
echo "=== SIM UP ==="
echo "  gzclient : Gazebo 3D world (separate window)"
echo ""
echo "To run the hover test in another terminal:"
echo "  source /opt/ros/humble/setup.bash && source /home/teo/ros2_ws/install/setup.bash"
echo "  ros2 run offboard_commander hover_test"
