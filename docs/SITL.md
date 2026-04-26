# PX4 SITL + Gazebo setup

This file lists the external dependencies and the launch flow for the Gazebo-based bench harnesses (`bench/sim_texture_probe.py`, `bench/anchor_hover_bench.py`) and the closed-loop hover test (`ros2/offboard_commander/hover_test.py`).

## Dependencies

| Component         | Version       | Notes                                   |
|-------------------|---------------|-----------------------------------------|
| Ubuntu            | 22.04         | ROS Humble's reference platform         |
| PX4-Autopilot     | v1.14+        | Built locally with `make px4_sitl gazebo-classic` |
| Gazebo Classic    | 11            | The plane-with-aerial-texture worlds need Classic, not Garden |
| ROS 2             | Humble        |                                          |
| Micro-XRCE-DDS-Agent | latest    | Bridges PX4 uORB to ROS 2 DDS            |
| Python            | 3.10          | Venv at `bench/.venv/` (torch, opencv, etc.) |

The `PX4-Autopilot/` and `Micro-XRCE-DDS-Agent/` source trees are gitignored — clone them yourself alongside this repo:

```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
```

## ROS 2 workspace

The two ROS 2 packages in `ros2/` (`motion_delta_msgs`, `offboard_commander`) plus PX4's `px4_msgs` need to live in a colcon workspace:

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/PX4/px4_msgs.git -b release/1.14
ln -s /path/to/this/repo/ros2/motion_delta_msgs .
ln -s /path/to/this/repo/ros2/offboard_commander .
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Gazebo resource path

Worlds and models in `gazebo_resources/` need to be on Gazebo's resource path:

```bash
export GAZEBO_RESOURCE_PATH="$(pwd)/gazebo_resources:$(pwd)/gazebo_resources/models:/usr/share/gazebo-11"
export GAZEBO_MODEL_PATH="$(pwd)/gazebo_resources/models"
```

## Ground-truth plugin (load-bearing)

Every world file in `gazebo_resources/worlds/` loads `libgazebo_ros_state.so` as a `WorldPlugin` to expose `/gazebo/model_states`. This is the only trustworthy ground-truth pose during a SITL test — the EKF can lie, the world plugin cannot. Verify after any sim launch:

```bash
ros2 topic hz /gazebo/model_states    # should be ~30–60 Hz
```

If this prints nothing, the plugin tag is missing from the world file or the resource path is wrong.

## Hardcoded paths in ROS 2 nodes

The nodes in `ros2/offboard_commander/` currently insert `/home/teo/Drone/bench/.venv/lib/python3.10/site-packages` into `sys.path` so they can find `torch` and `cv2` without polluting the system Python. To run on a different machine, edit the `sys.path.insert` lines at the top of `anchor_estimator.py`, `anchor_ev_shim.py`, and `displacement_estimator.py` — or set up a single venv on the system Python and remove those lines.

This is dev ergonomics, not a design choice. The shipping firmware is C++ (planned) — see roadmap in [`../README.md`](../README.md).

## Anchor capture during a flight

```bash
# Once at altitude over the area you want to anchor against:
ros2 service call /anchor/capture std_srvs/srv/Trigger
```

The next camera frame is stored as the anchor along with the current barometric altitude (which sets the metric scale for the homography decomposition).
