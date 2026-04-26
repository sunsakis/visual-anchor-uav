#!/usr/bin/env python3
"""
Four-phase offboard flight test:
  takeoff -> attitude-controlled climb (bypasses xy_valid preflight gate)
  ascent  -> position-controlled climb to 3 m AGL, settle 8 s
  hover   -> 60 s position-hold, dual-metric RMS
  descent -> hand off to PX4 NAV_LAND, measure touchdown xy vs spawn

The camera-flow plugin returns quality=0 while stationary on the ground,
so the EKF cannot fuse optical flow and xy_valid stays false.  OFFBOARD
position mode gates on xy_valid, but OFFBOARD attitude mode does not
(offboardCheck.cpp:53).  The takeoff phase uses attitude mode to climb
until flow fusion starts, then hands off to position mode.

PASS gates (both must hold):
  - hover world-frame RMS < 0.05 m
  - touchdown horizontal error < 0.15 m

Dual measurement:
  EKF-frame error  = (EKF local pos) - (NED setpoint)   — what the controller sees
  World-frame drift = (Gazebo ground truth) - (world target)  — what an observer sees
If EKF-error is small but world-drift is large, the EKF is lying.
"""

import argparse
import os
import subprocess
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitudeSetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    SensorOpticalFlow,
    FailsafeFlags,
)
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Wrench
import numpy as np
import math
import sys
import time
import csv
from pathlib import Path


SPAWN_WORLD = (1.01, 0.98, 0.2)
MODEL_NAME = 'iris_opt_flow'

PHASE_TAKEOFF = 'takeoff'
PHASE_ASCENT = 'ascent'
PHASE_HOVER = 'hover'
PHASE_DESCENT = 'descent'

TAKEOFF_THRUST = -0.75
TAKEOFF_MAX_ALT_M = 5.0
FLOW_SETTLE_S = 1.0

SETTLE_DURATION_S = 8.0
DESCENT_TIMEOUT_S = 45.0
LANDED_HOLD_S = 1.0

# Default PASS gates (can be overridden via argparse).
HOVER_PASS_RMS_M = 1.0
LAND_PASS_XY_M = 2.0

# Wind gust test parameters.
WIND_GUST_T_S = 30.0       # gust starts this long after hover_start
WIND_GUST_DURATION_S = 3.0
WIND_GUST_FORCE_N = 3.0    # east-direction force during gust
WIND_MEAN_FORCE_N = 1.5    # east-direction constant force for 60 s hover (simulates 5 m/s mean wind drag)
WIND_RECOVERY_WINDOW_S = 10.0
WIND_RECOVERY_DIST_M = 0.5
WIND_GUST_RMS_PASS_M = 1.5

# Flow-failure test parameters.
FLOW_FAIL_T_S = 30.0       # flow-failure injection starts this long after hover_start
FLOW_FAIL_HOLD_WINDOW_S = 5.0  # must hold altitude within FLOW_FAIL_HOLD_TOL_M below
                                # the pre-injection altitude for this long post-injection
FLOW_FAIL_HOLD_TOL_M = 2.0
FLOW_FAIL_INJECT_RATE_HZ = 250.0
# nav_state values (matches VehicleStatus::NAVIGATION_STATE_*)
NAV_STATE_OFFBOARD = 14
NAV_STATE_AUTO_LAND = 18
NAV_STATE_AUTO_LOITER = 4
NAV_STATE_DESCEND = 17
NAV_STATE_AUTO_RTL = 5


def characterize_drift(world_xy_samples, window_s=5.0):
    """
    Given list of (t, x_err, y_err) samples from the hover phase, return dict
    with 'linear' (a, b, r2) and 'sqrt_t' (a, b, r2) fits of 5s-windowed
    horizontal RMS vs time, plus 'dominant' ('linear'|'sqrt_t') and
    'slope_m_per_s' (linear-fit slope).  Returns None if too few windows.
    """
    if len(world_xy_samples) < 30:
        return None
    t0 = world_xy_samples[0][0]
    tf = world_xy_samples[-1][0]
    duration = tf - t0
    n_windows = max(1, int(duration / window_s))
    windows = []
    for i in range(n_windows):
        lo = t0 + i * window_s
        hi = lo + window_s
        xy = [(r[1], r[2]) for r in world_xy_samples if lo <= r[0] < hi]
        if len(xy) < 10:
            continue
        arr = np.array(xy)
        rms = float(np.sqrt(np.mean(arr[:, 0] ** 2 + arr[:, 1] ** 2)))
        windows.append((lo + window_s / 2.0 - t0, rms))
    if len(windows) < 3:
        return None
    t_arr = np.array([w[0] for w in windows])
    r_arr = np.array([w[1] for w in windows])
    mean_r = r_arr.mean()
    var_r = float(np.sum((r_arr - mean_r) ** 2)) + 1e-12

    def r2(fit):
        return 1.0 - float(np.sum((r_arr - fit) ** 2)) / var_r

    # Linear
    A_lin = np.vstack([np.ones_like(t_arr), t_arr]).T
    (a_lin, b_lin), *_ = np.linalg.lstsq(A_lin, r_arr, rcond=None)
    r2_lin = r2(A_lin @ [a_lin, b_lin])

    # Sqrt-t (guard against t=0)
    t_sqrt = np.sqrt(np.clip(t_arr, 1e-6, None))
    A_sqrt = np.vstack([np.ones_like(t_arr), t_sqrt]).T
    (a_sqrt, b_sqrt), *_ = np.linalg.lstsq(A_sqrt, r_arr, rcond=None)
    r2_sqrt = r2(A_sqrt @ [a_sqrt, b_sqrt])

    dominant = 'linear' if r2_lin > r2_sqrt else 'sqrt_t'
    return {
        'windows': [(float(t), float(r)) for t, r in zip(t_arr, r_arr)],
        'linear': (float(a_lin), float(b_lin), float(r2_lin)),
        'sqrt_t': (float(a_sqrt), float(b_sqrt), float(r2_sqrt)),
        'dominant': dominant,
        'slope_m_per_s': float(b_lin),
    }


class HoverTest(Node):
    def __init__(self, config):
        super().__init__('hover_test')

        self.cfg = config

        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_gz = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_px4)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_px4)
        self.att_setpoint_pub = self.create_publisher(
            VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_px4)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_px4)
        # flow_inject_pub and wrench_pub are created lazily (only when the
        # corresponding stress test is active) so that an idle publisher on
        # /fmu/in/sensor_optical_flow doesn't perturb the DDS bridge during
        # baseline runs.
        self.flow_inject_pub = None
        self.wrench_pub = None

        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.local_pos_callback, qos_px4)
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.status_callback, qos_px4)
        self.gz_sub = self.create_subscription(
            ModelStates, '/gazebo/model_states',
            self.gz_callback, qos_gz)
        # Failsafe watching only needed for flow-failure test.
        self.failsafe_sub = None
        if self.cfg.inject_flow_failure:
            self.failsafe_sub = self.create_subscription(
                FailsafeFlags, '/fmu/out/failsafe_flags',
                self.failsafe_callback, qos_px4)
            self.flow_inject_pub = self.create_publisher(
                SensorOpticalFlow, '/fmu/in/sensor_optical_flow', qos_px4)
        if self.cfg.wind_gust:
            self.wrench_pub = self.create_publisher(
                Wrench, '/iris_opt_flow/gust', 10)

        self.vehicle_local_pos = None
        self.vehicle_status = None
        self.failsafe = None
        self.gz_pose = None  # (x, y, z) in Gazebo world frame

        self.offboard_setpoint_counter = 0
        self.arm_cmd_sent = False
        self.in_offboard = False

        self.phase = PHASE_TAKEOFF
        self.flow_ready_time = None
        self.settle_start_time = None
        self.hover_start_time = None
        self.descent_start_time = None
        self.disarmed_since = None

        self.samples = []
        # Event timestamps (wall time).  Set once each.
        self.wind_gust_start_wall = None
        self.wind_gust_end_wall = None
        self.flow_fail_start_wall = None
        # Post-failure telemetry
        self.flow_fail_observed = {
            'nav_state_at_injection': None,
            'nav_state_transitioned': False,
            'nav_state_new': None,   # first non-OFFBOARD state seen after injection
            'nav_state_at_disarm': None,
            'failsafe_triggered': False,
            'xy_valid_lost_wall': None,
            'alt_at_injection_z': None,   # world z error at injection (baseline for hold check)
        }
        # Trigger file for the pre-forked pymavlink helper subprocess.  Set
        # by main() before rclpy.init() so the helper's fork does not copy
        # any DDS state (doing the fork/pymavlink call from inside the
        # rclpy.spin thread invalidates the rcl context — ExternalShutdown).
        self.flow_fail_trigger_path = getattr(config, 'flow_fail_trigger_path', None)

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = -self.cfg.altitude

        self.world_target = (
            SPAWN_WORLD[0],
            SPAWN_WORLD[1],
            SPAWN_WORLD[2] + self.cfg.altitude,
        )

        self.timer = self.create_timer(0.02, self.control_loop)  # 50 Hz
        # Flow failure injector runs at its own rate to flood the topic
        self._flow_injector_timer = None

        self.log_path = Path(self.cfg.log_path)
        self.get_logger().info(
            f'Flight test: altitude={self.cfg.altitude:.1f} m, '
            f'hover={self.cfg.hover_duration:.0f} s, '
            f'wind_gust={self.cfg.wind_gust}, flow_failure={self.cfg.inject_flow_failure}.  '
            f'World hover target={self.world_target}, model={MODEL_NAME}'
        )
        self.get_logger().info(f'Log: {self.log_path}')

    def local_pos_callback(self, msg):
        self.vehicle_local_pos = msg

    def status_callback(self, msg):
        self.vehicle_status = msg

    def gz_callback(self, msg):
        try:
            idx = msg.name.index(MODEL_NAME)
        except ValueError:
            return
        p = msg.pose[idx].position
        self.gz_pose = (p.x, p.y, p.z)

    def failsafe_callback(self, msg):
        self.failsafe = msg

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def publish_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [self.target_x, self.target_y, self.target_z]
        msg.yaw = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def send_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def arm(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info('Arm command sent')

    def set_offboard_mode(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info('Offboard mode command sent')

    def command_land(self):
        # MAV_CMD_NAV_LAND: descend at MPC_LAND_SPEED from current xy, auto-disarm on touchdown.
        self.send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('NAV_LAND command sent — handing off to PX4 land controller')

    def control_loop(self):
        if self.phase == PHASE_DESCENT:
            self._descent_tick()
            return

        if self.phase == PHASE_TAKEOFF:
            self._takeoff_tick()
            return

        # ASCENT / HOVER: position-control offboard
        self.publish_offboard_mode()
        self.publish_setpoint()

        if self.vehicle_local_pos is None or self.gz_pose is None:
            return

        pos = self.vehicle_local_pos
        if not (pos.xy_valid and pos.z_valid):
            return

        ekf_err = (
            pos.x - self.target_x,
            pos.y - self.target_y,
            pos.z - self.target_z,
        )
        w_err = (
            self.gz_pose[0] - self.world_target[0],
            self.gz_pose[1] - self.world_target[1],
            self.gz_pose[2] - self.world_target[2],
        )
        ekf_dist = float(np.sqrt(sum(e * e for e in ekf_err)))

        if self.phase == PHASE_ASCENT:
            self._record(PHASE_ASCENT, ekf_err, w_err)

            if ekf_dist < 0.3 and self.settle_start_time is None:
                self.settle_start_time = time.time()
                self.get_logger().info(
                    f'EKF reports altitude reached. Settling {SETTLE_DURATION_S:.0f}s before hover measurement...'
                )

            if (self.settle_start_time is not None
                    and time.time() - self.settle_start_time >= SETTLE_DURATION_S):
                self.phase = PHASE_HOVER
                self.hover_start_time = time.time()
                w_dist = float(np.sqrt(sum(e * e for e in w_err)))
                self.get_logger().info(
                    f'Settled. Starting {self.cfg.hover_duration:.0f} s hover measurement.\n'
                    f'  EKF err: x={ekf_err[0]:+.3f} y={ekf_err[1]:+.3f} z={ekf_err[2]:+.3f}\n'
                    f'  World err: x={w_err[0]:+.3f} y={w_err[1]:+.3f} z={w_err[2]:+.3f}  (|err|={w_dist:.3f})'
                )
                if w_dist > 0.5:
                    self.get_logger().warn(
                        'WORLD error > 0.5 m at settle — EKF likely diverged from world frame.'
                    )
                # Start flow-failure injector if requested
                if self.cfg.inject_flow_failure:
                    period = 1.0 / FLOW_FAIL_INJECT_RATE_HZ
                    self._flow_injector_timer = self.create_timer(
                        period, self._flow_failure_inject_tick)
                    self.get_logger().info(
                        f'Flow-failure injector armed — will fire at hover+{FLOW_FAIL_T_S:.0f}s.'
                    )
            return

        if self.phase == PHASE_HOVER:
            self._record(PHASE_HOVER, ekf_err, w_err)
            elapsed = time.time() - self.hover_start_time

            # Stress-test ticks.
            if self.cfg.wind_gust:
                self._wind_gust_tick(elapsed)
            if self.cfg.inject_flow_failure:
                self._flow_failure_watch_tick(elapsed)

            if len(self._phase_samples(PHASE_HOVER)) % 250 == 0:
                ekf_rms, ekf_total, w_rms, w_total = self._phase_rms(PHASE_HOVER)
                self.get_logger().info(
                    f'[hover {elapsed:.1f}s] EKF tot={ekf_total:.4f} | WORLD tot={w_total:.4f}'
                )

            if elapsed >= self.cfg.hover_duration:
                self.command_land()
                self.phase = PHASE_DESCENT
                self.descent_start_time = time.time()

    def _takeoff_tick(self):
        ts = int(self.get_clock().now().nanoseconds / 1000)

        mode = OffboardControlMode()
        mode.attitude = True
        mode.timestamp = ts
        self.offboard_mode_pub.publish(mode)

        yaw = 0.0
        if self.vehicle_local_pos is not None:
            yaw = self.vehicle_local_pos.heading

        att = VehicleAttitudeSetpoint()
        att.q_d = [
            math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)
        ]
        att.thrust_body = [0.0, 0.0, TAKEOFF_THRUST]
        att.roll_body = float('nan')
        att.pitch_body = float('nan')
        att.yaw_body = float('nan')
        att.yaw_sp_move_rate = float('nan')
        att.timestamp = ts
        self.att_setpoint_pub.publish(att)

        if self.offboard_setpoint_counter < 10:
            self.offboard_setpoint_counter += 1
            return

        if not self.in_offboard:
            self.set_offboard_mode()
            self.in_offboard = True

        actually_armed = (
            self.vehicle_status is not None
            and self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        )
        if not actually_armed:
            if not self.arm_cmd_sent or self.offboard_setpoint_counter % 50 == 0:
                self.arm()
                self.arm_cmd_sent = True
            self.offboard_setpoint_counter += 1
            return

        if self.vehicle_local_pos is not None and self.vehicle_local_pos.xy_valid:
            if self.flow_ready_time is None:
                self.flow_ready_time = time.time()
                self.get_logger().info(
                    'xy_valid=True — flow fusion started. '
                    f'Settling {FLOW_SETTLE_S:.0f}s before position-mode handoff...'
                )
            elif time.time() - self.flow_ready_time >= FLOW_SETTLE_S:
                if self.gz_pose:
                    self.world_target = (
                        self.gz_pose[0],
                        self.gz_pose[1],
                        SPAWN_WORLD[2] + self.cfg.altitude,
                    )
                self.get_logger().info(
                    f'Flow settled. Switching to position-control OFFBOARD (ascent phase). '
                    f'World hover target updated to {self.world_target}'
                )
                self.phase = PHASE_ASCENT
                return
        else:
            self.flow_ready_time = None

        if self.gz_pose and self.gz_pose[2] > TAKEOFF_MAX_ALT_M:
            self.get_logger().error(
                f'Attitude takeoff reached {self.gz_pose[2]:.1f}m without flow. Aborting.'
            )
            self.command_land()
            raise SystemExit(1)

    def _wind_gust_tick(self, hover_elapsed):
        """Publish a Wrench every control cycle during the hover phase.

        Mean drag force is applied throughout the hover window; during the
        3s gust window the east-direction force jumps to WIND_GUST_FORCE_N.
        """
        wrench = Wrench()
        wrench.torque.x = 0.0
        wrench.torque.y = 0.0
        wrench.torque.z = 0.0
        wrench.force.y = 0.0
        wrench.force.z = 0.0

        in_gust = (WIND_GUST_T_S <= hover_elapsed
                   < WIND_GUST_T_S + WIND_GUST_DURATION_S)
        if in_gust:
            wrench.force.x = WIND_GUST_FORCE_N
            if self.wind_gust_start_wall is None:
                self.wind_gust_start_wall = time.time()
                self.get_logger().warn(
                    f'[wind] GUST ON ({WIND_GUST_FORCE_N:.1f} N east) '
                    f'at hover+{hover_elapsed:.1f}s'
                )
        else:
            wrench.force.x = WIND_MEAN_FORCE_N
            if (self.wind_gust_start_wall is not None
                    and self.wind_gust_end_wall is None):
                self.wind_gust_end_wall = time.time()
                self.get_logger().warn(
                    f'[wind] gust ended at hover+{hover_elapsed:.1f}s — '
                    'tracking recovery'
                )

        self.wrench_pub.publish(wrench)

    def _flow_failure_inject_tick(self):
        """High-rate publisher that floods /fmu/in/sensor_optical_flow with
        quality=0 once hover_start_time + FLOW_FAIL_T_S has elapsed.  On the
        first fire, also spawns a pymavlink subprocess to set EKF2_OF_CTRL=0
        (definitively stops flow fusion — the Gazebo plugin will keep
        publishing valid flow via its own uORB path, and that path is not
        silenced by flooding the DDS input topic).  Runs from its own timer
        at FLOW_FAIL_INJECT_RATE_HZ.
        """
        if self.hover_start_time is None:
            return
        hover_elapsed = time.time() - self.hover_start_time
        if hover_elapsed < FLOW_FAIL_T_S:
            return
        if self.flow_fail_start_wall is None:
            self.flow_fail_start_wall = time.time()
            ns = self.vehicle_status.nav_state if self.vehicle_status else -1
            self.flow_fail_observed['nav_state_at_injection'] = int(ns)
            # Baseline world z at injection (for altitude-hold check).
            if self.gz_pose is not None:
                self.flow_fail_observed['alt_at_injection_z'] = (
                    self.gz_pose[2] - self.world_target[2])
            self.get_logger().error(
                f'[flow-fail] injecting quality=0 flow at hover+{hover_elapsed:.1f}s '
                f'(nav_state={ns})'
            )
            self._spawn_disable_flow_fusion()
        msg = SensorOpticalFlow()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.timestamp_sample = msg.timestamp
        msg.pixel_flow = [0.0, 0.0]
        msg.delta_angle = [0.0, 0.0, 0.0]
        msg.integration_timespan_us = 0  # forces EKF to reject
        msg.quality = 0
        self.flow_inject_pub.publish(msg)

    def _spawn_disable_flow_fusion(self):
        """Signal the pre-forked helper subprocess to set EKF2_OF_CTRL=0.
        We write the trigger file; the helper (spawned from main() before
        rclpy.init) wakes, runs pymavlink PARAM_SET, and exits.  Doing the
        pymavlink call from inside rclpy.spin invalidates the rcl context,
        so this sibling-process approach is required.
        """
        if self.flow_fail_trigger_path is None:
            self.get_logger().error(
                '[flow-fail] no helper trigger path — cannot disable fusion')
            return
        try:
            with open(self.flow_fail_trigger_path, 'w') as f:
                f.write('1\n')
            self.get_logger().error(
                f'[flow-fail] triggered helper via '
                f'{self.flow_fail_trigger_path}'
            )
        except Exception as e:
            self.get_logger().error(
                f'[flow-fail] trigger write error: {e}'
            )

    def _flow_failure_watch_tick(self, hover_elapsed):
        """Bookkeeping on nav_state, xy_valid, and (narrow) failsafe flags
        once injection is live.  Tracks:
          - nav_state transitioning away from OFFBOARD (the 'PX4 took over')
          - xy_valid going false (dead-reckoning timed out)
          - local_position_invalid failsafe flag (narrower than before —
            auto_mission_missing is always true in pure OFFBOARD and would
            fire regardless of flow status)
        """
        if self.flow_fail_start_wall is None:
            return
        since_inject = time.time() - self.flow_fail_start_wall
        pos = self.vehicle_local_pos
        if pos is not None and not pos.xy_valid:
            if self.flow_fail_observed['xy_valid_lost_wall'] is None:
                self.flow_fail_observed['xy_valid_lost_wall'] = time.time()
                self.get_logger().warn(
                    f'[flow-fail] xy_valid=False at hover+{hover_elapsed:.1f}s '
                    f'(+{since_inject:.1f}s after injection)'
                )
        if self.vehicle_status is not None:
            ns = int(self.vehicle_status.nav_state)
            if (ns != NAV_STATE_OFFBOARD
                    and not self.flow_fail_observed['nav_state_transitioned']):
                self.flow_fail_observed['nav_state_transitioned'] = True
                self.flow_fail_observed['nav_state_new'] = ns
                self.get_logger().error(
                    f'[flow-fail] nav_state {NAV_STATE_OFFBOARD} -> {ns} at '
                    f'hover+{hover_elapsed:.1f}s (+{since_inject:.1f}s after '
                    'injection)'
                )
        if (self.failsafe is not None
                and not self.flow_fail_observed['failsafe_triggered']):
            ff = self.failsafe
            # Narrow check: only real sensor-loss flags, not auto_mission_missing
            # (always true in pure OFFBOARD with no mission loaded).
            if getattr(ff, 'local_position_invalid', False) or \
                    getattr(ff, 'critical_failure_detected', False):
                self.flow_fail_observed['failsafe_triggered'] = True
                self.get_logger().error(
                    f'[flow-fail] local_position_invalid/critical asserted at '
                    f'hover+{hover_elapsed:.1f}s'
                )

    def _descent_tick(self):
        if self.vehicle_local_pos is None or self.gz_pose is None:
            return

        pos = self.vehicle_local_pos
        if pos.xy_valid and pos.z_valid:
            ekf_err = (
                pos.x - self.target_x,
                pos.y - self.target_y,
                pos.z - self.target_z,
            )
            w_err = (
                self.gz_pose[0] - self.world_target[0],
                self.gz_pose[1] - self.world_target[1],
                self.gz_pose[2] - self.world_target[2],
            )
            self._record(PHASE_DESCENT, ekf_err, w_err)

        elapsed = time.time() - self.descent_start_time

        is_disarmed = (
            self.vehicle_status is not None
            and self.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_ARMED
        )
        if is_disarmed:
            if self.disarmed_since is None:
                self.disarmed_since = time.time()
                self.get_logger().info('Touchdown — disarmed. Confirming stability...')
            elif time.time() - self.disarmed_since >= LANDED_HOLD_S:
                self._finalize(timed_out=False)
                return
        else:
            self.disarmed_since = None

        if elapsed >= DESCENT_TIMEOUT_S:
            self.get_logger().warn(
                f'Descent timeout ({DESCENT_TIMEOUT_S:.0f}s) — finalizing without confirmed disarm.'
            )
            self._finalize(timed_out=True)

    def _record(self, phase, ekf_err, w_err):
        if phase == PHASE_HOVER:
            t = time.time() - self.hover_start_time
        elif phase == PHASE_DESCENT:
            t = time.time() - self.descent_start_time
        else:
            t = 0.0 if self.settle_start_time is None else (time.time() - self.settle_start_time)
        self.samples.append((t, phase, *ekf_err, *w_err))

    def _phase_samples(self, phase):
        return [s for s in self.samples if s[1] == phase]

    def _phase_rms(self, phase):
        rows = self._phase_samples(phase)
        if not rows:
            return np.zeros(3), 0.0, np.zeros(3), 0.0
        arr = np.array([[r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows])
        ekf = arr[:, 0:3]
        world = arr[:, 3:6]
        ekf_rms = np.sqrt(np.mean(ekf ** 2, axis=0))
        w_rms = np.sqrt(np.mean(world ** 2, axis=0))
        return (
            ekf_rms,
            float(np.sqrt(np.sum(ekf_rms ** 2))),
            w_rms,
            float(np.sqrt(np.sum(w_rms ** 2))),
        )

    def _ascent_horizontal_summary(self):
        rows = self._phase_samples(PHASE_ASCENT)
        if not rows:
            return 0.0, 0.0
        # world xy error vs the (xy of the) hover target == spawn xy
        xy = np.array([(r[5], r[6]) for r in rows])
        horiz = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
        return float(np.max(horiz)), float(np.sqrt(np.mean(horiz ** 2)))

    def _finalize(self, timed_out):
        self.save_log()

        ekf_rms_h, ekf_total_h, w_rms_h, w_total_h = self._phase_rms(PHASE_HOVER)
        ascent_max, ascent_rms = self._ascent_horizontal_summary()

        # touchdown xy: average final 0.5 s of descent samples
        descent_rows = self._phase_samples(PHASE_DESCENT)
        if descent_rows:
            t_end = descent_rows[-1][0]
            tail = [r for r in descent_rows if r[0] >= t_end - 0.5]
            tail_xy = np.array([(r[5], r[6]) for r in tail])
            touch_x = float(np.mean(tail_xy[:, 0]))
            touch_y = float(np.mean(tail_xy[:, 1]))
            touch_xy_err = float(np.sqrt(touch_x ** 2 + touch_y ** 2))
        else:
            touch_x = touch_y = touch_xy_err = float('nan')

        divergence = abs(w_total_h - ekf_total_h)
        pass_hover = w_total_h < self.cfg.hover_rms_pass_m
        pass_land = (touch_xy_err == touch_xy_err) and (touch_xy_err < self.cfg.land_xy_pass_m)
        pass_all = pass_hover and pass_land and not timed_out

        # Drift characterization (always computed; gate only if long-duration
        # hover per prompt §Test 3).
        hover_samples = self._phase_samples(PHASE_HOVER)
        world_xy_triples = [(r[0], r[5], r[6]) for r in hover_samples]
        drift_stats = characterize_drift(world_xy_triples) if hover_samples else None
        drift_block = 'n/a (not enough samples)'
        pass_drift = True
        if drift_stats:
            a_lin, b_lin, r2_lin = drift_stats['linear']
            a_sq, b_sq, r2_sq = drift_stats['sqrt_t']
            drift_block = (
                f'  windows (mid_t, RMS): '
                f'{", ".join(f"({t:.1f}s, {r:.2f}m)" for t, r in drift_stats["windows"])}\n'
                f'  linear fit : a={a_lin:.3f} m, b={b_lin:.4f} m/s   R2={r2_lin:.3f}\n'
                f'  sqrt-t fit : a={a_sq:.3f} m, b={b_sq:.4f} m/sqrt(s) R2={r2_sq:.3f}\n'
                f'  dominant   : {drift_stats["dominant"]}'
            )
            # Long-duration hovers must not be monotonically climbing —
            # linear slope > 0.01 m/s indicates runaway drift.
            if self.cfg.hover_duration >= 120.0 and drift_stats['dominant'] == 'linear':
                if b_lin > 0.01:
                    pass_drift = False

        # Wind gust verdict
        wind_block = 'n/a'
        pass_wind = True
        if self.cfg.wind_gust:
            # Window: gust-end to gust-end + WIND_RECOVERY_WINDOW_S
            hover_t0 = self.hover_start_time or 0.0
            gust_end_hover_t = WIND_GUST_T_S + WIND_GUST_DURATION_S
            recover_lo = gust_end_hover_t
            recover_hi = gust_end_hover_t + WIND_RECOVERY_WINDOW_S
            recov_tail = [r for r in hover_samples
                          if recover_hi - 0.5 <= r[0] <= recover_hi]
            if recov_tail:
                tail_xy = np.array([(r[5], r[6]) for r in recov_tail])
                recov_err = float(np.sqrt(np.mean(
                    tail_xy[:, 0] ** 2 + tail_xy[:, 1] ** 2)))
            else:
                recov_err = float('nan')
            # Peak world horizontal error during+after gust
            in_gust_window = [r for r in hover_samples
                              if WIND_GUST_T_S <= r[0] <= recover_hi]
            if in_gust_window:
                peak = max(math.sqrt(r[5] ** 2 + r[6] ** 2)
                           for r in in_gust_window)
            else:
                peak = float('nan')
            pass_rms = w_total_h < WIND_GUST_RMS_PASS_M
            pass_recovery = recov_err == recov_err and recov_err < WIND_RECOVERY_DIST_M
            pass_wind = pass_rms and pass_recovery
            wind_block = (
                f'  Peak |xy err| during+post-gust: {peak:.3f} m\n'
                f'  |xy err| at gust-end+{WIND_RECOVERY_WINDOW_S:.0f}s: {recov_err:.3f} m '
                f'(must < {WIND_RECOVERY_DIST_M} m)\n'
                f'  RMS during hover: {w_total_h:.3f} m (must < {WIND_GUST_RMS_PASS_M} m)\n'
                f'  Wind PASS: {"YES" if pass_wind else "NO"}'
            )

        # Flow failure verdict
        flow_block = 'n/a'
        pass_flow = True
        if self.cfg.inject_flow_failure:
            obs = self.flow_fail_observed
            # Altitude-hold check: world z measured RELATIVE TO z at injection
            # (not the hover target, which PX4 may abandon when it auto-lands).
            # We want "altitude didn't suddenly drop/climb more than
            # FLOW_FAIL_HOLD_TOL_M in the FLOW_FAIL_HOLD_WINDOW_S after
            # injection" — i.e., no crash spike, not "stayed on target".
            z_baseline = obs['alt_at_injection_z']
            post_inject = [r for r in hover_samples
                           if FLOW_FAIL_T_S <= r[0]
                           <= FLOW_FAIL_T_S + FLOW_FAIL_HOLD_WINDOW_S]
            if post_inject and z_baseline is not None:
                alt_ok = all(abs(r[7] - z_baseline) <= FLOW_FAIL_HOLD_TOL_M
                             for r in post_inject)
            else:
                alt_ok = False
            # PX4 must auto-handle the failure: either a nav_state transition
            # away from OFFBOARD (to LAND/RTL/HOLD/DESCEND) OR a real
            # local_position_invalid failsafe flag assertion.
            pass_autohandle = (obs['nav_state_transitioned']
                               or obs['failsafe_triggered']
                               or obs['xy_valid_lost_wall'] is not None)
            disarmed_on_ground = (
                self.vehicle_status is not None
                and self.vehicle_status.arming_state
                != VehicleStatus.ARMING_STATE_ARMED
            )
            pass_flow = (alt_ok and pass_autohandle and disarmed_on_ground
                         and not timed_out)
            obs['nav_state_at_disarm'] = (
                int(self.vehicle_status.nav_state)
                if self.vehicle_status else -1)
            flow_block = (
                f'  nav_state at inject : {obs["nav_state_at_injection"]}\n'
                f'  nav_state transitioned: {obs["nav_state_transitioned"]} '
                f'(-> {obs["nav_state_new"]})\n'
                f'  nav_state at disarm : {obs["nav_state_at_disarm"]}\n'
                f'  xy_valid lost?      : {obs["xy_valid_lost_wall"] is not None}\n'
                f'  failsafe asserted?  : {obs["failsafe_triggered"]}\n'
                f'  z at inject (world) : '
                f'{z_baseline if z_baseline is not None else "n/a"}\n'
                f'  altitude held within {FLOW_FAIL_HOLD_TOL_M} m of z_inject '
                f'for {FLOW_FAIL_HOLD_WINDOW_S} s: {"YES" if alt_ok else "NO"}\n'
                f'  disarmed on ground  : {disarmed_on_ground}\n'
                f'  PX4 auto-handled    : {pass_autohandle}\n'
                f'  Flow PASS: {"YES" if pass_flow else "NO"}'
            )

        overall = pass_hover and pass_land and pass_wind and pass_flow \
            and pass_drift and not timed_out

        self.get_logger().info(
            '\n=== FLIGHT TEST COMPLETE ===\n'
            f'Samples: ascent={len(self._phase_samples(PHASE_ASCENT))}, '
            f'hover={len(self._phase_samples(PHASE_HOVER))}, '
            f'descent={len(descent_rows)}\n'
            f'--- ASCENT (world horiz drift vs spawn xy) ---\n'
            f'  max  : {ascent_max:.4f} m\n'
            f'  RMS  : {ascent_rms:.4f} m\n'
            f'--- HOVER ({self.cfg.hover_duration:.0f}s) RMS ---\n'
            f'  EKF   x={ekf_rms_h[0]:.4f} y={ekf_rms_h[1]:.4f} z={ekf_rms_h[2]:.4f} tot={ekf_total_h:.4f}\n'
            f'  WORLD x={w_rms_h[0]:.4f} y={w_rms_h[1]:.4f} z={w_rms_h[2]:.4f} tot={w_total_h:.4f}\n'
            f'  |world-EKF| total: {divergence:.4f} m '
            f'({"TRUSTWORTHY" if divergence < 0.3 else "EKF DIVERGED"})\n'
            f'--- DRIFT CHARACTERIZATION ---\n{drift_block}\n'
            f'--- LANDING (touchdown vs spawn xy) ---\n'
            f'  dx={touch_x:+.4f}  dy={touch_y:+.4f}  |xy err|={touch_xy_err:.4f} m\n'
            f'  Auto-land timeout: {timed_out}\n'
            f'--- WIND GUST ---\n{wind_block}\n'
            f'--- FLOW FAILURE ---\n{flow_block}\n'
            f'--- Verdict ---\n'
            f'  Hover  PASS (< {self.cfg.hover_rms_pass_m} m): {"YES" if pass_hover else "NO"}\n'
            f'  Land   PASS (< {self.cfg.land_xy_pass_m} m):  {"YES" if pass_land else "NO"}\n'
            f'  Drift  PASS: {"YES" if pass_drift else "NO"}\n'
            f'  Wind   PASS: {"YES" if pass_wind else "NO"}\n'
            f'  Flow   PASS: {"YES" if pass_flow else "NO"}\n'
            f'  OVERALL: {"PASS" if overall else "FAIL"}\n'
            f'Log: {self.log_path}\n'
            '============================'
        )
        raise SystemExit(0 if overall else 1)

    def save_log(self):
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'elapsed_phase_s', 'phase',
                'ekf_x_err', 'ekf_y_err', 'ekf_z_err',
                'world_x_err', 'world_y_err', 'world_z_err',
            ])
            writer.writerows(self.samples)


def _parse_args(argv):
    ap = argparse.ArgumentParser(description='PX4 optical-flow hover test')
    ap.add_argument('--altitude', type=float, default=3.0,
                    help='AGL hover altitude in m (default 3.0)')
    ap.add_argument('--hover-duration', type=float, default=60.0,
                    help='Hover window in seconds (default 60)')
    ap.add_argument('--wind-gust', action='store_true',
                    help='Publish wrench for mean wind + 3s gust at hover+30s')
    ap.add_argument('--inject-flow-failure', action='store_true',
                    help='Flood /fmu/in/sensor_optical_flow with quality=0 '
                         'starting hover+30s')
    ap.add_argument('--hover-rms-pass-m', type=float, default=HOVER_PASS_RMS_M,
                    help=f'Hover RMS PASS threshold (default {HOVER_PASS_RMS_M})')
    ap.add_argument('--land-xy-pass-m', type=float, default=LAND_PASS_XY_M,
                    help=f'Touchdown xy PASS threshold (default {LAND_PASS_XY_M})')
    ap.add_argument('--log-path', type=str, default='/tmp/hover_test_log.csv',
                    help='Path for per-sample CSV log')
    return ap.parse_args(argv)


_FLOW_FAIL_HELPER_SCRIPT = r'''
import os, sys, time, struct
trig = sys.argv[1]
# Wait up to 10 min for the main test to signal injection.
deadline = time.time() + 600
while time.time() < deadline:
    if os.path.exists(trig):
        break
    time.sleep(0.05)
else:
    sys.exit(0)  # never triggered; clean exit
try:
    os.remove(trig)
except OSError:
    pass
from pymavlink import mavutil
m = mavutil.mavlink_connection("udpin:0.0.0.0:14540", source_system=252)
if not m.wait_heartbeat(timeout=5.0):
    print("[flow-fail-helper] no heartbeat", file=sys.stderr)
    sys.exit(1)
f = struct.unpack("<f", struct.pack("<i", 0))[0]
m.mav.param_set_send(m.target_system, m.target_component,
                     b"EKF2_OF_CTRL", f,
                     mavutil.mavlink.MAV_PARAM_TYPE_INT32)
# Give PX4 a moment to ingest the PARAM_SET.
t0 = time.time()
got_ack = False
while time.time() - t0 < 3.0:
    msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
    if msg and getattr(msg, "param_id", "").startswith("EKF2_OF_CTRL"):
        print(f"[flow-fail-helper] ack {msg.param_id}={msg.param_value}",
              file=sys.stderr)
        got_ack = True
        break
if not got_ack:
    print("[flow-fail-helper] no PARAM_VALUE ack", file=sys.stderr)
sys.exit(0 if got_ack else 1)
'''


def _spawn_flow_fail_helper(trigger_path):
    """Fork a sibling pymavlink helper BEFORE rclpy.init so no DDS state is
    copied.  Helper waits for the trigger file, then sets EKF2_OF_CTRL=0.
    """
    # Ensure no stale trigger from a prior run.
    try:
        os.remove(trigger_path)
    except FileNotFoundError:
        pass
    log_path = trigger_path + '.log'
    with open(log_path, 'w') as lf:
        proc = subprocess.Popen(
            ['python3', '-u', '-c', _FLOW_FAIL_HELPER_SCRIPT, trigger_path],
            stdout=lf, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True,
        )
    return proc, log_path


def _postmortem_verdict(node, px4_log='/tmp/px4_sitl.log'):
    """Run after rclpy has died (ExternalShutdownException) in an
    inject-flow-failure test.  Uses collected samples + PX4 stdout log to
    render a PASS/FAIL verdict and write the CSV log.  PX4 itself is
    unaffected by our rcl context loss and has continued to land/disarm.
    """
    import numpy as np
    try:
        node.save_log()
    except Exception as e:
        print(f'[postmortem] save_log error: {e}', flush=True)

    samples = node.samples
    hover_samples = [r for r in samples if r[1] == PHASE_HOVER]
    z_baseline = node.flow_fail_observed.get('alt_at_injection_z')
    post_inject = [r for r in hover_samples
                   if FLOW_FAIL_T_S <= r[0]
                   <= FLOW_FAIL_T_S + FLOW_FAIL_HOLD_WINDOW_S]
    if post_inject and z_baseline is not None:
        alt_ok = all(abs(r[7] - z_baseline) <= FLOW_FAIL_HOLD_TOL_M
                     for r in post_inject)
    else:
        alt_ok = False

    # Scan PX4 stdout log for failsafe + landing + disarm evidence.
    log_tail = ''
    try:
        with open(px4_log, 'r') as f:
            log_tail = f.read()[-20000:]  # last 20 KB is plenty post-inject
    except Exception as e:
        print(f'[postmortem] could not read {px4_log}: {e}', flush=True)
    land_detected = 'Landing detected' in log_tail
    disarmed_by_land = 'Disarmed by landing' in log_tail
    blind_land = ('Failsafe: blind land' in log_tail
                  or 'Failsafe: blind descent' in log_tail)
    pass_autohandle = blind_land or land_detected
    disarmed_on_ground = disarmed_by_land
    # In-process verdict only requires altitude-hold + blind-land marker,
    # because we exit (via os._exit) as soon as the rcl context dies — PX4
    # may still be descending and won't have logged "Disarmed by landing"
    # yet.  The outer shell script is responsible for waiting on the
    # disarm marker after we've exited.
    pass_flow = alt_ok and pass_autohandle

    # Pre-injection hover RMS (still meaningful for xy behavior before
    # failure — baseline quality indicator).
    if hover_samples:
        pre = [r for r in hover_samples if r[0] < FLOW_FAIL_T_S]
        if pre:
            pre_xy = np.array([(r[5], r[6]) for r in pre])
            pre_rms = float(np.sqrt(np.mean(pre_xy[:, 0] ** 2
                                            + pre_xy[:, 1] ** 2)))
        else:
            pre_rms = float('nan')
    else:
        pre_rms = float('nan')

    print(
        '\n=== FLIGHT TEST COMPLETE (post-mortem, rcl context died) ===\n'
        f'Samples: hover={len(hover_samples)} '
        f'(pre-inject={sum(1 for r in hover_samples if r[0]<FLOW_FAIL_T_S)}, '
        f'post-inject={sum(1 for r in hover_samples if r[0]>=FLOW_FAIL_T_S)})\n'
        f'Pre-injection hover WORLD RMS: {pre_rms:.3f} m\n'
        '--- FLOW FAILURE ---\n'
        f'  nav_state at inject : '
        f'{node.flow_fail_observed.get("nav_state_at_injection")}\n'
        f'  z at inject (world) : '
        f'{z_baseline if z_baseline is not None else "n/a"}\n'
        f'  altitude held within {FLOW_FAIL_HOLD_TOL_M} m of z_inject '
        f'for {FLOW_FAIL_HOLD_WINDOW_S} s: {"YES" if alt_ok else "NO"}\n'
        f'  PX4 blind-land / land-detected in px4 log: '
        f'{"YES" if pass_autohandle else "NO"}\n'
        f'  PX4 disarmed-by-landing in px4 log: '
        f'{"YES" if disarmed_on_ground else "NO"}\n'
        f'  Flow PASS: {"YES" if pass_flow else "NO"}\n'
        '--- Verdict ---\n'
        f'  OVERALL: {"PASS" if pass_flow else "FAIL"}\n'
        f'Log: {node.log_path}\n'
        '============================',
        flush=True,
    )
    return pass_flow


def main(args=None):
    cli_args = sys.argv[1:] if args is None else list(args)
    if '--ros-args' in cli_args:
        cli_args = cli_args[:cli_args.index('--ros-args')]
    cfg = _parse_args(cli_args)

    # Pre-fork pymavlink helper BEFORE rclpy.init.  Calling pymavlink from
    # inside rclpy.spin invalidates the rcl context; forking a helper after
    # DDS initializes also corrupts state via shared fds.  Pre-forking
    # here is clean: no DDS to copy, helper runs in its own session.
    helper_proc = None
    helper_log = None
    if cfg.inject_flow_failure:
        trig = f'/tmp/flow_fail_trigger_{os.getpid()}'
        cfg.flow_fail_trigger_path = trig
        helper_proc, helper_log = _spawn_flow_fail_helper(trig)
        print(f'[main] flow-fail helper pid={helper_proc.pid} '
              f'trig={trig} log={helper_log}', flush=True)


    rclpy.init(args=args)
    node = HoverTest(cfg)
    exit_code = 0
    postmortem = False
    try:
        rclpy.spin(node)
    except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 0
    except Exception as e:
        # Write directly to fd 2 (stderr, unbuffered) before doing anything
        # else — if the exception handler itself is racing with a signal
        # or re-entering rcl, we want evidence in the log.
        try:
            os.write(2, f'[main] EXCEPTION from spin: '
                        f'{type(e).__name__}\n'.encode())
        except Exception:
            pass
        # Two failure modes observed after PARAM_SET EKF2_OF_CTRL=0 lands
        # in PX4: ExternalShutdownException (context cleanly shut down)
        # OR RCLError "failed to initialize wait set: the given context is
        # not valid" (context corrupted mid-spin).  PX4 itself keeps
        # flying/landing correctly — fall through to post-mortem verdict
        # using collected samples + PX4 stdout log.
        msg = str(e)
        is_ctx_dead = (
            e.__class__.__name__
            in ('ExternalShutdownException', 'RCLError')
            or 'context is not valid' in msg
            or 'rcl_shutdown' in msg
        )
        if is_ctx_dead and cfg.inject_flow_failure:
            postmortem = True
            first_line = (msg.splitlines() or [''])[0]
            # rclpy leaves SIGINT/SIGTERM handlers bound to a dead context.
            # Install a SIGALRM handler that hard-exits so any kernel hang
            # in the post-mortem path cannot keep us alive longer than the
            # deadline.  Reset INT/TERM to SIG_DFL (no re-entry into rcl).
            import signal

            def _postmortem_timeout(signum, frame):
                try:
                    sys.stdout.write(
                        '[alarm] post-mortem timeout, force-exit\n')
                    sys.stdout.flush()
                except Exception:
                    pass
                os._exit(1)

            for s in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(s, signal.SIG_DFL)
                except Exception:
                    pass
            try:
                signal.signal(signal.SIGALRM, _postmortem_timeout)
                signal.alarm(40)
            except Exception:
                pass
            # Use os.write (unbuffered) for trace before any Python I/O
            # which may interact with stale rcl state.
            os.write(2, f'[main] rcl context died ({e.__class__.__name__}) '
                        f'— saving CSV + running postmortem\n'.encode())
            # Save CSV FIRST — even if the rest of the function hangs we
            # have evidence on disk.
            try:
                node.save_log()
                os.write(2, f'[main] CSV saved to {node.log_path}\n'.encode())
            except Exception as se:
                os.write(2, f'[main] save_log error: {se}\n'.encode())
            # Run postmortem verdict with no wait.  PX4 may or may not
            # have disarmed yet — the outer shell script is responsible
            # for the final disarm check after we exit (it can re-read
            # /tmp/px4_sitl.log).  The sentinel file carries our partial
            # verdict for the shell script to grade.
            ok = _postmortem_verdict(node)
            exit_code = 0 if ok else 1
            os.write(2, f'[main] postmortem done ok={ok} '
                        f'exit_code={exit_code}\n'.encode())
        else:
            raise

    if postmortem:
        # rcl pybind11 state is corrupted; normal Python shutdown hangs on
        # interpreter finalizers touching the dead context.  Even os._exit
        # has been observed to wedge (presumably some atexit/finalizer runs
        # C code that touches the dead rcl context).  Flush output, then
        # SIGKILL ourselves via a forked child so we cannot be caught.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Write exit code to a sentinel file so the outer shell can grade.
        try:
            with open('/tmp/hover_test_postmortem_rc', 'w') as _rc:
                _rc.write(str(exit_code))
        except Exception:
            pass
        # Fork a killer: parent sleeps 1 s then SIGKILLs the main pid.
        import signal as _sig
        main_pid = os.getpid()
        killer_pid = os.fork()
        if killer_pid == 0:
            os.execvp('bash', ['bash', '-c',
                               f'sleep 1; kill -9 {main_pid}'])
        # Meanwhile we try os._exit — if it works, great; otherwise the
        # forked killer SIGKILLs us within 1 s.
        os._exit(exit_code)

    # Normal teardown (rcl still alive).
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass

    if helper_proc is not None:
        try:
            helper_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            helper_proc.terminate()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
