#!/usr/bin/env python3
"""Host-side shim: translate /motion_delta (ANCHOR path) into PX4's
external-vision odometry topic so the EKF can fuse it as position.

On first MotionDelta per reference_id, snapshot the current EKF local pos +
yaw as the anchor's world origin.  Each subsequent delta is rotated by the
anchor yaw and added to the origin to produce an NED xy position, published
on /fmu/in/vehicle_visual_odometry with RELIABLE QoS to match PX4's input
expectation.  Reset counter bumps on every new reference_id so EKF2 clears
its EV buffer when the anchor changes.

Scope pin (2026-04-21): only emits xy position fusion (cov[2] = large, z =
NaN).  Yaw / velocity / height are left to other sensors.
"""

import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")
sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")

import rclpy
from motion_delta_msgs.msg import MotionDelta
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition, VehicleOdometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)


def _yaw_from_q(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _px4_qos():
    q = QoSProfile(depth=5)
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.history = HistoryPolicy.KEEP_LAST
    q.durability = DurabilityPolicy.VOLATILE
    return q


def _reliable_qos():
    q = QoSProfile(depth=10)
    q.reliability = ReliabilityPolicy.RELIABLE
    q.history = HistoryPolicy.KEEP_LAST
    q.durability = DurabilityPolicy.VOLATILE
    return q


class AnchorEvShim(Node):
    def __init__(self):
        super().__init__("anchor_ev_shim")
        self.current_pos = None
        self.current_pos_valid = False
        self.current_yaw = None
        self.anchors = {}
        self.last_ref_id = None
        self.reset_counter = 0
        self.published_count = 0

        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self._pos_cb, _px4_qos(),
        )
        self.create_subscription(
            VehicleAttitude, "/fmu/out/vehicle_attitude",
            self._att_cb, _px4_qos(),
        )
        self.create_subscription(
            MotionDelta, "/motion_delta", self._motion_cb, _reliable_qos(),
        )
        self.ev_pub = self.create_publisher(
            VehicleOdometry, "/fmu/in/vehicle_visual_odometry", _reliable_qos(),
        )
        self.get_logger().info("anchor_ev_shim ready | waiting for /motion_delta")

    def _pos_cb(self, msg):
        self.current_pos_valid = bool(msg.xy_valid)
        self.current_pos = (float(msg.x), float(msg.y), float(msg.z))

    def _att_cb(self, msg):
        w, x, y, z = (float(msg.q[0]), float(msg.q[1]),
                      float(msg.q[2]), float(msg.q[3]))
        if math.isfinite(w):
            self.current_yaw = _yaw_from_q(w, x, y, z)

    def _motion_cb(self, md):
        if md.reference != MotionDelta.REFERENCE_ANCHOR:
            return
        if md.lock_status not in (MotionDelta.LOCK_LOCKED,
                                  MotionDelta.LOCK_DEGRADED):
            return
        ref_id = int(md.reference_id)
        if ref_id == 0:
            return

        if ref_id not in self.anchors:
            if (self.current_pos is None or self.current_yaw is None
                    or not self.current_pos_valid):
                return
            self.anchors[ref_id] = (*self.current_pos, float(self.current_yaw))
            if self.last_ref_id is not None and ref_id != self.last_ref_id:
                self.reset_counter = (self.reset_counter + 1) % 256
            self.last_ref_id = ref_id
            self.get_logger().info(
                f"anchor {ref_id} origin snapshot: "
                f"pos=({self.anchors[ref_id][0]:+.2f},"
                f"{self.anchors[ref_id][1]:+.2f},"
                f"{self.anchors[ref_id][2]:+.2f}) "
                f"yaw={math.degrees(self.anchors[ref_id][3]):+.1f}° "
                f"reset={self.reset_counter}"
            )

        ax, ay, _az, ayaw = self.anchors[ref_id]
        c, s = math.cos(ayaw), math.sin(ayaw)
        ned_x = ax + c * float(md.dx) - s * float(md.dy)
        ned_y = ay + s * float(md.dx) + c * float(md.dy)

        odom = VehicleOdometry()
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        odom.timestamp = now_us
        odom.timestamp_sample = int(md.timestamp_end)
        odom.pose_frame = VehicleOdometry.POSE_FRAME_NED
        odom.position = [float(ned_x), float(ned_y), float("nan")]
        odom.q = [float("nan"), 0.0, 0.0, 0.0]
        odom.velocity_frame = VehicleOdometry.VELOCITY_FRAME_UNKNOWN
        odom.velocity = [float("nan")] * 3
        odom.angular_velocity = [float("nan")] * 3
        pos_var = max(1e-4, float(md.cov[0]))
        odom.position_variance = [pos_var, pos_var, 1e6]
        odom.orientation_variance = [float("nan")] * 3
        odom.velocity_variance = [float("nan")] * 3
        odom.reset_counter = int(self.reset_counter)
        odom.quality = int(max(0, min(100, int(md.quality) * 100 // 255)))
        self.ev_pub.publish(odom)

        self.published_count += 1
        if self.published_count % 20 == 1:
            self.get_logger().info(
                f"EV ref={ref_id} ned=({ned_x:+.2f},{ned_y:+.2f}) "
                f"cov={pos_var:.4f} lock={md.lock_status} "
                f"n={self.published_count}"
            )


def main():
    rclpy.init()
    node = AnchorEvShim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for fn in (node.destroy_node, rclpy.try_shutdown):
            try:
                fn()
            except Exception:
                pass


if __name__ == "__main__":
    main()
