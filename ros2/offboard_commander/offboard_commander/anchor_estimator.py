#!/usr/bin/env python3
"""L2 anchor-path displacement estimator: IDLE -> LOCKED -> LOST -> (recapture).

/anchor/capture (std_srvs/Trigger) stores the next frame + baro h_ref + yaw_ref
+ t_ref. While LOCKED, each frame runs XFeat + RANSAC homography against the
anchor; H[0,2]/H[1,2] scaled by h_now/f_px give dx/dy in meters, atan2(H[1,0],
H[0,0]) gives dyaw. CSV is schema-parity with motion_delta{reference=ANCHOR}.
Torch/XFeat + cv2/numpy come from /home/teo/Drone/bench/.venv via sys.path.
"""

import csv, math, os, sys, time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/teo/Drone/bench/.venv/lib/python3.10/site-packages")
sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")
sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")

import cv2
import numpy as np
import rclpy
import torch
from gazebo_msgs.msg import ModelStates
from motion_delta_msgs.msg import MotionDelta
from px4_msgs.msg import VehicleAirData, VehicleAttitude, VehicleLocalPosition
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

MODEL_NAME = "iris_opt_flow"
LOSS_INLIERS, LOSS_REPROJ_PX, LOSS_STREAK = 30, 5.0, 3
COV_K = 10.0
IDLE, LOCKED, LOST = "IDLE", "LOCKED", "LOST"


def _px4_qos():
    q = QoSProfile(depth=5)
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.history = HistoryPolicy.KEEP_LAST
    q.durability = DurabilityPolicy.VOLATILE
    return q


def _yaw(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class AnchorEstimator(Node):
    def __init__(self):
        super().__init__("anchor_estimator")
        self.declare_parameter("image_topic", "/anchor_cam/image_raw")
        self.declare_parameter("camera_info_topic", "/anchor_cam/camera_info")
        self.declare_parameter("model_name", MODEL_NAME)
        image_topic = self.get_parameter("image_topic").value
        info_topic = self.get_parameter("camera_info_topic").value
        self.model_name = self.get_parameter("model_name").value

        self.f_px = self.baro_alt_m = self.yaw_now = None
        self.truth_xyz_yaw = None
        self.last_image_stamp_us = 0
        self.state = IDLE
        self.anchor_gray = self.h_ref = self.yaw_ref = self.truth_ref = None
        self.t_ref_us = self.anchor_id = self.loss_streak = self.frame_count = 0
        self.capture_pending = False

        self.get_logger().info("Loading XFeat on CPU...")
        self.xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat",
            pretrained=True, top_k=4096, trust_repo=True,
        )

        os.makedirs("/home/teo/Drone/logs", exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        self.csv_path = f"/home/teo/Drone/logs/anchor_estimator_{ts}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_file.write(
            f"# anchor_estimator | image={image_topic} info={info_topic} | "
            f"loss: inliers<{LOSS_INLIERS} or reproj>{LOSS_REPROJ_PX}px "
            f"x{LOSS_STREAK}\n"
        )
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow([
            "ts_us", "ts_ref_us", "state", "inliers", "total_matches",
            "reproj_err_px", "dx_m", "dy_m", "dyaw_rad",
            "cov_dx", "cov_dy", "cov_dyaw", "quality_u8",
            "baro_alt_m", "gt_dx_m", "gt_dy_m", "gt_dyaw_rad",
        ])

        cbi = MutuallyExclusiveCallbackGroup()
        cbs = MutuallyExclusiveCallbackGroup()
        cbv = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, image_topic, self.image_cb,
                                 qos_profile_sensor_data, callback_group=cbi)
        self.create_subscription(CameraInfo, info_topic, self.info_cb,
                                 qos_profile_sensor_data, callback_group=cbs)
        self.create_subscription(VehicleAirData, "/fmu/out/vehicle_air_data",
                                 self.air_cb, _px4_qos(), callback_group=cbs)
        # vehicle_air_data isn't in PX4's default DDS allowlist, so in SITL we
        # fall back to vehicle_local_position.z (EKF altitude, baro-primary via
        # EKF2_HGT_REF=2).  On hardware with a separate BMP388 we'd keep the
        # direct baro path; either source fills baro_alt_m.
        self.create_subscription(VehicleLocalPosition,
                                 "/fmu/out/vehicle_local_position",
                                 self.local_pos_cb, _px4_qos(), callback_group=cbs)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude",
                                 self.att_cb, _px4_qos(), callback_group=cbs)
        self.create_subscription(ModelStates, "/gazebo/model_states",
                                 self.model_cb, 10, callback_group=cbs)
        self.capture_srv = self.create_service(
            Trigger, "/anchor/capture", self.capture_cb, callback_group=cbv,
        )
        pub_qos = QoSProfile(depth=10)
        pub_qos.reliability = ReliabilityPolicy.RELIABLE
        pub_qos.history = HistoryPolicy.KEEP_LAST
        pub_qos.durability = DurabilityPolicy.VOLATILE
        self.motion_pub = self.create_publisher(
            MotionDelta, "/motion_delta", pub_qos,
        )
        self.get_logger().info(
            f"ready | state={self.state} | csv={self.csv_path}"
        )

    def info_cb(self, msg):
        if len(msg.k) >= 1 and float(msg.k[0]) > 0.0:
            self.f_px = float(msg.k[0])

    def air_cb(self, msg):
        self.baro_alt_m = float(msg.baro_alt_meter)

    def local_pos_cb(self, msg):
        if msg.z_valid and math.isfinite(float(msg.z)):
            self.baro_alt_m = float(-msg.z)

    def att_cb(self, msg):
        w, x, y, z = (float(msg.q[0]), float(msg.q[1]),
                      float(msg.q[2]), float(msg.q[3]))
        if math.isfinite(w):
            self.yaw_now = _yaw(w, x, y, z)

    def model_cb(self, msg):
        try:
            i = msg.name.index(self.model_name)
        except ValueError:
            return
        p, o = msg.pose[i].position, msg.pose[i].orientation
        self.truth_xyz_yaw = (float(p.x), float(p.y), float(p.z),
                              _yaw(float(o.w), float(o.x),
                                   float(o.y), float(o.z)))

    def capture_cb(self, _req, resp):
        missing = [n for n, v in (("baro", self.baro_alt_m),
                                  ("camera_info", self.f_px),
                                  ("image", self.last_image_stamp_us or None))
                   if v is None]
        if missing:
            resp.success, resp.message = False, f"Missing: {', '.join(missing)}"
            return resp
        self.capture_pending = True
        resp.success, resp.message = True, "Capture scheduled on next frame."
        return resp

    def image_cb(self, msg):
        ts_us = (msg.header.stamp.sec * 1_000_000
                 + msg.header.stamp.nanosec // 1000) or int(time.time() * 1e6)
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ("mono8", "8UC1"):
            gray = buf.reshape(msg.height, msg.width)
        elif msg.encoding == "rgb8":
            gray = cv2.cvtColor(buf.reshape(msg.height, msg.width, 3),
                                cv2.COLOR_RGB2GRAY)
        elif msg.encoding == "bgr8":
            gray = cv2.cvtColor(buf.reshape(msg.height, msg.width, 3),
                                cv2.COLOR_BGR2GRAY)
        else:
            self.get_logger().warn(f"unsupported encoding {msg.encoding}")
            return
        self.last_image_stamp_us = ts_us
        self.frame_count += 1

        if self.capture_pending:
            self._capture(gray, ts_us)
            self.capture_pending = False
            return
        if self.state != LOCKED:
            return
        self._match_and_emit(gray, ts_us)

    def _capture(self, gray, ts_us):
        self.anchor_gray = gray.copy()
        self.h_ref = float(self.baro_alt_m)
        self.yaw_ref = float(self.yaw_now) if self.yaw_now is not None else 0.0
        self.t_ref_us = ts_us
        self.truth_ref = self.truth_xyz_yaw
        self.anchor_id += 1
        self.loss_streak = 0
        self.state = LOCKED
        self.get_logger().info(
            f"ANCHOR {self.anchor_id} captured | h_ref={self.h_ref:.2f}m "
            f"yaw_ref={math.degrees(self.yaw_ref):+.1f}° -> LOCKED"
        )

    def _match_and_emit(self, gray, ts_us):
        h_now, f_px = float(self.baro_alt_m), float(self.f_px)
        inliers, total, reproj, H = self._xfeat_h(self.anchor_gray, gray)

        if H is not None and f_px > 0.0:
            dx_px, dy_px = float(H[0, 2]), float(H[1, 2])
            dyaw = math.atan2(float(H[1, 0]), float(H[0, 0]))
            s = h_now / f_px
            # anchor_cam SDF mount pose pitch=+π/2: image-v → body +X (confirmed
            # +X bench 2026-04-21), image-u → body +Y (sign verified by +Y bench).
            body_dx_live = dy_px * s
            body_dy_live = dx_px * s
            # Rotate live-body delta into anchor-body frame so dx/dy are
            # body-frame-flat aligned to anchor yaw (motion_delta contract).
            if self.yaw_now is not None and self.yaw_ref is not None:
                yaw_delta = _wrap(self.yaw_now - self.yaw_ref)
                c, sn = math.cos(yaw_delta), math.sin(yaw_delta)
                dx_m = c * body_dx_live - sn * body_dy_live
                dy_m = sn * body_dx_live + c * body_dy_live
            else:
                dx_m, dy_m = body_dx_live, body_dy_live
        else:
            dx_m = dy_m = dyaw = 0.0

        sigma2 = ((h_now / max(1.0, f_px)) ** 2
                  * (1.0 + COV_K / max(1, inliers)))
        quality = max(0, min(255, int(inliers)))

        if self.truth_ref and self.truth_xyz_yaw:
            gt_dx = self.truth_xyz_yaw[0] - self.truth_ref[0]
            gt_dy = self.truth_xyz_yaw[1] - self.truth_ref[1]
            gt_dyaw = _wrap(self.truth_xyz_yaw[3] - self.truth_ref[3])
        else:
            gt_dx = gt_dy = gt_dyaw = float("nan")

        self.csv.writerow([
            ts_us, self.t_ref_us, self.state,
            inliers, total, f"{reproj:.3f}",
            f"{dx_m:.4f}", f"{dy_m:.4f}", f"{dyaw:.4f}",
            f"{sigma2:.6f}", f"{sigma2:.6f}", f"{sigma2:.6f}", quality,
            f"{h_now:.3f}", f"{gt_dx:.4f}", f"{gt_dy:.4f}", f"{gt_dyaw:.4f}",
        ])

        degraded = (inliers < LOSS_INLIERS) or (reproj > LOSS_REPROJ_PX)
        if degraded:
            self.loss_streak += 1
            if self.loss_streak >= LOSS_STREAK:
                self.state = LOST
                self.get_logger().warn(
                    f"ANCHOR {self.anchor_id} LOST "
                    f"(in={inliers} rep={reproj:.2f}px)"
                )
        else:
            self.loss_streak = 0

        if self.state == LOST:
            lock_status = MotionDelta.LOCK_LOST
            pub_quality = 0
        elif degraded:
            lock_status = MotionDelta.LOCK_DEGRADED
            pub_quality = quality
        else:
            lock_status = MotionDelta.LOCK_LOCKED
            pub_quality = quality

        md = MotionDelta()
        now = self.get_clock().now().to_msg()
        md.header.stamp = now
        md.header.frame_id = "body_flat_anchor"
        md.timestamp_end = int(ts_us)
        md.timestamp_start = int(self.t_ref_us)
        md.reference = MotionDelta.REFERENCE_ANCHOR
        md.reference_id = int(self.anchor_id)
        md.dx = float(dx_m)
        md.dy = float(dy_m)
        md.dyaw = float(dyaw)
        md.cov = [float(sigma2), float(sigma2), float(sigma2)]
        md.quality = int(pub_quality)
        md.matched_features = int(min(inliers, 0xFFFF))
        md.lock_status = int(lock_status)
        self.motion_pub.publish(md)

        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f"a={self.anchor_id} {self.state} in={inliers}/{total} "
                f"rep={reproj:.2f}px d=({dx_m:+.2f},{dy_m:+.2f})m "
                f"yaw={math.degrees(dyaw):+.1f}° "
                f"gt=({gt_dx:+.2f},{gt_dy:+.2f})m h={h_now:.1f}m"
            )

    def _xfeat_h(self, a, b):
        a3 = np.stack([a, a, a], axis=-1) if a.ndim == 2 else a
        b3 = np.stack([b, b, b], axis=-1) if b.ndim == 2 else b
        with torch.inference_mode():
            mk0, mk1 = self.xfeat.match_xfeat(a3, b3, top_k=4096)
        total = int(len(mk0))
        if total < 8:
            return 0, total, float("inf"), None
        H, mask = cv2.findHomography(mk0, mk1, cv2.RANSAC, 3.0)
        if H is None or mask is None or mask.sum() == 0:
            return 0, total, float("inf"), None
        inl = int(mask.sum())
        idx = np.where(mask.ravel() > 0)[0]
        p0 = mk0[idx].reshape(-1, 1, 2).astype(np.float32)
        proj = cv2.perspectiveTransform(p0, H).reshape(-1, 2)
        err = float(np.linalg.norm(proj - mk1[idx], axis=1).mean())
        return inl, total, err, H

    def destroy_node(self):
        try:
            self.csv_file.flush(); self.csv_file.close()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = AnchorEstimator()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.get_logger().info(
                f"FINAL anchor_id={node.anchor_id} state={node.state} "
                f"frames={node.frame_count} csv={node.csv_path}"
            )
        except Exception:
            pass
        for fn in (node.destroy_node, rclpy.try_shutdown):
            try:
                fn()
            except Exception:
                pass


if __name__ == "__main__":
    main()
