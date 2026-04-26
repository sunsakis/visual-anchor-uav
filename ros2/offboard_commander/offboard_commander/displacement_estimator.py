#!/usr/bin/env python3
"""Displacement estimator — rotation-compensated variant.

Consumes:
  - /px4flow/image_raw           (sensor_msgs/Image, mono8)
  - /gazebo/model_states         (height Z + ground-truth pose)
  - /fmu/out/vehicle_odometry    (body-frame omega + FRD->NED quaternion)

Pipeline:
  1. Shi-Tomasi + pyramidal LK → median pixel flow (dx_px, dy_px)
  2. Rotation-induced pixel flow (small-angle, down-looking FRD body):
         flow_rot_px_x ≈  focal_px · omega_y_body · dt   (pitch rate)
         flow_rot_px_y ≈ -focal_px · omega_x_body · dt   (roll rate)
     When compensate=True, subtract from measured flow BEFORE scaling.
  3. Residual pixels → metric body-frame delta:
         (dx_body, dy_body) = (res_px_x, res_px_y) · height / focal_px
  4. Yaw-align body-frame delta into world-level frame (compensate=True only):
         dx_w = cos(yaw)·dx_body - sin(yaw)·dy_body
         dy_w = sin(yaw)·dx_body + cos(yaw)·dy_body
     When compensate=False, accumulate body/image-frame directly (v0-minimal).
  5. Accumulate (x_est, y_est).

Mode selection (A/B):
  - Env DISP_COMPENSATE=1|true|yes|on enables compensation (default: 0).
  - ROS param `compensate` (bool) overrides env if explicitly set.
  - CSV suffix base|comp so analyze_displacement.py can pick both.

Out of scope (deferred):
  - Farnebäck dense fallback
  - Kalman / IMU fusion
  - Publishing to /fmu/in/sensor_optical_flow
"""

import csv
import math
import os
import sys
import time
from datetime import datetime

# Force system numpy 1.21.5 (cv2 ABI-compatible) ahead of user's 2.2.6.
sys.path.insert(0, "/usr/lib/python3/dist-packages")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from gazebo_msgs.msg import ModelStates  # noqa: E402
from geometry_msgs.msg import Point  # noqa: E402
from px4_msgs.msg import VehicleOdometry  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup  # noqa: E402
from rclpy.executors import (  # noqa: E402
    ExternalShutdownException, MultiThreadedExecutor,
)
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from sensor_msgs.msg import Image  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

FOCAL_PX = 254.66527570774434
MODEL_NAME = "iris_opt_flow"
MIN_FEATURES = 4
LK_WIN = (11, 11)
LK_PYR_LEVELS = 3
ST_MAX_CORNERS = 60
ST_QUALITY = 0.005
ST_MIN_DIST = 2
# RANSAC-inlier mean: seed with median, keep features within RES threshold
# pixels of seed, take mean of kept features. Fights the median sub-pixel
# skew bias that dominates drift at 64×64.
INLIER_RES_PX = 0.5


def _px4_qos() -> QoSProfile:
    q = QoSProfile(depth=5)
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.history = HistoryPolicy.KEEP_LAST
    q.durability = DurabilityPolicy.VOLATILE
    return q


class DisplacementEstimator(Node):
    def __init__(self):
        super().__init__("displacement_estimator")
        env_comp = os.environ.get("DISP_COMPENSATE", "0").strip().lower()
        env_bool = env_comp in ("1", "true", "yes", "on")
        self.declare_parameter("compensate", env_bool)
        self.compensate = bool(
            self.get_parameter("compensate").get_parameter_value().bool_value
        )
        mode_tag = "comp" if self.compensate else "base"

        self.prev_gray = None
        self.height_z = None
        self.truth_x = self.truth_y = None
        self.truth_x0 = self.truth_y0 = None
        self.x_est = 0.0
        self.y_est = 0.0
        self.path_truth = 0.0
        self.frame_count = 0
        self.dropped = 0
        self.t0 = None
        self.t_prev = None

        # Latest odometry — held as last-seen sample (no interpolation).
        self.omega_x = self.omega_y = self.omega_z = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.odom_recv_t = None  # monotonic seconds

        # RViz trace — grows each frame; published as MarkerArray at ~10 Hz.
        self.est_pts: list[tuple[float, float]] = []
        self.truth_pts: list[tuple[float, float]] = []

        os.makedirs("/home/teo/Drone/logs", exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        self.csv_path = f"/home/teo/Drone/logs/displacement_{mode_tag}_{ts}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_file.write(
            f"# displacement_estimator rot-comp | mode={mode_tag} "
            f"compensate={self.compensate} | focal_px={FOCAL_PX} | "
            f"WxH=64x64 | win={LK_WIN} levels={LK_PYR_LEVELS}\n"
        )
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow([
            "t", "dt", "height", "n_features",
            "dx_px", "dy_px", "dx_m", "dy_m",
            "x_est_cum", "y_est_cum", "x_truth", "y_truth",
            "roll", "pitch", "yaw",
            "omega_x", "omega_y", "omega_z",
            "flow_rot_px_x", "flow_rot_px_y",
            "dx_body_m", "dy_body_m", "odom_age_s",
            "dx_px_median", "dy_px_median", "n_inliers", "inlier_res_rms",
            "quality",
        ])

        # Isolate the heavy image callback (LK + Shi-Tomasi) from the two
        # lightweight state subs so LK work doesn't starve odom reception.
        self.cb_image = MutuallyExclusiveCallbackGroup()
        self.cb_state = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Image, "/px4flow/image_raw", self.image_cb, 10,
            callback_group=self.cb_image,
        )
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self.model_cb, 10,
            callback_group=self.cb_state,
        )
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry",
            self.odom_cb, _px4_qos(),
            callback_group=self.cb_state,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/displacement_trace", 5,
        )
        self.get_logger().info(
            f"compensate={self.compensate} | logging to {self.csv_path}"
        )

    def odom_cb(self, msg: VehicleOdometry):
        self.omega_x = float(msg.angular_velocity[0])
        self.omega_y = float(msg.angular_velocity[1])
        self.omega_z = float(msg.angular_velocity[2])
        w, x, y, z = (
            float(msg.q[0]), float(msg.q[1]),
            float(msg.q[2]), float(msg.q[3]),
        )
        if not math.isfinite(w):
            return
        # roll (X), pitch (Y), yaw (Z) — FRD body → NED world.
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            self.pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            self.pitch = math.asin(sinp)
        self.yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        self.odom_recv_t = time.monotonic()

    def model_cb(self, msg: ModelStates):
        try:
            idx = msg.name.index(MODEL_NAME)
        except ValueError:
            return
        p = msg.pose[idx].position
        self.height_z = float(p.z)
        if self.truth_x is not None:
            dx = p.x - self.truth_x
            dy = p.y - self.truth_y
            self.path_truth += math.hypot(dx, dy)
        self.truth_x = float(p.x)
        self.truth_y = float(p.y)
        if self.truth_x0 is None:
            self.truth_x0 = self.truth_x
            self.truth_y0 = self.truth_y
        # Truth trace — expressed relative to the first sample so it lines up
        # with the estimator's zeroed frame.
        self.truth_pts.append(
            (self.truth_x - self.truth_x0, self.truth_y - self.truth_y0)
        )
        if len(self.truth_pts) > 5000:
            self.truth_pts = self.truth_pts[-5000:]

    def image_cb(self, msg: Image):
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
            self.t_prev = now
        dt = now - self.t_prev
        self.t_prev = now
        t = now - self.t0

        if self.height_z is None or self.truth_x0 is None:
            return

        if msg.encoding != "mono8":
            self.get_logger().warn(f"unexpected encoding {msg.encoding}")
            return
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)

        if self.prev_gray is None:
            self.prev_gray = gray
            return

        corners = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=ST_MAX_CORNERS,
            qualityLevel=ST_QUALITY,
            minDistance=ST_MIN_DIST,
        )

        dx_px = dy_px = float("nan")
        dx_med = dy_med = float("nan")
        n_kept = 0
        n_inliers = 0
        inlier_res_rms = float("nan")
        if corners is not None and len(corners) >= MIN_FEATURES:
            nxt, status, _err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, corners, None,
                winSize=LK_WIN, maxLevel=LK_PYR_LEVELS,
            )
            if nxt is not None:
                st = status.ravel() == 1
                good_prev = corners[st]
                good_next = nxt[st]
                n_kept = len(good_prev)
                if n_kept >= MIN_FEATURES:
                    dxy = (good_next - good_prev).reshape(-1, 2)
                    # Seed a 2-DOF translation with the median (robust center),
                    # then keep features within INLIER_RES_PX of the seed and
                    # average them — mean is unbiased on symmetric sub-pixel
                    # noise once the outlier tail is trimmed.
                    dx_med = float(np.median(dxy[:, 0]))
                    dy_med = float(np.median(dxy[:, 1]))
                    res = np.hypot(dxy[:, 0] - dx_med, dxy[:, 1] - dy_med)
                    mask = res < INLIER_RES_PX
                    n_inliers = int(mask.sum())
                    if n_inliers >= MIN_FEATURES:
                        dx_px = float(dxy[mask, 0].mean())
                        dy_px = float(dxy[mask, 1].mean())
                        inlier_res_rms = float(np.sqrt((res[mask] ** 2).mean()))
                    else:
                        # Too few inliers — fall back to median so we don't
                        # drop the frame; n_inliers captures the degradation.
                        dx_px = dx_med
                        dy_px = dy_med

        if n_kept < MIN_FEATURES or not math.isfinite(dx_px):
            self.dropped += 1
            self.prev_gray = gray
            return

        # Quality scalar on PX4's 0-255 scale — the field the host EKF
        # consumes to downweight us when flow is unreliable.
        # Composed of two orthogonal axes: (a) what fraction of tracked
        # features agreed on one motion (rejects swirl/aperture failure),
        # (b) how tight that agreement was (rejects noisy tracks). The 0.5
        # px denominator matches INLIER_RES_PX — saturates at the rejection
        # threshold.
        frac_inlier = n_inliers / max(1, n_kept)
        res_rms = inlier_res_rms if math.isfinite(inlier_res_rms) else INLIER_RES_PX
        tightness = max(0.0, 1.0 - res_rms / INLIER_RES_PX)
        quality = int(round(255.0 * frac_inlier * tightness))

        # Rotation-induced pixel flow from body rates (small-angle).
        # Per prompt rule-of-thumb; signs post-verified from logged columns.
        flow_rot_px_x = FOCAL_PX * self.omega_y * dt
        flow_rot_px_y = -FOCAL_PX * self.omega_x * dt

        if self.compensate:
            res_px_x = dx_px - flow_rot_px_x
            res_px_y = dy_px - flow_rot_px_y
        else:
            res_px_x = dx_px
            res_px_y = dy_px

        scale = self.height_z / FOCAL_PX
        dx_body = res_px_x * scale
        dy_body = res_px_y * scale

        if self.compensate:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            dx_acc = c * dx_body - s * dy_body
            dy_acc = s * dx_body + c * dy_body
        else:
            # v0-minimal baseline — no frame rotation (regression anchor).
            dx_acc = dx_body
            dy_acc = dy_body

        self.x_est += dx_acc
        self.y_est += dy_acc

        odom_age = (now - self.odom_recv_t) if self.odom_recv_t else -1.0

        self.csv.writerow([
            f"{t:.4f}", f"{dt:.4f}", f"{self.height_z:.3f}", n_kept,
            f"{dx_px:.4f}", f"{dy_px:.4f}",
            f"{dx_acc:.4f}", f"{dy_acc:.4f}",
            f"{self.x_est:.4f}", f"{self.y_est:.4f}",
            f"{self.truth_x - self.truth_x0:.4f}",
            f"{self.truth_y - self.truth_y0:.4f}",
            f"{self.roll:.4f}", f"{self.pitch:.4f}", f"{self.yaw:.4f}",
            f"{self.omega_x:.4f}", f"{self.omega_y:.4f}", f"{self.omega_z:.4f}",
            f"{flow_rot_px_x:.4f}", f"{flow_rot_px_y:.4f}",
            f"{dx_body:.4f}", f"{dy_body:.4f}",
            f"{odom_age:.4f}",
            f"{dx_med:.4f}", f"{dy_med:.4f}", n_inliers,
            f"{inlier_res_rms:.4f}", quality,
        ])

        self.prev_gray = gray
        self.frame_count += 1

        self.est_pts.append((self.x_est, self.y_est))
        if len(self.est_pts) > 5000:
            self.est_pts = self.est_pts[-5000:]
        if self.frame_count % 17 == 0:
            self._publish_trace()

        if self.frame_count % 200 == 0:
            tag = "COMP" if self.compensate else "BASE"
            self.get_logger().info(
                f"[{tag}] t={t:.1f}s est=({self.x_est:+.2f},{self.y_est:+.2f}) "
                f"truth=({self.truth_x-self.truth_x0:+.2f},"
                f"{self.truth_y-self.truth_y0:+.2f}) "
                f"n={n_kept} in={n_inliers} q={quality} "
                f"dropped={self.dropped}"
            )

    def _publish_trace(self):
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        for ns, mid, pts, rgb in (
            ("estimate", 0, self.est_pts, (1.0, 0.2, 0.2)),
            ("truth", 1, self.truth_pts, (0.2, 1.0, 0.2)),
        ):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns = ns
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.08
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 1.0
            m.points = [Point(x=x, y=y, z=0.0) for x, y in pts]
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = DisplacementEstimator()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.truth_x is not None and node.truth_x0 is not None:
            tx = node.truth_x - node.truth_x0
            ty = node.truth_y - node.truth_y0
        else:
            tx = ty = 0.0
        err = math.hypot(node.x_est - tx, node.y_est - ty)
        mode = "comp" if node.compensate else "base"
        summary = (
            f"FINAL[{mode}] est=({node.x_est:+.3f},{node.y_est:+.3f}) "
            f"truth=({tx:+.3f},{ty:+.3f}) "
            f"err_xy={err:.3f} m | path_truth={node.path_truth:.3f} m | "
            f"frames={node.frame_count} dropped={node.dropped} | "
            f"csv={node.csv_path}"
        )
        try:
            node.get_logger().info(summary)
        except Exception:
            pass
        print(summary, file=sys.stderr, flush=True)
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
