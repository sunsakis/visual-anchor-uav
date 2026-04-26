#!/usr/bin/env python3
"""Demo #2 — motion_delta on a real-geometry aerial trajectory in Gazebo.

Reuses the sim_texture_probe harness: spawns anchor_cam at 1280x800 / 110°
HFOV over the donbas_rural z18 texture at 100 m AGL, captures an anchor
frame, then teleports the camera along a circular trajectory (r=30 m, 60
waypoints) and runs XFeat + RANSAC homography against the anchor at each
waypoint. Decomposes the homography to dx/dy/dyaw in meters, compares
against ground truth (commanded pose), renders a pitch-style overlay, and
stitches frames to /tmp/motion_delta_demo2.gif.

Output has: top row = anchor | live, bottom row = match lines + telemetry
panel with motion_delta schema values, ground-truth comparison, error, and
a trajectory minimap inset.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/usr/lib/python3/dist-packages")
sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")
sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")

import math
import signal
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

PROJ = Path("/home/teo/Drone")
GAZEBO_RES = PROJ / "gazebo_resources"
MODEL_SDF = GAZEBO_RES / "models/anchor_cam/anchor_cam.sdf"
WORLD = GAZEBO_RES / "worlds/anchor_bench.world"

# Camera: v0 BOM baseline — 1280x800, 110° HFOV, rectilinear pinhole
CAM_W, CAM_H, CAM_HFOV_DEG = 1280, 800, 110.0
ALT_M = 100.0
TRAJ_RADIUS_M = 30.0
TRAJ_N = 60                  # waypoints
TRAJ_FPS = 12
OUT_DIR = Path("/tmp/demo2_out")
OUT_GIF = Path("/tmp/motion_delta_demo2.gif")


def rewrite_model(w, h, hfov_deg):
    hfov = min(math.radians(hfov_deg), math.pi - 1e-3)
    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="anchor_cam">
    <static>true</static>
    <link name="link">
      <pose>0 0 0 0 0 0</pose>
      <sensor name="anchor_cam" type="camera">
        <pose>0 0 0 0 1.5707963 0</pose>
        <always_on>true</always_on>
        <update_rate>30</update_rate>
        <visualize>false</visualize>
        <camera>
          <horizontal_fov>{hfov:.7f}</horizontal_fov>
          <image>
            <width>{w}</width>
            <height>{h}</height>
            <format>L8</format>
          </image>
          <clip><near>0.1</near><far>3000</far></clip>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <robotNamespace></robotNamespace>
          <cameraName>anchor_cam</cameraName>
          <imageTopicName>image_raw</imageTopicName>
          <cameraInfoTopicName>camera_info</cameraInfoTopicName>
          <frameName>anchor_cam_link</frameName>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
"""
    MODEL_SDF.write_text(sdf)


def build_env():
    env = os.environ.copy()
    px4_models = ("/home/teo/Drone/PX4-Autopilot/Tools/simulation/"
                  "gazebo-classic/sitl_gazebo-classic/models")
    extra_res = [str(GAZEBO_RES), f"{GAZEBO_RES}/models", "/usr/share/gazebo-11"]
    env["GAZEBO_RESOURCE_PATH"] = (
        ":".join(extra_res) + ":" + env.get("GAZEBO_RESOURCE_PATH", "")
    )
    env["GAZEBO_MODEL_PATH"] = (
        f"{GAZEBO_RES}/models:{px4_models}:" + env.get("GAZEBO_MODEL_PATH", "")
    )
    return env


def start_gzserver(env):
    log = open("/tmp/demo2_gz.log", "a")
    log.write(f"\n===== {time.strftime('%H:%M:%S')} launching gzserver =====\n")
    log.flush()
    return subprocess.Popen(
        ["gzserver", "--verbose", str(WORLD)],
        env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
    )


def kill_gzserver(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def gz_set_pose(model, x, y, z, env):
    subprocess.run(
        ["gz", "model", "-m", model,
         "-x", str(x), "-y", str(y), "-z", str(z),
         "-R", "0", "-P", "0", "-Y", "0"],
        env=env, check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
    )


class ImageGrabber(Node):
    def __init__(self, topic):
        super().__init__("demo2_grabber")
        self._latest = None
        self._stamp = 0
        self.create_subscription(Image, topic, self._cb, qos_profile_sensor_data)

    def _cb(self, msg):
        h, w = msg.height, msg.width
        enc = msg.encoding
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if enc in ("mono8", "8UC1"):
            arr = buf.reshape(h, w)
        elif enc == "rgb8":
            arr = cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2GRAY)
        elif enc == "bgr8":
            arr = cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_BGR2GRAY)
        else:
            return
        self._latest = arr
        self._stamp += 1

    def grab_fresh(self, min_frames=5, timeout=6.0):
        start = self._stamp
        t0 = time.time()
        while (self._stamp - start < min_frames) and (time.time() - t0 < timeout):
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._latest


def match_xfeat(xfeat, a, b):
    a3 = np.stack([a, a, a], axis=-1) if a.ndim == 2 else a
    b3 = np.stack([b, b, b], axis=-1) if b.ndim == 2 else b
    mk0, mk1 = xfeat.match_xfeat(a3, b3, top_k=4096)
    total = int(len(mk0))
    if total < 8:
        return None, np.asarray(mk0), np.asarray(mk1), None
    H, mask = cv2.findHomography(mk0, mk1, cv2.RANSAC, 3.0)
    return H, np.asarray(mk0), np.asarray(mk1), mask


def decompose(H, W, Himg):
    c = np.array([W / 2.0, Himg / 2.0, 1.0])
    m = H @ c
    m = m[:2] / m[2]
    dx_px = float(m[0] - W / 2.0)
    dy_px = float(m[1] - Himg / 2.0)
    dyaw = float(math.atan2(H[1, 0], H[0, 0]))
    return dx_px, dy_px, dyaw


def put(img, txt, org, col=(255, 255, 255), scale=0.55, th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th,
                cv2.LINE_AA)


def draw_minimap(panel, traj_xy_truth, traj_xy_est, cur_idx,
                 origin, half_size, scale_m):
    """Draw a trajectory minimap inside the given panel rect.
    panel: (Hp, Wp, 3) uint8
    traj_xy_truth / traj_xy_est: list of (x_m, y_m)
    origin: (cx, cy) pixel center in panel
    half_size: panel half-size in px (square)
    scale_m: meters per panel-half (world half-extent)
    """
    cx, cy = origin
    # Grid + axes
    cv2.rectangle(panel, (cx - half_size, cy - half_size),
                  (cx + half_size, cy + half_size), (80, 80, 80), 1)
    cv2.line(panel, (cx - half_size, cy), (cx + half_size, cy), (60, 60, 60), 1)
    cv2.line(panel, (cx, cy - half_size), (cx, cy + half_size), (60, 60, 60), 1)

    def to_px(x_m, y_m):
        px = int(cx + (x_m / scale_m) * half_size)
        py = int(cy - (y_m / scale_m) * half_size)
        return px, py

    # Ground truth as thin cyan
    for i in range(1, len(traj_xy_truth)):
        p0 = to_px(*traj_xy_truth[i - 1])
        p1 = to_px(*traj_xy_truth[i])
        cv2.line(panel, p0, p1, (255, 200, 0), 1, cv2.LINE_AA)
    # Estimate trajectory
    for i in range(1, min(cur_idx + 1, len(traj_xy_est))):
        if traj_xy_est[i] is None or traj_xy_est[i - 1] is None:
            continue
        p0 = to_px(*traj_xy_est[i - 1])
        p1 = to_px(*traj_xy_est[i])
        cv2.line(panel, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)
    # Anchor marker
    ax, ay = to_px(0.0, 0.0)
    cv2.drawMarker(panel, (ax, ay), (0, 200, 255), cv2.MARKER_TRIANGLE_UP, 10, 2)
    # Current position dot
    if cur_idx < len(traj_xy_truth):
        tp = to_px(*traj_xy_truth[cur_idx])
        cv2.circle(panel, tp, 6, (255, 200, 0), 2)
        if cur_idx < len(traj_xy_est) and traj_xy_est[cur_idx] is not None:
            ep = to_px(*traj_xy_est[cur_idx])
            cv2.circle(panel, ep, 5, (0, 255, 0), -1)
    put(panel, "trajectory (top-down)", (cx - half_size, cy - half_size - 6),
        (200, 200, 200), 0.45)
    put(panel, f"+/- {int(scale_m)} m", (cx + half_size - 70, cy + half_size + 16),
        (150, 150, 150), 0.4)


def render_frame(anchor_bgr, live_bgr, H, mk0, mk1, mask,
                 focal_px, truth_xy, traj_truth, traj_est, cur_idx):
    # Downsize the capture frames for display (1280x800 -> 480x300 each)
    DISP_W, DISP_H = 480, 300
    a = cv2.resize(anchor_bgr, (DISP_W, DISP_H))
    l = cv2.resize(live_bgr, (DISP_W, DISP_H))
    top_l, top_r = a.copy(), l.copy()

    # Warped quad on live
    sx = DISP_W / CAM_W
    sy = DISP_H / CAM_H
    pad_src = 80
    rect_src = np.array([
        [pad_src, pad_src], [CAM_W - pad_src, pad_src],
        [CAM_W - pad_src, CAM_H - pad_src], [pad_src, CAM_H - pad_src],
    ], dtype=np.float32).reshape(-1, 1, 2)
    rect_disp_anchor = (rect_src.reshape(-1, 2) * np.array([sx, sy])).astype(np.int32)
    cv2.polylines(top_l, [rect_disp_anchor], True, (0, 255, 0), 2)
    if H is not None:
        warped = cv2.perspectiveTransform(rect_src, H).reshape(-1, 2)
        warped_disp = (warped * np.array([sx, sy])).astype(np.int32)
        cv2.polylines(top_r, [warped_disp], True, (0, 255, 0), 2)

    top = np.concatenate([top_l, top_r], axis=1)  # 960 x 300

    # Bottom row = match panel (640x360) + telemetry panel (320x360)
    match_panel = np.zeros((360, 640, 3), dtype=np.uint8)
    # Draw anchor and live side-by-side at 320x200 each inside 640x240 strip,
    # leaving 120 px below for a caption and space around lines.
    MW, MH = 320, 200
    a_small = cv2.resize(anchor_bgr, (MW, MH))
    l_small = cv2.resize(live_bgr, (MW, MH))
    match_panel[20:20 + MH, :MW] = a_small
    match_panel[20:20 + MH, MW:MW + MW] = l_small
    if H is not None and mask is not None and len(mk0) > 0:
        sx_m = MW / CAM_W
        sy_m = MH / CAM_H
        inlier_mask = mask.ravel().astype(bool)
        for p0, p1 in zip(mk0[inlier_mask][::3], mk1[inlier_mask][::3]):
            x0 = int(p0[0] * sx_m)
            y0 = int(p0[1] * sy_m) + 20
            x1 = int(p1[0] * sx_m) + MW
            y1 = int(p1[1] * sy_m) + 20
            cv2.line(match_panel, (x0, y0), (x1, y1), (0, 255, 0), 1,
                     cv2.LINE_AA)
    put(match_panel, "XFeat inlier matches (anchor  <->  live)",
        (8, 14), (0, 200, 255), 0.45)

    # Telemetry panel
    tel = np.zeros((360, 320, 3), dtype=np.uint8)
    if H is not None and mask is not None:
        inliers = int(mask.sum())
        total = int(len(mk0))
        dx_px, dy_px, dyaw = decompose(H, CAM_W, CAM_H)
        # Gazebo's down-pitched camera rotates image axes 90° vs world:
        # calibrated empirically — world +x (east) shows up as image +v,
        # world +y (north) shows up as image -u.
        dx_m = +dy_px * ALT_M / focal_px
        dy_m = +dx_px * ALT_M / focal_px
        cov_xy = (1.0 / max(inliers, 1)) ** 2
        quality = min(255, inliers * 2)
        if inliers >= 30:
            lock, lock_col = "LOCKED", (0, 255, 0)
        elif inliers >= 10:
            lock, lock_col = "DEGRADED", (0, 200, 255)
        else:
            lock, lock_col = "LOST", (0, 0, 255)
    else:
        inliers, total = 0, 0
        dx_m = dy_m = dyaw = 0.0
        cov_xy = 1.0
        quality = 0
        lock, lock_col = "LOST", (0, 0, 255)

    tx, ty = truth_xy
    err_x = dx_m - tx
    err_y = dy_m - ty
    err_r = math.hypot(err_x, err_y)

    y = 20
    put(tel, "motion_delta", (8, y), (255, 255, 255), 0.6); y += 22
    put(tel, "reference = ANCHOR", (8, y), (200, 200, 200), 0.5); y += 22
    put(tel, f"lock: {lock}", (8, y), lock_col, 0.55); y += 24
    put(tel, f"matches: {total}   inliers: {inliers}", (8, y)); y += 22
    put(tel, f"quality: {quality}/255", (8, y)); y += 26
    put(tel, f"dx  est: {dx_m:+7.2f} m", (8, y), (200, 255, 200), 0.55); y += 20
    put(tel, f"    gt:  {tx:+7.2f} m  err {err_x:+5.2f}", (8, y),
        (180, 180, 180), 0.48); y += 20
    put(tel, f"dy  est: {dy_m:+7.2f} m", (8, y), (200, 255, 200), 0.55); y += 20
    put(tel, f"    gt:  {ty:+7.2f} m  err {err_y:+5.2f}", (8, y),
        (180, 180, 180), 0.48); y += 20
    put(tel, f"dyaw: {math.degrees(dyaw):+5.2f} deg", (8, y)); y += 22
    put(tel, f"|err|: {err_r:5.2f} m", (8, y),
        (0, 255, 0) if err_r < 0.5 else (0, 200, 255), 0.6); y += 8

    # Minimap inset in tel bottom
    traj_est_pt = (dx_m, dy_m) if H is not None else None
    traj_est[cur_idx] = traj_est_pt
    draw_minimap(tel, traj_truth, traj_est,
                 cur_idx=cur_idx,
                 origin=(160, 290),
                 half_size=55,
                 scale_m=TRAJ_RADIUS_M * 1.3)

    bottom = np.concatenate([match_panel, tel], axis=1)  # 960 x 360
    out = np.concatenate([top, bottom], axis=0)  # 960 x 660

    put(out, f"altitude {ALT_M:.0f} m AGL  |  1280x800 @ {CAM_HFOV_DEG:.0f}° HFOV  |  "
             "texture: donbas_rural z18 (Ukraine aerial)",
        (8, out.shape[0] - 10), (160, 160, 160), 0.45)
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for p in OUT_DIR.glob("*.png"):
        p.unlink()

    print("Loading XFeat (cached)...", flush=True)
    xfeat = torch.hub.load(
        "verlab/accelerated_features", "XFeat",
        pretrained=True, top_k=4096, trust_repo=True,
    )

    rewrite_model(CAM_W, CAM_H, CAM_HFOV_DEG)
    env = build_env()
    subprocess.run(["pkill", "-9", "-f", "gzserver"],
                   check=False, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    time.sleep(1.0)

    sim = start_gzserver(env)
    time.sleep(12.0)
    rclpy.init()
    grabber = ImageGrabber("/anchor_cam/image_raw")
    focal_px = (CAM_W / 2.0) / math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
    print(f"focal_px = {focal_px:.1f}", flush=True)

    # Anchor capture at (0, 0, ALT)
    gz_set_pose("anchor_cam", 0.0, 0.0, ALT_M, env)
    time.sleep(1.0)
    anchor = grabber.grab_fresh(min_frames=8, timeout=8.0)
    if anchor is None:
        print("no anchor frame; aborting")
        kill_gzserver(sim); rclpy.shutdown(); return
    anchor_bgr = cv2.cvtColor(anchor, cv2.COLOR_GRAY2BGR)
    cache = Path("/tmp/demo2_cache")
    cache.mkdir(exist_ok=True)
    cv2.imwrite(str(cache / "anchor.png"), anchor_bgr)

    # Build trajectory: circle of radius TRAJ_RADIUS_M, TRAJ_N points
    traj_truth = [(TRAJ_RADIUS_M * math.cos(2 * math.pi * i / TRAJ_N),
                   TRAJ_RADIUS_M * math.sin(2 * math.pi * i / TRAJ_N))
                  for i in range(TRAJ_N)]
    traj_est = [None] * TRAJ_N

    # Warmup
    _ = match_xfeat(xfeat, anchor, anchor)

    for i, (tx, ty) in enumerate(traj_truth):
        gz_set_pose("anchor_cam", tx, ty, ALT_M, env)
        time.sleep(0.6)
        live = grabber.grab_fresh(min_frames=4, timeout=5.0)
        if live is None:
            print(f"  wp {i}: no live frame", flush=True)
            continue
        live_bgr = cv2.cvtColor(live, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(cache / f"live_{i:03d}.png"), live_bgr)
        H, mk0, mk1, mask = match_xfeat(xfeat, anchor, live)
        frame = render_frame(anchor_bgr, live_bgr, H, mk0, mk1, mask,
                             focal_px, (tx, ty), traj_truth, traj_est, i)
        cv2.imwrite(str(OUT_DIR / f"f_{i:03d}.png"), frame)
        inl = int(mask.sum()) if mask is not None else 0
        tot = int(len(mk0)) if mk0 is not None else 0
        dx_m, dy_m, _ = (0.0, 0.0, 0.0)
        if H is not None:
            dxp, dyp, _ = decompose(H, CAM_W, CAM_H)
            dx_m = +dyp * ALT_M / focal_px
            dy_m = +dxp * ALT_M / focal_px
        err = math.hypot(dx_m - tx, dy_m - ty) if H is not None else float('nan')
        print(f"  wp {i:2d}/{TRAJ_N}  truth=({tx:+5.1f},{ty:+5.1f})  "
              f"est=({dx_m:+5.1f},{dy_m:+5.1f})  err={err:.2f} m  "
              f"inl={inl}/{tot}",
              flush=True)

    try:
        grabber.destroy_node()
    except Exception:
        pass
    kill_gzserver(sim)
    rclpy.shutdown()

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(TRAJ_FPS),
         "-i", str(OUT_DIR / "f_%03d.png"),
         "-vf",
         "scale=720:-1:flags=lanczos,split[a][b];"
         "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
         str(OUT_GIF)],
        check=True,
    )
    print(f"wrote {OUT_GIF}")


if __name__ == "__main__":
    main()
