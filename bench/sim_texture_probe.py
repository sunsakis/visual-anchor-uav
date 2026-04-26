#!/usr/bin/env python3
"""Layer 2 smoke test — can XFeat match anchor vs live frames over an
Eastern-Ukraine aerial ground texture across (res, HFOV) x altitude?

Sweeps 5 camera configs x 7 altitudes = 35 cells. Each cell teleports a
static downward-facing camera to (0,0,h) for anchor capture and to
(shift,0,h) for live capture, where shift = (1-overlap)*patch_width so
every cell has 50% ground overlap regardless of HFOV/altitude.

Texture: /home/teo/Drone/gazebo_resources/media/materials/textures/donbas_rural.jpg
World:   /home/teo/Drone/gazebo_resources/worlds/anchor_bench.world
Model:   /home/teo/Drone/gazebo_resources/models/anchor_cam/

Output:  /home/teo/Drone/logs/sim_texture_probe_<ts>.csv
Pass per cell: inliers >= 100 AND reproj <= 3 px.

Invocation: /home/teo/Drone/bench/.venv/bin/python3 sim_texture_probe.py
(ROS2 humble paths are inserted into sys.path at startup so rclpy + cv_bridge
work from the bench venv without sourcing setup.bash.)
"""
import os
import sys

# Force XFeat to CPU: GTX 970 (sm_52) is too old for torch 2.11 kernels
# (needs sm_75+). Setting this BEFORE torch import hides the GPU entirely.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# cv_bridge/numpy ABI fix + ROS2 humble rclpy path — must come before cv2/numpy/rclpy.
sys.path.insert(0, "/usr/lib/python3/dist-packages")
sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")
sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")

import csv
import math
import signal
import subprocess
import time
from pathlib import Path

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import torch  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402


PROJ = Path("/home/teo/Drone")
GAZEBO_RES = PROJ / "gazebo_resources"
MODEL_SDF = GAZEBO_RES / "models/anchor_cam/anchor_cam.sdf"
WORLD = GAZEBO_RES / "worlds/anchor_bench.world"
LOGS = PROJ / "logs"

# (name, width, height, hfov_deg, lens_type, undist_out_hfov_deg)
#   lens_type=None         -> pinhole (rectilinear); undist_out_hfov_deg must be None
#   lens_type="equidistant"-> Gazebo Classic fisheye via <lens> with cutoff=π
#   undist_out_hfov_deg    -> if set, remap the fisheye image to a virtual
#                             rectilinear pinhole at that HFOV before matching;
#                             shift/patch math then uses this HFOV.
# For raw fisheye with hfov >= 120° the 2h·tan(hfov/2) patch formula diverges at
# the rim, so we clamp the "useful" hfov to 120° for shift/gsd below.
CONFIGS = [
    ("baseline_1280x800_90",        1280, 800,  90.0,  None,          None),
    ("arducam_1280x800_83",         1280, 800,  83.0,  None,          None),
    ("arducam_1280x800_110",        1280, 800, 110.0,  None,          None),
    ("low_res_640x400_90",           640, 400,  90.0,  None,          None),
    ("high_res_1920x1200_90",       1920, 1200, 90.0,  None,          None),
    ("narrow_fov_1280x800_60",      1280, 800,  60.0,  None,          None),
    ("wide_fov_1280x800_120",       1280, 800, 120.0,  None,          None),
    ("fisheye_1280x800_180",        1280, 800, 180.0, "equidistant",  None),
    ("fisheye_180_undist_90",       1280, 800, 180.0, "equidistant",  90.0),
    ("fisheye_180_undist_120",      1280, 800, 180.0, "equidistant", 120.0),
]
USABLE_HFOV_CAP_DEG = 120.0  # clamp for patch/shift when fisheye
ALT_M = [1.0, 3.0, 10.0, 30.0, 50.0, 100.0, 200.0]
OVERLAP_FRAC = 0.5  # shift so live frame has 50% ground overlap with anchor

INLIERS_PASS = 100
REPROJ_PASS_PX = 3.0


def rewrite_model(w, h, hfov_deg, lens_type=None):
    # Gazebo Classic: <horizontal_fov> has a hard ceiling of π rad (180°);
    # fisheye goes through <lens> with a cutoff_angle that can equal π.
    hfov = min(math.radians(hfov_deg), math.pi - 1e-3)
    lens_block = ""
    if lens_type == "equidistant":
        lens_block = """
          <lens>
            <type>equidistant</type>
            <scale_to_hfov>true</scale_to_hfov>
            <cutoff_angle>3.1415</cutoff_angle>
            <env_texture_size>512</env_texture_size>
          </lens>"""
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
          <clip><near>0.1</near><far>3000</far></clip>{lens_block}
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
    # Assumes the invoking shell sourced /opt/ros/humble/setup.bash and PX4's
    # setup_gazebo.bash (see run_sim_probe.sh). We just append our own paths.
    env = os.environ.copy()
    px4_models = ("/home/teo/Drone/PX4-Autopilot/Tools/simulation/"
                  "gazebo-classic/sitl_gazebo-classic/models")
    extra_res = [
        str(GAZEBO_RES),
        f"{GAZEBO_RES}/models",
        "/usr/share/gazebo-11",
    ]
    env["GAZEBO_RESOURCE_PATH"] = (
        ":".join(extra_res) + ":" + env.get("GAZEBO_RESOURCE_PATH", "")
    )
    env["GAZEBO_MODEL_PATH"] = (
        f"{GAZEBO_RES}/models:{px4_models}:" + env.get("GAZEBO_MODEL_PATH", "")
    )
    # DO NOT unset DISPLAY — gazebo-classic's OGRE camera renderer needs a GL
    # context, and no X server = "Unable to create CameraSensor. Rendering is
    # disabled." Keep the user's DISPLAY; no gzclient is spawned anyway.
    return env


def start_gzserver(env):
    # Drop stdout/stderr to a file so we can post-mortem on failure.
    log = open("/tmp/sim_texture_probe_gz.log", "a")
    log.write(f"\n===== {time.strftime('%H:%M:%S')} launching gzserver =====\n")
    log.flush()
    return subprocess.Popen(
        ["gzserver", "--verbose", str(WORLD)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
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
        env=env,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )


class ImageGrabber(Node):
    def __init__(self, topic):
        super().__init__("anchor_probe_grabber")
        self._latest = None
        self._stamp = 0
        # gazebo_ros_camera publishes sensor data on a BestEffort profile;
        # a default RELIABLE subscriber receives no frames from it.
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

    def grab_fresh(self, min_frames=3, timeout=8.0):
        start = self._stamp
        t0 = time.time()
        while (self._stamp - start < min_frames) and (time.time() - t0 < timeout):
            rclpy.spin_once(self, timeout_sec=0.1)
        got = self._stamp - start
        if got < min_frames:
            print(f"    grab timeout: got {got}/{min_frames} frames in {timeout}s",
                  flush=True)
        return self._latest if got > 0 else None


def build_fisheye_undist_maps(w_in, h_in, w_out, h_out, out_hfov_deg,
                              fisheye_cutoff_rad=math.pi):
    """Remap table: Gazebo equidistant fisheye (r = f_in*theta, f_in=w_in/cutoff)
    -> rectilinear virtual pinhole at out_hfov_deg. Output pixels outside the
    cutoff angle are marked invalid (remap fill is 0)."""
    f_in = w_in / fisheye_cutoff_rad
    cx_in, cy_in = (w_in - 1) / 2.0, (h_in - 1) / 2.0
    f_out = (w_out / 2.0) / math.tan(math.radians(out_hfov_deg) / 2.0)
    cx_out, cy_out = (w_out - 1) / 2.0, (h_out - 1) / 2.0
    u, v = np.meshgrid(np.arange(w_out, dtype=np.float32),
                       np.arange(h_out, dtype=np.float32))
    x = (u - cx_out) / f_out
    y = (v - cy_out) / f_out
    rr = np.sqrt(x * x + y * y)
    theta = np.arctan(rr)  # pinhole ray -> angle from optical axis
    # Pixels past the fisheye cutoff are unobservable; mark them far off-image.
    invalid = theta >= (fisheye_cutoff_rad / 2.0)
    r_in = f_in * theta
    # phi = atan2(y, x); cos(phi)=x/rr, sin(phi)=y/rr (with rr=0 -> center)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_phi = np.where(rr > 1e-9, x / rr, 0.0)
        sin_phi = np.where(rr > 1e-9, y / rr, 0.0)
    map_x = (cx_in + r_in * cos_phi).astype(np.float32)
    map_y = (cy_in + r_in * sin_phi).astype(np.float32)
    map_x[invalid] = -1.0
    map_y[invalid] = -1.0
    return map_x, map_y


def match_xfeat(xfeat, a, b):
    a3 = np.stack([a, a, a], axis=-1) if a.ndim == 2 else a
    b3 = np.stack([b, b, b], axis=-1) if b.ndim == 2 else b
    mk0, mk1 = xfeat.match_xfeat(a3, b3, top_k=4096)
    total = int(len(mk0))
    if total < 8:
        return 0, total, float("inf")
    H, mask = cv2.findHomography(mk0, mk1, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return 0, total, float("inf")
    inl = int(mask.sum())
    if inl == 0:
        return 0, total, float("inf")
    idx = np.where(mask.ravel() > 0)[0]
    p0 = mk0[idx].reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(p0, H).reshape(-1, 2)
    err = float(np.linalg.norm(proj - mk1[idx], axis=1).mean())
    return inl, total, err


def main():
    LOGS.mkdir(exist_ok=True)
    # CLI: `--only <cfg_name>` to run one config; `--alts 60,80,100,...` to
    # override ALT_M. Used for targeted altitude-ceiling sweeps after the
    # broad 6x7 pass is done.
    only_cfg = None
    alts_override = None
    args = list(sys.argv[1:])
    while args:
        a = args.pop(0)
        if a == "--only":
            only_cfg = args.pop(0)
        elif a == "--alts":
            alts_override = [float(x) for x in args.pop(0).split(",")]
    global ALT_M
    if alts_override is not None:
        ALT_M = alts_override
    configs = [c for c in CONFIGS if (only_cfg is None or c[0] == only_cfg)]
    tag = f"_{only_cfg}" if only_cfg else ""
    out_path = LOGS / f"sim_texture_probe{tag}_{int(time.time())}.csv"

    print("Loading XFeat (cached)...", flush=True)
    xfeat = torch.hub.load(
        "verlab/accelerated_features", "XFeat",
        pretrained=True, top_k=4096, trust_repo=True,
    )

    env = build_env()
    rclpy.init()
    rows = []

    for cfg_name, w, h, hfov, lens_type, undist_hfov in configs:
        tag = f"{w}x{h}, HFOV={hfov}°"
        if lens_type:
            tag += f", lens={lens_type}"
        if undist_hfov is not None:
            tag += f", undist→{undist_hfov}°"
        print(f"\n=== config: {cfg_name} ({tag}) ===", flush=True)
        rewrite_model(w, h, hfov, lens_type=lens_type)
        # When we undistort fisheye to a virtual pinhole, geometry is governed
        # by the output HFOV, not the 180° capture. For raw fisheye (no undist)
        # we clamp the useful HFOV to 120° so 2h·tan(hfov/2) doesn't diverge.
        if undist_hfov is not None:
            hfov_for_geom = undist_hfov
        elif lens_type:
            hfov_for_geom = min(hfov, USABLE_HFOV_CAP_DEG)
        else:
            hfov_for_geom = hfov
        # Build remap LUT once per config if undistortion is active. Output
        # image keeps the same resolution as the capture (1280×800).
        remap_xy = None
        if undist_hfov is not None:
            remap_xy = build_fisheye_undist_maps(w, h, w, h, undist_hfov)

        subprocess.run(["pkill", "-9", "-f", "gzserver"],
                       check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        sim = start_gzserver(env)
        time.sleep(12.0)  # camera plugin + ROS bridge advertise takes time

        grabber = ImageGrabber("/anchor_cam/image_raw")

        for alt in ALT_M:
            patch_w = 2.0 * alt * math.tan(math.radians(hfov_for_geom) / 2.0)
            shift = (1.0 - OVERLAP_FRAC) * patch_w
            print(f"  alt={alt:>5.1f} m  patch={patch_w:>6.2f} m  shift={shift:>6.2f} m  "
                  f"gsd={patch_w / w * 1000:.2f} mm/px", flush=True)

            gz_set_pose("anchor_cam", 0.0, 0.0, alt, env)
            time.sleep(0.8)
            anchor = grabber.grab_fresh(min_frames=5, timeout=6.0)
            if anchor is None:
                rows.append(dict(cfg=cfg_name, w=w, h=h, hfov_deg=hfov, alt_m=alt,
                                 shift_m=round(shift, 3), gsd_mm_px=round(patch_w / w * 1000, 3),
                                 inliers=0, matches=0, reproj_px=float("inf"),
                                 xfeat_ms=0.0, note="no_anchor_frame"))
                print("    no anchor frame; skipping")
                continue
            if remap_xy is not None:
                anchor = cv2.remap(anchor, remap_xy[0], remap_xy[1],
                                   cv2.INTER_LINEAR, borderValue=0)

            gz_set_pose("anchor_cam", shift, 0.0, alt, env)
            time.sleep(0.8)
            live = grabber.grab_fresh(min_frames=5, timeout=6.0)
            if live is None:
                rows.append(dict(cfg=cfg_name, w=w, h=h, hfov_deg=hfov, alt_m=alt,
                                 shift_m=round(shift, 3), gsd_mm_px=round(patch_w / w * 1000, 3),
                                 inliers=0, matches=0, reproj_px=float("inf"),
                                 xfeat_ms=0.0, note="no_live_frame"))
                print("    no live frame; skipping")
                continue
            if remap_xy is not None:
                live = cv2.remap(live, remap_xy[0], remap_xy[1],
                                 cv2.INTER_LINEAR, borderValue=0)

            t0 = time.perf_counter()
            inl, tot, err = match_xfeat(xfeat, anchor, live)
            ms = (time.perf_counter() - t0) * 1000.0
            ok = inl >= INLIERS_PASS and err <= REPROJ_PASS_PX
            print(f"    inliers={inl:>4d}  matches={tot:>5d}  reproj={err:>7.2f} px  "
                  f"xfeat={ms:>6.1f} ms  [{'PASS' if ok else 'FAIL'}]", flush=True)
            rows.append(dict(
                cfg=cfg_name, w=w, h=h, hfov_deg=hfov, alt_m=alt,
                shift_m=round(shift, 3), gsd_mm_px=round(patch_w / w * 1000, 3),
                inliers=inl, matches=tot,
                reproj_px=(round(err, 3) if math.isfinite(err) else None),
                xfeat_ms=round(ms, 1), note=("pass" if ok else "fail"),
            ))

        try:
            grabber.destroy_node()
        except Exception:
            pass
        kill_gzserver(sim)
        time.sleep(1.5)

    rclpy.shutdown()

    # CSV
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")

    # Summary
    print("\n=== per-config pass rate ===")
    for cfg_name, *_ in configs:
        cell = [r for r in rows if r["cfg"] == cfg_name]
        p = sum(1 for r in cell if r["note"] == "pass")
        print(f"  {cfg_name}: {p}/{len(cell)} altitudes PASS  "
              f"inl={[r['inliers'] for r in cell]}")


if __name__ == "__main__":
    main()
