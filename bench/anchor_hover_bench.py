#!/usr/bin/env python3
"""L2 Task 4 + Task 5 anchor-path bench harness.

Teleport-based validation of anchor_estimator on anchor_bench.world:
 - gzserver with donbas_rural texture + gazebo_ros_state (ground truth) plugin
 - static anchor_cam rig at Arducam production specs (110° HFOV, 1280x800 L8)
 - fake /fmu/out/vehicle_air_data + /fmu/out/vehicle_attitude (50 Hz)
 - anchor_estimator subprocess matching camera frames against captured anchor

Protocol:
  Task 4: teleport to (0,0,ALT), /anchor/capture, 60 s static hover
          PASS: RMS(dx_m, dy_m) < 0.3 m over hover window
  Task 5a: teleport to (+10,0,ALT), hold 10 s, return to (0,0,ALT), hold 10 s
  Task 5b: teleport to (0,+10,ALT), hold 10 s, return to (0,0,ALT), hold 10 s
          PASS (per shifted phase):  target axis within ±0.05 m, off-axis <0.1 m
          PASS (per returned phase): |dx|,|dy| < 0.1 m
          LOCKED throughout all phases

ALT defaults to 100 m (v0 product target; bench L2 1c validated 110° floor is
20 m on z18, ceiling plane-limited). The 3 m spec in the L2 prompt predates
the 2026-04-21 high-altitude pivot.

CAMERA FRAME: post-2026-04-22 Task 1 fix, anchor_estimator emits body-frame-flat
meters (image-v→body+X, image-u→body+Y, yaw-corrected to anchor frame).
Bench drone yaw is always 0, so body-frame-flat ≡ world frame and the axis
gates below are meaningful. Sign of image-u→body+Y is verified by Task 5b.
"""

import csv
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# ROS2 paths must precede numpy so system packages resolve first.
sys.path.insert(0, "/usr/lib/python3/dist-packages")
sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")
sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")

import rclpy  # noqa: E402
from px4_msgs.msg import VehicleAirData, VehicleAttitude  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,  # noqa: E402
                       ReliabilityPolicy)


PROJ = Path("/home/teo/Drone")
GAZEBO_RES = PROJ / "gazebo_resources"
WORLD = GAZEBO_RES / "worlds/anchor_bench.world"
LOGS = PROJ / "logs"

ANCHOR_MODEL = "anchor_cam"
ALT_M = 100.0
TASK4_HOVER_S = 60.0
TASK5_SHIFT_M = 10.0
TASK5_DWELL_S = 10.0

RMS_PASS_M = 0.3


class BaroNoiseModel:
    """Realistic BMP388 + environment baro-altitude noise for hardware-parity runs.

    Three layers, composed additively on top of ground-truth z:
      bias   — fixed per-session offset (σ_bias ≈ 0.10 m, residual AGL cal error)
      drift  — AR(1) thermal / atmospheric wander (σ_ss ≈ 0.20 m, τ ≈ 30 s)
      white  — per-sample zero-mean Gaussian (σ ≈ 0.05 m after modest smoothing)

    Defaults are conservative-nominal for a calibrated BMP388 on a hovering
    quadcopter in still air: datasheet 0.08 Pa noise @ 0.66 Hz (ultra-high-res)
    → ~7 mm RMS per sample; field-typical drift from prop wash and warm-up pushes
    the effective RMS up. Tune via constructor kwargs if modeling harsher cases.

    Propagation into anchor estimator:
      dx_m = dx_px · (h_now / f_px)   (anchor_estimator.py:217)
      ⇒ scale error is strictly linear in h_now noise.
      Static hover (dx_true=0): insensitive to baro (0 × any_scale = 0).
      Shifted phase: mean-error ≈ dx_true · bias/h_AGL;
                     per-sample σ ≈ dx_true · √(σ_drift² + σ_white²)/h_AGL.
    """
    def __init__(self, seed=0, sigma_white=0.05, sigma_drift=0.20, tau_s=30.0,
                 sigma_bias=0.10):
        self.rng = random.Random(seed)
        self.sigma_white = sigma_white
        self.sigma_drift = sigma_drift
        self.tau_s = tau_s
        self.sigma_bias = sigma_bias
        self.bias = self.rng.gauss(0.0, sigma_bias)
        self.drift = 0.0
        self.t_last = None

    def sample(self, h_true, t_s):
        if self.t_last is None:
            self.t_last = t_s
        dt = max(1e-3, t_s - self.t_last)
        self.t_last = t_s
        alpha = math.exp(-dt / self.tau_s)
        innov_sigma = self.sigma_drift * math.sqrt(max(0.0, 1.0 - alpha * alpha))
        self.drift = alpha * self.drift + self.rng.gauss(0.0, innov_sigma)
        white = self.rng.gauss(0.0, self.sigma_white)
        return h_true + self.bias + self.drift + white

    def describe(self):
        return (f"bias_realized={self.bias:+.3f}m  σ_white={self.sigma_white}m  "
                f"σ_drift={self.sigma_drift}m  τ={self.tau_s}s  "
                f"σ_bias={self.sigma_bias}m")


def build_env():
    env = os.environ.copy()
    px4_models = (
        "/home/teo/Drone/PX4-Autopilot/Tools/simulation/"
        "gazebo-classic/sitl_gazebo-classic/models"
    )
    extra_res = [str(GAZEBO_RES), f"{GAZEBO_RES}/models", "/usr/share/gazebo-11"]
    env["GAZEBO_RESOURCE_PATH"] = (
        ":".join(extra_res) + ":" + env.get("GAZEBO_RESOURCE_PATH", "")
    )
    env["GAZEBO_MODEL_PATH"] = (
        f"{GAZEBO_RES}/models:{px4_models}:" + env.get("GAZEBO_MODEL_PATH", "")
    )
    return env


def start_gzserver(env, log_path):
    log = open(log_path, "w")
    log.write(f"===== {time.strftime('%H:%M:%S')} gzserver launch =====\n")
    log.flush()
    return subprocess.Popen(
        ["gzserver", "--verbose", str(WORLD)],
        env=env, stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def start_estimator(env, log_path):
    log = open(log_path, "w")
    log.write(f"===== {time.strftime('%H:%M:%S')} anchor_estimator launch =====\n")
    log.flush()
    return subprocess.Popen(
        ["ros2", "run", "offboard_commander", "anchor_estimator",
         "--ros-args", "-p", f"model_name:={ANCHOR_MODEL}"],
        env=env, stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def gz_teleport(env, x, y, z):
    return subprocess.run(
        ["gz", "model", "-m", ANCHOR_MODEL,
         "-x", f"{x}", "-y", f"{y}", "-z", f"{z}",
         "-R", "0", "-P", "0", "-Y", "0"],
        env=env, timeout=5,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def call_capture(env):
    r = subprocess.run(
        ["ros2", "service", "call", "/anchor/capture",
         "std_srvs/srv/Trigger", "{}"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    return "success=True" in r.stdout, r.stdout + r.stderr


def kill_group(proc):
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


def px4_qos():
    q = QoSProfile(depth=5)
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.history = HistoryPolicy.KEEP_LAST
    q.durability = DurabilityPolicy.VOLATILE
    return q


class FakePx4(Node):
    """Fills in /fmu/out/vehicle_air_data + /fmu/out/vehicle_attitude at 50 Hz
    so anchor_estimator has baro altitude (drives metric scale) and yaw (unused
    in this frame, identity quaternion) without PX4 running.

    If `baro_noise` is provided, ground-truth z is passed through it before
    publishing — simulates real hardware baro. Samples (t, h_true, h_meas) are
    streamed to `baro_log_path` if given; running |err| stats exposed via
    `baro_stats()`.
    """

    def __init__(self, z_provider, baro_noise=None, baro_log_path=None):
        super().__init__("bench_fake_px4")
        self.z_provider = z_provider
        self.baro_noise = baro_noise
        self.baro_log = open(baro_log_path, "w") if baro_log_path else None
        if self.baro_log:
            self.baro_log.write("t_s,h_true,h_meas,err_m\n")
        self.h_err_sq_sum = 0.0
        self.h_err_abs_sum = 0.0
        self.h_err_max = 0.0
        self.h_err_count = 0
        qos = px4_qos()
        self.pub_air = self.create_publisher(
            VehicleAirData, "/fmu/out/vehicle_air_data", qos)
        self.pub_att = self.create_publisher(
            VehicleAttitude, "/fmu/out/vehicle_attitude", qos)
        self.timer = self.create_timer(0.02, self._tick)

    def _tick(self):
        now_us = int(time.time() * 1e6)
        t_s = now_us / 1e6
        h_true = float(self.z_provider())
        h_meas = (self.baro_noise.sample(h_true, t_s)
                  if self.baro_noise is not None else h_true)
        err = h_meas - h_true
        self.h_err_sq_sum += err * err
        self.h_err_abs_sum += abs(err)
        self.h_err_max = max(self.h_err_max, abs(err))
        self.h_err_count += 1
        if self.baro_log:
            self.baro_log.write(
                f"{t_s:.6f},{h_true:.4f},{h_meas:.4f},{err:+.4f}\n"
            )

        air = VehicleAirData()
        air.timestamp = now_us
        air.timestamp_sample = now_us
        air.baro_alt_meter = h_meas
        self.pub_air.publish(air)

        att = VehicleAttitude()
        att.timestamp = now_us
        att.timestamp_sample = now_us
        att.q = [1.0, 0.0, 0.0, 0.0]
        self.pub_att.publish(att)

    def baro_stats(self):
        if self.h_err_count == 0:
            return None
        rms = math.sqrt(self.h_err_sq_sum / self.h_err_count)
        mean_abs = self.h_err_abs_sum / self.h_err_count
        return {"rms_m": rms, "mean_abs_m": mean_abs,
                "max_abs_m": self.h_err_max, "samples": self.h_err_count}

    def close_baro_log(self):
        try:
            if self.baro_log:
                self.baro_log.flush()
                self.baro_log.close()
        except Exception:
            pass


def find_csv_in_log(log_path):
    try:
        for line in Path(log_path).read_text().splitlines():
            if "csv=" in line:
                return line.split("csv=")[-1].strip().rstrip()
    except Exception:
        pass
    return None


SHIFT_TARGET_TOL_M = 0.05   # shifted-axis tolerance around +SHIFT
SHIFT_OFFAXIS_TOL_M = 0.10  # off-axis tolerance when shifted
RETURN_TOL_M = 0.10         # tolerance on each axis after return to origin
PHASE_SETTLE_FRAC = 0.5     # ignore first half of phase for numeric gates


def analyze(csv_path, task4_hover_s=TASK4_HOVER_S,
            task5_dwell_s=TASK5_DWELL_S):
    """Window the CSV by SIM time (ts_us from image headers).
    The first LOCKED row marks capture. Phase boundaries follow the protocol:
      hover:       [t0,                     t0 + hover_s)
      shifted_x:   [t0 + hover_s,           + 1*dwell_s)
      returned_x:  [+ 1*dwell_s,            + 2*dwell_s)
      shifted_y:   [+ 2*dwell_s,            + 3*dwell_s)
      returned_y:  [+ 3*dwell_s,            end]
    """
    import math
    if csv_path is None or not Path(csv_path).exists():
        print(f"[analyze] no CSV at {csv_path}")
        return None
    rows = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            break
        reader = csv.DictReader([line] + list(f))
        for r in reader:
            try:
                rows.append({
                    "ts_us": int(r["ts_us"]),
                    "state": r["state"],
                    "inliers": int(r["inliers"]),
                    "reproj": float(r["reproj_err_px"]),
                    "dx_m": float(r["dx_m"]),
                    "dy_m": float(r["dy_m"]),
                    "dyaw": float(r["dyaw_rad"]),
                    "gt_dx": float(r["gt_dx_m"]),
                    "gt_dy": float(r["gt_dy_m"]),
                })
            except (ValueError, KeyError):
                continue

    states = {}
    for r in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
    print(f"\n[analyze] rows={len(rows)} states={states}")

    locked_rows = [r for r in rows if r["state"] == "LOCKED"]
    if not locked_rows:
        print("[analyze] no LOCKED rows — capture failed or all frames LOST")
        return None

    t0_us = locked_rows[0]["ts_us"]
    dwell_us = int(task5_dwell_s * 1e6)
    t_hover_end = t0_us + int(task4_hover_s * 1e6)
    t_s1_end = t_hover_end + dwell_us        # end of shifted_x
    t_r1_end = t_s1_end + dwell_us            # end of returned_x
    t_s2_end = t_r1_end + dwell_us            # end of shifted_y
    # returned_y runs from t_s2_end to end of log

    def _between(t_lo, t_hi):
        return [r for r in locked_rows if t_lo <= r["ts_us"] < t_hi]

    hover = _between(t0_us, t_hover_end)
    shifted_x = _between(t_hover_end, t_s1_end)
    returned_x = _between(t_s1_end, t_r1_end)
    shifted_y = _between(t_r1_end, t_s2_end)
    returned_y = [r for r in locked_rows if r["ts_us"] >= t_s2_end]

    def _rms(vals):
        return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else 0.0

    # Trim transient at phase boundary — teleport settles in <300 ms in bench,
    # but XFeat sees a few mixed frames. Use the later half for numeric gates.
    def _settled(rs):
        if len(rs) < 4:
            return rs
        return rs[int(len(rs) * PHASE_SETTLE_FRAC):]

    any_lost = any(r["state"] == "LOST" for r in rows)

    # Task 4 stats: RMS of (dx_m - gt_dx_m, dy_m - gt_dy_m) — for static hover
    # gt is (0,0) so this reduces to RMS of (dx_m, dy_m).
    if hover:
        ex = [r["dx_m"] - r["gt_dx"] for r in hover]
        ey = [r["dy_m"] - r["gt_dy"] for r in hover]
        rms_x = _rms(ex); rms_y = _rms(ey)
        rms_xy = math.sqrt((rms_x ** 2 + rms_y ** 2))
        max_r = max(math.sqrt(x * x + y * y) for x, y in zip(ex, ey))
        inl = [r["inliers"] for r in hover]
        rep = [r["reproj"] for r in hover]
        dur = (hover[-1]["ts_us"] - hover[0]["ts_us"]) / 1e6
        print(f"\n[Task 4 hover — {len(hover)} LOCKED rows over {dur:.1f}s sim]")
        print(f"  err RMS:  dx={rms_x:.3f} m  dy={rms_y:.3f} m  "
              f"radial={rms_xy:.3f} m  max={max_r:.3f} m")
        print(f"  inliers:  mean={sum(inl)/len(inl):.0f}  "
              f"min={min(inl)}  max={max(inl)}")
        print(f"  reproj:   mean={sum(rep)/len(rep):.2f} px  "
              f"max={max(rep):.2f} px")
        verdict = "PASS" if rms_xy < RMS_PASS_M else "FAIL"
        print(f"  Task 4 gate: radial RMS {rms_xy:.3f} m "
              f"< {RMS_PASS_M:.2f} m => {verdict}")

    # Task 5: LOCKED persistence + axis-specific gates.
    def _phase_stats(label, rs, exp_dx, exp_dy, tol_target, tol_offaxis):
        """exp_dx, exp_dy: expected body-flat delta for this phase.
        tol_target: tolerance on the non-zero axis (±).
        tol_offaxis: tolerance on the zero axis (±).
        For returned phases, both axes are target (exp=0, tol=tol_target).
        """
        if not rs:
            print(f"  [{label}] no rows — FAIL")
            return False
        settled = _settled(rs)
        mean_dx = sum(r["dx_m"] for r in settled) / len(settled)
        mean_dy = sum(r["dy_m"] for r in settled) / len(settled)
        err_dx = mean_dx - exp_dx
        err_dy = mean_dy - exp_dy
        # For a shifted phase, err on target axis uses tol_target;
        # off-axis uses tol_offaxis. For returned, both use tol_target.
        tol_x = tol_target if exp_dx != 0 else tol_offaxis
        tol_y = tol_target if exp_dy != 0 else tol_offaxis
        ok_x = abs(err_dx) <= tol_x
        ok_y = abs(err_dy) <= tol_y
        ok = ok_x and ok_y
        dur = (rs[-1]["ts_us"] - rs[0]["ts_us"]) / 1e6
        inl = [r["inliers"] for r in rs]
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{label}] {len(rs)} rows ({dur:.1f}s)  "
              f"inl={min(inl)}..{max(inl)}  "
              f"mean(dx,dy)=({mean_dx:+.3f},{mean_dy:+.3f})  "
              f"exp=({exp_dx:+.1f},{exp_dy:+.1f})  "
              f"err=({err_dx:+.3f},{err_dy:+.3f})  "
              f"tol=(±{tol_x:.2f},±{tol_y:.2f}) => {verdict}")
        return ok

    print(f"\n[Task 5 drift-return @ ±{TASK5_SHIFT_M} m on X and Y]")
    r_sx = _phase_stats("shifted_x ", shifted_x,
                        +TASK5_SHIFT_M, 0.0,
                        SHIFT_TARGET_TOL_M, SHIFT_OFFAXIS_TOL_M)
    r_rx = _phase_stats("returned_x", returned_x,
                        0.0, 0.0,
                        RETURN_TOL_M, RETURN_TOL_M)
    r_sy = _phase_stats("shifted_y ", shifted_y,
                        0.0, +TASK5_SHIFT_M,
                        SHIFT_TARGET_TOL_M, SHIFT_OFFAXIS_TOL_M)
    r_ry = _phase_stats("returned_y", returned_y,
                        0.0, 0.0,
                        RETURN_TOL_M, RETURN_TOL_M)
    print(f"  LOCKED maintained throughout: {not any_lost}")
    overall = r_sx and r_rx and r_sy and r_ry and not any_lost
    print(f"  Task 5 overall: {'PASS' if overall else 'FAIL'}")


def main():
    LOGS.mkdir(exist_ok=True)
    env = build_env()

    # Minimal CLI: --no-task5, --hover-s S, --noisy-baro, --noise-seed N.
    do_task5 = "--no-task5" not in sys.argv
    noisy_baro = "--noisy-baro" in sys.argv
    noise_seed = 0
    if "--noise-seed" in sys.argv:
        i = sys.argv.index("--noise-seed")
        noise_seed = int(sys.argv[i + 1])
    hover_s = TASK4_HOVER_S
    if "--hover-s" in sys.argv:
        i = sys.argv.index("--hover-s")
        hover_s = float(sys.argv[i + 1])

    ts = time.strftime("%H%M%S")
    gz_log = f"/tmp/anchor_bench_gz_{ts}.log"
    est_log = f"{LOGS}/anchor_bench_estimator_{ts}.log"
    bench_log = f"{LOGS}/anchor_bench_{ts}.log"
    baro_log = f"{LOGS}/anchor_bench_baro_{ts}.csv" if noisy_baro else None

    bench_f = open(bench_log, "w")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        bench_f.write(line + "\n"); bench_f.flush()

    log(f"bench log      : {bench_log}")
    log(f"gzserver log   : {gz_log}")
    log(f"estimator log  : {est_log}")
    log(f"world          : {WORLD}")
    log(f"altitude       : {ALT_M} m AGL")
    log(f"camera         : 110° HFOV (Arducam prod. specs), 1280x800 L8")
    if noisy_baro:
        log(f"baro noise     : ON (BMP388+env model, seed={noise_seed})")
        log(f"baro samples   : {baro_log}")
    else:
        log(f"baro noise     : OFF (deterministic ground-truth z)")

    log("killing stale gz/ros processes...")
    subprocess.run(["pkill", "-9", "-f", "gzserver|gzclient|MicroXRCEAgent"],
                   check=False)
    time.sleep(1)

    gz = start_gzserver(env, gz_log)
    log("gzserver launched — waiting 8s for boot + plugin init...")
    time.sleep(8)
    if gz.poll() is not None:
        log("gzserver died during boot — see log")
        return 2

    # Teleport to starting pose BEFORE estimator subscribes, so first frames
    # are already at target altitude.
    if not gz_teleport(env, 0.0, 0.0, ALT_M):
        log("initial teleport failed")
        kill_group(gz)
        return 2
    log(f"teleported anchor_cam to (0, 0, {ALT_M})")
    time.sleep(2)

    rclpy.init()
    current_z = {"z": ALT_M}
    baro_noise = BaroNoiseModel(seed=noise_seed) if noisy_baro else None
    if baro_noise is not None:
        log(f"baro noise cfg : {baro_noise.describe()}")
    fake = FakePx4(lambda: current_z["z"], baro_noise=baro_noise,
                   baro_log_path=baro_log)
    stop_spin = threading.Event()

    def spin_fake():
        while not stop_spin.is_set():
            try:
                rclpy.spin_once(fake, timeout_sec=0.1)
            except Exception:
                break
    spinner = threading.Thread(target=spin_fake, daemon=True)
    spinner.start()
    log("fake /fmu/out/{vehicle_air_data,vehicle_attitude} publishing @ 50 Hz")

    est = start_estimator(env, est_log)
    log("anchor_estimator launched — waiting 25s for XFeat weights + ROS init...")
    time.sleep(25)
    if est.poll() is not None:
        log("anchor_estimator died during init — see log")
        kill_group(gz); stop_spin.set(); rclpy.shutdown()
        return 2

    log("calling /anchor/capture...")
    ok, cap_out = call_capture(env)
    log(f"capture: {'OK' if ok else 'FAILED'}")
    if not ok:
        log(cap_out[:500])
        kill_group(est); kill_group(gz); stop_spin.set(); rclpy.shutdown()
        return 2
    t_capture_us = int(time.time() * 1e6)

    # Task 4: static hover (camera held at (0,0,ALT))
    log(f"Task 4: holding (0,0,{ALT_M}) for {hover_s:.0f} s...")
    time.sleep(hover_s)
    t_task5_shift_us = int(time.time() * 1e6)

    t_end_us = t_task5_shift_us
    if do_task5:
        log(f"Task 5a: teleport (+{TASK5_SHIFT_M},0,{ALT_M})")
        gz_teleport(env, +TASK5_SHIFT_M, 0.0, ALT_M)
        time.sleep(TASK5_DWELL_S)
        log(f"Task 5a: return (0,0,{ALT_M})")
        gz_teleport(env, 0.0, 0.0, ALT_M)
        time.sleep(TASK5_DWELL_S)
        log(f"Task 5b: teleport (0,+{TASK5_SHIFT_M},{ALT_M})")
        gz_teleport(env, 0.0, +TASK5_SHIFT_M, ALT_M)
        time.sleep(TASK5_DWELL_S)
        log(f"Task 5b: return (0,0,{ALT_M})")
        gz_teleport(env, 0.0, 0.0, ALT_M)
        time.sleep(TASK5_DWELL_S)
        t_end_us = int(time.time() * 1e6)
    else:
        log("Task 5 skipped (--no-task5)")

    log("finalizing — killing estimator + gzserver...")
    kill_group(est)
    time.sleep(1)
    kill_group(gz)
    stop_spin.set()

    b = fake.baro_stats()
    if b is not None and noisy_baro:
        log(f"baro error     : RMS={b['rms_m']:.3f}m  "
            f"mean|err|={b['mean_abs_m']:.3f}m  "
            f"max|err|={b['max_abs_m']:.3f}m  N={b['samples']}")
        # Analytical prediction for shifted-phase mean error: ~dx_true·bias/h_AGL
        pred_shift = TASK5_SHIFT_M * abs(baro_noise.bias) / ALT_M
        pred_sigma_shift = (TASK5_SHIFT_M
                            * math.sqrt(baro_noise.sigma_drift ** 2
                                        + baro_noise.sigma_white ** 2)
                            / ALT_M)
        log(f"analytic shift : mean |err| ≈ {pred_shift*100:.1f} cm "
            f"(bias), σ_per-sample ≈ {pred_sigma_shift*100:.1f} cm "
            f"on {TASK5_SHIFT_M:.0f} m shift at {ALT_M:.0f} m AGL")
    fake.close_baro_log()
    try:
        fake.destroy_node(); rclpy.shutdown()
    except Exception:
        pass

    # Wait for estimator to flush CSV
    time.sleep(1)
    csv_path = find_csv_in_log(est_log)
    log(f"CSV: {csv_path}")

    # Wall-clock phase marks (for log correlation only — CSV uses sim time).
    log(f"wall marks: cap={t_capture_us} task5_start={t_task5_shift_us} "
        f"end={t_end_us}")
    analyze(csv_path, task4_hover_s=hover_s)

    bench_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
