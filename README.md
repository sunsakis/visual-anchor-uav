# Visual Anchor Positioning for GPS-Denied UAV

A drift-free, vision-only position-hold module for drones operating in GPS-denied environments. Matches the live downward camera frame against a stored reference image (the *anchor*) using a learned feature extractor + RANSAC homography, and emits a `motion_delta` message that the host flight controller's EKF fuses as an absolute-position measurement.

Bench-validated against six Ukraine terrain types in a PX4 SITL + Gazebo Classic harness. Designed to ship on an ARM SoC + edge NPU; current dev runs on Raspberry Pi 4.

> **Status:** sim-validated. Hardware bring-up (Pi 4 + OV9281 global-shutter mono camera + IMU + baro) is the next step, followed by ONNX export and INT8 quantization for embedded NPU deploy.

![demo](docs/demo.gif)

*Offline pipeline demo. Anchor frame (left) is a centre crop of an ESRI z16 aerial tile of Donbas, Ukraine. The live frame (right) drifts around the anchor on a 24-waypoint circle. Each frame: XFeat → mutual-NN matching → RANSAC homography → metres in the world frame. End-to-end error stays under 0.2 m across the trajectory. Reproduce with `python3 demo/demo_offline_aerial.py`.*

---

## The problem

GPS-jammed and GPS-denied operation is the default in contested airspace. A drone at 100 m altitude that loses GPS today either lands, drifts on dead-reckoning IMU + optical flow (which integrates error indefinitely), or fails over to VIO (which still drifts, slower). None of these are *drift-free*: error grows monotonically with time.

If the drone has a **stored reference image** of the area beneath it (a tile from a recent overflight, a pre-mission map, a live-shared image from a teammate), then re-localizing against that anchor is a *drift-free absolute measurement*: every match resets the error to whatever the matcher's reprojection error is, regardless of how long the flight has been running.

This module is the perception side of that idea. The output contract is `motion_delta` — a single ROS2 schema with a `reference` field — that the host EKF fuses without modification:

| `reference`        | Path on host EKF                            | Rate    | Drift          |
|--------------------|---------------------------------------------|---------|----------------|
| `ANCHOR`           | PX4 `EKF2_EV_CTRL` (external vision)        | 2–10 Hz | **drift-free** while LOCKED |
| `PREVIOUS_FRAME`   | PX4 `EKF2_OF_CTRL` (optical flow)           | 100 Hz  | bounded growth between anchors |

Two paths, one schema, one host EKF. The module never publishes a position estimate — only displacement deltas. Position state lives on the flight controller.

---

## Approach

```
                       Downward camera (OV9281, global shutter, ~110° HFOV)
                                          │
                                          ▼
         ┌────────────────────────────────────────────────────────────┐
         │                   Anchor estimator (10 Hz)                 │
         │   IDLE → LOCKED → LOST state machine                       │
         │   /anchor/capture stores: anchor frame, baro h_ref, yaw,t  │
         │                                                            │
         │   Live frame:                                              │
         │     • XFeat (PyTorch) → keypoints + descriptors            │
         │     • mutual-NN matching                                   │
         │     • cv2.findHomography(RANSAC, 3 px)                     │
         │     • H[0,2]/H[1,2] × h_now / f_px → dx, dy in meters      │
         │     • atan2(H[1,0], H[0,0]) → dyaw                         │
         │   Loss criterion: <30 inliers OR >5 px reproj × 3 frames   │
         └────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
         ┌────────────────────────────────────────────────────────────┐
         │              Short-window estimator (100 Hz)               │
         │   Pyramidal Lucas–Kanade on Shi–Tomasi corners             │
         │   Gyro pre-rotation; RANSAC median translation             │
         │   Dense DIS fallback when feature count drops              │
         │   Resets at every new anchor lock                          │
         └────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                  motion_delta (10/100 Hz)
                                          │
                                          ▼
                ┌────────────── EV/OF shim (host) ──────────────┐
                │  ANCHOR        → /fmu/in/vehicle_visual_odom  │
                │  PREVIOUS_FRAME → /fmu/in/sensor_optical_flow │
                └───────────────────────────────────────────────┘
                                          │
                                          ▼
                                  PX4 EKF2 (unmodified)
```

### Why XFeat and not ORB or SuperPoint+LightGlue

Bench data lives in `bench/layer15b_results.csv`. Headline:

| Matcher                | Cross-provider success | Median inference (ms) | Inliers (severodonetsk dx256) |
|------------------------|------------------------|-----------------------|-------------------------------|
| ORB                    | 3/6 (fails on rural)   | ~30                   | 0–1300                        |
| SuperPoint + LightGlue | 6/6                    | ~2900                 | 988                           |
| **XFeat**              | **6/6**                | **~115**              | **1748**                      |

XFeat is **~25× faster than SP+LG** at equal-or-better inlier counts, and **does not fail** on the rural and river terrain that breaks ORB. At ~115 ms per match on a desktop GPU, the anchor loop comfortably fits a 2–10 Hz target rate; INT8 quantization for the embedded NPU brings this to the same order of magnitude on a sub-1 W part.

### Why an anchor instead of pure VIO

VIO is dead-reckoning. Even a state-of-the-art VIO stack drifts ~0.1–1 % of distance traveled. At 100 m hover for 30 minutes, that's metres of error. Anchor matching is *absolute*: error is bounded by matcher reprojection accuracy (~1–3 px → sub-metre at 100 m AGL with a 110° lens), regardless of flight duration.

This trades robustness against transit and feature-poor scenes (where VIO would still work) for drift-free behavior over the operating envelope where the anchor is in view. The short-window optical-flow path bridges the gap between anchor locks.

---

## Bench results

### L1.5b — feature-matcher viability across providers

`bench/layer15b_harness.py`, results in `bench/layer15b_results.csv` (451 rows).

Six Ukraine terrain types: `donbas_rural`, `kharkiv_urban`, `izyum_fields`, `kramatorsk_mixed`, `oskil_river`, `severodonetsk`. Each tested in two regimes:
- **cross-provider** — ESRI tile vs Google tile of the same area (different sensors, different time of day, different season). Hardest test of matcher generality.
- **spatial-overlap** — single-provider with synthetic translations (64–256 px) and yaw rotations (5–30°). Tests scale + rotation invariance.

Result: **XFeat passes 100 %** of cross-provider tests with mean reprojection error 1.6 px; ORB falls over on rural terrain (3/6 fails); SP+LG passes but is 25× slower for no inlier benefit. XFeat picked.

### L2 Task 1 — sim texture + camera sweep (altitude floor / ceiling)

`bench/sim_texture_probe.py` against a Gazebo plane with z18 (0.40 m/px) ESRI tiles of the same six Ukraine locations. Six camera configurations × twelve altitudes (5–900 m) × six locations.

Result with the chosen baseline (1280×800, 90° HFOV): **70/72 PASS** across the v0 product band (≥30 m altitude). Texture-GSD-driven floor at 110° HFOV is **20 m** — that's the lowest altitude at which a 0.40 m/px tile resolves enough features for reliable matching, an artifact of sim resolution rather than the algorithm. No upper ceiling found within the simulator's geometry.

### L2 Task 4 — anchor estimator on a teleport trajectory

`bench/anchor_hover_bench.py`. Spawns the camera at 100 m AGL over `donbas_rural`, captures an anchor, then teleports along a 60-waypoint circular trajectory (r = 30 m) and measures `dx, dy, dyaw` against ground truth.

Result: 0.0 m RMS on three static hovers (sim floor; deterministic), 5 mm scale error on a 10 m drift-and-return. End-to-end pipeline measured; the IDLE → LOCKED → LOST state machine triggers correctly on inlier loss.

CSV outputs and per-test plots are checked into `bench/`.

---

## Repository layout

```
.
├── bench/                       Offline matcher + sim-probe bench harnesses
│   ├── layer1_harness.py            Layer 1: matcher viability, single image
│   ├── layer15_harness.py           Layer 1.5: spatial-overlap sweep
│   ├── layer15b_harness.py          Layer 1.5b: cross-provider sweep
│   ├── sim_texture_probe.py         L2 T1: 6 cameras × 12 altitudes in Gazebo
│   ├── anchor_hover_bench.py        L2 T4: anchor estimator on teleport traj
│   ├── analyze*.py                  Plotting / aggregation
│   ├── fetch_tiles*.py              ESRI / Google tile fetch (pre-mission map)
│   └── aerial/                      z16 fixtures for the six Ukraine terrains
│
├── ros2/
│   ├── motion_delta_msgs/           ROS2 message package (the output contract)
│   └── offboard_commander/          ROS2 nodes
│       ├── anchor_estimator.py        XFeat + RANSAC anchor path (10 Hz)
│       ├── displacement_estimator.py  Lucas–Kanade short-window path (100 Hz)
│       ├── anchor_ev_shim.py          ANCHOR → PX4 EV odometry translator
│       ├── preflight_gate.py          Flow-quality / lidar / illumination gate
│       └── hover_test.py              Dual-metric (EKF + ground truth) test
│
├── gazebo_resources/            Sim worlds + camera / iris models
│   ├── worlds/
│   │   ├── anchor_bench.world       Bench-only plane with z18 texture
│   │   └── iris_anchor.world        Full SITL with downward anchor camera
│   ├── models/anchor_cam/           Pinhole anchor camera SDF
│   └── models/iris_opt_flow/        PX4 iris + camera-flow plugin
│
└── demo/                        Visualization scripts (motion_delta overlay)
    ├── demo_offline_aerial.py             Offline trajectory demo (no Gazebo) → docs/demo.gif
    ├── demo1_motion_delta_xfeat_gif.py    Pipeline demo on handheld footage
    └── demo2_motion_delta_aerial.py       Aerial trajectory demo (Gazebo)
```

---

## Hardware target

v0 ships as a plug-and-play module — single connector to the host flight controller, draws power from the airframe, runs its own perception stack on board.

| Component         | Part                          | Notes                                    |
|-------------------|-------------------------------|------------------------------------------|
| Camera            | Arducam OV9281 mono + 110° HFOV M12 | 1280×800 @ 120 Hz, global shutter  |
| IMU               | Bosch BMI270                  | Qwiic chain                              |
| Barometer         | Bosch BMP388                  | Altitude → metric scale for homography   |
| Magnetometer (opt)| Bosch BMM150                  | Yaw redundancy in clean RF environments  |
| Compute (dev)     | Raspberry Pi 4                | XFeat fp32 fits comfortably              |
| Compute (ship)    | SoC + edge NPU (Coral / K230) | ONNX INT8 export — *next milestone*      |

**Approximate BOM cost: $150–170.**

---

## Roadmap

1. **Hardware bring-up** *(next)* — Pi 4 + OV9281 + Bosch IMU/baro stack on bench. First outdoor capture against a stored anchor. Validate the simulator's findings on real imagery.
2. **ONNX export + INT8 quantization** — XFeat → ONNX → quantized INT8 model. Latency bench on Pi 4 CPU, then on Coral Edge TPU and Kendryte K230. Goal: ≥10 Hz anchor rate within a sub-1 W power budget on ARM.
3. **Closed-loop hold flight** — module wired to a PX4 flight controller, position-hold demonstration with GPS off, anchor pre-loaded.
4. **Anchor handoff** — multi-anchor map; smooth handoff as the drone transits between anchor footprints. Short-window estimator carries the gap.
5. **Production module** — custom PCB, single connector to host FC, environmental sealing.

---

## Reproducing the bench

The offline bench (matcher viability, spatial-overlap, cross-provider) runs without Gazebo:

```bash
cd bench
python3 -m venv .venv && source .venv/bin/activate
pip install torch opencv-python numpy matplotlib pillow

# Layer 1: matcher viability on a single image
python3 layer1_harness.py

# Layer 1.5: spatial-overlap sweep (rotation + translation invariance)
python3 layer15_harness.py

# Layer 1.5b: cross-provider sweep (ESRI vs Google, 6 Ukraine terrains)
python3 layer15b_harness.py
```

The Gazebo-based sweeps (`sim_texture_probe.py`, `anchor_hover_bench.py`) require PX4 SITL + Gazebo Classic + ROS2 Humble + a custom DDS bridge. Setup notes in [`docs/SITL.md`](docs/SITL.md).

---

## Engineering notes

- **Sim is a measurement harness, not the deliverable.** The Gazebo numbers are a sanity floor. Real-imagery validation on hardware is the gate that matters; the sim-validated bench just keeps us from wasting hardware iteration cycles on obvious algorithmic mistakes.
- **The host EKF can lie.** A `vehicle_local_position` topic that says "PASS" while the drone drifts out of frame in Gazebo was the highest-yield bug class in this project. Every test that reports a metric also subscribes to `/gazebo/model_states` ground truth and reports both — the divergence between EKF and world is a bug detector.
- **C++ port is on the roadmap, not in this repo.** The hot path that benefits most from C++ is the 100 Hz Lucas–Kanade short-window estimator (Python GC pauses become measurable at 100 Hz on embedded ARM). The 10 Hz anchor path lives comfortably in Python — `cv2.findHomography` and the XFeat ONNX runtime are already C++ underneath.

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
