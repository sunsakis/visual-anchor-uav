#!/usr/bin/env python3
"""Offline anchor-matching demo (no Gazebo required).

Loads a Ukraine z16 aerial texture, picks a centre crop as the *anchor*,
then synthetically translates + rotates the source through N waypoints
to simulate a drone drifting around the anchor footprint at ~100 m AGL.

For each waypoint:
  1. Crop the live view at the same size as the anchor.
  2. Run XFeat + mutual-NN matching against the anchor.
  3. cv2.findHomography(RANSAC, 3 px) → H.
  4. Decompose H → dx / dy / dyaw in metres (uses fake 100 m altitude
     and assumed 110° HFOV — tuneable; this is a pipeline demo, not a
     quantitative bench).
  5. Render anchor | live | match lines + a telemetry panel.

Stitches the frames to docs/demo.gif via ffmpeg.

Run:
    .venv/bin/python3 demo/demo_offline_aerial.py
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
SRC_TILE = PROJ / "bench" / "aerial" / "donbas_rural.jpg"
OUT_DIR = PROJ / "docs" / "_demo_frames"
OUT_GIF = PROJ / "docs" / "demo.gif"

CROP = 512                      # anchor + live crop size, square
HFOV_DEG = 110.0                # matches v0 BOM lens
ALT_M = 100.0                   # synthetic altitude for metric scaling
N_WP = 24                       # waypoints around the trajectory
RADIUS_PX = 80                  # synthetic translation radius in source-pixel units
YAW_AMP_DEG = 0.0               # pure translation; rotation invariance proved in bench/layer15b
FPS = 8


def build_matcher():
    print("Loading XFeat...", flush=True)
    xfeat = torch.hub.load(
        "verlab/accelerated_features", "XFeat",
        pretrained=True, top_k=4096, trust_repo=True,
    )

    @torch.inference_mode()
    def run(anchor_gray, live_gray):
        a3 = np.stack([anchor_gray] * 3, axis=-1)
        b3 = np.stack([live_gray] * 3, axis=-1)
        mk0, mk1 = xfeat.match_xfeat(a3, b3, top_k=4096)
        if len(mk0) < 8:
            return None, np.asarray(mk0), np.asarray(mk1), None
        H, mask = cv2.findHomography(np.asarray(mk0), np.asarray(mk1),
                                     cv2.RANSAC, 3.0)
        return H, np.asarray(mk0), np.asarray(mk1), mask
    return run


def warp_view(src, cx, cy, yaw_deg):
    """Return a CROP×CROP view centred on (cx, cy) of `src`, rotated by yaw."""
    M = cv2.getRotationMatrix2D((cx, cy), yaw_deg, 1.0)
    M[0, 2] += CROP / 2.0 - cx
    M[1, 2] += CROP / 2.0 - cy
    return cv2.warpAffine(src, M, (CROP, CROP),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def decompose(H, w, h):
    """Centre-pixel translation + rotation from a homography."""
    if H is None:
        return None, None, None
    c = np.array([w / 2.0, h / 2.0, 1.0])
    cp = H @ c
    cp /= cp[2]
    dx_px = cp[0] - w / 2.0
    dy_px = cp[1] - h / 2.0
    dyaw = math.atan2(H[1, 0], H[0, 0])
    return dx_px, dy_px, dyaw


def render(anchor, live, mk0, mk1, mask, dx_m, dy_m, dyaw, gt, i, n):
    h, w = anchor.shape[:2]
    canvas = np.zeros((h + 200, 2 * w + 20, 3), dtype=np.uint8)
    canvas[:h, :w] = cv2.cvtColor(anchor, cv2.COLOR_GRAY2BGR)
    canvas[:h, w + 20:2 * w + 20] = cv2.cvtColor(live, cv2.COLOR_GRAY2BGR)

    if mask is not None and len(mk0) and len(mk1):
        inl_idx = np.where(mask.ravel() > 0)[0]
        sample = inl_idx[::max(1, len(inl_idx) // 60)]
        for k in sample:
            x0, y0 = mk0[k].astype(int)
            x1, y1 = mk1[k].astype(int)
            cv2.line(canvas, (x0, y0), (x1 + w + 20, y1), (60, 200, 60), 1)
            cv2.circle(canvas, (x0, y0), 2, (0, 255, 0), -1)
            cv2.circle(canvas, (x1 + w + 20, y1), 2, (0, 255, 0), -1)

    cv2.putText(canvas, "ANCHOR (donbas_rural, ESRI z16)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, f"LIVE  (waypoint {i+1}/{n})", (w + 28, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    panel_y0 = h + 8
    cv2.rectangle(canvas, (0, h), (canvas.shape[1], canvas.shape[0]),
                  (32, 32, 32), -1)
    inl = int(mask.sum()) if mask is not None else 0
    if dx_m is None:
        line1 = "motion_delta: NO LOCK"
        line2 = ""
        line3 = ""
    else:
        line1 = (f"motion_delta  dx={dx_m:+6.2f} m   "
                 f"dy={dy_m:+6.2f} m   dyaw={math.degrees(dyaw):+6.2f} deg")
        line2 = (f"ground truth  dx={gt[0]:+6.2f} m   "
                 f"dy={gt[1]:+6.2f} m   dyaw={gt[2]:+6.2f} deg")
        err = math.hypot(dx_m - gt[0], dy_m - gt[1])
        line3 = (f"|err|={err:5.2f} m    inliers={inl:4d}    "
                 f"reference=ANCHOR    lock=LOCKED")
    cv2.putText(canvas, line1, (12, panel_y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 220, 255), 1)
    cv2.putText(canvas, line2, (12, panel_y0 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(canvas, line3, (12, panel_y0 + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas,
                "XFeat + RANSAC homography  |  altitude 100 m (synthetic)  "
                "|  HFOV 110 deg",
                (12, panel_y0 + 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    return canvas


def main():
    src = cv2.imread(str(SRC_TILE), cv2.IMREAD_GRAYSCALE)
    if src is None:
        print(f"missing tile: {SRC_TILE}", file=sys.stderr)
        sys.exit(1)
    H, W = src.shape[:2]
    cx0, cy0 = W // 2, H // 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*.png"):
        p.unlink()

    anchor = warp_view(src, cx0, cy0, 0.0)
    matcher = build_matcher()
    _ = matcher(anchor, anchor)  # warmup

    focal_px = (CROP / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    px_per_m = focal_px / ALT_M

    for i in range(N_WP):
        ang = 2 * math.pi * i / N_WP
        dx_px_gt = RADIUS_PX * math.cos(ang)
        dy_px_gt = RADIUS_PX * math.sin(ang)
        yaw_gt = YAW_AMP_DEG * math.sin(ang)

        live = warp_view(src, cx0 + dx_px_gt, cy0 + dy_px_gt, yaw_gt)
        Hmat, mk0, mk1, mask = matcher(anchor, live)
        dxp, dyp, dyaw = decompose(Hmat, CROP, CROP)

        if dxp is None:
            dx_m = dy_m = None
        else:
            dx_m = dxp / px_per_m
            dy_m = dyp / px_per_m

        gt_dx_m = -dx_px_gt / px_per_m
        gt_dy_m = -dy_px_gt / px_per_m
        gt_dyaw = -math.radians(yaw_gt)
        gt = (gt_dx_m, gt_dy_m, math.degrees(gt_dyaw))

        frame = render(anchor, live, mk0, mk1, mask, dx_m, dy_m,
                       0.0 if dyaw is None else dyaw, gt, i, N_WP)
        cv2.imwrite(str(OUT_DIR / f"f_{i:03d}.png"), frame)

        inl = int(mask.sum()) if mask is not None else 0
        if dx_m is None:
            print(f"  wp {i:2d}/{N_WP}  NO LOCK  inl={inl}")
        else:
            err = math.hypot(dx_m - gt_dx_m, dy_m - gt_dy_m)
            print(f"  wp {i:2d}/{N_WP}  truth=({gt_dx_m:+5.2f},"
                  f"{gt_dy_m:+5.2f}) m  est=({dx_m:+5.2f},{dy_m:+5.2f}) m  "
                  f"err={err:.2f} m  inl={inl}")

    print("Stitching GIF...", flush=True)
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    palette = OUT_DIR / "palette.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
         "-i", str(OUT_DIR / "f_%03d.png"),
         "-vf", "fps=8,scale=900:-1:flags=lanczos,palettegen",
         str(palette)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
         "-i", str(OUT_DIR / "f_%03d.png"),
         "-i", str(palette),
         "-lavfi", "fps=8,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse",
         str(OUT_GIF)],
        check=True,
    )
    palette.unlink(missing_ok=True)
    print(f"wrote {OUT_GIF}")


if __name__ == "__main__":
    main()
