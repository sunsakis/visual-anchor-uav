#!/usr/bin/env python3
"""Re-render demo #2 from cached captures.

Uses /tmp/demo2_cache/{anchor.png, live_NNN.png} populated by the first
gazebo run, so visualization tweaks can iterate in ~60 s without
re-running Gazebo.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from demo2_common import (
    CAM_W, CAM_H, CAM_HFOV_DEG, ALT_M,
    TRAJ_RADIUS_M, TRAJ_N, TRAJ_FPS,
    decompose, render_frame, match_xfeat, pixels_to_world,
)

CACHE = Path("/tmp/demo2_cache")
OUT_DIR = Path("/tmp/demo2_out")
OUT_GIF = Path("/tmp/motion_delta_demo2.gif")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for p in OUT_DIR.glob("*.png"):
        p.unlink()

    anchor_bgr = cv2.imread(str(CACHE / "anchor.png"))
    assert anchor_bgr is not None, "anchor cache missing"
    anchor_gray = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY)

    print("Loading XFeat (cached)...", flush=True)
    xfeat = torch.hub.load(
        "verlab/accelerated_features", "XFeat",
        pretrained=True, top_k=4096, trust_repo=True,
    )

    focal_px = (CAM_W / 2.0) / math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
    traj_truth = [(TRAJ_RADIUS_M * math.cos(2 * math.pi * i / TRAJ_N),
                   TRAJ_RADIUS_M * math.sin(2 * math.pi * i / TRAJ_N))
                  for i in range(TRAJ_N)]
    traj_est = [None] * TRAJ_N

    # Warmup
    _ = match_xfeat(xfeat, anchor_gray, anchor_gray)

    for i, (tx, ty) in enumerate(traj_truth):
        live_path = CACHE / f"live_{i:03d}.png"
        if not live_path.exists():
            print(f"  wp {i}: missing cache", flush=True)
            continue
        live_bgr = cv2.imread(str(live_path))
        live_gray = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY)
        H, mk0, mk1, mask = match_xfeat(xfeat, anchor_gray, live_gray)
        frame = render_frame(anchor_bgr, live_bgr, H, mk0, mk1, mask,
                             focal_px, (tx, ty), traj_truth, traj_est, i)
        cv2.imwrite(str(OUT_DIR / f"f_{i:03d}.png"), frame)
        if H is not None:
            dxp, dyp, _ = decompose(H, CAM_W, CAM_H)
            dx_m, dy_m = pixels_to_world(dxp, dyp, focal_px, ALT_M)
            err = math.hypot(dx_m - tx, dy_m - ty)
            inl = int(mask.sum()) if mask is not None else 0
            print(f"  wp {i:2d}/{TRAJ_N}  truth=({tx:+5.1f},{ty:+5.1f})  "
                  f"est=({dx_m:+5.1f},{dy_m:+5.1f})  err={err:.2f} m  "
                  f"inl={inl}", flush=True)

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
