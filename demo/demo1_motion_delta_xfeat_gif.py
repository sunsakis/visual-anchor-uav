#!/usr/bin/env python3
"""Demo #1 — motion_delta overlay on the XFeat demo GIF.

Loads /tmp/xfeat.gif, uses frame 0 as the anchor, runs XFeat + RANSAC
homography against each subsequent frame, decomposes to dx/dy/dyaw, and
renders a pitch-style visualization with the motion_delta schema fields
overlaid. Stitches to /tmp/motion_delta_demo1.gif via ffmpeg.

Source footage is handheld (wall-view, not downward aerial), so the
"meters" output uses a fake 1 m altitude + assumed 60 deg HFOV. This is
a pipeline demo, not a quantitative benchmark.
"""
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

GIF_IN = Path('/tmp/xfeat.gif')
FRAMES_DIR = Path('/tmp/demo1_src')
OUT_DIR = Path('/tmp/demo1_out')
OUT_GIF = Path('/tmp/motion_delta_demo1.gif')

FAKE_ALTITUDE_M = 1.0
HFOV_DEG = 60.0


def extract_frames():
    FRAMES_DIR.mkdir(exist_ok=True)
    for p in FRAMES_DIR.glob('*.png'):
        p.unlink()
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-i', str(GIF_IN),
         str(FRAMES_DIR / 'f_%03d.png')],
        check=True,
    )
    return sorted(FRAMES_DIR.glob('*.png'))


def build_matcher():
    xfeat = torch.hub.load(
        'verlab/accelerated_features', 'XFeat',
        pretrained=True, top_k=4096, trust_repo=True,
    )

    @torch.inference_mode()
    def run(a, b):
        mk0, mk1 = xfeat.match_xfeat(a, b, top_k=4096)
        if len(mk0) < 8:
            return None, np.asarray(mk0), np.asarray(mk1), None
        H, mask = cv2.findHomography(mk0, mk1, cv2.RANSAC, 3.0)
        return H, np.asarray(mk0), np.asarray(mk1), mask
    return run


def decompose_homography(H, W, Himg):
    """Center-pixel translation + rotation decomposition."""
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


def render(anchor, target, H, mk0, mk1, mask, focal_px):
    Himg, W = anchor.shape[:2]
    top_l = anchor.copy()
    top_r = target.copy()
    pad = 40
    rect = np.array([
        [pad, pad], [W - pad, pad],
        [W - pad, Himg - pad], [pad, Himg - pad],
    ], dtype=np.float32).reshape(-1, 1, 2)
    cv2.polylines(top_l, [rect.astype(np.int32)], True, (0, 255, 0), 2)
    if H is not None:
        warped = cv2.perspectiveTransform(rect, H)
        cv2.polylines(top_r, [warped.astype(np.int32)], True, (0, 255, 0), 2)

    bottom = np.zeros((Himg, W * 2, 3), dtype=np.uint8)
    bottom[:, :W] = anchor
    bottom[:, W:] = target
    if H is not None and mask is not None and len(mk0) > 0:
        inlier_mask = mask.ravel().astype(bool)
        for p0, p1 in zip(mk0[inlier_mask], mk1[inlier_mask]):
            x0, y0 = int(p0[0]), int(p0[1])
            x1, y1 = int(p1[0]) + W, int(p1[1])
            cv2.line(bottom, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)

    top = np.concatenate([top_l, top_r], axis=1)
    out = np.concatenate([top, bottom], axis=0)

    if H is not None and mask is not None:
        inliers = int(mask.sum())
        total = int(len(mk0))
        dx_px, dy_px, dyaw = decompose_homography(H, W, Himg)
        dx_m = dx_px * FAKE_ALTITUDE_M / focal_px
        dy_m = dy_px * FAKE_ALTITUDE_M / focal_px
        cov_xy = (1.0 / max(inliers, 1)) ** 2
        quality = min(255, inliers * 2)
        if inliers >= 30:
            lock, lock_col = 'LOCKED', (0, 255, 0)
        elif inliers >= 10:
            lock, lock_col = 'DEGRADED', (0, 200, 255)
        else:
            lock, lock_col = 'LOST', (0, 0, 255)
    else:
        inliers, total = 0, int(len(mk0)) if mk0 is not None else 0
        dx_m = dy_m = dyaw = 0.0
        cov_xy = 1.0
        quality = 0
        lock, lock_col = 'LOST', (0, 0, 255)

    put(out, 'Anchor Frame', (8, 22), (0, 200, 255))
    put(out, 'Target Frame', (W + 8, 22), (0, 200, 255))

    y = Himg + 26
    put(out, 'motion_delta  reference = ANCHOR', (8, y), (255, 255, 255), 0.6)
    put(out, f'lock_status: {lock}', (8, y + 26), lock_col, 0.6)
    put(out, f'matches: {total}', (8, y + 52))
    put(out, f'inliers: {inliers}', (8, y + 74))
    put(out, f'quality: {quality}/255', (8, y + 96))

    put(out, f'dx:   {dx_m:+.3f} m', (W + 8, y), (200, 255, 200), 0.6)
    put(out, f'dy:   {dy_m:+.3f} m', (W + 8, y + 26), (200, 255, 200), 0.6)
    put(out, f'dyaw: {math.degrees(dyaw):+.2f} deg', (W + 8, y + 52),
        (200, 255, 200), 0.6)
    put(out, f'cov:  [{cov_xy:.4f}, {cov_xy:.4f}]', (W + 8, y + 78))
    put(out, 'DEMO: handheld footage  (altitude assumed 1 m, HFOV 60 deg)',
        (8, out.shape[0] - 10), (160, 160, 160), 0.5)

    return out


def main():
    frames = extract_frames()
    print(f'{len(frames)} frames extracted')
    OUT_DIR.mkdir(exist_ok=True)
    for p in OUT_DIR.glob('*.png'):
        p.unlink()

    matcher = build_matcher()
    anchor = cv2.imread(str(frames[0]))
    Himg, W = anchor.shape[:2]
    focal_px = (W / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)

    # Warmup
    _ = matcher(anchor, anchor)

    for i, fp in enumerate(frames):
        target = cv2.imread(str(fp))
        H, mk0, mk1, mask = matcher(anchor, target)
        out = render(anchor, target, H, mk0, mk1, mask, focal_px)
        cv2.imwrite(str(OUT_DIR / f'f_{i:03d}.png'), out)
        if i % 10 == 0:
            inl = int(mask.sum()) if mask is not None else 0
            print(f'frame {i:3d}/{len(frames)}: matches={len(mk0):4d} '
                  f'inliers={inl:4d} H={"ok" if H is not None else "fail"}')

    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-framerate', '14',
         '-i', str(OUT_DIR / 'f_%03d.png'),
         '-vf', 'scale=720:-1:flags=lanczos,split[a][b];'
                '[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer',
         str(OUT_GIF)],
        check=True,
    )
    print(f'wrote {OUT_GIF}')


if __name__ == '__main__':
    main()
