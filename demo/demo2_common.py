"""Shared render + math for demo #2 (no ROS imports)."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import math

import cv2
import numpy as np

# Camera: v0 BOM baseline — 1280x800, 110° HFOV, rectilinear pinhole
CAM_W, CAM_H, CAM_HFOV_DEG = 1280, 800, 110.0
ALT_M = 100.0
TRAJ_RADIUS_M = 30.0
TRAJ_N = 60
TRAJ_FPS = 12


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


def pixels_to_world(dx_px, dy_px, focal_px, altitude_m):
    """Gazebo's down-pitched camera rotates image axes 90° vs world:
    calibrated empirically — world +x (east) shows up as image +v,
    world +y (north) shows up as image +u."""
    dx_m = +dy_px * altitude_m / focal_px
    dy_m = +dx_px * altitude_m / focal_px
    return dx_m, dy_m


def put(img, txt, org, col=(255, 255, 255), scale=0.55, th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th,
                cv2.LINE_AA)


def draw_minimap(panel, traj_xy_truth, traj_xy_est, cur_idx,
                 origin, half_size, scale_m):
    cx, cy = origin
    cv2.rectangle(panel, (cx - half_size, cy - half_size),
                  (cx + half_size, cy + half_size), (80, 80, 80), 1)
    cv2.line(panel, (cx - half_size, cy), (cx + half_size, cy), (60, 60, 60), 1)
    cv2.line(panel, (cx, cy - half_size), (cx, cy + half_size), (60, 60, 60), 1)

    def to_px(x_m, y_m):
        px = int(cx + (x_m / scale_m) * half_size)
        py = int(cy - (y_m / scale_m) * half_size)
        return px, py

    for i in range(1, len(traj_xy_truth)):
        p0 = to_px(*traj_xy_truth[i - 1])
        p1 = to_px(*traj_xy_truth[i])
        cv2.line(panel, p0, p1, (255, 200, 0), 1, cv2.LINE_AA)
    for i in range(1, min(cur_idx + 1, len(traj_xy_est))):
        if traj_xy_est[i] is None or traj_xy_est[i - 1] is None:
            continue
        p0 = to_px(*traj_xy_est[i - 1])
        p1 = to_px(*traj_xy_est[i])
        cv2.line(panel, p0, p1, (0, 255, 0), 2, cv2.LINE_AA)
    ax, ay = to_px(0.0, 0.0)
    cv2.drawMarker(panel, (ax, ay), (0, 200, 255), cv2.MARKER_TRIANGLE_UP, 10, 2)
    if cur_idx < len(traj_xy_truth):
        tp = to_px(*traj_xy_truth[cur_idx])
        cv2.circle(panel, tp, 6, (255, 200, 0), 2)
        if cur_idx < len(traj_xy_est) and traj_xy_est[cur_idx] is not None:
            ep = to_px(*traj_xy_est[cur_idx])
            cv2.circle(panel, ep, 5, (0, 255, 0), -1)
    put(panel, "trajectory (top-down)",
        (cx - half_size, cy - half_size - 6), (200, 200, 200), 0.45)
    put(panel, "gt (cyan)   est (green)",
        (cx - half_size, cy + half_size + 16), (150, 150, 150), 0.4)


def render_frame(anchor_bgr, live_bgr, H, mk0, mk1, mask,
                 focal_px, truth_xy, traj_truth, traj_est, cur_idx):
    DISP_W, DISP_H = 480, 300
    a = cv2.resize(anchor_bgr, (DISP_W, DISP_H))
    l = cv2.resize(live_bgr, (DISP_W, DISP_H))
    top_l, top_r = a.copy(), l.copy()

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

    top = np.concatenate([top_l, top_r], axis=1)

    match_panel = np.zeros((360, 640, 3), dtype=np.uint8)
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

    tel = np.zeros((360, 320, 3), dtype=np.uint8)
    if H is not None and mask is not None:
        inliers = int(mask.sum())
        total = int(len(mk0))
        dx_px, dy_px, dyaw = decompose(H, CAM_W, CAM_H)
        dx_m, dy_m = pixels_to_world(dx_px, dy_px, focal_px, ALT_M)
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

    traj_est[cur_idx] = (dx_m, dy_m) if H is not None else None

    bottom = np.concatenate([match_panel, tel], axis=1)
    out = np.concatenate([top, bottom], axis=0)

    put(out, f"altitude {ALT_M:.0f} m AGL  |  {CAM_W}x{CAM_H} @ {CAM_HFOV_DEG:.0f}° HFOV  |  "
             "texture: donbas_rural z18 (Ukraine aerial)",
        (8, out.shape[0] - 10), (160, 160, 160), 0.45)
    return out
