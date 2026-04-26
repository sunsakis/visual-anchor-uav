#!/usr/bin/env python3
"""Layer 1 bench: SuperPoint+LightGlue vs ORB on synthetic aerial perturbations.

Install:
    pip install torch opencv-python numpy
    pip install git+https://github.com/cvg/LightGlue

Data: drop aerial JPEG/PNG tiles (Mapbox Satellite, Google Earth exports,
USGS 3DEP orthoimagery, OpenAerialMap) into ./aerial/ before running.

Run:
    python layer1_harness.py --anchors ./aerial --output layer1_results.csv
"""
import argparse
import csv
import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    HAVE_LIGHTGLUE = True
except ImportError:
    HAVE_LIGHTGLUE = False


def homography_tilt_yaw_scale(W, H, pitch_deg, roll_deg, yaw_deg, scale, f_px):
    cx, cy = W / 2, H / 2
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]], dtype=np.float64)
    pr, rr, yr = map(math.radians, (pitch_deg, roll_deg, yaw_deg))
    Rx = np.array([[1, 0, 0], [0, math.cos(pr), -math.sin(pr)], [0, math.sin(pr), math.cos(pr)]])
    Ry = np.array([[math.cos(rr), 0, math.sin(rr)], [0, 1, 0], [-math.sin(rr), 0, math.cos(rr)]])
    Rz = np.array([[math.cos(yr), -math.sin(yr), 0], [math.sin(yr), math.cos(yr), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    H_rot = K @ R @ np.linalg.inv(K)
    S = np.array([[scale, 0, cx * (1 - scale)], [0, scale, cy * (1 - scale)], [0, 0, 1]])
    return S @ H_rot


def apply_gamma(img, gamma):
    out = (img.astype(np.float32) / 255.0) ** (1.0 / gamma)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def add_shadow(img, strength, angle_deg=45):
    H, W = img.shape[:2]
    ang = math.radians(angle_deg)
    yy, xx = np.mgrid[0:H, 0:W]
    side = (xx - W / 2) * math.cos(ang) + (yy - H / 2) * math.sin(ang)
    mask = np.where(side > 0, strength, 0).astype(np.float32)
    return np.clip(img.astype(np.float32) * (1.0 - mask)[..., None], 0, 255).astype(np.uint8)


def match_orb(img_a, img_b):
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 8:
        return None, 0, len(good)
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H, int(mask.sum()) if mask is not None else 0, len(good)


class LightGlueMatcher:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    @torch.inference_mode()
    def match(self, img_a, img_b):
        def prep(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(self.device)

        fa = self.extractor.extract(prep(img_a))
        fb = self.extractor.extract(prep(img_b))
        out = self.matcher({'image0': fa, 'image1': fb})
        fa, fb, out = [rbd(x) for x in (fa, fb, out)]
        ka = fa['keypoints'].cpu().numpy()
        kb = fb['keypoints'].cpu().numpy()
        mi = out['matches'].cpu().numpy()
        if mi.shape[0] < 8:
            return None, 0, mi.shape[0]
        src, dst = ka[mi[:, 0]], kb[mi[:, 1]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        return H, int(mask.sum()) if mask is not None else 0, int(mi.shape[0])


def reproj_error(H_pred, H_true, W, H_img):
    if H_pred is None:
        return float('inf')
    corners = np.float32([[0, 0], [W, 0], [W, H_img], [0, H_img]]).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(corners, H_pred).reshape(-1, 2)
    t = cv2.perspectiveTransform(corners, H_true).reshape(-1, 2)
    return float(np.linalg.norm(p - t, axis=1).mean())


def perturbations():
    tilts = [0, 5, 10, 20, 30]
    scales = [0.7, 1.0, 1.4]
    yaws = [0, 30, 90]
    gammas = [1.0, 1.8]
    shadows = [0.0, 0.4]
    for t, s, y, g, sh in itertools.product(tilts, scales, yaws, gammas, shadows):
        yield {'tilt_deg': t, 'scale': s, 'yaw_deg': y, 'gamma': g, 'shadow': sh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchors', default='./aerial')
    ap.add_argument('--output', default='layer1_results.csv')
    ap.add_argument('--success-px', type=float, default=5.0)
    ap.add_argument('--min-inliers', type=int, default=30)
    args = ap.parse_args()

    matchers = {'orb': match_orb}
    if HAVE_LIGHTGLUE:
        matchers['superpoint_lightglue'] = LightGlueMatcher().match
    else:
        print('WARNING: lightglue not installed; running ORB only.')
        print('  install: pip install git+https://github.com/cvg/LightGlue')

    paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif'):
        paths.extend(Path(args.anchors).glob(ext))
    if not paths:
        raise SystemExit(f'no images in {args.anchors}')

    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'anchor', 'matcher', 'tilt_deg', 'scale', 'yaw_deg', 'gamma', 'shadow',
            'inliers', 'total_matches', 'reproj_err_px', 'success',
        ])
        w.writeheader()
        n = 0
        for p in sorted(paths):
            anchor = cv2.imread(str(p))
            if anchor is None:
                continue
            Himg, Wimg = anchor.shape[:2]
            f_px = 0.8 * Wimg  # rough pinhole focal matching ~65° HFOV for synthetic tilt warp

            for pert in perturbations():
                Htrue = homography_tilt_yaw_scale(
                    Wimg, Himg,
                    pitch_deg=pert['tilt_deg'], roll_deg=0,
                    yaw_deg=pert['yaw_deg'], scale=pert['scale'], f_px=f_px,
                )
                warped = cv2.warpPerspective(anchor, Htrue, (Wimg, Himg))
                if pert['gamma'] != 1.0:
                    warped = apply_gamma(warped, gamma=pert['gamma'])
                if pert['shadow'] > 0:
                    warped = add_shadow(warped, strength=pert['shadow'])

                for name, fn in matchers.items():
                    Hest, inl, tot = fn(anchor, warped)
                    err = reproj_error(Hest, Htrue, Wimg, Himg)
                    ok = Hest is not None and err < args.success_px and inl >= args.min_inliers
                    w.writerow({
                        'anchor': p.name, 'matcher': name,
                        'inliers': inl, 'total_matches': tot,
                        'reproj_err_px': f'{err:.2f}', 'success': int(ok),
                        **pert,
                    })
                    n += 1
        print(f'wrote {n} rows to {args.output}')


if __name__ == '__main__':
    main()
