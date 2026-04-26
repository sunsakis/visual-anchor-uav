#!/usr/bin/env python3
"""Layer 1.5 bench: multi-source + spatial-overlap matcher tests.

Two tests that expose what Layer 1's synthetic warps couldn't:

  A. Cross-provider: ESRI vs Google tiles covering the same ground area.
     Different capture dates, sensors, post-processing. Ground truth
     homography = identity. Tests real photometric + appearance variance.

  B. Spatial overlap: same-source crops at known pixel offsets. Tests the
     actual anchor-lock scenario: drone translates, anchor shows as a
     partially-overlapping region of the current view.

Both matchers (ORB, SP+LG) run on both tests. Answers: does ORB hold under
real photometric variance, or does it collapse hard enough to justify the
SuperPoint+LightGlue compute cost?
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from layer1_harness import LightGlueMatcher, match_orb, reproj_error, HAVE_LIGHTGLUE


def cross_provider_cases(esri_dir, google_dir):
    for p in sorted(Path(esri_dir).glob('*.jpg')):
        q = Path(google_dir) / p.name
        if not q.exists():
            continue
        a = cv2.imread(str(p))
        b = cv2.imread(str(q))
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        H_true = np.eye(3, dtype=np.float64)
        yield {
            'test': 'cross_provider', 'anchor': p.name, 'variant': 'esri_vs_google',
            'dx': 0, 'dy': 0,
        }, a, b, H_true


def spatial_overlap_cases(src_dir, crop_size=512, offsets=None):
    if offsets is None:
        offsets = [
            (0, 0), (32, 0), (64, 0), (128, 0), (192, 0), (256, 0),
            (0, 64), (0, 128), (0, 256),
            (64, 64), (128, 128), (192, 192),
        ]
    for p in sorted(Path(src_dir).glob('*.jpg')):
        img = cv2.imread(str(p))
        if img is None:
            continue
        H_img, W_img = img.shape[:2]
        if W_img < crop_size * 2 or H_img < crop_size * 2:
            continue
        bx, by = (W_img - crop_size) // 2, (H_img - crop_size) // 2
        anchor = img[by:by + crop_size, bx:bx + crop_size]
        for dx, dy in offsets:
            x1, y1 = bx + dx, by + dy
            if x1 + crop_size > W_img or y1 + crop_size > H_img:
                continue
            live = img[y1:y1 + crop_size, x1:x1 + crop_size]
            H_true = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)
            yield {
                'test': 'spatial_overlap', 'anchor': p.name,
                'variant': f'dx{dx}_dy{dy}', 'dx': dx, 'dy': dy,
            }, anchor, live, H_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--esri', default='./aerial')
    ap.add_argument('--google', default='./aerial_google')
    ap.add_argument('--output', default='layer15_results.csv')
    ap.add_argument('--success-px', type=float, default=5.0)
    ap.add_argument('--min-inliers', type=int, default=15)  # looser than L1 since crops are smaller
    ap.add_argument('--crop-size', type=int, default=512)
    args = ap.parse_args()

    matchers = {'orb': match_orb}
    if HAVE_LIGHTGLUE:
        matchers['superpoint_lightglue'] = LightGlueMatcher().match
    else:
        print('WARNING: lightglue not installed; running ORB only.')

    all_cases = []
    all_cases += list(cross_provider_cases(args.esri, args.google))
    all_cases += list(spatial_overlap_cases(args.esri, crop_size=args.crop_size))
    # Also run spatial-overlap on Google tiles for redundancy
    all_cases += list(spatial_overlap_cases(args.google, crop_size=args.crop_size))

    print(f'{len(all_cases)} case×matcher pairs to run '
          f'({len(all_cases)} cases × {len(matchers)} matchers)')

    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'test', 'anchor', 'variant', 'dx', 'dy', 'matcher',
            'inliers', 'total_matches', 'reproj_err_px', 'success',
        ])
        w.writeheader()
        n = 0
        for meta, a, b, H_true in all_cases:
            Wimg, Himg = a.shape[1], a.shape[0]
            for name, fn in matchers.items():
                Hest, inl, tot = fn(a, b)
                err = reproj_error(Hest, H_true, Wimg, Himg)
                ok = Hest is not None and err < args.success_px and inl >= args.min_inliers
                w.writerow({
                    **meta, 'matcher': name,
                    'inliers': inl, 'total_matches': tot,
                    'reproj_err_px': f'{err:.2f}', 'success': int(ok),
                })
                n += 1
        print(f'wrote {n} rows to {args.output}')


if __name__ == '__main__':
    main()
