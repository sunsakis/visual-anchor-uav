#!/usr/bin/env python3
"""Layer 1.5b: same cases as L1.5 plus XFeat, with inference_ms per call.

Purpose: decide whether XFeat's lighter compute footprint opens the NPU/MCU
door. Gate: ≥90% success AND ≥10× faster than SP+LG on the same cases.

Runs ORB, SP+LG, and XFeat end-to-end on cross-provider + spatial-overlap
cases and emits per-call wall-clock time for the full matcher function
(extraction + matching + homography).
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from layer1_harness import LightGlueMatcher, match_orb, reproj_error, HAVE_LIGHTGLUE
from layer15_harness import cross_provider_cases, spatial_overlap_cases


class XFeatMatcher:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xfeat = torch.hub.load(
            'verlab/accelerated_features', 'XFeat',
            pretrained=True, top_k=4096, trust_repo=True,
        )

    @torch.inference_mode()
    def match(self, img_a, img_b):
        mk0, mk1 = self.xfeat.match_xfeat(img_a, img_b, top_k=4096)
        if len(mk0) < 8:
            return None, 0, int(len(mk0))
        H, mask = cv2.findHomography(mk0, mk1, cv2.RANSAC, 3.0)
        return H, int(mask.sum()) if mask is not None else 0, int(len(mk0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--esri', default='./aerial')
    ap.add_argument('--google', default='./aerial_google')
    ap.add_argument('--output', default='layer15b_results.csv')
    ap.add_argument('--success-px', type=float, default=5.0)
    ap.add_argument('--min-inliers', type=int, default=15)
    ap.add_argument('--crop-size', type=int, default=512)
    args = ap.parse_args()

    matchers = {'orb': match_orb}
    if HAVE_LIGHTGLUE:
        matchers['superpoint_lightglue'] = LightGlueMatcher().match
    else:
        print('WARNING: lightglue not installed; running ORB only.')
    matchers['xfeat'] = XFeatMatcher().match

    all_cases = []
    all_cases += list(cross_provider_cases(args.esri, args.google))
    all_cases += list(spatial_overlap_cases(args.esri, crop_size=args.crop_size))
    all_cases += list(spatial_overlap_cases(args.google, crop_size=args.crop_size))

    # Per-matcher warmup so timings aren't polluted by first-call allocator hits.
    if all_cases:
        _, wa, wb, _ = all_cases[0]
        for name, fn in matchers.items():
            try:
                fn(wa, wb)
            except Exception as e:
                print(f'warmup {name} failed: {e}')

    print(f'{len(all_cases)} cases × {len(matchers)} matchers '
          f'= {len(all_cases) * len(matchers)} rows')

    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'test', 'anchor', 'variant', 'dx', 'dy', 'matcher',
            'inliers', 'total_matches', 'reproj_err_px', 'success',
            'inference_ms',
        ])
        w.writeheader()
        n = 0
        for meta, a, b, H_true in all_cases:
            Wimg, Himg = a.shape[1], a.shape[0]
            for name, fn in matchers.items():
                t0 = time.perf_counter()
                Hest, inl, tot = fn(a, b)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                err = reproj_error(Hest, H_true, Wimg, Himg)
                ok = Hest is not None and err < args.success_px and inl >= args.min_inliers
                w.writerow({
                    **meta, 'matcher': name,
                    'inliers': inl, 'total_matches': tot,
                    'reproj_err_px': f'{err:.2f}', 'success': int(ok),
                    'inference_ms': f'{dt_ms:.1f}',
                })
                n += 1
        print(f'wrote {n} rows to {args.output}')


if __name__ == '__main__':
    main()
