#!/usr/bin/env python3
"""Summarize layer15_results.csv — cross-provider vs spatial-overlap, per matcher."""
import csv
import math
from collections import defaultdict
from pathlib import Path


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r['dx'] = int(r['dx']); r['dy'] = int(r['dy'])
            r['inliers'] = int(r['inliers']); r['total_matches'] = int(r['total_matches'])
            try: r['reproj_err_px'] = float(r['reproj_err_px'])
            except ValueError: r['reproj_err_px'] = float('inf')
            r['success'] = int(r['success'])
            rows.append(r)
    return rows


def rate(rs): return 100.0 * sum(r['success'] for r in rs) / len(rs) if rs else 0.0


def precision(rs):
    errs = [r['reproj_err_px'] for r in rs if r['success'] and r['reproj_err_px'] < float('inf')]
    return sum(errs)/len(errs) if errs else float('nan')


def section(title):
    print('\n' + title); print('-' * len(title))


def main():
    rows = load(Path(__file__).parent / 'layer15_results.csv')
    print(f'{len(rows)} rows total')

    tests = sorted({r['test'] for r in rows})
    matchers = sorted({r['matcher'] for r in rows})
    anchors = sorted({r['anchor'] for r in rows})

    section('Overall success + precision by test × matcher')
    hdr = f'  {"test":<18}{"matcher":<24}{"success":>10}{"mean_reproj":>14}{"n":>6}'
    print(hdr)
    for t in tests:
        for m in matchers:
            rs = [r for r in rows if r['test'] == t and r['matcher'] == m]
            if not rs: continue
            print(f'  {t:<18}{m:<24}{rate(rs):>9.1f}%{precision(rs):>13.2f}px{len(rs):>6}')

    section('Cross-provider success by terrain × matcher (ESRI vs Google, identity GT)')
    hdr = f'  {"anchor":<28}' + ''.join(f'{m:>24}' for m in matchers)
    print(hdr)
    for a in anchors:
        line = f'  {a:<28}'
        for m in matchers:
            rs = [r for r in rows if r['test'] == 'cross_provider'
                  and r['anchor'] == a and r['matcher'] == m]
            if rs:
                ok = '✓' if rs[0]['success'] else '✗'
                err = rs[0]['reproj_err_px']
                inl = rs[0]['inliers']
                line += f'{ok} err={err:>5.1f}px inl={inl:>4}'.rjust(24)
            else:
                line += '—'.rjust(24)
        print(line)

    section('Spatial-overlap: success rate by pixel offset × matcher')
    offsets = sorted({(r['dx'], r['dy']) for r in rows if r['test'] == 'spatial_overlap'})
    hdr = f'  {"offset (px)":<14}{"overlap_frac":>14}' + ''.join(f'{m:>24}' for m in matchers)
    print(hdr)
    crop = 512
    for dx, dy in offsets:
        overlap = (1 - abs(dx)/crop) * (1 - abs(dy)/crop) if abs(dx) < crop and abs(dy) < crop else 0
        line = f'  ({dx:>3},{dy:>3})    {overlap*100:>12.0f}%'
        for m in matchers:
            rs = [r for r in rows if r['test'] == 'spatial_overlap'
                  and r['dx'] == dx and r['dy'] == dy and r['matcher'] == m]
            line += f'{rate(rs):>23.1f}%'
        print(line)

    section('Takeaway lines')
    for m in matchers:
        xp = [r for r in rows if r['test'] == 'cross_provider' and r['matcher'] == m]
        sp = [r for r in rows if r['test'] == 'spatial_overlap' and r['matcher'] == m]
        print(f'  {m:<24} cross-provider {rate(xp):>5.1f}%   spatial-overlap {rate(sp):>5.1f}%')


if __name__ == '__main__':
    main()
