#!/usr/bin/env python3
"""Summarize layer1_results.csv along the axes that matter for v0."""
import csv
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r['tilt_deg'] = float(r['tilt_deg'])
            r['scale'] = float(r['scale'])
            r['yaw_deg'] = float(r['yaw_deg'])
            r['gamma'] = float(r['gamma'])
            r['shadow'] = float(r['shadow'])
            r['inliers'] = int(r['inliers'])
            r['total_matches'] = int(r['total_matches'])
            try:
                r['reproj_err_px'] = float(r['reproj_err_px'])
            except ValueError:
                r['reproj_err_px'] = float('inf')
            r['success'] = int(r['success'])
            rows.append(r)
    return rows


def by(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def rate(rs):
    return 100.0 * sum(r['success'] for r in rs) / len(rs)


def reproj_success(rs):
    errs = [r['reproj_err_px'] for r in rs if r['success'] and r['reproj_err_px'] < float('inf')]
    return mean(errs) if errs else float('nan')


def section(title):
    print('\n' + title)
    print('-' * len(title))


def main():
    rows = load(Path(__file__).parent / 'layer1_results.csv')
    print(f'{len(rows)} rows total')

    section('Overall success rate by matcher')
    for (m,), rs in sorted(by(rows, 'matcher').items()):
        print(f'  {m:<22} {rate(rs):5.1f}%   mean-reproj-err-on-success {reproj_success(rs):5.2f} px   n={len(rs)}')

    section('Success rate by matcher x terrain (success %)')
    matchers = sorted({r['matcher'] for r in rows})
    anchors = sorted({r['anchor'] for r in rows})
    hdr = f'  {"terrain":<28}' + ''.join(f'{m:>22}' for m in matchers)
    print(hdr)
    for a in anchors:
        line = f'  {a:<28}'
        for m in matchers:
            rs = [r for r in rows if r['anchor'] == a and r['matcher'] == m]
            line += f'{rate(rs):>21.1f}%'
        print(line)

    section('Success rate by matcher x tilt (success %)')
    tilts = sorted({r['tilt_deg'] for r in rows})
    hdr = f'  {"tilt":<8}' + ''.join(f'{m:>22}' for m in matchers)
    print(hdr)
    for t in tilts:
        line = f'  {int(t):>3}°    '
        for m in matchers:
            rs = [r for r in rows if r['tilt_deg'] == t and r['matcher'] == m]
            line += f'{rate(rs):>21.1f}%'
        print(line)

    section('Success rate by matcher x gamma (photometric)')
    gammas = sorted({r['gamma'] for r in rows})
    hdr = f'  {"gamma":<8}' + ''.join(f'{m:>22}' for m in matchers)
    print(hdr)
    for g in gammas:
        line = f'  {g:<8}'
        for m in matchers:
            rs = [r for r in rows if r['gamma'] == g and r['matcher'] == m]
            line += f'{rate(rs):>21.1f}%'
        print(line)

    section('Success rate by matcher x shadow')
    shads = sorted({r['shadow'] for r in rows})
    hdr = f'  {"shadow":<8}' + ''.join(f'{m:>22}' for m in matchers)
    print(hdr)
    for s in shads:
        line = f'  {s:<8}'
        for m in matchers:
            rs = [r for r in rows if r['shadow'] == s and r['matcher'] == m]
            line += f'{rate(rs):>21.1f}%'
        print(line)

    section('Success rate by matcher x scale')
    scales = sorted({r['scale'] for r in rows})
    hdr = f'  {"scale":<8}' + ''.join(f'{m:>22}' for m in matchers)
    print(hdr)
    for s in scales:
        line = f'  {s:<8}'
        for m in matchers:
            rs = [r for r in rows if r['scale'] == s and r['matcher'] == m]
            line += f'{rate(rs):>21.1f}%'
        print(line)

    section('SP+LG worst-case terrain: success rate by tilt on donbas_rural')
    rural = [r for r in rows if r['anchor'] == 'donbas_rural.jpg' and r['matcher'] == 'superpoint_lightglue']
    for t in tilts:
        rs = [r for r in rural if r['tilt_deg'] == t]
        print(f'  tilt {int(t):>3}°:  {rate(rs):5.1f}%  (n={len(rs)})')

    section('SP+LG reprojection precision on successful cases (px)')
    for a in anchors:
        rs = [r for r in rows if r['anchor'] == a and r['matcher'] == 'superpoint_lightglue' and r['success']]
        errs = [r['reproj_err_px'] for r in rs]
        if errs:
            print(f'  {a:<28} mean {mean(errs):5.2f}  min {min(errs):5.2f}  max {max(errs):5.2f}  n={len(errs)}')


if __name__ == '__main__':
    main()
