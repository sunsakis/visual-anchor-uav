#!/usr/bin/env python3
"""Summarize layer15b_results.csv — success + inference_ms per matcher.

Gate: is XFeat ≥90% success AND ≥10× faster than SP+LG?
"""
import csv
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
            r['inference_ms'] = float(r['inference_ms'])
            rows.append(r)
    return rows


def rate(rs): return 100.0 * sum(r['success'] for r in rs) / len(rs) if rs else 0.0
def mean(xs): return sum(xs) / len(xs) if xs else float('nan')


def precision(rs):
    errs = [r['reproj_err_px'] for r in rs if r['success'] and r['reproj_err_px'] < float('inf')]
    return mean(errs) if errs else float('nan')


def section(t):
    print('\n' + t); print('-' * len(t))


def main():
    rows = load(Path(__file__).parent / 'layer15b_results.csv')
    print(f'{len(rows)} rows total')

    matchers = sorted({r['matcher'] for r in rows})
    tests = sorted({r['test'] for r in rows})
    anchors = sorted({r['anchor'] for r in rows})

    section('Success + mean inference time by test × matcher')
    hdr = f'  {"test":<18}{"matcher":<24}{"success":>10}{"reproj":>12}{"inf_ms":>10}{"n":>6}'
    print(hdr)
    for t in tests:
        for m in matchers:
            rs = [r for r in rows if r['test'] == t and r['matcher'] == m]
            if not rs: continue
            inf = mean([r['inference_ms'] for r in rs])
            print(f'  {t:<18}{m:<24}{rate(rs):>9.1f}%{precision(rs):>11.2f}px{inf:>9.1f}{len(rs):>6}')

    section('Overall: success + inference_ms + speedup vs SP+LG')
    sp_rows = [r for r in rows if r['matcher'] == 'superpoint_lightglue']
    sp_ms = mean([r['inference_ms'] for r in sp_rows]) if sp_rows else float('nan')
    for m in matchers:
        rs = [r for r in rows if r['matcher'] == m]
        ms = mean([r['inference_ms'] for r in rs])
        speedup = sp_ms / ms if ms > 0 else float('nan')
        print(f'  {m:<24} success {rate(rs):>5.1f}%   mean inference {ms:>6.1f} ms   speedup×SP+LG {speedup:>5.2f}')

    section('Cross-provider per terrain (success / inliers / inf_ms)')
    hdr = f'  {"anchor":<28}' + ''.join(f'{m:>26}' for m in matchers)
    print(hdr)
    for a in anchors:
        line = f'  {a:<28}'
        for m in matchers:
            rs = [r for r in rows if r['test'] == 'cross_provider' and r['anchor'] == a and r['matcher'] == m]
            if rs:
                r = rs[0]
                ok = 'OK' if r['success'] else 'NO'
                cell = f'{ok} inl={r["inliers"]:>4} {r["inference_ms"]:>5.0f}ms'
            else:
                cell = '—'
            line += cell.rjust(26)
        print(line)

    section('Spatial-overlap: success by offset × matcher')
    offsets = sorted({(r['dx'], r['dy']) for r in rows if r['test'] == 'spatial_overlap'})
    crop = 512
    hdr = f'  {"offset":<14}{"overlap":>10}' + ''.join(f'{m:>18}' for m in matchers)
    print(hdr)
    for dx, dy in offsets:
        ov = (1 - abs(dx)/crop) * (1 - abs(dy)/crop) if abs(dx) < crop and abs(dy) < crop else 0
        line = f'  ({dx:>3},{dy:>3})    {ov*100:>8.0f}%'
        for m in matchers:
            rs = [r for r in rows if r['test'] == 'spatial_overlap'
                  and r['dx'] == dx and r['dy'] == dy and r['matcher'] == m]
            line += f'{rate(rs):>17.1f}%'
        print(line)

    section('NPU/MCU gate')
    xf = [r for r in rows if r['matcher'] == 'xfeat']
    xf_success = rate(xf)
    xf_ms = mean([r['inference_ms'] for r in xf])
    orb = [r for r in rows if r['matcher'] == 'orb']
    orb_ms = mean([r['inference_ms'] for r in orb])
    speedup = sp_ms / xf_ms if xf_ms > 0 else float('nan')
    print(f'  XFeat success      : {xf_success:5.1f}% (gate: >=90%)')
    print(f'  XFeat vs SP+LG     : {speedup:5.2f}x faster (gate: >=10x)')
    print(f'  XFeat vs ORB       : {orb_ms / xf_ms if xf_ms > 0 else float("nan"):5.2f}x')
    door_open = xf_success >= 90 and speedup >= 10
    print(f'  NPU/MCU door       : {"OPEN" if door_open else "CLOSED"}')


if __name__ == '__main__':
    main()
