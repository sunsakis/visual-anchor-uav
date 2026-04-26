#!/usr/bin/env python3
"""Fetch Google Maps satellite tiles for the same locations as ESRI tiles.

Second source with different capture dates and processing — gives real photometric
variance for cross-source matcher testing. For bench research use only.
"""
import io
import math
import time
import urllib.request
from pathlib import Path

from PIL import Image

URL = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

LOCATIONS = [
    ('kharkiv_urban',    49.9935, 36.2304),
    ('izyum_fields',     49.2070, 37.3161),
    ('kramatorsk_mixed', 48.7245, 37.5564),
    ('oskil_river',      49.5530, 37.6780),
    ('severodonetsk',    48.9482, 38.4925),
    ('donbas_rural',     48.4500, 38.2500),
]

ZOOM = 16
GRID = 4


def deg2tile(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def fetch_tile(z, x, y, retries=3):
    url = URL.format(z=z, x=x, y=y)
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (drone-bench/0.1)',
            })
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert('RGB')
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise last


def main():
    out_dir = Path(__file__).parent / 'aerial_google'
    out_dir.mkdir(exist_ok=True)
    for name, lat, lon in LOCATIONS:
        cx, cy = deg2tile(lat, lon, ZOOM)
        x0, y0 = cx - GRID // 2, cy - GRID // 2
        canvas = Image.new('RGB', (256 * GRID, 256 * GRID))
        fails = 0
        for i in range(GRID):
            for j in range(GRID):
                try:
                    tile = fetch_tile(ZOOM, x0 + i, y0 + j)
                    canvas.paste(tile, (i * 256, j * 256))
                except Exception as e:
                    fails += 1
                    print(f'  {name}: tile ({x0+i},{y0+j}) failed: {e}')
        out_path = out_dir / f'{name}.jpg'
        canvas.save(out_path, 'JPEG', quality=92)
        tag = ' (incomplete)' if fails else ''
        print(f'saved {out_path}{tag}  [{GRID*GRID - fails}/{GRID*GRID} tiles]')


if __name__ == '__main__':
    main()
