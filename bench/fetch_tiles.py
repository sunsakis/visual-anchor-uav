#!/usr/bin/env python3
"""Fetch aerial tiles from ESRI World Imagery for bench test.

ESRI World Imagery tiles are free to use with attribution for non-commercial
testing. Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community.

Terrain variety is intentional — a matcher that passes on fields but fails on
forest is a narrow-domain result, not a go signal.
"""
import io
import math
import time
import urllib.request
from pathlib import Path

from PIL import Image

URL = 'https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'

LOCATIONS = [
    ('kharkiv_urban',      49.9935, 36.2304),   # dense city blocks
    ('izyum_fields',       49.2070, 37.3161),   # agricultural patchwork
    ('kramatorsk_mixed',   48.7245, 37.5564),   # suburban + industrial
    ('oskil_river',        49.5530, 37.6780),   # river + riparian forest
    ('severodonetsk',      48.9482, 38.4925),   # riverside town, mixed texture
    ('donbas_rural',       48.4500, 38.2500),   # open steppe + villages
]

ZOOM = 16      # ~2.4 m/px at these latitudes
GRID = 4       # 4x4 tiles -> 1024x1024 image (~2.5 km wide)


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
            req = urllib.request.Request(url, headers={'User-Agent': 'drone-bench/0.1'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert('RGB')
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise last


def main():
    out_dir = Path(__file__).parent / 'aerial'
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
