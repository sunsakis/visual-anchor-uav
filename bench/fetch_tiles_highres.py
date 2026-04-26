#!/usr/bin/env python3
"""Fetch ESRI World Imagery tiles at a configurable zoom/grid.

Example: zoom 18 at 16×16 = 4096×4096 px, ~1.6 km footprint at 49°N
(~0.39 m/px native). Writes `<name>_z{zoom}.jpg` in /home/teo/Drone/bench/aerial/.
Same LOCATIONS table as fetch_tiles.py (zoom-16 version).

Usage:
  python3 fetch_tiles_highres.py --zoom 18 --grid 16 --only donbas_rural
  python3 fetch_tiles_highres.py --zoom 18 --grid 16   # all 6 locations
"""
import argparse
import io
import math
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

URL = ("https://services.arcgisonline.com/arcgis/rest/services/"
       "World_Imagery/MapServer/tile/{z}/{y}/{x}")

LOCATIONS = [
    ("kharkiv_urban",     49.9935, 36.2304),
    ("izyum_fields",      49.2070, 37.3161),
    ("kramatorsk_mixed",  48.7245, 37.5564),
    ("oskil_river",       49.5530, 37.6780),
    ("severodonetsk",     48.9482, 38.4925),
    ("donbas_rural",      48.4500, 38.2500),
]


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
            req = urllib.request.Request(url, headers={"User-Agent": "drone-bench/0.2"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise last


def fetch_location(name, lat, lon, zoom, grid, out_dir):
    cx, cy = deg2tile(lat, lon, zoom)
    x0, y0 = cx - grid // 2, cy - grid // 2
    canvas = Image.new("RGB", (256 * grid, 256 * grid))
    fails = 0
    t0 = time.time()
    for i in range(grid):
        for j in range(grid):
            try:
                tile = fetch_tile(zoom, x0 + i, y0 + j)
                canvas.paste(tile, (i * 256, j * 256))
            except Exception as e:
                fails += 1
                print(f"  {name}: tile ({x0+i},{y0+j}) failed: {e}", flush=True)
    out_path = out_dir / f"{name}_z{zoom}.jpg"
    canvas.save(out_path, "JPEG", quality=92)
    meters_per_px_at_lat = (
        40075016.686 * math.cos(math.radians(lat)) / (256 * 2 ** zoom)
    )
    footprint_m = 256 * grid * meters_per_px_at_lat
    print(f"saved {out_path.name}  [{grid*grid - fails}/{grid*grid} tiles, "
          f"{footprint_m:.0f} m footprint, {meters_per_px_at_lat:.2f} m/px native, "
          f"{time.time()-t0:.1f} s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=18)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--only", help="fetch a single location by name")
    args = ap.parse_args()

    out_dir = Path(__file__).parent / "aerial"
    out_dir.mkdir(exist_ok=True)
    locs = [loc for loc in LOCATIONS if args.only is None or loc[0] == args.only]
    if not locs:
        print(f"no location matches --only {args.only!r}", file=sys.stderr)
        sys.exit(1)
    for name, lat, lon in locs:
        fetch_location(name, lat, lon, args.zoom, args.grid, out_dir)


if __name__ == "__main__":
    main()
