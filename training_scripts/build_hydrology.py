"""
Derives hydrology rasters from the Copernicus GLO-30 DEM at ~90 m with pysheds:
flow accumulation, height above nearest drainage (HAND) and distance to stream.
Also converts the OpenStreetMap road export into a compact point set for
distance-to-road lookups.

Outputs (Processing/*.tif are git-ignored like the other rasters):
  Processing/assam_dem_90m.tif
  Processing/assam_flowacc_90m.tif
  Processing/assam_hand_90m.tif
  Processing/assam_dist_stream_90m.tif
  models/v5/osm_roads_points.npz
"""
import os
import sys
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage
from pyproj import Transformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC = os.path.join(ROOT, 'Processing')
STREAM_KM2 = 5.0  # drainage area that starts a stream


def downsample_dem():
    src_path = os.path.join(PROC, 'assam_glo30_dem.tif')
    out_path = os.path.join(PROC, 'assam_dem_90m.tif')
    with rasterio.open(src_path) as src:
        h, w = src.height // 3, src.width // 3
        dem = src.read(1, out_shape=(h, w), resampling=Resampling.average).astype('float32')
        transform = src.transform * src.transform.scale(src.width / w, src.height / h)
        profile = src.profile.copy()
    dem[~np.isfinite(dem)] = -9999
    profile.update(height=h, width=w, transform=transform, dtype='float32', nodata=-9999,
                   compress='deflate', tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(dem, 1)
    return out_path, profile


def write(name, arr, profile):
    with rasterio.open(os.path.join(PROC, name), 'w', **profile) as dst:
        dst.write(arr.astype('float32'), 1)
    print(f"  wrote {name}")


def hydrology():
    from pysheds.grid import Grid
    dem_path, profile = downsample_dem()
    print("  DEM downsampled to", profile['height'], 'x', profile['width'])
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    dem = grid.fill_pits(dem)
    dem = grid.fill_depressions(dem)
    dem = grid.resolve_flats(dem)
    fdir = grid.flowdir(dem)
    acc = grid.accumulation(fdir)
    print("  flow accumulation done")

    lat_mid = (profile['transform'].f + profile['transform'].f + profile['transform'].e * profile['height']) / 2
    px_y = abs(profile['transform'].e) * 111320.0
    px_x = profile['transform'].a * 111320.0 * np.cos(np.radians(lat_mid))
    cells = STREAM_KM2 * 1e6 / (px_x * px_y)
    streams = acc > cells

    hand = grid.compute_hand(fdir, dem, streams)
    hand = np.where(np.isfinite(np.asarray(hand)), np.asarray(hand), -9999)
    print("  HAND done")
    dist = ndimage.distance_transform_edt(~np.asarray(streams, dtype=bool), sampling=(px_y, px_x))

    write('assam_flowacc_90m.tif', np.asarray(acc), profile)
    write('assam_hand_90m.tif', hand, profile)
    write('assam_dist_stream_90m.tif', dist, profile)


def roads(spacing_m=150.0):
    src = os.path.join(ROOT, 'data', 'v5', 'external', 'osm_roads_assam.json')
    with open(src, encoding='utf-8') as f:
        ways = json.load(f)['elements']
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32646", always_xy=True)
    pts = []
    for w in ways:
        g = w.get('geometry')
        if not g:
            continue
        x, y = to_utm.transform([p['lon'] for p in g], [p['lat'] for p in g])
        x, y = np.asarray(x), np.asarray(y)
        for i in range(len(x) - 1):
            seg = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
            n = max(1, int(seg // spacing_m))
            t = np.arange(n) / n
            pts.append(np.column_stack([x[i] + t * (x[i + 1] - x[i]), y[i] + t * (y[i + 1] - y[i])]))
        pts.append(np.array([[x[-1], y[-1]]]))
    pts = np.unique(np.round(np.vstack(pts) / 10) * 10, axis=0).astype('float32')
    out = os.path.join(ROOT, 'models', 'v5', 'osm_roads_points.npz')
    np.savez_compressed(out, xy_utm46=pts)
    print(f"  roads: {len(ways)} ways -> {len(pts)} points")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'roads'):
        roads()
    if which in ('all', 'hydrology'):
        hydrology()
