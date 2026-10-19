"""
Generates flood / no-flood labels for the 2019-2024 monsoons from Sentinel-1 RTC
backscatter (Microsoft Planetary Computer, anonymous access), using the same
temporal-inundation idea as the 2025 SAR labels:

  flooded     : VV(monsoon) < -18 dB, VV(dry season) > -15 dB, JRC water occurrence < 30 %
  not flooded : VV(monsoon) > -14 dB and VV(dry season) > -15 dB
  otherwise   : dropped (ambiguous or permanent water)

Dry-season scenes also yield observed no-flood rows. Features come from
feature_builder (terrain + hydrology) and the Open-Meteo archive (rain windows
ending at the acquisition time), exactly like the live API.

Output: data/v5/flood_s1_2019_2024.csv
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Point

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python-ml-service'))
sys.path.insert(0, os.path.dirname(__file__))
import feature_builder as fb  # noqa: E402
from build_v5_datasets import fetch_archive, windows_at  # noqa: E402

OUT = os.path.join(ROOT, 'data', 'v5', 'flood_s1_2019_2024.csv')
CACHE = os.path.join(ROOT, 'data', 'v5', 'cache', 's1')
os.makedirs(CACHE, exist_ok=True)
rng = np.random.default_rng(11)

YEARS = range(2019, 2025)
MONSOON_WINDOWS = [('06-10', '06-22'), ('07-01', '07-13'), ('07-16', '07-28'), ('08-10', '08-22')]
DRY_WINDOW = ('01-15', '02-28')
TILE_DEG = 0.036          # ~4 km
POINTS_PER_TILE = 30
PIX = 200                 # read tiles at ~20 m


def catalog():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                     modifier=planetary_computer.sign_inplace)


def pick_tiles(n_low=45, n_up=15):
    assam = gpd.read_file(os.path.join(ROOT, 'Processing', 'assam_background_area.gpkg')).to_crs(4326).geometry.iloc[0]
    hand_ds = fb._open(fb.HAND_PATH)
    tiles, low, up = [], 0, 0
    while low < n_low or up < n_up:
        lat, lon = rng.uniform(24.2, 27.9), rng.uniform(89.8, 95.9)
        if not assam.contains(Point(lon, lat)):
            continue
        if any((lat - a) ** 2 + (lon - b) ** 2 < 0.13 ** 2 for a, b, _ in tiles):
            continue
        hand = fb._point(hand_ds, lon, lat)
        if not np.isfinite(hand):
            continue
        if hand < 20 and low < n_low:
            tiles.append((lat, lon, 'lowland')); low += 1
        elif 20 <= hand < 120 and up < n_up:
            tiles.append((lat, lon, 'upland')); up += 1
    return tiles


def read_db(href, bbox):
    """Mean-resampled VV (dB) over a lon/lat bbox, PIX x PIX."""
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=4, GDAL_HTTP_RETRY_DELAY=5):
        with rasterio.open(href) as src:
            b = transform_bounds('EPSG:4326', src.crs, *bbox)
            win = from_bounds(*b, src.transform)
            arr = src.read(1, window=win, out_shape=(PIX, PIX), boundless=True, fill_value=0,
                           resampling=rasterio.enums.Resampling.average).astype('float64')
    arr[arr <= 0] = np.nan
    return 10 * np.log10(arr)


def _search(client, **kw):
    """STAC search with retries; Planetary Computer returns occasional 502s."""
    for attempt in range(6):
        try:
            return list(client.search(**kw).items())
        except Exception as e:
            print(f"  search retry {attempt}: {str(e)[:60]}")
            time.sleep(15 * (attempt + 1))
    return []


def scene_for(client, bbox, start, end):
    items = _search(client, collections=['sentinel-1-rtc'], bbox=bbox, datetime=f'{start}/{end}', max_items=20)
    items = [i for i in items if 'vv' in i.assets]
    if not items:
        return None
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    return min(items, key=lambda i: abs(pd.Timestamp(i.datetime).tz_localize(None) - mid))


def jrc_occurrence(client, bbox):
    items = _search(client, collections=['jrc-gsw'], bbox=bbox, max_items=1)
    if not items:
        return None
    with rasterio.open(items[0].assets['occurrence'].href) as src:
        win = from_bounds(*bbox, src.transform)
        arr = src.read(1, window=win, out_shape=(PIX, PIX), boundless=True, fill_value=0).astype(float)
    arr[arr > 100] = 0
    return arr


def process_tile(args):
    k, (lat, lon, kind) = args
    cache = os.path.join(CACHE, f'tile_{k}.json')
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    client = catalog()
    h = TILE_DEG / 2
    bbox = (lon - h, lat - h, lon + h, lat + h)
    occ = jrc_occurrence(client, bbox)
    pts = rng.integers(2, PIX - 2, size=(POINTS_PER_TILE, 2))
    out = {'tile': k, 'lat': lat, 'lon': lon, 'kind': kind, 'points': [], 'obs': []}
    for r, c in pts:
        out['points'].append({'row': int(r), 'col': int(c),
                              'lat': lat + h - (r + 0.5) / PIX * TILE_DEG,
                              'lon': lon - h + (c + 0.5) / PIX * TILE_DEG,
                              'jrc': float(occ[r, c]) if occ is not None else None})

    def sample(img):
        return [float(np.nanmedian(img[r - 1:r + 2, c - 1:c + 2])) for r, c in pts]

    for y in YEARS:
        dry = scene_for(client, bbox, f'{y}-{DRY_WINDOW[0]}', f'{y}-{DRY_WINDOW[1]}')
        if dry is None:
            continue
        try:
            vv_dry = sample(read_db(dry.assets['vv'].href, bbox))
        except Exception as e:
            print(f"  tile {k} {y} dry read failed: {str(e)[:60]}")
            continue
        out['obs'].append({'date': str(dry.datetime), 'kind': 'dry', 'vv': vv_dry, 'vv_dry': vv_dry})
        for a, b in MONSOON_WINDOWS:
            it = scene_for(client, bbox, f'{y}-{a}', f'{y}-{b}')
            if it is None:
                continue
            try:
                vv = sample(read_db(it.assets['vv'].href, bbox))
            except Exception as e:
                print(f"  tile {k} {it.id} read failed: {str(e)[:60]}")
                continue
            out['obs'].append({'date': str(it.datetime), 'kind': 'monsoon', 'vv': vv, 'vv_dry': vv_dry})
    with open(cache, 'w') as f:
        json.dump(out, f)
    print(f"  tile {k} ({kind}) done: {len(out['obs'])} scenes")
    return out


def label(vv, vv_dry, jrc):
    if not (np.isfinite(vv) and np.isfinite(vv_dry)) or vv_dry <= -15 or (jrc is not None and jrc >= 30):
        return None
    if vv < -18:
        return 1
    if vv > -14:
        return 0
    return None


def main():
    tiles_path = os.path.join(CACHE, 'tiles.json')
    if os.path.exists(tiles_path):
        tiles = [tuple(t) for t in json.load(open(tiles_path))]
    else:
        tiles = pick_tiles()
        json.dump(tiles, open(tiles_path, 'w'))
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(process_tile, enumerate(tiles)))

    rows = []
    for t in results:
        for o in t['obs']:
            when = pd.Timestamp(o['date']).tz_convert('Asia/Kolkata').tz_localize(None)
            for p, vv, vd in zip(t['points'], o['vv'], o['vv_dry']):
                lab = label(vv, vd, p['jrc'])
                if lab is None:
                    continue
                rows.append({'tile': t['tile'], 'tile_kind': t['kind'], 'tile_lat': round(t['lat'], 2),
                             'tile_lon': round(t['lon'], 2), 'latitude': p['lat'], 'longitude': p['lon'],
                             'acq_time': when, 'event_date': str(when.date()), 'season': o['kind'],
                             'vv_db': vv, 'vv_dry_db': vd, 'flood_label': lab})
    df = pd.DataFrame(rows)
    print("labels:", df.groupby(['season', 'flood_label']).size().to_dict())

    feats = {}
    for i, (la, lo) in enumerate(sorted(set(zip(df.latitude.round(5), df.longitude.round(5))))):
        feats[(la, lo)] = fb.terrain_features(la, lo)
        if i % 300 == 0:
            print(f"  terrain {i}")
    terrain = pd.DataFrame([feats[(a, b)] for a, b in zip(df.latitude.round(5), df.longitude.round(5))])

    rain_rows = []
    for y in sorted(df.acq_time.dt.year.unique()):
        sub = df[df.acq_time.dt.year == y]
        cells = sorted(set(zip(sub.tile_lat, sub.tile_lon)))
        hourly = fetch_archive(cells, f'{y}-01-01', f'{y}-09-30')
        for i, r in sub.iterrows():
            rain_rows.append((i, windows_at(hourly[(r.tile_lat, r.tile_lon)], r.acq_time)))
    rain = pd.DataFrame([r for _, r in sorted(rain_rows)], index=[i for i, _ in sorted(rain_rows)])

    df = pd.concat([df, terrain.set_index(df.index), rain], axis=1)
    df['augmented'] = 0
    df['source'] = 'S1_RTC_2019_2024'
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows")


if __name__ == '__main__':
    main()
