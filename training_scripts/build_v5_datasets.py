"""
Builds the v5 landslide and flood training sets with features computed by
python-ml-service/feature_builder.py (the same code the live API uses).

Landslide: inventory points (class 1) vs random Assam points (class 0, 'random')
plus hard negatives from hilly terrain 1-20 km from any mapped slide ('hard').
Flood: SAR-derived labels from data/v3, with rainfall and soil moisture re-fetched
from the Open-Meteo archive using the same hourly windows as weather_service.py.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import httpx
from scipy.spatial import cKDTree
from shapely.geometry import Point

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python-ml-service'))
import feature_builder as fb  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'data', 'v5')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
rng = np.random.default_rng(42)


def add_terrain(df):
    rows = []
    for i, (lat, lon) in enumerate(zip(df.latitude, df.longitude)):
        rows.append(fb.terrain_features(lat, lon))
        if i % 250 == 0:
            print(f"  terrain {i}/{len(df)}")
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def hard_negatives(pos_utm, assam, n):
    tree = cKDTree(pos_utm)
    xmin, ymin = pos_utm.min(0) - 20000
    xmax, ymax = pos_utm.max(0) + 20000
    out = []
    while len(out) < n:
        cand = np.column_stack([rng.uniform(xmin, xmax, 4000), rng.uniform(ymin, ymax, 4000)])
        d, _ = tree.query(cand)
        cand = cand[(d > 1000) & (d < 20000)]
        pts = gpd.GeoSeries([Point(x, y) for x, y in cand], crs=32646).to_crs(4326)
        pts = pts[pts.within(assam)]
        for p in pts:
            if fb._point(fb._open(fb.SLOPE_PATH), p.x, p.y) >= 8:
                out.append((p.y, p.x))
            if len(out) >= n:
                break
    return pd.DataFrame(out, columns=['latitude', 'longitude'])


def near_road_negatives(pos_utm, assam, n):
    """Hilly points 0-600 m from a road, 1-20 km from any mapped slide."""
    tree = cKDTree(pos_utm)
    roads = np.load(fb.ROADS_PATH)['xy_utm46']
    lo, hi = pos_utm.min(0) - 20000, pos_utm.max(0) + 20000
    roads = roads[(roads[:, 0] > lo[0]) & (roads[:, 0] < hi[0]) & (roads[:, 1] > lo[1]) & (roads[:, 1] < hi[1])]
    out = []
    while len(out) < n:
        base = roads[rng.integers(0, len(roads), 4000)]
        ang, r = rng.uniform(0, 2 * np.pi, 4000), rng.uniform(0, 600, 4000)
        cand = base + np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        d, _ = tree.query(cand)
        cand = cand[(d > 1000) & (d < 20000)]
        pts = gpd.GeoSeries([Point(x, y) for x, y in cand], crs=32646).to_crs(4326)
        pts = pts[pts.within(assam)]
        for p in pts:
            if fb._point(fb._open(fb.SLOPE_PATH), p.x, p.y) >= 8:
                out.append((p.y, p.x))
            if len(out) >= n:
                break
    return pd.DataFrame(out, columns=['latitude', 'longitude'])


def road_matched(cands, pos, n):
    """
    The inventory was surveyed along roads, so slides sit close to them. Sample hard
    negatives with the same distance-to-road profile so the model cannot learn survey bias.
    """
    bins = [0, 100, 250, 500, 1000, 2000, 5000, np.inf]
    d_pos = np.array([fb.distance_to_road(a, b) for a, b in zip(pos.latitude, pos.longitude)])
    d_neg = np.array([fb.distance_to_road(a, b) for a, b in zip(cands.latitude, cands.longitude)])
    want = np.histogram(d_pos, bins)[0] / len(d_pos) * n
    b_neg = np.digitize(d_neg, bins) - 1
    picked = []
    for k, w in enumerate(want):
        idx = np.flatnonzero(b_neg == k)
        take = min(len(idx), int(round(w)))
        picked.extend(rng.choice(idx, take, replace=False))
        if take < round(w):
            print(f"  road bin {bins[k]}-{bins[k + 1]} m: only {take}/{int(round(w))} candidates")
    return cands.iloc[sorted(picked)].reset_index(drop=True)


def build_landslide():
    print("Building landslide dataset...")
    pos = gpd.read_file(os.path.join(ROOT, 'Processing', 'assam_landslides_utm46.gpkg'))
    pos['key'] = pos.geometry.apply(lambda g: (round(g.x), round(g.y)))
    pos = pos.drop_duplicates('key')
    pos_utm = np.column_stack([pos.geometry.x, pos.geometry.y])
    pos_ll = pos.to_crs(4326)
    df_pos = pd.DataFrame({'latitude': pos_ll.geometry.y, 'longitude': pos_ll.geometry.x,
                           'district': pos['district'].values, 'class': 1, 'neg_type': 'none'})

    neg = gpd.read_file(os.path.join(ROOT, 'Processing', 'non_landslide_points_v3.gpkg'))
    df_rand = pd.DataFrame({'latitude': neg.geometry.y, 'longitude': neg.geometry.x,
                            'district': None, 'class': 0, 'neg_type': 'random'})

    assam = gpd.read_file(os.path.join(ROOT, 'Processing', 'assam_background_area.gpkg')).to_crs(4326).geometry.iloc[0]
    cands = pd.concat([hard_negatives(pos_utm, assam, 4000), near_road_negatives(pos_utm, assam, 4000)],
                      ignore_index=True)
    df_hard = road_matched(cands, df_pos, 1200)
    df_hard['district'] = None
    df_hard['class'] = 0
    df_hard['neg_type'] = 'hard'

    df = pd.concat([df_pos, df_rand, df_hard], ignore_index=True)
    df = add_terrain(df)
    df = df.dropna(subset=['elevation', 'slope'])
    df.to_csv(os.path.join(OUT_DIR, 'landslide_v5.csv'), index=False)
    print(df.groupby(['class', 'neg_type']).size())


def fetch_archive(coords, start, end):
    """Hourly precipitation + topsoil moisture from the Open-Meteo archive, cached on disk."""
    results = {}
    batch = 50
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        cache = os.path.join(CACHE_DIR, f'archive_{start}_{end}_{i}.json')
        if os.path.exists(cache):
            with open(cache) as f:
                data = json.load(f)
        else:
            params = {
                'latitude': ','.join(f'{c[0]:.5f}' for c in chunk),
                'longitude': ','.join(f'{c[1]:.5f}' for c in chunk),
                'start_date': start, 'end_date': end,
                'hourly': 'precipitation,soil_moisture_0_to_7cm',
                'timezone': 'Asia/Kolkata',
            }
            for attempt in range(5):
                try:
                    r = httpx.get('https://archive-api.open-meteo.com/v1/archive', params=params, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    print(f"  retry {attempt}: {e}")
                    time.sleep(20 * (attempt + 1))
            else:
                raise RuntimeError("Open-Meteo archive unavailable")
            data = data if isinstance(data, list) else [data]
            with open(cache, 'w') as f:
                json.dump(data, f)
        for c, d in zip(chunk, data):
            results[c] = d['hourly']
        print(f"  weather {min(i + batch, len(coords))}/{len(coords)}")
    return results


def windows_at(hourly, when):
    """Same windows as weather_service.process_weather_data, ending at `when` (local time)."""
    times = pd.to_datetime(hourly['time'])
    idx = int(np.searchsorted(times.values, np.datetime64(when), side='right'))
    precip = np.array([p if p is not None else 0.0 for p in hourly['precipitation']], dtype=float)
    soil = hourly.get('soil_moisture_0_to_7cm') or []

    def last(days):
        return float(precip[max(0, idx - days * 24):idx].sum())

    rain = {'24h_mm': last(1), '3d_mm': last(3), '7d_mm': last(7), '14d_mm': last(14), '30d_mm': last(30)}
    sm = soil[idx - 1] if 0 < idx <= len(soil) else None
    return fb.rainfall_features(rain, sm)


def build_flood():
    print("Building flood dataset...")
    fl = pd.read_csv(os.path.join(ROOT, 'data', 'v3', 'flood_features_v3_advanced.csv'))
    df = fl[['event_id', 'event_date', 'latitude', 'longitude', 'flood_label']].copy()
    df['latitude'] = df.latitude.round(5)
    df['longitude'] = df.longitude.round(5)

    coords = sorted(set(zip(df.latitude, df.longitude)))
    terrain = {}
    for i, (lat, lon) in enumerate(coords):
        terrain[(lat, lon)] = fb.terrain_features(lat, lon)
        if i % 100 == 0:
            print(f"  terrain {i}/{len(coords)}")

    hourly = fetch_archive(coords, '2025-05-15', '2025-09-30')
    rows = []
    for r in df.itertuples():
        # Sentinel-1 passes over Assam around 17:30 IST; use rain up to that hour
        when = pd.Timestamp(r.event_date) + pd.Timedelta(hours=17)
        feats = dict(terrain[(r.latitude, r.longitude)])
        feats.update(windows_at(hourly[(r.latitude, r.longitude)], when))
        rows.append(feats)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    df['augmented'] = 0

    # Dry-season negatives: flood-prone locations in Jan-Feb, when Assam does not flood.
    # They teach the model that low-lying terrain alone is not a flood; used for training only.
    pos_coords = sorted(set(zip(df.latitude[df.flood_label == 1], df.longitude[df.flood_label == 1])))
    dry_hourly = fetch_archive(pos_coords, '2024-12-01', '2025-02-28')
    dry_dates = pd.date_range('2025-01-05', '2025-02-25', freq='D')
    aug = []
    for lat, lon in pos_coords:
        for d in rng.choice(dry_dates, 2, replace=False):
            feats = dict(terrain[(lat, lon)])
            feats.update(windows_at(dry_hourly[(lat, lon)], pd.Timestamp(d) + pd.Timedelta(hours=17)))
            aug.append({'event_id': 'DRY', 'event_date': str(pd.Timestamp(d).date()), 'latitude': lat,
                        'longitude': lon, 'flood_label': 0, **feats, 'augmented': 1})
    df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, 'flood_v5.csv'), index=False)
    print(df.groupby(['augmented', 'flood_label']).size())


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'landslide'):
        build_landslide()
    if which in ('all', 'flood'):
        build_flood()
    fb.close_all()
