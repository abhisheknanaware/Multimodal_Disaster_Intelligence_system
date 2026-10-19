"""
Builds a dated landslide rainfall-trigger dataset (case-control in time).

Events: dated slides parsed from the GSI Assam inventory text, plus rain-triggered
NE India events from the NASA Global Landslide Catalog (2007-2016, location
accuracy <= 25 km). For each event cell the event day is a positive and ten random
monsoon-season days of the same year (>= 10 days from any event in that cell) are
negatives, so the model learns which weather triggers slides, not which places.
Weather comes from the Open-Meteo archive (daily).
"""
import os
import re
import sys
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import httpx
from dateutil import parser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python-ml-service'))
import feature_builder as fb  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'data', 'v5')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
rng = np.random.default_rng(7)

MONTH = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
PATTERNS = [
    (rf'(\d{{1,2}})(?:st|nd|rd|th)?[ ,.-]*({MONTH})[ ,.-]*((?:19|20)\d\d)', lambda a, b, c: f'{a} {b} {c}'),
    (rf'({MONTH})[ ,.-]*(\d{{1,2}})(?:st|nd|rd|th)?[ ,.-]+((?:19|20)\d\d)', lambda a, b, c: f'{b} {a} {c}'),
    (r'(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d\d)', lambda a, b, c: f'{a}/{b}/{c}'),
]


def parse_date(text):
    for pat, fmt in PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            try:
                return parser.parse(fmt(*m.groups()), dayfirst=True).date()
            except Exception:
                return None
    return None


def gsi_events():
    g = gpd.read_file(os.path.join(ROOT, 'Processing', 'assam_landslides_utm46.gpkg')).to_crs(4326)
    text = (g.source_text_before_coordinates.astype(str) + ' ' +
            g.source_text_after_coordinates.astype(str)).str.replace('�', ' ')
    dates = text.apply(parse_date)
    df = pd.DataFrame({'latitude': g.geometry.y, 'longitude': g.geometry.x,
                       'event_date': pd.to_datetime(dates), 'source': 'GSI'})
    return df.dropna(subset=['event_date'])


def glc_events():
    d = pd.read_csv(os.path.join(ROOT, 'data', 'v5', 'external', 'nasa_glc.csv'))
    d = d[d.latitude.between(21.5, 29.5) & d.longitude.between(88, 97.5)]
    d = d[d.landslide_trigger.isin(['downpour', 'rain', 'continuous_rain', 'monsoon', 'flooding'])]
    d = d[d.location_accuracy.isin(['exact', '1km', '5km', '10km', '25km'])]
    return pd.DataFrame({'latitude': d.latitude, 'longitude': d.longitude,
                         'event_date': pd.to_datetime(d.event_date, format='%m/%d/%Y %I:%M:%S %p').dt.normalize(),
                         'source': 'NASA_GLC'})


def fetch_year(cells, year):
    """Daily weather Mar 1 - Nov 30 for a list of (lat, lon) cells, cached."""
    out = {}
    for i in range(0, len(cells), 50):
        chunk = cells[i:i + 50]
        cache = os.path.join(CACHE_DIR, f'trigger_{year}_{i}_{len(chunk)}.json')
        if os.path.exists(cache):
            with open(cache) as f:
                data = json.load(f)
        else:
            params = {
                'latitude': ','.join(f'{c[0]:.2f}' for c in chunk),
                'longitude': ','.join(f'{c[1]:.2f}' for c in chunk),
                'start_date': f'{year}-03-01', 'end_date': f'{year}-11-30',
                'daily': 'precipitation_sum,precipitation_hours,soil_moisture_0_to_7cm_mean',
                'timezone': 'Asia/Kolkata',
            }
            for attempt in range(6):
                try:
                    r = httpx.get('https://archive-api.open-meteo.com/v1/archive', params=params, timeout=180)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    print(f"  retry {attempt}: {str(e)[:80]}")
                    time.sleep(30 * (attempt + 1))
            else:
                raise RuntimeError("Open-Meteo archive unavailable")
            data = data if isinstance(data, list) else [data]
            with open(cache, 'w') as f:
                json.dump(data, f)
        for c, d in zip(chunk, data):
            out[c] = pd.DataFrame(d['daily']).assign(time=lambda x: pd.to_datetime(x.time)).set_index('time')
    return out


def windows(daily, day):
    """Rain windows ending on `day` (inclusive), matching the live last-24h/3d/... sums."""
    p = daily.precipitation_sum.fillna(0.0)
    i = daily.index.get_loc(day)

    def last(n):
        return float(p.iloc[max(0, i - n + 1):i + 1].sum())

    rain = {'24h_mm': last(1), '3d_mm': last(3), '7d_mm': last(7), '14d_mm': last(14), '30d_mm': last(30)}
    feats = fb.rainfall_features(rain, daily.soil_moisture_0_to_7cm_mean.iloc[i])
    feats['rain_hours_1d'] = float(daily.precipitation_hours.fillna(0).iloc[i])
    feats['rain_max_1d_7d'] = float(p.iloc[max(0, i - 6):i + 1].max())
    return feats


def build():
    ev = pd.concat([gsi_events(), glc_events()], ignore_index=True)
    ev = ev[(ev.event_date.dt.year >= 2005) & ev.event_date.dt.month.between(4, 10)]
    ev['cell_lat'] = (ev.latitude * 10).round() / 10
    ev['cell_lon'] = (ev.longitude * 10).round() / 10
    ev = ev.drop_duplicates(['cell_lat', 'cell_lon', 'event_date'])
    print(f"events: {len(ev)} ({ev.source.value_counts().to_dict()})")

    rows = []
    for year, ev_y in ev.groupby(ev.event_date.dt.year):
        cells = sorted(set(zip(ev_y.cell_lat, ev_y.cell_lon)))
        weather = fetch_year(cells, year)
        season = pd.date_range(f'{year}-04-01', f'{year}-10-31', freq='D')
        for cell, ev_c in ev_y.groupby(['cell_lat', 'cell_lon']):
            daily = weather[cell]
            clim = fb.rain_climatology(*cell)
            event_days = set(ev_c.event_date)
            blocked = {d + pd.Timedelta(days=k) for d in event_days for k in range(-10, 11)}
            for d in event_days:
                rows.append({'cell': f'{cell[0]}_{cell[1]}', 'latitude': cell[0], 'longitude': cell[1],
                             'date': d.date(), 'year': year, 'label': 1,
                             'source': ev_c.source.iloc[0], **clim, **windows(daily, d)})
            free = [d for d in season if d not in blocked]
            for d in rng.choice(free, min(len(free), 10 * len(event_days)), replace=False):
                d = pd.Timestamp(d)
                rows.append({'cell': f'{cell[0]}_{cell[1]}', 'latitude': cell[0], 'longitude': cell[1],
                             'date': d.date(), 'year': year, 'label': 0,
                             'source': ev_c.source.iloc[0], **clim, **windows(daily, d)})
        print(f"  {year}: {len(cells)} cells, {len(ev_y)} events")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'landslide_trigger_v5.csv'), index=False)
    print(df.groupby(['source', 'label']).size())


if __name__ == '__main__':
    build()
