"""
Adds IMD gauge-gridded rainfall windows (0.25 deg, 2007-2025) to the landslide
trigger dataset and fits a quantile mapping from Open-Meteo/ERA5 daily rain to
IMD daily rain, so the live API can express its rainfall on the IMD scale.

IMD daily values cover 08:30 IST of the previous day to 08:30 IST of the stamped
day; they correlate best with reanalysis rain one day earlier, so IMD(D+1) is
used as the rain of day D.

Outputs:
  data/v5/landslide_trigger_v5.csv  (adds imd_* columns)
  models/v5/rain_qm_era5_to_imd.npz
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'data', 'v5', 'cache')


def imd_year(y):
    for p in (os.path.join(ROOT, 'data', 'v5', 'external', 'imd', f'RF25_ind{y}_rfp25.nc'),
              os.path.join(ROOT, 'rainfall', f'RF25_ind{y}_rfp25.nc')):
        if os.path.exists(p):
            da = xr.open_dataset(p)['RAINFALL'].sel(LATITUDE=slice(21.0, 30.0), LONGITUDE=slice(87.5, 98.0)).load()
            da['TIME'] = pd.to_datetime(da.TIME.values) - pd.Timedelta(days=1)
            return da
    return None


def nearest_valid(da, lat, lon):
    """Nearest IMD cell with data (IMD covers India only)."""
    valid = np.isfinite(da.isel(TIME=0).values) | np.isfinite(da.max('TIME').values)
    la, lo = np.meshgrid(da.LATITUDE.values, da.LONGITUDE.values, indexing='ij')
    d = np.where(valid, (la - lat) ** 2 + (lo - lon) ** 2, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return (i, j) if np.sqrt(d[i, j]) <= 0.5 else None


def main():
    trig_path = os.path.join(ROOT, 'data', 'v5', 'landslide_trigger_v5.csv')
    df = pd.read_csv(trig_path, parse_dates=['date'])
    cols = {k: np.full(len(df), np.nan) for k in
            ('imd_rain_1d', 'imd_rain_3d', 'imd_rain_7d', 'imd_rain_14d', 'imd_rain_30d', 'imd_max_1d_7d')}

    era5_pairs, imd_pairs = [], []
    years = sorted(df.date.dt.year.unique())
    for y in years:
        cur, nxt = imd_year(y), imd_year(y + 1)
        if cur is None:
            continue
        da = xr.concat([cur, nxt], 'TIME') if nxt is not None else cur
        rows = df.index[df.date.dt.year == y]
        cell_ij = {}
        for i in rows:
            key = (df.at[i, 'latitude'], df.at[i, 'longitude'])
            if key not in cell_ij:
                cell_ij[key] = nearest_valid(cur, *key)
            ij = cell_ij[key]
            if ij is None:
                continue
            s = da.isel(LATITUDE=ij[0], LONGITUDE=ij[1]).to_series().fillna(0.0)
            d = df.at[i, 'date']
            if d not in s.index:
                continue
            k = s.index.get_loc(d)

            def last(n):
                return float(s.iloc[max(0, k - n + 1):k + 1].sum())

            cols['imd_rain_1d'][i] = last(1)
            cols['imd_rain_3d'][i] = last(3)
            cols['imd_rain_7d'][i] = last(7)
            cols['imd_rain_14d'][i] = last(14)
            cols['imd_rain_30d'][i] = last(30)
            cols['imd_max_1d_7d'][i] = float(s.iloc[max(0, k - 6):k + 1].max())

        # Paired daily series for the quantile mapping (Apr-Oct, cells with IMD data)
        for f in glob.glob(os.path.join(CACHE, f'trigger_{y}_*.json')):
            for d in json.load(open(f)):
                ij = nearest_valid(cur, d['latitude'], d['longitude'])
                if ij is None:
                    continue
                e = pd.Series(d['daily']['precipitation_sum'], index=pd.to_datetime(d['daily']['time']), dtype=float)
                m = cur.isel(LATITUDE=ij[0], LONGITUDE=ij[1]).to_series()
                j = pd.concat([e, m], axis=1).dropna()
                j = j[(j.index.month >= 4) & (j.index.month <= 10)]
                era5_pairs.append(j.iloc[:, 0].values)
                imd_pairs.append(j.iloc[:, 1].values)
        print(f"  {y}: {int(np.isfinite(cols['imd_rain_1d'][rows]).sum())}/{len(rows)} rows with IMD")

    for k, v in cols.items():
        df[k] = v
    df.to_csv(trig_path, index=False)

    era5 = np.concatenate(era5_pairs)
    imd = np.concatenate(imd_pairs)
    q = np.linspace(0, 1, 2001)
    np.savez_compressed(os.path.join(ROOT, 'models', 'v5', 'rain_qm_era5_to_imd.npz'),
                        q=q, src=np.quantile(era5, q), dst=np.quantile(imd, q))
    print(f"QM fitted on {len(era5)} daily pairs; p50/p90/p99 ERA5 "
          f"{np.percentile(era5, [50, 90, 99]).round(1)} -> IMD {np.percentile(imd, [50, 90, 99]).round(1)}")
    print(df.groupby('label')[['rain_1d', 'imd_rain_1d', 'rain_3d', 'imd_rain_3d']].median().round(1))


if __name__ == '__main__':
    main()
