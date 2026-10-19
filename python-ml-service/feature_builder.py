"""
Shared feature construction for the v5 landslide, landslide-trigger and flood models.

The same functions are used by training_scripts/train_v5_models.py and by the
live API, so the model always sees features computed the same way.
"""
import os
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

workspace = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DEM_PATH = os.path.join(workspace, "Processing", "assam_glo30_dem.tif")
SLOPE_PATH = os.path.join(workspace, "Processing", "assam_slope_corrected.tif")
ASPECT_PATH = os.path.join(workspace, "Processing", "assam_aspect.tif")
TRI_PATH = os.path.join(workspace, "Processing", "assam_TRI.tif")
TPI_PATH = os.path.join(workspace, "Processing", "assam_TPI.tif")
RAIN_CLIM_PATH = os.path.join(workspace, "models", "v5", "imd_monsoon_climatology.npz")
ROADS_PATH = os.path.join(workspace, "models", "v5", "osm_roads_points.npz")
FLOWACC_PATH = os.path.join(workspace, "Processing", "assam_flowacc_90m.tif")
HAND_PATH = os.path.join(workspace, "Processing", "assam_hand_90m.tif")
DIST_STREAM_PATH = os.path.join(workspace, "Processing", "assam_dist_stream_90m.tif")
SOIL_PATH = os.path.join(workspace, "Processing", "assam_soilgrids_250m.tif")
LITHO_PATH = os.path.join(workspace, "Processing", "assam_lithology_250m.tif")
RAIN_QM_PATH = os.path.join(workspace, "models", "v5", "rain_qm_era5_to_imd.npz")

# SoilGrids 2.0 bands (stored units: clay/sand/silt g/kg, bdod cg/cm3, soc dg/kg, cfvo cm3/dm3)
SOIL_BANDS = [f"{p}_{d}" for d in ("0-5cm", "30-60cm") for p in ("clay", "sand", "silt", "bdod", "soc", "cfvo")]
# GLiM first-level classes: unconsolidated / siliciclastic / mixed / carbonate sediments,
# metamorphics, acid / basic / intermediate plutonics, water bodies
LITHO_CLASSES = ["su", "ss", "sm", "sc", "mt", "pa", "pb", "pi", "wb"]
LITHO_FEATURES = ["lith_unconsolidated", "lith_siliciclastic", "lith_mixed_sed", "lith_metamorphic", "lith_plutonic"]

_to_utm46 = Transformer.from_crs("EPSG:4326", "EPSG:32646", always_xy=True)

LANDSLIDE_FEATURES = [
    'elevation', 'slope', 'northness', 'eastness', 'tri', 'tpi',
    'slope_mean_500m', 'slope_max_500m', 'slope_std_500m',
    'relief_500m', 'relief_1500m', 'rel_elev_1500m', 'curvature_proxy',
    'slope_mean_1500m', 'relief_5km', 'elev_std_1500m',
    'rain_clim_jjas', 'heavy_rain_days',
    'dist_road_m', 'dist_stream_m', 'hand_m', 'log_flowacc', 'log_flowacc_2km',
    'clay_0-5cm', 'sand_0-5cm', 'bdod_0-5cm', 'soc_0-5cm', 'cfvo_0-5cm',
    'clay_30-60cm', 'sand_30-60cm', 'cfvo_30-60cm',
    'lith_unconsolidated', 'lith_siliciclastic', 'lith_mixed_sed', 'lith_metamorphic', 'lith_plutonic',
]

FLOOD_FEATURES = [
    'elevation', 'slope', 'relief_1500m', 'rel_elev_1500m', 'rel_elev_5km',
    'elev_pct_5km', 'low_flat_frac_1500m', 'slope_mean_1500m', 'elev_std_1500m',
    'rain_clim_jjas', 'heavy_rain_days',
    'rain_1d', 'rain_3d', 'rain_7d', 'soil_moisture',
    'hand_m', 'dist_stream_m', 'log_flowacc', 'log_flowacc_2km',
]

TRIGGER_FEATURES = [
    'rain_1d', 'rain_3d', 'rain_7d', 'rain_14d', 'rain_30d', 'rain_3d_share_14d',
    'soil_moisture', 'rain_hours_1d', 'rain_max_1d_7d',
    'rain_3d_norm', 'rain_14d_norm', 'rain_clim_jjas',
]

_datasets = {}
_rain_clim = None
_roads = None


def _open(path):
    ds = _datasets.get(path)
    if ds is None:
        ds = rasterio.open(path)
        _datasets[path] = ds
    return ds


def close_all():
    for ds in _datasets.values():
        ds.close()
    _datasets.clear()


def rain_climatology(lat: float, lon: float) -> dict:
    """Mean IMD Jun-Sep rainfall (mm) and heavy-rain days (>=64.5 mm) per year, 2020-2025."""
    global _rain_clim
    if _rain_clim is None:
        _rain_clim = dict(np.load(RAIN_CLIM_PATH))
    c = _rain_clim
    i = int(np.abs(c['lat'] - lat).argmin())
    j = int(np.abs(c['lon'] - lon).argmin())
    return {'rain_clim_jjas': float(c['jjas_mm'][i, j]), 'heavy_rain_days': float(c['heavy_days'][i, j])}


def distance_to_road(lat: float, lon: float) -> float:
    """Metres to the nearest OSM motorway/trunk/primary/secondary/tertiary road."""
    global _roads
    if _roads is None:
        from scipy.spatial import cKDTree
        _roads = cKDTree(np.load(ROADS_PATH)['xy_utm46'])
    x, y = _to_utm46.transform(lon, lat)
    return float(min(_roads.query((x, y))[0], 20000.0))


def hydrology_features(lat: float, lon: float) -> dict:
    """DEM-derived drainage descriptors (90 m) from training_scripts/build_hydrology.py."""
    acc_ds = _open(FLOWACC_PATH)
    hand = _point(_open(HAND_PATH), lon, lat)
    dist = _point(_open(DIST_STREAM_PATH), lon, lat)

    def log_max(radius_m):
        win = _window(acc_ds, lon, lat, radius_m)
        return float(np.log10(np.nanmax(win) + 1)) if np.isfinite(win).any() else np.nan

    return {
        'log_flowacc': log_max(500),
        'log_flowacc_2km': log_max(2000),
        'hand_m': hand,
        'dist_stream_m': dist,
    }


def soil_lithology_features(lat: float, lon: float) -> dict:
    """SoilGrids texture/density/gravel and GLiM rock type at a point (NaN if rasters missing)."""
    out = {}
    if os.path.exists(SOIL_PATH):
        ds = _open(SOIL_PATH)
        row, col = ds.index(lon, lat)
        if 0 <= row < ds.height and 0 <= col < ds.width:
            vals = ds.read(window=Window(col, row, 1, 1))[:, 0, 0].astype(float)
            vals[vals == ds.nodata] = np.nan
            out.update(dict(zip(SOIL_BANDS, vals)))
    code = 0
    if os.path.exists(LITHO_PATH):
        ds = _open(LITHO_PATH)
        row, col = ds.index(lon, lat)
        if 0 <= row < ds.height and 0 <= col < ds.width:
            code = int(ds.read(1, window=Window(col, row, 1, 1))[0, 0])
    cls = LITHO_CLASSES[code - 1] if 0 < code <= len(LITHO_CLASSES) else None
    out.update({
        'lith_unconsolidated': float(cls == 'su'),
        'lith_siliciclastic': float(cls == 'ss'),
        'lith_mixed_sed': float(cls in ('sm', 'sc')),
        'lith_metamorphic': float(cls == 'mt'),
        'lith_plutonic': float(cls in ('pa', 'pb', 'pi')),
    } if cls is not None else {f: np.nan for f in LITHO_FEATURES})
    for b in SOIL_BANDS:
        out.setdefault(b, np.nan)
    return out


_rain_qm = None


def to_imd_scale(rain: dict) -> dict:
    """
    Maps Open-Meteo (ERA5-like) rain windows onto the IMD gauge-gridded scale with a
    quantile mapping fitted on 2007-2025 NE India daily pairs. An n-day window is mapped
    through its mean daily rate: n * QM(sum / n).
    """
    global _rain_qm
    if _rain_qm is None:
        _rain_qm = dict(np.load(RAIN_QM_PATH))
    q = _rain_qm

    def qm(x):
        return float(np.interp(x, q['src'], q['dst'])) if np.isfinite(x) else np.nan

    out = dict(rain)
    for key, n in (('24h_mm', 1), ('3d_mm', 3), ('7d_mm', 7), ('14d_mm', 14), ('30d_mm', 30)):
        if key in rain and rain[key] is not None:
            out[key] = n * qm(rain[key] / n)
    if rain.get('max_daily_7d_mm') is not None:
        out['max_daily_7d_mm'] = qm(rain['max_daily_7d_mm'])
    return out


def trigger_features(rain: dict, soil_moisture, clim: dict) -> dict:
    """Inputs to the learned landslide rainfall-trigger model."""
    f = rainfall_features(rain, soil_moisture)
    f['rain_hours_1d'] = rain.get('hours_24h', np.nan)
    f['rain_max_1d_7d'] = rain.get('max_daily_7d_mm', np.nan)
    daily_clim = max(1.0, clim['rain_clim_jjas'] / 122.0)
    f['rain_3d_norm'] = f['rain_3d'] / (3 * daily_clim)
    f['rain_14d_norm'] = f['rain_14d'] / (14 * daily_clim)
    f['rain_clim_jjas'] = clim['rain_clim_jjas']
    return f


def _valid(arr, ds):
    arr = arr.astype('float64')
    if ds.nodata is not None:
        arr[arr == ds.nodata] = np.nan
    arr[arr < -1000] = np.nan
    return arr


def _point(ds, x, y):
    row, col = ds.index(x, y)
    if not (0 <= row < ds.height and 0 <= col < ds.width):
        return np.nan
    v = _valid(ds.read(1, window=Window(col, row, 1, 1)), ds)
    return float(v[0, 0])


def _window(ds, lon, lat, radius_m):
    """Square window of roughly radius_m around a lon/lat point on a geographic grid."""
    row, col = ds.index(lon, lat)
    px_m = abs(ds.res[1]) * 111320.0
    r = max(1, int(round(radius_m / px_m)))
    r0, c0 = max(0, row - r), max(0, col - r)
    r1, c1 = min(ds.height, row + r + 1), min(ds.width, col + r + 1)
    if r0 >= r1 or c0 >= c1:
        return np.full((1, 1), np.nan)
    return _valid(ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)), ds)


def terrain_features(lat: float, lon: float) -> dict:
    """Static terrain descriptors at a point (shared by both hazards)."""
    dem, slope_ds, aspect_ds = _open(DEM_PATH), _open(SLOPE_PATH), _open(ASPECT_PATH)
    tri_ds, tpi_ds = _open(TRI_PATH), _open(TPI_PATH)

    elevation = _point(dem, lon, lat)
    slope = _point(slope_ds, lon, lat)
    aspect = _point(aspect_ds, lon, lat)
    tri = _point(tri_ds, lon, lat)
    ux, uy = _to_utm46.transform(lon, lat)
    tpi = _point(tpi_ds, ux, uy)

    dem_500, dem_1500, dem_5k = (_window(dem, lon, lat, r) for r in (500, 1500, 5000))
    slope_500 = _window(slope_ds, lon, lat, 500)
    slope_1500 = _window(slope_ds, lon, lat, 1500)

    def stat(fn, arr):
        return float(fn(arr)) if np.isfinite(arr).any() else np.nan

    has_aspect = np.isfinite(aspect) and aspect >= 0
    return {
        'elevation': elevation,
        'slope': slope,
        'northness': math.cos(math.radians(aspect)) if has_aspect else np.nan,
        'eastness': math.sin(math.radians(aspect)) if has_aspect else np.nan,
        'tri': tri,
        'tpi': tpi,
        'slope_mean_500m': stat(np.nanmean, slope_500),
        'slope_max_500m': stat(np.nanmax, slope_500),
        'slope_std_500m': stat(np.nanstd, slope_500),
        'relief_500m': stat(np.nanmax, dem_500) - stat(np.nanmin, dem_500),
        'relief_1500m': stat(np.nanmax, dem_1500) - stat(np.nanmin, dem_1500),
        'rel_elev_1500m': elevation - stat(np.nanmin, dem_1500),
        'rel_elev_5km': elevation - stat(np.nanmin, dem_5k),
        'elev_pct_5km': stat(lambda a: np.nanmean(a < elevation), dem_5k),
        'curvature_proxy': elevation - stat(np.nanmean, dem_500),
        'low_flat_frac_1500m': _low_flat_fraction(dem_1500, slope_1500, elevation),
        'slope_mean_1500m': stat(np.nanmean, slope_1500),
        'relief_5km': stat(np.nanmax, dem_5k) - stat(np.nanmin, dem_5k),
        'elev_std_1500m': stat(np.nanstd, dem_1500),
        **rain_climatology(lat, lon),
        **hydrology_features(lat, lon),
        **soil_lithology_features(lat, lon),
        'dist_road_m': distance_to_road(lat, lon),
    }


def _low_flat_fraction(dem_win, slope_win, elevation):
    """Share of nearby pixels that are flat (<2 deg) and no higher than the point + 5 m."""
    if not np.isfinite(elevation) or dem_win.shape != slope_win.shape:
        return np.nan
    valid = np.isfinite(dem_win) & np.isfinite(slope_win)
    if not valid.any():
        return np.nan
    low_flat = (dem_win <= elevation + 5) & (slope_win < 2.0)
    return float(low_flat[valid].mean())


def rainfall_features(rain: dict, soil_moisture=None) -> dict:
    """Antecedent rainfall windows (mm) ending at the prediction time."""
    r1 = rain.get('24h_mm', np.nan)
    r3 = rain.get('3d_mm', np.nan)
    r14 = rain.get('14d_mm', np.nan)
    return {
        'rain_1d': r1,
        'rain_3d': r3,
        'rain_7d': rain.get('7d_mm', np.nan),
        'rain_14d': r14,
        'rain_30d': rain.get('30d_mm', np.nan),
        'rain_3d_share_14d': r3 / (r14 + 1.0) if np.isfinite(r3) and np.isfinite(r14) else np.nan,
        'soil_moisture': soil_moisture if soil_moisture is not None else np.nan,
    }


def landslide_rain_trigger(rain_24h: float, rain_72h: float) -> float:
    """
    Rainfall triggering factor in [0, 1] anchored to IMD daily rainfall classes
    (heavy >= 64.5 mm, very heavy >= 115.6 mm, extremely heavy >= 204.5 mm).
    The landslide inventory has no event dates, so the trigger cannot be learned
    from labels; it scales the learned susceptibility instead.
    """
    r24 = max(0.0, rain_24h or 0.0)
    r72 = max(0.0, rain_72h or 0.0)
    daily = 1.0 / (1.0 + math.exp(-(r24 - 90.0) / 25.0))
    multi = 1.0 / (1.0 + math.exp(-(r72 - 180.0) / 45.0))
    return round(0.15 + 0.85 * max(daily, multi), 4)
