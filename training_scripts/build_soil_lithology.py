"""
Builds local soil and lithology rasters for Assam so the live API never calls
external servers:

  Processing/assam_soilgrids_250m.tif  ISRIC SoilGrids 2.0 (CC-BY 4.0), 12 bands:
      clay, sand, silt, bdod, soc, cfvo at 0-5 cm and 30-60 cm
  Processing/assam_lithology_250m.tif  GLiM v1 first-level rock class codes
      (Hartmann & Moosdorf 2012), from data/v5/external/glim/LiMW_GIS_2015.gdb.zip
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC = os.path.join(ROOT, 'Processing')
sys.path.insert(0, os.path.join(ROOT, 'python-ml-service'))
import feature_builder as fb  # noqa: E402

WEST, SOUTH, EAST, NORTH = 89.5, 24.0, 96.0, 28.35
RES = 0.0025
WIDTH, HEIGHT = int(round((EAST - WEST) / RES)), int(round((NORTH - SOUTH) / RES))
TRANSFORM = from_origin(WEST, NORTH, RES, RES)


def build_soil():
    out = os.path.join(PROC, 'assam_soilgrids_250m.tif')
    bands = []
    for name in fb.SOIL_BANDS:
        prop, depth = name.split('_', 1)
        url = f"/vsicurl/https://files.isric.org/soilgrids/latest/data/{prop}/{prop}_{depth}_mean.vrt"
        dst = np.full((HEIGHT, WIDTH), -32768, dtype='int16')
        with rasterio.Env(GDAL_HTTP_TIMEOUT=120, GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=10,
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.vrt,.tif'):
            with rasterio.open(url) as src:
                # Read only the Assam window from the remote mosaic, then warp locally
                left, bottom, right, top = transform_bounds('EPSG:4326', src.crs, WEST - 0.1, SOUTH - 0.1,
                                                            EAST + 0.1, NORTH + 0.1, densify_pts=50)
                win = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
                arr = src.read(1, window=win)
                reproject(arr, dst, src_transform=src.window_transform(win), src_crs=src.crs,
                          src_nodata=src.nodata, dst_transform=TRANSFORM, dst_crs='EPSG:4326',
                          dst_nodata=-32768, resampling=Resampling.bilinear)
        bands.append(dst)
        valid = dst[dst != -32768]
        print(f"  {name}: median {np.median(valid) if valid.size else 'n/a'}")
    profile = dict(driver='GTiff', height=HEIGHT, width=WIDTH, count=len(bands), dtype='int16',
                   crs='EPSG:4326', transform=TRANSFORM, nodata=-32768, compress='deflate',
                   tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out, 'w', **profile) as dst:
        for i, b in enumerate(bands, 1):
            dst.write(b, i)
            dst.set_band_description(i, fb.SOIL_BANDS[i - 1])
    print(f"  wrote {out}")


def build_lithology():
    import geopandas as gpd
    from rasterio.features import rasterize
    src = os.path.join(ROOT, 'data', 'v5', 'external', 'glim', 'LiMW_GIS_2015.gdb.zip')
    layers = gpd.list_layers(src) if hasattr(gpd, 'list_layers') else None
    layer = layers.name.iloc[0] if layers is not None else None
    print(f"  layer: {layer}")
    import pyogrio
    from rasterio.warp import transform_bounds as tb
    layer_crs = pyogrio.read_info(src, layer=layer)['crs']
    box = tb('EPSG:4326', layer_crs, WEST, SOUTH, EAST, NORTH, densify_pts=50)
    g = gpd.read_file(src, layer=layer, bbox=box).to_crs(4326)
    col = next(c for c in g.columns if c.lower() == 'xx')
    print("  classes:", g[col].value_counts().to_dict())
    codes = {c: i + 1 for i, c in enumerate(fb.LITHO_CLASSES)}
    shapes = ((geom, codes.get(cls, 0)) for geom, cls in zip(g.geometry, g[col]) if geom is not None)
    arr = rasterize(shapes, out_shape=(HEIGHT, WIDTH), transform=TRANSFORM, fill=0, dtype='uint8')
    out = os.path.join(PROC, 'assam_lithology_250m.tif')
    with rasterio.open(out, 'w', driver='GTiff', height=HEIGHT, width=WIDTH, count=1, dtype='uint8',
                       crs='EPSG:4326', transform=TRANSFORM, nodata=0, compress='deflate') as dst:
        dst.write(arr, 1)
    print(f"  wrote {out}")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'soil'):
        build_soil()
    if which in ('all', 'lithology'):
        build_lithology()
