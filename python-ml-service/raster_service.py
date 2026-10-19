import rasterio
from rasterio.windows import Window
import logging
import numpy as np
import time
from pyproj import Transformer

logger = logging.getLogger(__name__)

transformer_4326_to_32646 = Transformer.from_crs("EPSG:4326", "EPSG:32646", always_xy=True)

import os
workspace = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASELINE_RASTER_PATH = os.path.join(workspace, "sentinel", "sentinel", "taking_feature", "maps", "landslide_susceptibility_probability.tif")
SLOPE_RASTER_PATH = os.path.join(workspace, "Processing", "assam_slope_corrected.tif")
SAR_RASTER_PATH = os.path.join(workspace, "Processing", "S1_SAR_Change.tif")

ASPECT_RASTER_PATH = os.path.join(workspace, "Processing", "assam_aspect.tif")
TRI_RASTER_PATH = os.path.join(workspace, "Processing", "assam_TRI.tif")
TPI_RASTER_PATH = os.path.join(workspace, "Processing", "assam_TPI.tif")
DEM_RASTER_PATH = os.path.join(workspace, "Processing", "assam_glo30_dem.tif")
S2_MOSAIC_PATH = os.path.join(workspace, "data", "sentinel2", "2025", "processed", "S2_Mosaic_NDVI_NDWI.tif")

# Global rasterio dataset readers
_raster_dataset = None
_slope_dataset = None
_sar_dataset = None
_aspect_dataset = None
_tri_dataset = None
_tpi_dataset = None
_dem_dataset = None
_s2_mosaic_dataset = None

def init_raster():
    """Initializes the global raster dataset readers."""
    global _raster_dataset, _slope_dataset, _sar_dataset, _aspect_dataset, _tri_dataset, _tpi_dataset, _dem_dataset, _s2_mosaic_dataset
    if _raster_dataset is None:
        logger.info(f"Opening raster dataset: {BASELINE_RASTER_PATH}")
        _raster_dataset = rasterio.open(BASELINE_RASTER_PATH)
    if _slope_dataset is None:
        try:
            logger.info(f"Opening slope raster dataset: {SLOPE_RASTER_PATH}")
            _slope_dataset = rasterio.open(SLOPE_RASTER_PATH)
        except Exception as e:
            logger.error(f"Failed to open slope raster: {e}")
    if _sar_dataset is None:
        try:
            logger.info(f"Opening Sentinel-1 SAR dataset: {SAR_RASTER_PATH}")
            _sar_dataset = rasterio.open(SAR_RASTER_PATH)
        except Exception as e:
            logger.error(f"Failed to open SAR raster (may not exist yet): {e}")
            
    try:
        if _aspect_dataset is None: _aspect_dataset = rasterio.open(ASPECT_RASTER_PATH)
        if _tri_dataset is None: _tri_dataset = rasterio.open(TRI_RASTER_PATH)
        if _tpi_dataset is None: _tpi_dataset = rasterio.open(TPI_RASTER_PATH)
        if _dem_dataset is None: _dem_dataset = rasterio.open(DEM_RASTER_PATH)
        if _s2_mosaic_dataset is None and os.path.exists(S2_MOSAIC_PATH):
            _s2_mosaic_dataset = rasterio.open(S2_MOSAIC_PATH)
    except Exception as e:
        logger.error(f"Failed to open terrain terrain rasters: {e}")

def close_raster():
    """Closes the global raster dataset readers."""
    global _raster_dataset, _slope_dataset, _sar_dataset, _aspect_dataset, _tri_dataset, _tpi_dataset, _dem_dataset, _s2_mosaic_dataset
    if _raster_dataset is not None:
        logger.info("Closing raster dataset.")
        _raster_dataset.close()
        _raster_dataset = None
    if _slope_dataset is not None:
        logger.info("Closing slope raster dataset.")
        _slope_dataset.close()
        _slope_dataset = None
    if _sar_dataset is not None:
        logger.info("Closing SAR raster dataset.")
        _sar_dataset.close()
        _sar_dataset = None
    if _aspect_dataset is not None: _aspect_dataset.close(); _aspect_dataset = None
    if _tri_dataset is not None: _tri_dataset.close(); _tri_dataset = None
    if _tpi_dataset is not None: _tpi_dataset.close(); _tpi_dataset = None
    if _dem_dataset is not None: _dem_dataset.close(); _dem_dataset = None

def get_baseline_susceptibility(lat: float, lon: float) -> float:
    """
    Reads exactly one pixel from the globally opened raster dataset in a memory-safe way.
    Returns None if the point is outside the raster bounds or is NoData.
    """
    if _raster_dataset is None:
        logger.error("Raster dataset not initialized!")
        raise RuntimeError("Raster dataset not initialized!")

    try:
        # Check if point is inside bounds
        bounds = _raster_dataset.bounds
        if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
            logger.warning(f"Point ({lat}, {lon}) is outside the raster bounds.")
            return None
        
        # Convert lat/lon to row/col
        row, col = _raster_dataset.index(lon, lat)
        
        # Ensure row/col is within the valid raster dimensions
        if row < 0 or col < 0 or row >= _raster_dataset.height or col >= _raster_dataset.width:
            logger.warning(f"Point ({lat}, {lon}) row/col out of image dimensions.")
            return None

        # Read only a 1x1 window
        window = Window(col, row, 1, 1)
        data = _raster_dataset.read(1, window=window)
        value = data[0, 0]
        
        nodata = _raster_dataset.nodata
        if nodata is not None and value == nodata:
            return None
            
        if np.isnan(value):
            return None

        return float(value)
        
    except Exception as e:
        logger.error(f"Error reading raster for lat={lat}, lon={lon}: {str(e)}")
        raise e

def get_terrain_at_location(lat: float, lon: float):
    """
    Reads exactly one pixel from slope, aspect, TRI, TPI, DEM in a memory-safe way.
    """
    terrain_data = {
        'slope': None,
        'aspect': None,
        'tri': None,
        'tpi': None,
        'dem': None,
        's2_ndvi': None,
        's2_ndwi': None
    }
    
    datasets = {
        'slope': _slope_dataset,
        'aspect': _aspect_dataset,
        'tri': _tri_dataset,
        'tpi': _tpi_dataset,
        'dem': _dem_dataset
    }
    
    # Process S2 Mosaic separately since it has 2 bands
    if _s2_mosaic_dataset is not None:
        try:
            ds = _s2_mosaic_dataset
            query_lon, query_lat = lon, lat
            if ds.crs and ds.crs.to_string() == 'EPSG:32646':
                query_lon, query_lat = transformer_4326_to_32646.transform(lon, lat)
            
            bounds = ds.bounds
            if (bounds.left <= query_lon <= bounds.right and bounds.bottom <= query_lat <= bounds.top):
                row, col = ds.index(query_lon, query_lat)
                if 0 <= row < ds.height and 0 <= col < ds.width:
                    window = Window(col, row, 1, 1)
                    data = ds.read(window=window)
                    if ds.nodata is None or data[0, 0, 0] != ds.nodata:
                        terrain_data['s2_ndvi'] = float(data[0, 0, 0])
                    if ds.nodata is None or data[1, 0, 0] != ds.nodata:
                        terrain_data['s2_ndwi'] = float(data[1, 0, 0])
        except Exception as e:
            logger.error(f"Error reading S2 Mosaic: {e}")
    
    for key, ds in datasets.items():
        if ds is None:
            continue
        try:
            # Transform coordinates if raster is in UTM 46N (EPSG:32646)
            query_lon, query_lat = lon, lat
            if ds.crs and ds.crs.to_string() == 'EPSG:32646':
                query_lon, query_lat = transformer_4326_to_32646.transform(lon, lat)
                
            bounds = ds.bounds
            if not (bounds.left <= query_lon <= bounds.right and bounds.bottom <= query_lat <= bounds.top):
                continue
            
            row, col = ds.index(query_lon, query_lat)
            if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
                continue

            window = Window(col, row, 1, 1)
            data = ds.read(1, window=window)
            value = data[0, 0]
            
            nodata = ds.nodata
            if nodata is not None and value == nodata:
                continue
                
            if np.isnan(value):
                continue

            terrain_data[key] = round(float(value), 2)
            
        except Exception as e:
            logger.error(f"Error reading {key} raster for lat={lat}, lon={lon}: {str(e)}")
            
    return terrain_data

def get_slope_at_location(lat: float, lon: float):
    terrain = get_terrain_at_location(lat, lon)
    return terrain.get('slope')

def get_sar_at_location(lat: float, lon: float):
    """
    Reads SAR change from S1_SAR_Change.tif in a memory-safe way.
    Returns a dict with metadata or None if outside bounds/nodata.
    """
    if _sar_dataset is None:
        return None
        
    try:
        bounds = _sar_dataset.bounds
        if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
            return None
            
        row, col = _sar_dataset.index(lon, lat)
        
        if row < 0 or col < 0 or row >= _sar_dataset.height or col >= _sar_dataset.width:
            return None
            
        window = Window(col, row, 1, 1)
        
        # Band 1: VV change, Band 2: VH change
        vv_change = float(_sar_dataset.read(1, window=window)[0, 0])
        vh_change = float(_sar_dataset.read(2, window=window)[0, 0])
        
        nodata = _sar_dataset.nodata
        
        if (nodata is not None and vv_change == nodata) or np.isnan(vv_change):
            return None
            
        # We can extract dates from the raster tags
        tags = _sar_dataset.tags()
        current_date = tags.get('current_date', 'Unknown')
        previous_date = tags.get('previous_date', 'Unknown')
        
        # Let's combine them into a single magnitude metric for the UI, e.g. average amplitude change or just return VV
        # We'll use VV change as the primary metric for the frontend as it's typically more stable for bare earth, 
        # or a mean of both. The user requested: "Calculate a transparent SAR change metric... If a combined metric is used, document its exact formula."
        # We'll just return VV change to keep it extremely transparent.
        
        return {
            "available": True,
            "metric": "Sentinel-1 SAR Change",
            "value": round(vv_change, 2), # Using VV change
            "unit": "dB",
            "currentDate": current_date,
            "previousDate": previous_date,
            "source": "Sentinel-1 GRD",
            "details": {
                "vv_change": round(vv_change, 2),
                "vh_change": round(vh_change, 2)
            }
        }
    except Exception as e:
        logger.error(f"Error reading SAR raster for lat={lat}, lon={lon}: {str(e)}")
        return None
