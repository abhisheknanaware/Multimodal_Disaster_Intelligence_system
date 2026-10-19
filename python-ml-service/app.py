from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import datetime
import logging
from weather_service import fetch_weather_data
from raster_service import get_terrain_at_location, get_sar_at_location, BASELINE_RASTER_PATH, init_raster, close_raster
from risk_engine import calculate_dynamic_risk, get_risk_class, calculate_flood_risk
from rio_tiler.io import Reader
from rio_tiler.colormap import cmap
import rasterio
import math
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Landslide Risk ML Service", version="1.0.0")

# Enable CORS for Node backend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing global resources...")
    init_raster()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Closing global resources...")
    close_raster()

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}

@app.get("/metadata")
async def get_metadata():
    return {
        "model": "CatBoost (Historical)",
        "features": ["Terrain", "Rainfall", "Sentinel-1"],
        "baseline_raster": "landslide_susceptibility_probability.tif",
        "flood_raster": "flood_susceptibility.tif"
    }

@app.get("/risk")
async def get_risk(latitude: float = Query(..., description="Latitude"),
                   longitude: float = Query(..., description="Longitude")):
    import time
    t0 = time.perf_counter()
    
    # 2. Query ML Baseline (1.92 GB raster)
    t_raster_start = time.perf_counter()
    try:
        terrain_data = get_terrain_at_location(latitude, longitude)
    except Exception as e:
        logger.error(f"Raster read error: {e}")
        terrain_data = {}

    if not terrain_data or terrain_data.get('dem') is None:
        raise HTTPException(status_code=404, detail="Coordinates outside historical data coverage or no valid terrain.")

    slope_val = terrain_data.get('slope')
    
    # 2.6 Query Sentinel-1 Raster
    sar_data = get_sar_at_location(latitude, longitude)
    if not sar_data:
        sar_data = {
            "available": False,
            "metric": "Sentinel-1 SAR Change",
            "change": None,
            "unit": None,
            "currentDate": None,
            "previousDate": None,
            "source": None
        }
    else:
        # Map metric value to 'change' for /risk consistency as requested
        sar_data["change"] = sar_data["value"]
        
    terrain_data['sar_change'] = sar_data.get("change")
    
    t_raster_end = time.perf_counter()
    raster_ms = round((t_raster_end - t_raster_start) * 1000, 2)
    
    # 3. Request Live Weather (Open-Meteo)
    t_weather_start = time.perf_counter()
    try:
        weather_metrics = await fetch_weather_data(latitude, longitude)
    except Exception as e:
        logger.error(f"Weather API failed: {e}")
        raise HTTPException(status_code=503, detail="Current weather data unavailable")
    t_weather_end = time.perf_counter()
    weather_ms = round((t_weather_end - t_weather_start) * 1000, 2)

    # 3. Calculate Risk
    t_calc_start = time.perf_counter()
    try:
        rainfall = weather_metrics["rainfall"]
        forecast = weather_metrics["forecast"]
        
        # Get real S2 data FIRST so it can be passed to risk engine
        from s2_service import get_s2_at_location
        s2_data = get_s2_at_location(latitude, longitude)
        
        soil_now = (weather_metrics.get("soil_moisture") or {}).get("value")
        current_risk_data = calculate_dynamic_risk(latitude, longitude, rainfall, soil_now)
        current_risk = current_risk_data.get("probability", 0.0)

        # Same rainfall windows, ending 24h / 72h ahead using the hourly forecast
        horizons = weather_metrics.get("horizons", {})

        def horizon_risk(key):
            h = horizons.get(key)
            if not h:
                return current_risk
            return calculate_dynamic_risk(latitude, longitude, h["rainfall"], h["soil_moisture"]).get("probability", 0.0)

        risk_24h = horizon_risk("24h")
        risk_72h = horizon_risk("72h")

        risk_class = get_risk_class(current_risk)

        riskFactorWeighting = {
            "baseline_susceptibility": current_risk_data.get("susceptibility"),
            "rainfall": current_risk_data.get("rainTrigger"),
            "slope": round(slope_val / 90.0, 3) if slope_val else 0,
            "sentinel1": 0.0
        }
        
        # Calculate flood risk concurrently or sequentially
        try:
            flood_data = await calculate_flood_risk(latitude, longitude)
        except Exception as f_err:
            logger.error(f"Flood risk calculation failed: {f_err}")
            flood_data = {"available": False, "error": str(f_err)}
            
    except Exception as e:
        logger.error(f"Risk calculation failed: {e}")
        raise HTTPException(status_code=500, detail="Risk engine failure")
    t_calc_end = time.perf_counter()
    calc_ms = round((t_calc_end - t_calc_start) * 1000, 2)
    
    t1 = time.perf_counter()
    total_ms = round((t1 - t0) * 1000, 2)

    return {
        "status": "ok",
        "metadata": {
            "model_version": "v5_catboost",
            "feature_version": "v5_shared_feature_builder",
            "sentinel2_available": s2_data.get("available", False),
            "sentinel1_available": sar_data.get("available", False) if sar_data else False,
            "rainfall_source": "Open-Meteo"
        },
        "location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "slope": {
            "degrees": slope_val,
            "source": "DEM-derived" if slope_val is not None else None
        },
        "rainfall": rainfall,
        "forecast": forecast,
        "soil_moisture": weather_metrics.get("soil_moisture"),
        "risk": {
            "current": current_risk,
            "24h": round(risk_24h, 4),
            "72h": round(risk_72h, 4)
        },
        "localContributions": current_risk_data.get("localContributions"),
        "globalFeatureImportance": current_risk_data.get("globalFeatureImportance"),
        "riskFactorWeighting": riskFactorWeighting,
        "risk_class": risk_class,
        "alertLevel": current_risk_data.get("alertLevel"),
        "timestamps": {
            "weather": weather_metrics.get("retrieved_at"),
            "generated": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        "sentinel1": sar_data,
        "sentinel2": s2_data,
        "flood": flood_data,
        "performance": {
            "raster_ms": raster_ms,
            "weather_ms": weather_ms,
            "calc_ms": calc_ms,
            "total_ms": total_ms
        }
    }

@app.get("/tiles/baseline/{z}/{x}/{y}.png")
async def get_baseline_tile(z: int, x: int, y: int):
    """
    Serves map tiles from the baseline GeoTIFF using rio-tiler.
    This prevents loading the full 1.92GB file into memory/browser.
    """
    from rio_tiler.errors import TileOutsideBounds
    from rio_tiler.colormap import cmap
    
    try:
        with Reader(BASELINE_RASTER_PATH) as src:
            # We want to render a tile
            # Using standard web mercator
            img = src.tile(x, y, z)
            
            # Since baseline is 0-1, we explicitly rescale it to 0-255 before applying colormap
            img.rescale(in_range=((0, 1),))
            
            # "rdylgn_r" is exactly what was intended (Red-Yellow-Green reversed)
            my_cmap = src.colormap or cmap.get("rdylgn_r")
            
            # Create a PNG
            content = img.render(img_format="PNG", colormap=my_cmap)
            
            return Response(content, media_type="image/png")
    except TileOutsideBounds:
        # Expected behavior outside the dataset bounds
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Tile generation error for {z}/{x}/{y}: {e}")
        # Return 204 for other expected missing tile issues as well, instead of breaking frontend
        return Response(status_code=204)

@app.get("/sar")
async def get_sar(latitude: float = Query(..., description="Latitude"),
                  longitude: float = Query(..., description="Longitude")):
    try:
        sar_data = get_sar_at_location(latitude, longitude)
        if sar_data is None:
            return {
              "available": False,
              "metric": "Sentinel-1 SAR Change",
              "value": None,
              "unit": None,
              "currentDate": None,
              "previousDate": None,
              "source": None
            }
        
        # Remove 'details' block for the specific /sar endpoint if present to match spec
        if 'details' in sar_data:
            del sar_data['details']
            
        return sar_data
        
    except Exception as e:
        logger.error(f"SAR read error: {e}")
        return {
          "available": False,
          "metric": "Sentinel-1 SAR Change",
          "value": None,
          "unit": None,
          "currentDate": None,
          "previousDate": None,
          "source": None
        }

@app.get("/flood")
async def get_flood_risk(lat: float, lon: float):
    try:
        import risk_engine
        flood_risk = await risk_engine.calculate_flood_risk(lat, lon)
        return flood_risk
    except Exception as e:
        return {"available": False, "error": str(e)}

@app.get("/tiles/flood/{z}/{x}/{y}.png")
async def get_flood_tile(z: int, x: int, y: int):
    from fastapi import Response
    return Response(status_code=204)

@app.get("/api/risk/landslide/debug")
async def debug_landslide(lat: float, lng: float):
    # Retrieve terrain
    try:
        terrain_data = get_terrain_at_location(lat, lng)
    except:
        terrain_data = {}
        
    slope_val = terrain_data.get('slope')
    
    try:
        weather = await fetch_weather_data(lat, lng)
        rain = weather.get("rainfall", {})
        r24 = rain.get("24h_mm", 0)
    except:
        weather = {"forecast": {}}
        rain = {}
        r24 = 0
        
    # Genuine ML calculation
    import risk_engine
    features_list = risk_engine.fb.LANDSLIDE_FEATURES
    dynamic_risk_data = calculate_dynamic_risk(lat, lng, rain, (weather.get("soil_moisture") or {}).get("value"))
    dynamic_risk = dynamic_risk_data.get("probability", 0.0)
    
    return {
        "location": {
            "latitude": lat,
            "longitude": lng
        },
        "model": {
            "name": "CatBoost Classifier (v5) x rainfall trigger",
            "version": "5.0",
            "features": features_list
        },
        "features": {
            "terrain": terrain_data,
            "rainfall_1d": r24
        },
        "prediction": {
            "probability": round(dynamic_risk, 4),
            "riskLevel": get_risk_class(dynamic_risk)
        },
        "metadata": {
            "pipeline": "MODEL-DRIVEN / GENUINE ML INFERENCE"
        }
    }

@app.get("/api/risk/flood/debug")
async def debug_flood(lat: float, lng: float):
    try:
        import risk_engine
        flood_data = await risk_engine.calculate_flood_risk(lat, lng)
        prob = flood_data.get("floodProbability", 0)
        
        risk_level = get_risk_class(prob) if prob else "Unknown"
        
        return {
            "location": {
                "latitude": lat,
                "longitude": lng
            },
            "model": {
                "name": "CatBoost Classifier",
                "version": "v5",
                "features": risk_engine.fb.FLOOD_FEATURES
            },
            "features": {
                "terrain": flood_data.get("terrain", {}),
                "rainfall": flood_data.get("rainfall", {}),
                "sentinel1": flood_data.get("sentinel1", {})
            },
            "prediction": {
                "probability": prob,
                "riskLevel": risk_level
            },
            "metadata": {
                "pipeline": "MODEL-DRIVEN / GENUINE ML INFERENCE"
            }
        }
    except Exception as e:
        return {"error": str(e)}
