import os
import json
import logging
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import rasterio
import catboost
from catboost import Pool

import feature_builder as fb
from weather_service import fetch_weather_data

logger = logging.getLogger(__name__)

workspace = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(workspace, "models", "v5"))


def _load(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        logger.error(f"Model not found: {path}")
        return None
    model = catboost.CatBoostClassifier()
    model.load_model(path)
    return model


landslide_model = _load("landslide_v5.cbm")
flood_model = _load("flood_v5.cbm")
trigger_model = _load("landslide_trigger_v5.cbm")

try:
    with open(os.path.join(MODEL_DIR, "metrics_v5.json")) as f:
        _metrics = json.load(f)
except Exception:
    _metrics = {}
LANDSLIDE_THRESHOLD = _metrics.get("landslide", {}).get("deploy_threshold", 0.5)
FLOOD_THRESHOLD = _metrics.get("flood", {}).get("deploy_threshold", 0.5)
TRIGGER_SCALE = _metrics.get("landslide_trigger", {}).get("trigger_scale", 1.0)


@lru_cache(maxsize=4096)
def _terrain(lat_r: float, lon_r: float) -> dict:
    return fb.terrain_features(lat_r, lon_r)


def terrain_at(lat: float, lon: float) -> dict:
    """Static terrain features, cached per ~10 m cell."""
    return _terrain(round(lat, 4), round(lon, 4))


def _predict(model, features: list, values: dict):
    row = pd.DataFrame([[values.get(f, np.nan) for f in features]], columns=features).astype(float)
    prob = float(model.predict_proba(row)[0][1])
    shap = model.get_feature_importance(Pool(row), type='ShapValues')[0][:-1]
    local = {f: round(float(v), 4) for f, v in zip(features, shap)}
    global_fi = {f: round(float(v), 3) for f, v in zip(features, model.get_feature_importance())}
    return prob, local, global_fi


def rain_trigger(lat: float, lon: float, rainfall: dict, soil_moisture=None):
    """
    Learned rainfall trigger in [0, 1] (1 = as wet as a typical intense landslide day),
    from dated GSI + NASA GLC events. Falls back to the IMD-threshold rule without the model.
    """
    if trigger_model is None:
        return fb.landslide_rain_trigger(rainfall.get("24h_mm", 0), rainfall.get("3d_mm", 0)), None
    values = fb.trigger_features(rainfall, soil_moisture, fb.rain_climatology(lat, lon))
    raw, local, _ = _predict(trigger_model, fb.TRIGGER_FEATURES, values)
    return round(min(1.0, raw / TRIGGER_SCALE), 4), local


def calculate_dynamic_risk(lat: float, lon: float, rainfall: dict, soil_moisture=None) -> dict:
    """
    Landslide hazard = learned susceptibility (terrain classifier) x learned rainfall trigger.
    `rainfall` holds the windows ending at the time being predicted (now or a forecast horizon).
    """
    result = {"probability": 0.0, "susceptibility": None, "rainTrigger": None,
              "localContributions": None, "globalFeatureImportance": None}
    if landslide_model is None:
        return result
    try:
        terrain = terrain_at(lat, lon)
        susceptibility, local, global_fi = _predict(landslide_model, fb.LANDSLIDE_FEATURES, terrain)
        trigger, trigger_local = rain_trigger(lat, lon, rainfall, soil_moisture)
        result.update({
            "probability": round(susceptibility * trigger, 4),
            "susceptibility": round(susceptibility, 4),
            "susceptibleClass": susceptibility >= LANDSLIDE_THRESHOLD,
            "rainTrigger": trigger,
            "triggerContributions": trigger_local,
            "localContributions": local,
            "globalFeatureImportance": global_fi,
        })
    except Exception as e:
        logger.error(f"Landslide risk failed: {e}")
    return result


def get_risk_class(risk_value: float) -> str:
    """
    Returns the risk classification based on the requested thresholds.
    0.0–0.2 = Very Low
    0.2–0.4 = Low
    0.4–0.6 = Moderate
    0.6–0.8 = High
    0.8–1.0 = Very High
    """
    if risk_value < 0.2:
        return "Very Low"
    elif risk_value < 0.4:
        return "Low"
    elif risk_value < 0.6:
        return "Moderate"
    elif risk_value < 0.8:
        return "High"
    else:
        return "Very High"


def _sentinel1_display(lat: float, lon: float) -> dict:
    """Latest processed Sentinel-1 backscatter, shown for context only (not a model input)."""
    s1_dir = os.path.join(workspace, "sentel", "sentinel", "output", "2026")
    data_dir = os.path.join(s1_dir, "S1A_IW_GRDH_1SDV_20260526T115657_20260526T115722_064688_082639_2896_Orb_Cal_Spk_TC.data")
    out = {"vv": None, "vh": None, "vvChange": None, "vhChange": None, "ratio": None,
           "currentDate": "2026-05-26", "previousDate": None}
    try:
        for band, key in (("Sigma0_VV.img", "vv"), ("Sigma0_VH.img", "vh")):
            path = os.path.join(data_dir, band)
            if os.path.exists(path):
                with rasterio.open(path) as src:
                    val = float(next(src.sample([(lon, lat)]))[0])
                    out[key] = round(val, 4) if val > -999 else None
        if out["vv"] and out["vh"]:
            out["ratio"] = round(out["vv"] / out["vh"], 4)
    except Exception as e:
        logger.error(f"Error extracting Sentinel-1: {e}")
    return out


async def calculate_flood_risk(lat: float, lon: float):
    rainfall_data, soil = None, None
    try:
        weather_metrics = await fetch_weather_data(lat, lon)
        if weather_metrics.get("source") != "Fallback":
            rainfall_data = weather_metrics.get("rainfall", {})
            soil = (weather_metrics.get("soil_moisture") or {}).get("value")
    except Exception as e:
        logger.error(f"Error extracting weather data: {e}")

    terrain = {}
    flood_prob, local, global_fi = None, None, {}
    try:
        terrain = terrain_at(lat, lon)
        if flood_model is not None and rainfall_data is not None and np.isfinite(terrain.get('elevation', np.nan)):
            values = dict(terrain)
            values.update(fb.rainfall_features(rainfall_data, soil))
            flood_prob, local, global_fi = _predict(flood_model, fb.FLOOD_FEATURES, values)
    except Exception as e:
        logger.error(f"Flood inference error: {e}")

    missing = []
    if rainfall_data is None:
        missing.append("live rainfall")
    if not np.isfinite(terrain.get('elevation', np.nan)):
        missing.append("terrain")

    return {
        "available": flood_prob is not None,
        "floodProbability": round(flood_prob, 4) if flood_prob is not None else None,
        "floodClass": flood_prob >= FLOOD_THRESHOLD if flood_prob is not None else None,
        "threshold": FLOOD_THRESHOLD,
        "model": "Flood CatBoost v5",
        "predictionDate": datetime.now().isoformat(),
        "missingFeatures": missing,
        "rainfall": {
            "intensity": rainfall_data.get("24h_mm") if rainfall_data else None,
            "1d": rainfall_data.get("24h_mm") if rainfall_data else None,
            "3d": rainfall_data.get("3d_mm") if rainfall_data else None,
            "7d": rainfall_data.get("7d_mm") if rainfall_data else None,
            "antecedent": rainfall_data.get("14d_mm") if rainfall_data else None
        },
        "soilMoisture": soil,
        "sentinel1": _sentinel1_display(lat, lon),
        "terrain": {
            "dem": round(terrain['elevation'], 2) if np.isfinite(terrain.get('elevation', np.nan)) else None,
            "slope": round(terrain['slope'], 2) if np.isfinite(terrain.get('slope', np.nan)) else None,
            "relativeElevation5km": round(terrain['rel_elev_5km'], 2) if np.isfinite(terrain.get('rel_elev_5km', np.nan)) else None,
        },
        "localContributions": local,
        "globalFeatureImportance": global_fi
    }
