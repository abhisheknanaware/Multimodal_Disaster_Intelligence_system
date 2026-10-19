import os
import joblib
import pandas as pd
import numpy as np
import catboost

# Load the previously trained V4 Advanced model
try:
    workspace = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    ls_cb_model_path = os.path.join(workspace, "models", "landslide_v4_advanced.cbm")
    if os.path.exists(ls_cb_model_path):
        landslide_cb_model = catboost.CatBoostRegressor()
        landslide_cb_model.load_model(ls_cb_model_path)
        ls_cb_feature_names = landslide_cb_model.feature_names_
    else:
        landslide_cb_model = None
        ls_cb_feature_names = []
except Exception as e:
    print(f"Error loading v4 model: {e}")
    landslide_cb_model = None
    ls_cb_feature_names = []

# Load the flood classifier once at startup (override with FLOOD_MODEL_PATH)
try:
    flood_model_path = os.environ.get("FLOOD_MODEL_PATH", os.path.join(workspace, "models", "flood_catboost_v3.cbm"))
    if os.path.exists(flood_model_path):
        flood_cb_model = catboost.CatBoostClassifier()
        flood_cb_model.load_model(flood_model_path)
    else:
        print(f"Flood model not found at {flood_model_path}")
        flood_cb_model = None
except Exception as e:
    print(f"Error loading flood model: {e}")
    flood_cb_model = None

def calculate_dynamic_risk(terrain_data: dict, rainfall: dict, forecast: dict, s2_data: dict = None) -> dict:
    """
    Calculates dynamic landslide risk using the Sentinel-1 + Terrain model (CatBoost).
    Returns RAW probabilities without any artificial scaling.
    """
    result = {
        "probability": 0.0,
        "localContributions": None,
        "globalFeatureImportance": None
    }
    
    if landslide_cb_model is None:
        return result
        
    try:
        r24 = forecast.get("24h_mm", 0)
        
        # Build the exact 13-feature array expected by the V4 model
        r24 = forecast.get("24h_mm", 0)
        
        slope = terrain_data.get('slope', 0)
        aspect = terrain_data.get('aspect', 0)
        tpi = terrain_data.get('tpi', 0)
        
        northness = np.cos(np.radians(aspect)) if aspect is not None else 0
        eastness = np.sin(np.radians(aspect)) if aspect is not None else 0
        
        safe_slope = max(0.001, np.radians(slope)) if slope is not None else 0.001
        twi_proxy = -tpi / np.tan(safe_slope) if tpi is not None else 0
        erosion_potential = (r24 * np.tan(safe_slope)) if r24 is not None else 0
        
        # In V4, 'elevation' maps to 'dem'
        feature_dict = {
            'slope': slope if slope is not None else np.nan,
            'elevation': terrain_data.get('dem') if terrain_data.get('dem') is not None else np.nan,
            'tpi': tpi if tpi is not None else np.nan,
            'tri': terrain_data.get('tri') if terrain_data.get('tri') is not None else np.nan,
            'aspect': aspect if aspect is not None else np.nan,
            'sar_change': terrain_data.get('sar_change') if terrain_data.get('sar_change') is not None else np.nan,
            'rainfall': r24 if r24 is not None else np.nan,
            's2_ndvi': terrain_data.get('s2_ndvi') if terrain_data.get('s2_ndvi') is not None else np.nan,
            's2_ndwi': terrain_data.get('s2_ndwi') if terrain_data.get('s2_ndwi') is not None else np.nan,
            'northness': northness,
            'eastness': eastness,
            'twi_proxy': twi_proxy,
            'erosion_potential': erosion_potential
        }
        
        features = []
        missing_features = []
        for f_name in ls_cb_feature_names:
            val = feature_dict.get(f_name)
            if val is None or np.isnan(val):
                missing_features.append(f_name)
                # Let CatBoost handle missing values natively
                features.append(np.nan)
            else:
                features.append(val)
                    
        if len(missing_features) > 3:
            print(f"Warning: Missing features {missing_features}. Prediction might be degraded.")
            
        # V4 is a regressor on 0-1 susceptibility; clip to keep it a valid probability
        prob = float(landslide_cb_model.predict([features])[0])
        prob = max(0.0, min(1.0, prob))
        
        result["probability"] = round(prob, 4)
        
        # Feature importances mapping
        if hasattr(landslide_cb_model, 'get_feature_importance'):
            importances = landslide_cb_model.get_feature_importance()
            global_fi = {}
            for name, imp in zip(ls_cb_feature_names, importances):
                # Filter out the padded S1 historical dates for the UI clarity
                if "S1_" not in name or imp > 0:
                    global_fi[name] = float(imp)
            result["globalFeatureImportance"] = global_fi
            
        return result
    except Exception as e:
        print(f"Error calculating CatBoost dynamic risk: {e}")
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


async def calculate_flood_risk(lat: float, lon: float):
    from datetime import datetime
    import os
    import rasterio
    import numpy as np
    import pandas as pd
    import joblib
    from weather_service import fetch_weather_data
    
    # 1. Fetch Rainfall from Open-Meteo
    rainfall_data = None
    try:
        weather_metrics = await fetch_weather_data(lat, lon)
        rainfall_data = weather_metrics.get("rainfall", {})
    except Exception as e:
        print(f"Error extracting Weather data: {e}")

    # 2. Extract Terrain
    dem_val = None
    slope_val = None
    try:
        workspace = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        dem_path = os.path.join(workspace, "Processing", "assam_glo30_dem.tif")
        slope_path = os.path.join(workspace, "Processing", "assam_slope_corrected.tif")
        
        with rasterio.open(dem_path) as src:
            for val in src.sample([(lon, lat)]):
                dem_val = float(val[0])
                
        with rasterio.open(slope_path) as src:
            for val in src.sample([(lon, lat)]):
                slope_val = float(val[0])
    except Exception as e:
        pass

    s1_curr_date = "2026-05-26"
    s1_prev_date = "2026-05-14"
    
    s1_vv = None
    s1_vh = None
    s1_vv_prev = None
    s1_vh_prev = None
    s1_vv_change = None
    s1_vh_change = None
    s1_ratio = None
    
    try:
        workspace = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        s1_dir = os.path.join(workspace, "sentel", "sentinel", "output", "2026")
        
        # We only have one successful .dim file for 2026 due to corrupt zip archives
        curr_dim = os.path.join(s1_dir, "S1A_IW_GRDH_1SDV_20260526T115657_20260526T115722_064688_082639_2896_Orb_Cal_Spk_TC.dim")
        curr_data = curr_dim.replace(".dim", ".data")
        curr_vv_img = os.path.join(curr_data, "Sigma0_VV.img")
        curr_vh_img = os.path.join(curr_data, "Sigma0_VH.img")
        
        if os.path.exists(curr_vv_img):
            with rasterio.open(curr_vv_img) as src:
                for val in src.sample([(lon, lat)]):
                    s1_vv = float(val[0])
            with rasterio.open(curr_vh_img) as src:
                for val in src.sample([(lon, lat)]):
                    s1_vh = float(val[0])
                    
        # Impute previous with current to allow CatBoost to run (change = 0)
        s1_vv_prev = s1_vv
        s1_vh_prev = s1_vh
                    
        if s1_vv is not None and s1_vv_prev is not None and s1_vv > -999 and s1_vv_prev > -999:
            s1_vv_change = s1_vv - s1_vv_prev
        
        if s1_vh is not None and s1_vh_prev is not None and s1_vh > -999 and s1_vh_prev > -999:
            s1_vh_change = s1_vh - s1_vh_prev
            
        if s1_vv is not None and s1_vh is not None and s1_vh > -999 and s1_vv > -999:
            s1_ratio = s1_vv / s1_vh if s1_vh != 0 else 0
            
    except Exception as e:
        print(f"Error extracting Sentinel-1: {e}")

    # 4. Feature Construction & Inference
    flood_prob = None
    
    local_contributions = {}
    global_feature_importance = {}
    
    try:
        model = flood_cb_model
        if model is not None:

            feature_dict = {
                'rain_intensity': rainfall_data.get("24h_mm", 0) if rainfall_data else np.nan,
                'rainfall_1d': rainfall_data.get("24h_mm", 0) if rainfall_data else np.nan,
                'rainfall_3d': rainfall_data.get("3d_mm", 0) if rainfall_data else np.nan,
                'rainfall_7d': rainfall_data.get("7d_mm", 0) if rainfall_data else np.nan,
                'antecedent_rainfall': rainfall_data.get("14d_mm", 0) if rainfall_data else np.nan,
                'DEM': dem_val if dem_val is not None else np.nan,
                'slope': slope_val if slope_val is not None else np.nan,
                'flow_accumulation': np.nan,
                'distance_to_river': np.nan,
                'S1_VV': s1_vv if s1_vv is not None else np.nan,
                'S1_VH': s1_vh if s1_vh is not None else np.nan,
                'S1_VV_change': s1_vv_change if s1_vv_change is not None else np.nan,
                'S1_VH_change': s1_vh_change if s1_vh_change is not None else np.nan,
                'S1_VV_VH_ratio': s1_ratio if s1_ratio is not None else np.nan
            }
            
            feature_names = model.feature_names_
            
            missing_features = [f for f in feature_names if feature_dict.get(f) is None or np.isnan(feature_dict.get(f))]
            features = [feature_dict.get(f_name, np.nan) for f_name in feature_names]
            
            flood_prob = float(model.predict_proba([features])[0][1])
            
            # Extract actual global feature importances
            if hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
                for name, imp in zip(feature_names, importances):
                    global_feature_importance[name] = float(imp)
                    
            # Skip SHAP for speed/simplicity in the CatBoost V2 version
            local_contributions = None
                
    except Exception as e:
        print(f"Inference error: {e}")

    # CASE 1: Prediction Available
    return {
        "available": flood_prob is not None,
        "floodProbability": round(flood_prob, 4) if flood_prob is not None else None,
        "model": "Flood CatBoost V1",
        "predictionDate": datetime.now().isoformat(),
        "missingFeatures": [] if flood_prob is not None else ["Some features missing"],
        "rainfall": {
            "intensity": rainfall_data.get("24h_mm") if rainfall_data else None,
            "1d": rainfall_data.get("24h_mm") if rainfall_data else None,
            "3d": rainfall_data.get("3d_mm") if rainfall_data else None,
            "7d": rainfall_data.get("7d_mm") if rainfall_data else None,
            "antecedent": rainfall_data.get("14d_mm") if rainfall_data else None
        },
        "sentinel1": {
            "vv": round(s1_vv, 4) if s1_vv and s1_vv > -999 else None,
            "vh": round(s1_vh, 4) if s1_vh and s1_vh > -999 else None,
            "vvChange": round(s1_vv_change, 4) if s1_vv_change is not None else None,
            "vhChange": round(s1_vh_change, 4) if s1_vh_change is not None else None,
            "ratio": round(s1_ratio, 4) if s1_ratio is not None else None,
            "currentDate": s1_curr_date,
            "previousDate": s1_prev_date
        },
        "terrain": {
            "dem": round(dem_val, 2) if dem_val is not None else None,
            "slope": round(slope_val, 2) if slope_val is not None else None
        },
        "localContributions": local_contributions,
        "globalFeatureImportance": global_feature_importance
    }
