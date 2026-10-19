import httpx
import logging
from datetime import datetime, timezone, timedelta
import time
from dateutil import parser

logger = logging.getLogger(__name__)

# Simple in-memory cache to prevent excessive API requests
# cache format: {(lat, lon): {"data": {...}, "timestamp": float}}
_weather_cache = {}
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

async def fetch_weather_data(lat: float, lon: float):
    """
    Fetches past and forecast rainfall from Open-Meteo for the given lat/lon.
    Uses an in-memory cache valid for 15 minutes.
    """
    cache_key = (round(lat, 4), round(lon, 4))
    now = time.time()
    
    # Check cache
    if cache_key in _weather_cache:
        cached = _weather_cache[cache_key]
        if now - cached["timestamp"] < CACHE_TTL_SECONDS:
            logger.info(f"Using cached weather data for {cache_key}")
            return cached["data"]

    # Open-Meteo requires past days and forecast days
    # We need: 30 days past for antecedent rainfall, and 3 days forecast for 72h forecast
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation,soil_moisture_0_to_7cm",
        "timezone": "auto",
        "past_days": 31,
        "forecast_days": 4
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            # Process the response to get our specific metrics
            metrics = process_weather_data(data)
            metrics["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            metrics["source"] = "Open-Meteo"
            
            # Update cache
            _weather_cache[cache_key] = {
                "data": metrics,
                "timestamp": now
            }
            
            return metrics
            
    except Exception as e:
        logger.error(f"Failed to fetch weather for {lat}, {lon}: {repr(e)}")
        # Graceful fallback instead of crashing
        return {
            "rainfall": {"24h_mm": 0, "3d_mm": 0, "7d_mm": 0, "14d_mm": 0, "30d_mm": 0},
            "forecast": {"24h_mm": 0, "72h_mm": 0},
            "soil_moisture": {"available": False, "value": None, "unit": "m³/m³", "metric": "Topsoil Moisture", "depth": "0-7 cm", "source": "Fallback (API Error)"},
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source": "Fallback"
        }

def process_weather_data(data: dict) -> dict:
    """
    Extracts past/forecast rainfall and current soil moisture.
    """
    hourly_time = data["hourly"]["time"]
    hourly_precip = data["hourly"]["precipitation"]
    hourly_soil = data["hourly"].get("soil_moisture_0_to_7cm", [])
    
    times = [parser.parse(t) for t in hourly_time]

    # Hourly times are local to the location; split past/forecast at the current local hour
    local_now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=data.get("utc_offset_seconds", 0))
    current_index = sum(1 for t in times if t <= local_now)
    if current_index == 0 or current_index >= len(times):
        current_index = min(31 * 24, len(times) - 1)
        
    precip = [p if p is not None else 0.0 for p in hourly_precip]
    forecast_precip = precip[current_index:]

    def windows_ending(idx):
        """Rainfall windows over the hours before `idx` (now, or a forecast horizon)."""
        past = precip[:idx]

        def last(n_days):
            return sum(past[-n_days * 24:])

        days = [sum(past[max(0, len(past) - (k + 1) * 24):len(past) - k * 24]) for k in range(7)]
        return {
            "24h_mm": round(last(1), 2),
            "3d_mm": round(last(3), 2),
            "7d_mm": round(last(7), 2),
            "14d_mm": round(last(14), 2),
            "30d_mm": round(last(30), 2),
            "hours_24h": sum(1 for p in past[-24:] if p >= 0.1),
            "max_daily_7d_mm": round(max(days) if days else 0.0, 2),
        }

    def soil_at(idx):
        return hourly_soil[idx - 1] if 0 < idx <= len(hourly_soil) else None

    def sum_next_n(n_days):
        return sum(forecast_precip[:n_days * 24])

    current_soil_moisture = soil_at(current_index)
    horizons = {}
    for label, hours in (("24h", 24), ("72h", 72)):
        idx = min(current_index + hours, len(precip))
        horizons[label] = {"rainfall": windows_ending(idx), "soil_moisture": soil_at(idx)}

    metrics = {
        "rainfall": windows_ending(current_index),
        "horizons": horizons,
        "forecast": {
            "24h_mm": round(sum_next_n(1), 2),
            "72h_mm": round(sum_next_n(3), 2)
        },
        "soil_moisture": {
            "available": current_soil_moisture is not None,
            "value": current_soil_moisture,
            "unit": "m³/m³",
            "metric": "Topsoil Moisture",
            "depth": "0-7 cm",
            "source": "Open-Meteo"
        }
    }
    return metrics
