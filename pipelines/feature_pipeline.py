"""
Feature Pipeline
=================
Fetches current AQI + pollutant data (AQICN) and weather data (OpenWeather),
engineers features, and writes a single row to the Hopsworks Feature Store.

Run manually:
    python pipelines/feature_pipeline.py

This same script is what GitHub Actions will call every hour once automated.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timezone

import config


# ---------------------------------------------------------------------------
# 1. RAW DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_aqicn_data(city: str, token: str) -> dict:
    """Fetch current AQI + pollutant readings for a city from AQICN."""
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != "ok":
        raise ValueError(f"AQICN API error: {payload}")

    data = payload["data"]
    iaqi = data.get("iaqi", {})

    return {
        "aqi": data.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "station_time": data.get("time", {}).get("iso"),
    }


def fetch_weather_data(lat: float, lon: float) -> dict:
    """Fetch current weather conditions from Open-Meteo.
    No API key required — free for non-commercial use, no rate-limit signup.
    Docs: https://open-meteo.com/en/docs
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current", {})

    return {
        "temp": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "pressure": current.get("pressure_msl"),
    }


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(raw_aqi: dict, raw_weather: dict, previous_aqi: float = None) -> dict:
    """Combine raw sources into a flat feature row with derived features."""
    now = datetime.now(timezone.utc)

    row = {
        "timestamp": now,
        "unix_time": int(now.timestamp()),
        "city": config.CITY,

        # Raw pollutant/AQI features
        "aqi": raw_aqi["aqi"],
        "pm25": raw_aqi["pm25"],
        "pm10": raw_aqi["pm10"],
        "o3": raw_aqi["o3"],
        "no2": raw_aqi["no2"],
        "so2": raw_aqi["so2"],
        "co": raw_aqi["co"],

        # Raw weather features
        "temp": raw_weather["temp"],
        "humidity": raw_weather["humidity"],
        "wind_speed": raw_weather["wind_speed"],
        "pressure": raw_weather["pressure"],

        # Time-based features
        "hour": int(now.hour),
        "day_of_week": int(now.weekday()),          # 0=Monday
        "day_of_month": int(now.day),
        "month": int(now.month),
        "is_weekend": int(now.weekday() >= 5),
    }

    # Derived feature: AQI rate of change vs previous reading (if available)
    if previous_aqi is not None and row["aqi"] is not None:
        row["aqi_change_rate"] = row["aqi"] - previous_aqi
    else:
        row["aqi_change_rate"] = 0.0

    return row


# ---------------------------------------------------------------------------
# 3. FEATURE STORE WRITE (Hopsworks)
# ---------------------------------------------------------------------------

def get_last_aqi_from_feature_store(fs) -> float:
    """Look up the most recent AQI value already stored, used to compute
    the change-rate feature. Returns None if the feature group is empty
    or doesn't exist yet (first run)."""
    try:
        fg = fs.get_feature_group(config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
        df = fg.read()
        if df.empty:
            return None
        df = df.sort_values("unix_time")
        return float(df.iloc[-1]["aqi"])
    except Exception:
        return None


def push_to_feature_store(row: dict):
    """Push a single feature row to Hopsworks. Creates the feature group
    on first run if it doesn't exist yet."""
    import hopsworks

    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()

    df = pd.DataFrame([row])

    # Explicitly cast numeric columns to float64. Without this, a column
    # that is entirely None (e.g. a pollutant your station doesn't report)
    # gets inferred as an ambiguous "null" dtype that Hopsworks rejects.
    numeric_cols = [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temp", "humidity", "wind_speed", "pressure",
        "aqi_change_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for AQI forecasting",
        primary_key=["city", "unix_time"],
        event_time="timestamp",
        online_enabled=True,
        time_travel_format="HUDI",
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"[OK] Inserted 1 row into '{config.FEATURE_GROUP_NAME}' feature group.")


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

def run(push_to_hopsworks: bool = True) -> dict:
    config.validate_config()

    print(f"[1/4] Fetching AQICN data for '{config.CITY}'...")
    raw_aqi = fetch_aqicn_data(config.CITY, config.AQICN_API_TOKEN)

    print("[2/4] Fetching weather data...")
    raw_weather = fetch_weather_data(config.CITY_LAT, config.CITY_LON)

    previous_aqi = None
    if push_to_hopsworks:
        try:
            import hopsworks
            project = hopsworks.login(
                api_key_value=config.HOPSWORKS_API_KEY,
                project=config.HOPSWORKS_PROJECT_NAME,
            )
            fs = project.get_feature_store()
            previous_aqi = get_last_aqi_from_feature_store(fs)
        except Exception as e:
            print(f"[WARN] Could not fetch previous AQI (fine on first run): {e}")

    print("[3/4] Engineering features...")
    row = engineer_features(raw_aqi, raw_weather, previous_aqi)
    print(f"       -> {row}")

    if push_to_hopsworks:
        print("[4/4] Pushing to Hopsworks Feature Store...")
        push_to_feature_store(row)
    else:
        print("[4/4] Skipped Hopsworks push (push_to_hopsworks=False) — dry run only.")

    return row


if __name__ == "__main__":
    # Set push_to_hopsworks=False for your very first local test run,
    # so you can confirm the API calls work before touching Hopsworks.
    run(push_to_hopsworks=True)
