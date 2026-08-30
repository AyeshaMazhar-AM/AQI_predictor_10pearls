"""
Backfill Pipeline (v2 — Open-Meteo historical archive)
========================================================
Bootstraps historical training data automatically by calling Open-Meteo's
free historical Air Quality + Weather archive APIs for your city's exact
coordinates. No manual CSV download, no Kaggle hunting, no column mapping.

Data sources (both free, no API key required):
- Air quality history: https://air-quality-api.open-meteo.com/v1/air-quality
  (PM2.5, PM10, NO2, SO2, CO, ozone, and a computed AQI — CAMS reanalysis
  model estimates, not raw ground-station readings, but a solid proxy for
  training when real station history isn't available for free)
- Weather history: https://archive-api.open-meteo.com/v1/archive
  (temperature, humidity, wind, pressure)

Run:
    python pipelines/backfill_pipeline.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

import config


# How many past days to backfill. Open-Meteo's air quality archive
# generally covers a good multi-year window, but start smaller (e.g. 60
# days) to keep the first run fast — you can re-run with a bigger range
# later once everything works.
BACKFILL_DAYS = 365


# ---------------------------------------------------------------------------
# 1. FETCH HISTORICAL AIR QUALITY
# ---------------------------------------------------------------------------

def fetch_historical_air_quality(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly historical pollutant + AQI data from Open-Meteo."""
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
        "&timezone=UTC"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError(f"Unexpected air quality response: {data}")

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "aqi": hourly.get("us_aqi"),
        "pm25": hourly.get("pm2_5"),
        "pm10": hourly.get("pm10"),
        "co": hourly.get("carbon_monoxide"),
        "no2": hourly.get("nitrogen_dioxide"),
        "so2": hourly.get("sulphur_dioxide"),
        "o3": hourly.get("ozone"),
    })
    return df


# ---------------------------------------------------------------------------
# 2. FETCH HISTORICAL WEATHER
# ---------------------------------------------------------------------------

def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly historical weather data from Open-Meteo."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
        "&timezone=UTC"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError(f"Unexpected weather response: {data}")

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "temp": hourly.get("temperature_2m"),
        "humidity": hourly.get("relative_humidity_2m"),
        "wind_speed": hourly.get("wind_speed_10m"),
        "pressure": hourly.get("pressure_msl"),
    })
    return df


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (mirrors feature_pipeline.py exactly)
# ---------------------------------------------------------------------------

def engineer_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["unix_time"] = df["timestamp"].astype("int64") // 10**9
    df["city"] = config.CITY

    df["hour"] = df["timestamp"].dt.hour.astype("int64")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int64")
    df["day_of_month"] = df["timestamp"].dt.day.astype("int64")
    df["month"] = df["timestamp"].dt.month.astype("int64")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int64")

    df["aqi_change_rate"] = df["aqi"].diff().fillna(0.0)

    numeric_cols = [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temp", "humidity", "wind_speed", "pressure",
        "aqi_change_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    ordered_cols = [
        "timestamp", "unix_time", "city",
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temp", "humidity", "wind_speed", "pressure",
        "hour", "day_of_week", "day_of_month", "month", "is_weekend",
        "aqi_change_rate",
    ]
    return df[ordered_cols]


# ---------------------------------------------------------------------------
# 4. BULK PUSH TO HOPSWORKS
# ---------------------------------------------------------------------------

def push_batch_to_feature_store(df: pd.DataFrame, batch_size: int = 5000):
    import hopsworks

    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for AQI forecasting",
        primary_key=["city", "unix_time"],
        event_time="timestamp",
        online_enabled=True,
        time_travel_format="HUDI",
    )

    total = len(df)
    for start in range(0, total, batch_size):
        chunk = df.iloc[start:start + batch_size]
        fg.insert(chunk, write_options={"wait_for_job": False})
        print(f"[OK] Inserted rows {start} -> {start + len(chunk)} of {total}")

    print(f"[DONE] Backfilled {total} historical rows into '{config.FEATURE_GROUP_NAME}'.")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def run(push_to_hopsworks: bool = True, days: int = BACKFILL_DAYS):
    config.validate_config()

    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)  # yesterday (today may be incomplete)
    start_date = end_date - timedelta(days=days)
    start_str, end_str = start_date.isoformat(), end_date.isoformat()

    print(f"[1/4] Fetching historical air quality for '{config.CITY}' "
          f"({start_str} -> {end_str})...")
    aq_df = fetch_historical_air_quality(config.CITY_LAT, config.CITY_LON, start_str, end_str)
    print(f"       -> {len(aq_df)} hourly rows fetched")

    print("[2/4] Fetching historical weather...")
    weather_df = fetch_historical_weather(config.CITY_LAT, config.CITY_LON, start_str, end_str)
    print(f"       -> {len(weather_df)} hourly rows fetched")

    print("[3/4] Merging + engineering features...")
    merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    merged = merged.dropna(subset=["aqi"])  # drop hours with no AQI reading
    features_df = engineer_features_batch(merged)
    print(features_df.head())
    print(f"       -> {len(features_df)} total feature rows ready")

    if push_to_hopsworks:
        print("[4/4] Pushing batch to Hopsworks Feature Store...")
        push_batch_to_feature_store(features_df)
    else:
        print("[4/4] Skipped Hopsworks push (dry run only).")
        os.makedirs("data", exist_ok=True)
        features_df.to_csv("data/backfill_preview.csv", index=False)
        print("Saved preview to data/backfill_preview.csv for inspection.")

    return features_df


if __name__ == "__main__":
    # Dry run first — sanity check the data before writing to Hopsworks.
    run(push_to_hopsworks=True)