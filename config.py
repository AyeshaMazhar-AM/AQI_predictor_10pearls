"""
Central configuration for the AQI Predictor project.
All pipelines import from here so credentials/settings live in one place.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API credentials ---
AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN")
# Weather comes from Open-Meteo (api.open-meteo.com) — no API key needed.
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# --- Location settings ---
CITY = os.getenv("CITY", "karachi")
CITY_LAT = float(os.getenv("CITY_LAT", "24.8607"))
CITY_LON = float(os.getenv("CITY_LON", "67.0011"))

# --- Feature store settings ---
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

# --- Sanity check helper ---
def validate_config():
    missing = []
    if not HOPSWORKS_API_KEY or HOPSWORKS_API_KEY == "your_hopsworks_key_here":
        missing.append("HOPSWORKS_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            f"Copy .env.example to .env and fill in your keys."
        )
