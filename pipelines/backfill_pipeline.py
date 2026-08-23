"""
Backfill Pipeline
==================
Bootstraps historical training data from a downloaded CSV file, applies the
SAME feature engineering as feature_pipeline.py, and bulk-inserts everything
into the same Hopsworks feature group. This lets your training pipeline have
weeks/months of data on day one, while feature_pipeline.py keeps adding real
live rows on top of it every hour going forward.

WHERE TO GET A HISTORICAL CSV (pick one):
------------------------------------------
1. AQICN's own historical data platform (best fit, matches your API):
   https://aqicn.org/data-platform/register/
   -> free registration -> download historical CSV for your city's station

2. Kaggle - search "Air Quality Index" or "<your city> air quality dataset"
   e.g. https://www.kaggle.com/datasets?search=air+quality

3. OpenAQ (openaq.org) - free historical air quality data via API, covers
   many cities globally, good fallback if #1/#2 don't have your city.

HOW TO USE THIS SCRIPT:
------------------------
1. Download a CSV from one of the sources above into data/historical_raw.csv
2. Open the COLUMN_MAPPING section below and map YOUR file's column names
   to our standard names (left side = our name, right side = your CSV's
   column name). Delete/comment out any you don't have.
3. Run: python pipelines/backfill_pipeline.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import timezone

import config


# ---------------------------------------------------------------------------
# CONFIGURE THIS FOR YOUR DOWNLOADED CSV
# ---------------------------------------------------------------------------

RAW_CSV_PATH = "data/historical_raw.csv"   # <-- change to your downloaded file

# Map OUR standard feature names -> the column name in YOUR csv.
# Set any value to None if that column doesn't exist in your file — the
# script will fill it with NaN and still work (models handle missing values).
COLUMN_MAPPING = {
    "datetime": "date",       # REQUIRED — any column with a parseable date/time
    "aqi":      "aqi",        # REQUIRED — the target variable
    "pm25":     "pm25",
    "pm10":     "pm10",
    "o3":       "o3",
    "no2":      "no2",
    "so2":      "so2",
    "co":       "co",
    "temp":     None,         # e.g. "temperature" if your file has it
    "humidity": None,
    "wind_speed": None,
    "pressure": None,
}


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN RAW CSV
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: str, mapping: dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if mapping["datetime"] not in df.columns:
        raise ValueError(
            f"Datetime column '{mapping['datetime']}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    if mapping["aqi"] not in df.columns:
        raise ValueError(
            f"AQI column '{mapping['aqi']}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[mapping["datetime"]], errors="coerce", utc=True)

    for feature_name, col_name in mapping.items():
        if feature_name == "datetime":
            continue
        if col_name is not None and col_name in df.columns:
            out[feature_name] = pd.to_numeric(df[col_name], errors="coerce")
        else:
            out[feature_name] = np.nan

    # Drop rows with no timestamp or no AQI — useless for training
    out = out.dropna(subset=["timestamp", "aqi"])
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    print(f"[OK] Loaded {len(out)} valid historical rows "
          f"({out['timestamp'].min()} -> {out['timestamp'].max()})")
    return out


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (mirrors feature_pipeline.py exactly)
# ---------------------------------------------------------------------------

def engineer_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["unix_time"] = df["timestamp"].astype("int64") // 10**9
    df["city"] = config.CITY

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # AQI change rate vs the previous chronological row
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0.0)

    # Column order matching the live feature group schema
    ordered_cols = [
        "timestamp", "unix_time", "city",
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temp", "humidity", "wind_speed", "pressure",
        "hour", "day_of_week", "day_of_month", "month", "is_weekend",
        "aqi_change_rate",
    ]
    return df[ordered_cols]


# ---------------------------------------------------------------------------
# 3. BULK PUSH TO HOPSWORKS
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
    )

    total = len(df)
    for start in range(0, total, batch_size):
        chunk = df.iloc[start:start + batch_size]
        fg.insert(chunk, write_options={"wait_for_job": False})
        print(f"[OK] Inserted rows {start} -> {start + len(chunk)} of {total}")

    print(f"[DONE] Backfilled {total} historical rows into "
          f"'{config.FEATURE_GROUP_NAME}'.")


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

def run(push_to_hopsworks: bool = True):
    config.validate_config()

    print(f"[1/3] Loading raw CSV from '{RAW_CSV_PATH}'...")
    raw_df = load_and_clean(RAW_CSV_PATH, COLUMN_MAPPING)

    print("[2/3] Engineering features for all historical rows...")
    features_df = engineer_features_batch(raw_df)
    print(features_df.head())

    if push_to_hopsworks:
        print("[3/3] Pushing batch to Hopsworks Feature Store...")
        push_batch_to_feature_store(features_df)
    else:
        print("[3/3] Skipped Hopsworks push (dry run only). "
              f"Would have inserted {len(features_df)} rows.")
        features_df.to_csv("data/backfill_preview.csv", index=False)
        print("Saved preview to data/backfill_preview.csv for inspection.")

    return features_df


if __name__ == "__main__":
    # Do a dry run FIRST to sanity-check the engineered features before
    # writing thousands of rows to Hopsworks.
    run(push_to_hopsworks=False)
