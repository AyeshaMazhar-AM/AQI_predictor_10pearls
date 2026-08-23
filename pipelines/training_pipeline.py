"""
Training Pipeline
==================
1. Fetches all historical features from the Hopsworks Feature Store
2. Builds three target variables: AQI 24h, 48h, and 72h into the future
3. Trains multiple models per horizon (Ridge, Random Forest, optionally a
   small neural network), evaluates with RMSE / MAE / R^2
4. Saves the best model per horizon to the Hopsworks Model Registry

Run:
    python pipelines/training_pipeline.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config

HORIZONS = {"24h": 24, "48h": 48, "72h": 72}  # hours into the future
FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temp", "humidity", "wind_speed", "pressure",
    "hour", "day_of_week", "day_of_month", "month", "is_weekend",
    "aqi_change_rate",
]
MODELS_DIR = "models"


# ---------------------------------------------------------------------------
# 1. FETCH FROM FEATURE STORE
# ---------------------------------------------------------------------------

def fetch_training_data() -> pd.DataFrame:
    import hopsworks

    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
    df = fg.read()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. BUILD FUTURE TARGETS (time-aware, not just row-shift)
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, find the AQI value ~N hours later by matching on actual
    elapsed time (not row position), since the hourly data can have gaps."""
    df = df.copy().set_index("timestamp")

    # Build a clean hourly-frequency AQI series for lookahead matching.
    # Small gaps get linearly interpolated; large gaps stay NaN (so we don't
    # fabricate targets across long missing stretches).
    hourly_aqi = df["aqi"].resample("h").mean().interpolate(limit=6)

    for label, hours in HORIZONS.items():
        future_index = df.index + pd.Timedelta(hours=hours)
        df[f"target_{label}"] = hourly_aqi.reindex(future_index).values

    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# 3. TRAIN / EVALUATE MODELS FOR ONE HORIZON
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_horizon(df: pd.DataFrame, target_col: str) -> dict:
    """Trains Ridge + Random Forest (+ a small NN if TensorFlow is available)
    for one forecast horizon, and returns the best model + its metrics."""
    # Only require the target itself to be present. For feature columns,
    # impute missing values with the column median rather than dropping
    # the row entirely — some pollutants (e.g. pm10, o3) aren't reported
    # by every station/reading, and dropping those rows wastes a lot of
    # otherwise-usable data.
    data = df.dropna(subset=[target_col]).reset_index(drop=True)
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].fillna(data[FEATURE_COLUMNS].median(numeric_only=True))

    if len(data) < 50:
        raise ValueError(
            f"Only {len(data)} usable rows for {target_col} — need more history. "
            f"Run backfill_pipeline.py with a larger BACKFILL_DAYS value."
        )

    # Chronological split — NEVER shuffle time series data.
    split_idx = int(len(data) * 0.85)
    train_df, test_df = data.iloc[:split_idx], data.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[target_col]

    results = {}

    # --- Ridge Regression baseline ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results["ridge"] = {
        "model": ridge,
        "metrics": evaluate(y_test, ridge.predict(X_test)),
    }

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest"] = {
        "model": rf,
        "metrics": evaluate(y_test, rf.predict(X_test)),
    }

    # --- Optional: small neural network (TensorFlow), if installed ---
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        nn = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_s.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        nn.compile(optimizer="adam", loss="mse")
        nn.fit(X_train_s, y_train, epochs=30, batch_size=32, verbose=0,
               validation_split=0.1,
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

        preds = nn.predict(X_test_s, verbose=0).flatten()
        results["neural_net"] = {
            "model": nn,
            "scaler": scaler,
            "metrics": evaluate(y_test, preds),
        }
    except ImportError:
        print("       [skip] TensorFlow not installed — skipping neural net for this horizon.")

    # Pick best by RMSE (lower is better)
    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    for name, r in results.items():
        tag = " <-- BEST" if name == best_name else ""
        m = r["metrics"]
        print(f"       {name:15s} RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}{tag}")

    return {
        "best_name": best_name,
        "best_model": results[best_name]["model"],
        "best_scaler": results[best_name].get("scaler"),
        "all_results": {k: v["metrics"] for k, v in results.items()},
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


# ---------------------------------------------------------------------------
# 4. SAVE TO MODEL REGISTRY
# ---------------------------------------------------------------------------

def save_to_registry(horizon_label: str, result: dict):
    import hopsworks

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_dir = os.path.join(MODELS_DIR, horizon_label)
    os.makedirs(model_dir, exist_ok=True)

    model_name = f"aqi_model_{horizon_label}"
    metrics = result["all_results"][result["best_name"]]

    if result["best_name"] == "neural_net":
        model_path = os.path.join(model_dir, "model.keras")
        result["best_model"].save(model_path)
        joblib.dump(result["best_scaler"], os.path.join(model_dir, "scaler.pkl"))
    else:
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(result["best_model"], model_path)

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump({
            "algorithm": result["best_name"],
            "horizon": horizon_label,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
        }, f, indent=2)

    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    mr = project.get_model_registry()

    hw_model = mr.sklearn.create_model(
        name=model_name,
        metrics=metrics,
        description=f"AQI forecast model, {horizon_label} horizon, algorithm={result['best_name']}",
    )
    hw_model.save(model_dir)
    print(f"       [OK] Saved '{model_name}' ({result['best_name']}) to Model Registry.")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def run(push_to_registry: bool = True):
    config.validate_config()

    print("[1/3] Fetching training data from Hopsworks...")
    df = fetch_training_data()
    print(f"       -> {len(df)} raw rows fetched")

    print("[2/3] Building 24h / 48h / 72h targets...")
    df = build_targets(df)

    summary = {}
    for label in HORIZONS:
        target_col = f"target_{label}"
        print(f"\n[3/3] Training models for horizon: {label}")
        result = train_horizon(df, target_col)
        summary[label] = result

        if push_to_registry:
            save_to_registry(label, result)

    print("\n=== SUMMARY (best model per horizon) ===")
    for label, result in summary.items():
        m = result["all_results"][result["best_name"]]
        print(f"{label:5s} -> {result['best_name']:15s} "
              f"RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    return summary


if __name__ == "__main__":
    # First run: set push_to_registry=False to just see metrics without
    # touching Hopsworks, in case you want to sanity check results first.
    run(push_to_registry=True)
