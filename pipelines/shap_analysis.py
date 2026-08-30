"""
SHAP Feature Importance Analysis
==================================
Loads the newest trained model for each forecast horizon from the Hopsworks
Model Registry, computes SHAP values against real training data, and saves
a feature-importance bar chart per horizon — the "Use SHAP or LIME for
feature importance explanations" requirement from the project brief.

Run:
    python pipelines/shap_analysis.py

Output:
    notebooks/eda_output/shap_<horizon>.png   (one chart per horizon)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import config

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = ["24h", "48h", "72h"]
FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temp", "humidity", "wind_speed", "pressure",
    "hour", "day_of_week", "day_of_month", "month", "is_weekend",
    "aqi_change_rate",
]


def load_training_data() -> pd.DataFrame:
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
    df = df[~((df["aqi"] == 161) & (df["pm25"] == 161))].reset_index(drop=True)

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median(numeric_only=True))
    return df, project


def load_latest_model(project, horizon: str):
    mr = project.get_model_registry()
    all_versions = mr.get_models(name=f"aqi_model_{horizon}")
    latest_version = max(m.version for m in all_versions)
    model_obj = mr.get_model(f"aqi_model_{horizon}", version=latest_version)
    model_dir = model_obj.download()

    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    if metadata["algorithm"] == "neural_net":
        return None, metadata  # skip — SHAP setup differs for keras, not covered here
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    return model, metadata


def analyze_horizon(model, algorithm: str, X: pd.DataFrame, horizon: str):
    # Use a small background sample for speed — SHAP doesn't need every row.
    sample = X.sample(min(200, len(X)), random_state=42)

    if algorithm == "ridge":
        explainer = shap.LinearExplainer(model, sample)
    else:  # random_forest or other tree-based
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    importance.head(12).sort_values().plot.barh(color="#2c6fbb")
    plt.title(f"SHAP feature importance — {horizon} forecast ({algorithm})")
    plt.xlabel("Mean |SHAP value| (impact on prediction)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_{horizon}.png"), dpi=120)
    plt.close()

    return importance


def run():
    config.validate_config()
    print("[1/2] Loading training data + models...")
    df, project = load_training_data()
    X = df[FEATURE_COLUMNS]

    summaries = {}
    for horizon in HORIZONS:
        print(f"\n[2/2] Analyzing {horizon}...")
        model, metadata = load_latest_model(project, horizon)
        if model is None:
            print(f"       [skip] {horizon} uses a neural net — SHAP for keras not covered by this script.")
            continue

        importance = analyze_horizon(model, metadata["algorithm"], X, horizon)
        summaries[horizon] = importance
        print(f"       Top 5 features driving {horizon} predictions:")
        print(importance.head(5).to_string())

    print(f"\nDone. Charts saved to {OUT_DIR}/shap_<horizon>.png")
    return summaries


if __name__ == "__main__":
    run()
