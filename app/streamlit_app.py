"""
AQI Predictor Dashboard
========================
Loads the 24h/48h/72h models + latest features from Hopsworks, computes a
3-day AQI forecast, and displays it with hazard alerts.

Run locally:
    streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone

import config

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

HORIZONS = ["24h", "48h", "72h"]
FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temp", "humidity", "wind_speed", "pressure",
    "hour", "day_of_week", "day_of_month", "month", "is_weekend",
    "aqi_change_rate",
]

# US EPA-style AQI bands: (upper bound, label, color)
AQI_BANDS = [
    (50, "Good", "#00e400"),
    (100, "Moderate", "#ffff00"),
    (150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (200, "Unhealthy", "#ff0000"),
    (300, "Very Unhealthy", "#8f3f97"),
    (500, "Hazardous", "#7e0023"),
]


def aqi_band(value: float):
    for upper, label, color in AQI_BANDS:
        if value <= upper:
            return label, color
    return "Hazardous", "#7e0023"


# ---------------------------------------------------------------------------
# DATA LOADING (cached so we don't hit Hopsworks on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=600)
def get_hopsworks_project():
    import hopsworks
    return hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )


@st.cache_data(ttl=600)
def load_latest_features() -> pd.DataFrame:
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    return df


@st.cache_data(ttl=600)
def load_history(days: int = 14) -> pd.DataFrame:
    df = load_latest_features()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return df[df["timestamp"] >= cutoff]


@st.cache_resource(ttl=3600)
def load_model(horizon: str):
    """Downloads the saved model + metadata for a given horizon from the
    Hopsworks Model Registry and loads it into memory."""
    import json
    import tensorflow as tf  # only imported if a neural net model exists

    project = get_hopsworks_project()
    mr = project.get_model_registry()
    model_obj = mr.get_model(f"aqi_model_{horizon}")
    model_dir = model_obj.download()

    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    if metadata["algorithm"] == "neural_net":
        model = tf.keras.models.load_model(os.path.join(model_dir, "model.keras"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    else:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        scaler = None

    return {"model": model, "scaler": scaler, "metadata": metadata}


def predict(bundle: dict, latest_row: pd.DataFrame) -> float:
    X = latest_row[FEATURE_COLUMNS]
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)
        pred = bundle["model"].predict(X, verbose=0).flatten()[0]
    else:
        pred = bundle["model"].predict(X)[0]
    return float(pred)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🌫️ AQI Predictor")
st.caption(f"Live 3-day Air Quality Index forecast for **{config.CITY.title()}**")

with st.spinner("Loading latest data..."):
    try:
        history = load_history(days=14)
        latest = history.iloc[[-1]]
    except Exception as e:
        st.error(f"Could not load data from Hopsworks: {e}")
        st.stop()

if history.empty:
    st.warning("No data found yet. Make sure feature_pipeline.py has run at least once.")
    st.stop()

latest_ts = latest["timestamp"].iloc[0]
current_aqi = float(latest["aqi"].iloc[0])
current_label, current_color = aqi_band(current_aqi)

# --- Current conditions ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current AQI", f"{current_aqi:.0f}", current_label)
col2.metric("PM2.5", f"{latest['pm25'].iloc[0]:.1f} µg/m³" if pd.notna(latest['pm25'].iloc[0]) else "—")
col3.metric("Temperature", f"{latest['temp'].iloc[0]:.1f}°C" if pd.notna(latest['temp'].iloc[0]) else "—")
col4.metric("Humidity", f"{latest['humidity'].iloc[0]:.0f}%" if pd.notna(latest['humidity'].iloc[0]) else "—")
st.caption(f"Last updated: {latest_ts.strftime('%Y-%m-%d %H:%M UTC')}")

st.divider()

# --- 3-day forecast ---
st.subheader("3-Day Forecast")

forecast_rows = []
load_errors = []
for h in HORIZONS:
    try:
        bundle = load_model(h)
        pred_value = predict(bundle, latest)
        hours = int(h.replace("h", ""))
        forecast_rows.append({
            "horizon": h,
            "datetime": latest_ts + timedelta(hours=hours),
            "predicted_aqi": pred_value,
            "algorithm": bundle["metadata"]["algorithm"],
        })
    except Exception as e:
        load_errors.append(f"{h}: {e}")

if load_errors:
    st.info(
        "Some forecast models aren't available yet — run training_pipeline.py "
        "to generate them.\n\n" + "\n".join(load_errors)
    )

if forecast_rows:
    forecast_df = pd.DataFrame(forecast_rows)

    # Hazard alert banner
    max_pred = forecast_df["predicted_aqi"].max()
    if max_pred >= 150:
        st.error(
            f"⚠️ **Hazardous air quality expected** — forecast AQI reaches "
            f"{max_pred:.0f} within the next 3 days. Sensitive groups should "
            f"limit outdoor exposure."
        )
    elif max_pred >= 100:
        st.warning(
            f"⚠️ Moderate-to-unhealthy air quality expected — forecast peaks "
            f"at {max_pred:.0f} AQI."
        )
    else:
        st.success(f"✅ Air quality is expected to stay in a healthy range (peak: {max_pred:.0f}).")

    cols = st.columns(len(forecast_df))
    for col, row in zip(cols, forecast_df.itertuples()):
        label, color = aqi_band(row.predicted_aqi)
        with col:
            st.markdown(f"**{row.horizon}** — {row.datetime.strftime('%a %b %d')}")
            st.markdown(
                f"<div style='background-color:{color};padding:16px;border-radius:8px;"
                f"text-align:center;color:#000;font-weight:600;'>"
                f"{row.predicted_aqi:.0f}<br><span style='font-size:0.8em'>{label}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"model: {row.algorithm}")

st.divider()

# --- Historical trend chart ---
st.subheader("Recent AQI Trend (last 14 days)")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history["timestamp"], y=history["aqi"],
    mode="lines", name="Historical AQI", line=dict(color="#4C72B0", width=2),
))
if forecast_rows:
    fig.add_trace(go.Scatter(
        x=[latest_ts] + list(forecast_df["datetime"]),
        y=[current_aqi] + list(forecast_df["predicted_aqi"]),
        mode="lines+markers", name="Forecast",
        line=dict(color="#DD8452", width=2, dash="dash"),
    ))
for upper, label, color in AQI_BANDS:
    fig.add_hline(y=upper, line_dash="dot", line_color=color, opacity=0.3)

fig.update_layout(
    height=400, xaxis_title="Time", yaxis_title="AQI",
    margin=dict(l=20, r=20, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Note: historical data before hourly live collection began was backfilled "
    "using Open-Meteo's CAMS atmospheric model (a reanalysis estimate), while "
    "live hourly readings come from AQICN ground-station data."
)
