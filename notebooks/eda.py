"""
Exploratory Data Analysis (EDA)
================================
Pulls all historical data from the Hopsworks Feature Store, generates
trend/distribution charts, and writes a markdown summary report — the
"Perform EDA to identify trends" requirement from the project brief.

Run:
    python notebooks/eda.py

Output:
    notebooks/eda_output/*.png   (charts)
    notebooks/eda_output/eda_report.md   (summary you can drop into your
                                           final report)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt

import config

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

POLLUTANTS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
WEATHER = ["temp", "humidity", "wind_speed", "pressure"]


def load_data() -> pd.DataFrame:
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

    # Exclude the known contaminated test rows (see training_pipeline.py)
    df = df[~((df["aqi"] == 161) & (df["pm25"] == 161))].reset_index(drop=True)
    return df


def plot_aqi_timeseries(df: pd.DataFrame):
    plt.figure(figsize=(11, 4))
    plt.plot(df["timestamp"], df["aqi"], linewidth=0.8, color="#2c6fbb")
    plt.axhline(50, color="green", linestyle="--", linewidth=0.6, alpha=0.5, label="Good/Moderate")
    plt.axhline(100, color="orange", linestyle="--", linewidth=0.6, alpha=0.5, label="Moderate/Unhealthy(SG)")
    plt.axhline(150, color="red", linestyle="--", linewidth=0.6, alpha=0.5, label="Unhealthy")
    plt.title("AQI over time")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_aqi_timeseries.png"), dpi=120)
    plt.close()


def plot_aqi_distribution(df: pd.DataFrame):
    plt.figure(figsize=(7, 4))
    plt.hist(df["aqi"].dropna(), bins=40, color="#2c6fbb", edgecolor="white")
    plt.title("Distribution of AQI values")
    plt.xlabel("AQI")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_aqi_distribution.png"), dpi=120)
    plt.close()


def plot_hourly_pattern(df: pd.DataFrame):
    hourly = df.groupby("hour")["aqi"].mean()
    plt.figure(figsize=(8, 4))
    plt.bar(hourly.index, hourly.values, color="#4C9F70")
    plt.title("Average AQI by hour of day")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Mean AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_hourly_pattern.png"), dpi=120)
    plt.close()
    return hourly


def plot_weekday_pattern(df: pd.DataFrame):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday = df.groupby("day_of_week")["aqi"].mean()
    plt.figure(figsize=(7, 4))
    plt.bar([day_names[i] for i in weekday.index], weekday.values, color="#D88B3E")
    plt.title("Average AQI by day of week")
    plt.ylabel("Mean AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_weekday_pattern.png"), dpi=120)
    plt.close()
    return weekday


def plot_correlations(df: pd.DataFrame):
    cols = ["aqi"] + [c for c in POLLUTANTS + WEATHER if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Feature correlation with AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_correlations.png"), dpi=120)
    plt.close()
    return corr["aqi"].drop("aqi").sort_values(key=abs, ascending=False)


def plot_pollutant_trends(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for ax, col in zip(axes.flat, POLLUTANTS):
        if col in df.columns:
            ax.plot(df["timestamp"], df[col], linewidth=0.6, color="#7A5195")
            ax.set_title(col.upper())
    fig.suptitle("Pollutant concentrations over time")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_pollutant_trends.png"), dpi=120)
    plt.close()


def write_report(df: pd.DataFrame, hourly, weekday, corr_with_aqi):
    peak_hour = int(hourly.idxmax())
    low_hour = int(hourly.idxmin())
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    peak_day = day_names[int(weekday.idxmax())]

    report = f"""# EDA Report — {config.CITY.title()} AQI

**Data range:** {df['timestamp'].min()} to {df['timestamp'].max()}
**Total rows analyzed:** {len(df)}

## Summary statistics

{df[['aqi'] + [c for c in POLLUTANTS if c in df.columns]].describe().round(2).to_markdown()}

## Key trends

- **AQI over time:** see `01_aqi_timeseries.png`. Dashed lines mark EPA
  thresholds (50=Good/Moderate boundary, 100=Moderate/Unhealthy-for-
  sensitive-groups, 150=Unhealthy).
- **Distribution:** see `02_aqi_distribution.png` — shows how often each
  AQI range occurs.
- **Hourly pattern:** AQI tends to peak around **{peak_hour}:00 UTC** and
  is lowest around **{low_hour}:00 UTC**. See `03_hourly_pattern.png`.
- **Weekly pattern:** **{peak_day}** shows the highest average AQI. See
  `04_weekday_pattern.png`.
- **Correlations with AQI** (strongest first):
{corr_with_aqi.round(3).to_string()}
  See `05_correlations.png` for the full matrix.
- **Pollutant trends:** see `06_pollutant_trends.png` for individual
  pollutant concentration trends over the collection period.

## Notes on data provenance

Historical rows (before live hourly collection began) were backfilled
using Open-Meteo's CAMS atmospheric reanalysis model. Live hourly rows use
Open-Meteo's current air quality estimate (switched from AQICN after
discovering the AQICN ground station for this city had been inactive
since March 2025). A small number of contaminated test rows
(aqi=161, pm25=161 exact duplicates, from testing against the dead AQICN
station) were excluded from this analysis.
"""
    with open(os.path.join(OUT_DIR, "eda_report.md"), "w") as f:
        f.write(report)
    print(report)


def run():
    config.validate_config()
    print("[1/3] Loading data from Hopsworks...")
    df = load_data()
    print(f"       -> {len(df)} rows (after excluding contaminated test rows)")

    print("[2/3] Generating charts...")
    plot_aqi_timeseries(df)
    plot_aqi_distribution(df)
    hourly = plot_hourly_pattern(df)
    weekday = plot_weekday_pattern(df)
    corr_with_aqi = plot_correlations(df)
    plot_pollutant_trends(df)

    print("[3/3] Writing report...")
    write_report(df, hourly, weekday, corr_with_aqi)
    print(f"\nDone. See {OUT_DIR}/ for charts and eda_report.md")


if __name__ == "__main__":
    run()
