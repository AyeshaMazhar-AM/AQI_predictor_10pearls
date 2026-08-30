# EDA Report — Karachi AQI

**Data range:** 2026-06-23 00:00:00+00:00 to 2026-08-23 09:17:20.274263+00:00
**Total rows analyzed:** 1473

## Summary statistics

|       |     aqi |    pm25 |    pm10 |      o3 |     no2 |     so2 |      co |
|:------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| count | 1473    | 1473    | 1473    | 1473    | 1473    | 1473    | 1473    |
| mean  |   76.83 |   23.18 |   56.76 |   54.71 |    8.38 |    5.73 |  202.48 |
| std   |    8.22 |    5.93 |   19.62 |   17.1  |    4.1  |    1.59 |   57.9  |
| min   |   61    |   10.9  |   19.6  |   28    |    2.1  |    2.9  |   97    |
| 25%   |   71    |   19    |   42.4  |   43    |    5.3  |    4.7  |  164    |
| 50%   |   76    |   22.5  |   53.6  |   49    |    7.6  |    5.4  |  201    |
| 75%   |   82    |   26.3  |   66.6  |   64    |   10.5  |    6.3  |  236    |
| max   |  109    |   47.7  |  143.4  |  153    |   26.8  |   15.9  |  612    |

## Key trends

- **AQI over time:** see `01_aqi_timeseries.png`. Dashed lines mark EPA
  thresholds (50=Good/Moderate boundary, 100=Moderate/Unhealthy-for-
  sensitive-groups, 150=Unhealthy).
- **Distribution:** see `02_aqi_distribution.png` — shows how often each
  AQI range occurs.
- **Hourly pattern:** AQI tends to peak around **14:00 UTC** and
  is lowest around **2:00 UTC**. See `03_hourly_pattern.png`.
- **Weekly pattern:** **Monday** shows the highest average AQI. See
  `04_weekday_pattern.png`.
- **Correlations with AQI** (strongest first):
pm25          0.664
pressure     -0.629
pm10          0.618
so2           0.417
wind_speed   -0.305
temp          0.304
o3            0.262
no2           0.204
co            0.133
humidity      0.013
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
