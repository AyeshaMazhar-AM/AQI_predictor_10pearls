# Pearls AQI Predictor

Serverless, end-to-end ML pipeline that predicts Air Quality Index (AQI) for the next 3 days.

## Status
- [x] Phase 1: Project scaffold
- [x] Phase 2: Feature pipeline (fetch + engineer + store)
- [ ] Phase 3: Historical backfill
- [ ] Phase 4: Training pipeline
- [ ] Phase 5: CI/CD automation (GitHub Actions)
- [ ] Phase 6: Streamlit dashboard
- [ ] Phase 7: SHAP explanations + alerts + EDA

## Setup

1. **Clone and install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get your API keys**
   - AQICN token (free): https://aqicn.org/data-platform/token/
   - OpenWeatherMap key (free, optional): https://openweathermap.org/api
   - Hopsworks account + API key (free): https://app.hopsworks.ai

3. **Configure environment**
   ```bash
   cp .env.example .env
   # then edit .env and fill in your keys + city
   ```

4. **Test the feature pipeline (dry run, no Hopsworks write)**

   Open `pipelines/feature_pipeline.py`, change the last line to:
   ```python
   run(push_to_hopsworks=False)
   ```
   Then run:
   ```bash
   python pipelines/feature_pipeline.py
   ```
   You should see the fetched AQI/weather data and engineered feature row printed.

5. **Once the dry run works, enable the Hopsworks write**

   Change `push_to_hopsworks=False` back to `True` and re-run. This creates
   the `aqi_features` feature group in your Hopsworks project on first run.

## Project structure
```
aqi-predictor/
├── config.py                    # central config, loads .env
├── pipelines/
│   ├── feature_pipeline.py      # fetch -> engineer -> push to feature store
│   ├── backfill_pipeline.py     # (next) historical data backfill
│   └── training_pipeline.py     # (next) model training
├── app/
│   └── streamlit_app.py         # (later) dashboard
├── .github/workflows/           # (later) hourly/daily automation
├── requirements.txt
└── .env.example
```

## Finding your AQICN city slug
Your `CITY` value in `.env` must match a station AQICN recognizes. Test it
directly in a browser first:
```
https://api.waqi.info/feed/YOUR_CITY/?token=YOUR_TOKEN
```
If that returns `"status":"ok"`, the slug is valid.
