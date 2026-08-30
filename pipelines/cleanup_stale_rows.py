"""
One-off cleanup: inspect the feature group for the duplicate/stale rows
inserted while the AQICN 'karachi' station was frozen at aqi=161, pm25=161
(discovered to be dead since March 2025). Run this once to review, then
retrain — training_pipeline.py has already been updated to automatically
filter these rows out.

Run:
    python pipelines/cleanup_stale_rows.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hopsworks
import config

project = hopsworks.login(
    api_key_value=config.HOPSWORKS_API_KEY,
    project=config.HOPSWORKS_PROJECT_NAME,
)
fs = project.get_feature_store()
fg = fs.get_feature_group(config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
df = fg.read()

print(f"Total rows: {len(df)}")

# The stale AQICN station always returned exactly aqi=161, pm25=161.
# Real readings vary, so an exact repeat of this specific combo is almost
# certainly one of the contaminated test rows, not genuine data.
suspect = df[(df["aqi"] == 161) & (df["pm25"] == 161)]
print(f"Suspected stale/duplicate rows (aqi=161, pm25=161): {len(suspect)}")
print(suspect[["timestamp", "aqi", "pm25"]].to_string())

print(
    "\nThese rows are artifacts of testing against a dead AQICN station, "
    "not genuine readings. training_pipeline.py now filters them out "
    "automatically before training, so no manual deletion is required."
)
