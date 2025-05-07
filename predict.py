import pandas as pd
import joblib
import os
from datetime import datetime

# File path
MODEL_TYPE = 'wind_onshore'
DATA_PATH = f'outData/{MODEL_TYPE}_energy_weather.csv'
MODEL_PATH = f'models/optimized_xgb_{MODEL_TYPE}_model.pkl'
OUTPUT_PATH = f'predictions/{MODEL_TYPE}_predictions.csv'
TARGET_COLUMN = {
    'solar': 'GB_GBN_solar_generation_actual',
    'wind': 'GB_GBN_wind_generation_actual',
    'wind_offshore': 'GB_GBN_wind_offshore_generation_actual',
    'wind_onshore': 'GB_GBN_wind_onshore_generation_actual'
}[MODEL_TYPE]

# Time feature
def createTimeFeatures(df):
    df['hour'] = df['utc_timestamp'].dt.hour
    df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
    df['month'] = df['utc_timestamp'].dt.month
    df['year'] = df['utc_timestamp'].dt.year
    df['quarter'] = df['utc_timestamp'].dt.quarter
    df['day_of_year'] = df['utc_timestamp'].dt.dayofyear
    return df

def createLagFeatures(df, target_col, lags=[24, 48]):
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df

def addInteractionFeatures(df):
    df['temp_x_wind'] = df['Basel Temperature [2 m elevation corrected]'] * df['Basel Wind Speed [10 m]']
    df['humidity_div_radiation'] = df['Basel Relative Humidity [2 m]'] / (df['Basel Shortwave Radiation'] + 1e-6)
    df['cloud_x_uv'] = df['Basel Cloud Cover Total'] * df['Basel UV Radiation']
    df['hour_x_radiation'] = df['hour'] * df['Basel Shortwave Radiation']
    df['wind_gust_diff'] = df['Basel Wind Gust'] - df['Basel Wind Speed [10 m]']
    return df

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=['utc_timestamp'])
df = createTimeFeatures(df)
df = createLagFeatures(df, TARGET_COLUMN)
df = addInteractionFeatures(df)

# Drop rows with NaNs introduced by lags
df = df.dropna()
timestamps = df['utc_timestamp']  # Save timestamps for output
X = df.drop(columns=['utc_timestamp', TARGET_COLUMN])

# Load trained model
model = joblib.load(MODEL_PATH)

# Predictions
y_pred = model.predict(X)

# Save results
os.makedirs('predictions', exist_ok=True)
results = pd.DataFrame({
    'utc_timestamp': timestamps,
    'predicted_energy_generation': y_pred
})
results.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")

print(results.tail())