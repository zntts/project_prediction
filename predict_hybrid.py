import pandas as pd
import numpy as np
import joblib
import os

# Prediction configuration
MODEL_NAME = 'hybrid'
MODEL_PATH = f'models/optimized_xgb_{MODEL_NAME}_model.pkl'
DATA_PATH = 'outData/hybrid_energy_weather.csv'
TARGET_COL = 'hybrid_generation_actual'

# Load model
print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)

# Helper functions
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

# Load and preprocess data
df = pd.read_csv(DATA_PATH, parse_dates=['utc_timestamp'])
if MODEL_NAME == 'hybrid':
    df = df.loc[:, ~df.columns.str.endswith('_y')]
    df.columns = df.columns.str.replace('_x', '')

# Keep original timestamp for later
timestamps = df['utc_timestamp'].copy()

df = createTimeFeatures(df)
df = createLagFeatures(df, TARGET_COL)
df = addInteractionFeatures(df)

# Add utc_timestamp back to df before dropping
df['utc_timestamp'] = timestamps

# Drop rows with NA after feature engineering
df_clean = df.dropna()

# Separate features and target
X = df_clean.drop(columns=[TARGET_COL, 'utc_timestamp'])
y = df_clean[TARGET_COL]

# Predict
print("Making predictions...")
y_pred = model.predict(X)

# Evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y, y_pred)
mae_val = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nPrediction Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae_val:.4f}")
print(f"R²: {r2:.4f}")

# Save results with utc_timestamp
output_df = df_clean[['utc_timestamp']].copy()
output_df['predicted_energy_generation'] = y_pred
output_df.to_csv(f'predictions/{MODEL_NAME}_predictions.csv', index=False)

print(f"Predictions with timestamps saved to models/{MODEL_NAME}_predictions.csv")
