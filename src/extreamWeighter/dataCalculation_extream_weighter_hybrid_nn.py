import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model

# File path
config = {
    'name': 'hybrid',
    'data_path': 'outData/hybrid_energy_weather.csv',
    'target': 'hybrid_generation_actual',
    'model_path': 'models_nn/hybrid/nn_hybrid_best_model.h5'
}

print(f"\nProcessing {config['name'].upper()}...")

df = pd.read_csv(config['data_path'], parse_dates=['utc_timestamp'])

# Create Time feature
df['hour'] = df['utc_timestamp'].dt.hour
df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
df['month'] = df['utc_timestamp'].dt.month
df['year'] = df['utc_timestamp'].dt.year
df['quarter'] = df['utc_timestamp'].dt.quarter
df['day_of_year'] = df['utc_timestamp'].dt.dayofyear
df['lag_24'] = df[config['target']].shift(24)
df['lag_48'] = df[config['target']].shift(48)

# Clean column names
df = df.dropna()
df.columns = df.columns.str.replace(r"[\[\]<>\(\)]", "", regex=True).str.replace(" ", "_")

# Prepare features and labels
X = df.drop(columns=[config['target'], 'utc_timestamp'], errors='ignore')
y = df[config['target']]

X_test = X[int(len(X)*0.8):].copy()
y_test = y[int(len(y)*0.8):]

# Extreme weather conditions
extreme_conditions = (
    (X_test.filter(like='Shortwave_Radiation').iloc[:, 0] < 50) |
    (X_test.filter(like='Wind_Gust').iloc[:, 0] > 20) |
    (X_test.filter(like='Temperature_2_m_elevation_corrected').iloc[:, 0] < 0)
)

X_extreme = X_test[extreme_conditions]
y_extreme = y_test[extreme_conditions]

# Selected features
selected_features = [
    "Basel_Temperature_2_m_elevation_corrected_x",
    "Basel_Growing_Degree_Days_2_m_elevation_corrected_x",
    "Basel_Temperature_900_mb_x",
    "Basel_Temperature_850_mb_x",
    "Basel_Temperature_800_mb_x",
    "Basel_Temperature_700_mb_x",
    "Basel_Temperature_500_mb_x",
    "Basel_Precipitation_Total_x",
    "Basel_Relative_Humidity_2_m_x",
    "Basel_Snowfall_Amount_x",
    "Basel_Snow_Depth_x",
    "Basel_Wind_Gust_x",
    "Basel_Wind_Speed_10_m_x",
    "Basel_Wind_Direction_10_m_x",
    "Basel_Wind_Speed_100_m_x",
    "Basel_Wind_Direction_100_m_x",
    "Basel_Wind_Speed_900_mb_x",
    "Basel_Wind_Direction_900_mb_x",
    "Basel_Wind_Speed_850_mb_x",
    "Basel_Wind_Direction_850_mb_x",
    "Basel_Wind_Speed_800_mb_x",
    "Basel_Wind_Direction_800_mb_x",
    "Basel_Wind_Speed_700_mb_x",
    "Basel_Wind_Direction_700_mb_x",
    "Basel_Wind_Speed_500_mb_x",
    "Basel_Wind_Direction_500_mb_x",
    "Basel_Wind_Speed_250_mb_x",
    "Basel_Wind_Direction_250_mb_x",
    "Basel_Cloud_Cover_Total_x",
    "Basel_Cloud_Cover_High_high_cld_lay_x",
    "Basel_Cloud_Cover_Medium_mid_cld_lay_x",
    "Basel_Cloud_Cover_Low_low_cld_lay_x",
    "Basel_CAPE_180-0_mb_above_gnd_x",
    "Basel_Sunshine_Duration_x",
    "Basel_Shortwave_Radiation_x",
    "Basel_Longwave_Radiation_x",
    "Basel_UV_Radiation_x",
    "Basel_Direct_Shortwave_Radiation_x",
    "Basel_Diffuse_Shortwave_Radiation_x",
    "Basel_Mean_Sea_Level_Pressure_MSL_x",
    "Basel_Geopotential_Height_1000_mb_x",
    "Basel_Geopotential_Height_850_mb_x",
    "Basel_Geopotential_Height_800_mb_x",
    "Basel_Geopotential_Height_700_mb_x",
    "Basel_Geopotential_Height_500_mb_x",
    "Basel_Evapotranspiration_x",
    "Basel_Potential_Evaporation_x",
    "Basel_FAO_Reference_Evapotranspiration_2_m_x",
    "Basel_Vapor_Pressure_Deficit_2_m_x",
    "Basel_PBL_Height_x",
    "Basel_Temperature_x",
    "Basel_Soil_Temperature_0-7_cm_down_x",
    "Basel_Soil_Temperature_7-28_cm_down_x",
    "Basel_Soil_Temperature_28-100_cm_down_x",
    "Basel_Soil_Temperature_100-255_cm_down_x",
    "Basel_Soil_Moisture_0-7_cm_down_x",
    "Basel_Soil_Moisture_7-28_cm_down_x",
    "Basel_Soil_Moisture_28-100_cm_down_x",
    "Basel_Soil_Moisture_100-255_cm_down_x",
    "Basel_Soil_Moisture_Available_To_Plant_0-7_cm_down_x",
    "Basel_Soil_Moisture_Available_To_Plant_7-28_cm_down_x",
    "Basel_Soil_Moisture_Available_To_Plant_28-100_cm_down_x",
    "Basel_Soil_Moisture_Available_To_Plant_100-255_cm_down_x",
    "Basel_Temperature_2_m_elevation_corrected_y",
    "Basel_Growing_Degree_Days_2_m_elevation_corrected_y",
    "Basel_Temperature_900_mb_y",
    "Basel_Temperature_850_mb_y",
    "Basel_Temperature_800_mb_y",
    "Basel_Temperature_700_mb_y",
    "Basel_Temperature_500_mb_y",
    "Basel_Precipitation_Total_y",
    "Basel_Relative_Humidity_2_m_y",
    "Basel_Snowfall_Amount_y",
    "Basel_Snow_Depth_y",
    "Basel_Wind_Gust_y",
    "Basel_Wind_Speed_10_m_y",
    "Basel_Wind_Direction_10_m_y"
]

# Error handeling
missing = [feat for feat in selected_features if feat not in X_extreme.columns]
if missing:
    raise ValueError(f"Some selected features are missing in the hybrid dataset: {missing}")

X_extreme = X_extreme[selected_features]

# Load model and predict
model = load_model(config['model_path'])
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_extreme)
y_pred_extreme = model.predict(X_scaled).flatten()

# Metrics calculation
mse_ext = mean_squared_error(y_extreme, y_pred_extreme)
mae_ext = mean_absolute_error(y_extreme, y_pred_extreme)
r2_ext = r2_score(y_extreme, y_pred_extreme)

print(f"\nExtreme Weather Performance (NN - {config['name'].capitalize()}):")
print(f"MAE: {mae_ext:.2f}")
print(f"MSE: {mse_ext:.2f}")
print(f"R² : {r2_ext:.4f}")

# Plot graph
plt.figure(figsize=(12, 5))
plt.plot(y_extreme.values, label='Actual', color='green')
plt.plot(y_pred_extreme, label='NN Prediction', color='blue', linestyle='--')
plt.title(f"{config['name'].capitalize()} Forecast During Extreme Weather (NN)")
textstr = f"MAE: {mae_ext:.2f}\nMSE: {mse_ext:.2f}\nR² : {r2_ext:.4f}"
plt.gcf().text(0.78, 0.55, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.xlabel("Time Step (Extreme Cases)")
plt.ylabel("Energy (MW)")
plt.legend()
plt.grid(True)

plt.tight_layout()
os.makedirs(f"models_nn/{config['name']}", exist_ok=True)
plt.savefig(f"models_nn/{config['name']}/{config['name']}_extreme_weather_plot_nn.png")
plt.close()
