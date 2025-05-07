import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# File path
datasets = {
    'solar': {
        'data_path': 'outData/solar_energy_weather.csv',
        'target': 'GB_GBN_solar_generation_actual',
        'model_path': 'models/solar/optimized_xgb_solar_model.pkl'
    },
    'wind': {
        'data_path': 'outData/wind_energy_weather.csv',
        'target': 'GB_GBN_wind_generation_actual',
        'model_path': 'models/wind/optimized_xgb_wind_model.pkl'
    },
    'wind_offshore': {
        'data_path': 'outData/wind_offshore_energy_weather.csv',
        'target': 'GB_GBN_wind_offshore_generation_actual',
        'model_path': 'models/wind_offshore/optimized_xgb_wind_offshore_model.pkl'
    },
    'wind_onshore': {
        'data_path': 'outData/wind_onshore_energy_weather.csv',
        'target': 'GB_GBN_wind_onshore_generation_actual',
        'model_path': 'models/wind_onshore/optimized_xgb_wind_onshore_model.pkl'
    },
    'hybrid': {
        'data_path': 'outData/hybrid_energy_weather.csv',
        'target': 'hybrid_generation_actual',
        'model_path': 'models/hybrid/optimized_xgb_hybrid_model.pkl'
    }
}

def loadData(name, config):
    df = pd.read_csv(config['data_path'], parse_dates=['utc_timestamp'])

    # Time-based features
    df['hour'] = df['utc_timestamp'].dt.hour
    df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
    df['month'] = df['utc_timestamp'].dt.month
    df['year'] = df['utc_timestamp'].dt.year
    df['quarter'] = df['utc_timestamp'].dt.quarter
    df['day_of_year'] = df['utc_timestamp'].dt.dayofyear

    df['lag_24'] = df[config['target']].shift(24)
    df['lag_48'] = df[config['target']].shift(48)

    if name == 'hybrid':
        tempCol = next(col for col in df.columns if 'Basel Temperature [2 m elevation corrected]' in col)
        windCol = next(col for col in df.columns if 'Basel Wind Speed [10 m]' in col)
        radiationCol = next(col for col in df.columns if 'Basel Shortwave Radiation' in col)
        humidityCol = next(col for col in df.columns if 'Basel Relative Humidity [2 m]' in col)
        cloudCol = next(col for col in df.columns if 'Basel Cloud Cover Total' in col)
        uvCol = next(col for col in df.columns if 'Basel UV Radiation' in col)
        gustCol = next(col for col in df.columns if 'Basel Wind Gust' in col)

        df['temp_x_wind'] = df[tempCol] * df[windCol]
        df['humidity_div_radiation'] = df[humidityCol] / (df[radiationCol] + 1e-6)
        df['cloud_x_uv'] = df[cloudCol] * df[uvCol]
        df['hour_x_radiation'] = df['hour'] * df[radiationCol]
        df['wind_gust_diff'] = df[gustCol] - df[windCol]
    else:
        df['temp_x_wind'] = df['Basel Temperature [2 m elevation corrected]'] * df['Basel Wind Speed [10 m]']
        df['humidity_div_radiation'] = df['Basel Relative Humidity [2 m]'] / (df['Basel Shortwave Radiation'] + 1e-6)
        df['cloud_x_uv'] = df['Basel Cloud Cover Total'] * df['Basel UV Radiation']
        df['hour_x_radiation'] = df['hour'] * df['Basel Shortwave Radiation']
        df['wind_gust_diff'] = df['Basel Wind Gust'] - df['Basel Wind Speed [10 m]']

    df = df.dropna()
    df.columns = df.columns.str.replace(r"[\[\]<>\(\)]", "", regex=True).str.replace(" ", "_")
    return df

def getExtremeWeather(df, target, name):
    X = df.drop(columns=[target])
    y = df[target]
    X_test = X[int(len(X)*0.8):]
    y_test = y[int(len(y)*0.8):]

    if name == 'hybrid':
        radiationCol = next(col for col in X_test.columns if 'Shortwave_Radiation' in col)
        gustCol = next(col for col in X_test.columns if 'Wind_Gust' in col)
        tempCol = next(col for col in X_test.columns if 'Temperature_2_m_elevation_corrected' in col)
    else:
        radiationCol = 'Basel_Shortwave_Radiation'
        gustCol = 'Basel_Wind_Gust'
        tempCol = 'Basel_Temperature_2_m_elevation_corrected'

    extremeMask = (
        (X_test[radiationCol] < 50) |
        (X_test[gustCol] > 20) |
        (X_test[tempCol] < 0)
    )

    return X_test[extremeMask], y_test[extremeMask]

def evaluateXgb(modelPath, X_extreme, y_extreme):
    model = joblib.load(modelPath)
    X_transformed = model.named_steps['preprocessor'].transform(X_extreme)
    X_selected = model.named_steps['selector'].transform(X_transformed)
    y_pred = model.named_steps['xgb'].predict(X_selected)

    mae = mean_absolute_error(y_extreme, y_pred)
    mse = mean_squared_error(y_extreme, y_pred)
    r2 = r2_score(y_extreme, y_pred)
    return y_pred, mae, mse, r2

def plot(name, y_true, y_pred, mae, mse, r2):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.values, label='Actual', color='green')
    plt.plot(y_pred, label='XGBoost Prediction', color='blue', linestyle='--')
    plt.title(f"{name.capitalize()} Forecast During Extreme Weather (XGBoost)")
    textstr = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nR² : {r2:.4f}"
    plt.gcf().text(0.78, 0.55, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.xlabel("Time Step (Extreme Cases)")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(f"models/{name}", exist_ok=True)
    plt.savefig(f"models/{name}/{name}_extreme_weather_plot.png")
    plt.close()

def processDataset(name, config):
    print(f"\nProcessing {name.upper()}...")
    df = loadData(name, config)
    X_extreme, y_extreme = getExtremeWeather(df, config['target'], name)
    y_pred, mae, mse, r2 = evaluateXgb(config['model_path'], X_extreme, y_extreme)

    print(f"\nExtreme Weather Performance (XGBoost - {name.capitalize()}):")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R² : {r2:.4f}")

    plot(name, y_extreme, y_pred, mae, mse, r2)

# Main loop
for name, config in datasets.items():
    processDataset(name, config)
