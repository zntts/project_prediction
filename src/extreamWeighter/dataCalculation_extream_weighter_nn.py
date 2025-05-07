import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model

# File path
datasets = {
    'solar': {
        'data_path': 'outData/solar_energy_weather.csv',
        'target': 'GB_GBN_solar_generation_actual',
        'model_path': 'models_nn/solar/nn_solar_best_model.h5'
    },
    'wind': {
        'data_path': 'outData/wind_energy_weather.csv',
        'target': 'GB_GBN_wind_generation_actual',
        'model_path': 'models_nn/wind/nn_wind_best_model.h5'
    }
}

def loadAndPrepareData(path, targetColumn):
    df = pd.read_csv(path, parse_dates=['utc_timestamp'])

    # Create time based features
    df['hour'] = df['utc_timestamp'].dt.hour
    df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
    df['month'] = df['utc_timestamp'].dt.month
    df['year'] = df['utc_timestamp'].dt.year
    df['quarter'] = df['utc_timestamp'].dt.quarter
    df['day_of_year'] = df['utc_timestamp'].dt.dayofyear

    # Lag features
    df['lag_24'] = df[targetColumn].shift(24)
    df['lag_48'] = df[targetColumn].shift(48)

    df['temp_x_wind'] = df['Basel Temperature [2 m elevation corrected]'] * df['Basel Wind Speed [10 m]']
    df['humidity_div_radiation'] = df['Basel Relative Humidity [2 m]'] / (df['Basel Shortwave Radiation'] + 1e-6)
    df['cloud_x_uv'] = df['Basel Cloud Cover Total'] * df['Basel UV Radiation']
    df['hour_x_radiation'] = df['hour'] * df['Basel Shortwave Radiation']
    df['wind_gust_diff'] = df['Basel Wind Gust'] - df['Basel Wind Speed [10 m]']

    df = df.dropna()
    df.columns = df.columns.str.replace(r"[\[\]<>\(\)]", "", regex=True).str.replace(" ", "_")
    return df

def splitData(df, targetColumn):
    timestamp = df['utc_timestamp'] if 'utc_timestamp' in df.columns else None
    X = df.drop(columns=[targetColumn, 'utc_timestamp'], errors='ignore')
    y = df[targetColumn]

    X_test = X[int(len(X)*0.8):].copy()
    y_test = y[int(len(y)*0.8):]
    return X_test, y_test

def filterExtremeConditions(X_test, y_test):
    condition = (
        (X_test['Basel_Shortwave_Radiation'] < 50) |
        (X_test['Basel_Wind_Gust'] > 20) |
        (X_test['Basel_Temperature_2_m_elevation_corrected'] < 0)
    )
    return X_test[condition], y_test[condition]

def evaluateModel(modelPath, X_extreme, y_extreme):
    model = load_model(modelPath)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_extreme)
    y_pred = model.predict(X_scaled).flatten()

    mae = mean_absolute_error(y_extreme, y_pred)
    mse = mean_squared_error(y_extreme, y_pred)
    r2 = r2_score(y_extreme, y_pred)
    return y_pred, mae, mse, r2

def plotResults(name, yTrue, yPred, mae, mse, r2):
    plt.figure(figsize=(12, 5))
    plt.plot(yTrue.values, label='Actual', color='green')
    plt.plot(yPred, label='NN Prediction', color='blue', linestyle='--')
    plt.title(f"{name.capitalize()} Forecast During Extreme Weather (NN)")
    textstr = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nR² : {r2:.4f}"
    plt.gcf().text(0.78, 0.55, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.xlabel("Time Step (Extreme Cases)")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(f"models_nn/{name}", exist_ok=True)
    plt.savefig(f"models_nn/{name}/{name}_extreme_weather_plot_nn.png")
    plt.close()

def processDataset(name, config):
    print(f"\nProcessing {name.upper()}...")

    df = loadAndPrepareData(config['data_path'], config['target'])
    X_test, y_test = splitData(df, config['target'])
    X_extreme, y_extreme = filterExtremeConditions(X_test, y_test)

    y_pred, mae, mse, r2 = evaluateModel(config['model_path'], X_extreme, y_extreme)

    print(f"\nExtreme Weather Performance (NN - {name.capitalize()}):")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R² : {r2:.4f}")

    plotResults(name, y_extreme, y_pred, mae, mse, r2)


# Main loop
for name, config in datasets.items():
    processDataset(name, config)
