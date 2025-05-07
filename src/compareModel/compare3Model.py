import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

# File path
configs = {
    'solar': {
        'data_path': 'outData/solar_energy_weather.csv',
        'target_column': 'GB_GBN_solar_generation_actual',
        'xgb_model_path': 'models/solar/optimized_xgb_solar_model.pkl',
        'nn_model_path': 'models_nn/solar/nn_solar_best_model.h5'
    },
    'wind': {
        'data_path': 'outData/wind_energy_weather.csv',
        'target_column': 'GB_GBN_wind_generation_actual',
        'xgb_model_path': 'models/wind/optimized_xgb_wind_model.pkl',
        'nn_model_path': 'models_nn/wind/nn_wind_best_model.h5'
    },
    'hybrid': {
        'data_path': 'outData/hybrid_energy_weather.csv',
        'target_column': 'hybrid_generation_actual',
        'xgb_model_path': 'models/hybrid/optimized_xgb_hybrid_model.pkl',
        'nn_model_path': 'models_nn/hybrid/nn_hybrid_best_model.h5'
    }
}

def evaluateModel(modelName, yTrue, yPred):
    mae = mean_absolute_error(yTrue, yPred)
    mse = mean_squared_error(yTrue, yPred)
    r2 = r2_score(yTrue, yPred)
    print(f"\n{modelName} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R² : {r2:.4f}")
    return mae, mse, r2

def addFeatures(df, name, targetCol):
    df['hour'] = df['utc_timestamp'].dt.hour
    df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
    df['month'] = df['utc_timestamp'].dt.month
    df['year'] = df['utc_timestamp'].dt.year
    df['quarter'] = df['utc_timestamp'].dt.quarter
    df['day_of_year'] = df['utc_timestamp'].dt.day_of_year
    df['lag_24'] = df[targetCol].shift(24)
    df['lag_48'] = df[targetCol].shift(48)

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
    return df

def plot(name, yTrue, xgbPred, linregPred, nnPred, xgbMetrics, linregMetrics, nnMetrics):
    plt.figure(figsize=(14, 6))
    plt.plot(yTrue[:100], label='Actual', color='green')
    plt.plot(xgbPred[:100], label=f'XGBoost\nMAE:{xgbMetrics[0]:.1f}, MSE:{xgbMetrics[1]:.0f}, R2:{xgbMetrics[2]:.2f}', linestyle='--')
    plt.plot(linregPred[:100], label=f'Linear Regression\nMAE:{linregMetrics[0]:.1f}, MSE:{linregMetrics[1]:.0f}, R2:{linregMetrics[2]:.2f}', linestyle=':')
    plt.plot(nnPred[:100], label=f'Neural Network\nMAE:{nnMetrics[0]:.1f}, MSE:{nnMetrics[1]:.0f}, R2:{nnMetrics[2]:.2f}', linestyle='-.')
    plt.title(f"Model Comparison for {name.capitalize()}: First 100 Predictions")
    plt.xlabel("Time Step (Hours)")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("comparePrediction", exist_ok=True)
    plt.savefig(f"comparePrediction/{name}_comparison_plot.png")
    plt.close()

def processDataset(name, cfg):
    print(f"\nProcessing {name.upper()}...")
    df = pd.read_csv(cfg['data_path'], parse_dates=['utc_timestamp'])
    df = addFeatures(df, name, cfg['target_column'])

    df = df.dropna()
    df.columns = df.columns.str.replace(r"[\[\]<>\(\)]", "", regex=True).str.replace(" ", "_")

    X = df.drop(columns=[cfg['target_column'], 'utc_timestamp'])
    y = df[cfg['target_column']]

    X_train = X[:int(len(X)*0.8)]
    X_test = X[int(len(X)*0.8):]
    y_train = y[:int(len(y)*0.8)]
    y_test = y[int(len(y)*0.8):]

    # XGBoost
    xgbModel = joblib.load(cfg['xgb_model_path'])
    xgbPred = xgbModel.named_steps['xgb'].predict(
        xgbModel.named_steps['selector'].transform(
            xgbModel.named_steps['preprocessor'].transform(X_test)
        )
    )

    # Linear Regression
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    linreg = LinearRegression()
    linreg.fit(X_scaled, y_train)
    linregPred = linreg.predict(X_test_scaled)

    # Neural Network
    nnModel = load_model(cfg['nn_model_path'])
    nnInput = X_test_scaled[:, :77] if name == 'hybrid' else X_test_scaled
    nnPred = nnModel.predict(nnInput).flatten()

    # Evaluation
    xgbMetrics = evaluateModel("XGBoost", y_test, xgbPred)
    linregMetrics = evaluateModel("Linear Regression", y_test, linregPred)
    nnMetrics = evaluateModel("Neural Network", y_test, nnPred)

    # Plot graph
    plot(name, y_test.values, xgbPred, linregPred, nnPred, xgbMetrics, linregMetrics, nnMetrics)

# Main loop
for name, cfg in configs.items():
    processDataset(name, cfg)
