import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model

# File path
datasets = {
    'solar': {
        'data_path': 'outData/solar_energy_weather.csv',
        'target': 'GB_GBN_solar_generation_actual',
        'nn_model_path': 'models_nn/solar/nn_solar_best_model.h5',
        'xgb_model_path': 'models/solar/optimized_xgb_solar_model.pkl'
    },
    'wind': {
        'data_path': 'outData/wind_energy_weather.csv',
        'target': 'GB_GBN_wind_generation_actual',
        'nn_model_path': 'models_nn/wind/nn_wind_best_model.h5',
        'xgb_model_path': 'models/wind/optimized_xgb_wind_model.pkl'
    },
    'hybrid': {
        'data_path': 'outData/hybrid_energy_weather.csv',
        'target': 'hybrid_generation_actual',
        'nn_model_path': 'models_nn/hybrid/nn_hybrid_best_model.h5',
        'xgb_model_path': 'models/hybrid/optimized_xgb_hybrid_model.pkl'
    }
}

def addFeatures(df, name):
    if name == 'hybrid':
        tempCol = next(col for col in df.columns if 'Basel_Temperature_2_m_elevation_corrected' in col)
        windCol = next(col for col in df.columns if 'Basel_Wind_Speed_10_m' in col)
        radiationCol = next(col for col in df.columns if 'Basel_Shortwave_Radiation' in col)
        humidityCol = next(col for col in df.columns if 'Basel_Relative_Humidity_2_m' in col)
        cloudCol = next(col for col in df.columns if 'Basel_Cloud_Cover_Total' in col)
        uvCol = next(col for col in df.columns if 'Basel_UV_Radiation' in col)
        gustCol = next(col for col in df.columns if 'Basel_Wind_Gust' in col)
    else:
        tempCol = 'Basel_Temperature_2_m_elevation_corrected'
        windCol = 'Basel_Wind_Speed_10_m'
        radiationCol = 'Basel_Shortwave_Radiation'
        humidityCol = 'Basel_Relative_Humidity_2_m'
        cloudCol = 'Basel_Cloud_Cover_Total'
        uvCol = 'Basel_UV_Radiation'
        gustCol = 'Basel_Wind_Gust'

    df['temp_x_wind'] = df[tempCol] * df[windCol]
    df['humidity_div_radiation'] = df[humidityCol] / (df[radiationCol] + 1e-6)
    df['cloud_x_uv'] = df[cloudCol] * df[uvCol]
    df['hour_x_radiation'] = df['hour'] * df[radiationCol]
    df['wind_gust_diff'] = df[gustCol] - df[windCol]
    return df

def evaluateModel(modelName, yTrue, yPred):
    mae = mean_absolute_error(yTrue, yPred)
    mse = mean_squared_error(yTrue, yPred)
    r2 = r2_score(yTrue, yPred)
    print(f"\n{modelName} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R² : {r2:.4f}")
    return mae, mse, r2

def processDataset(name, config):
    print(f"\nProcessing {name.upper()}...")

    df = pd.read_csv(config['data_path'], parse_dates=['utc_timestamp'])

    df['hour'] = df['utc_timestamp'].dt.hour
    df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
    df['month'] = df['utc_timestamp'].dt.month
    df['year'] = df['utc_timestamp'].dt.year
    df['quarter'] = df['utc_timestamp'].dt.quarter
    df['day_of_year'] = df['utc_timestamp'].dt.dayofyear
    df['lag_24'] = df[config['target']].shift(24)
    df['lag_48'] = df[config['target']].shift(48)

    df = df.dropna()
    df.columns = df.columns.str.replace(r"[\[\]<>\(\)]", "", regex=True).str.replace(" ", "_")
    df = addFeatures(df, name)

    X = df.drop(columns=[config['target'], 'utc_timestamp'], errors='ignore')
    y = df[config['target']]

    X_train = X[:int(len(X)*0.8)]
    X_test = X[int(len(X)*0.8):]
    y_train = y[:int(len(y)*0.8)]
    y_test = y[int(len(y)*0.8):]

    # XGBoost Prediction
    xgbModel = joblib.load(config['xgb_model_path'])
    xgbPred = xgbModel.named_steps['xgb'].predict(
        xgbModel.named_steps['selector'].transform(
            xgbModel.named_steps['preprocessor'].transform(X_test)
        )
    )

    # Neural Network Prediction
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    nnModel = load_model(config['nn_model_path'])
    nnInput = X_test_scaled[:, :77] if name == 'hybrid' else X_test_scaled
    nnPred = nnModel.predict(nnInput).flatten()

    # Evaluation
    xgbMae, xgbMse, xgbR2 = evaluateModel("XGBoost", y_test, xgbPred)
    nnMae, nnMse, nnR2 = evaluateModel("Neural Network", y_test, nnPred)

    # Plot
    plotModel(name, y_test.values[:100], xgbPred[:100], nnPred[:100], xgbMae, xgbMse, xgbR2, nnMae, nnMse, nnR2)

def plotModel(name, yTrue, xgbPred, nnPred, xgbMae, xgbMse, xgbR2, nnMae, nnMse, nnR2):
    plt.figure(figsize=(14, 6))
    plt.plot(yTrue, label='Actual', color='green')
    plt.plot(xgbPred, label=f'XGBoost\nMAE:{xgbMae:.1f}, MSE:{xgbMse:.0f}, R2:{xgbR2:.2f}', linestyle='--')
    plt.plot(nnPred, label=f'Neural Network\nMAE:{nnMae:.1f}, MSE:{nnMse:.0f}, R2:{nnR2:.2f}', linestyle='-.')
    plt.title(f"Model Comparison for {name.capitalize()}: First 100 Predictions NN vs XGBoost")
    plt.xlabel("Time Step (Hours)")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("comparePrediction", exist_ok=True)
    plt.savefig(f"comparePrediction/{name}_comparison_plot.png")
    plt.close()

# Main loop
for name, config in datasets.items():
    processDataset(name, config)
