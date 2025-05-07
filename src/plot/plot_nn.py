import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
import json

# File path
modelDir = 'models_nn'
outputDir = 'models_nn'

datasets = {
    'solar': 'outData/solar_energy_weather.csv',
    'wind': 'outData/wind_energy_weather.csv',
    'wind_offshore': 'outData/wind_offshore_energy_weather.csv',
    'wind_onshore': 'outData/wind_onshore_energy_weather.csv'
}

targets = {
    'solar': 'GB_GBN_solar_generation_actual',
    'wind': 'GB_GBN_wind_generation_actual',
    'wind_offshore': 'GB_GBN_wind_offshore_generation_actual',
    'wind_onshore': 'GB_GBN_wind_onshore_generation_actual'
}

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

def computeNaive(y_true, shift=24):
    y_naive = y_true.shift(shift).dropna()
    y_true_aligned = y_true[shift:]
    mse = mean_squared_error(y_true_aligned, y_naive)
    r2 = r2_score(y_true_aligned, y_naive)
    return mse, r2, y_naive, y_true_aligned

def plotBaselinePrediction(y_test, y_pred, dataset_name, r2, mse):
    textstr = f"R² = {r2:.4f}\nMSE = {mse:.2f}"
    y_naive = y_test.shift(24).dropna()
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label='Actual', color='green')
    plt.plot(y_pred, label='Prediction', color='red', linestyle='--')
    plt.plot(y_naive.values, label='Naive Prediction', color='gray', linestyle=':')
    plt.gcf().text(0.85, 0.6, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f'{dataset_name.capitalize()} Energy Prediction (NN)')
    plt.xlabel('Time Step (Hour)')
    plt.ylabel('Energy Generation (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, f"{dataset_name}/{dataset_name}_baseline_linear_plot.png"))
    plt.close()

def plotResiduals(residuals, dataset_name):
    plt.figure(figsize=(10, 4))
    textstr = f"Mean = {residuals.mean():.2f}\nStd = {residuals.std():.2f}"
    plt.gcf().text(0.78, 0.75, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.scatter(range(len(residuals)), residuals, alpha=0.6, color='orange')
    plt.axhline(0, linestyle='--', color='black')
    plt.title(f'{dataset_name.capitalize()} Energy Residuals Plot (NN)')
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, f"{dataset_name}/{dataset_name}_residuals.png"))
    plt.close()

def plotMSE(naive_mse, model_mse, dataset_name):
    textstr = f"{dataset_name.capitalize()} MSE = {model_mse:.2f}\nNaive MSE= {naive_mse:.2f}"
    plt.figure(figsize=(5, 4))
    plt.bar(['Naive', 'NN'], [naive_mse, model_mse], color=['gray', 'red'])
    plt.gcf().text(0.57, 0.60, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f"{dataset_name.capitalize()} - MSE Comparison (NN)")
    plt.ylabel("MSE (MW²)")
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, f"{dataset_name}/{dataset_name}_mse_bar.png"))
    plt.close()

# main loop for solar, wind, wind_offshore, wind_onshore
for dataset_name in datasets:
    print(f"\nProcessing {dataset_name}...")

    # Load model from data set
    model_path = os.path.join(modelDir, f"{dataset_name}/nn_{dataset_name}_best_model.h5")
    model = load_model(model_path)

    # Load and preprocess data
    df = pd.read_csv(datasets[dataset_name], parse_dates=['utc_timestamp'])
    target_col = targets[dataset_name]
    df = createTimeFeatures(df)
    df = createLagFeatures(df, target_col)
    df = addInteractionFeatures(df)
    df = df.drop(columns=['utc_timestamp']).dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    preprocessor = ColumnTransformer([('num', RobustScaler(), numeric_features)])
    selector = SelectKBest(score_func=f_regression, k='all')

    X_scaled = preprocessor.fit_transform(X)
    X_selected = selector.fit_transform(X_scaled, y)

    valid_start = int(0.8 * len(X_selected))
    X_test = X_selected[valid_start:]
    y_test = y[valid_start:]

    # Predict for NN
    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    mse_val = mean_squared_error(y_test, y_pred)
    mae_val = mae(y_test, y_pred)

    # Predict for Naive baseline
    naive_mse, naive_r2, _, _ = computeNaive(y_test)
    residuals = y_test - y_pred
    
    # Create graph 
    plotBaselinePrediction(y_test, y_pred, dataset_name, r2, mse_val)
    plotResiduals(residuals, dataset_name)
    plotMSE(naive_mse, mse_val, dataset_name)

    print(f"Complete saved data for {dataset_name}")

    metrics = { 
        "NeuralNetwork": {"MSE": mse_val, "MAE": mae_val, "R2": r2},
        "NaiveBaseline": {"MSE": naive_mse, "R2": naive_r2},
        "ResidualSummary": {"mean": residuals.mean(), "std": residuals.std()}
    }

    with open(os.path.join(outputDir, f"{dataset_name}/{dataset_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
