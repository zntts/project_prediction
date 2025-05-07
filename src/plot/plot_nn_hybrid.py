import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error as mae
import json

# File path
data_path = 'outData/hybrid_energy_weather.csv'
target_col = 'hybrid_generation_actual'
model_path = 'models_nn/hybrid/nn_hybrid_best_model.h5'
output_dir = 'models_nn'
os.makedirs(output_dir, exist_ok=True)

def createTimeFeatures(df):
    df['hour'] = df['utc_timestamp'].dt.hour
    df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
    df['month'] = df['utc_timestamp'].dt.month
    df['year'] = df['utc_timestamp'].dt.year
    df['quarter'] = df['utc_timestamp'].dt.quarter
    df['day_of_year'] = df['utc_timestamp'].dt.dayofyear
    return df

def createLagFeatures(df, target_col, lags=[24]):
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

def plotBaselinePrediction(y_test, y_pred_nn, y_naive, dataset_name, r2_val, mse_val):
    textstr = f"R² = {r2_val:.4f}\nMSE = {mse_val:.2f}"
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual', color='green')
    plt.plot(y_pred_nn, label='NN Prediction', color='red', linestyle='--')
    plt.plot(y_naive.values, label='Naive Prediction', color='gray', linestyle=':')
    plt.gcf().text(0.78, 0.75, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f'{dataset_name.capitalize()} Energy Prediction (NN)')
    plt.xlabel('Time Step (Hour)')
    plt.ylabel('Energy (MW)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}/{dataset_name}_forecast_plot.png"))
    plt.close()

def plotResiduals(residuals, dataset_name):
    plt.figure(figsize=(10, 4))
    textstr_resid = f"Mean = {residuals.mean():.2f}\nStd = {residuals.std():.2f}"
    plt.gcf().text(0.78, 0.75, textstr_resid, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.scatter(range(len(residuals)), residuals, alpha=0.6, color='orange')
    plt.axhline(0, linestyle='--', color='black')
    plt.title(f"{dataset_name.capitalize()} Residuals Plot (NN)")
    plt.xlabel('Sample Index')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}/{dataset_name}_residuals.png"))
    plt.close()

def plotMSE(naive_mse, mse_val, dataset_name):
    plt.figure(figsize=(5, 4))
    textstr = f"{dataset_name.capitalize()} mse = {mse_val:.2f}\nNaive mse = {naive_mse:.2f}"
    plt.gcf().text(0.57, 0.60, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.bar(['Naive', 'NN'], [naive_mse, mse_val], color=['gray', 'red'])
    plt.title(f"{dataset_name.capitalize()} - MSE Comparison (NN)")
    plt.ylabel("MSE (MW²)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}/{dataset_name}_mse_bar.png"))
    plt.close()

print("Generating plots for hybrid NN model...")

 # Load data
df = pd.read_csv(data_path, parse_dates=['utc_timestamp'])
df = df.loc[:, ~df.columns.str.endswith('_y')]
df.columns = df.columns.str.replace('_x', '', regex=False)
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

# Predict for hybrid
model = load_model(model_path)
y_pred_nn = model.predict(X_test).flatten()
mae_val = mae(y_test, y_pred_nn)

# Predict for hybrid Naive
naive_mse, naive_r2, y_naive, y_true_aligned = computeNaive(y_test)
mse_val = mean_squared_error(y_test, y_pred_nn)
r2_val = r2_score(y_test, y_pred_nn)

residuals = y_test - y_pred_nn

# Plot for all graph
plotBaselinePrediction(y_test, y_pred_nn, y_naive, 'hybrid', r2_val, mse_val)
plotResiduals(residuals, 'hybrid')
plotMSE(naive_mse, mse_val, 'hybrid')

print("Complete saved data for hybrid")

metrics = { 
    "NeuralNetwork": {"MSE": mse_val, "MAE": mae_val, "R2": r2_val},
    "NaiveBaseline": {"MSE": naive_mse, "R2": naive_r2},
    "ResidualSummary": {"mean": residuals.mean(), "std": residuals.std()}
}

with open(os.path.join(output_dir, f"hybrid/hybrid_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
