import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json

# File path
model_names = ['solar', 'wind', 'wind_offshore', 'wind_onshore', 'hybrid']
root_dir = 'models'

def plotForecast(model_name, df):
    output_dir = os.path.join(root_dir, model_name)
    actual = df['actual']
    xgb_pred = df['xgboost_predicted']
    naive_pred = df['naive_predicted']

    r2 = r2_score(actual, xgb_pred)
    mse = mean_squared_error(actual, xgb_pred)

    plt.figure(figsize=(14, 6), dpi=120)
    plt.plot(actual, label='Actual', color='green')
    plt.plot(xgb_pred, label='XGBoost', color='red', linestyle='--')
    plt.plot(naive_pred, label='Naive', color='gray', linestyle=':')
    textstr = f"R² = {r2:.4f}\nMSE = {mse:.2f}"
    plt.gcf().text(0.78, 0.75, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f"{model_name.capitalize()} Forecast Comparison (XGBoost)")
    plt.xlabel('Time Step (Hour)')
    plt.ylabel("Energy (MW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_forecast_plot.png"))
    plt.close()


def plotResiduals(model_name, residuals):
    output_dir = os.path.join(root_dir, model_name)

    plt.figure(figsize=(12, 4))
    textstr = f"Mean = {residuals.mean():.2f}\nStd = {residuals.std():.2f}"
    plt.scatter(range(len(residuals)), residuals, alpha=0.6, color='orange')
    plt.axhline(0, linestyle='--', color='black')
    plt.gcf().text(0.78, 0.75, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f"{model_name.capitalize()} Residuals Plot (XGBoost)")
    plt.xlabel('Sample Index')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_residuals.png"))
    plt.close()


def plotMSEBar(model_name, mse_xgb, mse_naive):
    output_dir = os.path.join(root_dir, model_name)

    plt.figure(figsize=(6, 4))
    textstr = f"{model_name.capitalize()} mse = {mse_xgb:.2f}\nNaive mse = {mse_naive:.2f}"
    plt.gcf().text(0.60, 0.60, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.bar(['Naive', 'XGBoost'], [mse_naive, mse_xgb], color=['gray', 'red'])
    plt.title(f"{model_name.capitalize()} MSE Comparison (XGBoost)")
    plt.ylabel("MSE (MW²)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_mse_bar.png"))
    plt.close()


# Main Loop
for model_name in model_names:
    print(f"\nPlotting results for {model_name}...")
    output_dir = os.path.join(root_dir, model_name)
    forecast_path = os.path.join(output_dir, f"{model_name}_forecast_output.csv")

    if not os.path.exists(forecast_path):
        print(f"{forecast_path} not found. Skipping...")
        continue

    df = pd.read_csv(forecast_path)
    actual = df['actual']
    xgb_pred = df['xgboost_predicted']
    naive_pred = df['naive_predicted']
    residuals = actual - xgb_pred

    mse_xgb = mean_squared_error(actual, xgb_pred)
    mse_naive = mean_squared_error(actual, naive_pred)

    plotForecast(model_name, df)
    plotResiduals(model_name, residuals)
    plotMSEBar(model_name, mse_xgb, mse_naive)
    print(f"Saved plots for {model_name}")

    print("\nAll XGBoost result plots generated.")
    mae_xgb = np.mean(np.abs(actual - xgb_pred))
    r2_xgb = r2_score(actual, xgb_pred)
    r2_naive = r2_score(actual, naive_pred)

    # Save metrics
    metrics = { 
        "XGBoost": {"MSE": mse_xgb, "MAE": mae_xgb, "R2": r2_xgb},
        "NaiveBaseline": {"MSE": mse_naive, "R2": r2_naive},
        "ResidualSummary": {"mean": residuals.mean(), "std": residuals.std()}
    }

    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
