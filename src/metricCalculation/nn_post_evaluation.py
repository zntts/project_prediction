import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

#  File path
datasets = ['solar', 'wind','wind_offshore','wind_onshore','hybrid']
output_dir = 'models_nn'

for dataset in datasets:
    print(f"\nEvaluating NN model for: {dataset}")

    model_path = os.path.join(output_dir, f"{dataset}/nn_{dataset}_best_model.h5")
    forecast_path = os.path.join(output_dir, f"{dataset}/{dataset}_forecast_output.csv")

    # Check for file existence
    if not os.path.exists(model_path):
        print(f"Missing model file: {model_path}")
        continue
    if not os.path.exists(forecast_path):
        print(f"Missing forecast file: {forecast_path}")
        continue

    # Load forecasted results
    df = pd.read_csv(forecast_path)
    if 'actual' not in df.columns or 'nn_predicted' not in df.columns:
        print(f"Forecast file does not contain 'actual' and 'nn_predicted' columns")
        continue

    y_test = df['actual'].values
    y_pred = df['nn_predicted'].values

    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae_val = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Metrics for {dataset}:")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae_val:.4f}")
    print(f"  R²   : {r2:.4f}")

    # Save to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
        'Value': [mse, rmse, mae_val, r2]
    })

    metrics_path = os.path.join(output_dir, f"{dataset}/{dataset}_nn_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")
