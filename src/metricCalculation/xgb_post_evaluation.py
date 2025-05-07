import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# File path
datasets = ['solar', 'wind', 'hybrid', 'wind_offshore', 'wind_onshore']

best_overall_score = float('inf')
best_dataset = None
best_hyperparameters = {}

for dataset_name in datasets:
    print(f"\nProcessing {dataset_name.capitalize()}...")

    output_dir = os.path.join('models', dataset_name)
    model_path = os.path.join(output_dir, f"optimized_xgb_{dataset_name}_model.pkl")
    forecast_path = os.path.join(output_dir, f"{dataset_name}_forecast_output.csv")
    hyperparam_path = os.path.join(output_dir, f"{dataset_name}_hyperparameters.csv")

    if not os.path.exists(model_path) or not os.path.exists(forecast_path):
        print(f"Missing model or forecast for {dataset_name}, skipping.")
        continue

    model = joblib.load(model_path)
    forecast_df = pd.read_csv(forecast_path)
    xgb_model = model.named_steps['xgb']

    hyperparameters = {
        'Maximum Tree Depth': xgb_model.max_depth,
        'Minimum Child Weight': xgb_model.min_child_weight,
        'Learning Rate (Shrinkage)': xgb_model.learning_rate,
        'Number of Estimators (Trees)': xgb_model.n_estimators,
        'Subsample': xgb_model.subsample,
        'Colsample By Tree': xgb_model.colsample_bytree,
        'Reg Alpha': xgb_model.reg_alpha,
        'Reg Lambda': xgb_model.reg_lambda
    }
    df_hyper = pd.DataFrame(list(hyperparameters.items()), columns=['Hyperparameter', 'Value'])
    df_hyper.to_csv(hyperparam_path, index=False)
    print(f"Hyperparameters saved to: {hyperparam_path}")

    y_test = forecast_df['actual'].values
    y_test_pred = forecast_df['xgboost_predicted'].values

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae_val = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print(f"Metrics for {dataset_name.capitalize()}:")
    print(f"  MSE   : {mse:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae_val:.4f}")
    print(f"  R²    : {r2:.4f}")

    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
        'Value': [mse, rmse, mae_val, r2]
    })
    metrics_df.to_csv(os.path.join(output_dir, f"{dataset_name}_metrics_summary.csv"), index=False)
    print(f"Saved: {dataset_name}_metrics_summary.csv")

    if mse < best_overall_score:
        best_overall_score = mse
        best_dataset = dataset_name
        best_hyperparameters = hyperparameters

    selector = model.named_steps['selector']
    preprocessor = model.named_steps['preprocessor']

    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = preprocessor.get_feature_names()

    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
    importances = xgb_model.feature_importances_

    feature_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances[:len(selected_features)]
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 5 Features:")
    print(feature_df.head(5))

    feature_df.to_csv(os.path.join(output_dir, "selectkbest_feature_importance.csv"), index=False)
    print(f"Saved: selectkbest_feature_importance.csv")

summary_path = 'models/evaluation/summary_xgboost.json'
summary_data = {
    "Best Dataset": best_dataset,
    "Best MSE": best_overall_score,
    "Best Hyperparameters": best_hyperparameters
}
with open(summary_path, "w") as f:
    json.dump(summary_data, f, indent=2)

print("\nBest Performing Dataset:")
print(f"  Dataset : {best_dataset}")
print(f"  Best MSE: {best_overall_score:.4f}")
