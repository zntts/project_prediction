import pandas as pd
import numpy as np
import os
import time
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error as mae
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# File path
datasets = {
    'hybrid': 'outData/hybrid_energy_weather.csv'
}
targets = {
    'hybrid': 'hybrid_generation_actual'
}
output_dir = 'models_nn'
os.makedirs(output_dir, exist_ok=True)
best_score = float('inf')
best_combo = {}
summary_path = 'models_nn/evaluation/summary_nn.json'
os.makedirs(os.path.dirname(summary_path), exist_ok=True)

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

def createNNModel(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def loadAndPreprocessData(data_path, target_column):
    df = pd.read_csv(data_path, parse_dates=['utc_timestamp'])
    df = df.loc[:, ~df.columns.str.endswith('_y')]
    df.columns = df.columns.str.replace('_x', '', regex=False)
    df = createTimeFeatures(df)
    df = createLagFeatures(df, target_column)
    df = addInteractionFeatures(df)
    df = df.drop(columns=['utc_timestamp']).dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    preprocessor = ColumnTransformer(transformers=[('num', RobustScaler(), numeric_features)])
    selector = SelectKBest(score_func=f_regression, k='all')
    X_scaled = preprocessor.fit_transform(X)
    X_selected = selector.fit_transform(X_scaled, y)
    train_end = int(0.6 * len(X_selected))
    valid_end = int(0.8 * len(X_selected))
    return X_selected[:train_end], X_selected[valid_end:], y[:train_end], y[valid_end:], X.columns, selector

# Main loop
for dataset_name in datasets:
    print(f"\nTraining hybrid NN model for {dataset_name}")
    lr = 0.001
    batch = 32
    ep = 100

    X_train, X_test, y_train, y_test, feature_names, selector = loadAndPreprocessData(
        datasets[dataset_name], targets[dataset_name])

    model = createNNModel(X_train.shape[1], learning_rate=lr)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=ep, batch_size=batch, callbacks=[early_stop], verbose=0)
    y_pred_nn = model.predict(X_test).flatten()

    model_path = os.path.join(output_dir, f"{dataset_name}/nn_{dataset_name}_best_model.h5")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    with open(os.path.join(output_dir, f"{dataset_name}/{dataset_name}_selected_features.txt"), "w") as f:
        for feat in feature_names:
            f.write(f"{feat}\n")

    naive_mse, naive_r2, y_naive, y_true_aligned = computeNaive(y_test)
    residuals = y_test - y_pred_nn
    mae_val = mae(y_test, y_pred_nn)
    mse_val = mean_squared_error(y_test, y_pred_nn)
    r2_val = r2_score(y_test, y_pred_nn)

    metrics = {
        "NeuralNetwork": {"MSE": mse_val, "MAE": mae_val, "R2": r2_val},
        "NaiveBaseline": {"MSE": naive_mse, "R2": naive_r2},
        "ResidualSummary": {"mean": residuals.mean(), "std": residuals.std()}
    }
    with open(os.path.join(output_dir, f"{dataset_name}/{dataset_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    aligned_len = len(y_true_aligned)
    pred_df = pd.DataFrame({
        'actual': y_true_aligned.values,
        'nn_predicted': y_pred_nn[-aligned_len:],
        'naive_predicted': y_naive.values
    })
    pred_df.to_csv(os.path.join(output_dir, f"{dataset_name}/{dataset_name}_forecast_output.csv"), index=False)

    # Compute best config
    if mse_val < best_score:
        best_score = mse_val
        best_combo = {
            'dataset': dataset_name,
            'learning_rate': lr,
            'batch_size': batch,
            'epochs': ep,
            'mse': mse_val
        }

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual', color='green')
    plt.plot(y_pred_nn, label='NN Prediction', color='red', linestyle='--')
    plt.plot(y_naive.values, label='Naive Prediction', color='gray', linestyle=':')
    textstr = f"R² = {r2_val:.4f}\nMSE = {mse_val:.2f}"
    plt.gcf().text(0.78, 0.75, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f'{dataset_name.capitalize()} Energy Prediction (NN)')
    plt.xlabel('Time Step (Hour)')
    plt.ylabel('Energy (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_forecast_plot.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    textstr_resid = f"Mean = {residuals.mean():.2f}\nStd = {residuals.std():.2f}"
    plt.gcf().text(0.78, 0.75, textstr_resid, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.scatter(range(len(residuals)), residuals, alpha=0.6, color='orange')
    plt.axhline(0, linestyle='--', color='black')
    plt.title(f"{dataset_name.capitalize()} - Residuals (NN)")
    plt.xlabel('Sample Index')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_residuals.png"))
    plt.close()

    plt.figure(figsize=(5, 4))
    textstr = f"MSE = {mse_val:.2f}"
    plt.gcf().text(0.68, 0.60, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.bar(['Naive', 'NN'], [naive_mse, mse_val], color=['gray', 'red'])
    plt.title(f"{dataset_name.capitalize()} - MSE Comparison (NN)")
    plt.ylabel("MSE (MW²)")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_mse_bar.png"))
    plt.close()

# Save
with open(summary_path, 'w') as f:
    json.dump(best_combo, f, indent=2)

print("\nBest Performing NN Configuration:")
print(json.dumps(best_combo, indent=2))
print(f"Summary saved to: {summary_path}")