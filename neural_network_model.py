import pandas as pd
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
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

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
epochs_list = [50, 100]

# File path
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

output_dir = 'models_nn'
os.makedirs(output_dir, exist_ok=True)

best_score = float('inf')
best_combo = {}
os.makedirs(os.path.join(output_dir, 'evaluation'), exist_ok=True)
summary_path = os.path.join(output_dir, 'evaluation/summary_nn.json')

def computeNaive(y_true, shift=24):
    y_naive = y_true.shift(shift).dropna()
    y_true_aligned = y_true[shift:]
    mse = mean_squared_error(y_true_aligned, y_naive)
    r2 = r2_score(y_true_aligned, y_naive)
    return mse, r2, y_naive, y_true_aligned

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

def createNNModel(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def loadAndPreprocessData(data_path, target_column):
    df = pd.read_csv(data_path, parse_dates=['utc_timestamp'])
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
    X_train = X_selected[:train_end]
    y_train = y[:train_end]
    X_test = X_selected[valid_end:]
    y_test = y[valid_end:]

    return X_train, X_test, y_train, y_test, X.columns, selector

start_time = time.time()
# main loop
for dataset_name in datasets:
    print(f"\nStarting hyperparameter tuning for {dataset_name}...")

    data_path = datasets[dataset_name]
    target_col = targets[dataset_name]

    X_train, X_test, y_train, y_test, feature_names, selector = loadAndPreprocessData(data_path, target_col)

    best_r2 = -np.inf
    best_mse = np.inf
    best_params = None
    best_model = None

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                print(f"\nTesting: lr={lr}, batch={batch_size}, epochs={epochs}")

                model = createNNModel(X_train.shape[1], lr)
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=0
                )

                y_pred_nn = model.predict(X_test).flatten()
                current_r2 = r2_score(y_test, y_pred_nn)
                current_mse = mean_squared_error(y_test, y_pred_nn)

                print(f"Results: R²={current_r2:.4f}, MSE={current_mse:.4f}")

                if current_r2 > best_r2:
                    best_r2 = current_r2
                    best_params = (lr, batch_size, epochs)
                    best_model = model

                # Compute tune best mse
                if current_mse < best_score:
                    best_score = current_mse
                    best_combo = {
                        'dataset': dataset_name,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'mse': current_mse
                    }

    print(f"\nBest model for {dataset_name} found with R²={best_r2:.4f} MSE={best_mse:.4f}")
    print(f"Optimal parameters: Learning Rate={best_params[0]}, Batch Size={best_params[1]}, Epochs={best_params[2]}")

    model_path = os.path.join(output_dir, f"{dataset_name}/nn_{dataset_name}_best_model.h5")
    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
    best_model.save(model_path)

    # Save summary
    with open(summary_path, 'w') as f:
        json.dump(best_combo, f, indent=2)

print("\nAll models tuned and best configuration saved successfully!")
print(json.dumps(best_combo, indent=2))
