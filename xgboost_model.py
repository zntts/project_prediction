import pandas as pd
import numpy as np
import os
import time
import json
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error as mae
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor

# File path
datasets = {
    'solar': 'outData/solar_energy_weather.csv',
    'wind': 'outData/wind_energy_weather.csv',
    'wind_offshore': 'outData/wind_offshore_energy_weather.csv',
    'wind_onshore': 'outData/wind_onshore_energy_weather.csv',
    'hybrid': 'outData/hybrid_energy_weather.csv'
}

targets = {
    'solar': 'GB_GBN_solar_generation_actual',
    'wind': 'GB_GBN_wind_generation_actual',
    'wind_offshore': 'GB_GBN_wind_offshore_generation_actual',
    'wind_onshore': 'GB_GBN_wind_onshore_generation_actual',
    'hybrid': 'hybrid_generation_actual'
}

def createTimeFeatures(df):
    df['hour'] = df['utc_timestamp'].dt.hour
    df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
    df['month'] = df['utc_timestamp'].dt.month
    df['year'] = df['utc_timestamp'].dt.year
    df['quarter'] = df['utc_timestamp'].dt.quarter
    df['day_of_year'] = df['utc_timestamp'].dt.dayofyear
    return df

def createLagFeatures(df, targetCol, lags=[24, 48]):
    for lag in lags:
        df[f'lag_{lag}'] = df[targetCol].shift(lag)
    return df

def addInteractionFeatures(df):
    try:
        df['temp_x_wind'] = df['Basel Temperature [2 m elevation corrected]'] * df['Basel Wind Speed [10 m]']
        df['humidity_div_radiation'] = df['Basel Relative Humidity [2 m]'] / (df['Basel Shortwave Radiation'] + 1e-6)
        df['cloud_x_uv'] = df['Basel Cloud Cover Total'] * df['Basel UV Radiation']
        df['hour_x_radiation'] = df['hour'] * df['Basel Shortwave Radiation']
        df['wind_gust_diff'] = df['Basel Wind Gust'] - df['Basel Wind Speed [10 m]']
    except KeyError as e:
        print(f"Skipping interaction features due to missing column: {e}")
    return df

def computeNaiveBaseline(yTrue, shift=24):
    yNaive = yTrue.shift(shift).dropna()
    yAligned = yTrue[shift:]
    return mean_squared_error(yAligned, yNaive), r2_score(yAligned, yNaive), yNaive, yAligned

def shapPlot(model, XTransformed, selectedFeatures, outputPath, modelName):
    explainer = shap.Explainer(model, XTransformed)
    shapValues = explainer(XTransformed)
    XShap = pd.DataFrame(XTransformed, columns=selectedFeatures)

    shap.summary_plot(shapValues, features=XShap, feature_names=selectedFeatures, plot_type='bar', show=False)
    plt.gcf().set_size_inches(10, 5 + len(selectedFeatures) * 0.4)
    plt.title(f"Feature Importance {modelName} (XGBoost)", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(outputPath, f"{modelName}_shap_summary.png"))
    plt.close()

def plots(modelName, yTest, yPred, yNaive, yTrueAligned, residuals, mseVal, naiveMse, r2Val, outputPath):
    alignedLen = len(yTrueAligned)

    plt.figure(figsize=(14, 6), dpi=120)
    plt.plot(yTrueAligned.values, label='Actual', color='green')
    plt.plot(yPred[-alignedLen:], label='XGBoost', color='red', linestyle='--')
    plt.plot(yNaive.values, label='Naive', color='gray', linestyle=':')
    textstr = f"R² = {r2Val:.4f}\nMSE = {mseVal:.2f}"
    plt.gcf().text(0.78, 0.75, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f"{modelName.capitalize()} Forecast Comparison (XGBoost)")
    plt.xlabel('Time Step (Hour)')
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath, f"{modelName}_forecast_plot.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.scatter(range(len(residuals)), residuals, alpha=0.6, color='orange')
    plt.axhline(0, linestyle='--', color='black')
    textstr = f"Mean = {residuals.mean():.2f}\nStd = {residuals.std():.2f}"
    plt.gcf().text(0.78, 0.75, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f"{modelName.capitalize()} Residuals (XGBoost)")
    plt.xlabel('Sample Index')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath, f"{modelName}_residuals.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(['Naive', 'XGBoost'], [naiveMse, mseVal], color=['gray', 'red'])
    plt.gcf().text(0.68, 0.60, f"MSE = {mseVal:.2f}", fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f"{modelName.capitalize()} MSE Comparison (XGBoost)")
    plt.ylabel("MSE (MW²)")
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath, f"{modelName}_mse_bar.png"))
    plt.close()

def trainModel(dataPath, targetCol, modelName):
    print(f"\nTraining model for {modelName}...")
    outputDir = os.path.join("models", modelName)
    os.makedirs(outputDir, exist_ok=True)

    df = pd.read_csv(dataPath, parse_dates=['utc_timestamp'])
    df = createTimeFeatures(df)
    df = createLagFeatures(df, targetCol)
    df = addInteractionFeatures(df)
    df = df.drop(columns=['utc_timestamp']).dropna()
    df.columns = df.columns.str.replace(r"[\[\]<>\(\)]", "", regex=True).str.replace(" ", "_")

    X = df.drop(columns=[targetCol])
    y = df[targetCol]
    X_train, y_train = X[:int(0.6 * len(X))], y[:int(0.6 * len(X))]
    X_test, y_test = X[int(0.8 * len(X)):], y[int(0.8 * len(y)):]

    numericCols = X.select_dtypes(include=['float64', 'int64']).columns
    preprocessor = ColumnTransformer([('num', RobustScaler(), numericCols)])
    selector = SelectFromModel(XGBRegressor(n_estimators=100, random_state=42), threshold="mean")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('xgb', XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=2,
            eval_metric='rmse',
            verbosity=0,
            random_state=42
        ))
    ])

    gridParams = {
        'xgb__learning_rate': [0.01, 0.03],
        'xgb__n_estimators': [200, 300],
        'xgb__max_depth': [3, 5],
        'xgb__subsample': [0.8],
        'xgb__colsample_bytree': [0.8],
        'xgb__reg_lambda': [1, 2],
        'xgb__reg_alpha': [0, 1]
    }

    gridSearch = GridSearchCV(pipeline, gridParams, cv=TimeSeriesSplit(n_splits=5),
                              scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, return_train_score=True)
    gridSearch.fit(X_train, y_train)
    bestModel = gridSearch.best_estimator_

    xgbModel = bestModel.named_steps['xgb']
    bestParams = {
        'Maximum Tree Depth': xgbModel.max_depth,
        'Minimum Child Weight': xgbModel.min_child_weight,
        'Learning Rate (Shrinkage)': xgbModel.learning_rate,
        'Number of Estimators (Trees)': xgbModel.n_estimators,
        'Subsample': xgbModel.subsample,
        'Column Sample by Tree': xgbModel.colsample_bytree,
        'Regularization Alpha': xgbModel.reg_alpha,
        'Regularization Lambda': xgbModel.reg_lambda
    }
    pd.DataFrame(list(bestParams.items()), columns=['Hyperparameter', 'Value'])\
        .to_csv(os.path.join(outputDir, f"{modelName}_hyperparameters.csv"), index=False)

    X_test_trans = bestModel.named_steps['selector'].transform(
        bestModel.named_steps['preprocessor'].transform(X_test)
    )
    selectedFeatures = X.columns[bestModel.named_steps['selector'].get_support()]
    y_pred = xgbModel.predict(X_test_trans)

    shapPlot(xgbModel, X_test_trans, selectedFeatures, outputDir, modelName)
    pd.DataFrame({'Feature': selectedFeatures, 'Importance': xgbModel.feature_importances_})\
        .to_csv(os.path.join(outputDir, f"{modelName}_feature_importance.csv"), index=False)

    mseVal = mean_squared_error(y_test, y_pred)
    r2Val = r2_score(y_test, y_pred)
    maeVal = mae(y_test, y_pred)
    residuals = y_test - y_pred

    naiveMse, naiveR2, yNaive, yTrueAligned = computeNaiveBaseline(y_test)

    pd.DataFrame({
        'actual': yTrueAligned.values,
        'xgboost_predicted': y_pred[-len(yTrueAligned):],
        'naive_predicted': yNaive.values
    }).to_csv(os.path.join(outputDir, f"{modelName}_forecast_output.csv"), index=False)

    metrics = {
        "XGBoost": {"MSE": mseVal, "MAE": maeVal, "R2": r2Val},
        "Naive": {"MSE": naiveMse, "R2": naiveR2},
        "Residual Summary": {"mean": residuals.mean(), "std": residuals.std()}
    }
    with open(os.path.join(outputDir, f"{modelName}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plots(modelName, y_test, y_pred, yNaive, yTrueAligned, residuals, mseVal, naiveMse, r2Val, outputDir)
    pd.DataFrame(gridSearch.cv_results_).to_csv(os.path.join(outputDir, f"{modelName}_cv_results.csv"), index=False)
    joblib.dump(bestModel, os.path.join(outputDir, f"optimized_xgb_{modelName}_model.pkl"))
    print(f"{modelName} done. R²: {r2Val:.4f}, Naive R²: {naiveR2:.4f}")

if __name__ == "__main__":
    for name in datasets:
        start = time.time()
        trainModel(datasets[name], targets[name], name)
        print(f"Duration: {time.time() - start:.2f}s")
    print("All XGBoost models trained.")
