# =========================
# Battery SoH Analysis – Modular Functions
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Feature engineering
# -------------------------
def feature_engineering(data):
    # Temperature features
    data['Temp_avg'] = data[['Temp1','Temp2','Temp3','Temp4']].mean(axis=1)
    data['Temp_diff'] = data[['Temp1','Temp2','Temp3','Temp4']].max(axis=1) - \
                        data[['Temp1','Temp2','Temp3','Temp4']].min(axis=1)
    # Use std deviation for imbalance
    data['Temp_Imbalance'] = data[['Temp1','Temp2','Temp3','Temp4']].std(axis=1)

    # Voltage difference
    data['Volt_diff'] = data['C_N_High'] - data['C_N_Low']

    # Smoothed signals
    data['Current_smooth'] = data['Current'].rolling(3).mean()
    data['Voltage_smooth'] = data['Voltage'].rolling(3).mean()
    data = data.dropna(subset=['Current_smooth','Voltage_smooth'])

    # First differences
    data['dSoC'] = data['SOC'].diff()
    data['dVolt'] = data['Voltage_smooth'].diff()
    data['dTemp'] = data['Temp_avg'].diff()

    # Stress index
    data['Stress_Index'] = (
        np.abs(data['Current_smooth']) *
        data['Temp_avg'] *
        data['Volt_diff']
    )

    # Normalized cycle
    data['Cycle_norm'] = data['Cycle'] / data['Cycle'].max()
    return data

# -------------------------
# Train/test split
# -------------------------
def split_data(data):
    features = [
        'Voltage_smooth','Current_smooth','SOC','Cycle_norm',
        'Temp_avg','Temp_diff','Volt_diff','dSoC','dVolt','dTemp','Stress_Index'
    ]
    X = data[features]
    y = data['SoH']
    split_idx = int(len(data) * 0.8)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

# -------------------------
# Model training
# -------------------------
def train_model(X_train, y_train):
    rf_model = RandomForestRegressor(
        n_estimators=500, max_depth=22,
        min_samples_leaf=2, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

# -------------------------
# Evaluation
# -------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, mae, rmse

def summarize_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R²'],
        'Value': [round(mae, 3), round(rmse, 3), round(r2, 3)]
    })

# -------------------------
# Abnormal aging detection
# -------------------------
def detect_abnormal_aging(y_test, y_pred, data_test, sigma_factor=3):
    residuals = np.abs(y_test.values - y_pred)
    threshold = residuals.mean() + sigma_factor * residuals.std()
    data_test['Abnormal_Aging'] = residuals > threshold
    return data_test, data_test['Abnormal_Aging'].sum()

# -------------------------
# Visualization
# -------------------------
def plot_soh(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(y_test.to_numpy(), label='Actual SoH', linewidth=2)
    ax.plot(y_pred, label='Predicted SoH', linewidth=2, linestyle='--')
    ax.set_xlabel("Time Index"); ax.set_ylabel("SoH (%)")
    ax.set_title("Battery SoH Degradation")
    ax.legend(); ax.grid(True); fig.tight_layout()
    return fig

def plot_thermal(data, threshold=45):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(data['Temp1'], label='Temp1')
    ax.plot(data['Temp2'], label='Temp2')
    ax.axhline(threshold, linestyle='--', color='r', label=f'Thermal Risk ({threshold}°C)')
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Battery Thermal Behavior")
    ax.legend(); ax.grid(True)
    return fig

# -------------------------
# Second-life decision
# -------------------------
def second_life_decision(current_soh):
    if current_soh >= 80:
        return "Grade A – EV Use", "Battery is in excellent health. Suitable for EV reuse."
    elif current_soh >= 60:
        return "Grade B – Solar / Home Storage", "Battery shows moderate degradation. Suitable for stationary storage."
    else:
        return "Grade C – Recycling", "Battery is significantly degraded. Recommended for recycling."

def summarize_second_life(data):
    deg_data = data[['Cycle','SoH']].dropna().sort_values('Cycle')
    current_cycle = deg_data['Cycle'].max()
    current_soh = data.loc[data['Cycle'] == current_cycle, 'SoH'].mean()
    grade, description = second_life_decision(current_soh)
    return pd.DataFrame({
        'Metric': ['State of Health (SoH)', 'Second-Life Grade', 'Decision Description'],
        'Value': [f"{current_soh:.2f} %", grade, description]
    })

# -------------------------
# Feature importance
# -------------------------
def get_feature_importance(rf_model, features, width=10, height=4):
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

    importance_df = pd.DataFrame({
        'Factor': importances.index,
        'Impact (%)': (importances.values * 100).round(2)
    })

    fig, ax = plt.subplots(figsize=(width, height))
    importances.plot(kind='bar', ax=ax, color="#4C78A8", title="Factors affecting the degradation")
    ax.set_ylabel("Importance")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()

    return importance_df, fig