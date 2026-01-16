import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

from model import (
    feature_engineering, split_data, train_model,
    evaluate_model, detect_abnormal_aging,
    summarize_second_life,
    get_feature_importance, summarize_metrics,
    plot_thermal
)

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Battery SoH Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Theme-safe CSS
# -------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
div[data-testid="metric-container"] {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# =================================================
# ANIMATION FUNCTIONS
# =================================================

def animated_soh_plot(y_test, y_pred, delay=0.002):
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Battery SoH Degradation (Live)")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("SoH (%)")
    ax.grid(True)
    actual_line, = ax.plot([], [], label="Actual SoH", linewidth=2)
    pred_line, = ax.plot([], [], "--", label="Predicted SoH", linewidth=2)
    ax.legend()
    y_test_vals = y_test.to_numpy()
    x_vals = np.arange(len(y_test_vals))
    ymin = min(y_test_vals.min(), y_pred.min()) - 2
    ymax = max(y_test_vals.max(), y_pred.max()) + 2
    for i in range(1, len(x_vals) + 1):
        actual_line.set_data(x_vals[:i], y_test_vals[:i])
        pred_line.set_data(x_vals[:i], y_pred[:i])
        ax.set_xlim(0, len(x_vals))
        ax.set_ylim(ymin, ymax)
        placeholder.pyplot(fig)
        time.sleep(delay)

def animated_thermal_plot(data, delay=0.002, step=5, threshold=45):
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Battery Thermal Behavior (Live)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Time Index")
    ax.grid(True)
    t1, = ax.plot([], [], label="Temp1", linewidth=2)
    t2, = ax.plot([], [], label="Temp2", linewidth=2)
    ax.axhline(threshold, linestyle="--", color="red", label=f"Thermal Risk ({threshold}°C)")
    ax.legend()
    # Downsample data
    temp1 = data["Temp1"].values[::step]
    temp2 = data["Temp2"].values[::step]
    x_vals = range(len(temp1))
    ymin = min(temp1.min(), temp2.min()) - 2
    ymax = max(temp1.max(), temp2.max()) + 2
    for i in range(1, len(temp1) + 1):
        t1.set_data(x_vals[:i], temp1[:i])
        t2.set_data(x_vals[:i], temp2[:i])
        ax.set_xlim(0, len(temp1))
        ax.set_ylim(ymin, ymax)
        placeholder.pyplot(fig)
        time.sleep(delay)

def animated_feature_importance(importance_df, delay=0.002):
    placeholder = st.empty()
    fig, ax = plt.subplots(figsize=(12, 4))
    features = importance_df["Factor"].values
    values = importance_df["Impact (%)"].values
    colors = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756",
        "#72B7B2", "#EECA3B", "#B279A2", "#FF9DA6",
        "#9D755D", "#BAB0AC", "#1F77B4"
    ]
    for i in range(1, len(values) + 1):
        ax.clear()
        ax.bar(features[:i], values[:i], color=colors[:i])
        ax.set_title("Feature Importance (Live Build)")
        ax.set_ylabel("Impact (%)")
        ax.set_ylim(0, max(values) + 5)
        ax.set_xticks(range(i))
        ax.set_xticklabels(features[:i], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        placeholder.pyplot(fig)
        time.sleep(delay)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🔋 Battery State of Health (SoH) Dashboard")
st.caption("Animated ML-Based Battery Health Monitoring System")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload Battery Dataset", type=["csv", "xlsx"])
    threshold = st.slider("Thermal Risk Threshold (°C)", 30, 70, 45)
    st.info("💡 Light/Dark mode can be changed from Streamlit settings.")

# -------------------------------------------------
# Main Logic
# -------------------------------------------------
if uploaded_file:
    try:
        # Load data
        data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        data.columns = data.columns.str.strip()
        # ✅ Normalize column names
        data.rename(columns={
            'Pack Vol': 'Voltage',
            'Curent': 'Current',
            'Soc': 'SOC'
        }, inplace=True)
        # Compute SoH if capacity columns exist
        if "Rem. Ah" in data.columns and "Full Cap" in data.columns:
            data["SoH"] = (data["Rem. Ah"] / data["Full Cap"]) * 100
            data = data[data["SoH"].notna()]
        else:
            st.error("Dataset must include 'Rem. Ah' and 'Full Cap' columns to calculate SoH.")
            st.stop()

        # Feature engineering
        data = feature_engineering(data)

        # Split
        X_train, X_test, y_train, y_test = split_data(data)

        # Train
        model = train_model(X_train, y_train)

        # Evaluate
        y_pred, mae, rmse = evaluate_model(model, X_test, y_test)

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance",
            "📉 SoH (Animated)",
            "🌡️ Thermal (Animated)",
            "♻️ Second-Life",
            "🧠 Feature Importance (Animated)"
        ])

        # Performance
        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")
            c3.metric("Samples", len(y_test))
            st.dataframe(summarize_metrics(y_test, y_pred), use_container_width=True)
            _, abnormal_count = detect_abnormal_aging(
                y_test, y_pred, data.iloc[len(X_train):].copy()
            )
            st.warning(f"⚠️ Abnormal aging events detected: {abnormal_count}")

        # SoH animated
        with tab2:
            st.info("▶️ Building SoH graph in real time...")
            animated_soh_plot(y_test, y_pred)

        # Thermal animated
        with tab3:
            st.info("▶️ Building thermal graph in real time...")
            animated_thermal_plot(data, threshold=threshold)
            # Optional static thermal plot
            st.pyplot(plot_thermal(data, threshold=threshold))

        # Second life
        with tab4:
            st.dataframe(summarize_second_life(data), use_container_width=True)
            st.caption("Thresholds: ≥80% → EV Use, ≥60% → Stationary Storage, <60% → Recycling")

        # Feature importance animated
        with tab5:
            features = [
                'Voltage_smooth','Current_smooth','SOC','Cycle_norm',
                'Temp_avg','Temp_diff','Volt_diff',
                'dSoC','dVolt','dTemp','Stress_Index'
            ]
            importance_df, _ = get_feature_importance(model, features)
            animated_feature_importance(importance_df)
            st.dataframe(importance_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("⬅️ Upload a dataset from the sidebar to begin.")