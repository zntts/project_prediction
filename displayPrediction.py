import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

#  Set page config 
st.set_page_config(page_title="Renewable Energy Forecast Dashboard", layout="wide")

#  Page Title 
st.title("Renewable Energy Forecast Dashboard")
st.markdown("View actual vs predicted generation for solar, wind, hybrid, and more.")

#  Sidebar 
model_options = ['solar', 'wind', 'wind_onshore', 'wind_offshore', 'hybrid']
selected_model = st.sidebar.selectbox("Select Energy Source:", model_options)

#  File Path 
prediction_file = f"predictions/{selected_model}_predictions.csv"

#  Error Handling and Data Loading 
if not os.path.exists(prediction_file):
    st.error(f"Prediction file for '{selected_model}' not found at: {prediction_file}")
else:
    df = pd.read_csv(prediction_file, parse_dates=['utc_timestamp'])
    df.sort_values(by='utc_timestamp', inplace=True)

    #  Date Range Selector 
    min_date = df['utc_timestamp'].min()
    max_date = df['utc_timestamp'].max()

    date_range = st.sidebar.date_input(
        "Select Forecast Date Range:",
        [min_date.date(), max_date.date()],
        min_value=min_date.date(),
        max_value=max_date.date()
    )

    if len(date_range) == 2:
        df = df[
            (df['utc_timestamp'].dt.date >= date_range[0]) &
            (df['utc_timestamp'].dt.date <= date_range[1])
        ]

    #  Column Detection Logic 
    prediction_cols = [col for col in df.columns if 'predicted' in col.lower()]
    actual_cols = [col for col in df.columns if 'actual' in col.lower() or 'generation_actual' in col.lower()]

    if not prediction_cols:
        st.error(f"No prediction columns found in {selected_model} dataset")
    else:
        predicted_col = prediction_cols[0]
        actual_col = actual_cols[0] if actual_cols else None

        st.subheader(f"Forecast - {selected_model.capitalize()}")
        st.write("Available Columns:", df.columns.tolist())

        #  Forecast Plot 
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['utc_timestamp'], df[predicted_col], label='Predicted', color='red', linestyle='--')
        if actual_col:
            ax.plot(df['utc_timestamp'], df[actual_col], label='Actual', color='blue')
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Energy Generation (MW)")
        ax.set_title(f"{selected_model.capitalize()} Forecast")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        #  Metrics 
        if actual_col:
            mse = mean_squared_error(df[actual_col], df[predicted_col])
            r2 = r2_score(df[actual_col], df[predicted_col])
            st.markdown("")
            st.subheader("Model Performance Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
            col2.metric("R² Score", f"{r2:.4f}")
        else:
            st.info("Actual values not available for this model. Only predictions are shown.")

        #  Battery Simulation 
        st.markdown("")
        st.subheader("Simulated Energy Storage Decision")

        battery_capacity = 10000
        battery_state = 5000
        charge_efficiency = 0.9
        discharge_efficiency = 0.9
        small_threshold = 1e-2
        max_rate = 500

        actions = []
        levels = []

        # Create variable demand
        base_demand = df[predicted_col].mean()
        demand_noise = np.sin(np.linspace(0, 10 * np.pi, len(df))) * 300
        demand_series = base_demand + demand_noise

        for idx, pred in enumerate(df[predicted_col]):
            demand = demand_series[idx]
            net_energy = pred - demand

            if abs(net_energy) < small_threshold:
                actions.append("Idle")
            elif net_energy > 0:
                charge_amount = min(net_energy * charge_efficiency, max_rate)
                battery_state = min(battery_capacity, battery_state + charge_amount)
                actions.append("Store")
            else:
                discharge_needed = abs(net_energy) / discharge_efficiency
                discharge_amount = min(discharge_needed, max_rate)
                battery_state = max(0, battery_state - discharge_amount)
                actions.append("Discharge")

            levels.append(battery_state)

        df['Battery_Action'] = actions
        df['Battery_Level'] = levels
        df['Smoothed_Battery_Level'] = df['Battery_Level'].rolling(window=24, min_periods=1).mean()

        #  Dual Chart View 
        st.subheader("Battery Level Visualization")
        col1, col2 = st.columns(2)

        with col1:
            st.line_chart(df.set_index('utc_timestamp')['Smoothed_Battery_Level'])
            st.caption("Line Chart: 24-Hour Rolling Average")

        with col2:
            st.area_chart(df.set_index('utc_timestamp')['Smoothed_Battery_Level'])
            st.caption("Area Chart: Battery Fill Over Time")

        #  Action Log 
        with st.expander("Battery Actions Log"):
            st.dataframe(df[['utc_timestamp', predicted_col, 'Battery_Action', 'Battery_Level']].tail(50))

        #  Raw Data 
        with st.expander("View Raw Data"):
            st.dataframe(df.tail(50))