import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_path = r"predictions/hybrid_predictions.csv"
df = pd.read_csv(file_path, parse_dates=["utc_timestamp"])
df.columns = df.columns.str.strip()
df.sort_values("utc_timestamp", inplace=True)

# Detect prediction column
predicted_col = [col for col in df.columns if "generation" in col.lower()][0]
predictions = df[predicted_col]

# Simulated demand
base_demand = predictions.mean()
demand_variation = np.sin(np.linspace(0, 10 * np.pi, len(df))) * 300
demand_series = base_demand + demand_variation

# Battery config
battery_capacity = 15000
battery_level = battery_capacity / 2
charge_eff = 0.9
discharge_eff = 0.9
max_rate = 500
threshold = 25

# Simulation tracking
battery_logs = []
total_energy_stored = 0
total_energy_discharged = 0
full_hits = 0
empty_hits = 0
idle_count = 0

for i in range(len(df)):
    ts = df.iloc[i]["utc_timestamp"]
    predicted = predictions.iloc[i]
    demand = demand_series[i]
    net_energy = predicted - demand

    energy_stored = 0
    energy_discharged = 0

    if abs(net_energy) < threshold:
        action = "Idle"
        idle_count += 1
    elif net_energy > 0:
        charge = min(net_energy * charge_eff, max_rate)
        if battery_level + charge >= battery_capacity:
            full_hits += 1
        energy_stored = min(charge, battery_capacity - battery_level)
        battery_level = min(battery_capacity, battery_level + charge)
        total_energy_stored += energy_stored
        action = "Store"
    else:
        discharge = min(abs(net_energy) / discharge_eff, max_rate)
        if battery_level - discharge <= 0:
            empty_hits += 1
        energy_discharged = min(discharge, battery_level)
        battery_level = max(0, battery_level - discharge)
        total_energy_discharged += energy_discharged
        action = "Discharge"

    battery_logs.append({
        "utc_timestamp": ts,
        "predicted_energy_generation": round(predicted, 2),
        "Battery_Action": action,
        "Battery_Level": round(battery_level, 2),
        "Energy_Stored": round(energy_stored, 2),
        "Energy_Discharged": round(energy_discharged, 2)
    })

# Create log dataframe and save
log_df = pd.DataFrame(battery_logs)
log_df.to_csv("battery_action_log.csv", index=False)

# Metrics summary
total_entries = len(log_df)
idle_percentage = round(100 * idle_count / total_entries, 2)

metrics = {
    "Total Energy Stored (MWh)": round(total_energy_stored, 2),
    "Total Energy Discharged (MWh)": round(total_energy_discharged, 2),
    "Battery Full Events": full_hits,
    "Battery Empty Events": empty_hits,
    "Idle Time (%)": idle_percentage
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("battery_performance_metrics.csv", index=False)

# Smoothed Plot
log_df["Smoothed_Stored"] = log_df["Energy_Stored"].rolling(168).mean()
log_df["Smoothed_Discharged"] = log_df["Energy_Discharged"].rolling(168).mean()

plt.figure(figsize=(15, 5))
plt.plot(log_df["utc_timestamp"], log_df["Smoothed_Stored"], label="Smoothed Energy Stored", color="green", alpha=0.7)
plt.plot(log_df["utc_timestamp"], log_df["Smoothed_Discharged"], label="Smoothed Energy Discharged", color="red", alpha=0.7)
plt.title("Smoothed Energy Stored vs Discharged Over Time (7-day Average)")
plt.xlabel("Timestamp")
plt.ylabel("Energy (MW)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig("energy_flow_trend_smoothed.png", dpi=300)
plt.show()

# Weekly Stacked Area Plot
weekly_df = log_df.set_index("utc_timestamp")
weekly_df = weekly_df.select_dtypes(include=["number"]).resample("W").mean()
weekly_df = weekly_df[["Energy_Stored", "Energy_Discharged"]].dropna()

plt.figure(figsize=(15, 5))
plt.stackplot(
    weekly_df.index,
    weekly_df["Energy_Stored"],
    weekly_df["Energy_Discharged"],
    labels=["Energy Stored", "Energy Discharged"],
    colors=["green", "red"],
    alpha=0.5
)
plt.title("Weekly Energy Flow Area Chart (Smoothed)")
plt.xlabel("Timestamp")
plt.ylabel("Energy (MW)")
plt.legend(loc="upper left")
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig("energy_flow_area_chart_weekly.png", dpi=300)
plt.show()

# Monthly Summary Table
log_df["Month"] = pd.to_datetime(log_df["utc_timestamp"]).dt.to_period("M")
monthly_summary = log_df.groupby("Month")[["Energy_Stored", "Energy_Discharged"]].mean().reset_index()
monthly_summary.columns = ["Month", "Avg_Energy_Stored", "Avg_Energy_Discharged"]
monthly_summary.to_csv("monthly_energy_summary.csv", index=False)

print("\nMonthly Energy Summary (First 12 Rows):")
print(monthly_summary.head(12))
