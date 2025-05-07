import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# File path
file_path = r"predictions/hybrid_predictions.csv"
df = pd.read_csv(file_path, parse_dates=["utc_timestamp"])
df.columns = df.columns.str.strip()
df.sort_values("utc_timestamp", inplace=True)

# Auto-detect generation column 
auto_cols = [col for col in df.columns if "generation" in col.lower()]
if not auto_cols:
    raise ValueError("No column with 'generation' found.")
predicted_col = auto_cols[0]
predictions = df[predicted_col]

# Simulate synthetic demand 
base_demand = predictions.mean()
demand_noise = np.sin(np.linspace(0, 10 * np.pi, len(df))) * 300
demand_series = base_demand + demand_noise

# Battery setup 
battery_capacity = 15000
battery_state = battery_capacity / 2
charge_efficiency = 0.9
discharge_efficiency = 0.9
max_rate = 500
small_threshold = 1e-2

actions, levels = [], []

for idx, pred in enumerate(predictions):
    demand = demand_series[idx]
    net_energy = pred - demand
    if abs(net_energy) < small_threshold:
        actions.append("Idle")
    elif net_energy > 0:
        charge = min(net_energy * charge_efficiency, max_rate)
        battery_state = min(battery_capacity, battery_state + charge)
        actions.append("Store")
    else:
        discharge = min(abs(net_energy) / discharge_efficiency, max_rate)
        battery_state = max(0, battery_state - discharge)
        actions.append("Discharge")
    levels.append(battery_state)

df["Battery_Level"] = levels
df["Battery_Action"] = actions
df["Smoothed_Battery_Level"] = df["Battery_Level"].rolling(window=24, min_periods=1).mean()

# Plotting setup 
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (18, 7),
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.color": "lightgray",
    "grid.alpha": 0.5
})
fig, axs = plt.subplots(1, 2)
years = pd.date_range(start='2015', end='2020', freq='YS')
y_limit = (0, battery_capacity + 1000)

for ax in axs:
    ax.set_ylim(*y_limit)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Energy Stored (MW)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    for year in years:
        ax.axvline(year, color='gray', linestyle='--', linewidth=0.5)
    ax.axhspan(0, 2000, facecolor='red', alpha=0.05)
    ax.axhspan(12000, battery_capacity + 1000, facecolor='green', alpha=0.05)
    ax.grid(True)

# Left plot: Smoothed battery + predictions 
axs[0].plot(df["utc_timestamp"], df["Smoothed_Battery_Level"], label="Smoothed Level", color="royalblue", linewidth=1.8)
axs[0].plot(df["utc_timestamp"], predictions.rolling(24).mean(), label="Predicted Energy", color="darkorange", linewidth=1.5)
axs[0].set_title("Smoothed Battery Level (24-Hour Rolling Avg)")
axs[0].legend(loc="upper right", fontsize=10)

# Right plot: Raw battery level 
axs[1].fill_between(df["utc_timestamp"], df["Battery_Level"], color="skyblue", alpha=0.7)
axs[1].set_title("Raw Battery Level Over Time")

# Add average battery line 
avg_level = df["Battery_Level"].mean()
axs[1].axhline(avg_level, color='gray', linestyle='--', alpha=0.5, label='Average Level')

# Annotate peak battery level 
max_idx = df["Battery_Level"].idxmax()
max_time = df.loc[max_idx, "utc_timestamp"]
max_val = df.loc[max_idx, "Battery_Level"]
axs[1].annotate("Peak",
                xy=(max_time, max_val),
                xytext=(max_time, max_val + 300),
                ha='center',
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=10, color="green")

axs[1].legend(loc="lower right", fontsize=10)

# Finalize 
fig.suptitle("Battery Energy Storage Simulation (Hybrid Forecast, 2015–2020)", fontsize=18, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("battery_storage_simulation_perfect.png", dpi=400)
plt.savefig("battery_storage_simulation_perfect.pdf", dpi=400)
df.to_csv("battery_storage_simulation_data.csv", index=False)
plt.show()

print("Perfect final version generated and exported (PNG, PDF, CSV)")
