import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Example using solar data
df = pd.read_csv('outData/solar_energy_weather.csv', parse_dates=['utc_timestamp'])
df = df.drop(columns=['utc_timestamp'])  # Drop timestamp
correlation_matrix = df.corr()

# Plot heatmap (optional but looks great in your report)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()

# Identify features with high correlation
threshold = 0.9
high_corr_pairs = [
    (col1, col2)
    for col1 in correlation_matrix.columns
    for col2 in correlation_matrix.columns
    if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold
]

print("Highly correlated pairs (|corr| > 0.9):")
for pair in high_corr_pairs:
    print(pair)