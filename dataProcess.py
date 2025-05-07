import pandas as pd

# Load energy data
df_energy = pd.read_csv('./data/energy_generate_60min.csv', parse_dates=['utc_timestamp'])

# Load weather data
df_weather = pd.read_csv('./data/weather.csv')

# Convert weather timestamp (YYYYMMDDTHHMM) to (YYYY-MM-DDTHH:MM:SSZ)
df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'], format='%Y%m%dT%H%M', utc=True)
df_weather['timestamp'] = df_weather['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')  # Format to match energy data

# Convert energy timestamp to match format
df_energy['utc_timestamp'] = pd.to_datetime(df_energy['utc_timestamp'], utc=True)
df_energy['utc_timestamp'] = df_energy['utc_timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Convert both timestamps back to datetime for merging
df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
df_energy['utc_timestamp'] = pd.to_datetime(df_energy['utc_timestamp'])

# Select relevant columns
solar_cols = ['utc_timestamp', 'GB_GBN_solar_capacity', 'GB_GBN_solar_generation_actual', 'GB_GBN_solar_profile']
wind_cols = ['utc_timestamp', 'GB_GBN_wind_capacity', 'GB_GBN_wind_generation_actual', 'GB_GBN_wind_profile']
wind_offshore_cols = ['utc_timestamp', 'GB_GBN_wind_offshore_capacity', 'GB_GBN_wind_offshore_generation_actual', 'GB_GBN_wind_offshore_profile']
wind_onshore_cols = ['utc_timestamp', 'GB_GBN_wind_onshore_capacity', 'GB_GBN_wind_onshore_generation_actual', 'GB_GBN_wind_onshore_profile']

# Merge datasets
solar = df_energy[solar_cols].merge(df_weather, left_on='utc_timestamp', right_on='timestamp', how='inner')
wind = df_energy[wind_cols].merge(df_weather, left_on='utc_timestamp', right_on='timestamp', how='inner')
wind_offshore = df_energy[wind_offshore_cols].merge(df_weather, left_on='utc_timestamp', right_on='timestamp', how='inner')
wind_onshore = df_energy[wind_onshore_cols].merge(df_weather, left_on='utc_timestamp', right_on='timestamp', how='inner')

# Drop redundant timestamp column
solar.drop(columns=['timestamp'], inplace=True)
wind.drop(columns=['timestamp'], inplace=True)
wind_offshore.drop(columns=['timestamp'], inplace=True)
wind_onshore.drop(columns=['timestamp'], inplace=True)

# Merge wind and solar data on utc_timestamp
hybrid = solar.merge(wind, on='utc_timestamp', how='inner')

# Create hybrid energy columns
hybrid['hybrid_capacity'] = hybrid['GB_GBN_solar_capacity'] + hybrid['GB_GBN_wind_capacity']
hybrid['hybrid_generation_actual'] = hybrid['GB_GBN_solar_generation_actual'] + hybrid['GB_GBN_wind_generation_actual']
hybrid['hybrid_profile'] = hybrid['GB_GBN_solar_profile'] + hybrid['GB_GBN_wind_profile']

# Remove individual solar/wind columns
hybrid.drop(columns=[
    'GB_GBN_solar_capacity',
    'GB_GBN_solar_generation_actual',
    'GB_GBN_solar_profile',
    'GB_GBN_wind_capacity',
    'GB_GBN_wind_generation_actual',
    'GB_GBN_wind_profile'
], inplace=True)

# Save the hybrid dataset
hybrid.to_csv('./outData/hybrid_energy_weather.csv', index=False)

# Save the cleaned datasets inside ./outData
solar.to_csv('./outData/solar_energy_weather.csv', index=False)
wind.to_csv('./outData/wind_energy_weather.csv', index=False)
wind_offshore.to_csv('./outData/wind_offshore_energy_weather.csv', index=False)
wind_onshore.to_csv('./outData/wind_onshore_energy_weather.csv', index=False)

print("Data successfully merged and saved in ./outData!")
print("Hybrid energy dataset successfully saved in ./outData/hybrid_energy_weather.csv!")