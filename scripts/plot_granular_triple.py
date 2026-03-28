import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# UK Accidents - monthly
uk = pd.read_csv('data/archive24/accident_data.csv', usecols=['Date','Year','Number_of_Vehicles'])
uk['dt'] = pd.to_datetime(uk['Date'], format='%d/%m/%Y', errors='coerce')
uk['ym'] = uk['dt'].dt.to_period('M')
uk_monthly = uk.groupby('ym').agg(
    uk_acc_avg_vehicles=('Number_of_Vehicles', 'mean'),
    uk_acc_count=('Number_of_Vehicles', 'count'),
).reset_index()
uk_monthly['date'] = uk_monthly['ym'].dt.to_timestamp()
uk_monthly['year'] = uk_monthly['date'].dt.year

# Climate - monthly
climate = pd.read_csv('data/archive21/train_timeseries/train_timeseries.csv',
    usecols=['date','T2M'], nrows=5000000)
climate['dt'] = pd.to_datetime(climate['date'], errors='coerce')
climate['ym'] = climate['dt'].dt.to_period('M')
clim_monthly = climate.groupby('ym').agg(climate_avg_temp=('T2M', 'mean')).reset_index()
clim_monthly['date'] = clim_monthly['ym'].dt.to_timestamp()

# Coal - yearly (spread to monthly)
coal = pd.read_csv('data/archive27/world-coal-production.csv', sep=';')
coal_yearly = coal.groupby('Year').agg(coal_avg_production=('Value (Million Tonnes)', 'mean')).reset_index()
coal_yearly.columns = ['year', 'coal_avg_production']

# Merge monthly UK + Climate, then join yearly coal
merged = uk_monthly.merge(clim_monthly, on='ym', suffixes=('_uk','_clim'))
merged = merged.merge(coal_yearly, on='year')
merged = merged.sort_values('date_uk')

print(f"Monthly data points: {len(merged)}")
print(f"Year range: {merged['year'].min()}-{merged['year'].max()}")

# Z-scores
def zscore(s):
    return (s - s.mean()) / s.std()

def norm(s):
    return (s - s.min()) / (s.max() - s.min())

merged['z_uk_veh'] = zscore(merged['uk_acc_avg_vehicles'])
merged['z_coal'] = zscore(merged['coal_avg_production'])
merged['z_product'] = merged['z_uk_veh'] * merged['z_coal']

r = merged['z_product'].corr(merged['climate_avg_temp'])
print(f"Monthly correlation (z_product vs temp): r = {r:.4f}")

# Also raw correlations
r_uk_temp = merged['uk_acc_avg_vehicles'].corr(merged['climate_avg_temp'])
r_coal_temp = merged['coal_avg_production'].corr(merged['climate_avg_temp'])
r_uk_coal = merged['uk_acc_avg_vehicles'].corr(merged['coal_avg_production'])
print(f"UK vehicles vs temp: r = {r_uk_temp:.4f}")
print(f"Coal vs temp: r = {r_coal_temp:.4f}")
print(f"UK vehicles vs coal: r = {r_uk_coal:.4f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(merged['date_uk'], norm(merged['z_uk_veh']), '-', color='#4C72B0', lw=1.5, alpha=0.7, label='UK Vehicles per Crash (z)')
ax.plot(merged['date_uk'], norm(merged['z_coal']), '-', color='#55A868', lw=1.5, alpha=0.7, label='Coal Production (z)')
ax.plot(merged['date_uk'], norm(merged['z_product']), '-', color='#8172B2', lw=2, label='Z-Product (UK Crashes x Coal)')
ax.plot(merged['date_uk'], norm(merged['climate_avg_temp']), '-', color='#C44E52', lw=2, label='Avg Temperature')

ax.set_title(f'UK Crash Vehicles x Coal Production vs Temperature — Monthly (r = {r:.4f})', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Normalised Value')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('granular_triple.png', dpi=150, bbox_inches='tight')
print("Saved to granular_triple.png")
