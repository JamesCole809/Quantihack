import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Load data ----

# Chess
games = pd.read_csv('data/archive4/games.csv', usecols=['created_at','white_rating'], engine='python')
games['year'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce').dt.year
a4 = games.groupby('year').agg(chess_avg_white=('white_rating', 'mean')).reset_index()

# Climate
climate = pd.read_csv('data/archive21/train_timeseries/train_timeseries.csv',
    usecols=['date','WS10M'], nrows=5000000)
climate['year'] = pd.to_datetime(climate['date'], errors='coerce').dt.year
a21 = climate.groupby('year').agg(climate_avg_wind=('WS10M', 'mean')).reset_index()

# Asylum
asilo_ca = pd.read_csv('data/archive3/AsiloCA.csv', encoding='latin1')
yr_col = [c for c in asilo_ca.columns if 'o' in c.lower()][-1]
a3 = asilo_ca.groupby(yr_col).agg(asylum_region_total=('Solicitantes', 'sum')).reset_index()
a3.columns = ['year', 'asylum_region_total']

# UK Accidents
uk_acc = pd.read_csv('data/archive24/accident_data.csv', usecols=['Year','Number_of_Vehicles'])
a24 = uk_acc.groupby('Year').agg(uk_acc_avg_vehicles=('Number_of_Vehicles', 'mean')).reset_index()
a24.columns = ['year', 'uk_acc_avg_vehicles']

# Coal
coal = pd.read_csv('data/archive27/world-coal-production.csv', sep=';')
a27_prod = coal.groupby('Year').agg(coal_avg_production=('Value (Million Tonnes)', 'mean')).reset_index()
a27_prod.columns = ['year', 'coal_avg_production']
a27_max = coal.groupby('Year').agg(coal_total_production=('Value (Million Tonnes)', 'sum')).reset_index()
a27_max.columns = ['year', 'coal_total_production']
a27 = a27_prod.merge(a27_max, on='year')

# US Accidents
acc = pd.read_csv('data/archive7/accident.csv', usecols=['accident_id','YEAR','FATALS'])
a7 = acc.groupby('YEAR').agg(us_acc_avg_fatals=('FATALS', 'mean')).reset_index()
a7.columns = ['year', 'us_acc_avg_fatals']

vehicle = pd.read_csv('data/archive7/vehicle.csv', usecols=['Year','MOD_YEAR'])
a7v = vehicle.groupby('Year').agg(us_vehicle_avg_mod_year=('MOD_YEAR', lambda x: x[x < 9999].mean())).reset_index()
a7v.columns = ['year', 'us_vehicle_avg_mod_year']

def zscore(s):
    return (s - s.mean()) / s.std()

def norm(s):
    return (s - s.min()) / (s.max() - s.min())

fig, axes = plt.subplots(3, 1, figsize=(13, 14))
fig.suptitle('Absurd Z-Score Correlations', fontsize=16, fontweight='bold')

# ---- Plot 1: Chess skill x Wind speed predicts asylum applications ----
m1 = a4.merge(a21, on='year').merge(a3, on='year').dropna()
m1['z_chess'] = zscore(m1['chess_avg_white'])
m1['z_wind'] = zscore(m1['climate_avg_wind'])
m1['z_product'] = m1['z_chess'] * m1['z_wind']
r1 = m1['z_product'].corr(m1['asylum_region_total'])

ax = axes[0]
ax.plot(m1['year'], norm(m1['z_chess']), 'o-', color='#4C72B0', lw=2, ms=7, label='Chess Avg White Rating (z)')
ax.plot(m1['year'], norm(m1['z_wind']), 's-', color='#55A868', lw=2, ms=7, label='Avg Wind Speed (z)')
ax.plot(m1['year'], norm(m1['z_product']), 'D-', color='#8172B2', lw=2.5, ms=8, label='Z-Product (Chess x Wind)')
ax.plot(m1['year'], norm(m1['asylum_region_total']), '^-', color='#C44E52', lw=2, ms=7, label='Asylum Applications (Spain)')
ax.set_title(f'Chess Skill x Wind Speed Predicts Spanish Asylum Applications (r = {r1:.4f})', fontsize=12)
ax.set_xlabel('Year'); ax.set_ylabel('Normalised')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ---- Plot 2: UK accident vehicles x Coal production predicts temperature ----
m2 = a24.merge(a27, on='year').merge(a21, on='year').dropna()
m2['z_uk_veh'] = zscore(m2['uk_acc_avg_vehicles'])
m2['z_coal'] = zscore(m2['coal_avg_production'])
m2['z_product'] = m2['z_uk_veh'] * m2['z_coal']
# Use climate_avg_wind as proxy for temp — actually load temp
climate2 = pd.read_csv('data/archive21/train_timeseries/train_timeseries.csv',
    usecols=['date','T2M'], nrows=5000000)
climate2['year'] = pd.to_datetime(climate2['date'], errors='coerce').dt.year
a21t = climate2.groupby('year').agg(climate_avg_temp=('T2M', 'mean')).reset_index()
m2 = m2.merge(a21t, on='year').dropna()
r2 = m2['z_product'].corr(m2['climate_avg_temp'])

ax = axes[1]
ax.plot(m2['year'], norm(m2['z_uk_veh']), 'o-', color='#4C72B0', lw=2, ms=7, label='UK Vehicles per Crash (z)')
ax.plot(m2['year'], norm(m2['z_coal']), 's-', color='#55A868', lw=2, ms=7, label='Coal Production (z)')
ax.plot(m2['year'], norm(m2['z_product']), 'D-', color='#8172B2', lw=2.5, ms=8, label='Z-Product (UK Crashes x Coal)')
ax.plot(m2['year'], norm(m2['climate_avg_temp']), '^-', color='#C44E52', lw=2, ms=7, label='Avg Temperature')
ax.set_title(f'UK Crash Vehicles x Coal Production Predicts Temperature (r = {r2:.4f})', fontsize=12)
ax.set_xlabel('Year'); ax.set_ylabel('Normalised')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ---- Plot 3: Coal / US fatality rate predicts chess game length ----
games2 = pd.read_csv('data/archive4/games.csv', usecols=['created_at','turns'], engine='python')
games2['year'] = pd.to_datetime(games2['created_at'], unit='ms', errors='coerce').dt.year
a4t = games2.groupby('year').agg(chess_avg_turns=('turns', 'mean')).reset_index()

m3 = a27.merge(a7, on='year').merge(a4t, on='year').dropna()
m3['z_coal'] = zscore(m3['coal_total_production'])
m3['z_fatals'] = zscore(m3['us_acc_avg_fatals'])
safe = m3['z_fatals'].abs() > 0.1
m3['z_ratio'] = np.nan
m3.loc[safe, 'z_ratio'] = m3.loc[safe, 'z_coal'] / m3.loc[safe, 'z_fatals']
m3 = m3.dropna(subset=['z_ratio'])
r3 = m3['z_ratio'].corr(m3['chess_avg_turns'])

ax = axes[2]
ax.plot(m3['year'], norm(m3['z_coal']), 'o-', color='#4C72B0', lw=2, ms=7, label='Coal Production (z)')
ax.plot(m3['year'], norm(m3['z_fatals']), 's-', color='#55A868', lw=2, ms=7, label='US Fatality Rate (z)')
ax.plot(m3['year'], norm(m3['z_ratio']), 'D-', color='#8172B2', lw=2.5, ms=8, label='Z-Ratio (Coal / Fatality)')
ax.plot(m3['year'], norm(m3['chess_avg_turns']), '^-', color='#C44E52', lw=2, ms=7, label='Chess Avg Turns')
ax.set_title(f'Coal Production / US Fatality Rate Predicts Chess Game Length (r = {r3:.4f})', fontsize=12)
ax.set_xlabel('Year'); ax.set_ylabel('Normalised')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('zscore_correlations.png', dpi=150, bbox_inches='tight')
print(f"Plot 1: r = {r1:.4f}, n = {len(m1)}")
print(f"Plot 2: r = {r2:.4f}, n = {len(m2)}")
print(f"Plot 3: r = {r3:.4f}, n = {len(m3)}")
print("Saved to zscore_correlations.png")
