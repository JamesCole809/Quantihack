import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Build signal ----
coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
coal_yr = coal.groupby('Year')['Value (Million Tonnes)'].max().reset_index()
coal_yr.columns = ['year', 'coal_max']

acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['year'] = pd.to_datetime(acc0515['Date'], errors='coerce').dt.year
crashes = acc0515.groupby('year')['Number_of_Casualties'].mean().reset_index()
crashes.columns = ['year', 'crash_casualties']

weather = pd.read_parquet('data/archive (10)/all_weather_data.parquet')
weather['year'] = pd.to_datetime(weather['date'], errors='coerce').dt.year
hum_col = [c for c in weather.columns if 'humidity' in c.lower()][0]
humidity = weather.groupby('year')[hum_col].mean().reset_index()
humidity.columns = ['year', 'humidity']

# DBA Agriculture ETF
dba = yf.download('DBA', start='2008-01-01', end='2015-01-01', progress=False, auto_adjust=True)
dba['year'] = dba.index.year
dba_yr = dba.groupby('year')['Close'].mean().reset_index()
dba_yr.columns = ['year', 'dba_price']

# Wheat
wheat = yf.download('ZW=F', start='2008-01-01', end='2015-01-01', progress=False, auto_adjust=True)
wheat['year'] = wheat.index.year
wheat_yr = wheat.groupby('year')['Close'].mean().reset_index()
wheat_yr.columns = ['year', 'wheat_price']

# Merge 2009-2014
merged = coal_yr.merge(crashes, on='year').merge(humidity, on='year').merge(dba_yr, on='year').merge(wheat_yr, on='year')
merged = merged[(merged['year'] >= 2009) & (merged['year'] <= 2014)].sort_values('year').reset_index(drop=True)

def z(s):
    return (s - s.mean()) / s.std()

def norm(s):
    return (s - s.min()) / (s.max() - s.min())

merged['z_product'] = z(merged['coal_max']) * z(merged['crash_casualties'])

r_hum = merged['z_product'].corr(merged['humidity'])
r_dba = merged['z_product'].corr(merged['dba_price'])
r_wht = merged['z_product'].corr(merged['wheat_price'])

labels = [f'Y{i+1}' for i in range(len(merged))]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(range(6), norm(merged['z_product']), 'D-', color='#8172B2', lw=2.5, ms=10,
        label='Z-Signal (Coal Production x Crash Casualties)')
ax.plot(range(6), norm(merged['humidity']), 's-', color='#4C72B0', lw=2, ms=9,
        label='UK Humidity')
ax.plot(range(6), norm(merged['dba_price']), '^-', color='#C44E52', lw=2, ms=9,
        label='DBA Agriculture ETF')

ax.fill_between(range(6), norm(merged['z_product']), alpha=0.08, color='#8172B2')

ax.set_title(f'Coal Production x UK Crash Casualties  \u2192  Humidity  \u2192  Agriculture ETF\n'
             f'Signal\u2192Humidity r={r_hum:.2f}  |  Signal\u2192Agriculture r={r_dba:.2f}  |  Humidity\u2192Agriculture r={r_hum:.2f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Normalised Value (0\u20131)', fontsize=12)
ax.set_xticks(range(6))
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/wheat_signal_final.png', dpi=150, bbox_inches='tight')

print(f'Signal -> Humidity:     r = {r_hum:.4f}')
print(f'Signal -> DBA:          r = {r_dba:.4f}')
print(f'Signal -> Wheat:        r = {r_wht:.4f}')
print()
print(merged[['year', 'z_product', 'humidity', 'dba_price', 'wheat_price']].to_string(index=False))
print('\nSaved to plots/wheat_signal_final.png')
