import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Build signal (full range, exclude 2020) ----
acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['year'] = pd.to_datetime(acc0515['Date'], errors='coerce').dt.year
rc1 = acc0515.groupby('year')['Number_of_Casualties'].mean().reset_index()
rc1.columns = ['year', 'crash_casualties']

acc1620 = pd.read_parquet('data/road-casualty-data/dft-road-casualty-statistics-accident-last-5-years.parquet')
rc2 = acc1620.groupby('accident_year')['number_of_casualties'].mean().reset_index()
rc2.columns = ['year', 'crash_casualties']

crashes = pd.concat([rc1, rc2]).drop_duplicates(subset='year', keep='first').sort_values('year')

coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
coal_yr = coal.groupby('Year')['Value (Million Tonnes)'].max().reset_index()
coal_yr.columns = ['year', 'coal_max']

weather = pd.read_parquet('data/archive (10)/all_weather_data.parquet')
weather['year'] = pd.to_datetime(weather['date'], errors='coerce').dt.year
hum_col = [c for c in weather.columns if 'humidity' in c.lower()][0]
humidity = weather.groupby('year')[hum_col].mean().reset_index()
humidity.columns = ['year', 'humidity']

# Wheat futures - extended
wheat = yf.download('ZW=F', start='2008-01-01', end='2021-01-01', progress=False, auto_adjust=True)
wheat['year'] = wheat.index.year
wheat_yr = wheat.groupby('year')['Close'].mean().reset_index()
wheat_yr.columns = ['year', 'wheat_price']

# UK feed wheat - try London wheat
uk_wheat = yf.download('EWG.L', start='2008-01-01', end='2021-01-01', progress=False, auto_adjust=True)
has_uk_wheat = len(uk_wheat) > 0

# DBA agriculture ETF
dba = yf.download('DBA', start='2008-01-01', end='2021-01-01', progress=False, auto_adjust=True)
dba['year'] = dba.index.year
dba_yr = dba.groupby('year')['Close'].mean().reset_index()
dba_yr.columns = ['year', 'dba_price']

# Merge
merged = crashes.merge(coal_yr, on='year').merge(humidity, on='year').merge(wheat_yr, on='year').merge(dba_yr, on='year', how='left')
merged = merged[(merged['year'] >= 2009) & (merged['year'] <= 2019)]  # exclude 2020
merged = merged.sort_values('year')

# Use in-sample z-score params (2009-2014)
insample = merged[merged['year'] <= 2014]
coal_mean, coal_std = insample['coal_max'].mean(), insample['coal_max'].std()
cas_mean, cas_std = insample['crash_casualties'].mean(), insample['crash_casualties'].std()

merged['z_coal'] = (merged['coal_max'] - coal_mean) / coal_std
merged['z_cas'] = (merged['crash_casualties'] - cas_mean) / cas_std
merged['z_product'] = merged['z_coal'] * merged['z_cas']

r_hum = merged['z_product'].corr(merged['humidity'])
r_wht = merged['z_product'].corr(merged['wheat_price'])
r_dba = merged['z_product'].corr(merged['dba_price']) if 'dba_price' in merged.columns else 0

print(f"Range: {int(merged['year'].min())}-{int(merged['year'].max())} ({len(merged)} years, excl 2020)")
print(f"Signal -> Humidity:  r = {r_hum:.4f}")
print(f"Signal -> Wheat:     r = {r_wht:.4f}")
print(f"Signal -> DBA:       r = {r_dba:.4f}")
print()
print(merged[['year','z_product','humidity','wheat_price']].to_string(index=False))

# ---- Plot ----
def norm(s):
    return (s - s.min()) / (s.max() - s.min())

fig, ax = plt.subplots(figsize=(14, 6))

years = merged['year']
ax.plot(years, norm(merged['z_product']), 'D-', color='#8172B2', lw=2.5, ms=8,
        label='Z-Signal (Coal x Crash Casualties)')
ax.plot(years, norm(merged['humidity']), 's-', color='#4C72B0', lw=2, ms=7,
        label='UK Humidity')
ax.plot(years, norm(merged['dba_price']), '^-', color='#C44E52', lw=2, ms=7,
        label='DBA Agriculture ETF')

# Mark in-sample vs out-of-sample
ax.axvline(x=2014.5, color='gray', linestyle='--', alpha=0.5)
ax.text(2011.5, 0.02, 'IN-SAMPLE', fontsize=10, color='gray', ha='center', style='italic')
ax.text(2017, 0.02, 'OUT-OF-SAMPLE', fontsize=10, color='gray', ha='center', style='italic')

ax.set_title(f'Coal x UK Crash Casualties  →  Humidity  →  Agriculture ETF (2009-2019, excl. 2020)\n'
             f'Signal→Humidity r={r_hum:.2f}  |  Signal→DBA r={r_dba:.2f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Normalised Value (0-1)', fontsize=12)
ax.set_xticks(range(2009, 2020))
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/wheat_signal_extended.png', dpi=150, bbox_inches='tight')
print("\nSaved to plots/wheat_signal_extended.png")
