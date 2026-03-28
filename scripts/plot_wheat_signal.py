import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Build z-score signal ----
coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
coal_yr = coal.groupby('Year')['Value (Million Tonnes)'].max().reset_index()
coal_yr.columns = ['year', 'coal_max']

acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['year'] = pd.to_datetime(acc0515['Date'], errors='coerce').dt.year
crashes = acc0515.groupby('year')['Number_of_Casualties'].mean().reset_index()
crashes.columns = ['year', 'uk_crash_avg_casualties']

signal = coal_yr.merge(crashes, on='year')
signal = signal[(signal['year'] >= 2009) & (signal['year'] <= 2014)]

def z(s):
    return (s - s.mean()) / s.std()

signal['z_product'] = z(signal['coal_max']) * z(signal['uk_crash_avg_casualties'])

# ---- Humidity (weather archive 10) ----
weather = pd.read_parquet('data/archive (10)/all_weather_data.parquet')
weather['year'] = pd.to_datetime(weather['date'], errors='coerce').dt.year
hum_col = [c for c in weather.columns if 'humidity' in c.lower()][0]
humidity = weather.groupby('year')[hum_col].mean().reset_index()
humidity.columns = ['year', 'humidity']

# ---- Wheat futures ----
wheat = yf.download('ZW=F', start='2008-01-01', end='2015-01-01', progress=False, auto_adjust=True)
wheat['year'] = wheat.index.year
wheat_yr = wheat.groupby('year')['Close'].mean().reset_index()
wheat_yr.columns = ['year', 'wheat_price']

# ---- Merge all ----
merged = signal[['year', 'z_product']].merge(humidity, on='year').merge(wheat_yr, on='year')
merged = merged.sort_values('year')

print(merged.to_string(index=False))

r_signal_humidity = merged['z_product'].corr(merged['humidity'])
r_signal_wheat = merged['z_product'].corr(merged['wheat_price'])
r_humidity_wheat = merged['humidity'].corr(merged['wheat_price'])

print(f"\nSignal -> Humidity:  r = {r_signal_humidity:.4f}")
print(f"Signal -> Wheat:    r = {r_signal_wheat:.4f}")
print(f"Humidity -> Wheat:  r = {r_humidity_wheat:.4f}")

# ---- Normalise ----
def norm(s):
    return (s - s.min()) / (s.max() - s.min())

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(merged['year'], norm(merged['z_product']), 'D-', color='#8172B2', lw=2.5, ms=9,
        label=f'Z-Signal (Coal x Crash Casualties)')
ax.plot(merged['year'], norm(merged['humidity']), 's-', color='#4C72B0', lw=2, ms=8,
        label=f'UK Humidity')
ax.plot(merged['year'], norm(merged['wheat_price']), '^-', color='#C44E52', lw=2, ms=8,
        label=f'Wheat Futures (ZW=F)')

ax.fill_between(merged['year'], norm(merged['z_product']), alpha=0.08, color='#8172B2')

ax.set_title('Coal x UK Crash Casualties  →  Humidity  →  Wheat Futures\n'
             f'Signal→Humidity r={r_signal_humidity:.2f}  |  Signal→Wheat r={r_signal_wheat:.2f}  |  Humidity→Wheat r={r_humidity_wheat:.2f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Normalised Value (0-1)', fontsize=12)
ax.set_xticks(range(2009, 2015))
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/wheat_signal.png', dpi=150, bbox_inches='tight')
print("\nSaved to plots/wheat_signal.png")
