import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Coal (yearly, spread to half-year) ----
coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
coal_yr = coal.groupby('Year')['Value (Million Tonnes)'].max().reset_index()
coal_yr.columns = ['year', 'coal_max']

# ---- UK Crash casualties (bi-annual) ----
acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['dt'] = pd.to_datetime(acc0515['Date'], errors='coerce')
acc0515['year'] = acc0515['dt'].dt.year
acc0515['half'] = (acc0515['dt'].dt.month > 6).astype(int)  # 0=H1, 1=H2
crashes_h = acc0515.groupby(['year', 'half'])['Number_of_Casualties'].mean().reset_index()
crashes_h['year'] = crashes_h['year'].astype(int)
crashes_h['month'] = crashes_h['half'] * 6 + 1
crashes_h['date'] = pd.to_datetime(crashes_h[['year']].assign(month=crashes_h['month'], day=1))
crashes_h = crashes_h.merge(coal_yr, on='year')

# ---- Humidity (bi-annual) ----
weather = pd.read_parquet('data/archive (10)/all_weather_data.parquet')
weather['dt'] = pd.to_datetime(weather['date'], errors='coerce')
weather['year'] = weather['dt'].dt.year
weather['half'] = (weather['dt'].dt.month > 6).astype(int)
hum_col = [c for c in weather.columns if 'humidity' in c.lower()][0]
humidity_h = weather.groupby(['year', 'half'])[hum_col].mean().reset_index()
humidity_h.columns = ['year', 'half', 'humidity']

# ---- Wheat futures (bi-annual) ----
wheat = yf.download('ZW=F', start='2008-01-01', end='2015-01-01', progress=False, auto_adjust=True)
wheat['year'] = wheat.index.year
wheat['half'] = (wheat.index.month > 6).astype(int)
wheat_h = wheat.groupby(['year', 'half'])['Close'].mean().reset_index()
wheat_h.columns = ['year', 'half', 'wheat_price']

# ---- Merge ----
merged = crashes_h[['year', 'half', 'date', 'Number_of_Casualties', 'coal_max']].copy()
merged.columns = ['year', 'half', 'date', 'crash_casualties', 'coal_max']
merged = merged.merge(humidity_h, on=['year', 'half'])
merged = merged.merge(wheat_h, on=['year', 'half'])
merged = merged[(merged['date'] >= '2009-01-01') & (merged['date'] <= '2014-12-31')]
merged = merged.sort_values('date')

print(f"Bi-annual data points: {len(merged)}")

def z(s):
    return (s - s.mean()) / s.std()

def norm(s):
    return (s - s.min()) / (s.max() - s.min())

merged['z_product'] = z(merged['coal_max']) * z(merged['crash_casualties'])

r_sig_hum = merged['z_product'].corr(merged['humidity'])
r_sig_wht = merged['z_product'].corr(merged['wheat_price'])
r_hum_wht = merged['humidity'].corr(merged['wheat_price'])

print(f"Signal -> Humidity:  r = {r_sig_hum:.4f}")
print(f"Signal -> Wheat:    r = {r_sig_wht:.4f}")
print(f"Humidity -> Wheat:  r = {r_hum_wht:.4f}")

# ---- Plot ----
labels = [f"{'H1' if h==0 else 'H2'} {y}" for y, h in zip(merged['year'], merged['half'])]

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(range(len(merged)), norm(merged['z_product']), 'D-', color='#8172B2', lw=2.5, ms=8,
        label='Z-Signal (Coal x Crash Casualties)')
ax.plot(range(len(merged)), norm(merged['humidity']), 's-', color='#4C72B0', lw=2, ms=7,
        label='UK Humidity')
ax.plot(range(len(merged)), norm(merged['wheat_price']), '^-', color='#C44E52', lw=2, ms=7,
        label='Wheat Futures (ZW=F)')

ax.set_title('Coal x UK Crash Casualties  →  Humidity  →  Wheat Futures (Bi-Annual)\n'
             f'Signal→Humidity r={r_sig_hum:.2f}  |  Signal→Wheat r={r_sig_wht:.2f}  |  Humidity→Wheat r={r_hum_wht:.2f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Normalised Value (0-1)', fontsize=12)
ax.set_xticks(range(len(merged)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/wheat_signal_biannual.png', dpi=150, bbox_inches='tight')
print("Saved to plots/wheat_signal_biannual.png")
