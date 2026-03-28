import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Coal (yearly, spread to quarterly) ----
coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
coal_yr = coal.groupby('Year')['Value (Million Tonnes)'].max().reset_index()
coal_yr.columns = ['year', 'coal_max']

# ---- UK Crash casualties (quarterly) ----
acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['dt'] = pd.to_datetime(acc0515['Date'], errors='coerce')
acc0515['quarter'] = acc0515['dt'].dt.to_period('Q')
crashes_q = acc0515.groupby('quarter')['Number_of_Casualties'].mean().reset_index()
crashes_q['date'] = crashes_q['quarter'].dt.to_timestamp()
crashes_q['year'] = crashes_q['date'].dt.year

# Join coal as constant per year
crashes_q = crashes_q.merge(coal_yr, on='year')

# ---- Humidity (quarterly) ----
weather = pd.read_parquet('data/archive (10)/all_weather_data.parquet')
weather['dt'] = pd.to_datetime(weather['date'], errors='coerce')
weather['quarter'] = weather['dt'].dt.to_period('Q')
hum_col = [c for c in weather.columns if 'humidity' in c.lower()][0]
humidity_q = weather.groupby('quarter')[hum_col].mean().reset_index()
humidity_q.columns = ['quarter', 'humidity']
humidity_q['date'] = humidity_q['quarter'].dt.to_timestamp()

# ---- Wheat futures (quarterly) ----
wheat = yf.download('ZW=F', start='2008-01-01', end='2015-01-01', progress=False, auto_adjust=True)
wheat['quarter'] = wheat.index.to_period('Q')
wheat_q = wheat.groupby('quarter')['Close'].mean().reset_index()
wheat_q.columns = ['quarter', 'wheat_price']
wheat_q['date'] = wheat_q['quarter'].dt.to_timestamp()

# ---- Merge on quarter ----
merged = crashes_q[['quarter', 'date', 'Number_of_Casualties', 'coal_max']].copy()
merged.columns = ['quarter', 'date', 'crash_casualties', 'coal_max']
merged = merged.merge(humidity_q[['quarter', 'humidity']], on='quarter')
merged = merged.merge(wheat_q[['quarter', 'wheat_price']], on='quarter')
merged = merged[(merged['date'] >= '2009-01-01') & (merged['date'] <= '2014-12-31')]
merged = merged.sort_values('date')

print(f"Quarterly data points: {len(merged)}")

# Z-scores
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
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(merged['date'], norm(merged['z_product']), 'D-', color='#8172B2', lw=2, ms=6,
        label='Z-Signal (Coal x Crash Casualties)')
ax.plot(merged['date'], norm(merged['humidity']), 's-', color='#4C72B0', lw=1.5, ms=5,
        label='UK Humidity')
ax.plot(merged['date'], norm(merged['wheat_price']), '^-', color='#C44E52', lw=1.5, ms=5,
        label='Wheat Futures (ZW=F)')

ax.set_title('Coal x UK Crash Casualties  →  Humidity  →  Wheat Futures (Quarterly)\n'
             f'Signal→Humidity r={r_sig_hum:.2f}  |  Signal→Wheat r={r_sig_wht:.2f}  |  Humidity→Wheat r={r_hum_wht:.2f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Quarter', fontsize=12)
ax.set_ylabel('Normalised Value (0-1)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/wheat_signal_quarterly.png', dpi=150, bbox_inches='tight')
print("Saved to plots/wheat_signal_quarterly.png")
