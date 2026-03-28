"""
6-year sliding window z-score analysis.
Finds and plots the top z-score product correlations using coal,
UK crash, and weather data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Load data ----
acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['year'] = pd.to_datetime(acc0515['Date'], errors='coerce').dt.year
crashes = acc0515.groupby('year').agg(
    uk_crash_count=('Accident_Index', 'count'),
    uk_crash_avg_casualties=('Number_of_Casualties', 'mean'),
).reset_index()

coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
coal_yr = coal.groupby('Year').agg(
    coal_total=('Value (Million Tonnes)', 'sum'),
    coal_max=('Value (Million Tonnes)', 'max'),
).reset_index()
coal_yr.columns = ['year', 'coal_total', 'coal_max']

weather = pd.read_parquet('data/archive (10)/all_weather_data.parquet')
weather['year'] = pd.to_datetime(weather['date'], errors='coerce').dt.year
wcols = {}
for c in weather.columns:
    if 'humidity' in c.lower(): wcols[c] = 'w10_humidity'
    elif 'min_temp' in c.lower(): wcols[c] = 'w10_min_temp'
    elif 'max_temp' in c.lower(): wcols[c] = 'w10_max_temp'
    elif 'rain' in c.lower(): wcols[c] = 'w10_rain'
    elif 'wind_speed' in c.lower(): wcols[c] = 'w10_wind'
weather = weather.rename(columns=wcols)
avail = [c for c in wcols.values() if c in weather.columns]
w10_yr = weather.groupby('year').agg({c: 'mean' for c in avail}).reset_index()

merged = crashes.merge(coal_yr, on='year').merge(w10_yr, on='year', how='outer')
merged = merged.sort_values('year').reset_index(drop=True)

cols_by_ds = {
    'UK_Crashes': ['uk_crash_count', 'uk_crash_avg_casualties'],
    'Coal': ['coal_total', 'coal_max'],
    'Weather': [c for c in avail],
}

def z(s):
    return (s - s.mean()) / s.std()

def norm(s):
    mn, mx = s.min(), s.max()
    if mx == mn: return s * 0
    return (s - mn) / (mx - mn)

# ---- Scan 6-year windows ----
all_results = []
min_yr = int(merged['year'].min())
max_yr = int(merged['year'].max())

for start in range(min_yr, max_yr - 4):
    end = start + 5
    window = merged[(merged['year'] >= start) & (merged['year'] <= end)].copy()
    if len(window) < 6:
        continue
    for ds_a, cols_a in cols_by_ds.items():
        for ds_b, cols_b in cols_by_ds.items():
            if ds_a >= ds_b: continue
            for ca in cols_a:
                for cb in cols_b:
                    common = window[[ca, cb]].dropna().index
                    if len(common) < 6: continue
                    product = z(window.loc[common, ca]) * z(window.loc[common, cb])
                    for ds_c, cols_c in cols_by_ds.items():
                        if ds_c in (ds_a, ds_b): continue
                        for cc in cols_c:
                            common2 = product.index.intersection(window[cc].dropna().index)
                            if len(common2) < 6: continue
                            r = product.loc[common2].corr(window.loc[common2, cc])
                            if not np.isnan(r) and abs(r) > 0.9:
                                all_results.append((start, end, ds_a, ca, ds_b, cb, ds_c, cc, r, len(common2)))

all_results.sort(key=lambda x: abs(x[8]), reverse=True)

print(f"Found {len(all_results)} triples with |r| > 0.9\n")
seen = set()
shown = 0
for start, end, da, ca, db, cb, dc, cc, r, n in all_results:
    key = (ca, cb, cc)
    if key in seen: continue
    seen.add(key)
    shown += 1
    if shown > 15: break
    print(f"  {shown:>3}  r={r:>8.4f}  {start}-{end}  {ca} x {cb} -> {cc}")

# ---- Plot top 3 ----
fig, axes = plt.subplots(3, 1, figsize=(13, 14))
fig.suptitle('Best Z-Score Product Correlations (6-year windows)', fontsize=16, fontweight='bold')

plotted = 0
seen_plot = set()
for start, end, da, ca, db, cb, dc, cc, r, n in all_results:
    key = (ca, cb, cc)
    if key in seen_plot: continue
    seen_plot.add(key)
    window = merged[(merged['year'] >= start) & (merged['year'] <= end)].dropna(subset=[ca, cb, cc])
    if len(window) < 6: continue
    ax = axes[plotted]
    zp = z(window[ca]) * z(window[cb])
    ax.plot(window['year'], norm(zp), 'D-', color='#8172B2', lw=2.5, ms=8, label=f'Z-Product ({ca} x {cb})')
    ax.plot(window['year'], norm(window[cc]), '^-', color='#C44E52', lw=2, ms=8, label=cc)
    ax.set_title(f'{ca} x {cb} vs {cc} ({start}-{end}, r = {r:.4f})', fontsize=11)
    ax.set_xlabel('Year'); ax.set_ylabel('Normalised')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xticks(range(start, end + 1))
    plotted += 1
    if plotted >= 3: break

plt.tight_layout()
plt.savefig('plots/zscore_6yr.png', dpi=150, bbox_inches='tight')
print("\nSaved to plots/zscore_6yr.png")
