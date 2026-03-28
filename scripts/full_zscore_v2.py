"""
Main correlation scanner.
Loads coal, UK crash, and weather data, then finds the strongest
z-score product correlations across 6-year sliding windows.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---- UK Road Casualties 2005-2015 ----
acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['year'] = pd.to_datetime(acc0515['Date'], errors='coerce').dt.year
crashes = acc0515.groupby('year').agg(
    uk_crash_count=('Accident_Index', 'count'),
    uk_crash_avg_casualties=('Number_of_Casualties', 'mean'),
).reset_index()

# ---- Coal (1981-2021) ----
coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
a_coal = coal.groupby('Year').agg(
    coal_total=('Value (Million Tonnes)', 'sum'),
    coal_max=('Value (Million Tonnes)', 'max'),
).reset_index()
a_coal.columns = ['year', 'coal_total', 'coal_max']

# ---- Weather / Humidity (2009-2024) ----
weather = pd.read_parquet('data/archive (10)/all_weather_data.parquet')
weather['year'] = pd.to_datetime(weather['date'], errors='coerce').dt.year
wcols = {}
for c in weather.columns:
    if 'humidity' in c.lower(): wcols[c] = 'weather_humidity'
    elif 'min_temp' in c.lower(): wcols[c] = 'weather_min_temp'
    elif 'max_temp' in c.lower(): wcols[c] = 'weather_max_temp'
    elif 'rain' in c.lower(): wcols[c] = 'weather_rain'
    elif 'wind_speed' in c.lower(): wcols[c] = 'weather_wind'
weather = weather.rename(columns=wcols)
avail = [c for c in wcols.values() if c in weather.columns]
a_weather = weather.groupby('year').agg({c: 'mean' for c in avail}).reset_index()

print(f"UK Crashes: {int(crashes['year'].min())}-{int(crashes['year'].max())} ({len(crashes)} years)")
print(f"Coal:       {int(a_coal['year'].min())}-{int(a_coal['year'].max())} ({len(a_coal)} years)")
print(f"Weather:    {int(a_weather['year'].min())}-{int(a_weather['year'].max())} ({len(a_weather)} years)")

# ---- Z-score product scan across 6-year windows ----
def z(s):
    return (s - s.mean()) / s.std()

archives = {'UK_Crashes': crashes, 'Coal': a_coal, 'Weather': a_weather}
dataset_tags = {'uk_crash': 'UK_Crashes', 'coal_': 'Coal', 'weather_': 'Weather'}

def get_ds(col):
    for prefix, ds in dataset_tags.items():
        if col.startswith(prefix):
            return ds
    return 'unknown'

merged = crashes.merge(a_coal, on='year').merge(a_weather, on='year')
numeric_cols = [c for c in merged.columns if c != 'year']

print(f"\n{'='*100}")
print("Z-SCORE PRODUCTS (A * B) -> C  across 6-year sliding windows")
print(f"{'='*100}")

results = []
for start in range(2005, 2016):
    end = start + 5
    w = merged[(merged['year'] >= start) & (merged['year'] <= end)].dropna()
    if len(w) < 6:
        continue
    for ca in numeric_cols:
        for cb in numeric_cols:
            ds_a, ds_b = get_ds(ca), get_ds(cb)
            if ds_a >= ds_b or ds_a == 'unknown' or ds_b == 'unknown':
                continue
            product = z(w[ca]) * z(w[cb])
            for cc in numeric_cols:
                ds_c = get_ds(cc)
                if ds_c in (ds_a, ds_b) or ds_c == 'unknown':
                    continue
                r = product.corr(w[cc])
                if not np.isnan(r) and abs(r) > 0.9:
                    results.append((start, end, ds_a, ca, ds_b, cb, ds_c, cc, r, len(w)))

results.sort(key=lambda x: abs(x[8]), reverse=True)

seen = set()
shown = 0
for start, end, da, ca, db, cb, dc, cc, r, n in results:
    key = (ca, cb, cc)
    if key in seen:
        continue
    seen.add(key)
    shown += 1
    print(f"  {shown:>3}  r={r:>8.4f}  {start}-{end}  {ca} x {cb} -> {cc}")
    if shown >= 15:
        break
