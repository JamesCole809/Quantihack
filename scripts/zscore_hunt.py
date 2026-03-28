"""
Z-score product, ratio, and squared search.
Tests all combinations of coal, UK crashes, and weather variables.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['year'] = pd.to_datetime(acc0515['Date'], errors='coerce').dt.year
crashes = acc0515.groupby('year').agg(
    uk_crash_count=('Accident_Index', 'count'),
    uk_crash_avg_casualties=('Number_of_Casualties', 'mean'),
).reset_index()

coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
a_coal = coal.groupby('Year').agg(
    coal_total=('Value (Million Tonnes)', 'sum'),
    coal_max=('Value (Million Tonnes)', 'max'),
).reset_index()
a_coal.columns = ['year', 'coal_total', 'coal_max']

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

merged = crashes.merge(a_coal, on='year').merge(a_weather, on='year').sort_values('year')

dataset_tags = {'uk_crash': 'UK_Crashes', 'coal_': 'Coal', 'weather_': 'Weather'}
def get_ds(col):
    for prefix, ds in dataset_tags.items():
        if col.startswith(prefix): return ds
    return 'unknown'

def z(s):
    return (s - s.mean()) / s.std()

numeric_cols = [c for c in merged.columns if c != 'year']

print("Z-SCORE PRODUCTS (A * B) -> C  (6-year windows, |r| > 0.9)")
print("=" * 80)

results = []
for start in range(2005, 2016):
    end = start + 5
    w = merged[(merged['year'] >= start) & (merged['year'] <= end)].dropna()
    if len(w) < 6: continue
    for ca in numeric_cols:
        for cb in numeric_cols:
            da, db = get_ds(ca), get_ds(cb)
            if da >= db or da == 'unknown' or db == 'unknown': continue
            product = z(w[ca]) * z(w[cb])
            for cc in numeric_cols:
                dc = get_ds(cc)
                if dc in (da, db) or dc == 'unknown': continue
                r = product.corr(w[cc])
                if not np.isnan(r) and abs(r) > 0.9:
                    results.append((start, end, ca, cb, cc, r))

results.sort(key=lambda x: abs(x[5]), reverse=True)
seen = set()
for start, end, ca, cb, cc, r in results:
    key = (ca, cb, cc)
    if key in seen: continue
    seen.add(key)
    print(f"  r={r:>8.4f}  {start}-{end}  z({ca}) * z({cb}) -> {cc}")
    if len(seen) >= 15: break
