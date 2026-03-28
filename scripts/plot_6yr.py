import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Load UK Crashes (2005-2022) ----
acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['year'] = pd.to_datetime(acc0515['Date'], errors='coerce').dt.year
rc1 = acc0515.groupby('year').agg(
    count=('Accident_Index', 'count'),
    avg_vehicles=('Number_of_Vehicles', 'mean'),
    avg_casualties=('Number_of_Casualties', 'mean'),
).reset_index()

acc1620 = pd.read_parquet('data/road-casualty-data/dft-road-casualty-statistics-accident-last-5-years.parquet')
rc2 = acc1620.groupby('accident_year').agg(
    count=('accident_index', 'count'),
    avg_vehicles=('number_of_vehicles', 'mean'),
    avg_casualties=('number_of_casualties', 'mean'),
).reset_index()
rc2.columns = ['year', 'count', 'avg_vehicles', 'avg_casualties']

crashes = pd.concat([rc1, rc2]).drop_duplicates(subset='year', keep='first').sort_values('year')
crashes.columns = ['year', 'uk_crash_count', 'uk_crash_avg_vehicles', 'uk_crash_avg_casualties']

# ---- Load Coal ----
coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
coal_yr = coal.groupby('Year').agg(
    coal_total=('Value (Million Tonnes)', 'sum'),
    coal_max=('Value (Million Tonnes)', 'max'),
).reset_index()
coal_yr.columns = ['year', 'coal_total', 'coal_max']

# ---- Load UK Weather ----
ukw = pd.read_csv('data/uk_weather_1961_2024/land_uk_daily_regions.csv')
ukw_yr = ukw.groupby('year').agg(
    ukw_temp=('temp', 'mean'),
    ukw_wind=('wind_speed', 'mean'),
    ukw_precip=('precipitation', 'mean'),
    ukw_dewpoint=('dewpoint_temp', 'mean'),
).reset_index()

# ---- Load Asylum ----
asilo_ca = pd.read_csv('data/asylum_spain/AsiloCA.csv', encoding='latin1')
yr_col = [c for c in asilo_ca.columns if 'o' in c.lower()][-1]
asilo_es = pd.read_csv('data/asylum_spain/AsiloEspaa.csv', encoding='latin1')
yr_col2 = [c for c in asilo_es.columns if 'o' in c.lower()][-1]
a3a = asilo_ca.groupby(yr_col).agg(asylum_total=('Solicitantes', 'sum')).reset_index()
a3a.columns = ['year', 'asylum_total']
a3b = asilo_es.groupby(yr_col2).agg(asylum_nat_total=('Total', 'sum'), asylum_admitted=('Admitidas', 'sum')).reset_index()
a3b.columns = ['year', 'asylum_nat_total', 'asylum_admitted']
asylum = a3a.merge(a3b, on='year', how='outer')

# ---- Load UK Rainfall ----
rain = pd.read_csv('data/uk_rainfall_2018_2023/Uk_rainfall_data.csv')
rain['year'] = rain['Period'].str[:4].astype(int)
rain_yr = rain.groupby('year').agg(
    rain_avg=('Avg rainfall(in mm)', 'mean'),
    rain_temp=('Avg temp(in centigrade)', 'mean'),
).reset_index()

# ---- Load Weather (archive 10) ----
weather = pd.read_parquet('data/archive (10)/all_weather_data.parquet')
weather['year'] = pd.to_datetime(weather['date'], errors='coerce').dt.year
wcols = {}
for c in weather.columns:
    if 'min_temp' in c.lower(): wcols[c] = 'w10_min_temp'
    elif 'max_temp' in c.lower(): wcols[c] = 'w10_max_temp'
    elif 'rain' in c.lower(): wcols[c] = 'w10_rain'
    elif 'humidity' in c.lower(): wcols[c] = 'w10_humidity'
    elif 'wind_speed' in c.lower(): wcols[c] = 'w10_wind'
weather = weather.rename(columns=wcols)
avail = [c for c in wcols.values() if c in weather.columns]
w10_yr = weather.groupby('year').agg({c: 'mean' for c in avail}).reset_index()

# ---- Merge everything ----
merged = crashes.copy()
for df in [coal_yr, ukw_yr, asylum, rain_yr, w10_yr]:
    merged = merged.merge(df, on='year', how='outer')
merged = merged.sort_values('year').reset_index(drop=True)

# ---- Try all 6-year windows ----
cols_by_ds = {
    'UK_Crashes': ['uk_crash_count', 'uk_crash_avg_vehicles', 'uk_crash_avg_casualties'],
    'Coal': ['coal_total', 'coal_max'],
    'UK_Weather': ['ukw_temp', 'ukw_wind', 'ukw_precip', 'ukw_dewpoint'],
    'Asylum': ['asylum_total', 'asylum_nat_total', 'asylum_admitted'],
    'UK_Rainfall': ['rain_avg', 'rain_temp'],
    'Weather10': ['w10_min_temp', 'w10_max_temp', 'w10_rain', 'w10_humidity', 'w10_wind'],
}

def z(s):
    return (s - s.mean()) / s.std()

def norm(s):
    mn, mx = s.min(), s.max()
    if mx == mn: return s * 0
    return (s - mn) / (mx - mn)

all_results = []

# Slide 6-year windows
min_yr = int(merged['year'].min())
max_yr = int(merged['year'].max())

for start in range(min_yr, max_yr - 4):
    end = start + 5
    window = merged[(merged['year'] >= start) & (merged['year'] <= end)].copy()
    if len(window) < 6:
        continue

    for ds_a, cols_a in cols_by_ds.items():
        for ds_b, cols_b in cols_by_ds.items():
            if ds_a >= ds_b:
                continue
            for ca in cols_a:
                for cb in cols_b:
                    sa = window[ca].dropna()
                    sb = window[cb].dropna()
                    common = sa.index.intersection(sb.index)
                    if len(common) < 6:
                        continue
                    product = z(window.loc[common, ca]) * z(window.loc[common, cb])

                    for ds_c, cols_c in cols_by_ds.items():
                        if ds_c in (ds_a, ds_b):
                            continue
                        for cc in cols_c:
                            sc = window.loc[common, cc].dropna()
                            common2 = product.index.intersection(sc.index)
                            if len(common2) < 6:
                                continue
                            r = product.loc[common2].corr(window.loc[common2, cc])
                            if not np.isnan(r) and abs(r) > 0.9:
                                all_results.append((start, end, ds_a, ca, ds_b, cb, ds_c, cc, r, len(common2)))

all_results.sort(key=lambda x: abs(x[8]), reverse=True)

print(f"Found {len(all_results)} z-score triples with |r| > 0.9 across 6-year windows\n")
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'Window':<12} {'A':<14} {'A col':<26} {'B':<14} {'B col':<18} {'C':<14} {'C col'}")
print("-" * 140)
seen = set()
shown = 0
for start, end, da, ca, db, cb, dc, cc, r, n in all_results:
    key = (ca, cb, cc)
    if key in seen:
        continue
    seen.add(key)
    shown += 1
    if shown > 20:
        break
    print(f"{shown:>4}  {r:>8.4f}  {n:>3}  {start}-{end:<7} {da:<14} {ca:<26} {db:<14} {cb:<18} {dc:<14} {cc}")

# Plot top 3 unique
fig, axes = plt.subplots(3, 1, figsize=(13, 14))
fig.suptitle('Best Z-Score Product Correlations (6-year windows)', fontsize=16, fontweight='bold')

plotted = 0
seen_plot = set()
for start, end, da, ca, db, cb, dc, cc, r, n in all_results:
    key = (ca, cb, cc)
    if key in seen_plot:
        continue
    seen_plot.add(key)

    window = merged[(merged['year'] >= start) & (merged['year'] <= end)].copy().dropna(subset=[ca, cb, cc])
    if len(window) < 6:
        continue

    ax = axes[plotted]
    zp = z(window[ca]) * z(window[cb])

    ax.plot(window['year'], norm(zp), 'D-', color='#8172B2', lw=2.5, ms=8, label=f'Z-Product ({ca} x {cb})')
    ax.plot(window['year'], norm(window[cc]), '^-', color='#C44E52', lw=2, ms=8, label=cc)
    ax.set_title(f'{ca} x {cb} vs {cc} ({start}-{end}, r = {r:.4f})', fontsize=11)
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalised')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(start, end + 1))

    plotted += 1
    if plotted >= 3:
        break

plt.tight_layout()
plt.savefig('plots/zscore_6yr.png', dpi=150, bbox_inches='tight')
print("\nSaved to plots/zscore_6yr.png")
