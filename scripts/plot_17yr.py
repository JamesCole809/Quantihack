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

# ---- Load Coal (1981-2021) ----
coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
coal_yr = coal.groupby('Year').agg(
    coal_total=('Value (Million Tonnes)', 'sum'),
    coal_max=('Value (Million Tonnes)', 'max'),
).reset_index()
coal_yr.columns = ['year', 'coal_total', 'coal_max']

# ---- Load UK Weather (1961-2024) ----
ukw = pd.read_csv('data/uk_weather_1961_2024/land_uk_daily_regions.csv')
ukw_yr = ukw.groupby('year').agg(
    ukw_temp=('temp', 'mean'),
    ukw_wind=('wind_speed', 'mean'),
    ukw_precip=('precipitation', 'mean'),
    ukw_dewpoint=('dewpoint_temp', 'mean'),
).reset_index()

# ---- Merge on 2005-2021 ----
merged = crashes.merge(coal_yr, on='year').merge(ukw_yr, on='year')
merged = merged[(merged['year'] >= 2005) & (merged['year'] <= 2021)].reset_index(drop=True)
print(f"Data: {merged['year'].min()}-{merged['year'].max()} ({len(merged)} years)")

# Z-scores
def z(s):
    return (s - s.mean()) / s.std()

def norm(s):
    return (s - s.min()) / (s.max() - s.min())

# Find best z-score product -> third variable
cols_by_ds = {
    'UK_Crashes': ['uk_crash_count', 'uk_crash_avg_vehicles', 'uk_crash_avg_casualties'],
    'Coal': ['coal_total', 'coal_max'],
    'UK_Weather': ['ukw_temp', 'ukw_wind', 'ukw_precip', 'ukw_dewpoint'],
}

results = []
for ds_a, cols_a in cols_by_ds.items():
    for ds_b, cols_b in cols_by_ds.items():
        if ds_a >= ds_b:
            continue
        for ca in cols_a:
            for cb in cols_b:
                product = z(merged[ca]) * z(merged[cb])
                for ds_c, cols_c in cols_by_ds.items():
                    if ds_c in (ds_a, ds_b):
                        continue
                    for cc in cols_c:
                        r = product.corr(merged[cc])
                        if not np.isnan(r):
                            results.append((ds_a, ca, ds_b, cb, ds_c, cc, r))

results.sort(key=lambda x: abs(x[6]), reverse=True)

print("\nTop z-score products:")
for i, (da, ca, db, cb, dc, cc, r) in enumerate(results[:10], 1):
    print(f"  {i}. {ca} x {cb} -> {cc}: r={r:.4f}")

# Plot top 3
fig, axes = plt.subplots(3, 1, figsize=(13, 14))
fig.suptitle('Z-Score Product Correlations (2005-2021, 17 years)', fontsize=16, fontweight='bold')

for idx, ax in enumerate(axes):
    da, ca, db, cb, dc, cc, r = results[idx]
    za = z(merged[ca])
    zb = z(merged[cb])
    zp = za * zb

    ax.plot(merged['year'], norm(zp), 'D-', color='#8172B2', lw=2.5, ms=7, label=f'Z-Product ({ca} x {cb})')
    ax.plot(merged['year'], norm(merged[cc]), '^-', color='#C44E52', lw=2, ms=7, label=cc)
    ax.set_title(f'{ca} x {cb} vs {cc} (r = {r:.4f})', fontsize=11)
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalised')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(2005, 2022))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/zscore_17yr.png', dpi=150, bbox_inches='tight')
print("\nSaved to plots/zscore_17yr.png")
