import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")

# ---- UK Road Casualties 2005-2015 ----
acc0515 = pd.read_parquet('data/road-casualty-data/Accidents0515.parquet')
acc0515['year'] = pd.to_datetime(acc0515['Date'], errors='coerce').dt.year
a_rc1 = acc0515.groupby('year').agg(
    rc1_count=('Accident_Index', 'count'),
    rc1_avg_vehicles=('Number_of_Vehicles', 'mean'),
    rc1_avg_casualties=('Number_of_Casualties', 'mean'),
    rc1_avg_severity=('Accident_Severity', 'mean'),
).reset_index()

# ---- UK Road Casualties 2016-2020 ----
acc1620 = pd.read_parquet('data/road-casualty-data/dft-road-casualty-statistics-accident-last-5-years.parquet')
a_rc2 = acc1620.groupby('accident_year').agg(
    rc2_count=('accident_index', 'count'),
    rc2_avg_vehicles=('number_of_vehicles', 'mean'),
    rc2_avg_casualties=('number_of_casualties', 'mean'),
    rc2_avg_severity=('accident_severity', 'mean'),
).reset_index()
a_rc2.columns = ['year'] + list(a_rc2.columns[1:])

# Combine UK accidents into one long series
combined_acc = pd.DataFrame()
for yr in range(2005, 2021):
    if yr <= 2015:
        row = a_rc1[a_rc1['year'] == yr]
        if len(row):
            combined_acc = pd.concat([combined_acc, pd.DataFrame({
                'year': [yr], 'uk_crash_count': [row['rc1_count'].values[0]],
                'uk_crash_avg_vehicles': [row['rc1_avg_vehicles'].values[0]],
                'uk_crash_avg_casualties': [row['rc1_avg_casualties'].values[0]],
            })])
    else:
        row = a_rc2[a_rc2['year'] == yr]
        if len(row):
            combined_acc = pd.concat([combined_acc, pd.DataFrame({
                'year': [yr], 'uk_crash_count': [row['rc2_count'].values[0]],
                'uk_crash_avg_vehicles': [row['rc2_avg_vehicles'].values[0]],
                'uk_crash_avg_casualties': [row['rc2_avg_casualties'].values[0]],
            })])

# Add UK accidents 2019
uk19 = pd.read_csv('data/uk_accidents_2019/accident data.csv', usecols=['Index','Number_of_Vehicles','Number_of_Casualties'])
combined_acc = pd.concat([combined_acc, pd.DataFrame({
    'year': [2019], 'uk_crash_count': [len(uk19)],
    'uk_crash_avg_vehicles': [uk19['Number_of_Vehicles'].mean()],
    'uk_crash_avg_casualties': [uk19['Number_of_Casualties'].mean()],
})])

# Add UK accidents 2021-2022
uk2122 = pd.read_csv('data/uk_accidents_2021_2022/Road Accident Data.csv', usecols=['Accident Date','Number_of_Vehicles','Number_of_Casualties'])
uk2122['year'] = pd.to_datetime(uk2122['Accident Date'], format='%d-%m-%Y', errors='coerce').dt.year
for yr in [2021, 2022]:
    sub = uk2122[uk2122['year'] == yr]
    if len(sub):
        combined_acc = pd.concat([combined_acc, pd.DataFrame({
            'year': [yr], 'uk_crash_count': [len(sub)],
            'uk_crash_avg_vehicles': [sub['Number_of_Vehicles'].mean()],
            'uk_crash_avg_casualties': [sub['Number_of_Casualties'].mean()],
        })])

combined_acc = combined_acc.drop_duplicates(subset='year', keep='first').sort_values('year').reset_index(drop=True)
print(f"UK Crashes: {int(combined_acc['year'].min())}-{int(combined_acc['year'].max())} ({len(combined_acc)} years)")

# ---- Weather (archive 10) 2009-2024 ----
weather = pd.read_parquet('data/archive (10)/all_weather_data.parquet')
weather['year'] = pd.to_datetime(weather['date'], errors='coerce').dt.year
wcols = {'min_temp \xb0c': 'weather_min_temp', 'max_temp \xb0c': 'weather_max_temp',
         'rain mm': 'weather_rain', 'humidity %': 'weather_humidity',
         'wind_speed km/h': 'weather_wind'}
rename_map = {}
for orig, new in wcols.items():
    matches = [c for c in weather.columns if orig in c.lower() or c == orig]
    if matches:
        rename_map[matches[0]] = new
weather = weather.rename(columns=rename_map)
available = [c for c in wcols.values() if c in weather.columns]
a_weather = weather.groupby('year').agg({c: 'mean' for c in available}).reset_index()
print(f"Weather: {int(a_weather['year'].min())}-{int(a_weather['year'].max())} ({len(a_weather)} years)")

# ---- UK Weather 1961-2024 ----
ukw = pd.read_csv('data/uk_weather_1961_2024/land_uk_daily_regions.csv')
a_ukw = ukw.groupby('year').agg(
    ukw_avg_temp=('temp', 'mean'),
    ukw_avg_dewpoint=('dewpoint_temp', 'mean'),
    ukw_avg_wind=('wind_speed', 'mean'),
    ukw_avg_precip=('precipitation', 'mean'),
).reset_index()
print(f"UK Weather: {int(a_ukw['year'].min())}-{int(a_ukw['year'].max())} ({len(a_ukw)} years)")

# ---- Coal (1981-2021) ----
coal = pd.read_csv('data/world_coal_production/world-coal-production.csv', sep=';')
a_coal = coal.groupby('Year').agg(
    coal_total=('Value (Million Tonnes)', 'sum'),
    coal_avg=('Value (Million Tonnes)', 'mean'),
    coal_max=('Value (Million Tonnes)', 'max'),
).reset_index()
a_coal.columns = ['year'] + list(a_coal.columns[1:])
print(f"Coal: {int(a_coal['year'].min())}-{int(a_coal['year'].max())} ({len(a_coal)} years)")

# ---- Asylum (2012-2021) ----
asilo_ca = pd.read_csv('data/asylum_spain/AsiloCA.csv', encoding='latin1')
yr_col = [c for c in asilo_ca.columns if 'o' in c.lower()][-1]
asilo_es = pd.read_csv('data/asylum_spain/AsiloEspaa.csv', encoding='latin1')
yr_col2 = [c for c in asilo_es.columns if 'o' in c.lower()][-1]
a3a = asilo_ca.groupby(yr_col).agg(asylum_total=('Solicitantes', 'sum'), asylum_avg=('Solicitantes', 'mean')).reset_index()
a3a.columns = ['year'] + list(a3a.columns[1:])
a3b = asilo_es.groupby(yr_col2).agg(asylum_nat_total=('Total', 'sum'), asylum_admitted=('Admitidas', 'sum')).reset_index()
a3b.columns = ['year'] + list(a3b.columns[1:])
a_asylum = a3a.merge(a3b, on='year', how='outer')
print(f"Asylum: {int(a_asylum['year'].min())}-{int(a_asylum['year'].max())} ({len(a_asylum)} years)")

# ---- UK Rainfall (2018-2023) ----
rain = pd.read_csv('data/uk_rainfall_2018_2023/Uk_rainfall_data.csv')
rain['year'] = rain['Period'].str[:4].astype(int)
a_rain = rain.groupby('year').agg(
    rain_avg_rainfall=('Avg rainfall(in mm)', 'mean'),
    rain_avg_temp=('Avg temp(in centigrade)', 'mean'),
).reset_index()
print(f"UK Rainfall: {int(a_rain['year'].min())}-{int(a_rain['year'].max())} ({len(a_rain)} years)")

# ---- Build archives ----
archives = {
    'UK_Crashes': combined_acc,
    'UK_Weather_Long': a_ukw,
    'Coal': a_coal,
    'Asylum': a_asylum,
    'UK_Rainfall': a_rain,
}
# Only add weather if it has data
if len(available) > 0:
    archives['Weather'] = a_weather

# ---- Merge all on year ----
merged_all = pd.DataFrame({'year': range(1961, 2025)})
for df in archives.values():
    merged_all = merged_all.merge(df, on='year', how='left')

numeric_cols = [c for c in merged_all.columns if c != 'year']
zscores = merged_all[['year']].copy()
for c in numeric_cols:
    s = merged_all[c]
    if s.std() > 0:
        zscores[f'z_{c}'] = (s - s.mean()) / s.std()

dataset_tags = {
    'uk_crash': 'UK_Crashes', 'rc1_': 'UK_Crashes', 'rc2_': 'UK_Crashes',
    'ukw_': 'UK_Weather_Long', 'weather_': 'Weather',
    'coal_': 'Coal', 'asylum_': 'Asylum',
    'rain_': 'UK_Rainfall',
}

def get_ds(col):
    col = col.replace('z_', '')
    for prefix, ds in dataset_tags.items():
        if col.startswith(prefix):
            return ds
    return 'unknown'

z_cols = [c for c in zscores.columns if c.startswith('z_')]

# ---- PART 1: Standard cross-archive correlations ----
print("\n" + "=" * 130)
print("CROSS-ARCHIVE CORRELATIONS (by year, min 5 overlapping years)")
print("=" * 130)

results = []
archive_names = list(archives.keys())
for i in range(len(archive_names)):
    for j in range(i+1, len(archive_names)):
        na, nb = archive_names[i], archive_names[j]
        da, db = archives[na], archives[nb]
        merged = da.merge(db, on='year', how='inner')
        if len(merged) < 5:
            continue
        cols_a = [c for c in da.columns if c != 'year']
        cols_b = [c for c in db.columns if c != 'year']
        for ca in cols_a:
            for cb in cols_b:
                sub = merged[[ca, cb]].dropna()
                if len(sub) < 5:
                    continue
                r = sub[ca].corr(sub[cb])
                if not np.isnan(r):
                    results.append((na, ca, nb, cb, r, len(sub)))

results.sort(key=lambda x: abs(x[4]), reverse=True)
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'Archive A':<16} {'Column A':<26} {'Archive B':<16} {'Column B'}")
print("-" * 130)
for idx, (a, ca, b, cb, r, n) in enumerate(results[:40], 1):
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {a:<16} {ca:<26} {b:<16} {cb}")

# ---- PART 2: Z-score products ----
print("\n" + "=" * 130)
print("Z-SCORE PRODUCTS (A * B) CORRELATED WITH C  (min 5 overlapping years)")
print("=" * 130)

triple_results = []
for i in range(len(z_cols)):
    for j in range(i+1, len(z_cols)):
        za, zb = z_cols[i], z_cols[j]
        ds_a, ds_b = get_ds(za), get_ds(zb)
        if ds_a == ds_b or ds_a == 'unknown' or ds_b == 'unknown':
            continue
        product = zscores[za] * zscores[zb]
        for k in range(len(z_cols)):
            zc = z_cols[k]
            ds_c = get_ds(zc)
            if ds_c in (ds_a, ds_b) or ds_c == 'unknown':
                continue
            mask = product.notna() & zscores[zc].notna()
            if mask.sum() < 5:
                continue
            r = product[mask].corr(zscores[zc][mask])
            if not np.isnan(r) and abs(r) > 0.85:
                triple_results.append((ds_a, za, ds_b, zb, ds_c, zc, r, int(mask.sum())))

triple_results.sort(key=lambda x: abs(x[6]), reverse=True)
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'A':<16} {'A col':<24} {'B':<16} {'B col':<24} {'C':<16} {'C col'}")
print("-" * 140)
for idx, (da, za, db, zb, dc, zc, r, n) in enumerate(triple_results[:30], 1):
    ca = za.replace('z_','')
    cb = zb.replace('z_','')
    cc = zc.replace('z_','')
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {da:<16} {ca:<24} {db:<16} {cb:<24} {dc:<16} {cc}")

# Show data for best results
if results:
    best = results[0]
    da, db = archives[best[0]], archives[best[2]]
    m = da.merge(db, on='year', how='inner')[['year', best[1], best[3]]].dropna()
    print(f"\nBest standard correlation ({best[0]} vs {best[2]}, r={best[4]:.4f}, n={best[5]}):")
    print(m.to_string(index=False))
