import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---- Load all archives ----

# Archive 1: Air quality (2019-2020)
aq1 = pd.read_csv('data/archive/data.csv', usecols=['day','avg_daily_qqty','pollutant_code'])
aq1['year'] = pd.to_datetime(aq1['day'], errors='coerce').dt.year
a1 = aq1.groupby('year').agg(
    aq_avg_qty=('avg_daily_qqty', 'mean'), aq_median_qty=('avg_daily_qqty', 'median'),
    aq_std_qty=('avg_daily_qqty', 'std'), aq_count=('avg_daily_qqty', 'count'),
).reset_index()

# Archive 3: Asylum (2012-2021)
asilo_ca = pd.read_csv('data/archive3/AsiloCA.csv', encoding='latin1')
yr_col = [c for c in asilo_ca.columns if 'o' in c.lower()][-1]
asilo_es = pd.read_csv('data/archive3/AsiloEspaa.csv', encoding='latin1')
yr_col2 = [c for c in asilo_es.columns if 'o' in c.lower()][-1]
a3a = asilo_ca.groupby(yr_col).agg(asylum_region_total=('Solicitantes', 'sum'), asylum_region_avg=('Solicitantes', 'mean'), asylum_region_max=('Solicitantes', 'max')).reset_index()
a3a.columns = ['year'] + list(a3a.columns[1:])
a3b = asilo_es.groupby(yr_col2).agg(asylum_nat_total=('Total', 'sum'), asylum_nat_men=('Hombres', 'sum'), asylum_nat_women=('Mujeres', 'sum'), asylum_nat_admitted=('Admitidas', 'sum')).reset_index()
a3b.columns = ['year'] + list(a3b.columns[1:])
a3 = a3a.merge(a3b, on='year', how='outer')

# Archive 4: Chess (2013-2017)
games = pd.read_csv('data/archive4/games.csv', usecols=['created_at','turns','id','white_rating','black_rating','opening_ply'], engine='python')
games['year'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce').dt.year
a4 = games.groupby('year').agg(
    chess_avg_white=('white_rating', 'mean'), chess_avg_black=('black_rating', 'mean'),
    chess_avg_turns=('turns', 'mean'), chess_games=('id', 'count'), chess_opening_ply=('opening_ply', 'mean'),
).reset_index()

# Archive 7: US Accidents (2012-2016)
acc = pd.read_csv('data/archive7/accident.csv', usecols=['accident_id','YEAR','FATALS','VE_TOTAL','PERSONS'])
a7 = acc.groupby('YEAR').agg(
    us_acc_count=('accident_id', 'count'), us_acc_avg_fatals=('FATALS', 'mean'),
    us_acc_total_fatals=('FATALS', 'sum'), us_acc_avg_vehicles=('VE_TOTAL', 'mean'),
    us_acc_avg_persons=('PERSONS', 'mean'),
).reset_index()
a7.columns = ['year'] + list(a7.columns[1:])
vehicle = pd.read_csv('data/archive7/vehicle.csv', usecols=['Year','TRAV_SP','MOD_YEAR'])
a7v = vehicle.groupby('Year').agg(
    us_vehicle_avg_speed=('TRAV_SP', lambda x: x[(x < 998) & (x > 0)].mean()),
    us_vehicle_avg_mod_year=('MOD_YEAR', lambda x: x[x < 9999].mean()),
).reset_index()
a7v.columns = ['year'] + list(a7v.columns[1:])
a7 = a7.merge(a7v, on='year', how='outer')

# Archive 21: Climate (2000-2016)
climate = pd.read_csv('data/archive21/train_timeseries/train_timeseries.csv',
    usecols=['date','PRECTOT','T2M','T2M_MAX','T2M_MIN','WS10M','PS','T2MDEW'], nrows=5000000)
climate['year'] = pd.to_datetime(climate['date'], errors='coerce').dt.year
a21 = climate.groupby('year').agg(
    climate_avg_temp=('T2M', 'mean'), climate_avg_precip=('PRECTOT', 'mean'),
    climate_avg_wind=('WS10M', 'mean'), climate_avg_pressure=('PS', 'mean'),
    climate_avg_dewpoint=('T2MDEW', 'mean'), climate_max_temp=('T2M_MAX', 'mean'),
    climate_min_temp=('T2M_MIN', 'mean'),
).reset_index()

# Archive 23: Solar (2017-2019)
solar = pd.read_csv('data/archive23/2017_2019.csv')
a23 = solar.groupby('Year').agg(
    solar_avg_temp=('Temperature', 'mean'), solar_avg_ghi=('GHI', 'mean'),
    solar_avg_dni=('DNI', 'mean'), solar_avg_dhi=('DHI', 'mean'),
    solar_avg_wind=('Wind Speed', 'mean'), solar_avg_humidity=('Relative Humidity', 'mean'),
    solar_avg_pressure=('Pressure', 'mean'),
).reset_index()
a23.columns = ['year'] + list(a23.columns[1:])

# Archive 24: UK Accidents (2005-2010)
uk_acc = pd.read_csv('data/archive24/accident_data.csv', usecols=['Year','Number_of_Casualties','Number_of_Vehicles','Speed_limit'])
a24 = uk_acc.groupby('Year').agg(
    uk_acc_count=('Number_of_Casualties', 'count'),
    uk_acc_avg_casualties=('Number_of_Casualties', 'mean'),
    uk_acc_avg_vehicles=('Number_of_Vehicles', 'mean'),
    uk_acc_avg_speed_limit=('Speed_limit', 'mean'),
).reset_index()
a24.columns = ['year'] + list(a24.columns[1:])

# Archive 27: Coal (1981-2021)
coal = pd.read_csv('data/archive27/world-coal-production.csv', sep=';')
a27 = coal.groupby('Year').agg(
    coal_total_production=('Value (Million Tonnes)', 'sum'),
    coal_avg_production=('Value (Million Tonnes)', 'mean'),
    coal_max_production=('Value (Million Tonnes)', 'max'),
    coal_num_countries=('Country', 'nunique'),
).reset_index()
a27.columns = ['year'] + list(a27.columns[1:])

archives = {
    'AirQuality': a1, 'Asylum': a3, 'Chess': a4, 'US_Accidents': a7,
    'Climate': a21, 'Solar': a23, 'UK_Accidents': a24, 'Coal': a27,
}

# ---- Merge all on year, compute z-scores ----
merged_all = pd.DataFrame({'year': range(1981, 2022)})
for df in archives.values():
    merged_all = merged_all.merge(df, on='year', how='left')

numeric_cols = [c for c in merged_all.columns if c != 'year']
zscores = merged_all[['year']].copy()
for c in numeric_cols:
    s = merged_all[c]
    if s.std() > 0:
        zscores[f'z_{c}'] = (s - s.mean()) / s.std()

dataset_tags = {
    'aq_': 'AirQuality', 'asylum_': 'Asylum', 'chess_': 'Chess',
    'us_acc': 'US_Accidents', 'us_vehicle': 'US_Accidents',
    'climate_': 'Climate', 'solar_': 'Solar',
    'uk_acc': 'UK_Accidents', 'coal_': 'Coal',
}

def get_ds(col):
    col = col.replace('z_', '')
    for prefix, ds in dataset_tags.items():
        if col.startswith(prefix):
            return ds
    return 'unknown'

z_cols = [c for c in zscores.columns if c.startswith('z_')]

# ---- PART 1: Z-score of A * Z-score of B correlated with C ----
print("=" * 140)
print("Z-SCORE PRODUCTS (A * B) CORRELATED WITH C  (min 4 overlapping years)")
print("=" * 140)

triple_results = []

for i in range(len(z_cols)):
    for j in range(i+1, len(z_cols)):
        za, zb = z_cols[i], z_cols[j]
        ds_a, ds_b = get_ds(za), get_ds(zb)
        if ds_a == ds_b:
            continue

        product = zscores[za] * zscores[zb]

        for k in range(len(z_cols)):
            zc = z_cols[k]
            ds_c = get_ds(zc)
            if ds_c in (ds_a, ds_b):
                continue

            mask = product.notna() & zscores[zc].notna()
            if mask.sum() < 4:
                continue

            r = product[mask].corr(zscores[zc][mask])
            if not np.isnan(r) and abs(r) > 0.85:
                triple_results.append((ds_a, za, ds_b, zb, ds_c, zc, r, int(mask.sum())))

triple_results.sort(key=lambda x: abs(x[6]), reverse=True)

print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'A dataset':<13} {'A col':<25} {'B dataset':<13} {'B col':<25} {'C dataset':<13} {'C col'}")
print("-" * 140)
for idx, (da, za, db, zb, dc, zc, r, n) in enumerate(triple_results[:30], 1):
    ca = za.replace('z_','')
    cb = zb.replace('z_','')
    cc = zc.replace('z_','')
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {da:<13} {ca:<25} {db:<13} {cb:<25} {dc:<13} {cc}")

# ---- PART 2: Z-score ratio (A/B) correlated with C ----
print("\n" + "=" * 140)
print("Z-SCORE RATIOS (A / B) CORRELATED WITH C  (min 4 overlapping years)")
print("=" * 140)

ratio_results = []

for i in range(len(z_cols)):
    for j in range(len(z_cols)):
        if i == j:
            continue
        za, zb = z_cols[i], z_cols[j]
        ds_a, ds_b = get_ds(za), get_ds(zb)
        if ds_a == ds_b:
            continue

        # Avoid division by near-zero
        safe = zscores[zb].abs() > 0.1
        ratio = pd.Series(np.nan, index=zscores.index)
        ratio[safe] = zscores[za][safe] / zscores[zb][safe]

        for k in range(len(z_cols)):
            zc = z_cols[k]
            ds_c = get_ds(zc)
            if ds_c in (ds_a, ds_b):
                continue

            mask = ratio.notna() & zscores[zc].notna()
            if mask.sum() < 4:
                continue

            r = ratio[mask].corr(zscores[zc][mask])
            if not np.isnan(r) and abs(r) > 0.9:
                ratio_results.append((ds_a, za, ds_b, zb, ds_c, zc, r, int(mask.sum())))

ratio_results.sort(key=lambda x: abs(x[6]), reverse=True)

print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'A dataset':<13} {'A col':<25} {'B dataset':<13} {'B col':<25} {'C dataset':<13} {'C col'}")
print("-" * 140)
for idx, (da, za, db, zb, dc, zc, r, n) in enumerate(ratio_results[:20], 1):
    ca = za.replace('z_','')
    cb = zb.replace('z_','')
    cc = zc.replace('z_','')
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {da:<13} {ca:<25} {db:<13} {cb:<25} {dc:<13} {cc}")

# ---- PART 3: Squared z-scores correlated across archives ----
print("\n" + "=" * 140)
print("SQUARED Z-SCORES (A^2) CORRELATED WITH B  (min 5 overlapping years)")
print("=" * 140)

sq_results = []
for i in range(len(z_cols)):
    for j in range(len(z_cols)):
        if i == j:
            continue
        za, zb = z_cols[i], z_cols[j]
        ds_a, ds_b = get_ds(za), get_ds(zb)
        if ds_a == ds_b:
            continue

        sq = zscores[za] ** 2
        mask = sq.notna() & zscores[zb].notna()
        if mask.sum() < 5:
            continue
        r = sq[mask].corr(zscores[zb][mask])
        if not np.isnan(r) and abs(r) > 0.85:
            sq_results.append((ds_a, za, ds_b, zb, r, int(mask.sum())))

sq_results.sort(key=lambda x: abs(x[4]), reverse=True)
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'A dataset':<13} {'A col (squared)':<30} {'B dataset':<13} {'B col'}")
print("-" * 110)
for idx, (da, za, db, zb, r, n) in enumerate(sq_results[:20], 1):
    ca = za.replace('z_','')
    cb = zb.replace('z_','')
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {da:<13} {ca:<30} {db:<13} {cb}")
