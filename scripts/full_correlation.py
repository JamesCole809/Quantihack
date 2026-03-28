import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ---- Archive 1: Air quality ----
aq1 = pd.read_csv('data/archive/data.csv', usecols=['day','avg_daily_qqty','pollutant_code'])
aq1['year'] = pd.to_datetime(aq1['day'], errors='coerce').dt.year
a1 = aq1.groupby('year').agg(
    aq_avg_qty=('avg_daily_qqty', 'mean'), aq_median_qty=('avg_daily_qqty', 'median'),
    aq_std_qty=('avg_daily_qqty', 'std'), aq_max_qty=('avg_daily_qqty', 'max'), aq_count=('avg_daily_qqty', 'count'),
).reset_index()

# ---- Archive 3: Asylum ----
asilo_ca = pd.read_csv('data/archive3/AsiloCA.csv', encoding='latin1')
yr_col = [c for c in asilo_ca.columns if 'o' in c.lower()][-1]
asilo_es = pd.read_csv('data/archive3/AsiloEspaa.csv', encoding='latin1')
yr_col2 = [c for c in asilo_es.columns if 'o' in c.lower()][-1]
a3a = asilo_ca.groupby(yr_col).agg(asylum_region_total=('Solicitantes', 'sum'), asylum_region_avg=('Solicitantes', 'mean'), asylum_region_max=('Solicitantes', 'max')).reset_index()
a3a.columns = ['year'] + list(a3a.columns[1:])
a3b = asilo_es.groupby(yr_col2).agg(asylum_nat_total=('Total', 'sum'), asylum_nat_men=('Hombres', 'sum'), asylum_nat_women=('Mujeres', 'sum'), asylum_nat_admitted=('Admitidas', 'sum')).reset_index()
a3b.columns = ['year'] + list(a3b.columns[1:])
a3 = a3a.merge(a3b, on='year', how='outer')

# ---- Archive 4: Chess ----
games = pd.read_csv('data/archive4/games.csv', usecols=['created_at','turns','id','white_rating','black_rating','opening_ply'], engine='python')
games['year'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce').dt.year
a4 = games.groupby('year').agg(
    chess_avg_white=('white_rating', 'mean'), chess_avg_black=('black_rating', 'mean'),
    chess_avg_turns=('turns', 'mean'), chess_games=('id', 'count'), chess_opening_ply=('opening_ply', 'mean'),
).reset_index()

# ---- Archive 5: Music reviews ----
conn = sqlite3.connect('data/archive5/database.sqlite')
reviews = pd.read_sql('SELECT reviewid, score, best_new_music, pub_year FROM reviews', conn)
conn.close()
a5 = reviews.groupby('pub_year').agg(
    music_avg_score=('score', 'mean'), music_std_score=('score', 'std'),
    music_reviews=('reviewid', 'count'), music_best_new=('best_new_music', 'mean'),
).reset_index()
a5.columns = ['year'] + list(a5.columns[1:])

# ---- Archive 7: Accidents ----
acc = pd.read_csv('data/archive7/accident.csv', usecols=['accident_id','YEAR','MONTH','FATALS','VE_TOTAL','PERSONS','HOUR','PEDS'])
a7 = acc.groupby('YEAR').agg(
    acc_count=('accident_id', 'count'), acc_avg_fatals=('FATALS', 'mean'),
    acc_total_fatals=('FATALS', 'sum'), acc_avg_vehicles=('VE_TOTAL', 'mean'),
    acc_avg_persons=('PERSONS', 'mean'),
).reset_index()
a7.columns = ['year'] + list(a7.columns[1:])

vehicle = pd.read_csv('data/archive7/vehicle.csv', usecols=['Year','TRAV_SP','MOD_YEAR','accident_id'])
a7v = vehicle.groupby('Year').agg(
    vehicle_avg_speed=('TRAV_SP', lambda x: x[(x < 998) & (x > 0)].mean()),
    vehicle_avg_mod_year=('MOD_YEAR', lambda x: x[x < 9999].mean()),
    vehicle_count=('accident_id', 'count'),
).reset_index()
a7v.columns = ['year'] + list(a7v.columns[1:])
a7 = a7.merge(a7v, on='year', how='outer')

# ---- Archive 19: Flights ----
flights = pd.read_parquet('data/archive19/flight_data.parquet')
print(f"Flight years: {sorted(flights['Year'].unique())}")

a19 = flights.groupby('Year').agg(
    flight_count=('Flight_Number_Marketing_Airline', 'count'),
    flight_avg_dep_delay=('DepDelay', 'mean'),
    flight_avg_arr_delay=('ArrDelay', 'mean'),
    flight_avg_distance=('Distance', 'mean'),
    flight_avg_air_time=('AirTime', 'mean'),
    flight_cancel_rate=('Cancelled', 'mean'),
    flight_divert_rate=('Diverted', 'mean'),
    flight_avg_taxi_out=('TaxiOut', 'mean'),
).reset_index()
a19.columns = ['year'] + list(a19.columns[1:])

# ---- Build archives dict ----
archives = {
    'AirQuality': a1,
    'Asylum': a3,
    'Chess': a4,
    'Music': a5,
    'Accidents': a7,
    'Flights': a19,
}

# ---- PART 1: Standard cross-archive correlations ----
archive_names = list(archives.keys())
results = []

for i in range(len(archive_names)):
    for j in range(i+1, len(archive_names)):
        na, nb = archive_names[i], archive_names[j]
        da, db = archives[na], archives[nb]
        merged = da.merge(db, on='year', how='inner')
        if len(merged) < 3:
            continue
        cols_a = [c for c in da.columns if c != 'year']
        cols_b = [c for c in db.columns if c != 'year']
        for ca in cols_a:
            for cb in cols_b:
                sub = merged[[ca, cb]].dropna()
                if len(sub) < 3:
                    continue
                r = sub[ca].corr(sub[cb])
                if not np.isnan(r):
                    results.append((na, ca, nb, cb, r, len(sub)))

results.sort(key=lambda x: abs(x[4]), reverse=True)

print("\n" + "=" * 130)
print("CROSS-ARCHIVE CORRELATIONS (by year, min 3 overlapping years)")
print("=" * 130)
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'Archive A':<15} {'Column A':<28} {'Archive B':<15} {'Column B'}")
print("-" * 130)
for idx, (a, ca, b, cb, r, n) in enumerate(results[:50], 1):
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {a:<15} {ca:<28} {b:<15} {cb}")

# ---- PART 2: Z-score cross-archive correlations ----
print("\n" + "=" * 130)
print("Z-SCORE CROSS-ARCHIVE CORRELATIONS")
print("=" * 130)

merged_all = a1.copy()
for df in [a3, a4, a5, a7, a19]:
    merged_all = merged_all.merge(df, on='year', how='outer')
merged_all = merged_all.sort_values('year')

numeric_cols = [c for c in merged_all.columns if c != 'year' and merged_all[c].dtype in ['float64', 'int64']]
zscores = merged_all[['year']].copy()
for c in numeric_cols:
    s = merged_all[c]
    zscores[f'z_{c}'] = (s - s.mean()) / s.std()

dataset_tags = {
    'aq_': 'AirQuality', 'asylum_': 'Asylum', 'chess_': 'Chess',
    'music_': 'Music', 'acc_': 'Accidents', 'vehicle_': 'Accidents',
    'flight_': 'Flights',
}

def get_ds(col):
    col = col.replace('z_', '')
    for prefix, ds in dataset_tags.items():
        if col.startswith(prefix):
            return ds
    return 'unknown'

z_cols = [c for c in zscores.columns if c.startswith('z_')]
z_results = []
for i in range(len(z_cols)):
    for j in range(i+1, len(z_cols)):
        c1, c2 = z_cols[i], z_cols[j]
        ds1, ds2 = get_ds(c1), get_ds(c2)
        if ds1 == ds2:
            continue
        sub = zscores[[c1, c2]].dropna()
        if len(sub) < 3:
            continue
        r = sub[c1].corr(sub[c2])
        if not np.isnan(r):
            z_results.append((ds1, c1, ds2, c2, r, len(sub)))

z_results.sort(key=lambda x: abs(x[4]), reverse=True)
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'Archive A':<15} {'Z-Column A':<30} {'Archive B':<15} {'Z-Column B'}")
print("-" * 130)
for idx, (a, ca, b, cb, r, n) in enumerate(z_results[:30], 1):
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {a:<15} {ca:<30} {b:<15} {cb}")

# ---- PART 3: Z-score product (A*B) correlated with third archive C ----
print("\n" + "=" * 130)
print("Z-SCORE PRODUCT (A*B) CORRELATED WITH THIRD ARCHIVE C")
print("=" * 130)

top_z_pairs = z_results[:10]
triple_results = []

for ds_a, za, ds_b, zb, r_ab, n_ab in top_z_pairs:
    product = zscores[za] * zscores[zb]
    product_df = pd.DataFrame({'year': zscores['year'], 'z_product': product})

    for ds_name, ds_df in archives.items():
        if ds_name in (ds_a, ds_b):
            continue
        merged = product_df.merge(ds_df, on='year', how='inner')
        for col in [c for c in ds_df.columns if c != 'year']:
            sub = merged[['z_product', col]].dropna()
            if len(sub) < 3:
                continue
            r = sub['z_product'].corr(sub[col])
            if not np.isnan(r):
                triple_results.append((ds_a, za, ds_b, zb, r_ab, ds_name, col, r, len(sub)))

triple_results.sort(key=lambda x: abs(x[7]), reverse=True)
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'Pair':<55} {'Third Archive':<15} {'Third Column'}")
print("-" * 130)
for idx, (da, za, db, zb, rab, dc, col, r, n) in enumerate(triple_results[:20], 1):
    pair = f"{da}:{za.replace('z_','')} x {db}:{zb.replace('z_','')}"
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {pair:<55} {dc:<15} {col}")