import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ---- Archive 1: Air quality (2019-2020) ----
aq1 = pd.read_csv('data/archive/data.csv', usecols=['day','avg_daily_qqty','pollutant_code'])
aq1['year'] = pd.to_datetime(aq1['day'], errors='coerce').dt.year
a1 = aq1.groupby('year').agg(
    aq_avg_qty=('avg_daily_qqty', 'mean'), aq_median_qty=('avg_daily_qqty', 'median'),
    aq_std_qty=('avg_daily_qqty', 'std'), aq_count=('avg_daily_qqty', 'count'),
).reset_index()

# ---- Archive 3: Asylum (2012-2021) ----
asilo_ca = pd.read_csv('data/archive3/AsiloCA.csv', encoding='latin1')
yr_col = [c for c in asilo_ca.columns if 'o' in c.lower()][-1]
asilo_es = pd.read_csv('data/archive3/AsiloEspaa.csv', encoding='latin1')
yr_col2 = [c for c in asilo_es.columns if 'o' in c.lower()][-1]
a3a = asilo_ca.groupby(yr_col).agg(asylum_region_total=('Solicitantes', 'sum'), asylum_region_avg=('Solicitantes', 'mean'), asylum_region_max=('Solicitantes', 'max')).reset_index()
a3a.columns = ['year'] + list(a3a.columns[1:])
a3b = asilo_es.groupby(yr_col2).agg(asylum_nat_total=('Total', 'sum'), asylum_nat_men=('Hombres', 'sum'), asylum_nat_women=('Mujeres', 'sum'), asylum_nat_admitted=('Admitidas', 'sum')).reset_index()
a3b.columns = ['year'] + list(a3b.columns[1:])
a3 = a3a.merge(a3b, on='year', how='outer')

# ---- Archive 4: Chess (2013-2017) ----
games = pd.read_csv('data/archive4/games.csv', usecols=['created_at','turns','id','white_rating','black_rating','opening_ply'], engine='python')
games['year'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce').dt.year
a4 = games.groupby('year').agg(
    chess_avg_white=('white_rating', 'mean'), chess_avg_black=('black_rating', 'mean'),
    chess_avg_turns=('turns', 'mean'), chess_games=('id', 'count'), chess_opening_ply=('opening_ply', 'mean'),
).reset_index()

# ---- Archive 5: Music reviews (1999-2017) ----
conn = sqlite3.connect('data/archive5/database.sqlite')
reviews = pd.read_sql('SELECT reviewid, score, best_new_music, pub_year FROM reviews', conn)
conn.close()
a5 = reviews.groupby('pub_year').agg(
    music_avg_score=('score', 'mean'), music_std_score=('score', 'std'),
    music_reviews=('reviewid', 'count'), music_best_new=('best_new_music', 'mean'),
).reset_index()
a5.columns = ['year'] + list(a5.columns[1:])

# ---- Archive 7: US Accidents (2012-2016) ----
acc = pd.read_csv('data/archive7/accident.csv', usecols=['accident_id','YEAR','FATALS','VE_TOTAL','PERSONS'])
a7 = acc.groupby('YEAR').agg(
    acc_count=('accident_id', 'count'), acc_avg_fatals=('FATALS', 'mean'),
    acc_total_fatals=('FATALS', 'sum'), acc_avg_vehicles=('VE_TOTAL', 'mean'),
    acc_avg_persons=('PERSONS', 'mean'),
).reset_index()
a7.columns = ['year'] + list(a7.columns[1:])

# ---- Archive 21: Climate/Soil (2000-2016) ----
climate = pd.read_csv('data/archive21/train_timeseries/train_timeseries.csv',
    usecols=['date','PRECTOT','T2M','T2M_MAX','T2M_MIN','WS10M','PS','QV2M','RH_out' if False else 'T2MDEW'],
    nrows=5000000)
climate['year'] = pd.to_datetime(climate['date'], errors='coerce').dt.year
a21 = climate.groupby('year').agg(
    climate_avg_temp=('T2M', 'mean'), climate_avg_precip=('PRECTOT', 'mean'),
    climate_avg_wind=('WS10M', 'mean'), climate_avg_pressure=('PS', 'mean'),
    climate_avg_dewpoint=('T2MDEW', 'mean'), climate_max_temp=('T2M_MAX', 'mean'),
    climate_min_temp=('T2M_MIN', 'mean'),
).reset_index()

# ---- Archive 22: Energy (2016) ----
energy = pd.read_csv('data/archive22/energydata_complete.csv')
energy['year'] = pd.to_datetime(energy['date'], errors='coerce').dt.year
a22 = energy.groupby('year').agg(
    energy_avg_appliances=('Appliances', 'mean'),
    energy_avg_lights=('lights', 'mean'),
    energy_avg_t_out=('T_out', 'mean'),
    energy_avg_windspeed=('Windspeed', 'mean'),
    energy_avg_pressure=('Press_mm_hg', 'mean'),
).reset_index()

# ---- Archive 23: Solar (2017-2019) ----
solar = pd.read_csv('data/archive23/2017_2019.csv')
a23 = solar.groupby('Year').agg(
    solar_avg_temp=('Temperature', 'mean'),
    solar_avg_ghi=('GHI', 'mean'),
    solar_avg_dni=('DNI', 'mean'),
    solar_avg_dhi=('DHI', 'mean'),
    solar_avg_wind=('Wind Speed', 'mean'),
    solar_avg_humidity=('Relative Humidity', 'mean'),
    solar_avg_pressure=('Pressure', 'mean'),
).reset_index()
a23.columns = ['year'] + list(a23.columns[1:])

# ---- Archive 24: UK Accidents (2005-2010) ----
uk_acc = pd.read_csv('data/archive24/accident_data.csv', usecols=['Year','Accident_Severity','Number_of_Casualties','Number_of_Vehicles','Speed_limit'])
a24 = uk_acc.groupby('Year').agg(
    uk_acc_count=('Number_of_Casualties', 'count'),
    uk_acc_avg_casualties=('Number_of_Casualties', 'mean'),
    uk_acc_avg_vehicles=('Number_of_Vehicles', 'mean'),
    uk_acc_avg_speed_limit=('Speed_limit', 'mean'),
).reset_index()
a24.columns = ['year'] + list(a24.columns[1:])

# ---- Archive 27: Coal (1981-2021) ----
coal = pd.read_csv('data/archive27/world-coal-production.csv', sep=';')
a27 = coal.groupby('Year').agg(
    coal_total_production=('Value (Million Tonnes)', 'sum'),
    coal_avg_production=('Value (Million Tonnes)', 'mean'),
    coal_max_production=('Value (Million Tonnes)', 'max'),
    coal_num_countries=('Country', 'nunique'),
).reset_index()
a27.columns = ['year'] + list(a27.columns[1:])

# ---- Build archives dict ----
archives = {
    'AirQuality': a1,
    'Asylum': a3,
    'Chess': a4,
    'Music': a5,
    'US_Accidents': a7,
    'Climate': a21,
    'Energy': a22,
    'Solar': a23,
    'UK_Accidents': a24,
    'Coal': a27,
}

print("Archive year ranges:")
for name, df in archives.items():
    print(f"  {name:<15} {int(df['year'].min())}-{int(df['year'].max())} ({len(df)} years)")

# ---- Cross-archive correlations ----
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
print("TOP 50 CROSS-ARCHIVE CORRELATIONS (by year)")
print("=" * 130)
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'Archive A':<15} {'Column A':<28} {'Archive B':<15} {'Column B'}")
print("-" * 130)
for idx, (a, ca, b, cb, r, n) in enumerate(results[:50], 1):
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {a:<15} {ca:<28} {b:<15} {cb}")

# ---- Highlight pairs with most overlapping years ----
print("\n" + "=" * 130)
print("TOP CORRELATIONS WITH 6+ OVERLAPPING YEARS")
print("=" * 130)
long_results = [r for r in results if r[5] >= 6]
long_results.sort(key=lambda x: abs(x[4]), reverse=True)
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'Archive A':<15} {'Column A':<28} {'Archive B':<15} {'Column B'}")
print("-" * 130)
for idx, (a, ca, b, cb, r, n) in enumerate(long_results[:30], 1):
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {a:<15} {ca:<28} {b:<15} {cb}")

# Show data for top long correlation
if long_results:
    best = long_results[0]
    da, db = archives[best[0]], archives[best[2]]
    m = da.merge(db, on='year', how='inner')[['year', best[1], best[3]]].dropna()
    print(f"\nBest long correlation data ({best[0]} vs {best[2]}, r={best[4]:.4f}):")
    print(m.to_string(index=False))
