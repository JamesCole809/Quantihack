import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ---- Archive 1: Air quality ----
aq1 = pd.read_csv('data/archive/data.csv')
aq1['year'] = pd.to_datetime(aq1['day'], errors='coerce').dt.year
a1 = aq1.groupby('year').agg(
    avg_qty=('avg_daily_qqty', 'mean'),
    median_qty=('avg_daily_qqty', 'median'),
    std_qty=('avg_daily_qqty', 'std'),
    max_qty=('avg_daily_qqty', 'max'),
    count=('avg_daily_qqty', 'count'),
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
games = pd.read_csv('data/archive4/games.csv')
games['year'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce').dt.year
a4 = games.groupby('year').agg(
    chess_avg_white=('white_rating', 'mean'), chess_avg_black=('black_rating', 'mean'),
    chess_avg_turns=('turns', 'mean'), chess_games=('id', 'count'), chess_opening_ply=('opening_ply', 'mean'),
).reset_index()

# ---- Archive 5: Music reviews ----
conn = sqlite3.connect('data/archive5/database.sqlite')
reviews = pd.read_sql('SELECT * FROM reviews', conn)
conn.close()
a5 = reviews.groupby('pub_year').agg(
    music_avg_score=('score', 'mean'), music_std_score=('score', 'std'),
    music_reviews=('reviewid', 'count'), music_best_new=('best_new_music', 'mean'),
).reset_index()
a5.columns = ['year'] + list(a5.columns[1:])

# ---- Archive 7: FARS accidents ----
acc = pd.read_csv('data/archive7/accident.csv')
a7 = acc.groupby('YEAR').agg(
    accident_count=('accident_id', 'count'),
    accident_avg_fatals=('FATALS', 'mean'),
    accident_total_fatals=('FATALS', 'sum'),
    accident_avg_persons=('PERSONS', 'mean'),
    accident_avg_vehicles=('VE_TOTAL', 'mean'),
    accident_avg_hour=('HOUR', 'mean'),
    accident_avg_peds=('PEDS', 'mean'),
).reset_index()
a7.columns = ['year'] + list(a7.columns[1:])

person = pd.read_csv('data/archive7/person.csv')
a7p = person.groupby('Year').agg(
    person_avg_age=('AGE', lambda x: x[x < 998].mean()),
    person_count=('accident_id', 'count'),
    person_avg_drinking=('DRINKING', 'mean'),
).reset_index()
a7p.columns = ['year'] + list(a7p.columns[1:])
a7 = a7.merge(a7p, on='year', how='outer')

vehicle = pd.read_csv('data/archive7/vehicle.csv')
a7v = vehicle.groupby('Year').agg(
    vehicle_count=('accident_id', 'count'),
    vehicle_avg_speed=('TRAV_SP', lambda x: x[(x < 998) & (x > 0)].mean()),
    vehicle_avg_mod_year=('MOD_YEAR', lambda x: x[x < 9999].mean()),
).reset_index()
a7v.columns = ['year'] + list(a7v.columns[1:])
a7 = a7.merge(a7v, on='year', how='outer')

# ---- Archive 6: Blood cells (no year - skip) ----

archives = {
    'Archive1 (Air Quality)': a1,
    'Archive3 (Asylum)': a3,
    'Archive4 (Chess)': a4,
    'Archive5 (Music)': a5,
    'Archive7 (Accidents)': a7,
}

# ---- All cross-archive pairs ----
archive_names = list(archives.keys())
results = []

for i in range(len(archive_names)):
    for j in range(i+1, len(archive_names)):
        name_a, name_b = archive_names[i], archive_names[j]
        df_a, df_b = archives[name_a], archives[name_b]
        merged = df_a.merge(df_b, on='year', how='inner')
        if len(merged) < 3:
            continue
        cols_a = [c for c in df_a.columns if c != 'year']
        cols_b = [c for c in df_b.columns if c != 'year']
        for ca in cols_a:
            for cb in cols_b:
                sub = merged[[ca, cb]].dropna()
                if len(sub) < 3:
                    continue
                r = sub[ca].corr(sub[cb])
                if not np.isnan(r):
                    results.append((name_a, ca, name_b, cb, r, len(sub)))

results.sort(key=lambda x: abs(x[4]), reverse=True)

print("=" * 130)
print("CROSS-ARCHIVE CORRELATIONS (by year, min 3 overlapping years)")
print("=" * 130)
print(f"{'Rank':>4}  {'r':>8}  {'n':>3}  {'Archive A':<22} {'Column A':<28} {'Archive B':<22} {'Column B'}")
print("-" * 130)
for idx, (a, ca, b, cb, r, n) in enumerate(results[:50], 1):
    print(f"{idx:>4}  {r:>8.4f}  {n:>3}  {a:<22} {ca:<28} {b:<22} {cb}")

# Top 3 data points
for rank, (a, ca, b, cb, r, n) in enumerate(results[:3], 1):
    df_a, df_b = archives[a], archives[b]
    merged = df_a.merge(df_b, on='year', how='inner')[['year', ca, cb]].dropna()
    print(f"\n--- #{rank}: {a} '{ca}' vs {b} '{cb}' (r={r:.4f}) ---")
    print(merged.to_string(index=False))
