import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Chess - skip 'moves' column (huge strings cause OOM)
games = pd.read_csv('data/archive4/games.csv', usecols=['created_at','turns','id','white_rating','black_rating','opening_ply'], engine='python')
games['dt'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce')
games['ym'] = games['dt'].dt.to_period('M')
chess = games.groupby('ym').agg(
    chess_avg_turns=('turns', 'mean'),
    chess_games=('id', 'count'),
    chess_avg_white=('white_rating', 'mean'),
    chess_avg_black=('black_rating', 'mean'),
    chess_opening_ply=('opening_ply', 'mean'),
).reset_index()

# Accidents
acc = pd.read_csv('data/archive7/accident.csv', usecols=['accident_id','YEAR','MONTH','FATALS','VE_TOTAL','PERSONS','HOUR','PEDS'])
acc['ym'] = pd.to_datetime(acc[['YEAR','MONTH']].assign(DAY=1)).dt.to_period('M')
accidents = acc.groupby('ym').agg(
    acc_count=('accident_id', 'count'),
    acc_avg_fatals=('FATALS', 'mean'),
    acc_total_fatals=('FATALS', 'sum'),
    acc_avg_vehicles=('VE_TOTAL', 'mean'),
    acc_avg_persons=('PERSONS', 'mean'),
    acc_avg_hour=('HOUR', 'mean'),
).reset_index()

# Music reviews
conn = sqlite3.connect('data/archive5/database.sqlite')
reviews = pd.read_sql('SELECT reviewid, score, best_new_music, pub_date FROM reviews', conn)
conn.close()
reviews['dt'] = pd.to_datetime(reviews['pub_date'], errors='coerce')
reviews['ym'] = reviews['dt'].dt.to_period('M')
music = reviews.dropna(subset=['ym']).groupby('ym').agg(
    music_avg_score=('score', 'mean'),
    music_std_score=('score', 'std'),
    music_reviews=('reviewid', 'count'),
    music_best_new=('best_new_music', 'mean'),
).reset_index()

# Air quality
aq = pd.read_csv('data/archive/data.csv', usecols=['day','avg_daily_qqty','pollutant_code'], engine='python')
aq['dt'] = pd.to_datetime(aq['day'], errors='coerce')
aq['ym'] = aq['dt'].dt.to_period('M')
airq = aq.dropna(subset=['ym']).groupby('ym').agg(
    aq_avg_qty=('avg_daily_qqty', 'mean'),
    aq_median_qty=('avg_daily_qqty', 'median'),
    aq_std_qty=('avg_daily_qqty', 'std'),
    aq_count=('avg_daily_qqty', 'count'),
).reset_index()

# ---- Monthly cross-archive correlations ----
monthly_datasets = {
    'Chess': chess,
    'Accidents': accidents,
    'Music': music,
    'Air Quality': airq,
}

results = []
names = list(monthly_datasets.keys())

for i in range(len(names)):
    for j in range(i+1, len(names)):
        na, nb = names[i], names[j]
        da, db = monthly_datasets[na], monthly_datasets[nb]
        merged = da.merge(db, on='ym', how='inner')
        if len(merged) < 5:
            continue
        cols_a = [c for c in da.columns if c != 'ym']
        cols_b = [c for c in db.columns if c != 'ym']
        for ca in cols_a:
            for cb in cols_b:
                sub = merged[[ca, cb]].dropna()
                if len(sub) < 5:
                    continue
                r = sub[ca].corr(sub[cb])
                if not np.isnan(r):
                    results.append((na, ca, nb, cb, r, len(sub)))

results.sort(key=lambda x: abs(x[4]), reverse=True)

print("=" * 130)
print("MONTHLY CROSS-ARCHIVE CORRELATIONS (min 5 overlapping months)")
print("=" * 130)
print(f"{'Rank':>4}  {'r':>8}  {'n':>4}  {'Archive A':<15} {'Column A':<22} {'Archive B':<15} {'Column B'}")
print("-" * 130)
for idx, (a, ca, b, cb, r, n) in enumerate(results[:40], 1):
    print(f"{idx:>4}  {r:>8.4f}  {n:>4}  {a:<15} {ca:<22} {b:<15} {cb}")
