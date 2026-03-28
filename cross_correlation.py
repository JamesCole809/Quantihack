import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CROSS-DATASET CORRELATION ANALYSIS")
print("=" * 80)

# ---- Load & aggregate all datasets by year ----

# Air quality
aq = pd.read_csv('data/archive/data.csv')
aq['year'] = pd.to_datetime(aq['day'], errors='coerce').dt.year
aq_year = aq.groupby('year').agg(
    aq_avg_qty=('avg_daily_qqty', 'mean'),
    aq_median_qty=('avg_daily_qqty', 'median'),
    aq_std_qty=('avg_daily_qqty', 'std'),
    aq_max_qty=('avg_daily_qqty', 'max'),
    aq_count=('avg_daily_qqty', 'count'),
    aq_avg_pollutant_code=('pollutant_code', 'mean'),
).reset_index()

# Asylum regions
asilo_ca = pd.read_csv('data/archive3/AsiloCA.csv', encoding='latin1')
asylum_year = asilo_ca.groupby(asilo_ca.columns[3]).agg(
    asylum_total_requests=('Solicitantes', 'sum'),
    asylum_avg_requests=('Solicitantes', 'mean'),
    asylum_max_requests=('Solicitantes', 'max'),
    asylum_num_regions=(asilo_ca.columns[1], 'nunique'),
).reset_index()
asylum_year.columns = ['year'] + list(asylum_year.columns[1:])

# Asylum nationalities
asilo_es = pd.read_csv('data/archive3/AsiloEspaa.csv', encoding='latin1')
year_col = [c for c in asilo_es.columns if 'o' in c.lower() or 'year' in c.lower() or 'a' in c.lower()][-1]
asylum_nat_year = asilo_es.groupby(year_col).agg(
    asylum_nat_total=('Total', 'sum'),
    asylum_nat_men=('Hombres', 'sum'),
    asylum_nat_women=('Mujeres', 'sum'),
    asylum_nat_admitted=('Admitidas', 'sum'),
    asylum_nat_countries=('Nacionalidad ', 'nunique'),
    asylum_nat_admit_rate=('Admitidas', lambda x: x.sum()),
).reset_index()
asylum_nat_year.columns = ['year'] + list(asylum_nat_year.columns[1:])
asylum_nat_year['asylum_admit_pct'] = asylum_nat_year['asylum_nat_admitted'] / asylum_nat_year['asylum_nat_total']
asylum_nat_year['asylum_gender_ratio'] = asylum_nat_year['asylum_nat_men'] / asylum_nat_year['asylum_nat_women']

# Chess games
games = pd.read_csv('data/archive4/games.csv')
games['year'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce').dt.year
chess_year = games.groupby('year').agg(
    chess_avg_white_rating=('white_rating', 'mean'),
    chess_avg_black_rating=('black_rating', 'mean'),
    chess_avg_turns=('turns', 'mean'),
    chess_num_games=('id', 'count'),
    chess_avg_opening_ply=('opening_ply', 'mean'),
    chess_white_win_rate=('winner', lambda x: (x == 'white').mean()),
    chess_draw_rate=('winner', lambda x: (x == 'draw').mean()),
    chess_rating_spread=('white_rating', 'std'),
).reset_index()

# Music reviews
conn = sqlite3.connect('data/archive5/database.sqlite')
reviews = pd.read_sql('SELECT * FROM reviews', conn)
genres = pd.read_sql('SELECT * FROM genres', conn)
years_tbl = pd.read_sql('SELECT * FROM years', conn)
conn.close()
rev = reviews.merge(genres, on='reviewid', how='left').merge(years_tbl, on='reviewid', how='left', suffixes=('', '_album'))
music_year = rev.groupby('pub_year').agg(
    music_avg_score=('score', 'mean'),
    music_median_score=('score', 'median'),
    music_std_score=('score', 'std'),
    music_num_reviews=('reviewid', 'count'),
    music_best_new_pct=('best_new_music', 'mean'),
    music_min_score=('score', 'min'),
    music_max_score=('score', 'max'),
).reset_index()
music_year.columns = ['year'] + list(music_year.columns[1:])

# ---- Merge all on year ----
merged = aq_year.copy()
for df in [asylum_year, asylum_nat_year, chess_year, music_year]:
    merged = merged.merge(df, on='year', how='outer')

merged = merged.sort_values('year').reset_index(drop=True)
print(f"\nMerged shape: {merged.shape}")
print(f"Year range: {merged['year'].min()} - {merged['year'].max()}")
print(f"\nColumns: {merged.columns.tolist()}")

# ---- Identify which columns come from which dataset ----
dataset_tags = {
    'aq_': 'Air Quality',
    'asylum_total': 'Asylum Regions', 'asylum_avg': 'Asylum Regions', 'asylum_max': 'Asylum Regions', 'asylum_num_regions': 'Asylum Regions',
    'asylum_nat': 'Asylum Nationalities', 'asylum_admit': 'Asylum Nationalities', 'asylum_gender': 'Asylum Nationalities',
    'chess_': 'Chess',
    'music_': 'Music Reviews',
}

def get_dataset(col):
    for prefix, ds in dataset_tags.items():
        if col.startswith(prefix):
            return ds
    return 'unknown'

# ---- Compute ALL cross-dataset pairs ----
numeric_cols = [c for c in merged.columns if c != 'year' and merged[c].dtype in ['float64', 'int64']]

cross_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        c1, c2 = numeric_cols[i], numeric_cols[j]
        ds1, ds2 = get_dataset(c1), get_dataset(c2)
        if ds1 == ds2:
            continue  # skip within-dataset
        
        # Only use rows where both have data
        mask = merged[[c1, c2]].dropna()
        if len(mask) < 3:
            continue  # need at least 3 points
        
        r = mask[c1].corr(mask[c2])
        if not np.isnan(r):
            cross_pairs.append((ds1, c1, ds2, c2, r, len(mask)))

cross_pairs.sort(key=lambda x: abs(x[4]), reverse=True)

print(f"\n{'=' * 80}")
print(f"TOP 30 CROSS-DATASET CORRELATIONS (min 3 overlapping years)")
print(f"{'=' * 80}")
print(f"{'Rank':>4}  {'|r|':>6}  {'r':>7}  {'n':>3}  {'Dataset 1':<20} {'Column 1':<30} {'Dataset 2':<20} {'Column 2'}")
print("-" * 130)
for idx, (ds1, c1, ds2, c2, r, n) in enumerate(cross_pairs[:30], 1):
    print(f"{idx:>4}  {abs(r):>6.4f}  {r:>7.4f}  {n:>3}  {ds1:<20} {c1:<30} {ds2:<20} {c2}")

# ---- Highlight the winner ----
best = cross_pairs[0]
print(f"\n{'=' * 80}")
print(f"STRONGEST CROSS-DATASET CORRELATION:")
print(f"  {best[0]}: {best[1]}")
print(f"  {best[2]}: {best[3]}")
print(f"  r = {best[4]:.4f}  (n = {best[5]} overlapping years)")
print(f"{'=' * 80}")

# Show the actual data points for the top correlation
c1, c2 = best[1], best[3]
subset = merged[['year', c1, c2]].dropna()
print(f"\nData points:")
print(subset.to_string(index=False))
