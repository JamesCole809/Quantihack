import pandas as pd
import numpy as np
import sqlite3
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CROSS-DATASET CORRELATION ANALYSIS")
print("=" * 80)

# Load all datasets
datasets = {}

# Archive 1 & 2: Air quality data
for i, name in [(1, 'archive'), (2, 'archive2')]:
    df = pd.read_csv(f'data/{name}/data.csv')
    datasets[f'air_quality_{i}'] = df
    print(f"\nair_quality_{i}: {df.shape} - cols: {df.columns.tolist()}")

    spo = pd.read_csv(f'data/{name}/spo.csv')
    datasets[f'spo_{i}'] = spo
    print(f"spo_{i}: {spo.shape} - cols: {spo.columns.tolist()[:10]}")

    zone = pd.read_csv(f'data/{name}/zone.csv')
    datasets[f'zone_{i}'] = zone
    print(f"zone_{i}: {zone.shape} - cols: {zone.columns.tolist()[:10]}")

# Archive 3: Asylum data
asilo_ca = pd.read_csv('data/archive3/AsiloCA.csv')
asilo_es = pd.read_csv('data/archive3/AsiloEspaa.csv')
datasets['asylum_regions'] = asilo_ca
datasets['asylum_nationalities'] = asilo_es
print(f"\nasylum_regions: {asilo_ca.shape} - cols: {asilo_ca.columns.tolist()}")
print(f"asylum_nationalities: {asilo_es.shape} - cols: {asilo_es.columns.tolist()}")

# Archive 4: Chess games
games = pd.read_csv('data/archive4/games.csv')
datasets['chess_games'] = games
print(f"\nchess_games: {games.shape} - cols: {games.columns.tolist()}")

# Archive 5: Music reviews
conn = sqlite3.connect('data/archive5/database.sqlite')
reviews = pd.read_sql('SELECT * FROM reviews', conn)
genres = pd.read_sql('SELECT * FROM genres', conn)
years = pd.read_sql('SELECT * FROM years', conn)
reviews_full = reviews.merge(genres, on='reviewid', how='left').merge(years, on='reviewid', how='left', suffixes=('', '_album'))
datasets['music_reviews'] = reviews_full
conn.close()
print(f"\nmusic_reviews: {reviews_full.shape} - cols: {reviews_full.columns.tolist()}")

# Archive 6: Blood cells
blood = pd.read_csv('data/archive6/blood_cell_anomaly_detection.csv')
datasets['blood_cells'] = blood
print(f"\nblood_cells: {blood.shape} - cols: {blood.columns.tolist()[:15]}...")

cyto = pd.read_csv('data/archive6/cytodiffusion_benchmark_scores.csv')
datasets['cyto_benchmark'] = cyto
print(f"cyto_benchmark: {cyto.shape} - cols: {cyto.columns.tolist()}")

# ============================================================
# WITHIN-DATASET CORRELATIONS
# ============================================================
print("\n" + "=" * 80)
print("WITHIN-DATASET CORRELATIONS (top pairs per dataset)")
print("=" * 80)

all_correlations = []

for name, df in datasets.items():
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        continue
    
    corr = numeric.corr()
    # Get upper triangle pairs
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            c1, c2 = corr.columns[i], corr.columns[j]
            val = corr.iloc[i, j]
            if not np.isnan(val) and abs(val) > 0.01:
                pairs.append((c1, c2, val))
                all_correlations.append((name, c1, c2, val))
    
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if pairs:
        print(f"\n--- {name} ---")
        for c1, c2, val in pairs[:5]:
            print(f"  {c1} <-> {c2}: r={val:.4f}")

# ============================================================
# CROSS-DATASET CORRELATIONS (by year/time)
# ============================================================
print("\n" + "=" * 80)
print("CROSS-DATASET CORRELATIONS (linked by year)")
print("=" * 80)

# Asylum by year
asylum_by_year = asilo_ca.groupby('Año')['Solicitantes'].sum().reset_index()
asylum_by_year.columns = ['year', 'total_asylum_requests']

asylum_nat_by_year = asilo_es.groupby('Año')['Total'].sum().reset_index()
asylum_nat_by_year.columns = ['year', 'total_asylum_nationality']

# Music reviews by year (publication year)
music_by_year = reviews_full.groupby('pub_year').agg(
    avg_score=('score', 'mean'),
    num_reviews=('reviewid', 'count'),
    best_new_pct=('best_new_music', 'mean')
).reset_index()
music_by_year.columns = ['year', 'avg_music_score', 'num_music_reviews', 'best_new_music_pct']

# Chess: extract year from created_at (epoch ms)
games['year'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce').dt.year
chess_by_year = games.groupby('year').agg(
    avg_white_rating=('white_rating', 'mean'),
    avg_black_rating=('black_rating', 'mean'),
    avg_turns=('turns', 'mean'),
    num_games=('id', 'count')
).reset_index()

# Air quality by year
aq = datasets['air_quality_1'].copy()
aq['year'] = pd.to_datetime(aq['day'], errors='coerce').dt.year
aq_by_year = aq.groupby('year').agg(
    avg_pollutant_qty=('avg_daily_qqty', 'mean'),
    num_measurements=('avg_daily_qqty', 'count')
).reset_index()

# Merge all yearly aggregates
yearly = asylum_by_year.copy()
for df in [asylum_nat_by_year, music_by_year, chess_by_year, aq_by_year]:
    yearly = yearly.merge(df, on='year', how='outer')

print(f"\nYearly merged dataset: {yearly.shape}")
print(yearly.to_string())

numeric_yearly = yearly.select_dtypes(include=[np.number]).drop(columns=['year'], errors='ignore')
if numeric_yearly.shape[1] >= 2:
    corr = numeric_yearly.corr()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            c1, c2 = corr.columns[i], corr.columns[j]
            val = corr.iloc[i, j]
            if not np.isnan(val):
                pairs.append((c1, c2, val))
    
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\nCross-dataset yearly correlations (sorted by |r|):")
    for c1, c2, val in pairs:
        print(f"  {c1} <-> {c2}: r={val:.4f}")

# ============================================================
# OVERALL STRONGEST CORRELATIONS
# ============================================================
print("\n" + "=" * 80)
print("TOP 20 STRONGEST CORRELATIONS ACROSS ALL DATASETS")
print("=" * 80)

# Add cross-dataset correlations
for c1, c2, val in pairs:
    all_correlations.append(('cross_dataset_yearly', c1, c2, val))

# Sort by absolute value, exclude trivial (r=1.0 self or near-duplicates)
all_correlations = [(ds, c1, c2, v) for ds, c1, c2, v in all_correlations if abs(v) < 0.9999]
all_correlations.sort(key=lambda x: abs(x[3]), reverse=True)

for ds, c1, c2, val in all_correlations[:20]:
    print(f"  [{ds}] {c1} <-> {c2}: r={val:.4f}")

print("\n" + "=" * 80)
print(f"STRONGEST OVERALL: [{all_correlations[0][0]}] {all_correlations[0][1]} <-> {all_correlations[0][2]}: r={all_correlations[0][3]:.4f}")
print("=" * 80)
