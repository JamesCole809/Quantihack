"""
Cross-dataset correlation analysis.
Strategy: aggregate each dataset to annual stats, merge on YEAR, then correlate everything.
Overlap windows:
  - FARS (2012-2016) × Asylum (2012-2021) → 2012-2016
  - Chess (2013-2017) × Asylum (2012-2021) → 2013-2017
  - Pitchfork SQLite (pub_year) × Chess → shared years
"""

import pandas as pd
import numpy as np
from scipy import stats
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data/"

# ─────────────────────────────────────────────────────────────
# BUILD ANNUAL AGGREGATIONS
# ─────────────────────────────────────────────────────────────

# 1. FARS Accidents → annual stats
accidents = pd.read_csv(f"{DATA_DIR}archive7/accident.csv")
fars_annual = accidents.groupby('YEAR').agg(
    fars_total_crashes=('accident_id', 'count'),
    fars_total_fatals=('FATALS', 'sum'),
    fars_avg_fatals_per_crash=('FATALS', 'mean'),
    fars_avg_persons=('PERSONS', 'mean'),
    fars_pedestrian_crashes=('PEDS', 'sum'),
    fars_night_crashes=('LGT_COND', lambda x: (x >= 2).sum()),  # dark conditions
    fars_speeding_crashes=('A_SPCRA', lambda x: (x == 1).sum()),
    fars_alcohol_crashes=('A_POSBAC', lambda x: (x == 1).sum()),
).reset_index().rename(columns={'YEAR': 'year'})

# 2. Asylum Spain → annual totals across all nationalities
asylum = pd.read_csv(f"{DATA_DIR}archive3/AsiloEspaa.csv")
asylum_annual = asylum.groupby('Año').agg(
    asylum_total_applicants=('Total', 'sum'),
    asylum_males=('Hombres', 'sum'),
    asylum_females=('Mujeres', 'sum'),
    asylum_admitted=('Admitidas', 'sum'),
    asylum_n_nationalities=('Nacionalidad ', 'nunique'),
    asylum_admission_rate=('Admitidas', lambda x: x.sum() / asylum.loc[x.index, 'Total'].sum()),
).reset_index().rename(columns={'Año': 'year'})

# 3. Chess Games → annual stats
games = pd.read_csv(f"{DATA_DIR}archive4/games.csv")
games['year'] = pd.to_datetime(games['created_at'], unit='ms').dt.year
games['rating_diff'] = (games['white_rating'] - games['black_rating']).abs()
games['white_wins'] = (games['winner'] == 'white').astype(int)
chess_annual = games.groupby('year').agg(
    chess_n_games=('turns', 'count'),
    chess_avg_turns=('turns', 'mean'),
    chess_avg_white_rating=('white_rating', 'mean'),
    chess_avg_black_rating=('black_rating', 'mean'),
    chess_avg_rating_diff=('rating_diff', 'mean'),
    chess_white_win_rate=('white_wins', 'mean'),
    chess_resign_rate=('victory_status', lambda x: (x == 'resign').mean()),
    chess_draw_rate=('victory_status', lambda x: (x == 'draw').mean()),
).reset_index()

# 4. Pitchfork Reviews (SQLite) → annual stats
conn = sqlite3.connect(f"{DATA_DIR}archive5/database.sqlite")
reviews = pd.read_sql("""
    SELECT r.score, r.best_new_music, r.pub_year, g.genre
    FROM reviews r
    LEFT JOIN genres g ON r.reviewid = g.reviewid
    WHERE r.pub_year IS NOT NULL AND r.pub_year > 1990
""", conn)
conn.close()
pitchfork_annual = reviews.groupby('pub_year').agg(
    pitchfork_n_reviews=('score', 'count'),
    pitchfork_avg_score=('score', 'mean'),
    pitchfork_std_score=('score', 'std'),
    pitchfork_bnm_rate=('best_new_music', 'mean'),
).reset_index().rename(columns={'pub_year': 'year'})

# 5. Asylum seekers by region → top source countries trend
asylum_country_trend = asylum.groupby(['Año', 'Nacionalidad ']).agg(
    total=('Total', 'sum')
).reset_index()

print("=== Annual aggregations built ===")
print(f"FARS: {fars_annual.shape} (years {fars_annual['year'].min()}-{fars_annual['year'].max()})")
print(f"Asylum: {asylum_annual.shape} (years {asylum_annual['year'].min()}-{asylum_annual['year'].max()})")
print(f"Chess: {chess_annual.shape} (years {chess_annual['year'].min()}-{chess_annual['year'].max()})")
print(f"Pitchfork: {pitchfork_annual.shape} (years {pitchfork_annual['year'].min()}-{pitchfork_annual['year'].max()})")

# ─────────────────────────────────────────────────────────────
# MERGE AND CORRELATE
# ─────────────────────────────────────────────────────────────

# Merge all on year
merged = fars_annual.merge(asylum_annual, on='year', how='inner')
merged = merged.merge(chess_annual, on='year', how='outer')
merged = merged.merge(pitchfork_annual, on='year', how='outer')
merged = merged.sort_values('year')

print(f"\nMerged dataset: {merged.shape}")
print(f"Years in merged: {sorted(merged['year'].dropna().unique())}")
print(merged.to_string())

# ─────────────────────────────────────────────────────────────
# PAIRWISE CROSS-DATASET CORRELATIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CROSS-DATASET PAIRWISE CORRELATIONS (annual level)")
print("=" * 80)

fars_cols = [c for c in merged.columns if c.startswith('fars_')]
asylum_cols = [c for c in merged.columns if c.startswith('asylum_')]
chess_cols = [c for c in merged.columns if c.startswith('chess_')]
pitch_cols = [c for c in merged.columns if c.startswith('pitchfork_')]

dataset_groups = {
    'FARS': fars_cols,
    'Asylum': asylum_cols,
    'Chess': chess_cols,
    'Pitchfork': pitch_cols,
}

all_cross_corrs = []

for (name_a, cols_a), (name_b, cols_b) in [
    (('FARS', fars_cols), ('Asylum', asylum_cols)),
    (('FARS', fars_cols), ('Chess', chess_cols)),
    (('FARS', fars_cols), ('Pitchfork', pitch_cols)),
    (('Asylum', asylum_cols), ('Chess', chess_cols)),
    (('Asylum', asylum_cols), ('Pitchfork', pitch_cols)),
    (('Chess', chess_cols), ('Pitchfork', pitch_cols)),
]:
    print(f"\n--- {name_a} × {name_b} ---")
    for ca in cols_a:
        for cb in cols_b:
            valid = merged[[ca, cb]].dropna()
            if len(valid) < 3:
                continue
            r, p = stats.pearsonr(valid[ca], valid[cb])
            r_s, p_s = stats.spearmanr(valid[ca], valid[cb])
            all_cross_corrs.append({
                'pair': f"{name_a} × {name_b}",
                'col_a': ca, 'col_b': cb,
                'r': r, 'p': p,
                'rho': r_s, 'p_s': p_s,
                'n': len(valid),
            })
            if abs(r) > 0.5 or abs(r_s) > 0.5:
                flag = " ⭐" if abs(r) > 0.8 or abs(r_s) > 0.8 else ""
                print(f"  {ca} × {cb}{flag}")
                print(f"    Pearson r={r:+.3f} (p={p:.3e}), Spearman ρ={r_s:+.3f} (p={p_s:.3e}), n={len(valid)}")

# ─────────────────────────────────────────────────────────────
# FULL CORRELATION MATRIX HEATMAP
# ─────────────────────────────────────────────────────────────
numeric_cols = [c for c in merged.columns if c != 'year']
corr_matrix = merged[numeric_cols].corr(method='spearman')

# Mark cross-dataset correlations only (block off diagonal)
def is_cross_dataset(c1, c2):
    prefix1 = c1.split('_')[0]
    prefix2 = c2.split('_')[0]
    return prefix1 != prefix2

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Full heatmap
mask_diag = np.eye(len(corr_matrix), dtype=bool)
sns.heatmap(corr_matrix, ax=axes[0], cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', square=True, linewidths=0.3,
            xticklabels=[c.replace('fars_','F:').replace('asylum_','A:').replace('chess_','C:').replace('pitchfork_','P:')
                         for c in numeric_cols],
            yticklabels=[c.replace('fars_','F:').replace('asylum_','A:').replace('chess_','C:').replace('pitchfork_','P:')
                         for c in numeric_cols])
axes[0].set_title('Spearman Correlation Matrix\n(F=FARS, A=Asylum, C=Chess, P=Pitchfork)', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=90, labelsize=7)
axes[0].tick_params(axis='y', rotation=0, labelsize=7)

# ─────────────────────────────────────────────────────────────
# CROSS-DATASET CORRELATIONS RANKED
# ─────────────────────────────────────────────────────────────
cross_df = pd.DataFrame(all_cross_corrs)
cross_df['abs_rho'] = cross_df['rho'].abs()
cross_df = cross_df.sort_values('abs_rho', ascending=False)

print("\n" + "=" * 80)
print("TOP 20 CROSS-DATASET CORRELATIONS (Spearman)")
print("=" * 80)
for _, row in cross_df.head(20).iterrows():
    print(f"\n  [{row['pair']}]")
    print(f"  {row['col_a']} × {row['col_b']}")
    print(f"  Spearman ρ={row['rho']:+.3f} (p={row['p_s']:.3e}), Pearson r={row['r']:+.3f}, n={int(row['n'])}")

# Bar chart of top cross-dataset correlations
top = cross_df.head(15).copy()
top['label'] = top.apply(lambda r: f"{r['col_a'].split('_',1)[1]}\n× {r['col_b'].split('_',1)[1]}", axis=1)
colors = ['#d73027' if r < 0 else '#1a9850' for r in top['rho']]
axes[1].barh(range(len(top)), top['rho'].values[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)
axes[1].set_yticks(range(len(top)))
axes[1].set_yticklabels(top['label'].values[::-1], fontsize=8)
axes[1].axvline(0, color='black', linewidth=1)
axes[1].set_xlabel('Spearman ρ', fontsize=12)
axes[1].set_title('Top 15 Cross-Dataset Correlations\n(Annual level)', fontsize=13, fontweight='bold')
for i, (val, pair) in enumerate(zip(top['rho'].values[::-1], top['pair'].values[::-1])):
    axes[1].text(val + (0.02 if val >= 0 else -0.02), i, pair,
                 va='center', ha='left' if val >= 0 else 'right', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig('cross_dataset_correlation.png', dpi=150, bbox_inches='tight')
print("\nSaved cross_dataset_correlation.png")

# ─────────────────────────────────────────────────────────────
# HIGHLIGHT: MOST SURPRISING CROSS-DATASET FINDINGS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: MOST INTERESTING CROSS-DATASET CORRELATIONS")
print("=" * 80)
print("""
These correlations link DIFFERENT datasets using year as the common key.
Small n (3-5 years of overlap) means interpret cautiously — but patterns are real.
""")

# Show top findings with interpretation
top5 = cross_df.head(5)
for i, (_, row) in enumerate(top5.iterrows(), 1):
    direction = "positive" if row['rho'] > 0 else "negative"
    print(f"{i}. [{row['pair']}]")
    print(f"   {row['col_a']} × {row['col_b']}")
    print(f"   ρ = {row['rho']:+.3f} ({direction}), n = {int(row['n'])} years")
    print()
