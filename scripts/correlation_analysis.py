import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data/"

# Load all CSV datasets
datasets = {}

# 1. Air Quality France
aq = pd.read_csv(f"{DATA_DIR}Air Quality Data France/data.csv")
aq_poll = pd.read_csv(f"{DATA_DIR}Air Quality Data France/pollutant.csv")
aq_zone = pd.read_csv(f"{DATA_DIR}Air Quality Data France/zone.csv")
datasets["Air Quality France"] = aq

# 2. Pandemics
pandemics = pd.read_csv(f"{DATA_DIR}archive (1)/Historical_Pandemic_Epidemic_Dataset.csv")
datasets["Pandemics"] = pandemics

# 3. FARS Accidents
accidents = pd.read_csv(f"{DATA_DIR}archive7/accident.csv")
datasets["FARS Accidents"] = accidents

# 4. Chess Games
games = pd.read_csv(f"{DATA_DIR}archive4/games.csv")
datasets["Chess Games"] = games

# 5. Blood Cell Anomaly
blood = pd.read_csv(f"{DATA_DIR}archive6/blood_cell_anomaly_detection.csv")
datasets["Blood Cell Anomaly"] = blood

# 6. Asylum Seekers
asylum_ca = pd.read_csv(f"{DATA_DIR}archive3/AsiloCA.csv")
asylum_es = pd.read_csv(f"{DATA_DIR}archive3/AsiloEspaa.csv")
datasets["Asylum CA"] = asylum_ca
datasets["Asylum Spain"] = asylum_es

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
for name, df in datasets.items():
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n{name}: {df.shape[0]} rows x {df.shape[1]} cols | Numeric cols: {len(num_cols)}")
    print(f"  Numeric: {num_cols[:10]}{'...' if len(num_cols)>10 else ''}")

# ============================================================
# WITHIN-DATASET CORRELATIONS
# ============================================================
print("\n" + "=" * 80)
print("TOP WITHIN-DATASET CORRELATIONS (unexpected/strong)")
print("=" * 80)

all_corrs = []

for name, df in datasets.items():
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if num_df.shape[1] < 2:
        continue

    # Sample if too large
    if len(num_df) > 5000:
        num_df = num_df.sample(5000, random_state=42)

    for c1, c2 in combinations(num_df.columns, 2):
        valid = num_df[[c1, c2]].dropna()
        if len(valid) < 30:
            continue
        # Skip trivially related columns (same prefix, index-like)
        if c1.lower() in ('index', 'id') or c2.lower() in ('index', 'id'):
            continue

        r, p = stats.pearsonr(valid[c1], valid[c2])
        if p < 0.001 and abs(r) > 0.3:
            all_corrs.append({
                'dataset': name,
                'col1': c1,
                'col2': c2,
                'r': r,
                'p': p,
                'n': len(valid),
                'type': 'within'
            })

all_corrs.sort(key=lambda x: abs(x['r']), reverse=True)

# Print top within-dataset correlations (skip obvious ones, show interesting)
seen = set()
count = 0
for c in all_corrs:
    if c['type'] != 'within':
        continue
    key = (c['dataset'], frozenset([c['col1'], c['col2']]))
    if key in seen:
        continue
    seen.add(key)
    print(f"\n  [{c['dataset']}] {c['col1']} vs {c['col2']}")
    print(f"    r = {c['r']:.4f}, p = {c['p']:.2e}, n = {c['n']}")
    count += 1
    if count >= 20:
        break

# ============================================================
# CROSS-DATASET CORRELATIONS
# ============================================================
print("\n" + "=" * 80)
print("CROSS-DATASET CORRELATIONS")
print("=" * 80)

cross_corrs = []

# --- Pandemics: aggregate by century or era for temporal analysis ---
# Internal pandemic analysis: does spread predict death rate?
print("\n--- Pandemic Internal Analysis ---")
p = pandemics.copy()
for c1, c2 in [
    ('Spread_Score', 'Case_Fatality_Rate_Pct'),
    ('Spread_Score', 'Estimated_Deaths'),
    ('Duration_Years', 'Case_Fatality_Rate_Pct'),
    ('Economic_Impact_Billion_USD', 'Estimated_Deaths'),
    ('Continents_Affected', 'Estimated_Deaths'),
    ('Start_Year', 'Case_Fatality_Rate_Pct'),
    ('Duration_Years', 'Estimated_Deaths'),
    ('Start_Year', 'Economic_Impact_Billion_USD'),
]:
    valid = p[[c1, c2]].dropna()
    if len(valid) < 5:
        continue
    r, pval = stats.pearsonr(valid[c1], valid[c2])
    r_s, pval_s = stats.spearmanr(valid[c1], valid[c2])
    print(f"  {c1} vs {c2}: Pearson r={r:.3f} (p={pval:.3e}), Spearman rho={r_s:.3f} (p={pval_s:.3e}), n={len(valid)}")

# --- Chess: rating vs game outcome ---
print("\n--- Chess Games Analysis ---")
g = games.copy()
g['rating_diff'] = g['white_rating'] - g['black_rating']
g['white_wins'] = (g['winner'] == 'white').astype(int)
g['avg_rating'] = (g['white_rating'] + g['black_rating']) / 2

for c1, c2 in [
    ('rating_diff', 'white_wins'),
    ('rating_diff', 'turns'),
    ('avg_rating', 'turns'),
    ('white_rating', 'turns'),
]:
    valid = g[[c1, c2]].dropna()
    r, pval = stats.pearsonr(valid[c1], valid[c2])
    print(f"  {c1} vs {c2}: r={r:.3f} (p={pval:.3e}), n={len(valid)}")

# --- Blood cells: anomaly score vs clinical measurements ---
print("\n--- Blood Cell Anomaly Analysis ---")
b = blood.copy()
for c1, c2 in [
    ('cytodiffusion_anomaly_score', 'wbc_count_per_ul'),
    ('cytodiffusion_anomaly_score', 'hemoglobin_g_dl'),
    ('cytodiffusion_anomaly_score', 'chromatin_density'),
    ('cytodiffusion_anomaly_score', 'circularity'),
    ('anomaly_label', 'wbc_count_per_ul'),
    ('anomaly_label', 'platelet_count_per_ul'),
    ('cell_diameter_um', 'cell_area_px'),
    ('wbc_count_per_ul', 'hemoglobin_g_dl'),
    ('rbc_count_millions_per_ul', 'hemoglobin_g_dl'),
    ('hematocrit_pct', 'hemoglobin_g_dl'),
]:
    valid = b[[c1, c2]].dropna()
    if len(valid) < 30:
        continue
    r, pval = stats.pearsonr(valid[c1], valid[c2])
    print(f"  {c1} vs {c2}: r={r:.3f} (p={pval:.3e}), n={len(valid)}")

# --- FARS Accidents: conditions vs fatalities ---
print("\n--- FARS Accidents Analysis ---")
a = accidents.copy()
for c1, c2 in [
    ('PERSONS', 'FATALS'),
    ('VE_TOTAL', 'FATALS'),
    ('VE_TOTAL', 'PERSONS'),
    ('HOUR', 'FATALS'),
    ('LGT_COND', 'FATALS'),
    ('MONTH', 'FATALS'),
]:
    valid = a[[c1, c2]].dropna()
    if len(valid) < 30:
        continue
    r, pval = stats.pearsonr(valid[c1], valid[c2])
    print(f"  {c1} vs {c2}: r={r:.3f} (p={pval:.3e}), n={len(valid)}")

# --- CROSS-DATASET: Pandemics over time vs pattern analysis ---
print("\n--- Cross-Dataset: Pandemic Severity Over Time ---")
# Are pandemics getting less deadly over centuries?
p = pandemics.copy()
r, pval = stats.spearmanr(p['Start_Year'].dropna(), p['Case_Fatality_Rate_Pct'].dropna()[:len(p['Start_Year'].dropna())])
print(f"  Start_Year vs Case_Fatality_Rate: Spearman rho={r:.3f} (p={pval:.3e})")

# Group pandemics by era
era_stats = p.groupby('Era').agg({
    'Case_Fatality_Rate_Pct': 'mean',
    'Estimated_Deaths': 'mean',
    'Duration_Years': 'mean',
    'Spread_Score': 'mean',
    'Economic_Impact_Billion_USD': 'mean',
}).round(2)
print(f"\n  Pandemic stats by Era:\n{era_stats.to_string()}")

# ============================================================
# MOST SURPRISING / INTERESTING FINDINGS
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY: MOST INTERESTING CORRELATIONS FOUND")
print("=" * 80)

# Collect everything into a final ranked list
final = []
for c in all_corrs:
    # Skip trivially obvious correlations
    trivial_pairs = {
        frozenset(['cell_area_px', 'perimeter_px']),
        frozenset(['hematocrit_pct', 'hemoglobin_g_dl']),
        frozenset(['VE_TOTAL', 'VE_FORMS']),
        frozenset(['LATITUDE', 'LONGITUD']),
    }
    pair = frozenset([c['col1'], c['col2']])
    if pair in trivial_pairs:
        continue
    # Interestingness = high |r| + non-obvious
    final.append(c)

final.sort(key=lambda x: abs(x['r']), reverse=True)

print("\nTop 15 strongest non-trivial correlations across all datasets:\n")
for i, c in enumerate(final[:15], 1):
    direction = "positive" if c['r'] > 0 else "negative"
    print(f"  {i:2d}. [{c['dataset']}] {c['col1']} <-> {c['col2']}")
    print(f"      r = {c['r']:+.4f} ({direction}), p = {c['p']:.2e}, n = {c['n']}")
