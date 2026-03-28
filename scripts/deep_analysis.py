"""
Deep multi-dataset analysis with 9 focused panels.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data/"

# ── Load datasets ──────────────────────────────────────────────
pandemics = pd.read_csv(f"{DATA_DIR}archive (1)/Historical_Pandemic_Epidemic_Dataset.csv")
blood = pd.read_csv(f"{DATA_DIR}archive6/blood_cell_anomaly_detection.csv")
games = pd.read_csv(f"{DATA_DIR}archive4/games.csv")
accidents = pd.read_csv(f"{DATA_DIR}archive7/accident.csv")
asylum_raw = pd.read_csv(f"{DATA_DIR}archive3/AsiloEspaa.csv")
conn = sqlite3.connect(f"{DATA_DIR}archive5/database.sqlite")
reviews = pd.read_sql("""
    SELECT r.score, r.best_new_music, r.pub_year, g.genre
    FROM reviews r LEFT JOIN genres g ON r.reviewid = g.reviewid
    WHERE r.pub_year IS NOT NULL AND r.pub_year > 1999
""", conn)
conn.close()

fig = plt.figure(figsize=(22, 28))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)
fig.suptitle("Deep Multi-Dataset Analysis", fontsize=17, fontweight='bold', y=1.005)

# ════════════════════════════════════════════════════════════════
# 1. PANDEMICS: Transmission type vs fatality rate (violin)
# ════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
p = pandemics.copy()
# Simplify transmission
def simplify_tx(t):
    if 'Airborne' in str(t) or 'Droplet' in str(t): return 'Airborne/\nDroplet'
    if 'Vector' in str(t): return 'Vector'
    if 'Waterborne' in str(t) or 'Fecal' in str(t): return 'Waterborne/\nFecal-Oral'
    if 'Contact' in str(t): return 'Contact'
    if 'Sexual' in str(t) or 'Blood' in str(t): return 'Sexual/\nBlood'
    return 'Other'
p['tx_simple'] = p['Primary_Transmission'].apply(simplify_tx)
tx_order = p.groupby('tx_simple')['Case_Fatality_Rate_Pct'].median().sort_values(ascending=False).index
tx_data = [p[p['tx_simple']==tx]['Case_Fatality_Rate_Pct'].dropna().values for tx in tx_order]
bp = ax1.boxplot(tx_data, labels=tx_order, patch_artist=True, notch=False)
colors_bp = ['#d73027','#fc8d59','#fee090','#91bfdb','#4575b4']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
# overlay points
for i, (tx, d) in enumerate(zip(tx_order, tx_data), 1):
    ax1.scatter(np.random.normal(i, 0.06, len(d)), d, alpha=0.7, s=30, color='black', zorder=3)
ax1.set_ylabel('Case Fatality Rate (%)', fontsize=10)
ax1.set_title('Pandemic: Which Transmission\nMode is Most Deadly?', fontsize=11, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)
for label in ax1.get_xticklabels():
    label.set_fontsize(8)

# ════════════════════════════════════════════════════════════════
# 2. PANDEMICS: Containment method success (fatality rate)
# ════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
def simplify_contain(c):
    if 'Vaccine' in str(c): return 'Vaccine'
    if 'Antibiotic' in str(c): return 'Antibiotics'
    if 'Quarantine' in str(c): return 'Quarantine'
    if 'Sanitation' in str(c): return 'Sanitation'
    if 'Natural' in str(c): return 'Natural Decline'
    return 'Other'
p['contain_simple'] = p['Containment_Method'].apply(simplify_contain)
contain_stats = p.groupby('contain_simple').agg(
    median_cfr=('Case_Fatality_Rate_Pct','median'),
    mean_cfr=('Case_Fatality_Rate_Pct','mean'),
    n=('Case_Fatality_Rate_Pct','count'),
    mean_deaths=('Estimated_Deaths','mean'),
).sort_values('mean_cfr')
bars = ax2.barh(contain_stats.index, contain_stats['mean_cfr'],
                color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(contain_stats))),
                edgecolor='black', linewidth=0.7)
for bar, (_, row) in zip(bars, contain_stats.iterrows()):
    ax2.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
             f"n={int(row['n'])}", va='center', fontsize=8)
ax2.set_xlabel('Mean Case Fatality Rate (%)', fontsize=10)
ax2.set_title('Pandemic: Containment Method\nvs Fatality Rate', fontsize=11, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)

# ════════════════════════════════════════════════════════════════
# 3. PANDEMICS: Bacterial vs Viral — spread & lethality bubble chart
# ════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[0, 2])
for ptype, color, marker in [('Virus','#d73027','o'), ('Bacteria','#4575b4','s')]:
    sub = p[p['Pathogen_Type']==ptype]
    sizes = np.sqrt(sub['Duration_Years'].clip(1)) * 40
    ax3.scatter(sub['Spread_Score'], sub['Case_Fatality_Rate_Pct'],
                s=sizes, c=color, alpha=0.65, edgecolors='black', linewidths=0.5,
                marker=marker, label=ptype, zorder=3)
ax3.set_xlabel('Spread Score', fontsize=10)
ax3.set_ylabel('Case Fatality Rate (%)', fontsize=10)
ax3.set_title('Virus vs Bacteria:\nSpread vs Lethality\n(bubble = duration)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
r_v, _ = stats.spearmanr(p[p['Pathogen_Type']=='Virus']['Spread_Score'],
                          p[p['Pathogen_Type']=='Virus']['Case_Fatality_Rate_Pct'])
r_b, _ = stats.spearmanr(p[p['Pathogen_Type']=='Bacteria']['Spread_Score'],
                          p[p['Pathogen_Type']=='Bacteria']['Case_Fatality_Rate_Pct'])
ax3.text(0.03, 0.97, f"Virus ρ={r_v:.2f}\nBacteria ρ={r_b:.2f}",
         transform=ax3.transAxes, fontsize=8.5, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

# ════════════════════════════════════════════════════════════════
# 4. BLOOD CELLS: PCA — disease category clusters
# ════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 0])
morph_cols = ['cell_diameter_um','nucleus_area_pct','chromatin_density','cytoplasm_ratio',
              'circularity','eccentricity','granularity_score','lobularity_score',
              'membrane_smoothness','stain_intensity','cytodiffusion_anomaly_score']
b = blood[morph_cols + ['disease_category']].dropna()
scaler = StandardScaler()
X = scaler.fit_transform(b[morph_cols])
pca = PCA(n_components=2)
coords = pca.fit_transform(X)
b_pca = b.copy()
b_pca['PC1'] = coords[:,0]
b_pca['PC2'] = coords[:,1]

disease_colors = {
    'Normal_WBC':'#74add1','Normal_RBC':'#abd9e9','Normal_Platelet':'#e0f3f8',
    'Leukemia':'#d73027','Anemia':'#f46d43','Infection':'#fdae61',
    'Sickle_Cell_Anemia':'#9e0142','Artefact':'#888888'
}
for disease, color in disease_colors.items():
    sub = b_pca[b_pca['disease_category']==disease]
    if len(sub) == 0: continue
    ax4.scatter(sub['PC1'], sub['PC2'], c=color, label=disease, alpha=0.5, s=12, zorder=2)
ax4.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
ax4.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
ax4.set_title('Blood Cells: PCA of Morphology\nColoured by Disease', fontsize=11, fontweight='bold')
ax4.legend(fontsize=7, loc='upper right', markerscale=2)
ax4.grid(True, alpha=0.2)

# ════════════════════════════════════════════════════════════════
# 5. BLOOD CELLS: Which features discriminate Leukemia vs Normal?
# ════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 1])
features = ['cell_diameter_um','nucleus_area_pct','chromatin_density','cytoplasm_ratio',
            'circularity','eccentricity','granularity_score','lobularity_score',
            'membrane_smoothness','stain_intensity']
leuk = blood[blood['disease_category']=='Leukemia'][features]
norm = blood[blood['disease_category']=='Normal_WBC'][features]

results = []
for f in features:
    stat, pv_mwu = stats.mannwhitneyu(leuk[f].dropna(), norm[f].dropna(), alternative='two-sided')
    # effect size: rank biserial correlation
    n1, n2 = leuk[f].dropna().shape[0], norm[f].dropna().shape[0]
    r_effect = 1 - (2*stat)/(n1*n2)
    results.append({'feature': f, 'effect_size': r_effect, 'p': pv_mwu})
res_df = pd.DataFrame(results).sort_values('effect_size', key=abs, ascending=False)

colors_bar = ['#d73027' if v > 0 else '#4575b4' for v in res_df['effect_size']]
ax5.barh(res_df['feature'], res_df['effect_size'], color=colors_bar, edgecolor='black', linewidth=0.5)
ax5.axvline(0, color='black', linewidth=1)
ax5.set_xlabel('Effect Size (rank-biserial r)\n+ = higher in Leukemia', fontsize=9)
ax5.set_title('Blood Cells: Feature Differences\nLeukemia vs Normal WBC', fontsize=11, fontweight='bold')
ax5.grid(True, axis='x', alpha=0.3)
for i, (_, row) in enumerate(res_df.iterrows()):
    ax5.text(row['effect_size'] + (0.01 if row['effect_size']>0 else -0.01),
             i, f"p={row['p']:.1e}", va='center',
             ha='left' if row['effect_size']>0 else 'right', fontsize=7)

# ════════════════════════════════════════════════════════════════
# 6. CHESS: Top openings — win rate & avg game length
# ════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[1, 2])
g = games.copy()
g['white_wins'] = (g['winner']=='white').astype(int)
g['black_wins'] = (g['winner']=='black').astype(int)
# Simplify opening name to first two words
g['opening_short'] = g['opening_name'].str.split(':').str[0].str.strip()
top_openings = g['opening_short'].value_counts().head(12).index
opening_stats = g[g['opening_short'].isin(top_openings)].groupby('opening_short').agg(
    n=('turns','count'),
    avg_turns=('turns','mean'),
    white_win_rate=('white_wins','mean'),
    black_win_rate=('black_wins','mean'),
).sort_values('white_win_rate', ascending=False)

y = range(len(opening_stats))
ax6.barh(y, opening_stats['white_win_rate'], color='#f7f7f7', edgecolor='#333', linewidth=0.7,
         label='White win rate', height=0.6)
ax6.barh(y, -opening_stats['black_win_rate'], color='#333333', edgecolor='#333', linewidth=0.7,
         label='Black win rate', height=0.6)
ax6.set_yticks(y)
ax6.set_yticklabels(opening_stats.index, fontsize=8)
ax6.axvline(0, color='black', linewidth=1.2)
ax6.set_xlim(-0.65, 0.65)
ax6.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax6.set_xticklabels(['50%','25%','0','25%','50%'], fontsize=8)
ax6.set_title('Chess Openings:\nWin Rate (White vs Black)', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8, loc='lower right')
ax6.grid(True, axis='x', alpha=0.3)

# ════════════════════════════════════════════════════════════════
# 7. CHESS: Opening ply depth vs game outcome
# ════════════════════════════════════════════════════════════════
ax7 = fig.add_subplot(gs[2, 0])
g['draw'] = (g['winner']=='draw').astype(int)
ply_bins = pd.cut(g['opening_ply'], bins=[0,4,8,12,16,30], labels=['1-4','5-8','9-12','13-16','17+'])
ply_stats = g.groupby(ply_bins, observed=True).agg(
    avg_turns=('turns','mean'),
    white_win_rate=('white_wins','mean'),
    draw_rate=('draw','mean'),
    n=('turns','count'),
).reset_index()
x = range(len(ply_stats))
ax7.plot(x, ply_stats['avg_turns'], 'o-', color='steelblue', linewidth=2, markersize=8, label='Avg turns')
ax7b = ax7.twinx()
ax7b.plot(x, ply_stats['white_win_rate'], 's--', color='darkorange', linewidth=2, markersize=7, label='White win rate')
ax7b.plot(x, ply_stats['draw_rate'], '^--', color='green', linewidth=2, markersize=7, label='Draw rate')
ax7.set_xticks(x)
ax7.set_xticklabels(ply_stats['opening_ply'].astype(str), fontsize=9)
ax7.set_xlabel('Opening Depth (ply)', fontsize=10)
ax7.set_ylabel('Avg Game Length (turns)', fontsize=10, color='steelblue')
ax7b.set_ylabel('Win/Draw Rate', fontsize=10)
ax7.set_title('Chess: Opening Depth\nvs Game Outcome', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.25)
lines1, labels1 = ax7.get_legend_handles_labels()
lines2, labels2 = ax7b.get_legend_handles_labels()
ax7.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper left')

# ════════════════════════════════════════════════════════════════
# 8. FARS: Crash fatality by hour of day
# ════════════════════════════════════════════════════════════════
ax8 = fig.add_subplot(gs[2, 1])
a = accidents[accidents['HOUR'] < 24].copy()
hour_stats = a.groupby('HOUR').agg(
    n_crashes=('FATALS','count'),
    total_fatals=('FATALS','sum'),
    avg_fatals=('FATALS','mean'),
    alcohol_rate=('A_POSBAC', lambda x: (x==1).mean()),
).reset_index()
ax8.fill_between(hour_stats['HOUR'], hour_stats['n_crashes'],
                 alpha=0.3, color='steelblue', label='Crash count')
ax8.plot(hour_stats['HOUR'], hour_stats['n_crashes'], color='steelblue', linewidth=2)
ax8b = ax8.twinx()
ax8b.plot(hour_stats['HOUR'], hour_stats['alcohol_rate']*100, 'o-',
          color='crimson', linewidth=2, markersize=5, label='Alcohol rate (%)')
ax8.set_xlabel('Hour of Day', fontsize=10)
ax8.set_ylabel('Number of Fatal Crashes', fontsize=10, color='steelblue')
ax8b.set_ylabel('Alcohol-Involved (%)', fontsize=10, color='crimson')
ax8.set_xticks(range(0,24,2))
ax8.set_title('FARS: Fatal Crashes by Hour\n& Alcohol Involvement', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.25)
lines1, labels1 = ax8.get_legend_handles_labels()
lines2, labels2 = ax8b.get_legend_handles_labels()
ax8.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper left')

# ════════════════════════════════════════════════════════════════
# 9. PITCHFORK: Genre score trends over time (line)
# ════════════════════════════════════════════════════════════════
ax9 = fig.add_subplot(gs[2, 2])
main_genres = ['rock','electronic','rap','pop/r&b','experimental','metal','folk/country']
genre_colors = {
    'rock':'#d73027','electronic':'#4575b4','rap':'#fdae61',
    'pop/r&b':'#9970ab','experimental':'#1a9850','metal':'#333333',
    'folk/country':'#8c510a'
}
reviews_filtered = reviews[reviews['genre'].isin(main_genres)]
genre_year = reviews_filtered.groupby(['pub_year','genre'])['score'].mean().reset_index()
for genre in main_genres:
    sub = genre_year[genre_year['genre']==genre]
    if len(sub) < 5: continue
    ax9.plot(sub['pub_year'], sub['score'], linewidth=2, alpha=0.85,
             color=genre_colors[genre], label=genre)
ax9.set_xlabel('Year', fontsize=10)
ax9.set_ylabel('Avg Pitchfork Score', fontsize=10)
ax9.set_title('Pitchfork: Genre Scores\nOver Time (1999–2017)', fontsize=11, fontweight='bold')
ax9.legend(fontsize=8, loc='lower left')
ax9.grid(True, alpha=0.25)
ax9.set_ylim(5.5, 8.5)

# ════════════════════════════════════════════════════════════════
# 10-12. CROSS-DATASET DETRENDED: Remove year effect, check residuals
# ════════════════════════════════════════════════════════════════
accidents_ann = accidents.groupby('YEAR').agg(
    alcohol_crashes=('A_POSBAC', lambda x: (x==1).sum()),
    ped_crashes=('PEDS','sum'),
).reset_index().rename(columns={'YEAR':'year'})
games['year_g'] = pd.to_datetime(games['created_at'], unit='ms').dt.year
games['rdiff'] = (games['white_rating']-games['black_rating']).abs()
chess_ann2 = games.groupby('year_g').agg(
    avg_rating_diff=('rdiff','mean'),
    avg_white_rating=('white_rating','mean'),
    n_games=('turns','count'),
    draw_rate=('victory_status',lambda x:(x=='draw').mean()),
).reset_index().rename(columns={'year_g':'year'})
asylum_ann = asylum_raw.groupby('Año').agg(total=('Total','sum')).reset_index().rename(columns={'Año':'year','total':'asylum_total'})
pitch_ann = reviews.groupby('pub_year').agg(avg_score=('score','mean')).reset_index().rename(columns={'pub_year':'year'})

merged = (accidents_ann
    .merge(chess_ann2, on='year')
    .merge(asylum_ann, on='year')
    .merge(pitch_ann, on='year')
    .sort_values('year'))

def detrend(series):
    """Remove linear time trend, return residuals."""
    y = series.values
    x = np.arange(len(y))
    slope, intercept, *_ = stats.linregress(x, y)
    return y - (slope * x + intercept)

ax10 = fig.add_subplot(gs[3, 0])
if len(merged) >= 4:
    d_alcohol = detrend(merged['alcohol_crashes'])
    d_rdiff   = detrend(merged['avg_rating_diff'])
    r_raw, _  = stats.pearsonr(merged['alcohol_crashes'], merged['avg_rating_diff'])
    r_det, p_det = stats.pearsonr(d_alcohol, d_rdiff)
    ax10.scatter(d_alcohol, d_rdiff, c=merged['year'], cmap='viridis', s=80, edgecolors='black', zorder=3)
    for _, row in merged.iterrows():
        i = merged.index.get_loc(_)
        ax10.annotate(str(int(row['year'])), (d_alcohol[i], d_rdiff[i]),
                      textcoords='offset points', xytext=(4,4), fontsize=8)
    ax10.set_xlabel('Alcohol Crashes (detrended)', fontsize=10)
    ax10.set_ylabel('Chess Rating Diff (detrended)', fontsize=10)
    ax10.set_title(f'DETRENDED: Alcohol Crashes\nvs Chess Rating Gap\nRaw r={r_raw:+.3f} → Detrended r={r_det:+.3f} (p={p_det:.3f})',
                   fontsize=10, fontweight='bold')
    ax10.grid(True, alpha=0.25)
    ax10.axhline(0, color='gray', alpha=0.5); ax10.axvline(0, color='gray', alpha=0.5)

ax11 = fig.add_subplot(gs[3, 1])
if len(merged) >= 4:
    d_asylum = detrend(merged['asylum_total'])
    d_chess   = detrend(merged['avg_white_rating'])
    r_raw2, _ = stats.pearsonr(merged['asylum_total'], merged['avg_white_rating'])
    r_det2, p_det2 = stats.pearsonr(d_asylum, d_chess)
    ax11.scatter(d_asylum, d_chess, c=merged['year'], cmap='viridis', s=80, edgecolors='black', zorder=3)
    for _, row in merged.iterrows():
        i = merged.index.get_loc(_)
        ax11.annotate(str(int(row['year'])), (d_asylum[i], d_chess[i]),
                      textcoords='offset points', xytext=(4,4), fontsize=8)
    ax11.set_xlabel('Asylum Applicants (detrended)', fontsize=10)
    ax11.set_ylabel('Chess White Rating (detrended)', fontsize=10)
    ax11.set_title(f'DETRENDED: Asylum Seekers\nvs Chess Ratings\nRaw r={r_raw2:+.3f} → Detrended r={r_det2:+.3f} (p={p_det2:.3f})',
                   fontsize=10, fontweight='bold')
    ax11.grid(True, alpha=0.25)
    ax11.axhline(0, color='gray', alpha=0.5); ax11.axvline(0, color='gray', alpha=0.5)

ax12 = fig.add_subplot(gs[3, 2])
# Year-over-year changes (first differences)
if len(merged) >= 4:
    diff_df = merged.copy()
    for col in ['alcohol_crashes','avg_rating_diff','asylum_total','avg_white_rating','avg_score']:
        diff_df[f'd_{col}'] = diff_df[col].diff()
    diff_df = diff_df.dropna()

    r1, p1 = stats.pearsonr(diff_df['d_alcohol_crashes'], diff_df['d_avg_rating_diff'])
    r2, p2 = stats.pearsonr(diff_df['d_asylum_total'], diff_df['d_avg_white_rating'])
    r3, p3 = stats.pearsonr(diff_df['d_asylum_total'], diff_df['d_avg_score'])

    pairs_label = ['Alcohol↓ & Chess\nRating Gap', 'Asylum & Chess\nWhite Rating', 'Asylum &\nPitchfork Score']
    rs = [r1, r2, r3]
    ps_vals = [p1, p2, p3]
    bar_colors = ['#d73027' if r<0 else '#1a9850' for r in rs]
    bars = ax12.bar(pairs_label, rs, color=bar_colors, edgecolor='black', linewidth=0.7)
    for bar, r, pv in zip(bars, rs, ps_vals):
        ax12.text(bar.get_x()+bar.get_width()/2,
                  bar.get_height()+(0.02 if r>0 else -0.05),
                  f"r={r:+.2f}\np={pv:.3f}", ha='center', fontsize=9,
                  va='bottom' if r>0 else 'top')
    ax12.axhline(0, color='black', linewidth=1)
    ax12.set_ylim(-1.1, 1.1)
    ax12.set_ylabel('Pearson r (first differences)', fontsize=10)
    ax12.set_title('FIRST-DIFFERENCE TEST:\nDo Correlations Hold After\nRemoving Trends?', fontsize=10, fontweight='bold')
    ax12.grid(True, axis='y', alpha=0.3)

plt.savefig('deep_analysis.png', dpi=150, bbox_inches='tight')
print("Saved deep_analysis.png")

# ── Print summary stats for the markdown ──────────────────────
print("\n=== STATS FOR REPORT ===")
# Pandemic transmission
tx_stats = p.groupby('tx_simple')[['Case_Fatality_Rate_Pct','Spread_Score']].agg(['mean','median','count'])
print("\nTransmission stats:\n", tx_stats)

# Containment
print("\nContainment stats:\n", contain_stats)

# Blood cell Leukemia vs Normal
print("\nLeukemia vs Normal top features:\n", res_df.to_string())

# Chess openings
print("\nTop openings:\n", opening_stats.to_string())

# FARS peak hours
peak_crash = hour_stats.loc[hour_stats['n_crashes'].idxmax()]
peak_alcohol = hour_stats.loc[hour_stats['alcohol_rate'].idxmax()]
print(f"\nPeak crash hour: {int(peak_crash['HOUR'])}:00 ({int(peak_crash['n_crashes'])} crashes)")
print(f"Peak alcohol hour: {int(peak_alcohol['HOUR'])}:00 ({peak_alcohol['alcohol_rate']*100:.1f}% alcohol-involved)")

# Pitchfork genre
genre_avg = reviews_filtered.groupby('genre')['score'].mean().sort_values(ascending=False)
print("\nGenre avg scores:\n", genre_avg)

# Detrending results
if len(merged) >= 4:
    print(f"\nDetrending results:")
    print(f"  Alcohol vs Chess Rating Diff: raw r={r_raw:+.3f} → detrended r={r_det:+.3f} (p={p_det:.3f})")
    print(f"  Asylum vs Chess Rating: raw r={r_raw2:+.3f} → detrended r={r_det2:+.3f} (p={p_det2:.3f})")
    print(f"\nFirst-difference correlations:")
    print(f"  Alcohol Δ vs Chess Rating Gap Δ: r={r1:+.3f} (p={p1:.3f})")
    print(f"  Asylum Δ vs Chess Rating Δ: r={r2:+.3f} (p={p2:.3f})")
    print(f"  Asylum Δ vs Pitchfork Score Δ: r={r3:+.3f} (p={p3:.3f})")
