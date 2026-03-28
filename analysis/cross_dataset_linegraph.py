import pandas as pd
import numpy as np
from scipy import stats
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data/"

# ── Load & aggregate ─────────────────────────────────────────
accidents = pd.read_csv(f"{DATA_DIR}archive7/accident.csv")
fars = accidents.groupby('YEAR').agg(
    alcohol_crashes=('A_POSBAC', lambda x: (x==1).sum()),
    ped_crashes=('PEDS','sum'),
    total_fatals=('FATALS','sum'),
).reset_index().rename(columns={'YEAR':'year'})

asylum_raw = pd.read_csv(f"{DATA_DIR}archive3/AsiloEspaa.csv")
asylum = asylum_raw.groupby('Año').agg(
    total_applicants=('Total','sum'),
    admission_rate=('Admitidas', lambda x: x.sum()/asylum_raw.loc[x.index,'Total'].sum()),
).reset_index().rename(columns={'Año':'year'})

games = pd.read_csv(f"{DATA_DIR}archive4/games.csv")
games['year'] = pd.to_datetime(games['created_at'], unit='ms').dt.year
games['rating_diff'] = (games['white_rating'] - games['black_rating']).abs()
chess = games.groupby('year').agg(
    avg_white_rating=('white_rating','mean'),
    avg_rating_diff=('rating_diff','mean'),
    draw_rate=('victory_status', lambda x: (x=='draw').mean()),
    resign_rate=('victory_status', lambda x: (x=='resign').mean()),
    n_games=('turns','count'),
).reset_index()

conn = sqlite3.connect(f"{DATA_DIR}archive5/database.sqlite")
pitchfork = pd.read_sql(
    "SELECT pub_year as year, AVG(score) as avg_score, AVG(best_new_music) as bnm_rate, COUNT(*) as n_reviews "
    "FROM reviews WHERE pub_year > 1999 GROUP BY pub_year", conn)
conn.close()

# ── Merge on year ────────────────────────────────────────────
df = (fars
    .merge(asylum, on='year')
    .merge(chess, on='year')
    .merge(pitchfork, on='year')
    .sort_values('year'))

def norm(s):
    base = s.iloc[0]
    return (s / base) * 100 if base != 0 else s

# ── Pairs to plot ────────────────────────────────────────────
pairs = [
    # (x_col, y_col, x_label, y_label, title, x_color, y_color, r_val)
    ('year', 'alcohol_crashes',   'alcohol_crashes',   'avg_rating_diff',
     'FARS: Alcohol Crashes',     'Chess: Rating Diff',
     'Alcohol Crashes ↔ Chess Rating Gap\n(r = −0.995)',
     '#d73027', '#4575b4'),

    ('year', 'total_applicants',  'total_applicants',  'avg_white_rating',
     'Asylum Applicants (Spain)', 'Chess White Rating',
     'Asylum Seekers ↔ Chess Ratings\n(ρ = +1.000)',
     '#f46d43', '#74add1'),

    ('year', 'total_applicants',  'total_applicants',  'avg_score',
     'Asylum Applicants (Spain)', 'Pitchfork Avg Score',
     'Asylum Seekers ↔ Pitchfork Scores\n(ρ = +1.000)',
     '#f46d43', '#9970ab'),

    ('year', 'draw_rate',         'draw_rate',         'avg_score',
     'Chess Draw Rate',           'Pitchfork Avg Score',
     'Chess Draw Rate ↔ Music Score\n(r = +0.923, p = 0.025)',
     '#1a9850', '#9970ab'),

    ('year', 'ped_crashes',       'ped_crashes',       'n_games',
     'Pedestrian Crashes (AZ)',   'Chess Games Played',
     'Pedestrian Crashes ↔ Online Chess\n(r = +0.941)',
     '#d73027', '#1a9850'),

    ('year', 'admission_rate',    'admission_rate',    'bnm_rate',
     'Asylum Admission Rate',     'Pitchfork BNM Rate',
     'Asylum Approval ↔ Best New Music Rate\n(ρ = −1.000)',
     '#f46d43', '#9970ab'),
]

fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle("Cross-Dataset Correlations — Line Graphs (Annual Data, Dual Axis)",
             fontsize=15, fontweight='bold', y=1.01)
axes = axes.flatten()

for ax, (_, _, col_a, col_b, label_a, label_b, title, color_a, color_b) in zip(axes, pairs):
    sub = df[['year', col_a, col_b]].dropna()

    ax2 = ax.twinx()

    l1, = ax.plot(sub['year'], sub[col_a], 'o-', color=color_a, linewidth=2.5,
                  markersize=8, label=label_a, zorder=3)
    l2, = ax2.plot(sub['year'], sub[col_b], 's--', color=color_b, linewidth=2.5,
                   markersize=8, label=label_b, zorder=3)

    # Year labels on points
    for _, row in sub.iterrows():
        ax.annotate(str(int(row['year'])),
                    (row['year'], row[col_a]),
                    textcoords='offset points', xytext=(0, 8),
                    fontsize=8, color=color_a, ha='center')

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel(label_a, fontsize=11, color=color_a)
    ax2.set_ylabel(label_b, fontsize=11, color=color_b)
    ax.tick_params(axis='y', colors=color_a)
    ax2.tick_params(axis='y', colors=color_b)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(sub['year'].astype(int))
    ax.grid(True, alpha=0.2)

    # Compute correlation for annotation
    r, p = stats.pearsonr(sub[col_a], sub[col_b])
    rho, _ = stats.spearmanr(sub[col_a], sub[col_b])
    ax.text(0.03, 0.05, f"r={r:+.3f}  ρ={rho:+.3f}\np={p:.3e}  n={len(sub)} yrs",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('cross_dataset_linegraph.png', dpi=150, bbox_inches='tight')
print("Saved cross_dataset_linegraph.png")
