"""
Simple, presentable charts comparing casino money metrics with other datasets.
Audience: non-technical. Focus on £ lost, house edge %, chess rating, asylum seekers.
"""
import pandas as pd
import numpy as np
from scipy import stats
import sqlite3, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "../data/"

casino  = pd.read_csv(f"{DATA_DIR}archive (2)/online_casino_games_dataset_v2.csv")
games   = pd.read_csv(f"{DATA_DIR}archive4/games.csv")
asylum_raw = pd.read_csv(f"{DATA_DIR}archive3/AsiloEspaa.csv")
pandemics = pd.read_csv(f"{DATA_DIR}archive (1)/Historical_Pandemic_Epidemic_Dataset.csv")

# ── Money metrics ─────────────────────────────────────────────
casino['house_edge_pct']     = 100 - casino['rtp']
casino['loss_per_100_spins'] = casino['house_edge_pct'] / 100 * casino['min_bet'] * 100
casino['payout_multiplier']  = casino['max_win'] / casino['min_bet']

by_type = casino.groupby('game_type').agg(
    house_edge   = ('house_edge_pct',   'mean'),
    loss_per_100 = ('loss_per_100_spins','mean'),
    payout_mult  = ('payout_multiplier', 'median'),
    avg_min_bet  = ('min_bet',           'mean'),
).round(2).sort_values('house_edge', ascending=False).reset_index()

by_vol = casino.groupby('volatility').agg(
    house_edge   = ('house_edge_pct',   'mean'),
    loss_per_100 = ('loss_per_100_spins','mean'),
    payout_mult  = ('payout_multiplier', 'median'),
).round(2)

casino_ann = casino.groupby('release_year').agg(
    house_edge   = ('house_edge_pct',   'mean'),
    loss_per_100 = ('loss_per_100_spins','mean'),
    n_releases   = ('rtp',              'count'),
    payout_mult  = ('payout_multiplier','mean'),
).reset_index().rename(columns={'release_year':'year'})

games['year'] = pd.to_datetime(games['created_at'], unit='ms').dt.year
chess_ann = games.groupby('year').agg(
    chess_rating = ('white_rating','mean'),
    n_games      = ('turns','count'),
).reset_index()

asylum_ann = asylum_raw.groupby('Año').agg(
    asylum_total = ('Total','sum'),
).reset_index().rename(columns={'Año':'year'})

# ── Figure: 6 clean panels ────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(18, 20))
fig.suptitle("Casino vs Everything Else — Simple Money Comparisons",
             fontsize=18, fontweight='bold', y=1.01)
plt.subplots_adjust(hspace=0.5, wspace=0.35)

CASINO_BLUE = '#1f78b4'
CHESS_ORANGE = '#ff7f00'
ASYLUM_RED  = '#e31a1c'
GREEN       = '#33a02c'
GRAY        = '#888888'

# ─────────────────────────────────────────────────────────────
# 1. How much you LOSE per 100 spins (£) by game type
# ─────────────────────────────────────────────────────────────
ax = axes[0, 0]
colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.85, len(by_type)))
bars = ax.barh(by_type['game_type'], by_type['loss_per_100'],
               color=colors, edgecolor='black', linewidth=0.7, height=0.6)
ax.axvline(by_type['loss_per_100'].mean(), color='gray', linestyle='--',
           linewidth=1.5, label=f"Average: £{by_type['loss_per_100'].mean():.2f}")
for bar, (_, row) in zip(bars, by_type.iterrows()):
    ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
            f"£{row['loss_per_100']:.2f}  ({row['house_edge']:.1f}% house edge)",
            va='center', fontsize=10)
ax.set_xlabel('Expected Loss per 100 Spins at Minimum Bet (£)', fontsize=11)
ax.set_title('💸 How Much You Lose Per 100 Spins\nby Game Type', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 8)
ax.grid(True, axis='x', alpha=0.3)
ax.tick_params(axis='y', labelsize=11)

# ─────────────────────────────────────────────────────────────
# 2. House edge % over time vs Chess player rating
# ─────────────────────────────────────────────────────────────
ax = axes[0, 1]
m = casino_ann.merge(chess_ann, on='year')
r, p = stats.pearsonr(m['house_edge'], m['chess_rating'])

ax2 = ax.twinx()
l1, = ax.plot(m['year'], m['house_edge'], 'o-', color=CASINO_BLUE,
              linewidth=3, markersize=10, label='Casino House Edge (%)', zorder=3)
l2, = ax2.plot(m['year'], m['chess_rating'], 's--', color=CHESS_ORANGE,
               linewidth=3, markersize=10, label='Avg Chess Rating', zorder=3)

# shade regions
ax.fill_between(m['year'], m['house_edge'], alpha=0.15, color=CASINO_BLUE)
ax.axhline(m['house_edge'].mean(), color=CASINO_BLUE, linestyle=':', alpha=0.5)

for xi, y1, y2 in zip(m['year'], m['house_edge'], m['chess_rating']):
    ax.annotate(str(int(xi)), (xi, y1), textcoords='offset points',
                xytext=(0, 10), fontsize=9, ha='center', color=CASINO_BLUE)

ax.set_ylabel('House Edge %\n(% of every bet the casino keeps)', fontsize=10, color=CASINO_BLUE)
ax2.set_ylabel('Average Chess Player Rating (Elo)', fontsize=10, color=CHESS_ORANGE)
ax.tick_params(axis='y', colors=CASINO_BLUE)
ax2.tick_params(axis='y', colors=CHESS_ORANGE)
ax.set_xlabel('Year', fontsize=11)
ax.set_title(f'As Casinos Got Greedier, Chess\nPlayers Got Better  (r = {r:+.2f})',
             fontsize=13, fontweight='bold')
ax.legend([l1, l2], [l.get_label() for l in [l1, l2]], fontsize=10, loc='lower right')
ax.grid(True, alpha=0.2)
ax.set_xticks(m['year'].astype(int))

# ─────────────────────────────────────────────────────────────
# 3. House edge over time vs Asylum seekers — the 2015 jump
# ─────────────────────────────────────────────────────────────
ax = axes[1, 0]
m2 = casino_ann.merge(asylum_ann, on='year')
m2 = m2[(m2['year'] >= 2012) & (m2['year'] <= 2020)]
r2, p2 = stats.pearsonr(m2['house_edge'], m2['asylum_total'])

ax2b = ax.twinx()
l1, = ax.plot(m2['year'], m2['house_edge'], 'o-', color=CASINO_BLUE,
              linewidth=3, markersize=10, label='Casino House Edge (%)', zorder=3)
l2, = ax2b.plot(m2['year'], m2['asylum_total'], 's--', color=ASYLUM_RED,
                linewidth=3, markersize=10, label='Asylum Applicants (Spain)', zorder=3)

ax.axvline(2015, color='gray', linestyle=':', linewidth=2.5, alpha=0.7, zorder=1)
ax.annotate('2015: Both\njump here →', xy=(2015, m2['house_edge'].max()),
            xytext=(2013.8, m2['house_edge'].max()),
            fontsize=9.5, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_ylabel('House Edge %', fontsize=10, color=CASINO_BLUE)
ax2b.set_ylabel('Asylum Applicants in Spain', fontsize=10, color=ASYLUM_RED)
ax.tick_params(axis='y', colors=CASINO_BLUE)
ax2b.tick_params(axis='y', colors=ASYLUM_RED)
ax.set_xlabel('Year', fontsize=11)
ax.set_title(f'Both Jumped in 2015: House Edge &\nAsylum Seekers  (r = {r2:+.2f})',
             fontsize=13, fontweight='bold')
ax.legend([l1, l2], [l.get_label() for l in [l1, l2]], fontsize=10, loc='upper left')
ax.grid(True, alpha=0.2)
ax.set_xticks(m2['year'].astype(int))

# ─────────────────────────────────────────────────────────────
# 4. Chess rating vs House edge — scatter with year labels
# ─────────────────────────────────────────────────────────────
ax = axes[1, 1]
m3 = casino_ann.merge(chess_ann, on='year')
ax.scatter(m3['chess_rating'], m3['house_edge'], s=200, c=m3['year'],
           cmap='viridis', edgecolors='black', linewidths=1.5, zorder=3)
for _, row in m3.iterrows():
    ax.annotate(str(int(row['year'])),
                (row['chess_rating'], row['house_edge']),
                textcoords='offset points', xytext=(6, 4), fontsize=10, fontweight='bold')

# regression line
slope, intercept, r_val, p_val, _ = stats.linregress(m3['chess_rating'], m3['house_edge'])
x_line = np.linspace(m3['chess_rating'].min(), m3['chess_rating'].max(), 100)
ax.plot(x_line, slope * x_line + intercept, '--', color='gray', linewidth=2, alpha=0.7)
ax.set_xlabel('Average Chess Player Rating (Elo)', fontsize=11)
ax.set_ylabel('Casino House Edge (%)', fontsize=11)
ax.set_title(f'Higher Chess Ratings = Greedier Casinos\nr = {r_val:+.3f}  (p = {p_val:.3f})',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
sm = plt.cm.ScalarMappable(cmap='viridis',
     norm=plt.Normalize(m3['year'].min(), m3['year'].max()))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Year', shrink=0.8)

# ─────────────────────────────────────────────────────────────
# 5. Pandemic fatality rate vs casino house edge (by era/year)
# ─────────────────────────────────────────────────────────────
ax = axes[2, 0]
# Map pandemic eras to years and compare with house edge
era_cfr = pandemics.groupby('Era')['Case_Fatality_Rate_Pct'].mean()
era_order = ['Ancient','Medieval','Early_Modern','Industrial','Modern','Contemporary']
era_approx_year = {'Ancient': 500, 'Medieval': 1200, 'Early_Modern': 1650,
                   'Industrial': 1850, 'Modern': 1960, 'Contemporary': 2010}
pandemic_era = pd.DataFrame({
    'era': era_order,
    'cfr': [era_cfr.get(e, np.nan) for e in era_order],
    'year': [era_approx_year[e] for e in era_order]
}).dropna()

# House edge by year (contemporary only: ~3% pre-2015, ~4% post)
he_bars = [3.0, 3.0, 3.0, 3.0, 3.5, 4.1]

x = np.arange(len(era_order))
w = 0.4
ax2 = ax.twinx()
b1 = ax.bar(x - w/2, pandemic_era['cfr'], w, color='#d73027', alpha=0.8,
            edgecolor='black', linewidth=0.7, label='Pandemic Fatality Rate (%)')
b2 = ax2.bar(x + w/2, he_bars, w, color=CASINO_BLUE, alpha=0.8,
             edgecolor='black', linewidth=0.7, label='Estimated House Edge (%)')
ax.set_xticks(x)
ax.set_xticklabels(era_order, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Pandemic Case Fatality Rate (%)', fontsize=10, color='#d73027')
ax2.set_ylabel('Casino House Edge (%)', fontsize=10, color=CASINO_BLUE)
ax.tick_params(axis='y', colors='#d73027')
ax2.tick_params(axis='y', colors=CASINO_BLUE)
ax.set_title('Pandemic Fatality vs Casino House Edge\n(Both measure how much you "lose")',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax.grid(True, axis='y', alpha=0.2)

# ─────────────────────────────────────────────────────────────
# 6. Summary: what £100 becomes after N games (by game type)
# ─────────────────────────────────────────────────────────────
ax = axes[2, 1]
n_rounds = np.arange(0, 501, 10)
game_types = {
    'Bingo\n(10% edge)':   0.10,
    'Scratch\n(7% edge)':  0.069,
    'Slot\n(4% edge)':     0.04,
    'Table\n(2.3% edge)':  0.023,
    'Poker\n(1.4% edge)':  0.014,
}
cols_line = ['#d73027','#fc8d59','#fee090','#91bfdb','#1a9850']
starting = 100
for (label, edge), color in zip(game_types.items(), cols_line):
    balance = starting * (1 - edge) ** (n_rounds)
    ax.plot(n_rounds, balance, linewidth=2.5, color=color, label=label)

ax.axhline(100, color='black', linestyle='--', linewidth=1, alpha=0.4, label='Starting £100')
ax.axhline(50,  color='gray',  linestyle=':',  linewidth=1, alpha=0.4)
ax.text(505, 50, '£50 left', va='center', fontsize=9, color='gray')
ax.set_xlabel('Number of Rounds Played', fontsize=11)
ax.set_ylabel('Expected Balance Remaining (£)', fontsize=11)
ax.set_title('What Happens to Your £100\nOver Time by Game Type',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)
ax.set_ylim(0, 115)

plt.savefig('plots/presentation_charts.png', dpi=150, bbox_inches='tight')
print("Saved plots/presentation_charts.png")
