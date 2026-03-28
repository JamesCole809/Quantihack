import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
games = pd.read_csv('data/archive4/games.csv')
games['year'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce').dt.year
a4 = games.groupby('year').agg(chess_games=('id', 'count'), chess_avg_turns=('turns', 'mean')).reset_index()

acc = pd.read_csv('data/archive7/accident.csv')
vehicle = pd.read_csv('data/archive7/vehicle.csv')
a7 = acc.groupby('YEAR').agg(accident_avg_vehicles=('VE_TOTAL', 'mean')).reset_index()
a7.columns = ['year', 'accident_avg_vehicles']
a7v = vehicle.groupby('Year').agg(vehicle_avg_speed=('TRAV_SP', lambda x: x[(x < 998) & (x > 0)].mean())).reset_index()
a7v.columns = ['year', 'vehicle_avg_speed']
a7 = a7.merge(a7v, on='year', how='outer')

asilo_es = pd.read_csv('data/archive3/AsiloEspaa.csv', encoding='latin1')
yr_col = [c for c in asilo_es.columns if 'o' in c.lower()][-1]
a3 = asilo_es.groupby(yr_col).agg(asylum_nat_women=('Mujeres', 'sum')).reset_index()
a3.columns = ['year', 'asylum_nat_women']

conn = sqlite3.connect('data/archive5/database.sqlite')
reviews = pd.read_sql('SELECT * FROM reviews', conn)
conn.close()
a5 = reviews.groupby('pub_year').agg(music_avg_score=('score', 'mean'), music_std_score=('score', 'std')).reset_index()
a5.columns = ['year', 'music_avg_score', 'music_std_score']

def normalise(s):
    return (s - s.min()) / (s.max() - s.min())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cross-Archive Correlations — Normalised Line Graphs', fontsize=16, fontweight='bold')

# 1: Chess games vs Vehicle avg speed
m1 = a4.merge(a7, on='year').dropna(subset=['chess_games', 'vehicle_avg_speed'])
ax = axes[0, 0]
ax.plot(m1['year'], normalise(m1['chess_games']), 'o-', color='#4C72B0', linewidth=2, markersize=7, label='Chess Games')
ax.plot(m1['year'], normalise(m1['vehicle_avg_speed']), 's-', color='#C44E52', linewidth=2, markersize=7, label='Avg Vehicle Speed')
ax.set_title('Archive4 vs Archive7 (r = -0.9745)', fontsize=11)
ax.set_xlabel('Year'); ax.set_ylabel('Normalised Value')
ax.legend(); ax.grid(alpha=0.3)

# 2: Chess avg turns vs Accident avg vehicles
m2 = a4.merge(a7, on='year').dropna(subset=['chess_avg_turns', 'accident_avg_vehicles'])
ax = axes[0, 1]
ax.plot(m2['year'], normalise(m2['chess_avg_turns']), 'o-', color='#4C72B0', linewidth=2, markersize=7, label='Chess Avg Turns')
ax.plot(m2['year'], normalise(m2['accident_avg_vehicles']), 's-', color='#C44E52', linewidth=2, markersize=7, label='Accident Avg Vehicles')
ax.set_title('Archive4 vs Archive7 (r = 0.9710)', fontsize=11)
ax.set_xlabel('Year'); ax.set_ylabel('Normalised Value')
ax.legend(); ax.grid(alpha=0.3)

# 3: Chess games vs Music std score
m3 = a4.merge(a5, on='year').dropna(subset=['chess_games', 'music_std_score'])
ax = axes[1, 0]
ax.plot(m3['year'], normalise(m3['chess_games']), 'o-', color='#4C72B0', linewidth=2, markersize=7, label='Chess Games')
ax.plot(m3['year'], normalise(m3['music_std_score']), 's-', color='#55A868', linewidth=2, markersize=7, label='Music Score Std Dev')
ax.set_title('Archive4 vs Archive5 (r = 0.9652)', fontsize=11)
ax.set_xlabel('Year'); ax.set_ylabel('Normalised Value')
ax.legend(); ax.grid(alpha=0.3)

# 4: Asylum women vs Music avg score
m4 = a3.merge(a5, on='year').dropna(subset=['asylum_nat_women', 'music_avg_score'])
ax = axes[1, 1]
ax.plot(m4['year'], normalise(m4['asylum_nat_women']), 'o-', color='#8172B2', linewidth=2, markersize=7, label='Asylum Applications (Women)')
ax.plot(m4['year'], normalise(m4['music_avg_score']), 's-', color='#CCB974', linewidth=2, markersize=7, label='Avg Music Review Score')
ax.set_title('Archive3 vs Archive5 (r = 0.9588)', fontsize=11)
ax.set_xlabel('Year'); ax.set_ylabel('Normalised Value')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cross_archive_lines.png', dpi=150, bbox_inches='tight')
print("Saved to cross_archive_lines.png")
