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
a7 = acc.groupby('YEAR').agg(accident_avg_vehicles=('VE_TOTAL', 'mean')).reset_index()
a7.columns = ['year', 'accident_avg_vehicles']

vehicle = pd.read_csv('data/archive7/vehicle.csv')
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Top Cross-Archive Correlations', fontsize=16, fontweight='bold')

# 1: Chess games vs Vehicle avg speed
m1 = a4.merge(a7, on='year', how='inner').dropna(subset=['chess_games', 'vehicle_avg_speed'])
ax = axes[0, 0]
ax2 = ax.twinx()
ax.bar(m1['year'], m1['chess_games'], color='#4C72B0', alpha=0.7, label='Chess Games')
ax2.plot(m1['year'], m1['vehicle_avg_speed'], 'o-', color='#C44E52', linewidth=2, markersize=8, label='Avg Vehicle Speed')
ax.set_xlabel('Year')
ax.set_ylabel('Chess Games', color='#4C72B0')
ax2.set_ylabel('Avg Vehicle Speed (mph)', color='#C44E52')
ax.set_title(f'Archive4 vs Archive7\nChess Games vs Vehicle Speed (r = -0.9745)', fontsize=11)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# 2: Chess avg turns vs Accident avg vehicles
m2 = a4.merge(a7, on='year', how='inner').dropna(subset=['chess_avg_turns', 'accident_avg_vehicles'])
ax = axes[0, 1]
ax.scatter(m2['chess_avg_turns'], m2['accident_avg_vehicles'], s=100, c='#55A868', edgecolors='black', zorder=5)
for _, row in m2.iterrows():
    ax.annotate(int(row['year']), (row['chess_avg_turns'], row['accident_avg_vehicles']), textcoords="offset points", xytext=(8, 5), fontsize=9)
z = np.polyfit(m2['chess_avg_turns'], m2['accident_avg_vehicles'], 1)
x_line = np.linspace(m2['chess_avg_turns'].min(), m2['chess_avg_turns'].max(), 50)
ax.plot(x_line, np.polyval(z, x_line), '--', color='gray', alpha=0.7)
ax.set_xlabel('Chess Avg Turns')
ax.set_ylabel('Accident Avg Vehicles')
ax.set_title(f'Archive4 vs Archive7\nChess Turns vs Accident Vehicles (r = 0.9710)', fontsize=11)

# 3: Chess games vs Music std score
m3 = a4.merge(a5, on='year', how='inner').dropna(subset=['chess_games', 'music_std_score'])
ax = axes[1, 0]
ax.scatter(m3['chess_games'], m3['music_std_score'], s=100, c='#8172B2', edgecolors='black', zorder=5)
for _, row in m3.iterrows():
    ax.annotate(int(row['year']), (row['chess_games'], row['music_std_score']), textcoords="offset points", xytext=(8, 5), fontsize=9)
z = np.polyfit(m3['chess_games'], m3['music_std_score'], 1)
x_line = np.linspace(m3['chess_games'].min(), m3['chess_games'].max(), 50)
ax.plot(x_line, np.polyval(z, x_line), '--', color='gray', alpha=0.7)
ax.set_xlabel('Chess Games')
ax.set_ylabel('Music Score Std Dev')
ax.set_title(f'Archive4 vs Archive5\nChess Games vs Music Score Spread (r = 0.9652)', fontsize=11)

# 4: Asylum nat women vs Music avg score
m4 = a3.merge(a5, on='year', how='inner').dropna(subset=['asylum_nat_women', 'music_avg_score'])
ax = axes[1, 1]
ax.scatter(m4['asylum_nat_women'], m4['music_avg_score'], s=100, c='#CCB974', edgecolors='black', zorder=5)
for _, row in m4.iterrows():
    ax.annotate(int(row['year']), (row['asylum_nat_women'], row['music_avg_score']), textcoords="offset points", xytext=(8, 5), fontsize=9)
z = np.polyfit(m4['asylum_nat_women'], m4['music_avg_score'], 1)
x_line = np.linspace(m4['asylum_nat_women'].min(), m4['asylum_nat_women'].max(), 50)
ax.plot(x_line, np.polyval(z, x_line), '--', color='gray', alpha=0.7)
ax.set_xlabel('Asylum Applications (Women)')
ax.set_ylabel('Avg Music Review Score')
ax.set_title(f'Archive3 vs Archive5\nAsylum Applications vs Music Scores (r = 0.9588)', fontsize=11)

plt.tight_layout()
plt.savefig('cross_archive_correlations.png', dpi=150, bbox_inches='tight')
print("Saved to cross_archive_correlations.png")
