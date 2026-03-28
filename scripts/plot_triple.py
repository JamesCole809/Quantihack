import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
games = pd.read_csv('data/archive4/games.csv', usecols=['created_at','turns','id','white_rating','black_rating','opening_ply'], engine='python')
games['year'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce').dt.year
a4 = games.groupby('year').agg(chess_games=('id', 'count')).reset_index()

conn = sqlite3.connect('data/archive5/database.sqlite')
reviews = pd.read_sql('SELECT score, pub_year FROM reviews', conn)
conn.close()
a5 = reviews.groupby('pub_year').agg(music_std_score=('score', 'std')).reset_index()
a5.columns = ['year', 'music_std_score']

vehicle = pd.read_csv('data/archive7/vehicle.csv', usecols=['Year','TRAV_SP'])
a7v = vehicle.groupby('Year').agg(vehicle_avg_speed=('TRAV_SP', lambda x: x[(x < 998) & (x > 0)].mean())).reset_index()
a7v.columns = ['year', 'vehicle_avg_speed']

# Merge all three
merged = a4.merge(a5, on='year').merge(a7v, on='year').dropna()

# Compute z-scores
for col in ['chess_games', 'music_std_score', 'vehicle_avg_speed']:
    merged[f'z_{col}'] = (merged[col] - merged[col].mean()) / merged[col].std()

merged['z_product'] = merged['z_chess_games'] * merged['z_music_std_score']

def norm(s):
    return (s - s.min()) / (s.max() - s.min())

r = merged['z_product'].corr(merged['vehicle_avg_speed'])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(merged['year'], norm(merged['z_chess_games']), 'o-', color='#4C72B0', linewidth=2, markersize=8, label='Chess Games (z)')
ax.plot(merged['year'], norm(merged['z_music_std_score']), 's-', color='#55A868', linewidth=2, markersize=8, label='Music Score Std Dev (z)')
ax.plot(merged['year'], norm(merged['z_product']), 'D-', color='#8172B2', linewidth=2.5, markersize=9, label='Z-Product (Chess x Music)')
ax.plot(merged['year'], norm(merged['vehicle_avg_speed']), '^-', color='#C44E52', linewidth=2, markersize=8, label='Avg Vehicle Speed')

ax.set_title(f'Chess Games x Music Score Spread vs Vehicle Speed (r = {r:.4f})', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Normalised Value')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('triple_correlation.png', dpi=150, bbox_inches='tight')
print(f"r = {r:.4f}, n = {len(merged)}")
print(merged[['year', 'chess_games', 'music_std_score', 'z_product', 'vehicle_avg_speed']].to_string(index=False))
print("Saved to triple_correlation.png")