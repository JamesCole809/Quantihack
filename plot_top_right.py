import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Chess - aggregate by month
games = pd.read_csv('data/archive4/games.csv')
games['dt'] = pd.to_datetime(games['created_at'], unit='ms', errors='coerce')
games['year_month'] = games['dt'].dt.to_period('M')
chess_monthly = games.groupby('year_month').agg(chess_avg_turns=('turns', 'mean')).reset_index()
chess_monthly['date'] = chess_monthly['year_month'].dt.to_timestamp()

# Accidents - aggregate by month
acc = pd.read_csv('data/archive7/accident.csv')
acc['date'] = pd.to_datetime(acc[['YEAR', 'MONTH']].assign(DAY=1))
acc['year_month'] = acc['date'].dt.to_period('M')
acc_monthly = acc.groupby('year_month').agg(accident_avg_vehicles=('VE_TOTAL', 'mean')).reset_index()
acc_monthly['date'] = acc_monthly['year_month'].dt.to_timestamp()

# Merge on month
merged = chess_monthly.merge(acc_monthly, on='year_month', suffixes=('_chess', '_acc'))
merged = merged.sort_values('date_chess')

print(f"Overlapping months: {len(merged)}")
print(merged[['year_month', 'chess_avg_turns', 'accident_avg_vehicles']].to_string(index=False))

r = merged['chess_avg_turns'].corr(merged['accident_avg_vehicles'])
print(f"\nMonthly correlation: r = {r:.4f}")

# Normalise
def norm(s):
    return (s - s.min()) / (s.max() - s.min())

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(merged['date_chess'], norm(merged['chess_avg_turns']), 'o-', color='#4C72B0', linewidth=2, markersize=6, label='Chess Avg Turns')
ax.plot(merged['date_chess'], norm(merged['accident_avg_vehicles']), 's-', color='#C44E52', linewidth=2, markersize=6, label='Accident Avg Vehicles')
ax.set_title(f'Archive4 (Chess) vs Archive7 (Accidents) — Monthly (r = {r:.4f})', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Normalised Value')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chess_vs_accidents_monthly.png', dpi=150, bbox_inches='tight')
print("\nSaved to chess_vs_accidents_monthly.png")
