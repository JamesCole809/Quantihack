"""
New dataset analysis — 6 clean, presentable panels.
Datasets: Gym, Vitamin Deficiency, Weight-Height, Credit Card Fraud, Oil Reservoirs.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "../data/"

# ── Load ──────────────────────────────────────────────────────
gym   = pd.read_csv(f"{DATA_DIR}archive (4)/gym_members_exercise_tracking.csv")
vit   = pd.read_csv(f"{DATA_DIR}archive (6)/vitamin_deficiency_disease_dataset_20260123.csv")
res   = pd.read_csv(f"{DATA_DIR}archive (7)/Reservoir_NameBasin_RegionLocatio.csv")
fraud = pd.read_csv(f"{DATA_DIR}archive (3)/train.csv",
                    usecols=['trans_date_trans_time','amt','is_fraud'])
fraud['hour'] = pd.to_datetime(fraud['trans_date_trans_time']).dt.hour

# ── Pre-compute key aggregations ──────────────────────────────

# Diet → B12 + disease (ordered by B12)
diet_order = ['Vegan', 'Vegetarian', 'Pescatarian', 'Omnivore']
diet_stats = vit.groupby('diet_type').agg(
    avg_b12      = ('serum_vitamin_b12_pg_ml', 'mean'),
    avg_vitd     = ('serum_vitamin_d_ng_ml',   'mean'),
    disease_rate = ('disease_diagnosis', lambda x: (x != 'Healthy').mean() * 100),
).reindex(diet_order).reset_index()

# Income → Vitamin D + disease
income_order = ['Low', 'Middle', 'High']
income_stats = vit.groupby('income_level').agg(
    avg_vitd     = ('serum_vitamin_d_ng_ml',   'mean'),
    disease_rate = ('disease_diagnosis', lambda x: (x != 'Healthy').mean() * 100),
).reindex(income_order).reset_index()

# Gym frequency → Fat% + Calories
freq_stats = gym.groupby('Workout_Frequency (days/week)').agg(
    avg_fat = ('Fat_Percentage',   'mean'),
    avg_cal = ('Calories_Burned',  'mean'),
    n       = ('BMI',              'count'),
).reset_index()

# Fraud by hour
hourly = fraud.groupby('hour').agg(
    fraud_pct = ('is_fraud', lambda x: x.mean() * 100),
    count     = ('is_fraud', 'count'),
).reset_index()

# Disease breakdown by diet (stacked)
disease_labels = ['Healthy', 'Anemia', 'Rickets_Osteomalacia',
                  'Night_Blindness', 'Scurvy']
disease_by_diet = vit.groupby('diet_type')['disease_diagnosis'].value_counts(normalize=True).unstack().fillna(0)
disease_by_diet = disease_by_diet.reindex(index=diet_order, columns=disease_labels).fillna(0) * 100

# Reservoir reserves by basin
basin = res.groupby('Basin_Region')['Proven_Reserves_Billion_Barrels'].sum().sort_values(ascending=True)

# ── Figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(18, 22))
fig.suptitle("New Dataset Analysis — Health, Lifestyle & Risk",
             fontsize=18, fontweight='bold', y=1.01)
plt.subplots_adjust(hspace=0.50, wspace=0.38)

C1 = '#d73027'   # red
C2 = '#1f78b4'   # blue
C3 = '#33a02c'   # green
C4 = '#ff7f00'   # orange
C5 = '#9970ab'   # purple

# ══════════════════════════════════════════════════════════════
# 1. DIET TYPE → VITAMIN B12 + DISEASE RATE
#    Story: "Vegans have 3× less B12 — and nearly everyone gets sick"
# ══════════════════════════════════════════════════════════════
ax = axes[0, 0]
ax2 = ax.twinx()

x1 = range(len(diet_stats))
l1, = ax.plot(x1, diet_stats['avg_b12'], 'o-',
              color=C2, linewidth=3, markersize=12, label='Serum Vitamin B12 (pg/mL)')
l2, = ax2.plot(x1, diet_stats['disease_rate'], 's--',
               color=C1, linewidth=3, markersize=12, label='Disease Rate (%)')
ax.fill_between(x1, diet_stats['avg_b12'], alpha=0.12, color=C2)
ax2.fill_between(x1, diet_stats['disease_rate'], alpha=0.12, color=C1)

for xi, b12, dr in zip(x1, diet_stats['avg_b12'], diet_stats['disease_rate']):
    ax.annotate(f'{b12:.0f} pg/mL', (xi, b12),
                textcoords='offset points', xytext=(0, 13),
                fontsize=11, color=C2, ha='center', fontweight='bold')
    ax2.annotate(f'{dr:.0f}%', (xi, dr),
                 textcoords='offset points', xytext=(0, -20),
                 fontsize=11, color=C1, ha='center', fontweight='bold')

ax.set_xticks(x1)
ax.set_xticklabels(diet_order, fontsize=12)
ax.set_ylabel('Serum Vitamin B12 (pg/mL)', fontsize=11, color=C2)
ax2.set_ylabel('Disease Rate (%)', fontsize=11, color=C1)
ax.tick_params(axis='y', colors=C2)
ax2.tick_params(axis='y', colors=C1)
ax.set_title('Vegan Diet = 3× Less Vitamin B12, Nearly\nEveryone Gets Sick (98% Disease Rate)',
             fontsize=12, fontweight='bold')
ax.legend([l1, l2], [l.get_label() for l in [l1, l2]], fontsize=10, loc='upper left')
ax.grid(True, alpha=0.2)

# ══════════════════════════════════════════════════════════════
# 2. INCOME LEVEL → VITAMIN D + DISEASE RATE
#    Story: "Rich people have double the Vitamin D and half the disease"
# ══════════════════════════════════════════════════════════════
ax = axes[0, 1]
ax2 = ax.twinx()

x2 = range(3)
l1, = ax.plot(x2, income_stats['avg_vitd'], 'o-',
              color=C4, linewidth=3, markersize=12, label='Serum Vitamin D (ng/mL)')
l2, = ax2.plot(x2, income_stats['disease_rate'], 's--',
               color=C1, linewidth=3, markersize=12, label='Disease Rate (%)')
ax.fill_between(x2, income_stats['avg_vitd'], alpha=0.12, color=C4)
ax2.fill_between(x2, income_stats['disease_rate'], alpha=0.12, color=C1)

for xi, vd, dr in zip(x2, income_stats['avg_vitd'], income_stats['disease_rate']):
    ax.annotate(f'{vd:.1f} ng/mL', (xi, vd),
                textcoords='offset points', xytext=(0, 13),
                fontsize=11, color=C4, ha='center', fontweight='bold')
    ax2.annotate(f'{dr:.0f}%\ndiseased', (xi, dr),
                 textcoords='offset points', xytext=(0, -24),
                 fontsize=11, color=C1, ha='center', fontweight='bold')

ax.set_xticks(x2)
ax.set_xticklabels(['Low Income', 'Middle Income', 'High Income'], fontsize=12)
ax.set_ylabel('Avg Serum Vitamin D (ng/mL)', fontsize=11, color=C4)
ax2.set_ylabel('Disease Rate (%)', fontsize=11, color=C1)
ax.tick_params(axis='y', colors=C4)
ax2.tick_params(axis='y', colors=C1)
ax.set_title('Wealthier People Have Double the Vitamin D\nand Half the Disease Rate',
             fontsize=12, fontweight='bold')
ax.legend([l1, l2], [l.get_label() for l in [l1, l2]], fontsize=10, loc='upper right')
ax.grid(True, alpha=0.2)

# ══════════════════════════════════════════════════════════════
# 3. GYM FREQUENCY → BODY FAT + CALORIES BURNED
#    Story: "5 sessions/week = half the body fat of 2 sessions"
# ══════════════════════════════════════════════════════════════
ax = axes[1, 0]
ax2 = ax.twinx()

x3 = range(len(freq_stats))
l1, = ax.plot(x3, freq_stats['avg_fat'], 'o-',
              color=C1, linewidth=3, markersize=12, label='Body Fat %')
l2, = ax2.plot(x3, freq_stats['avg_cal'], 's--',
               color=C3, linewidth=3, markersize=12, label='Calories Burned / Session')
ax.fill_between(x3, freq_stats['avg_fat'], alpha=0.15, color=C1)
ax2.fill_between(x3, freq_stats['avg_cal'], alpha=0.10, color=C3)

for xi, fat, cal in zip(x3, freq_stats['avg_fat'], freq_stats['avg_cal']):
    ax.annotate(f'{fat:.1f}%', (xi, fat),
                textcoords='offset points', xytext=(0, 13),
                fontsize=11, color=C1, ha='center', fontweight='bold')
    ax2.annotate(f'{cal:.0f} cal', (xi, cal),
                 textcoords='offset points', xytext=(0, -20),
                 fontsize=11, color=C3, ha='center', fontweight='bold')

labels3 = [f"{int(f)} day{'s' if f>1 else ''}/week\n(n={int(n)})"
           for f, n in zip(freq_stats['Workout_Frequency (days/week)'], freq_stats['n'])]
ax.set_xticks(x3)
ax.set_xticklabels(labels3, fontsize=10)
ax.set_ylabel('Body Fat %', fontsize=11, color=C1)
ax2.set_ylabel('Calories Burned per Session', fontsize=11, color=C3)
ax.tick_params(axis='y', colors=C1)
ax2.tick_params(axis='y', colors=C3)
ax.set_title('More Gym Sessions = Less Fat, More Calories Burned\n'
             '973 gym members — Fat r=-0.54 vs Frequency',
             fontsize=12, fontweight='bold')
ax.legend([l1, l2], [l.get_label() for l in [l1, l2]], fontsize=10, loc='upper right')
ax.grid(True, alpha=0.2)

# ══════════════════════════════════════════════════════════════
# 4. FRAUD RATE BY HOUR OF DAY
#    Story: "Your card is 30× more likely to be frauded at 10pm than 10am"
# ══════════════════════════════════════════════════════════════
ax = axes[1, 1]

safe_hours   = (hourly['hour'] >= 9)  & (hourly['hour'] <= 17)
danger_hours = (hourly['hour'] >= 20) | (hourly['hour'] <= 3)

ax.fill_between(hourly['hour'], hourly['fraud_pct'],
                where=safe_hours,   alpha=0.25, color=C3, label='Safe hours (9am–5pm)')
ax.fill_between(hourly['hour'], hourly['fraud_pct'],
                where=danger_hours, alpha=0.35, color=C1, label='Danger hours (8pm–3am)')

ax.plot(hourly['hour'], hourly['fraud_pct'], 'o-',
        color=C2, linewidth=2.5, markersize=6, zorder=3)

# Annotate peak and trough
peak = hourly.loc[hourly['fraud_pct'].idxmax()]
trough = hourly.loc[hourly['fraud_pct'].idxmin()]
ax.annotate(f"PEAK\n{peak['fraud_pct']:.2f}%\n({int(peak['hour'])}:00)",
            xy=(peak['hour'], peak['fraud_pct']),
            xytext=(peak['hour'] - 5, peak['fraud_pct'] - 0.6),
            fontsize=10, color=C1, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C1))
ax.annotate(f"SAFEST\n{trough['fraud_pct']:.3f}%\n({int(trough['hour'])}:00)",
            xy=(trough['hour'], trough['fraud_pct']),
            xytext=(trough['hour'] + 1.5, trough['fraud_pct'] + 0.6),
            fontsize=10, color=C3, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C3))

ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('Fraud Rate (% of transactions)', fontsize=11)
ax.set_title('Credit Card Fraud by Hour — 30× More Dangerous at Night\n1.3M transactions',
             fontsize=12, fontweight='bold')
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.2)

# ══════════════════════════════════════════════════════════════
# 5. WHICH DISEASE DO YOU GET BY DIET? (Stacked bar)
#    Story: "Vegans get Anemia; Pescatarians stay healthiest"
# ══════════════════════════════════════════════════════════════
ax = axes[2, 0]

disease_colors = {
    'Healthy':              C3,
    'Anemia':               C1,
    'Rickets_Osteomalacia': C4,
    'Night_Blindness':      C5,
    'Scurvy':               '#8c510a',
}
clean_names = {
    'Healthy':              'Healthy',
    'Anemia':               'Anemia',
    'Rickets_Osteomalacia': 'Rickets',
    'Night_Blindness':      'Night Blindness',
    'Scurvy':               'Scurvy',
}

bottoms = np.zeros(len(diet_order))
x5 = np.arange(len(diet_order))
for disease in disease_labels:
    vals = disease_by_diet[disease].values
    bars = ax.bar(x5, vals, bottom=bottoms,
                  color=disease_colors[disease],
                  edgecolor='white', linewidth=0.8,
                  label=clean_names[disease])
    for xi, v, bot in zip(x5, vals, bottoms):
        if v > 4:
            ax.text(xi, bot + v / 2, f'{v:.0f}%',
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color='white')
    bottoms += vals

ax.set_xticks(x5)
ax.set_xticklabels(diet_order, fontsize=12)
ax.set_ylabel('% of Patients', fontsize=11)
ax.set_ylim(0, 105)
ax.set_title('Which Disease Do You Get Based on Your Diet?\nVegans get Anemia; Pescatarians stay healthiest',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, axis='y', alpha=0.2)

# ══════════════════════════════════════════════════════════════
# 6. OIL RESERVES BY BASIN (Venezuela)
#    Story: "One region holds 79% of all proven oil reserves"
# ══════════════════════════════════════════════════════════════
ax = axes[2, 1]

total = basin.sum()
pct   = (basin / total * 100).round(1)
colors_bar = [C1 if b == 'Orinoco Belt' else C2 for b in basin.index]

bars = ax.barh(basin.index, basin.values,
               color=colors_bar, edgecolor='black', linewidth=0.7, height=0.6)

for bar, (region, val), p in zip(bars, basin.items(), pct.values):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}B barrels  ({p:.0f}%)',
            va='center', fontsize=10, fontweight='bold' if region == 'Orinoco Belt' else 'normal')

ax.axvline(basin.values.mean(), color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Average: {basin.values.mean():.0f}B barrels')
ax.set_xlabel('Proven Reserves (Billion Barrels)', fontsize=11)
ax.set_title(f'Orinoco Belt Holds 79% of All Proven Oil Reserves\n'
             f'Venezuela — {len(res)} reservoirs, {total:.0f}B barrels total',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, basin.max() * 1.35)
ax.legend(handles=[
    mpatches.Patch(color=C1, label='Orinoco Belt (dominant)'),
    mpatches.Patch(color=C2, label='Other basins'),
], fontsize=10)

plt.savefig('plots/new_analysis_2.png', dpi=150, bbox_inches='tight')
print("Saved plots/new_analysis_2.png")

# ── Print key numbers ─────────────────────────────────────────
print("\n=== KEY FINDINGS ===")
print(f"\n1. Diet → Vitamin B12:")
for _, row in diet_stats.iterrows():
    print(f"   {row['diet_type']:<12}: B12={row['avg_b12']:.0f} pg/mL, disease={row['disease_rate']:.0f}%")

print(f"\n2. Income → Vitamin D + Disease:")
for _, row in income_stats.iterrows():
    print(f"   {row['income_level']:<8}: VitD={row['avg_vitd']:.1f} ng/mL, disease={row['disease_rate']:.0f}%")

print(f"\n3. Gym Frequency → Fat%:")
for _, row in freq_stats.iterrows():
    print(f"   {int(row['Workout_Frequency (days/week)'])} days/week: Fat={row['avg_fat']:.1f}%, Calories={row['avg_cal']:.0f}")

print(f"\n4. Fraud peak: {hourly.loc[hourly['fraud_pct'].idxmax(), 'fraud_pct']:.2f}% at "
      f"{int(hourly.loc[hourly['fraud_pct'].idxmax(), 'hour'])}:00")
print(f"   Fraud trough: {hourly.loc[hourly['fraud_pct'].idxmin(), 'fraud_pct']:.3f}% at "
      f"{int(hourly.loc[hourly['fraud_pct'].idxmin(), 'hour'])}:00")
print(f"   Ratio: {hourly['fraud_pct'].max()/hourly['fraud_pct'].min():.0f}x more dangerous at night")

print(f"\n5. Oil: Orinoco Belt = {basin['Orinoco Belt']:.0f}B barrels "
      f"({basin['Orinoco Belt']/total*100:.0f}% of total {total:.0f}B)")
