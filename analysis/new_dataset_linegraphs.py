"""
Simple, presentable line graphs from the new datasets.
All comparisons are cross-dataset where possible and easy to explain.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "../data/"

# ── Load ──────────────────────────────────────────────────────
gym   = pd.read_csv(f"{DATA_DIR}archive (4)/gym_members_exercise_tracking.csv")
vit   = pd.read_csv(f"{DATA_DIR}archive (6)/vitamin_deficiency_disease_dataset_20260123.csv")
wh    = pd.read_csv(f"{DATA_DIR}archive (8)/weight-height.csv")
fraud = pd.read_csv(f"{DATA_DIR}archive (3)/train.csv",
                    usecols=['trans_date_trans_time','amt','category','is_fraud'])
fraud['date']  = pd.to_datetime(fraud['trans_date_trans_time'])
fraud['month'] = fraud['date'].dt.to_period('M').dt.to_timestamp()
casino = pd.read_csv(f"{DATA_DIR}archive (2)/online_casino_games_dataset_v2.csv")
casino['house_edge_pct'] = 100 - casino['rtp']

fig, axes = plt.subplots(3, 2, figsize=(18, 22))
fig.suptitle("New Dataset Correlations — Line Graphs", fontsize=17, fontweight='bold', y=1.01)
plt.subplots_adjust(hspace=0.48, wspace=0.35)

C1, C2, C3 = '#d73027', '#1f78b4', '#33a02c'
C4, C5     = '#ff7f00', '#9970ab'

# ══════════════════════════════════════════════════════════════
# 1. AGE → BMI + BLOOD PRESSURE  (Weight-Height, 50k people)
#    Story: "As you age, your BMI and blood pressure both rise"
# ══════════════════════════════════════════════════════════════
ax = axes[0, 0]
wh['age_bin'] = pd.cut(wh['Age'], bins=[18,25,35,45,55,65,80],
                        labels=['18–25','26–35','36–45','46–55','56–65','65–80'])
age_stats = wh.groupby('age_bin', observed=True).agg(
    avg_bmi    = ('BMI',        'mean'),
    avg_sys_bp = ('Systolic_BP','mean'),
    n          = ('BMI',        'count'),
).reset_index()

ax2 = ax.twinx()
l1, = ax.plot(range(len(age_stats)), age_stats['avg_bmi'], 'o-',
              color=C1, linewidth=2.8, markersize=9, label='Average BMI')
l2, = ax2.plot(range(len(age_stats)), age_stats['avg_sys_bp'], 's--',
               color=C2, linewidth=2.8, markersize=9, label='Systolic Blood Pressure')
ax.fill_between(range(len(age_stats)), age_stats['avg_bmi'], alpha=0.12, color=C1)
ax2.fill_between(range(len(age_stats)), age_stats['avg_sys_bp'], alpha=0.12, color=C2)

r_bmi, _ = stats.pearsonr(wh['Age'], wh['BMI'])
r_bp,  _ = stats.pearsonr(wh['Age'], wh['Systolic_BP'])

ax.set_xticks(range(len(age_stats)))
ax.set_xticklabels(age_stats['age_bin'].astype(str), fontsize=10)
ax.set_xlabel('Age Group', fontsize=11)
ax.set_ylabel('Average BMI', fontsize=11, color=C1)
ax2.set_ylabel('Systolic Blood Pressure (mmHg)', fontsize=11, color=C2)
ax.tick_params(axis='y', colors=C1)
ax2.tick_params(axis='y', colors=C2)
ax.set_title(f'As You Age, BMI and Blood Pressure Rise\n'
             f'50,000 people  |  BMI r={r_bmi:+.2f}, BP r={r_bp:+.2f}',
             fontsize=12, fontweight='bold')
ax.legend([l1,l2],[l.get_label() for l in [l1,l2]], fontsize=10, loc='upper left')
ax.grid(True, alpha=0.2)

# ══════════════════════════════════════════════════════════════
# 2. GYM EXPERIENCE → CALORIES BURNED + SESSION LENGTH
#    Story: "The more experienced you are, the harder you train"
# ══════════════════════════════════════════════════════════════
ax = axes[0, 1]
exp_map  = {1:'Beginner', 2:'Intermediate', 3:'Advanced'}
gym['exp_label'] = gym['Experience_Level'].map(exp_map)
exp_stats = gym.groupby('Experience_Level').agg(
    avg_calories = ('Calories_Burned',         'mean'),
    avg_duration = ('Session_Duration (hours)', 'mean'),
    avg_freq     = ('Workout_Frequency (days/week)','mean'),
    avg_fat      = ('Fat_Percentage',           'mean'),
    n            = ('Calories_Burned',           'count'),
).reset_index()

ax2 = ax.twinx()
l1, = ax.plot([0,1,2], exp_stats['avg_calories'], 'o-',
              color=C1, linewidth=2.8, markersize=11, label='Calories Burned / Session')
l2, = ax2.plot([0,1,2], exp_stats['avg_fat'], 's--',
               color=C3, linewidth=2.8, markersize=11, label='Body Fat %')
for xi, cal, fat in zip([0,1,2], exp_stats['avg_calories'], exp_stats['avg_fat']):
    ax.annotate(f'{cal:.0f} cal', (xi, cal), textcoords='offset points',
                xytext=(0, 12), fontsize=10, color=C1, ha='center', fontweight='bold')
    ax2.annotate(f'{fat:.1f}%', (xi, fat), textcoords='offset points',
                 xytext=(0,-18), fontsize=10, color=C3, ha='center', fontweight='bold')

r_cal, _ = stats.pearsonr(gym['Experience_Level'], gym['Calories_Burned'])
r_fat, _ = stats.pearsonr(gym['Experience_Level'], gym['Fat_Percentage'])
ax.fill_between([0,1,2], exp_stats['avg_calories'], alpha=0.12, color=C1)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Beginner', 'Intermediate', 'Advanced'], fontsize=11)
ax.set_ylabel('Calories Burned per Session', fontsize=11, color=C1)
ax2.set_ylabel('Body Fat %', fontsize=11, color=C3)
ax.tick_params(axis='y', colors=C1)
ax2.tick_params(axis='y', colors=C3)
ax.set_title(f'Gym Experience → More Calories Burned, Less Fat\n'
             f'973 members  |  Calories r={r_cal:+.2f}, Fat r={r_fat:+.2f}',
             fontsize=12, fontweight='bold')
ax.legend([l1,l2],[l.get_label() for l in [l1,l2]], fontsize=10, loc='center left')
ax.grid(True, alpha=0.2)

# ══════════════════════════════════════════════════════════════
# 3. SUN EXPOSURE → VITAMIN D + DISEASE RATE  (Vitamin dataset)
#    Story: "More sun = nearly double the Vitamin D, half the disease"
# ══════════════════════════════════════════════════════════════
ax = axes[1, 0]
# sun_exposure is categorical: Low / Moderate / High
sun_order = ['Low', 'Moderate', 'High']
sun_stats = vit.groupby('sun_exposure').agg(
    avg_vitd     = ('serum_vitamin_d_ng_ml','mean'),
    disease_rate = ('disease_diagnosis',    lambda x: (x!='Healthy').mean()*100),
).reindex(sun_order).reset_index()

ax2 = ax.twinx()
l1, = ax.plot(range(3), sun_stats['avg_vitd'], 'o-',
              color=C4, linewidth=3, markersize=12, label='Serum Vitamin D (ng/mL)')
l2, = ax2.plot(range(3), sun_stats['disease_rate'], 's--',
               color=C1, linewidth=3, markersize=12, label='Disease Rate (%)')
ax.fill_between(range(3), sun_stats['avg_vitd'], alpha=0.15, color=C4)
ax2.fill_between(range(3), sun_stats['disease_rate'], alpha=0.15, color=C1)

for xi, vd, dr in zip(range(3), sun_stats['avg_vitd'], sun_stats['disease_rate']):
    ax.annotate(f'{vd:.1f} ng/mL', (xi, vd), textcoords='offset points',
                xytext=(0, 13), fontsize=11, color=C4, ha='center', fontweight='bold')
    ax2.annotate(f'{dr:.0f}%\ndiseased', (xi, dr), textcoords='offset points',
                 xytext=(0,-22), fontsize=10, color=C1, ha='center', fontweight='bold')

# encode Low=0, Moderate=1, High=2 for correlation
sun_enc = vit['sun_exposure'].map({'Low':0,'Moderate':1,'High':2})
r_sun, _ = stats.pearsonr(sun_enc, vit['serum_vitamin_d_ng_ml'])
ax.set_xticks(range(3))
ax.set_xticklabels(['Low Sun', 'Moderate Sun', 'High Sun'], fontsize=11)
ax.set_ylabel('Avg Serum Vitamin D (ng/mL)', fontsize=11, color=C4)
ax2.set_ylabel('Disease Rate (%)', fontsize=11, color=C1)
ax.tick_params(axis='y', colors=C4)
ax2.tick_params(axis='y', colors=C1)
ax.set_title(f'More Sunlight → Nearly Double Vitamin D, Less Disease\n'
             f'4,000 patients  |  Sun vs Vitamin D r={r_sun:+.2f}',
             fontsize=12, fontweight='bold')
ax.legend([l1,l2],[l.get_label() for l in [l1,l2]], fontsize=10, loc='center right')
ax.grid(True, alpha=0.2)

# ══════════════════════════════════════════════════════════════
# 4. CREDIT CARD FRAUD RATE OVER TIME  (Fraud dataset)
#    Story: "Fraud spikes in winter — and online shopping is the target"
# ══════════════════════════════════════════════════════════════
ax = axes[1, 1]
monthly = fraud.groupby('month').agg(
    fraud_rate = ('is_fraud', 'mean'),
    avg_fraud_amt = ('amt', lambda x: x[fraud.loc[x.index,'is_fraud']==1].mean()),
    total_txns = ('is_fraud', 'count'),
).reset_index()

ax2 = ax.twinx()
l1, = ax.plot(monthly['month'], monthly['fraud_rate']*100, 'o-',
              color=C1, linewidth=2.5, markersize=7, label='Fraud Rate (%)')
l2, = ax2.plot(monthly['month'], monthly['avg_fraud_amt'], 's--',
               color=C2, linewidth=2.5, markersize=7, label='Avg Fraud Amount ($)')
ax.fill_between(monthly['month'], monthly['fraud_rate']*100, alpha=0.12, color=C1)

# Annotate peak
peak_idx = monthly['fraud_rate'].idxmax()
peak_row  = monthly.loc[peak_idx]
ax.annotate(f"Peak: {peak_row['fraud_rate']*100:.2f}%",
            xy=(peak_row['month'], peak_row['fraud_rate']*100),
            xytext=(monthly['month'].iloc[3], monthly['fraud_rate'].max()*100+0.2),
            fontsize=10, color=C1, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C1))

ax.set_ylabel('Fraud Rate (% of transactions)', fontsize=11, color=C1)
ax2.set_ylabel('Avg Fraudulent Transaction ($)', fontsize=11, color=C2)
ax.tick_params(axis='y', colors=C1)
ax2.tick_params(axis='y', colors=C2)
ax.set_title('Credit Card Fraud Rate Over Time\n1.3M transactions, 2019–2020',
             fontsize=12, fontweight='bold')
ax.legend([l1,l2],[l.get_label() for l in [l1,l2]], fontsize=10, loc='upper left')
ax.grid(True, alpha=0.2)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

# ══════════════════════════════════════════════════════════════
# 5. FRAUD RATE BY CATEGORY  (bar-style line for comparison)
#    Story: "Online shopping is 9× more fraud-prone than in-store"
# ══════════════════════════════════════════════════════════════
ax = axes[2, 0]
cat_fraud = (fraud.groupby('category')
    .agg(fraud_rate=('is_fraud','mean'), avg_amt=('amt','mean'))
    .sort_values('fraud_rate', ascending=False)
    .reset_index())
cat_fraud['fraud_pct'] = cat_fraud['fraud_rate'] * 100

# Colour online vs offline
online_cats = {'shopping_net','misc_net','grocery_net'}
colors_bar  = [C1 if c in online_cats else C2 for c in cat_fraud['category']]
clean_labels = [c.replace('_',' ').title() for c in cat_fraud['category']]

bars = ax.barh(range(len(cat_fraud)), cat_fraud['fraud_pct'],
               color=colors_bar, edgecolor='black', linewidth=0.5, height=0.7)
for bar, val in zip(bars, cat_fraud['fraud_pct']):
    ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=9.5, fontweight='bold')

import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(color=C1, label='Online transaction'),
    mpatches.Patch(color=C2, label='In-person transaction'),
], fontsize=10, loc='lower right')
ax.set_yticks(range(len(cat_fraud)))
ax.set_yticklabels(clean_labels, fontsize=9)
ax.set_xlabel('Fraud Rate (%)', fontsize=11)
ax.set_title('Online Shopping is the Biggest Fraud Target\nFraud rate by spending category',
             fontsize=12, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

# ══════════════════════════════════════════════════════════════
# 6. TRANSACTION AMOUNT → FRAUD RISK  (Fraud dataset)
#    Story: "Large transactions are 100× more likely to be fraud"
# ══════════════════════════════════════════════════════════════
ax = axes[2, 1]

fraud['amt_bin'] = pd.cut(fraud['amt'],
                           bins=[0, 20, 100, 500, fraud['amt'].max()+1],
                           labels=['$0–20\n(small)', '$20–100\n(medium)',
                                   '$100–500\n(large)', '$500+\n(very large)'])
amt_stats = fraud.groupby('amt_bin', observed=True).agg(
    fraud_rate = ('is_fraud','mean'),
    avg_amt    = ('amt','mean'),
    count      = ('is_fraud','count'),
).reset_index()
amt_stats['fraud_pct'] = amt_stats['fraud_rate'] * 100

ax2 = ax.twinx()
x4 = range(len(amt_stats))
l1, = ax.plot(x4, amt_stats['fraud_pct'], 'o-',
              color=C1, linewidth=3, markersize=12, label='Fraud Rate (%)')
l2, = ax2.plot(x4, amt_stats['avg_amt'], 's--',
               color=C2, linewidth=3, markersize=12, label='Avg Transaction ($)')
ax.fill_between(x4, amt_stats['fraud_pct'], alpha=0.15, color=C1)

for xi, fpct, avg in zip(x4, amt_stats['fraud_pct'], amt_stats['avg_amt']):
    ax.annotate(f'{fpct:.2f}%', (xi, fpct), textcoords='offset points',
                xytext=(0, 13), fontsize=11, color=C1, ha='center', fontweight='bold')
    ax2.annotate(f'${avg:.0f}', (xi, avg), textcoords='offset points',
                 xytext=(0,-20), fontsize=10, color=C2, ha='center', fontweight='bold')

ax.set_xticks(x4)
ax.set_xticklabels(amt_stats['amt_bin'].astype(str), fontsize=11)
ax.set_ylabel('Fraud Rate (%)', fontsize=11, color=C1)
ax2.set_ylabel('Average Transaction Amount ($)', fontsize=11, color=C2)
ax.tick_params(axis='y', colors=C1)
ax2.tick_params(axis='y', colors=C2)
ax.set_title('Bigger Transactions = Massively Higher Fraud Risk\n1.3M transactions — Credit Card Fraud dataset',
             fontsize=12, fontweight='bold')
ax.legend([l1,l2],[l.get_label() for l in [l1,l2]], fontsize=10, loc='upper left')
ax.grid(True, alpha=0.2)

plt.savefig('plots/new_dataset_linegraphs.png', dpi=150, bbox_inches='tight')
print("Saved plots/new_dataset_linegraphs.png")

# ── Key numbers ───────────────────────────────────────────────
print(f"\nBMI vs Age r={stats.pearsonr(wh['Age'],wh['BMI'])[0]:.3f}")
print(f"Experience vs Calories r={stats.pearsonr(gym['Experience_Level'],gym['Calories_Burned'])[0]:.3f}")
print(f"Online fraud rate: {fraud[fraud['category'].isin(online_cats)]['is_fraud'].mean()*100:.3f}%")
print(f"In-store fraud rate: {fraud[~fraud['category'].isin(online_cats)]['is_fraud'].mean()*100:.3f}%")
print(f"\nSun Exposure → Vitamin D:")
print(sun_stats[['sun_exposure','avg_vitd','disease_rate']])
sun_enc2 = vit['sun_exposure'].map({'Low':0,'Moderate':1,'High':2})
print(f"Sun vs Vitamin D r={stats.pearsonr(sun_enc2, vit['serum_vitamin_d_ng_ml'])[0]:.3f}")
print(f"\nTransaction Amount → Fraud Rate:")
print(amt_stats[['amt_bin','fraud_pct','avg_amt','count']])
