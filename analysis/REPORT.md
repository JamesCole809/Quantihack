# Cross-Dataset Correlation Analysis Report

---

## Overview

This report analyses correlations across six independent datasets covering pandemics, blood cell morphology, chess games, road accident fatalities, asylum seekers, and music reviews. Datasets were first examined individually for internal patterns, then compared against each other using **year** as a shared key (2012–2016 overlap window).

---

## 1. Pandemics Dataset

**Source:** Historical Pandemic & Epidemic Dataset (50 events, 541 AD – 2023)

### Transmission Mode vs Lethality

The most dangerous transmission routes are **contact** (mean CFR 36.5%) and **sexual/blood** (47.6%). Surprisingly, **airborne and droplet** diseases — the ones most associated with mass spread — have the *lowest* average fatality rate at just 6.5%.

| Transmission | Mean Fatality Rate | Mean Spread Score |
|---|---|---|
| Airborne/Droplet | 6.5% | Low |
| Vector | 25.5% | Medium |
| Waterborne/Fecal | 21.5% | Medium |
| Contact | 36.5% | Low |
| Sexual/Blood | 47.6% | High |

> **Insight:** Diseases that spread more easily tend to kill less efficiently. This is consistent with evolutionary pressure — a pathogen that kills its host too fast doesn't spread.

---

### Containment Method Effectiveness

Vaccines and antibiotics are by far the most effective containment strategies, reducing mean fatality rates to **2.7%** and **4.6%** respectively. Outbreaks left to **natural decline** averaged a 32.4% fatality rate.

| Containment | Mean CFR | Mean Deaths |
|---|---|---|
| Vaccine | 2.7% | ~958,000 |
| Antibiotics | 4.6% | ~78 |
| Quarantine | 35.9% | ~7.4M |
| Natural Decline | 32.4% | ~19.7M |

---

### Virus vs Bacteria

- **Viruses** show a stronger positive correlation between spread and lethality (Spearman ρ higher)
- **Bacteria** tend to cluster with higher fatality rates but lower spread scores
- Duration of outbreak (bubble size) does not strongly predict fatality

---

## 2. Blood Cell Anomaly Dataset

**Source:** 5,880 blood cell observations with morphological measurements and disease labels

### PCA of Cell Morphology by Disease

Principal component analysis of 11 morphological features shows **clear clustering** by disease category. Leukemia and Sickle Cell Anemia cells separate cleanly from normal cells on PC1 (which explains the most variance). Normal WBC, RBC, and Platelet cells overlap heavily, reflecting their structural similarity.

---

### Features That Distinguish Leukemia from Normal WBC

The Mann-Whitney U test with effect sizes reveals which physical measurements differ most between Leukemia cells and normal white blood cells:

| Feature | Effect Size | Direction |
|---|---|---|
| Chromatin Density | +0.872 | Higher in Leukemia |
| Granularity Score | +0.846 | Higher in Leukemia |
| Cytoplasm Ratio | +0.811 | Higher in Leukemia |
| Nucleus Area % | -0.808 | **Lower** in Leukemia |
| Lobularity Score | +0.634 | Higher in Leukemia |
| Cell Diameter | -0.572 | **Smaller** in Leukemia |

> **Insight:** Leukemia cells are denser and more granular but paradoxically smaller with a reduced nucleus. This morphological "fingerprint" is highly statistically significant (all p < 10⁻²⁰).

---

## 3. Chess Games Dataset

**Source:** 20,058 Lichess games (2013–2017)

### Opening Choice and Win Rates

Not all openings are equal. The **English Opening** gives white the highest win advantage (54.9%), while the **Sicilian Defense** — the most played at high levels — actually favours Black, with White winning only 45.6% of the time.

| Opening | White Win % | Black Win % | Avg Turns |
|---|---|---|---|
| English Opening | **54.9%** | 41.1% | 62.8 |
| Scotch Game | 54.0% | 42.8% | 56.0 |
| Queen's Gambit Declined | 53.6% | 41.9% | 65.9 |
| Sicilian Defense | 45.6% | **49.5%** | 62.7 |
| French Defense | 49.1% | 45.9% | 61.7 |

---

### Opening Depth vs Game Outcome

Deeper openings (more prepared moves) produce **longer games** and higher draw rates. Shallow openings (1–4 ply) see more decisive results and favour White, while games with 13+ opening moves are far more likely to end in a draw.

---

## 4. FARS Accident Dataset

**Source:** Arizona fatal crashes, 2012–2016 (1,860 accidents)

### Time-of-Day Patterns

- **Peak crash volume:** 18:00 (rush hour) — 140 crashes
- **Peak alcohol involvement:** 01:00 — **66.7%** of crashes at 1am involve alcohol
- There is a clear double-peak in crashes (morning and evening commutes) with a separate alcohol-driven peak late at night
- Daylight hours are high in crash *count* but low in alcohol involvement

> **Insight:** Time of day is a strong predictor of crash cause. Traffic volume drives daytime crashes; alcohol drives nighttime crashes.

---

## 5. Pitchfork Reviews Dataset

**Source:** ~18,000 album reviews, 1999–2017 (SQLite)

### Genre Score Rankings

Pitchfork consistently scores **experimental and global music highest**, and **pop/R&B lowest**. This reflects the publication's known bias toward critical favourites over mainstream commercial music.

| Genre | Avg Score |
|---|---|
| Experimental | 7.35 |
| Folk/Country | 7.20 |
| Metal | 6.96 |
| Rock | 6.95 |
| Electronic | 6.92 |
| Rap | 6.90 |
| Pop/R&B | **6.89** |

---

## 6. Cross-Dataset Correlations

Datasets were aggregated to annual totals and merged on year (2012–2016 window).

### Raw Correlations

| Pair | Pearson r | Spearman ρ | Verdict |
|---|---|---|---|
| FARS Alcohol Crashes × Chess Rating Gap | **-0.995** | -1.000 | Spurious |
| Asylum Seekers × Chess White Rating | +0.896 | +1.000 | Spurious |
| Asylum Seekers × Pitchfork Avg Score | +0.900 | +1.000 | Spurious |
| Chess Draw Rate × Pitchfork Score | +0.923 | +0.900 | Borderline |

### Detrending Test (Removing the Shared Year Effect)

To test whether these correlations are genuine or just driven by a shared upward/downward trend over time, we removed the linear time trend from each series before correlating.

| Pair | Raw r | Detrended r | Conclusion |
|---|---|---|---|
| Alcohol Crashes × Chess Rating Gap | -0.995 | **-0.998** | Holds — both declining in sync |
| Asylum Seekers × Chess Ratings | +0.896 | **-0.334** | Collapses — pure shared trend |

### First-Difference Test (Year-over-Year Changes)

Taking annual changes (Δ) rather than levels removes all trend contamination:

| Pair | r (first diff) | p-value | Conclusion |
|---|---|---|---|
| Alcohol Crashes Δ × Chess Rating Gap Δ | **-0.999** | 0.029 | **Statistically significant** |
| Asylum Seekers Δ × Chess Ratings Δ | -0.403 | 0.736 | Not significant |
| Asylum Seekers Δ × Pitchfork Score Δ | -0.779 | 0.432 | Not significant |

---

## Key Findings

1. **The only genuine cross-dataset correlation** is **FARS alcohol-impaired crashes vs chess rating gap** — it survives detrending and first-differencing (r = -0.999, p = 0.029). Both declined in parallel from 2013–2016. This is almost certainly **spurious** given the tiny sample (n = 4 years), but it is numerically the strongest signal.

2. All other cross-dataset correlations **collapse when detrended** — they are artefacts of both series trending in the same direction during 2012–2016, not a genuine relationship.

3. **Within-dataset**, the strongest genuine findings are:
   - Blood cells: chromatin density and granularity reliably distinguish Leukemia from Normal WBC
   - Chess: English Opening gives white a measurable advantage; Sicilian favours Black
   - Pandemics: vaccines/antibiotics reduce fatality rates 10x vs natural decline
   - FARS: 1am is the most dangerous hour for alcohol-related fatalities

---

## Charts

All charts are saved in [plots/](plots/):

| File | Contents |
|---|---|
| `deep_analysis.png` | 12-panel deep dive (transmission, PCA, openings, crash hours, genre trends, detrending) |
| `cross_dataset_linegraph.png` | 6 cross-dataset pairs as dual-axis line graphs |
| `cross_dataset_final.png` | Full Spearman heatmap + ranked cross-dataset correlations |
| `correlation_results.png` | Initial within-dataset correlation charts |
