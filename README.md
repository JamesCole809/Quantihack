# Quantihack

Cross-dataset correlation analysis — finding spurious (and sometimes hilarious) correlations between completely unrelated datasets.

## Project Structure

```
Quantihack/
├── data/                          # All datasets
│   ├── air_quality_france/        # French air quality measurements (2019-2020)
│   ├── air_quality_france_2/      # French air quality (duplicate)
│   ├── asylum_spain/              # Spanish asylum applications (2012-2021)
│   ├── chess_games/               # Lichess chess games (2013-2017)
│   ├── climate_soil/              # US climate & soil time series (2000-2016)
│   ├── energy_consumption/        # Building energy consumption (2016)
│   ├── f1_laptimes/               # F1 2026 lap times & telemetry
│   ├── pandemic_data/             # Historical pandemic/epidemic data
│   ├── solar_weather/             # Solar irradiance & weather (2017-2019)
│   ├── train_delays/              # Train delay data (2016)
│   ├── uk_accidents_2005_2010/    # UK road accidents (2005-2010)
│   ├── uk_accidents_2021_2022/    # UK road accidents (2021-2022)
│   ├── us_accidents_fars/         # US fatal accidents FARS (2012-2016)
│   ├── us_flights/                # US flight data (2018-2024)
│   └── world_coal_production/     # Global coal production (1981-2021)
├── scripts/                       # Analysis & plotting scripts
├── plots/                         # Generated visualisations
├── notebooks/                     # Jupyter notebooks
├── analysis/                      # Reports
└── README.md
```

## Key Findings

### Best Cross-Dataset Correlations (by year)

| r | n | Dataset A | Dataset B | Columns |
|---|---|-----------|-----------|---------|
| -0.995 | 6 | UK Accidents | Coal Production | accident count vs coal max production |
| -0.990 | 6 | UK Accidents | Coal Production | accident count vs coal avg production |
| 0.959 | 6 | Asylum (Spain) | ~~Pitchfork~~ | asylum women vs avg music score |
| -0.961 | 5 | Asylum (Spain) | Climate | peak asylum region vs avg wind speed |
| 0.941 | 6 | Climate | UK Accidents | min temperature vs avg vehicles per crash |

### Best Z-Score Triple Correlations

| r | Pair (A x B) | Predicts C |
|---|-------------|------------|
| -0.9998 | Chess skill x Wind speed | Spanish asylum applications |
| 0.9982 | US crash vehicle age x Atm. pressure | Chess opening depth |
| 0.9855 | UK crash vehicles x Coal production | Average temperature |

All correlations are spurious — no causal relationships exist between these datasets.

## Running Analysis

```bash
cd Quantihack
python scripts/full_correlation3.py    # Main cross-archive correlation
python scripts/zscore_hunt.py          # Z-score product/ratio analysis
python scripts/plot_zscore3.py         # Generate z-score plots
```
