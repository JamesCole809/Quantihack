# UK Road Accidents x Weather Report

## Project Scope

This folder is the final-project workspace for the UK road accidents dataset and the UK weather dataset.
The analysis merges both sources at **daily national level** across the shared **2009-01-01 to 2020-12-31** window.

## Data Used

- Road accidents: 2005-2015 legacy STATS19 accident file + 2016-2020 recent accident file, restricted to the 2009-2020 overlap window
- Weather: daily weather observations aggregated from the selected weather dataset into one national daily weather profile

Key scale:

- Total accidents analysed: **1,637,020**
- Daily observations in merged table: **4,380**
- Average accidents per day: **373.7**
- Average weather locations contributing each day: **503.1**

## Main Findings

1. **The merge is credible.** Daily rainfall anomalies align strongly with the share of crashes police coded as rainy-weather crashes: **r = 0.831**. Wind anomalies also align well with police high-wind coding: **r = 0.678**.

2. **Rain changes severity more than volume.** After comparing wet vs dry days *within the same month*, wetter days showed almost no change in accident count (**-0.1%**) but noticeably lower fatal share (**-12.6%**) and lower serious-or-fatal share (**-10.6%**). The pattern suggests rain increases caution and lowers impact speed even when collisions still happen.

3. **Warmer-than-normal days are more severe.** Maximum-temperature anomalies correlate positively with serious-or-fatal accident share (**r = 0.248**). The hottest 10% of days had a **+26.7%** higher serious-or-fatal share than the coldest 10% of days.

4. **Cloudier-than-normal days have fewer crashes.** Accident-count anomalies are negatively correlated with cloud-cover anomalies (**r = -0.219**), while darker/cloudier conditions move together (**r = 0.438**). A plausible interpretation is lower exposure or more cautious driving under dull conditions.

5. **Weekend crashes are fewer but harsher.** Weekend days had **-20.2%** fewer crashes than weekdays, but the fatal share was **+49.8%** higher. That finding is not caused by weather alone, but it is useful project context when discussing severity.

## Files Generated

- `data/processed/daily_uk_accidents_weather.parquet`: merged daily modelling table
- `data/processed/monthly_uk_accidents_weather.csv`: monthly summary table
- `data/processed/wet_vs_dry_by_month.csv`: within-month wet/dry comparison table
- `outputs/uk_accidents_weather_dashboard.png`: project dashboard
- `outputs/summary_metrics.json`: machine-readable metrics used in this report

## Important Limitation

The weather dataset does not include station coordinates in this project workspace, and some location names appear broader than a strict UK-only list. Because of that, the analysis is intentionally framed as a **daily national weather proxy** rather than a precise station-to-accident spatial join. The findings are useful for project-level interpretation, but they should not be presented as location-level causal proof.
