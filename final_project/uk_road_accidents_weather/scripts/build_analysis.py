from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

WEATHER_PATH = RAW_DIR / "weather_daily.parquet"
ACCIDENTS_2005_2015_PATH = RAW_DIR / "road_accidents_2005_2015.parquet"
ACCIDENTS_2016_2020_PATH = RAW_DIR / "road_accidents_2016_2020.parquet"
REPORT_PATH = PROJECT_DIR / "REPORT.md"

OVERLAP_START = pd.Timestamp("2009-01-01")
OVERLAP_END = pd.Timestamp("2020-12-31")


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_weather_daily() -> pd.DataFrame:
    weather = pd.read_parquet(WEATHER_PATH).rename(
        columns={
            "min_temp °c": "min_temp_c",
            "max_temp °c": "max_temp_c",
            "rain mm": "rain_mm",
            "humidity %": "humidity_pct",
            "cloud_cover %": "cloud_cover_pct",
            "wind_speed km/h": "wind_speed_kmh",
            "wind_direction_numerical": "wind_direction_deg",
        }
    )
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather[(weather["date"] >= OVERLAP_START) & (weather["date"] <= OVERLAP_END)].copy()

    daily = (
        weather.groupby("date")
        .agg(
            weather_locations=("location", "nunique"),
            avg_min_temp_c=("min_temp_c", "mean"),
            avg_max_temp_c=("max_temp_c", "mean"),
            avg_rain_mm=("rain_mm", "mean"),
            avg_humidity_pct=("humidity_pct", "mean"),
            avg_cloud_cover_pct=("cloud_cover_pct", "mean"),
            avg_wind_speed_kmh=("wind_speed_kmh", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["avg_temp_range_c"] = daily["avg_max_temp_c"] - daily["avg_min_temp_c"]
    return daily


def standardize_accidents(path: Path, legacy: bool) -> pd.DataFrame:
    raw = pd.read_parquet(path)

    if legacy:
        raw = raw.rename(
            columns={
                "Date": "date",
                "Accident_Severity": "accident_severity",
                "Number_of_Vehicles": "number_of_vehicles",
                "Number_of_Casualties": "number_of_casualties",
                "Weather_Conditions": "weather_conditions",
                "Road_Surface_Conditions": "road_surface_conditions",
                "Urban_or_Rural_Area": "urban_or_rural_area",
                "Light_Conditions": "light_conditions",
            }
        )

    columns = [
        "date",
        "accident_severity",
        "number_of_vehicles",
        "number_of_casualties",
        "weather_conditions",
        "road_surface_conditions",
        "urban_or_rural_area",
        "light_conditions",
    ]

    accidents = raw[columns].copy()
    accidents["date"] = pd.to_datetime(accidents["date"], dayfirst=True)
    return accidents


def load_daily_accidents() -> pd.DataFrame:
    legacy = standardize_accidents(ACCIDENTS_2005_2015_PATH, legacy=True)
    recent = standardize_accidents(ACCIDENTS_2016_2020_PATH, legacy=False)

    accidents = pd.concat([legacy, recent], ignore_index=True)
    accidents = accidents[
        (accidents["date"] >= OVERLAP_START) & (accidents["date"] <= OVERLAP_END)
    ].copy()

    accidents["is_fatal"] = (accidents["accident_severity"] == 1).astype(int)
    accidents["is_serious_or_fatal"] = accidents["accident_severity"].isin([1, 2]).astype(int)
    accidents["is_rain_code"] = accidents["weather_conditions"].isin([2, 5]).astype(int)
    accidents["is_high_wind_code"] = accidents["weather_conditions"].isin([4, 5, 6]).astype(int)
    accidents["is_fog_code"] = (accidents["weather_conditions"] == 7).astype(int)
    accidents["is_dark"] = accidents["light_conditions"].isin([4, 5, 6, 7]).astype(int)
    accidents["is_rural"] = (accidents["urban_or_rural_area"] == 2).astype(int)

    daily = (
        accidents.groupby("date")
        .agg(
            accidents=("accident_severity", "size"),
            total_vehicles=("number_of_vehicles", "sum"),
            total_casualties=("number_of_casualties", "sum"),
            mean_vehicles=("number_of_vehicles", "mean"),
            mean_casualties=("number_of_casualties", "mean"),
            fatal_count=("is_fatal", "sum"),
            serious_or_fatal_count=("is_serious_or_fatal", "sum"),
            rain_code_accidents=("is_rain_code", "sum"),
            high_wind_code_accidents=("is_high_wind_code", "sum"),
            fog_code_accidents=("is_fog_code", "sum"),
            dark_accidents=("is_dark", "sum"),
            rural_accidents=("is_rural", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )

    share_map = {
        "fatal_count": "fatal_share",
        "serious_or_fatal_count": "serious_or_fatal_share",
        "rain_code_accidents": "rain_code_share",
        "high_wind_code_accidents": "high_wind_code_share",
        "fog_code_accidents": "fog_code_share",
        "dark_accidents": "dark_share",
        "rural_accidents": "rural_share",
    }
    for count_col, share_col in share_map.items():
        daily[share_col] = daily[count_col] / daily["accidents"]

    return daily


def add_calendar_and_anomalies(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    merged["month_name"] = merged["date"].dt.month_name().str.slice(stop=3)
    merged["weekday_num"] = merged["date"].dt.dayofweek
    merged["weekday_name"] = merged["date"].dt.day_name()
    merged["is_weekend"] = merged["weekday_num"] >= 5

    anomaly_columns = [
        "accidents",
        "fatal_share",
        "serious_or_fatal_share",
        "rain_code_share",
        "high_wind_code_share",
        "dark_share",
        "avg_rain_mm",
        "avg_max_temp_c",
        "avg_humidity_pct",
        "avg_cloud_cover_pct",
        "avg_wind_speed_kmh",
    ]
    groups = ["month", "weekday_num"]
    for column in anomaly_columns:
        merged[f"{column}_anom"] = merged[column] - merged.groupby(groups)[column].transform("mean")

    return merged


def build_monthly_summary(merged: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        merged.groupby(["year", "month", "month_name"])
        .agg(
            total_accidents=("accidents", "sum"),
            mean_daily_accidents=("accidents", "mean"),
            mean_fatal_share=("fatal_share", "mean"),
            mean_serious_or_fatal_share=("serious_or_fatal_share", "mean"),
            mean_rain_mm=("avg_rain_mm", "mean"),
            mean_max_temp_c=("avg_max_temp_c", "mean"),
            mean_cloud_cover_pct=("avg_cloud_cover_pct", "mean"),
            mean_wind_speed_kmh=("avg_wind_speed_kmh", "mean"),
            mean_weather_locations=("weather_locations", "mean"),
        )
        .reset_index()
    )
    monthly["month_start"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )
    return monthly.sort_values("month_start")


def build_wet_vs_dry_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for month, group in merged.groupby("month"):
        dry_cut = group["avg_rain_mm"].quantile(0.2)
        wet_cut = group["avg_rain_mm"].quantile(0.8)

        bucketed = group.assign(
            rain_bucket=np.where(
                group["avg_rain_mm"] <= dry_cut,
                "dry",
                np.where(group["avg_rain_mm"] >= wet_cut, "wet", "mid"),
            )
        )
        summary = (
            bucketed[bucketed["rain_bucket"].isin(["dry", "wet"])]
            .groupby("rain_bucket")
            .agg(
                mean_accidents=("accidents", "mean"),
                mean_fatal_share=("fatal_share", "mean"),
                mean_serious_or_fatal_share=("serious_or_fatal_share", "mean"),
            )
        )

        if {"dry", "wet"}.issubset(summary.index):
            rows.append(
                {
                    "month": int(month),
                    "month_name": group["month_name"].iloc[0],
                    "wet_vs_dry_accidents_pct": (
                        summary.loc["wet", "mean_accidents"] / summary.loc["dry", "mean_accidents"] - 1
                    )
                    * 100,
                    "wet_vs_dry_fatal_pct": (
                        summary.loc["wet", "mean_fatal_share"] / summary.loc["dry", "mean_fatal_share"] - 1
                    )
                    * 100,
                    "wet_vs_dry_serious_or_fatal_pct": (
                        summary.loc["wet", "mean_serious_or_fatal_share"]
                        / summary.loc["dry", "mean_serious_or_fatal_share"]
                        - 1
                    )
                    * 100,
                }
            )

    wet_dry = pd.DataFrame(rows).sort_values("month")
    overall = pd.DataFrame(
        [
            {
                "month": 0,
                "month_name": "All",
                "wet_vs_dry_accidents_pct": wet_dry["wet_vs_dry_accidents_pct"].mean(),
                "wet_vs_dry_fatal_pct": wet_dry["wet_vs_dry_fatal_pct"].mean(),
                "wet_vs_dry_serious_or_fatal_pct": wet_dry["wet_vs_dry_serious_or_fatal_pct"].mean(),
            }
        ]
    )
    return pd.concat([wet_dry, overall], ignore_index=True)


def compute_summary_metrics(merged: pd.DataFrame, wet_vs_dry: pd.DataFrame) -> dict[str, float | int | str]:
    overall_wet_dry = wet_vs_dry.loc[wet_vs_dry["month_name"] == "All"].iloc[0]

    metrics: dict[str, float | int | str] = {
        "analysis_window_start": OVERLAP_START.strftime("%Y-%m-%d"),
        "analysis_window_end": OVERLAP_END.strftime("%Y-%m-%d"),
        "total_daily_rows": int(len(merged)),
        "total_accidents": int(merged["accidents"].sum()),
        "avg_daily_accidents": float(merged["accidents"].mean()),
        "avg_weather_locations_per_day": float(merged["weather_locations"].mean()),
        "rain_validation_corr": float(merged["avg_rain_mm_anom"].corr(merged["rain_code_share_anom"])),
        "wind_validation_corr": float(
            merged["avg_wind_speed_kmh_anom"].corr(merged["high_wind_code_share_anom"])
        ),
        "temp_vs_severe_corr": float(
            merged["avg_max_temp_c_anom"].corr(merged["serious_or_fatal_share_anom"])
        ),
        "cloud_vs_accidents_corr": float(
            merged["avg_cloud_cover_pct_anom"].corr(merged["accidents_anom"])
        ),
        "dark_vs_cloud_corr": float(merged["avg_cloud_cover_pct_anom"].corr(merged["dark_share_anom"])),
        "wet_vs_dry_accidents_pct": float(overall_wet_dry["wet_vs_dry_accidents_pct"]),
        "wet_vs_dry_fatal_pct": float(overall_wet_dry["wet_vs_dry_fatal_pct"]),
        "wet_vs_dry_serious_or_fatal_pct": float(overall_wet_dry["wet_vs_dry_serious_or_fatal_pct"]),
        "weekend_accident_delta_pct": float(
            (
                merged.loc[merged["is_weekend"], "accidents"].mean()
                / merged.loc[~merged["is_weekend"], "accidents"].mean()
                - 1
            )
            * 100
        ),
        "weekend_fatal_share_delta_pct": float(
            (
                merged.loc[merged["is_weekend"], "fatal_share"].mean()
                / merged.loc[~merged["is_weekend"], "fatal_share"].mean()
                - 1
            )
            * 100
        ),
    }

    hottest = merged["avg_max_temp_c"] >= merged["avg_max_temp_c"].quantile(0.9)
    coldest = merged["avg_max_temp_c"] <= merged["avg_max_temp_c"].quantile(0.1)
    metrics["hot_vs_cold_serious_or_fatal_pct"] = float(
        (
            merged.loc[hottest, "serious_or_fatal_share"].mean()
            / merged.loc[coldest, "serious_or_fatal_share"].mean()
            - 1
        )
        * 100
    )

    return metrics


def add_trend_line(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> None:
    coef = np.polyfit(x, y, deg=1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = np.polyval(coef, xs)
    ax.plot(xs, ys, color=color, linewidth=2)


def make_dashboard(
    merged: pd.DataFrame, monthly: pd.DataFrame, wet_vs_dry: pd.DataFrame, metrics: dict[str, float | int | str]
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "UK Road Accidents x Weather, 2009-2020",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    monthly_view = monthly.copy()
    ax = axes[0, 0]
    ax.plot(
        monthly_view["month_start"],
        monthly_view["total_accidents"],
        color="#b22222",
        linewidth=2,
        label="Monthly accidents",
    )
    ax.set_title("Monthly accident totals", fontweight="bold")
    ax.set_ylabel("Accidents")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(
        monthly_view["month_start"],
        monthly_view["mean_rain_mm"],
        color="#1f77b4",
        linewidth=2,
        alpha=0.85,
        label="Mean rain mm",
    )
    ax2.plot(
        monthly_view["month_start"],
        monthly_view["mean_max_temp_c"],
        color="#ff8c00",
        linewidth=2,
        alpha=0.85,
        label="Mean max temp C",
    )
    ax2.set_ylabel("Weather scale")
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper right", fontsize=9)

    ax = axes[0, 1]
    overall = wet_vs_dry[wet_vs_dry["month_name"] == "All"].iloc[0]
    labels = ["Accidents", "Fatal share", "Serious/fatal share"]
    values = [
        overall["wet_vs_dry_accidents_pct"],
        overall["wet_vs_dry_fatal_pct"],
        overall["wet_vs_dry_serious_or_fatal_pct"],
    ]
    colors = ["#4c78a8" if value >= 0 else "#d95f02" for value in values]
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Wet vs dry days within the same month", fontweight="bold")
    ax.set_ylabel("Percent change on wetter days")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (1.2 if value >= 0 else -1.2),
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    ax = axes[0, 2]
    x = merged["avg_rain_mm_anom"]
    y = merged["rain_code_share_anom"]
    ax.scatter(x, y, alpha=0.25, s=16, color="#1f77b4", edgecolors="none")
    add_trend_line(ax, x, y, color="#08306b")
    ax.set_title("External rain vs police rain coding", fontweight="bold")
    ax.set_xlabel("Rain anomaly (mm)")
    ax.set_ylabel("Rain-coded crash share anomaly")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.95,
        f"r = {metrics['rain_validation_corr']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )

    ax = axes[1, 0]
    x = merged["avg_max_temp_c_anom"]
    y = merged["serious_or_fatal_share_anom"]
    ax.scatter(x, y, alpha=0.25, s=16, color="#ff8c00", edgecolors="none")
    add_trend_line(ax, x, y, color="#b35806")
    ax.set_title("Warmer-than-normal days are more severe", fontweight="bold")
    ax.set_xlabel("Max temperature anomaly (C)")
    ax.set_ylabel("Serious/fatal share anomaly")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.95,
        f"r = {metrics['temp_vs_severe_corr']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )

    ax = axes[1, 1]
    x = merged["avg_cloud_cover_pct_anom"]
    y = merged["accidents_anom"]
    ax.scatter(x, y, alpha=0.25, s=16, color="#6a3d9a", edgecolors="none")
    add_trend_line(ax, x, y, color="#542788")
    ax.set_title("Cloudier-than-normal days have fewer crashes", fontweight="bold")
    ax.set_xlabel("Cloud cover anomaly (%)")
    ax.set_ylabel("Accident count anomaly")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.95,
        f"r = {metrics['cloud_vs_accidents_corr']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )

    ax = axes[1, 2]
    x = merged["avg_wind_speed_kmh_anom"]
    y = merged["high_wind_code_share_anom"]
    ax.scatter(x, y, alpha=0.25, s=16, color="#2a9d8f", edgecolors="none")
    add_trend_line(ax, x, y, color="#005f56")
    ax.set_title("Wind speeds vs police high-wind coding", fontweight="bold")
    ax.set_xlabel("Wind speed anomaly (km/h)")
    ax.set_ylabel("High-wind crash share anomaly")
    ax.grid(alpha=0.25)
    ax.text(
        0.03,
        0.95,
        f"r = {metrics['wind_validation_corr']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUTS_DIR / "uk_accidents_weather_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: dict[str, float | int | str]) -> None:
    report = f"""# UK Road Accidents x Weather Report

## Project Scope

This folder is the final-project workspace for the UK road accidents dataset and the UK weather dataset.
The analysis merges both sources at **daily national level** across the shared **2009-01-01 to 2020-12-31** window.

## Data Used

- Road accidents: 2005-2015 legacy STATS19 accident file + 2016-2020 recent accident file, restricted to the 2009-2020 overlap window
- Weather: daily weather observations aggregated from the selected weather dataset into one national daily weather profile

Key scale:

- Total accidents analysed: **{metrics['total_accidents']:,}**
- Daily observations in merged table: **{metrics['total_daily_rows']:,}**
- Average accidents per day: **{metrics['avg_daily_accidents']:.1f}**
- Average weather locations contributing each day: **{metrics['avg_weather_locations_per_day']:.1f}**

## Main Findings

1. **The merge is credible.** Daily rainfall anomalies align strongly with the share of crashes police coded as rainy-weather crashes: **r = {metrics['rain_validation_corr']:.3f}**. Wind anomalies also align well with police high-wind coding: **r = {metrics['wind_validation_corr']:.3f}**.

2. **Rain changes severity more than volume.** After comparing wet vs dry days *within the same month*, wetter days showed almost no change in accident count (**{metrics['wet_vs_dry_accidents_pct']:+.1f}%**) but noticeably lower fatal share (**{metrics['wet_vs_dry_fatal_pct']:+.1f}%**) and lower serious-or-fatal share (**{metrics['wet_vs_dry_serious_or_fatal_pct']:+.1f}%**). The pattern suggests rain increases caution and lowers impact speed even when collisions still happen.

3. **Warmer-than-normal days are more severe.** Maximum-temperature anomalies correlate positively with serious-or-fatal accident share (**r = {metrics['temp_vs_severe_corr']:.3f}**). The hottest 10% of days had a **{metrics['hot_vs_cold_serious_or_fatal_pct']:+.1f}%** higher serious-or-fatal share than the coldest 10% of days.

4. **Cloudier-than-normal days have fewer crashes.** Accident-count anomalies are negatively correlated with cloud-cover anomalies (**r = {metrics['cloud_vs_accidents_corr']:.3f}**), while darker/cloudier conditions move together (**r = {metrics['dark_vs_cloud_corr']:.3f}**). A plausible interpretation is lower exposure or more cautious driving under dull conditions.

5. **Weekend crashes are fewer but harsher.** Weekend days had **{metrics['weekend_accident_delta_pct']:+.1f}%** fewer crashes than weekdays, but the fatal share was **{metrics['weekend_fatal_share_delta_pct']:+.1f}%** higher. That finding is not caused by weather alone, but it is useful project context when discussing severity.

## Files Generated

- `data/processed/daily_uk_accidents_weather.parquet`: merged daily modelling table
- `data/processed/monthly_uk_accidents_weather.csv`: monthly summary table
- `data/processed/wet_vs_dry_by_month.csv`: within-month wet/dry comparison table
- `outputs/uk_accidents_weather_dashboard.png`: project dashboard
- `outputs/summary_metrics.json`: machine-readable metrics used in this report

## Important Limitation

The weather dataset does not include station coordinates in this project workspace, and some location names appear broader than a strict UK-only list. Because of that, the analysis is intentionally framed as a **daily national weather proxy** rather than a precise station-to-accident spatial join. The findings are useful for project-level interpretation, but they should not be presented as location-level causal proof.
"""
    REPORT_PATH.write_text(report)


def main() -> None:
    ensure_dirs()

    weather_daily = load_weather_daily()
    accidents_daily = load_daily_accidents()
    merged = add_calendar_and_anomalies(accidents_daily.merge(weather_daily, on="date", how="inner"))
    monthly = build_monthly_summary(merged)
    wet_vs_dry = build_wet_vs_dry_summary(merged)
    metrics = compute_summary_metrics(merged, wet_vs_dry)

    merged.to_parquet(PROCESSED_DIR / "daily_uk_accidents_weather.parquet", index=False)
    monthly.to_csv(PROCESSED_DIR / "monthly_uk_accidents_weather.csv", index=False)
    wet_vs_dry.to_csv(PROCESSED_DIR / "wet_vs_dry_by_month.csv", index=False)
    (OUTPUTS_DIR / "summary_metrics.json").write_text(json.dumps(metrics, indent=2))

    make_dashboard(merged, monthly, wet_vs_dry, metrics)
    write_report(metrics)

    print("Saved:")
    print(f"  {PROCESSED_DIR / 'daily_uk_accidents_weather.parquet'}")
    print(f"  {PROCESSED_DIR / 'monthly_uk_accidents_weather.csv'}")
    print(f"  {PROCESSED_DIR / 'wet_vs_dry_by_month.csv'}")
    print(f"  {OUTPUTS_DIR / 'summary_metrics.json'}")
    print(f"  {OUTPUTS_DIR / 'uk_accidents_weather_dashboard.png'}")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()
