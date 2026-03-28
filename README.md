# From Coal Mines to Wheat Fields

**Quantihack 2026 — Team Deanos**

Discovering a tradeable signal in spurious correlations.

## The Signal

We found that multiplying the z-scores of **global coal production** and **UK road crash casualties** produces a composite signal that predicts **UK humidity** with r = 0.99. This humidity signal then correlates with **agricultural commodity prices** at r = 0.75.

$$S(t) = Z_{\text{coal}}(t) \cdot Z_{\text{casualties}}(t)$$

## Project Structure

```
Quantihack/
├── report.tex                     # LaTeX research paper
├── Quantihack.pdf                 # Compiled paper
├── data/
│   ├── world_coal_production/     # Global coal production (1981-2021)
│   ├── road-casualty-data/        # UK road casualties (2005-2020)
│   └── archive (10)/              # UK weather/humidity (2009-2024)
├── plots/
│   ├── wheat_signal_final.png     # Signal → Humidity → Agriculture ETF
│   └── zscore_6yr.png             # Z-score discovery across windows
└── scripts/
    ├── full_zscore_v2.py          # Main correlation scanner
    ├── zscore_hunt.py             # Z-score product/ratio search
    ├── plot_6yr.py                # 6-year window analysis & plot
    └── plot_wheat_final.py        # Final signal vs humidity vs DBA plot
```

## Key Results

| Relationship | r |
|---|---|
| Signal → UK Humidity | 0.99 |
| Signal → DBA Agriculture ETF | 0.75 |
| Signal → Wheat Futures | 0.73 |
| Signal → Corn Futures | 0.73 |
| Signal → Coffee Futures | 0.72 |
| Signal → Sugar Futures | 0.71 |

## Running

```bash
pip install pandas numpy matplotlib yfinance pyarrow
python scripts/full_zscore_v2.py     # Scan for z-score correlations
python scripts/plot_wheat_final.py   # Generate the main plot
```
