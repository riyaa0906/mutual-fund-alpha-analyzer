# Mutual Fund Alpha Analyzer

Analyzes whether Indian large-cap mutual fund managers genuinely create alpha
after accounting for expense ratios, or simply ride market momentum.

## Problem
Over 1,000 Indian mutual funds claim to beat the market. Most retail investors
lack tools to verify whether outperformance is real or just beta in disguise.

## Approach
- Fetched 5-year monthly NAV data for 12 large-cap funds via the AMFI API
- Downloaded Nifty 50 benchmark data via yfinance
- Deducted expense ratios (TER) before calculating returns
- Ran CAPM regression to isolate Jensen's Alpha from beta-driven returns
- Ranked funds by true alpha with statistical significance testing (t-stat, p-value)

## Key Metrics Computed
| Metric | What it tells you |
|---|---|
| Jensen's Alpha | Manager skill beyond market exposure |
| Beta | How much the fund moves with the market |
| Sharpe Ratio | Return per unit of total risk |
| Information Ratio | Return above benchmark per unit of active risk |
| Tracking Error | How much the fund deviates from the index |

## Formula
α = Rp − [Rf + β(Rm − Rf)]

Where Rf = 6.5% (10-year Indian G-sec), Rm = Nifty 50 monthly return

## Tech Stack
Python · Pandas · NumPy · SciPy · Seaborn · Matplotlib · yfinance

## How to Run
```bash
pip install -r requirements.txt
python main.py
```

## Output
- `alpha_results.csv` - all funds ranked by alpha with full metrics
- `alpha_dashboard.png` - 6-panel visualization dashboard