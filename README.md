# Mutual Fund Alpha Analyzer

Analyzes 5-year NAV data of large-cap Indian mutual funds against the Nifty 50 benchmark.
Calculates risk-adjusted returns (Jensen's Alpha) after expense ratio deduction using CAPM,
and ranks funds by true alpha generated — separating consistent outperformers from market-riders.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output

- `alpha_results.csv` — ranked table with Alpha, Beta, Sharpe, Info Ratio
- `alpha_dashboard.png` — 6-panel Seaborn visualization dashboard

## Methodology

- **Data**: AMFI API (NAV history) + Yahoo Finance (Nifty 50)
- **Alpha**: Jensen's Alpha via CAPM — `α = Rp − [Rf + β(Rm − Rf)]`
- **Expense adjustment**: Monthly TER drag deducted before regression
- **Risk-free rate**: 6.5% (10-year Indian G-sec proxy)
- **Lookback**: 5 years of monthly returns
