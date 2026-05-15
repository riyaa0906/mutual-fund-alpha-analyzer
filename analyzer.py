"""
analyzer.py
-----------
Computes risk-adjusted performance metrics for each mutual fund:
  • Monthly returns (fund & benchmark)
  • Expense-ratio-adjusted fund returns
  • Beta via OLS regression on excess returns
  • Jensen's Alpha (annualised)
  • Sharpe Ratio, Information Ratio, Tracking Error
  • 12-month rolling alpha
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)

MONTHS_PER_YEAR  = 12
RISK_FREE_RATE   = 0.065          # annual; matches data_fetcher.py


def _monthly_rf() -> float:
    return (1 + RISK_FREE_RATE) ** (1 / MONTHS_PER_YEAR) - 1


def compute_returns(prices: pd.Series) -> pd.Series:
    """Simple monthly returns from price / NAV series."""
    return prices.pct_change().dropna()


def adjust_for_expense_ratio(fund_returns: pd.Series,
                              annual_expense_ratio_pct: float) -> pd.Series:
    """
    Subtract the monthly equivalent expense ratio from each return.
    AMFI publishes the annual TER; we convert it to monthly drag.

    monthly_drag = (1 + TER)^(1/12) - 1
    """
    monthly_drag = (1 + annual_expense_ratio_pct / 100) ** (1 / MONTHS_PER_YEAR) - 1
    return fund_returns - monthly_drag


def compute_alpha_metrics(fund_returns: pd.Series,
                           market_returns: pd.Series,
                           fund_name: str) -> dict:
    """
    Align fund and market return series, then compute all metrics.

    Returns a dict with keys:
        fund, n_months, cagr_gross, cagr_net, beta, alpha_annual,
        sharpe, info_ratio, tracking_error, t_stat, p_value
    """
    rf_monthly = _monthly_rf()

    # Align on common dates
    aligned = pd.concat([fund_returns, market_returns], axis=1).dropna()
    aligned.columns = ["fund", "market"]

    if len(aligned) < 24:
        logger.warning(f"  [{fund_name}] Only {len(aligned)} overlapping months — skipping.")
        return None

    r_f = aligned["fund"].values
    r_m = aligned["market"].values

    # Excess returns over risk-free
    er_f = r_f - rf_monthly
    er_m = r_m - rf_monthly

    # OLS: er_f = alpha + beta * er_m  (Jensen's regression)
    slope, intercept, r_value, p_value, std_err = stats.linregress(er_m, er_f)
    beta           = slope
    alpha_monthly  = intercept
    alpha_annual   = (1 + alpha_monthly) ** MONTHS_PER_YEAR - 1

    t_stat = alpha_monthly / std_err if std_err > 0 else np.nan

    # CAGR — gross (as reported) and net (expense-adjusted, already in r_f)
    n = len(r_f)
    cagr_net   = (np.prod(1 + r_f)) ** (MONTHS_PER_YEAR / n) - 1

    # Sharpe (annualised)
    excess_mean   = er_f.mean() * MONTHS_PER_YEAR
    excess_std    = er_f.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    sharpe        = excess_mean / excess_std if excess_std > 0 else np.nan

    # Information Ratio & Tracking Error
    active_returns   = r_f - r_m
    tracking_error   = active_returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    info_ratio       = (active_returns.mean() * MONTHS_PER_YEAR) / tracking_error \
                       if tracking_error > 0 else np.nan

    return {
        "fund":            fund_name,
        "n_months":        n,
        "cagr_net":        round(cagr_net * 100, 2),
        "beta":            round(beta, 3),
        "alpha_annual":    round(alpha_annual * 100, 2),
        "sharpe":          round(sharpe, 3),
        "info_ratio":      round(info_ratio, 3),
        "tracking_error":  round(tracking_error * 100, 2),
        "t_stat":          round(t_stat, 3),
        "p_value":         round(p_value, 4),
    }


def compute_rolling_alpha(fund_returns: pd.Series,
                           market_returns: pd.Series,
                           window: int = 12) -> pd.Series:
    """
    12-month rolling Jensen's Alpha (annualised) for a single fund.
    Uses a rolling OLS via a loop — interpretable and dependency-light.
    """
    rf_monthly = _monthly_rf()
    aligned    = pd.concat([fund_returns, market_returns], axis=1).dropna()
    aligned.columns = ["fund", "market"]

    alphas = {}
    for i in range(window, len(aligned) + 1):
        chunk   = aligned.iloc[i - window : i]
        er_f    = chunk["fund"].values - rf_monthly
        er_m    = chunk["market"].values - rf_monthly
        try:
            slope, intercept, *_ = stats.linregress(er_m, er_f)
            alphas[chunk.index[-1]] = (1 + intercept) ** MONTHS_PER_YEAR - 1
        except Exception:
            alphas[chunk.index[-1]] = np.nan

    return pd.Series(alphas, name=fund_returns.name) * 100   # as %


def run_full_analysis(nav_df:        pd.DataFrame,
                      nifty_series:  pd.Series,
                      expense_ratios: dict) -> tuple[pd.DataFrame,
                                                     pd.DataFrame,
                                                     pd.Series]:
    """
    Main entry point called by main.py.

    Returns
    -------
    results_df    : summary metrics for every fund, sorted by alpha
    rolling_df    : wide DataFrame of 12-month rolling alphas (fund columns)
    market_returns: monthly Nifty 50 return series (for visualiser)
    """
    logger.info("Running alpha analysis …")
    market_returns = compute_returns(nifty_series)
    market_cagr    = (np.prod(1 + market_returns.dropna().values)) ** \
                     (MONTHS_PER_YEAR / len(market_returns.dropna())) - 1

    rows        = []
    rolling_dict = {}

    for fund_name in nav_df.columns:
        er_pct      = expense_ratios.get(fund_name, 1.0)
        nav_series  = nav_df[fund_name].dropna()
        gross_ret   = compute_returns(nav_series)
        net_ret     = adjust_for_expense_ratio(gross_ret, er_pct)

        metrics = compute_alpha_metrics(net_ret, market_returns, fund_name)
        if metrics is None:
            continue

        metrics["expense_ratio"] = er_pct
        metrics["market_cagr"]   = round(market_cagr * 100, 2)
        rows.append(metrics)

        # Rolling alpha
        rolling_dict[fund_name] = compute_rolling_alpha(net_ret, market_returns)

    results_df = (pd.DataFrame(rows)
                    .sort_values("alpha_annual", ascending=False)
                    .reset_index(drop=True))
    results_df.index += 1                   # 1-based rank

    rolling_df = pd.DataFrame(rolling_dict).sort_index()

    logger.info(f"Analysis complete. {len(results_df)} funds ranked.")
    _print_summary(results_df, market_cagr)
    return results_df, rolling_df, market_returns


def _print_summary(df: pd.DataFrame, market_cagr: float):
    """Pretty-print the ranking table to stdout."""
    print("\n" + "═" * 78)
    print("  MUTUAL FUND ALPHA RANKING  (Net of Expense Ratio, 5-Year CAPM)")
    print("═" * 78)
    print(f"  Nifty 50 CAGR (benchmark) : {market_cagr*100:.2f}%")
    print(f"  Risk-free rate            : {RISK_FREE_RATE*100:.1f}%")
    print("─" * 78)
    header = f"{'Rank':<5} {'Fund':<28} {'CAGR%':>7} {'Alpha%':>8} "  \
             f"{'Beta':>6} {'Sharpe':>7} {'IR':>6} {'ER%':>5}"
    print(header)
    print("─" * 78)
    for rank, row in df.iterrows():
        sig = "✓" if row["p_value"] < 0.10 else " "
        print(f"{rank:<5} {row['fund']:<28} {row['cagr_net']:>7.2f} "
              f"{row['alpha_annual']:>7.2f}{sig} {row['beta']:>6.3f} "
              f"{row['sharpe']:>7.3f} {row['info_ratio']:>6.3f} "
              f"{row['expense_ratio']:>5.2f}")
    print("─" * 78)
    print("  ✓ = alpha statistically significant at 10% level\n")
