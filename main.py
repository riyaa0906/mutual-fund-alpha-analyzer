"""
main.py
-------
Entry point. Run this file to execute the full pipeline:
  1. Fetch NAV data from AMFI + Nifty 50 from Yahoo Finance
  2. Compute Jensen's Alpha, Sharpe, Beta, Info Ratio for each fund
  3. Print ranked table to console
  4. Save 6-panel Seaborn dashboard as alpha_dashboard.png
"""

from data_fetcher import load_all_data
from analyzer    import run_full_analysis
from visualizer  import plot_dashboard

def main():
    print("\n🔍  Mutual Fund Alpha Analyzer — Starting pipeline...\n")

    # Step 1: Fetch data
    nav_df, nifty_series, expense_ratios = load_all_data()

    # Step 2: Run analysis
    results_df, rolling_df, market_returns = run_full_analysis(
        nav_df, nifty_series, expense_ratios
    )

    # Step 3: Save results to CSV
    results_df.to_csv("alpha_results.csv")
    print("📄  Results saved → alpha_results.csv")

    # Step 4: Build dashboard
    plot_dashboard(
        results_df     = results_df,
        rolling_df     = rolling_df,
        nav_df         = nav_df,
        market_returns = market_returns,
        expense_ratios = expense_ratios,
        save_path      = "alpha_dashboard.png"
    )

if __name__ == "__main__":
    main()
