"""
data_fetcher.py
---------------
Fetches 5-year NAV history for large-cap mutual funds from the AMFI API
(https://api.mfapi.in) and Nifty 50 benchmark data via yfinance.
"""

import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Large-cap funds: {display_name: (scheme_code, expense_ratio_%)} ──────────
LARGE_CAP_FUNDS = {
    "Mirae Asset Large Cap":        (118989, 0.54),
    "Axis Bluechip":                (120503, 0.55),
    "ICICI Pru Bluechip":           (120586, 1.26),
    "SBI Bluechip":                 (119598, 0.83),
    "Kotak Bluechip":               (120594, 0.58),
    "Canara Robeco Bluechip":       (120599, 0.44),
    "Invesco India Largecap":       (120828, 0.63),
    "Bandhan Large Cap":          (118479, 0.59),
    "Edelweiss Large Cap":         (118617, 0.53),
    "Nippon India Large Cap":      (118632, 1.67),
    "HDFC Large Cap":             (119018, 1.61),
    "Tata Large Cap":             (119160, 0.87),
}

NIFTY_TICKER   = "^NSEI"
RISK_FREE_RATE = 0.065          # 6.5% annualised (10-yr G-sec proxy)
LOOKBACK_YEARS = 5


def _date_range():
    end   = datetime.today()
    start = end - timedelta(days=365 * LOOKBACK_YEARS)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ── AMFI ─────────────────────────────────────────────────────────────────────

def fetch_nav_history(scheme_code: int, fund_name: str) -> pd.Series | None:
    """Return a monthly-resampled NAV series for the given scheme code."""
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.warning(f"  [{fund_name}] API error: {e}")
        return None

    records = raw.get("data", [])
    if not records:
        logger.warning(f"  [{fund_name}] Empty data returned.")
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()

    start, end = _date_range()
    df = df.loc[start:end]
    if df.empty:
        logger.warning(f"  [{fund_name}] No data in the 5-year window.")
        return None

    # Resample to month-end NAV
    monthly = df["nav"].resample("ME").last().dropna()
    monthly.name = fund_name
    logger.info(f"  [{fund_name}] {len(monthly)} monthly observations fetched.")
    return monthly


def fetch_all_nav_data() -> pd.DataFrame:
    """
    Download NAV histories for all tracked funds and return a single
    wide DataFrame (index = month-end dates, columns = fund names).
    """
    logger.info("Fetching NAV data from AMFI …")
    series_list = []

    for name, (code, _) in LARGE_CAP_FUNDS.items():
        s = fetch_nav_history(code, name)
        if s is not None:
            series_list.append(s)
        time.sleep(0.3)          # be polite to the free API

    if not series_list:
        raise RuntimeError("No fund data could be fetched. Check connectivity.")

    nav_df = pd.concat(series_list, axis=1).sort_index()
    logger.info(f"NAV data ready: {nav_df.shape[1]} funds, "
                f"{nav_df.shape[0]} months ({nav_df.index[0].date()} – "
                f"{nav_df.index[-1].date()})")
    return nav_df


# ── Nifty 50 ─────────────────────────────────────────────────────────────────

def fetch_nifty50() -> pd.Series:
    """Download Nifty 50 and return a monthly close-price series."""
    logger.info("Fetching Nifty 50 from Yahoo Finance …")
    start, end = _date_range()
    df = yf.download(NIFTY_TICKER, start=start, end=end,
                     interval="1mo", progress=False, auto_adjust=True)

    if df.empty:
        raise RuntimeError("Could not fetch Nifty 50 data.")

    series = df["Close"].squeeze()
    series.index = series.index.to_period("M").to_timestamp("M")   # month-end
    series.name  = "Nifty50"
    logger.info(f"Nifty 50 ready: {len(series)} monthly observations.")
    return series


# ── Convenience wrapper ───────────────────────────────────────────────────────

def load_all_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Returns
    -------
    nav_df        : wide DataFrame of NAVs (funds × months)
    nifty_series  : monthly Nifty 50 close prices
    expense_ratios: {fund_name: expense_ratio_%}
    """
    nav_df       = fetch_all_nav_data()
    nifty_series = fetch_nifty50()
    expense_ratios = {name: er for name, (_, er) in LARGE_CAP_FUNDS.items()
                      if name in nav_df.columns}
    return nav_df, nifty_series, expense_ratios
