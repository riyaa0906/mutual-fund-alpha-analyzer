"""
visualizer.py
-------------
Produces a 6-panel Seaborn/Matplotlib dashboard:

  1. Alpha Ranking Bar Chart        — fund ranks by Jensen's Alpha (%)
  2. Risk-Return Scatter            — CAGR vs Volatility, bubble = alpha size
  3. Alpha vs Expense Ratio         — does higher TER eat alpha?
  4. Rolling 12-Month Alpha         — consistency of outperformance over time
  5. Beta Distribution              — market-sensitivity comparison
  6. Outperformer vs Market Rider   — stacked attribution (alpha + beta contribution)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE_POS  = "#2ecc71"   # green  — positive alpha
PALETTE_NEG  = "#e74c3c"   # red    — negative alpha
ACCENT       = "#3498db"   # blue   — benchmark / neutral
DARK_BG      = "#1a1a2e"
PANEL_BG     = "#16213e"
TEXT_COLOR   = "#eaeaea"
GRID_COLOR   = "#2a2a4a"

sns.set_theme(style="darkgrid", palette="muted")

def _alpha_colors(alpha_series: pd.Series) -> list[str]:
    return [PALETTE_POS if v >= 0 else PALETTE_NEG for v in alpha_series]


def _fund_volatility(nav_df: pd.DataFrame,
                     expense_ratios: dict,
                     market_returns: pd.Series) -> pd.Series:
    """Annualised volatility (net returns) for each fund."""
    vols = {}
    for col in nav_df.columns:
        er  = expense_ratios.get(col, 1.0)
        ret = nav_df[col].pct_change().dropna()
        # monthly drag
        drag = (1 + er / 100) ** (1 / 12) - 1
        net  = ret - drag
        vols[col] = net.std(ddof=1) * np.sqrt(12) * 100
    return pd.Series(vols)


def plot_dashboard(results_df:     pd.DataFrame,
                   rolling_df:     pd.DataFrame,
                   nav_df:         pd.DataFrame,
                   market_returns: pd.Series,
                   expense_ratios: dict,
                   save_path:      str = "alpha_dashboard.png"):
    """
    Build and save the full 6-panel dashboard.

    Parameters
    ----------
    results_df      : output of analyzer.run_full_analysis()
    rolling_df      : wide rolling-alpha DataFrame
    nav_df          : raw NAV prices (for vol calc)
    market_returns  : Nifty 50 monthly returns
    expense_ratios  : {fund_name: TER %}
    save_path       : file to write
    """
    fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
    fig.suptitle("Mutual Fund Alpha Analyzer — Large Cap Universe (5-Year CAPM)",
                 fontsize=18, color=TEXT_COLOR, fontweight="bold", y=0.98)

    axes = fig.subplot_mosaic(
        [["alpha_bar", "alpha_bar", "rr_scatter"],
         ["rolling",   "rolling",   "beta_dist"],
         ["er_alpha",  "attribution", "attribution"]],
        gridspec_kw={"hspace": 0.45, "wspace": 0.35}
    )

    for ax in axes.values():
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.5)

    df = results_df.copy().reset_index(drop=True)

    # ── 1. Alpha Ranking Bar ──────────────────────────────────────────────────
    ax = axes["alpha_bar"]
    colors = _alpha_colors(df["alpha_annual"])
    bars = ax.barh(df["fund"][::-1], df["alpha_annual"][::-1],
                   color=colors[::-1], edgecolor="none", height=0.65)
    ax.axvline(0, color=TEXT_COLOR, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Jensen's Alpha (%, annualised, net of expense ratio)")
    ax.set_title("① Alpha Ranking", fontweight="bold")

    # Value labels
    for bar, val in zip(bars, df["alpha_annual"][::-1]):
        ha = "left" if val >= 0 else "right"
        offset = 0.05 if val >= 0 else -0.05
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}%", va="center", ha=ha,
                color=TEXT_COLOR, fontsize=8, fontweight="bold")

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

    # ── 2. Risk-Return Scatter ────────────────────────────────────────────────
    ax = axes["rr_scatter"]
    vols = _fund_volatility(nav_df, expense_ratios, market_returns)
    df["vol"] = df["fund"].map(vols)

    market_vol  = market_returns.std(ddof=1) * np.sqrt(12) * 100
    market_cagr = df["market_cagr"].iloc[0]

    bubble_size = np.abs(df["alpha_annual"]) * 40 + 80
    colors_sc   = _alpha_colors(df["alpha_annual"])

    sc = ax.scatter(df["vol"], df["cagr_net"],
                    s=bubble_size, c=colors_sc,
                    alpha=0.85, edgecolors=TEXT_COLOR, linewidth=0.4, zorder=3)
    ax.scatter(market_vol, market_cagr, s=180, color=ACCENT,
               marker="D", zorder=4, label="Nifty 50", edgecolors="white", linewidth=0.8)

    # Fund labels
    for _, row in df.iterrows():
        short = row["fund"].split()[-1]          # last word as short label
        ax.annotate(short,
                    (row["vol"], row["cagr_net"]),
                    textcoords="offset points", xytext=(5, 3),
                    color=TEXT_COLOR, fontsize=7, alpha=0.9)

    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE_POS,
               markersize=8, label="Positive Alpha"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE_NEG,
               markersize=8, label="Negative Alpha"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=ACCENT,
               markersize=8, label="Nifty 50"),
    ]
    ax.legend(handles=legend_els, facecolor=PANEL_BG, labelcolor=TEXT_COLOR,
              fontsize=8, loc="lower right")

    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("5-Year CAGR (%, net)")
    ax.set_title("② Risk–Return (bubble ∝ |alpha|)", fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    # ── 3. Rolling 12-Month Alpha ─────────────────────────────────────────────
    ax = axes["rolling"]
    # Plot top-3 and bottom-3 by alpha
    top3    = df.nlargest(3, "alpha_annual")["fund"].tolist()
    bottom3 = df.nsmallest(3, "alpha_annual")["fund"].tolist()

    palette_top    = sns.color_palette("Greens_d", n_colors=3)
    palette_bottom = sns.color_palette("Reds_d",   n_colors=3)

    for i, fund in enumerate(top3):
        if fund in rolling_df.columns:
            rolling_df[fund].plot(ax=ax, color=palette_top[i],
                                  linewidth=1.6, label=fund)
    for i, fund in enumerate(bottom3):
        if fund in rolling_df.columns:
            rolling_df[fund].plot(ax=ax, color=palette_bottom[i],
                                  linewidth=1.6, linestyle="--", label=fund)

    ax.axhline(0, color=TEXT_COLOR, linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("Rolling 12-M Alpha (%)")
    ax.set_title("④ Rolling 12-Month Alpha — Top 3 vs Bottom 3", fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=7,
              ncol=2, loc="upper left")
    ax.fill_between(rolling_df.index, 0, 0, color="none")   # axis ref

    # ── 4. Beta Distribution ─────────────────────────────────────────────────
    ax = axes["beta_dist"]
    beta_colors = [PALETTE_POS if b < 1 else PALETTE_NEG for b in df["beta"]]
    ax.bar(range(len(df)), df.sort_values("beta")["beta"].values,
           color=[PALETTE_POS if b < 1 else PALETTE_NEG
                  for b in df.sort_values("beta")["beta"].values],
           edgecolor="none")
    ax.axhline(1.0, color=ACCENT, linewidth=1.5, linestyle="--", alpha=0.8,
               label="β = 1 (market)")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        [f.split()[-1] for f in df.sort_values("beta")["fund"].tolist()],
        rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("Beta (β)")
    ax.set_title("⑤ Market Beta Distribution", fontweight="bold")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=8)

    # ── 5. Expense Ratio vs Alpha ─────────────────────────────────────────────
    ax = axes["er_alpha"]
    ax.scatter(df["expense_ratio"], df["alpha_annual"],
               c=_alpha_colors(df["alpha_annual"]), s=100,
               edgecolors=TEXT_COLOR, linewidth=0.4, zorder=3)

    # Trend line
    m, b = np.polyfit(df["expense_ratio"], df["alpha_annual"], 1)
    xs   = np.linspace(df["expense_ratio"].min(), df["expense_ratio"].max(), 50)
    ax.plot(xs, m * xs + b, color=ACCENT, linewidth=1.5,
            linestyle="--", alpha=0.8, label=f"Trend (slope={m:.1f})")

    ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, linestyle=":", alpha=0.5)
    for _, row in df.iterrows():
        ax.annotate(row["fund"].split()[0],
                    (row["expense_ratio"], row["alpha_annual"]),
                    textcoords="offset points", xytext=(4, 3),
                    color=TEXT_COLOR, fontsize=7)

    ax.set_xlabel("Expense Ratio (% TER)")
    ax.set_ylabel("Alpha (%, annualised)")
    ax.set_title("③ Does Higher TER Kill Alpha?", fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=8)

    # ── 6. Return Attribution (Alpha vs Beta contribution) ────────────────────
    ax = axes["attribution"]
    market_ret_annual = df["market_cagr"].iloc[0]
    rf_annual         = 6.5   # %

    df["beta_contrib"] = df["beta"] * (market_ret_annual - rf_annual)
    df["rf_contrib"]   = rf_annual
    df_s = df.sort_values("alpha_annual", ascending=False)

    x      = np.arange(len(df_s))
    width  = 0.55

    ax.bar(x, df_s["rf_contrib"],        width, label="Risk-free",       color="#7f8c8d")
    ax.bar(x, df_s["beta_contrib"],      width, bottom=df_s["rf_contrib"],
           label="Beta contribution",    color=ACCENT, alpha=0.85)
    ax.bar(x, df_s["alpha_annual"],      width,
           bottom=df_s["rf_contrib"] + df_s["beta_contrib"],
           color=[PALETTE_POS if a >= 0 else PALETTE_NEG for a in df_s["alpha_annual"]],
           label="True Alpha")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f.split()[0] for f in df_s["fund"].tolist()],
        rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("Return attribution (%)")
    ax.set_title("⑥ Return Attribution — Alpha vs Beta vs Risk-Free", fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=8, ncol=3)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close()
    print(f"\n✅  Dashboard saved → {save_path}")
