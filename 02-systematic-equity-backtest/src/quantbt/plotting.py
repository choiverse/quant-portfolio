"""Figures for the project, built on matplotlib only.

Every figure follows the same rules so the report reads as one document: a fixed
colorblind-validated categorical order (series get slots in order and are never
recycled), a single hue for magnitude, a blue-red diverging ramp with a neutral
midpoint for signed quantities, one y-axis per panel, recessive gridlines, and a
legend whenever more than one series is on screen.

Public entry points
-------------------
``tearsheet``                 strategy equity / drawdown / metrics
``data_overview_figure``      what is in the dataset and where the gaps are
``return_distribution_figure``  distributional shape of the return panel
``factor_diagnostics_figure`` is the *signal* predictive (IC, quantiles, decay)
``cost_analysis_figure``      what trading frictions do to the result
``robustness_figure``         parameter and cost surfaces
``rolling_risk_figure``       time-varying risk and the monthly calendar
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from . import diagnostics, eda, metrics
from .style import (
    AXIS,
    CRITICAL,
    DIV_BLUE_RED,
    GOOD,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SEQ_BLUE,
    SERIES,
    SURFACE,
    annotate_source,
    use_style,
)

SOURCE = "Source: Kaggle S&P 500 daily OHLCV (all_stocks_5yr.csv), Feb 2013 – Feb 2018 · universe frozen on end-of-sample membership."


def _metrics_table(ax, summ: pd.DataFrame) -> None:
    """Render a performance summary as a table on a blank axis."""
    ax.axis("off")
    fmt = summ.astype(object).copy()
    pct_rows = ["Total Return", "CAGR", "Ann. Volatility", "Max Drawdown", "Hit Rate"]
    for row in fmt.index:
        if row in pct_rows:
            fmt.loc[row] = summ.loc[row].map(lambda x: f"{x:.1%}")
        else:
            fmt.loc[row] = summ.loc[row].map(lambda x: f"{x:.2f}")

    table = ax.table(
        cellText=fmt.values,
        rowLabels=fmt.index,
        colLabels=fmt.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(color=INK, fontweight="bold")
        else:
            cell.set_text_props(color=INK_SECONDARY)


def _label_last(ax, series: pd.Series, color: str, text: str) -> None:
    """Direct-label the end of a line — identity without relying on color alone."""
    ax.annotate(
        text,
        xy=(series.index[-1], series.iloc[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=color,
        fontsize=8,
        fontweight="bold",
        va="center",
    )


# --------------------------------------------------------------------------
# 1. Strategy tearsheet
# --------------------------------------------------------------------------
def tearsheet(
    results: dict[str, pd.Series],
    flagship: str,
    title: str = "Strategy Tearsheet",
    savepath: str | None = None,
):
    """Equity curves (log), the flagship's drawdown band, and headline metrics."""
    use_style()
    fig = plt.figure(figsize=(11, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.4, 1.7], hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    for i, (label, rets) in enumerate(results.items()):
        eq = metrics.equity_curve(rets)
        color = SERIES[i]
        ax1.plot(eq.index, eq.values, label=label, color=color,
                 linewidth=2.4 if label == flagship else 1.8)
        _label_last(ax1, eq, color, f"{eq.iloc[-1]:.2f}×")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log scale)")
    ax1.set_title(title, fontsize=13)
    ax1.legend(loc="upper left", ncol=len(results))
    ax1.axhline(1.0, color=AXIS, linewidth=0.8, zorder=0)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    dd = metrics.drawdown_series(results[flagship])
    ax2.fill_between(dd.index, dd.values * 100, 0, color=CRITICAL, alpha=0.35, linewidth=0)
    ax2.plot(dd.index, dd.values * 100, color=CRITICAL, linewidth=1.2)
    trough = dd.idxmin()
    ax2.annotate(
        f"max DD {dd.min():.1%}",
        xy=(trough, dd.min() * 100),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=8,
        color=CRITICAL,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.85),
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title(f"{flagship} — underwater curve")

    summ = pd.DataFrame(
        {label: metrics.performance_summary(r, name=label) for label, r in results.items()}
    )
    _metrics_table(fig.add_subplot(gs[2]), summ)

    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 2. Dataset overview
# --------------------------------------------------------------------------
def data_overview_figure(raw: pd.DataFrame, prices: pd.DataFrame, savepath: str | None = None):
    """Four views of the raw panel: coverage, completeness, activity, integrity."""
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    fig.suptitle(
        "Dataset profile — S&P 500 daily panel, before any strategy logic",
        fontsize=13, fontweight="bold", color=INK,
    )

    # (a) Listed names per day — the survivorship fingerprint.
    ax = axes[0, 0]
    cov = eda.listing_coverage(raw)
    ax.plot(cov.index, cov.values, color=SERIES[0], linewidth=1.8)
    ax.fill_between(cov.index, cov.values, cov.min(), color=SERIES[0], alpha=0.12, linewidth=0)
    ax.set_title("(a) Tickers with a price on each day")
    ax.set_ylabel("names listed")
    ax.annotate(
        f"+{cov.iloc[-1] - cov.iloc[0]} names over the sample\n"
        "monotone rise = frozen end-of-sample universe",
        xy=(0.03, 0.93), xycoords="axes fraction", va="top",
        fontsize=8, color=INK_SECONDARY,
    )

    # (b) How complete each ticker's history is.
    ax = axes[0, 1]
    comp = eda.ticker_completeness(raw)
    ax.hist(comp.values * 100, bins=40, color=SERIES[1], edgecolor="white", linewidth=0.5)
    ax.axvline(98, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax.annotate(
        f"98% filter\ndrops {int((comp < 0.98).sum())} names",
        xy=(98, ax.get_ylim()[1] * 0.75), xytext=(-72, 0), textcoords="offset points",
        fontsize=8, color=INK_SECONDARY, fontweight="bold",
    )
    ax.set_title("(b) History completeness per ticker")
    ax.set_xlabel("% of trading days present")
    ax.set_ylabel("tickers")
    ax.set_yscale("log")

    # (c) Traded value — is the universe actually liquid enough to trade?
    ax = axes[1, 0]
    dollar_vol = (raw["close"] * raw["volume"]).groupby(raw["date"]).median() / 1e6
    ax.plot(dollar_vol.index, dollar_vol.values, color=SERIES[2], linewidth=1.4)
    ax.set_title("(c) Median daily dollar volume across the universe")
    ax.set_ylabel("USD millions")
    ax.annotate(
        f"median over sample: ${dollar_vol.median():,.0f}M\n"
        "10 bps of turnover is a defensible cost here",
        xy=(0.03, 0.93), xycoords="axes fraction", va="top",
        fontsize=8, color=INK_SECONDARY,
    )

    # (d) Integrity checks.
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("(d) Data-integrity checks")
    flags = eda.quality_flags(raw)
    cells = [[f"{c:,}", v] for c, v in zip(flags["count"], flags["verdict"])]
    table = ax.table(
        cellText=cells,
        rowLabels=[i if len(i) < 34 else i[:31] + "…" for i in flags.index],
        colLabels=["count", "verdict"],
        cellLoc="center",
        loc="center",
        colWidths=[0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(color=INK, fontweight="bold")
        elif col == 1:
            verdict = flags["verdict"].iloc[row - 1]
            cell.set_text_props(
                color=GOOD if verdict == "PASS" else CRITICAL if verdict == "FAIL" else INK_MUTED,
                fontweight="bold",
            )
        else:
            cell.set_text_props(color=INK_SECONDARY)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 3. Return distribution
# --------------------------------------------------------------------------
def return_distribution_figure(returns: pd.DataFrame, savepath: str | None = None):
    """Is the return panel Gaussian? (No — and every Sharpe below inherits that.)"""
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle(
        "Return panel — distributional shape of ~600,000 stock-days",
        fontsize=13, fontweight="bold", color=INK,
    )

    flat = returns.to_numpy().ravel()
    flat = flat[np.isfinite(flat)]
    mu, sd = flat.mean(), flat.std(ddof=1)
    stats = eda.return_distribution_stats(returns)

    # (a) Histogram vs the fitted normal, log-count so the tails are visible.
    ax = axes[0]
    ax.hist(flat, bins=300, range=(-0.12, 0.12), color=SERIES[0],
            label="observed", linewidth=0)
    grid = np.linspace(-0.12, 0.12, 400)
    normal = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    normal *= flat.size * (0.24 / 300)
    ax.plot(grid, normal, color=SERIES[1], linewidth=2.0, label="fitted normal")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.7)
    ax.set_xlim(-0.12, 0.12)
    ax.set_title("(a) Daily returns vs a normal")
    ax.set_xlabel("daily return")
    ax.set_ylabel("stock-days (log)")
    ax.legend(loc="upper right")
    ax.annotate(
        "the normal curve dives to zero\nwhere thousands of days actually sit",
        xy=(0.03, 0.60), xycoords="axes fraction", va="top",
        fontsize=8, color=INK_SECONDARY,
    )

    # (b) Q-Q plot against the normal.
    ax = axes[1]
    probs = np.linspace(0.0005, 0.9995, 1500)
    emp = np.quantile(flat, probs)
    theo = mu + sd * eda.normal_ppf(probs)
    lim = abs(theo).max() * 1.6
    ax.plot([-lim, lim], [-lim, lim], color=INK_MUTED, linewidth=1.2,
            linestyle="--", label="normal reference")
    ax.plot(theo, emp, color=SERIES[0], linewidth=2.0, label="empirical quantiles")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("(b) Q-Q vs normal — tails bend away")
    ax.set_xlabel("theoretical quantile")
    ax.set_ylabel("empirical quantile")
    ax.legend(loc="upper left")
    ax.annotate(
        f"excess kurtosis {stats['Excess kurtosis']:.1f}\nskewness {stats['Skewness']:.2f}",
        xy=(0.62, 0.12), xycoords="axes fraction",
        fontsize=8, color=INK_SECONDARY, fontweight="bold",
    )

    # (c) Cross-sectional dispersion — the raw material of a relative-value bet.
    ax = axes[2]
    disp = eda.cross_sectional_dispersion(returns)
    smooth = disp["std"].rolling(21).mean().dropna() * 100
    ax.plot(smooth.index, smooth.values, color=SERIES[2], linewidth=1.8,
            label="cross-sectional σ (21d avg)")
    ax.set_title("(c) Dispersion across names")
    ax.set_ylabel("daily cross-sectional σ (%)")
    ax.legend(loc="lower right")
    ax.annotate(
        "low dispersion = little to\nseparate winners from losers",
        xy=(0.04, 0.96), xycoords="axes fraction", va="top",
        fontsize=8, color=INK_SECONDARY,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 4. Factor diagnostics
# --------------------------------------------------------------------------
def factor_diagnostics_figure(
    ic_series: dict[str, pd.Series],
    quantile_ann: dict[str, pd.Series],
    decay: dict[str, pd.DataFrame],
    savepath: str | None = None,
):
    """Signal-level evidence, independent of portfolio construction or costs."""
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    fig.suptitle(
        "Factor diagnostics — does the score rank the cross-section?",
        fontsize=13, fontweight="bold", color=INK,
    )

    # (a) Rolling mean IC through time.
    ax = axes[0, 0]
    for i, (name, ic) in enumerate(ic_series.items()):
        roll = ic.rolling(126).mean().dropna()
        ax.plot(roll.index, roll.values, color=SERIES[i], linewidth=1.8, label=name)
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_title("(a) 6-month rolling mean rank IC")
    ax.set_ylabel("rank IC (21d forward)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.28)  # headroom so the legend clears the lines
    ax.legend(loc="upper left", ncol=2)
    ax.annotate(
        "sign flips for years at a time — the full-sample\nmean IC is not a stable property",
        xy=(0.02, 0.05), xycoords="axes fraction",
        fontsize=8, color=INK_SECONDARY,
    )

    # (b) IC distribution.
    ax = axes[0, 1]
    for i, (name, ic) in enumerate(ic_series.items()):
        ax.hist(ic.values, bins=60, color=SERIES[i], alpha=0.55, label=name, linewidth=0)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    for i, (name, ic) in enumerate(ic_series.items()):
        ax.axvline(ic.mean(), color=SERIES[i], linestyle="--", linewidth=1.6)
    ax.set_title("(b) Distribution of daily IC")
    ax.set_xlabel("rank IC")
    ax.set_ylabel("days")
    ax.legend(loc="upper right")
    note = " · ".join(f"{n}: mean {ic.mean():+.3f}, IR {ic.mean()/ic.std(ddof=1):+.2f}"
                      for n, ic in ic_series.items())
    ax.annotate(note, xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
                fontsize=7.5, color=INK_SECONDARY)

    # (c) Quantile monotonicity.
    ax = axes[1, 0]
    names = list(quantile_ann)
    width = 0.8 / len(names)
    x = np.arange(len(next(iter(quantile_ann.values()))))
    for i, name in enumerate(names):
        vals = quantile_ann[name].values * 100
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width * 0.9,
               color=SERIES[i], label=name, linewidth=0)
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{q + 1}" for q in x])
    ax.set_title("(c) Annualized return by score quantile (gross, common sample)")
    ax.set_ylabel("annualized return (%)")
    ax.set_xlabel("Q1 = lowest score  →  Q5 = highest score")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.22)
    ax.legend(loc="upper left", ncol=len(names))
    ax.annotate(
        "a real factor steps up left→right;\nthese are flat within noise",
        xy=(0.98, 0.90), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, color=INK_SECONDARY,
    )

    # (d) IC decay by forward horizon.
    ax = axes[1, 1]
    for i, (name, df) in enumerate(decay.items()):
        ax.plot(df.index, df["mean_ic"].values, color=SERIES[i],
                marker="o", linewidth=1.8, label=name)
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_title("(d) IC decay — how far ahead does the score see?")
    ax.set_xlabel("forward horizon (trading days, log)")
    ax.set_ylabel("mean rank IC")
    ax.legend(loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 5. Cost analysis
# --------------------------------------------------------------------------
def cost_analysis_figure(
    results: dict,
    sensitivity: dict[str, pd.DataFrame],
    savepath: str | None = None,
):
    """Where the gross edge goes: turnover, the cost wedge, and break-even."""
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    fig.suptitle(
        "Transaction costs — the gap between a gross signal and a tradable strategy",
        fontsize=13, fontweight="bold", color=INK,
    )
    names = list(results)

    # (a) Gross vs net equity for each strategy.
    ax = axes[0, 0]
    for i, name in enumerate(names):
        res = results[name]
        gross = metrics.equity_curve(res.gross_returns)
        net = metrics.equity_curve(res.returns)
        ax.plot(gross.index, gross.values, color=SERIES[i], linewidth=1.9,
                label=f"{name} — gross")
        ax.plot(net.index, net.values, color=SERIES[i], linewidth=1.5,
                linestyle="--", label=f"{name} — net")
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    ax.set_title("(a) Gross (solid) vs net of costs (dashed)")
    ax.set_ylabel("growth of $1")
    ax.legend(loc="upper left", ncol=1)

    # (b) Cumulative cost drag.
    ax = axes[0, 1]
    for i, name in enumerate(names):
        drag = diagnostics.cost_drag(results[name])
        ax.plot(drag.index, drag.values * 100, color=SERIES[i], linewidth=1.9, label=name)
        _label_last(ax, drag * 100, SERIES[i], f"{drag.iloc[-1]:.0%}")
    ax.set_title("(b) Cumulative performance paid away in costs")
    ax.set_ylabel("gross − net equity (% of initial capital)")
    ax.legend(loc="upper left")

    # (c) Turnover per rebalance.
    ax = axes[1, 0]
    for i, name in enumerate(names):
        to = results[name].turnover
        to = to[to > 0]
        ax.plot(to.index, to.values * 100, color=SERIES[i], linewidth=1.2,
                marker="o", markersize=2.5, label=f"{name} (avg {to.mean():.0%})")
    ax.set_title("(c) One-way turnover at each rebalance")
    ax.set_ylabel("turnover (% of gross book)")
    ax.legend(loc="upper left")

    # (d) Sharpe vs assumed cost, with the break-even marked.
    ax = axes[1, 1]
    for i, (name, sens) in enumerate(sensitivity.items()):
        ax.plot(sens.index, sens["sharpe"].values, color=SERIES[i],
                linewidth=2.0, label=name)
        be = diagnostics.breakeven_cost(sens)
        if np.isfinite(be):
            ax.plot([be], [0], marker="o", markersize=8, color=SERIES[i],
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            ax.annotate(f"break-even\n{be:.1f} bps", xy=(be, 0), xytext=(6, 12),
                        textcoords="offset points", fontsize=8,
                        color=SERIES[i], fontweight="bold")
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.axvline(10, color=INK_MUTED, linestyle=":", linewidth=1.2)
    ax.annotate("assumed\n10 bps", xy=(10, ax.get_ylim()[0]), xytext=(4, 8),
                textcoords="offset points", fontsize=8, color=INK_MUTED)
    ax.set_title("(d) Sharpe vs cost per unit of turnover")
    ax.set_xlabel("cost (bps of one-way turnover)")
    ax.set_ylabel("annualized Sharpe")
    ax.legend(loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 6. Robustness surfaces
# --------------------------------------------------------------------------
def robustness_figure(grids: dict[str, pd.DataFrame], savepath: str | None = None):
    """Sharpe over parameter grids — a lone bright cell is an overfit, not an edge."""
    use_style()
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.4), squeeze=False)
    fig.suptitle(
        "Parameter robustness — net Sharpe across the whole grid, not one lucky cell",
        fontsize=13, fontweight="bold", color=INK,
    )

    vmax = max(abs(np.nanmax(g.values)) for g in grids.values())
    vmax = max(vmax, abs(min(np.nanmin(g.values) for g in grids.values())), 0.2)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for ax, (name, grid) in zip(axes[0], grids.items()):
        im = ax.imshow(grid.values, cmap=DIV_BLUE_RED.reversed(), norm=norm, aspect="auto")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{c:.0%}" for c in grid.columns])
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels(grid.index)
        ax.set_xlabel("quantile per leg")
        ax.set_ylabel("lookback (trading days)")
        ax.set_title(name)
        ax.grid(False)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                val = grid.values[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=INK if abs(val) < vmax * 0.55 else "white")
        fig.colorbar(im, ax=ax, label="net Sharpe", fraction=0.046, pad=0.03)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig


# --------------------------------------------------------------------------
# 7. Rolling risk & calendar
# --------------------------------------------------------------------------
def rolling_risk_figure(
    results: dict[str, pd.Series],
    flagship: str,
    savepath: str | None = None,
):
    """When did it work? Rolling Sharpe, rolling vol, and a monthly calendar."""
    use_style()
    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1.6], hspace=0.55)
    fig.suptitle(
        "Stability through time — a full-sample Sharpe hides everything below",
        fontsize=13, fontweight="bold", color=INK,
    )

    # (a) Rolling 1-year Sharpe.
    ax = fig.add_subplot(gs[0])
    for i, (name, r) in enumerate(results.items()):
        rs = diagnostics.rolling_sharpe(r, window=252)
        ax.plot(rs.index, rs.values, color=SERIES[i], linewidth=1.8, label=name)
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_title("(a) Trailing 1-year Sharpe")
    ax.set_ylabel("Sharpe")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.30)
    ax.legend(loc="upper left", ncol=len(results))

    # (b) Rolling 1-year volatility.
    ax = fig.add_subplot(gs[1])
    for i, (name, r) in enumerate(results.items()):
        vol = r.rolling(252).std(ddof=1).dropna() * np.sqrt(252) * 100
        ax.plot(vol.index, vol.values, color=SERIES[i], linewidth=1.8, label=name)
    ax.set_title("(b) Trailing 1-year annualized volatility")
    ax.set_ylabel("volatility (%)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.30)
    ax.legend(loc="upper left", ncol=len(results))

    # (c) Monthly return calendar for the flagship.
    ax = fig.add_subplot(gs[2])
    table = diagnostics.monthly_return_table(results[flagship]) * 100
    lim = np.nanmax(np.abs(table.values))
    im = ax.imshow(table.values, cmap=DIV_BLUE_RED.reversed(),
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([pd.Timestamp(2000, m, 1).strftime("%b") for m in table.columns])
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_title(f"(c) {flagship} — monthly returns (%)")
    ax.grid(False)
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            val = table.values[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=7.5,
                        color=INK if abs(val) < lim * 0.55 else "white")
    fig.colorbar(im, ax=ax, label="monthly return (%)", fraction=0.03, pad=0.02)
    ax.annotate(
        "blank = signal still warming up (needs a 12-month lookback)",
        xy=(0, -0.30), xycoords="axes fraction", fontsize=8, color=INK_MUTED,
    )

    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.09, right=0.94)
    annotate_source(fig, SOURCE)
    if savepath:
        fig.savefig(savepath)
    return fig
