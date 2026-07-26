"""Figures for the project, built on matplotlib only.

Every figure follows the same rules as the rest of the portfolio so the reports
read as one document: a fixed colorblind-validated categorical order (series
get slots in order and are never recycled), a single hue for magnitude, a
blue-red diverging ramp with a neutral midpoint for signed quantities, one
y-axis per panel, recessive gridlines, and a legend whenever more than one
series is on screen.

One rule is specific to this project. Any Sharpe ratio drawn as a bar is drawn
with its bootstrap interval. The headline of the study is a difference between
two subsamples, one of which has 232 days in it; a bar chart without error bars
would present that difference as more settled than the data supports.

Public entry points
-------------------
``screening_figure``          the funnel, and how much of it is noise
``spread_figure``             one pair's spread, z-score and trades
``regime_figure``             the market split into calm and turbulent
``volatility_figure``         realized vs EWMA vs GARCH
``performance_figure``        equity, drawdown, exposure, cost sensitivity
``attribution_figure``        performance by regime — the headline
``headline_figure``           the four-panel composite for the README
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

from . import metrics
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

SOURCE = (
    "Source: Kaggle S&P 500 daily OHLCV (all_stocks_5yr.csv), Feb 2013 - Feb 2018 · "
    "names with a complete history only · walk-forward: 252d formation / 126d trading."
)

REGIME_COLOR = {"calm": SERIES[0], "turbulent": SERIES[1]}


def _save(fig, savepath: str | None):
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
    return fig


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


def _year_ticks(ax) -> None:
    """One tick per year. Narrow panels default to a tick every other month,
    which at this figure width overlaps into an unreadable smear."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _note(ax, text: str, loc: str = "upper left", xy: tuple | None = None) -> None:
    """A short interpretive caption inside the panel."""
    if xy is None:
        xy = (0.015, 0.97) if "left" in loc else (0.985, 0.97)
    ax.text(
        *xy, text, transform=ax.transAxes, fontsize=7.8, color=INK_SECONDARY,
        va="top", ha="left" if "left" in loc else "right",
        bbox=dict(facecolor=SURFACE, edgecolor=GRID, linewidth=0.6,
                  boxstyle="round,pad=0.35", alpha=0.92),
    )


def _shade_regimes(ax, regime: pd.Series) -> None:
    """Shade the turbulent stretches behind whatever else is on the axis."""
    turb = (regime == "turbulent").astype(int)
    if turb.sum() == 0:
        return
    change = turb.diff().fillna(turb.iloc[0])
    starts = list(turb.index[change == 1])
    ends = list(turb.index[change == -1])
    if turb.iloc[0] == 1:
        starts = [turb.index[0]] + starts
    if len(ends) < len(starts):
        ends.append(turb.index[-1])
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, color=SERIES[1], alpha=0.11, linewidth=0, zorder=0)


# --------------------------------------------------------------------------
# 1. The screen
# --------------------------------------------------------------------------
def screening_figure(
    candidates: pd.DataFrame,
    window_table: pd.DataFrame,
    n_possible: int,
    traded_half_life: pd.Series,
    level: float = 0.05,
    savepath: str | None = None,
):
    """What the screen tested, what passed, and how much of that is noise.

    The left panel is the whole point: the 5% line is not a discovery
    threshold, it is a rate. Every test run has a 1-in-20 chance of landing
    left of it on pure noise, and the annotation says how many that implies.
    """
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # -- distance vs evidence
    ax = axes[0]
    passed = candidates["passed"]
    ax.scatter(candidates.loc[~passed, "ssd"], candidates.loc[~passed, "stat"],
               s=9, color=INK_MUTED, alpha=0.45, linewidth=0, label="not cointegrated")
    ax.scatter(candidates.loc[passed, "ssd"], candidates.loc[passed, "stat"],
               s=11, color=SERIES[0], alpha=0.75, linewidth=0, label="rejects at 5%")
    crit = float(candidates["crit_5pct"].median())
    ax.axhline(crit, color=CRITICAL, linewidth=1.2, linestyle="--",
               label=f"5% critical value ({crit:.2f})")
    ax.set_xlabel("stage-1 distance (SSD of normalised paths)")
    ax.set_ylabel("Engle-Granger statistic")
    ax.set_title("Stage 1 buys tractability, not truth")
    ax.legend(loc="lower left", framealpha=0.9)
    n_tested = len(candidates)
    _note(ax, f"{n_tested:,} tested of {n_possible:,} possible\n"
              f"{int(passed.sum()):,} reject · {n_tested * level:,.0f} expected "
              f"by chance alone", loc="upper right")

    # -- passed vs expected, per window
    ax = axes[1]
    x = np.arange(len(window_table))
    ax.bar(x, window_table["pairs_passed"], color=SERIES[0], label="rejected the null")
    ax.plot(x, window_table["expected_false_positives"], color=CRITICAL,
            marker="o", linewidth=1.8, label="expected under the null")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.strftime("%b %Y") for d in pd.to_datetime(window_table["trading_start"])],
        rotation=45, ha="right",
    )
    ax.set_ylabel("pairs")
    ax.set_ylim(0, window_table["pairs_passed"].max() * 1.32)
    ax.set_title("Real structure, but not all of it")
    ax.legend(loc="upper right", framealpha=0.9)
    ratio = window_table["pairs_passed"].sum() / window_table["expected_false_positives"].sum()
    _note(ax, f"{ratio:.1f}x the null rate overall\n"
              f"~{1/ratio:.0%} of rejections are noise")

    # -- half-life distribution: everything that rejected, and what got traded
    ax = axes[2]
    hl_all = candidates.loc[passed, "half_life"].replace([np.inf, -np.inf], np.nan).dropna()
    traded = pd.Series(traded_half_life).replace([np.inf, -np.inf], np.nan).dropna()

    hi = max(float(hl_all.max()), 20.0) * 1.05
    bins = np.linspace(0, hi, 45)
    ax.hist(hl_all, bins=bins, color=INK_MUTED, alpha=0.6,
            edgecolor=SURFACE, linewidth=0.4, label=f"rejects at 5% (n={len(hl_all):,})")
    ax.axvline(float(hl_all.median()), color=INK_SECONDARY, linewidth=1.3, linestyle="--")

    # Traded pairs are ~5% as numerous, so they need their own axis to be seen.
    ax2 = ax.twinx()
    ax2.hist(traded, bins=bins, color=SERIES[2], alpha=0.85,
             edgecolor=SURFACE, linewidth=0.4, label=f"traded (n={len(traded):,})")
    ax2.axvline(float(traded.median()), color=SERIES[2], linewidth=1.6)
    ax2.set_ylabel("pairs traded", color=SERIES[2])
    ax2.tick_params(axis="y", colors=SERIES[2])
    ax2.grid(False)

    ax.set_xlim(0, hi)
    ax.set_xlabel("Ornstein-Uhlenbeck half-life (trading days)")
    ax.set_ylabel("pairs rejecting the null")
    ax.set_title("The screen selects the fastest reverters")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels_ = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels_, loc="upper right", framealpha=0.9)
    _note(ax,
          f"median {hl_all.median():.1f}d rejecting → "
          f"{traded.median():.1f}d traded\n"
          f"slowest rejection {hl_all.max():.0f}d, so the\n"
          f"60d filter never binds — and a\n"
          f"{traded.median():.0f}-day spread is the hardest\n"
          f"to capture with a 1-day lag",
          loc="upper right", xy=(0.985, 0.70))

    annotate_source(fig, SOURCE)
    fig.tight_layout()
    return _save(fig, savepath)


# --------------------------------------------------------------------------
# 2. One spread, traded
# --------------------------------------------------------------------------
def spread_figure(
    spread: pd.Series,
    z: pd.Series,
    position: pd.Series,
    spec,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 4.0,
    formation_end=None,
    savepath: str | None = None,
):
    """The mechanics, on one pair: spread, z-score bands, and the trades taken."""
    use_style()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})

    ax = axes[0]
    ax.plot(spread.index, spread.values, color=SERIES[0], linewidth=1.5)
    ax.axhline(spec.mu, color=INK_SECONDARY, linewidth=1.0)
    for k, style in ((entry, "--"), (stop, ":")):
        for sign in (1, -1):
            ax.axhline(spec.mu + sign * k * spec.sigma, color=AXIS,
                       linewidth=0.9, linestyle=style)
    if formation_end is not None:
        ax.axvline(formation_end, color=CRITICAL, linewidth=1.4)
        ax.annotate("formation ends —\nnothing after this point\nwas used to fit it",
                    xy=(formation_end, ax.get_ylim()[1]), xytext=(6, -8),
                    textcoords="offset points", fontsize=7.8, color=CRITICAL,
                    fontweight="bold", va="top")
    ax.set_ylabel(f"log spread: {spec.y} - {spec.beta:.2f}·{spec.x}")
    ax.set_title(f"{spec.name} — half-life {spec.half_life:.0f}d, "
                 f"Engle-Granger {spec.stat:.2f}", fontsize=12)

    ax = axes[1]
    ax.plot(z.index, z.values, color=SERIES[0], linewidth=1.3)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1.0)
    for k in (entry, -entry):
        ax.axhline(k, color=AXIS, linewidth=0.9, linestyle="--")
    for k in (exit, -exit):
        ax.axhline(k, color=GOOD, linewidth=0.9, linestyle="-.")
    for k in (stop, -stop):
        ax.axhline(k, color=CRITICAL, linewidth=0.9, linestyle=":")

    pos = position.reindex(z.index).fillna(0.0)
    ax.fill_between(z.index, -stop, stop, where=pos > 0, color=SERIES[2],
                    alpha=0.16, linewidth=0, label="long spread")
    ax.fill_between(z.index, -stop, stop, where=pos < 0, color=SERIES[1],
                    alpha=0.16, linewidth=0, label="short spread")
    ax.set_ylabel("z-score (formation moments)")
    ax.set_ylim(-stop * 1.15, stop * 1.15)
    ax.legend(loc="upper left", ncol=2)
    _note(ax, f"entry ±{entry:g} · exit ±{exit:g} · stop ±{stop:g}", loc="upper right")

    annotate_source(fig, SOURCE)
    fig.tight_layout()
    return _save(fig, savepath)


# --------------------------------------------------------------------------
# 3. Regimes
# --------------------------------------------------------------------------
def regime_figure(
    market: pd.Series,
    prob_turbulent: pd.Series,
    regime: pd.Series,
    summary: pd.DataFrame,
    savepath: str | None = None,
):
    """The market, split into two states by the HMM, with what the states are."""
    use_style()
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1]})

    ax = axes[0]
    _shade_regimes(ax, regime)
    ax.plot(market.index, market.values * 100, color=INK_SECONDARY, linewidth=0.7)
    ax.set_ylabel("market return (%/day)")
    ax.set_title("Equal-weight market, shaded by the filtered turbulent state",
                 fontsize=12)
    _note(ax, "shading uses only past data —\nthe filter never sees the future")

    ax = axes[1]
    ax.plot(prob_turbulent.index, prob_turbulent.values, color=SERIES[1], linewidth=1.3)
    ax.axhline(0.5, color=AXIS, linewidth=0.9, linestyle="--")
    ax.set_ylabel("P(turbulent)")
    ax.set_ylim(-0.03, 1.03)

    ax = axes[2]
    eq = metrics.equity_curve(market)
    ax.plot(eq.index, eq.values, color=SERIES[0], linewidth=1.6)
    _shade_regimes(ax, regime)
    ax.set_ylabel("market growth of $1")

    rows = []
    for name, r in summary.iterrows():
        rows.append(
            f"{name}: {r['ann_vol']:.1%} ann vol · "
            f"stays {r['stay_prob']:.1%}/day · "
            f"runs {r['expected_duration_days']:.0f}d · "
            f"{r['realized_share']:.0%} of days"
        )
    axes[1].text(
        0.015, 0.05, "\n".join(rows), transform=axes[1].transAxes,
        fontsize=8, color=INK_SECONDARY, va="bottom",
        bbox=dict(facecolor=SURFACE, edgecolor=GRID, linewidth=0.6,
                  boxstyle="round,pad=0.4", alpha=0.92),
    )

    annotate_source(fig, SOURCE)
    fig.tight_layout()
    return _save(fig, savepath)


# --------------------------------------------------------------------------
# 4. Volatility estimators
# --------------------------------------------------------------------------
def volatility_figure(
    realized: pd.Series,
    ewma: pd.Series,
    garch: pd.Series,
    garch_fit,
    savepath: str | None = None,
):
    """Three estimators of the same quantity, and where they disagree."""
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    for i, (label, s) in enumerate(
        (("realized (21d)", realized), ("EWMA (λ=0.94)", ewma), ("GARCH(1,1)", garch))
    ):
        ax.plot(s.index, s.values * 100, label=label, color=SERIES[i],
                linewidth=1.3 if i else 1.0, alpha=0.95 if i else 0.7)
    ax.set_ylabel("annualised volatility (%)")
    ax.set_title("Three answers to 'how volatile was today'", fontsize=12)
    ax.legend(loc="upper left", ncol=3)
    _note(ax,
          f"GARCH persistence α+β = {garch_fit.persistence:.3f}\n"
          f"variance half-life {garch_fit.half_life:.0f}d\n"
          f"long-run vol {np.sqrt(garch_fit.long_run_var * 252):.1%}",
          loc="upper right")

    ax = axes[1]
    joined = pd.concat([realized, ewma, garch], axis=1).dropna()
    joined.columns = ["realized", "ewma", "garch"]
    ax.scatter(joined["realized"] * 100, joined["garch"] * 100, s=7,
               color=SERIES[2], alpha=0.4, linewidth=0)
    lim = [0, joined.max().max() * 100 * 1.05]
    ax.plot(lim, lim, color=AXIS, linewidth=1.0, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("realized (21d), %")
    ax.set_ylabel("GARCH(1,1), %")
    ax.set_title("GARCH leads the rolling window")
    corr = joined["realized"].corr(joined["garch"])
    _note(ax, f"correlation {corr:.2f}")

    annotate_source(fig, SOURCE)
    fig.tight_layout()
    return _save(fig, savepath)


# --------------------------------------------------------------------------
# 5. Performance
# --------------------------------------------------------------------------
def performance_figure(
    result,
    cost_sweep: pd.DataFrame,
    charged_bps: float = 10.0,
    savepath: str | None = None,
):
    """Gross vs net, the drawdown, how much capital was at work, and the bill."""
    use_style()
    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1.15], hspace=0.55, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    for i, (label, r) in enumerate(
        (("gross", result.gross_returns), (f"net of {charged_bps:g} bps", result.returns))
    ):
        eq = metrics.equity_curve(r)
        ax.plot(eq.index, eq.values, label=label, color=SERIES[i], linewidth=2.0)
        _label_last(ax, eq, SERIES[i], f"{eq.iloc[-1]:.3f}×")
    ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=0)
    ax.set_ylabel("growth of $1")
    ax.set_title("Costs eat the entire edge", fontsize=13)
    ax.legend(loc="upper left", ncol=2)
    _note(ax,
          f"gross Sharpe {metrics.sharpe_ratio(result.gross_returns):.2f}  →  "
          f"net {metrics.sharpe_ratio(result.returns):.2f}\n"
          f"breakeven cost {result.breakeven_cost_bps():.1f} bps",
          loc="upper right")

    ax = fig.add_subplot(gs[1, 0])
    dd = metrics.drawdown_series(result.returns)
    ax.fill_between(dd.index, dd.values * 100, 0, color=CRITICAL, alpha=0.35, linewidth=0)
    ax.plot(dd.index, dd.values * 100, color=CRITICAL, linewidth=1.1)
    ax.set_ylabel("drawdown (%)")
    ax.set_title("Net drawdown")

    ax = fig.add_subplot(gs[1, 1])
    ge = result.gross_exposure
    ax.fill_between(ge.index, ge.values, 0, color=SERIES[0], alpha=0.3, linewidth=0)
    ax.plot(ge.index, ge.values, color=SERIES[0], linewidth=1.0)
    ax.set_ylabel("gross exposure")
    ax.set_title("Capital actually at work")
    _note(ax, f"mean {ge.mean():.2f} of a 1.0 budget\nidle capital is left idle")

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(cost_sweep.index, cost_sweep["Sharpe"], color=SERIES[0],
            marker="o", linewidth=1.8)
    ax.axhline(0, color=AXIS, linewidth=0.9)
    ax.axvline(charged_bps, color=CRITICAL, linewidth=1.2, linestyle="--")
    ax.annotate(f"charged {charged_bps:g} bps", xy=(charged_bps, ax.get_ylim()[1]),
                xytext=(5, -12), textcoords="offset points", fontsize=8,
                color=CRITICAL, fontweight="bold")
    ax.set_xlabel("transaction cost (bps of one-way turnover)")
    ax.set_ylabel("net Sharpe")
    ax.set_title("Cost sensitivity")

    ax = fig.add_subplot(gs[2, 1])
    monthly = result.turnover.resample("ME").mean()
    ax.bar(monthly.index, monthly.values, width=20, color=SERIES[3], alpha=0.9)
    ax.set_ylabel("avg daily turnover")
    ax.set_title("Trading intensity")

    annotate_source(fig, SOURCE)
    return _save(fig, savepath)


# --------------------------------------------------------------------------
# 6. The headline: attribution by regime
# --------------------------------------------------------------------------
def attribution_figure(
    table: pd.DataFrame,
    contribution: pd.DataFrame,
    net: pd.Series,
    regime: pd.Series,
    savepath: str | None = None,
):
    """Where the performance came from — and how sure we can be of it."""
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    order = [r for r in ("calm", "turbulent") if r in table.index]
    colors = [REGIME_COLOR.get(r, SERIES[0]) for r in order]
    x = np.arange(len(order))

    # -- gross vs net Sharpe with bootstrap intervals
    ax = axes[0]
    width = 0.36
    ax.bar(x - width / 2, table.loc[order, "gross_sharpe"], width,
           color=colors, alpha=0.45, label="gross")
    ax.bar(x + width / 2, table.loc[order, "net_sharpe"], width,
           color=colors, label="net of 10 bps")
    lo = table.loc[order, "net_sharpe"] - table.loc[order, "net_sharpe_boot_lo"]
    hi = table.loc[order, "net_sharpe_boot_hi"] - table.loc[order, "net_sharpe"]
    ax.errorbar(x + width / 2, table.loc[order, "net_sharpe"],
                yerr=[lo.values, hi.values], fmt="none", ecolor=INK,
                elinewidth=1.2, capsize=4)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n({int(table.loc[r, 'days'])} days)" for r in order])
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("The average of these two is meaningless")
    ax.legend(loc="upper left")
    _note(ax, "bars are 90% block-bootstrap\nintervals on the net Sharpe",
          loc="upper right")

    # -- breakeven cost vs what is charged
    ax = axes[1]
    be = table.loc[order, "breakeven_bps"].replace([np.inf, -np.inf], np.nan)
    ax.bar(x, be, color=colors)
    ax.axhline(10.0, color=CRITICAL, linewidth=1.3, linestyle="--")
    ax.annotate("charged 10 bps", xy=(ax.get_xlim()[1], 10.0), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=8,
                color=CRITICAL, fontweight="bold")
    for xi, v in zip(x, be):
        if np.isfinite(v):
            ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("breakeven cost (bps)")
    ax.set_title("How much friction each regime can absorb")

    # -- share of days vs share of P&L
    ax = axes[2]
    common = [r for r in order if r in contribution.index]
    xi = np.arange(len(common))
    ax.bar(xi - 0.18, contribution.loc[common, "day_share"], 0.36,
           color=INK_MUTED, label="share of days")
    ax.bar(xi + 0.18, contribution.loc[common, "pnl_share"], 0.36,
           color=[REGIME_COLOR.get(r, SERIES[0]) for r in common],
           label="share of P&L")
    ax.axhline(0, color=INK_SECONDARY, linewidth=1.0)
    ax.set_xticks(xi)
    ax.set_xticklabels(common)
    ax.set_ylabel("share")
    ax.set_title("A third of the days, all of the profit")
    ax.legend(loc="upper left")

    annotate_source(fig, SOURCE)
    fig.tight_layout()
    return _save(fig, savepath)


# --------------------------------------------------------------------------
# 7. Composite for the README
# --------------------------------------------------------------------------
def headline_figure(
    result,
    table: pd.DataFrame,
    regime: pd.Series,
    market_vol: pd.Series,
    savepath: str | None = None,
):
    """Four panels: the whole argument in one image."""
    use_style()
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)
    order = [r for r in ("calm", "turbulent") if r in table.index]
    colors = [REGIME_COLOR.get(r, SERIES[0]) for r in order]
    x = np.arange(len(order))

    # equity, shaded by regime
    ax = fig.add_subplot(gs[0, :2])
    _shade_regimes(ax, regime)
    for i, (label, r) in enumerate(
        (("gross", result.gross_returns), ("net of 10 bps", result.returns))
    ):
        eq = metrics.equity_curve(r)
        ax.plot(eq.index, eq.values, label=label, color=SERIES[i], linewidth=2.0)
        _label_last(ax, eq, SERIES[i], f"{eq.iloc[-1]:.3f}×")
    ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=0)
    ax.set_ylabel("growth of $1")
    ax.set_title("Cointegration pairs, S&P 500 2014-2018 — shaded where the HMM "
                 "says turbulent", fontsize=12)
    ax.legend(loc="upper left", ncol=2)

    # market vol for context
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(market_vol.index, market_vol.values * 100, color=INK_SECONDARY,
            linewidth=1.1)
    _shade_regimes(ax, regime)
    ax.set_ylabel("market vol (%, ann.)")
    ax.set_title("What 'turbulent' means")
    _year_ticks(ax)

    # net Sharpe by regime with CI
    ax = fig.add_subplot(gs[1, 0])
    lo = table.loc[order, "net_sharpe"] - table.loc[order, "net_sharpe_boot_lo"]
    hi = table.loc[order, "net_sharpe_boot_hi"] - table.loc[order, "net_sharpe"]
    ax.bar(x, table.loc[order, "net_sharpe"], color=colors)
    ax.errorbar(x, table.loc[order, "net_sharpe"], yerr=[lo.values, hi.values],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("net Sharpe")
    ax.set_title("Net Sharpe by regime")

    # breakeven
    ax = fig.add_subplot(gs[1, 1])
    be = table.loc[order, "breakeven_bps"].replace([np.inf, -np.inf], np.nan)
    ax.bar(x, be, color=colors)
    ax.axhline(10.0, color=CRITICAL, linewidth=1.3, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("breakeven cost (bps)")
    ax.set_title("Friction each regime survives")

    # the summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    lines = [
        f"unconditional net Sharpe   {metrics.sharpe_ratio(result.returns):+.2f}",
        f"gross Sharpe               {metrics.sharpe_ratio(result.gross_returns):+.2f}",
        f"breakeven cost             {result.breakeven_cost_bps():.1f} bps",
        "",
    ]
    for r in order:
        row = table.loc[r]
        lines.append(
            f"{r:<10} net {row['net_sharpe']:+.2f} "
            f"[{row['net_sharpe_boot_lo']:+.2f}, {row['net_sharpe_boot_hi']:+.2f}]"
        )
    ax.text(0.0, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=9.2,
            family="monospace", color=INK_SECONDARY, va="top")
    ax.text(0.0, 0.16,
            "The unconditional number is an\naverage of a strategy that works\n"
            "and one that burns money.",
            transform=ax.transAxes, fontsize=8.6, color=INK, va="top",
            fontweight="bold")

    annotate_source(fig, SOURCE)
    return _save(fig, savepath)
