"""The report figures.

Every figure is generated from a table that is also written to
``reports/tables/``, so nothing in the write-up is hand-copied and no number in
an image can drift away from the number beside it.

Conventions, applied everywhere and shared with projects 01-03:

- Categorical hues come from ``style.SERIES`` **in fixed order and are never
  cycled**; a series keeps its colour across every panel in which it appears,
  so "the orange line" means the same model in figure 3 and figure 6.
- Magnitude uses the single-hue blue ramp, signed quantities the blue-red
  diverging ramp with a neutral midpoint. No rainbow anywhere.
- One y-axis per panel. Two quantities of different scale get two panels.
- Colour is never the only channel: every multi-series panel carries a legend,
  and the panels with few enough series carry direct labels as well.
- Grid and axes are recessive; text uses the ink tokens rather than the series
  colour it describes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import Formatter, FuncFormatter, PercentFormatter

from . import metrics, pca, style

# One colour per model, fixed for the whole report. Slots 0-2 of the shared
# categorical order are reserved for these three and used for nothing else, so
# "the orange line" is the boosted tree in every figure it appears in.
MODEL_COLORS = {
    "best_feature": style.SERIES[2],
    "ridge": style.SERIES[0],
    "gbm": style.SERIES[1],
}
MODEL_LABELS = {
    "best_feature": "best single feature",
    "ridge": "ridge",
    "gbm": "gradient boosting",
}

# Every *other* two-level comparison in the report — training against
# out-of-sample, daily against staggered, factor against residual — uses this
# one fixed pair. Reusing the model hues for those would put a blue bar labelled
# "in sample" on the same axis as a tick labelled "ridge", which is exactly the
# confusion the fixed-order rule exists to prevent. Violet/yellow separates at
# dE 41 under protanopia against a floor of 8, so the pair is safe on its own.
PAIR = (style.SERIES[5], style.SERIES[3])

class _AdaptivePercent(Formatter):
    """Percent ticks carrying just enough decimals to stay distinguishable.

    A fixed zero-decimal formatter is right for an equity curve spanning tens of
    percent and wrong for the attribution panels, where the whole axis covers
    four points and a 0.5-point tick spacing renders as
    ``-2%, -2%, -3%, -3%`` — two different gridlines with the same label. The
    decimal count is read off the spacing the locator actually chose.
    """

    def __call__(self, x: float, pos=None) -> str:
        ticks = self.axis.get_ticklocs() if self.axis is not None else ()
        step = float(np.min(np.diff(ticks))) if len(ticks) > 1 else abs(x)
        decimals = 0 if step >= 0.01 - 1e-12 else (1 if step >= 1e-3 else 2)
        return f"{x * 100:.{decimals}f}%"


def _pct() -> Formatter:
    """A fresh percent formatter — one instance per axis.

    ``Formatter`` instances hold a back-reference to the axis they are attached
    to, so a module-level singleton shared across every panel would leave that
    reference pointing at whichever axis was configured last and read the wrong
    tick spacing.
    """
    return _AdaptivePercent()


def _thin_dates(ax, n: int = 5) -> None:
    """Cap the number of date ticks — narrow panels overlap their labels."""
    ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator(maxticks=n))


def _tidy(ax, title: str | None = None, ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, loc="left", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel)


def _zero_line(ax, axis: str = "y") -> None:
    """A reference line at zero — the only line allowed to be non-recessive."""
    fn = ax.axhline if axis == "y" else ax.axvline
    fn(0.0, color=style.INK_SECONDARY, lw=0.9, ls=(0, (4, 3)), zorder=1)


def _label_last(ax, series: pd.Series, color: str, text: str) -> None:
    """Direct-label the end of a line, so identity is not colour-alone."""
    s = series.dropna()
    if s.empty:
        return
    ax.annotate(
        text,
        xy=(s.index[-1], s.iloc[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=style.INK_SECONDARY,
    )


# --------------------------------------------------------------------------
# Figure 01 — the factor structure
# --------------------------------------------------------------------------


def figure_factor_structure(
    eigenvalues: np.ndarray,
    n_obs: int,
    n_assets: int,
    sigma2: float,
    factor_returns: pd.DataFrame,
    market: pd.Series,
    factor_count: pd.Series,
    path,
    source: str = "",
):
    """Spectrum against the Marchenko-Pastur bulk, and what the factors are."""
    style.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))

    lo, hi = pca.mp_edges(n_obs, n_assets)
    lo_a, hi_a = pca.mp_edges(n_obs, n_assets, sigma2)

    # (a) the bulk, with the deviating eigenvalues off the end of the axis.
    # Both theoretical curves are drawn: the plain law, which assumes the whole
    # matrix is noise and visibly sits to the right of the data, and the same
    # law once the market mode's variance is taken out of the noise pool, which
    # is the one that actually describes the bulk.
    ax = axes[0, 0]
    bulk = eigenvalues[eigenvalues <= hi * 1.45]
    ax.hist(bulk, bins=70, density=True, color=style.SERIES[0], alpha=0.55,
            edgecolor=style.SURFACE, linewidth=0.4, label="observed eigenvalues")
    grid = np.linspace(1e-6, hi * 1.05, 800)
    ax.plot(grid, pca.mp_density(grid, n_obs, n_assets), color=style.INK_MUTED,
            lw=1.6, ls="--", label="Marchenko-Pastur, all variance noise")
    ax.plot(grid, pca.mp_density(grid, n_obs, n_assets, sigma2), color=style.INK,
            lw=2.0, label=f"same law, market removed (sigma^2 = {sigma2:.2f})")
    top_y = ax.get_ylim()[1]
    ax.axvline(hi, color=style.CRITICAL, lw=1.4, ls="--")
    ax.annotate(f"edge {hi:.2f}", xy=(hi, top_y * 0.42), xytext=(-5, 0),
                textcoords="offset points", ha="right", fontsize=8,
                color=style.CRITICAL, rotation=90)
    ax.axvline(hi_a, color=style.SERIES[3], lw=1.4, ls=":")
    ax.annotate(f"adjusted {hi_a:.2f}", xy=(hi_a, top_y * 0.42), xytext=(-5, 0),
                textcoords="offset points", ha="right", fontsize=8,
                color=style.SERIES[3], rotation=90)
    n_above = int((eigenvalues > hi).sum())
    ax.set_xlabel("eigenvalue of the correlation matrix")
    ax.legend(loc="upper right", fontsize=7.5)
    _tidy(ax, f"a · {n_above} of {n_assets} eigenvalues are not noise", "density")

    # (b) variance explained — magnitude, so one hue, light to dark.
    ax = axes[0, 1]
    k = 20
    share = eigenvalues[:k] / eigenvalues.sum()
    colors = style.SEQ_BLUE(np.linspace(0.85, 0.25, k))
    ax.bar(np.arange(1, k + 1), share, color=colors, width=0.75)
    ax.yaxis.set_major_formatter(_pct())
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("principal component")
    ax.annotate(f"PC1 = {share[0]:.1%} of all\ncross-sectional variance",
                xy=(1, share[0]), xytext=(12, -6), textcoords="offset points",
                fontsize=8.5, color=style.INK_SECONDARY)
    _tidy(ax, "b · the market mode dominates", "share of total variance")

    # (c) PC1 against the equal-weight market.
    ax = axes[1, 0]
    pc1 = factor_returns.iloc[:, 0]
    corr = float(pc1.corr(market))
    cum_pc1 = metrics.equity_curve(pc1) - 1.0
    cum_mkt = metrics.equity_curve(market) - 1.0
    ax.plot(cum_pc1.index, cum_pc1, color=style.SERIES[0], label="PC1 portfolio")
    ax.plot(cum_mkt.index, cum_mkt, color=style.SERIES[1], lw=1.6, ls="--",
            label="equal-weight market")
    ax.yaxis.set_major_formatter(_pct())
    ax.legend(loc="upper left")
    _tidy(ax, f"c · PC1 is the market (rho = {corr:.3f})", "cumulative return")

    # (d) how many factors the rolling model kept.
    ax = axes[1, 1]
    fc = factor_count.dropna()
    ax.step(fc.index, fc.to_numpy(), color=style.SERIES[0], where="post")
    ax.fill_between(fc.index, 0, fc.to_numpy(), step="post",
                    color=style.SERIES[0], alpha=0.15)
    ax.set_ylim(0, max(fc.max() * 1.3, 4))
    _tidy(ax, "d · factors retained by the rolling 252-day model",
          "components above the edge")

    fig.tight_layout()
    if source:
        style.annotate_source(fig, source)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 02 — the features
# --------------------------------------------------------------------------


def figure_features(
    ic_split: pd.DataFrame,
    corr: pd.DataFrame,
    path,
    source: str = "",
):
    """Feature information ratios in each period, and whether the sign survives.

    ``ic_split`` must carry ``IR_train`` and ``IR_oos`` columns indexed by
    feature name.
    """
    style.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0),
                             gridspec_kw={"width_ratios": [1.45, 1.0, 1.15]})

    order = ic_split["IR_train"].abs().sort_values().index
    d = ic_split.loc[order]
    y = np.arange(len(d))

    # (a) paired bars — two series, so two fixed hues plus a legend.
    ax = axes[0]
    ax.barh(y - 0.2, d["IR_train"], height=0.38, color=PAIR[0],
            label="training period")
    ax.barh(y + 0.2, d["IR_oos"], height=0.38, color=PAIR[1],
            label="out-of-sample")
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=8)
    _zero_line(ax, axis="x")
    ax.legend(loc="lower left")
    _tidy(ax, "a · feature IC information ratio", None)
    ax.set_xlabel("daily IC mean / std")

    # (b) does the sign survive? A quadrant plot answers it directly.
    ax = axes[1]
    same = np.sign(d["IR_train"]) == np.sign(d["IR_oos"])
    ax.scatter(d["IR_train"][same], d["IR_oos"][same], s=46,
               color=PAIR[0], zorder=3, label="sign held")
    ax.scatter(d["IR_train"][~same], d["IR_oos"][~same], s=46,
               color=PAIR[1], marker="X", zorder=3, label="sign flipped")
    lim = float(np.abs(d[["IR_train", "IR_oos"]].to_numpy()).max()) * 1.25
    ax.axhspan(-lim, 0, xmin=0.5, color=style.CRITICAL, alpha=0.05, zorder=0)
    ax.axhspan(0, lim, xmin=0, xmax=0.5, color=style.CRITICAL, alpha=0.05, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color=style.INK_MUTED, lw=0.9, ls=":", zorder=1)
    _zero_line(ax)
    _zero_line(ax, axis="x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("IR in the training period")
    ax.set_ylabel("IR out of sample")
    ax.legend(loc="upper left")
    _tidy(ax, f"b · {int(same.sum())} of {len(d)} features keep their sign")

    # (c) correlation heatmap — signed, so the diverging ramp.
    ax = axes[2]
    im = ax.imshow(corr.to_numpy(), cmap=style.DIV_BLUE_RED, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr)))
    ax.set_yticks(np.arange(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6.5)
    ax.set_yticklabels(corr.index, fontsize=6.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("rank correlation", fontsize=8)
    cb.outline.set_visible(False)
    _tidy(ax, "c · the 14 features are not 14 independent bets")

    fig.tight_layout()
    if source:
        style.annotate_source(fig, source)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 03 — signal quality
# --------------------------------------------------------------------------


def figure_signal_quality(
    ic_by_model: dict[str, pd.Series],
    decay: pd.DataFrame,
    quantiles: pd.Series,
    path,
    source: str = "",
):
    """Cumulative IC, how fast the signal decays, and the quantile profile."""
    style.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    ax = axes[0]
    for name, ic in ic_by_model.items():
        cum = ic.dropna().cumsum()
        ax.plot(cum.index, cum, color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    _zero_line(ax)
    _thin_dates(ax, 6)
    ax.legend(loc="lower left")
    _tidy(ax, "a · cumulative out-of-sample IC", "sum of daily rank IC")

    ax = axes[1]
    ax.plot(decay.index, decay["IC mean"], color=style.SERIES[0], marker="o")
    _zero_line(ax)
    ax.set_xscale("log")
    ax.set_xticks(list(decay.index))
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("forward horizon (trading days)")
    _tidy(ax, "b · IC decay of the best model", "mean rank IC")

    ax = axes[2]
    vals = quantiles * 252 / 5      # 5-day means -> annualized
    # Signed quantity, so the two poles of the diverging ramp — not the status
    # red, which is reserved and would read as an alert rather than a sign.
    colors = [style.DIV_BLUE_RED(0.82 if v < 0 else 0.18) for v in vals]
    ax.bar(np.arange(len(vals)), vals, color=colors, width=0.7)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(vals.index)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    _zero_line(ax)
    ax.set_xlabel("signal quintile (Q5 = most favoured)")
    _tidy(ax, "c · mean forward return by quintile", "annualized")

    fig.tight_layout()
    if source:
        style.annotate_source(fig, source)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 04 — what the models learned
# --------------------------------------------------------------------------


def figure_models(
    is_oos: pd.DataFrame,
    alpha_sweep: pd.DataFrame,
    importances: pd.Series,
    staged: pd.DataFrame,
    path,
    source: str = "",
):
    """In-sample against out-of-sample skill, and why regularisation does not help."""
    style.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))

    ax = axes[0, 0]
    x = np.arange(len(is_oos))
    ax.bar(x - 0.2, is_oos["in_sample"], width=0.38, color=PAIR[0],
           label="in sample")
    ax.bar(x + 0.2, is_oos["out_of_sample"], width=0.38, color=PAIR[1],
           label="out of sample")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(i, i) for i in is_oos.index], fontsize=8)
    _zero_line(ax)
    ax.legend(loc="upper left")
    _tidy(ax, "a · skill in sample does not survive the fold boundary",
          "mean rank IC")

    ax = axes[0, 1]
    ax.plot(alpha_sweep.index, alpha_sweep["in_sample_ic"], color=PAIR[0],
            marker="s", ls="--", label="in sample")
    ax.plot(alpha_sweep.index, alpha_sweep["oos_ic"], color=PAIR[1],
            marker="o", label="out of sample")
    ax.set_xscale("log")
    _zero_line(ax)
    ax.set_xlabel("ridge penalty (alpha)")
    ax.legend(loc="center right")
    _tidy(ax, "b · shrinking the model changes nothing", "mean rank IC")

    ax = axes[1, 0]
    imp = importances.sort_values()
    ax.barh(np.arange(len(imp)), imp.to_numpy(),
            color=style.SEQ_BLUE(np.linspace(0.3, 0.9, len(imp))))
    ax.set_yticks(np.arange(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=7.5)
    ax.xaxis.set_major_formatter(_pct())
    _tidy(ax, "c · where the boosted trees spend their splits",
          None)
    ax.set_xlabel("share of total squared-error reduction")

    ax = axes[1, 1]
    ax.plot(staged.index, staged["oos_ic"], color=style.SERIES[0])
    _zero_line(ax)
    ax.set_xlabel("trees in the ensemble")
    _tidy(ax, "d · more capacity does not buy skill", "mean out-of-sample IC")

    fig.tight_layout()
    if source:
        style.annotate_source(fig, source)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 05 — performance and costs
# --------------------------------------------------------------------------


def figure_performance(
    equity: dict[str, pd.Series],
    cost_sweep: dict[str, pd.DataFrame],
    turnover: pd.DataFrame,
    drawdown: dict[str, pd.Series],
    path,
    source: str = "",
):
    """Net equity, the cost curve, and what rebalancing frequency costs."""
    style.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))

    ax = axes[0, 0]
    for name, eq in equity.items():
        curve = eq - 1.0
        ax.plot(curve.index, curve, color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    _zero_line(ax)
    ax.yaxis.set_major_formatter(_pct())
    _thin_dates(ax, 6)
    ax.legend(loc="lower left")
    _tidy(ax, "a · net cumulative return, 10 bps per unit of turnover", "growth of 1")

    ax = axes[0, 1]
    for name, sweep in cost_sweep.items():
        ax.plot(sweep.index, sweep["Sharpe"], color=MODEL_COLORS[name],
                marker="o", ms=3.5, label=MODEL_LABELS[name])
    _zero_line(ax)
    ax.axvline(10.0, color=style.INK_SECONDARY, lw=0.9, ls=":")
    ax.annotate("charged", xy=(10.0, ax.get_ylim()[0]), xytext=(4, 8),
                textcoords="offset points", fontsize=8, color=style.INK_SECONDARY)
    ax.set_xlabel("transaction cost (bps of one-way turnover)")
    ax.legend(loc="upper right")
    _tidy(ax, "b · the strategy is under water at every cost", "net Sharpe")

    ax = axes[1, 0]
    x = np.arange(len(turnover))
    ax.bar(x - 0.2, turnover["daily"], width=0.38, color=PAIR[1],
           label="rebalanced daily")
    ax.bar(x + 0.2, turnover["staggered"], width=0.38, color=PAIR[0],
           label="staggered over the 5-day horizon")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(i, i) for i in turnover.index], fontsize=8)
    ax.legend(loc="upper right")
    _tidy(ax, "c · staggering cuts turnover by more than half",
          "mean daily one-way turnover")

    ax = axes[1, 1]
    for name, dd in drawdown.items():
        ax.plot(dd.index, dd, color=MODEL_COLORS[name], lw=1.5,
                label=MODEL_LABELS[name])
        ax.fill_between(dd.index, dd, 0, color=MODEL_COLORS[name], alpha=0.10)
    ax.yaxis.set_major_formatter(_pct())
    _thin_dates(ax, 6)
    ax.legend(loc="lower left")
    _tidy(ax, "d · drawdown", "below prior peak")

    fig.tight_layout()
    if source:
        style.annotate_source(fig, source)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 06 — alpha or factor exposure?
# --------------------------------------------------------------------------


def figure_attribution(
    decomposition: pd.DataFrame,
    rolling_beta: pd.Series,
    neutral_equity: dict[str, pd.Series],
    residual_ic: pd.DataFrame,
    path,
    source: str = "",
):
    """The project's question, in four panels: was any of it alpha?"""
    style.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))

    # (a) signed decomposition — diverging poles, one bar pair per strategy.
    ax = axes[0, 0]
    x = np.arange(len(decomposition))
    ax.bar(x - 0.2, decomposition["Factor-explained"], width=0.38,
           color=PAIR[0], label="explained by the PCA factors")
    ax.bar(x + 0.2, decomposition["Residual alpha"], width=0.38,
           color=PAIR[1], label="residual alpha")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(i, i) for i in decomposition.index],
                       fontsize=8)
    ax.yaxis.set_major_formatter(_pct())
    _zero_line(ax)
    ax.legend(loc="lower left")
    _tidy(ax, "a · annualized return, split by source", "per year")

    ax = axes[0, 1]
    rb = rolling_beta.dropna()
    ax.plot(rb.index, rb, color=style.SERIES[0])
    ax.fill_between(rb.index, rb, 0, color=style.SERIES[0], alpha=0.15)
    _zero_line(ax)
    _thin_dates(ax, 6)
    _tidy(ax, "b · rolling 126-day exposure to PC1", "beta to the market mode")

    ax = axes[1, 0]
    for color, (name, eq) in zip(PAIR, neutral_equity.items()):
        curve = eq - 1.0
        ax.plot(curve.index, curve, color=color, label=name)
    _zero_line(ax)
    ax.yaxis.set_major_formatter(_pct())
    _thin_dates(ax, 6)
    ax.legend(loc="lower left")
    _tidy(ax, "c · the same signal, with the factor exposure removed",
          "net cumulative return")

    ax = axes[1, 1]
    y = np.arange(len(residual_ic))
    ax.barh(y - 0.2, residual_ic["vs residual return"], height=0.38,
            color=PAIR[0], label="vs idiosyncratic return")
    ax.barh(y + 0.2, residual_ic["vs total return"], height=0.38,
            color=PAIR[1], label="vs the return actually earned")
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(i, i) for i in residual_ic.index],
                       fontsize=8)
    _zero_line(ax, axis="x")
    ax.set_xlabel("out-of-sample IC information ratio")
    ax.legend(loc="lower left")
    _tidy(ax, "d · the model predicts what the book does not earn")

    fig.tight_layout()
    if source:
        style.annotate_source(fig, source)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# The composite headline image
# --------------------------------------------------------------------------


def figure_headline(
    eigenvalues: np.ndarray,
    n_obs: int,
    n_assets: int,
    ic_split: pd.DataFrame,
    equity: dict[str, pd.Series],
    decomposition: pd.DataFrame,
    path,
    source: str = "",
):
    """The four panels that carry the result, for the README and the portfolio page."""
    style.use_style()
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.1))

    lo, hi = pca.mp_edges(n_obs, n_assets)
    ax = axes[0]
    bulk = eigenvalues[eigenvalues <= hi * 1.6]
    ax.hist(bulk, bins=60, density=True, color=style.SERIES[0], alpha=0.55,
            edgecolor=style.SURFACE, linewidth=0.4)
    grid = np.linspace(max(lo * 0.5, 1e-6), hi * 1.05, 500)
    ax.plot(grid, pca.mp_density(grid, n_obs, n_assets), color=style.INK, lw=2.0)
    ax.axvline(hi, color=style.CRITICAL, lw=1.4, ls="--")
    n_above = int((eigenvalues > hi).sum())
    ax.set_xlabel("eigenvalue")
    _tidy(ax, f"{n_above} real factors, {n_assets - n_above} noise", "density")

    ax = axes[1]
    same = np.sign(ic_split["IR_train"]) == np.sign(ic_split["IR_oos"])
    ax.scatter(ic_split["IR_train"][same], ic_split["IR_oos"][same], s=44,
               color=PAIR[0], zorder=3, label="sign held")
    ax.scatter(ic_split["IR_train"][~same], ic_split["IR_oos"][~same], s=44,
               marker="X", color=PAIR[1], zorder=3, label="sign flipped")
    ax.legend(loc="upper left", fontsize=7.5)
    lim = float(np.abs(ic_split[["IR_train", "IR_oos"]].to_numpy()).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], color=style.INK_MUTED, lw=0.9, ls=":")
    _zero_line(ax)
    _zero_line(ax, axis="x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("IR in training")
    ax.set_ylabel("IR out of sample")
    _tidy(ax, f"only {int(same.sum())}/{len(ic_split)} features keep their sign")

    ax = axes[2]
    for name, eq in equity.items():
        curve = eq - 1.0
        ax.plot(curve.index, curve, color=MODEL_COLORS[name],
                label=MODEL_LABELS[name])
    _zero_line(ax)
    ax.yaxis.set_major_formatter(_pct())
    _thin_dates(ax, 4)
    ax.legend(loc="lower left", fontsize=7.5)
    _tidy(ax, "every model loses money net", "cumulative")

    ax = axes[3]
    x = np.arange(len(decomposition))
    ax.bar(x - 0.2, decomposition["Factor-explained"], width=0.38,
           color=PAIR[0], label="factor exposure")
    ax.bar(x + 0.2, decomposition["Residual alpha"], width=0.38,
           color=PAIR[1], label="residual alpha")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(i, i) for i in decomposition.index],
                       fontsize=7.5, rotation=12)
    ax.yaxis.set_major_formatter(_pct())
    _zero_line(ax)
    ax.legend(loc="upper left", fontsize=7.5)
    _tidy(ax, "and none of it was alpha", "annualized")

    fig.tight_layout()
    if source:
        style.annotate_source(fig, source)
    fig.savefig(path)
    plt.close(fig)
    return path
