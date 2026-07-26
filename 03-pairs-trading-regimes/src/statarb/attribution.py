"""Decomposing performance by volatility regime.

The point of this module is one comparison: gross and net performance, side by
side, conditioned on the regime the market was in. A strategy can have a real
gross edge that shows up only in turbulent markets and still be worthless, if
turbulence is also when its costs are highest and its positions largest. Those
two facts are invisible in an unconditional Sharpe ratio, which averages them
into a single number and reports "roughly zero".

Regime labels passed in here must be *causal* — see ``regimes.fit_causal``.
Conditioning realised P&L on a regime label that was fitted with hindsight
produces a decomposition that looks sharp and means nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def sharpe_stderr(returns: pd.Series, periods_per_year: int = metrics.TRADING_DAYS) -> float:
    """Asymptotic standard error of an annualised Sharpe ratio.

    ``SE(SR) = sqrt((1 + SR^2/2) / n)`` on the per-period Sharpe, then scaled
    to annual. It assumes i.i.d. returns, which daily strategy returns are not
    quite, so it is the optimistic bound — ``block_bootstrap_sharpe`` is the
    one to believe when the two disagree.

    The reason this function exists: a Sharpe ratio computed on a few hundred
    days has a standard error near 1. Quoting such a number to two decimals
    without it invites the reader to take a coin flip for a result.
    """
    n = len(returns.dropna())
    if n < 2:
        return np.nan
    sr_period = metrics.sharpe_ratio(returns) / np.sqrt(periods_per_year)
    return float(np.sqrt((1.0 + 0.5 * sr_period**2) / n) * np.sqrt(periods_per_year))


def block_bootstrap_sharpe(
    returns: pd.Series,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.90,
) -> dict[str, float]:
    """Percentile confidence interval for the Sharpe ratio, by moving-block bootstrap.

    Resampling individual days would destroy the serial dependence that makes
    a pair strategy's returns autocorrelated — positions are held for days at
    a time, so consecutive returns are not independent draws. Sampling
    contiguous blocks of ``block`` days (a trading month) keeps that structure
    inside each block and only randomises the order of the blocks.

    Returns the point estimate, the bootstrap standard deviation, and the
    lower/upper percentile bounds.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < block * 2:
        return {"sharpe": np.nan, "boot_sd": np.nan, "lo": np.nan, "hi": np.nan}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))

    # Build every resample at once: (n_boot, n_blocks, block) -> (n_boot, n)
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    samples = r[idx].reshape(n_boot, -1)[:, :n]

    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, np.nan) * np.sqrt(metrics.TRADING_DAYS)

    tail = (1.0 - ci) / 2.0
    return {
        "sharpe": metrics.sharpe_ratio(returns),
        "boot_sd": float(np.nanstd(sr, ddof=1)),
        "lo": float(np.nanquantile(sr, tail)),
        "hi": float(np.nanquantile(sr, 1.0 - tail)),
    }


def regime_table(
    net: pd.Series,
    gross: pd.Series,
    regime: pd.Series,
    turnover: pd.Series | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-regime performance, gross and net, with the cost drag between them.

    Annualised figures are computed from the days belonging to each regime,
    which are not contiguous. That is the right treatment for a mean and a
    standard deviation, and the wrong one for a drawdown — a drawdown path
    stitched from non-adjacent days is not a drawdown anybody experienced — so
    no drawdown column appears here.

    Every Sharpe ratio comes with a standard error and a bootstrap interval.
    A regime is by definition a subsample, so these are the numbers computed
    on the fewest observations in the whole study and the ones most likely to
    be over-read.
    """
    idx = net.index.intersection(gross.index).intersection(regime.dropna().index)
    net, gross, reg = net.loc[idx], gross.loc[idx], regime.loc[idx]
    turn = turnover.loc[idx] if turnover is not None else None

    rows = []
    for label, mask in reg.groupby(reg).groups.items():
        n, g = net.loc[mask], gross.loc[mask]
        row = {
            "regime": label,
            "days": len(n),
            "share_of_days": len(n) / len(reg),
            "gross_ann_return": g.mean() * metrics.TRADING_DAYS,
            "net_ann_return": n.mean() * metrics.TRADING_DAYS,
            "ann_volatility": metrics.ann_volatility(n),
            "gross_sharpe": metrics.sharpe_ratio(g),
            "net_sharpe": metrics.sharpe_ratio(n),
            "hit_rate": metrics.hit_rate(n),
        }
        boot = block_bootstrap_sharpe(n, n_boot=n_boot, seed=seed)
        row["net_sharpe_se"] = sharpe_stderr(n)
        row["net_sharpe_boot_lo"] = boot["lo"]
        row["net_sharpe_boot_hi"] = boot["hi"]
        row["cost_drag_ann"] = row["gross_ann_return"] - row["net_ann_return"]
        if turn is not None:
            row["avg_turnover"] = float(turn.loc[mask].mean())
            row["breakeven_bps"] = (
                float(1e4 * g.mean() / turn.loc[mask].mean())
                if turn.loc[mask].mean() > 0
                else np.inf
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("regime").sort_index()


def contribution(net: pd.Series, regime: pd.Series) -> pd.DataFrame:
    """How much of the total P&L each regime actually produced.

    Shares of *cumulative* return, not of annualised return: a regime covering
    8% of the days can dominate the equity curve, and the annualised column in
    ``regime_table`` hides that by construction.
    """
    idx = net.index.intersection(regime.dropna().index)
    n, reg = net.loc[idx], regime.loc[idx]
    total = n.sum()

    out = n.groupby(reg).agg(["sum", "count"])
    out.columns = ["pnl_sum", "days"]
    out["pnl_share"] = out["pnl_sum"] / total if total != 0 else np.nan
    out["day_share"] = out["days"] / len(n)
    return out.sort_index()


def rolling_regime_sharpe(
    net: pd.Series,
    prob_turbulent: pd.Series,
    window: int = 126,
) -> pd.DataFrame:
    """Rolling Sharpe alongside the rolling regime probability.

    Used for the figure that has to answer "did the strategy stop working, or
    did the market change?" — the two series moving together is the claim; the
    chart is what lets a reader disagree with it.
    """
    idx = net.index.intersection(prob_turbulent.dropna().index)
    n, p = net.loc[idx], prob_turbulent.loc[idx]
    roll = n.rolling(window)
    sharpe = (roll.mean() / roll.std(ddof=1)) * np.sqrt(metrics.TRADING_DAYS)
    return pd.DataFrame(
        {"rolling_sharpe": sharpe, "p_turbulent": p.rolling(window).mean()}
    )


def turnover_profile(
    turnover: pd.Series, regime: pd.Series, cost_bps: float = 10.0
) -> pd.DataFrame:
    """Trading intensity by regime, and what it costs annually."""
    idx = turnover.index.intersection(regime.dropna().index)
    t, reg = turnover.loc[idx], regime.loc[idx]
    out = t.groupby(reg).agg(["mean", "max", "count"])
    out.columns = ["avg_daily_turnover", "max_daily_turnover", "days"]
    out["ann_cost"] = out["avg_daily_turnover"] * (cost_bps / 1e4) * metrics.TRADING_DAYS
    return out.sort_index()
