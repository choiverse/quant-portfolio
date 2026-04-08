"""Signal-level diagnostics — is the *alpha* real, before any portfolio is built?

A backtest equity curve is one number's worth of evidence and it confounds the
signal with the portfolio-construction and cost assumptions layered on top. The
standard research answer is to interrogate the signal directly:

- **Information coefficient (IC)** — the cross-sectional rank correlation between
  today's score and the forward return. Positive and stable is what you want; the
  *information ratio* of the IC series (mean / std) says whether it is reliable
  rather than merely large on average.
- **Quantile monotonicity** — sort names into buckets by score and check that
  average forward return increases monotonically across buckets. A signal that
  only "works" in the extreme bucket is usually a handful of outliers.
- **Decay** — how IC behaves as the forward horizon lengthens. This is what tells
  you how often you have to trade, and therefore how much cost you will pay.
- **Cost and parameter sensitivity** — the honest version of a backtest is a
  surface, not a point.

All functions take/return pandas objects aligned on the (date x ticker) panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .backtest import Backtester

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Forward returns and information coefficient
# --------------------------------------------------------------------------
def forward_returns(returns: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Cumulative return over the *next* ``horizon`` days, per name.

    The value on date ``t`` is the return earned from the close of ``t`` to the
    close of ``t + horizon`` — i.e. strictly future information, which is what a
    score observed at ``t`` is supposed to predict.
    """
    cum = (1.0 + returns).cumprod()
    return cum.shift(-horizon) / cum - 1.0


def information_coefficient(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    horizon: int = 21,
    method: str = "spearman",
    min_names: int = 20,
) -> pd.Series:
    """Daily cross-sectional rank correlation between score and forward return.

    ``method='spearman'`` gives the *rank* IC, which is the standard choice: it
    is robust to the fat tails and outliers that dominate raw return
    correlations. Returns one number per date.
    """
    fwd = forward_returns(returns, horizon)
    idx = score.index.intersection(fwd.index)
    s, f = score.loc[idx], fwd.loc[idx]

    valid = (s.notna() & f.notna()).sum(axis=1)
    ic = s.corrwith(f, axis=1, method=method)
    ic[valid < min_names] = np.nan
    ic.name = f"IC({horizon}d)"
    return ic.dropna()


def ic_summary(ic: pd.Series, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    """Headline statistics of an IC series.

    ``IC IR`` is mean/std of the IC — the signal analogue of a Sharpe ratio. The
    t-statistic uses an effective sample size of the number of *independent*
    observations, which for overlapping forward windows is far smaller than the
    raw row count; we report the naive t-stat and flag the overlap in the docs
    rather than pretending to a correction we have not estimated.
    """
    mean, sd = ic.mean(), ic.std(ddof=1)
    return pd.Series(
        {
            "Mean IC": mean,
            "Std IC": sd,
            "IC IR": mean / sd if sd else np.nan,
            "t-stat (naive)": mean / sd * np.sqrt(len(ic)) if sd else np.nan,
            "% positive": (ic > 0).mean(),
            "N obs": float(len(ic)),
        },
        name=ic.name,
    )


def ic_decay(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 42, 63, 126, 252),
    method: str = "spearman",
) -> pd.DataFrame:
    """Mean IC as a function of the forward horizon — the signal's half-life."""
    rows = []
    for h in horizons:
        ic = information_coefficient(score, returns, horizon=h, method=method)
        rows.append({"horizon": h, "mean_ic": ic.mean(), "ic_ir": ic.mean() / ic.std(ddof=1)})
    return pd.DataFrame(rows).set_index("horizon")


# --------------------------------------------------------------------------
# Quantile portfolios
# --------------------------------------------------------------------------
def quantile_weights(score: pd.DataFrame, n_quantiles: int = 5) -> list[pd.DataFrame]:
    """Long-only equal-weight target weights for each score quantile.

    Bucket 0 is the lowest-scoring names, bucket ``n-1`` the highest. Ranks are
    computed within each date's cross-section, so the buckets are always equally
    populated regardless of how the raw score is distributed.
    """
    ranks = score.rank(axis=1, pct=True, na_option="keep")
    buckets = []
    for q in range(n_quantiles):
        lo, hi = q / n_quantiles, (q + 1) / n_quantiles
        mask = (ranks > lo) & (ranks <= hi) if q else (ranks >= 0.0) & (ranks <= hi)
        counts = mask.sum(axis=1).replace(0, np.nan)
        buckets.append(mask.div(counts, axis=0).fillna(0.0))
    return buckets


def quantile_performance(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    n_quantiles: int = 5,
    rebalance_every: int = 21,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Backtest every score quantile as its own long-only book.

    Costs default to zero here on purpose: this panel answers "does the score
    order the cross-section?", which is a question about the signal and should
    not be entangled with the trading assumption. Returns ``(daily_returns,
    annualized_mean)`` where columns are ``Q1`` (lowest) ... ``Qn`` (highest).
    """
    bt = Backtester(returns, rebalance_every=rebalance_every, cost_bps=cost_bps)
    out = {}
    for q, w in enumerate(quantile_weights(score, n_quantiles), start=1):
        out[f"Q{q}"] = bt.run(w, name=f"Q{q}").returns
    daily = pd.DataFrame(out)
    ann = daily.mean() * TRADING_DAYS
    return daily, ann


# --------------------------------------------------------------------------
# Robustness surfaces
# --------------------------------------------------------------------------
def cost_sensitivity(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_grid: np.ndarray,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Sharpe and CAGR as a function of the assumed cost per unit of turnover.

    The break-even cost — where Sharpe crosses zero — is the single most useful
    number a backtest can report, because it converts "is this profitable?" into
    "how good does my execution have to be?".
    """
    rows = []
    for bps in cost_grid:
        bt = Backtester(returns, rebalance_every=rebalance_every, cost_bps=float(bps))
        r = bt.run(weights).returns
        rows.append(
            {
                "cost_bps": float(bps),
                "sharpe": metrics.sharpe_ratio(r),
                "cagr": metrics.cagr(r),
            }
        )
    return pd.DataFrame(rows).set_index("cost_bps")


def breakeven_cost(sens: pd.DataFrame) -> float:
    """Linearly interpolate the cost level at which Sharpe hits zero.

    Returns NaN when the curve never crosses zero inside the tested grid.
    """
    s = sens["sharpe"]
    sign = np.sign(s.values)
    cross = np.where(np.diff(sign) != 0)[0]
    if len(cross) == 0:
        return np.nan
    i = cross[0]
    x0, x1 = s.index[i], s.index[i + 1]
    y0, y1 = s.iloc[i], s.iloc[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def parameter_grid(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    signal_fn,
    lookbacks,
    quantiles,
    weight_fn,
    rebalance_every: int = 21,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Sharpe over a (lookback x quantile) grid.

    A single parameter pair looking good is not evidence; a *neighbourhood* of
    pairs looking good is. The output is meant to be read as a heatmap — a lone
    bright cell surrounded by dark ones is overfit.
    """
    out = pd.DataFrame(index=pd.Index(lookbacks, name="lookback"),
                       columns=pd.Index(quantiles, name="quantile"), dtype=float)
    bt = Backtester(returns, rebalance_every=rebalance_every, cost_bps=cost_bps)
    for lb in lookbacks:
        score = signal_fn(prices, lb)
        for q in quantiles:
            w = weight_fn(score, quantile=q)
            out.loc[lb, q] = metrics.sharpe_ratio(bt.run(w).returns)
    return out


# --------------------------------------------------------------------------
# Return-series views
# --------------------------------------------------------------------------
def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Trailing annualized Sharpe — shows *when* a strategy worked, not just if."""
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1)
    return (mu / sd * np.sqrt(TRADING_DAYS)).dropna()


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Calendar table of monthly compounded returns (rows=year, cols=month).

    Months in which the book was flat for every single day — the signal's
    warm-up period, before a 12-month lookback can be computed — come back as
    NaN rather than 0.0, so the calendar does not advertise a run of "flat"
    months that the strategy never actually had the chance to trade.
    """
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    never_traded = returns.ne(0).resample("ME").sum() == 0
    monthly[never_traded] = np.nan
    frame = monthly.to_frame("ret")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    return frame.pivot(index="year", columns="month", values="ret")


def cost_drag(result) -> pd.Series:
    """Cumulative performance lost to transaction costs, in equity-curve units."""
    return metrics.equity_curve(result.gross_returns) - metrics.equity_curve(result.returns)
