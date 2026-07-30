"""Turning a prediction into a book.

A model outputs a number per (date, ticker). A backtest needs weights. What
happens in between is where a plausible signal becomes an implausible strategy,
and there are only two decisions to make — how many names to hold, and how
often to change them — but the second one dominates everything in this project.

**The turnover problem.** The label is a 5-day forward return, so the model is
answering "what will happen over the next week". Rebalancing to a fresh
prediction every day therefore trades a position five times over the horizon it
was meant to be held for, and pays for the privilege each time. The fix used
here is the staggered-portfolio construction of Jegadeesh and Titman (1993):
run ``h`` overlapping sub-books, each rebalanced once every ``h`` days, each
holding ``1/h`` of the capital. The aggregate book is the trailing ``h``-day
average of the daily target weights — a one-line rolling mean, and exactly the
right one, because it holds each day's prediction for precisely the horizon it
was trained to predict.

``§4`` of the write-up reports what this is worth, and it is the difference
between a strategy and a fee generator.

Reference
---------
- Jegadeesh, N. and Titman, S. (1993). *Returns to buying winners and selling
  losers.* Journal of Finance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def long_short_weights(
    score: pd.DataFrame,
    quantile: float = 0.2,
    gross: float = 1.0,
    min_names: int = 20,
) -> pd.DataFrame:
    """Dollar-neutral weights: long the top ``quantile``, short the bottom.

    Equal-weighted within each leg, each leg taking ``gross/2`` of capital, so
    the book is dollar-neutral with gross exposure ``gross``. Identical in
    construction to project 02's function — deliberately, so that a Sharpe here
    and a Sharpe there are produced by the same portfolio rule and any
    difference is the signal rather than the sizing.

    Vectorized over the whole panel rather than looped by date: with 1,000
    dates and four models plus a robustness grid, the loop is the slowest thing
    in the project by a wide margin and it does not need to be.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")

    ranks = score.rank(axis=1, method="first")
    counts = score.notna().sum(axis=1)
    k = np.maximum(1, np.floor(quantile * counts)).astype(float)

    enough = counts >= min_names
    is_long = ranks.gt(counts - k, axis=0)
    is_short = ranks.le(k, axis=0)

    leg = (gross / 2.0) / k
    weights = (
        is_long.astype(float).mul(leg, axis=0)
        - is_short.astype(float).mul(leg, axis=0)
    )
    weights = weights.where(enough, 0.0)
    return weights.fillna(0.0)


def overlapping(weights: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Average the last ``horizon`` days of target weights.

    This *is* the staggered-portfolio book: the average of ``h`` sub-books,
    each opened on a different day and held for ``h`` days. Turnover falls by
    roughly a factor of ``h`` because only ``1/h`` of the book is being rolled
    on any given day, while the average holding period rises to exactly the
    horizon the model was trained on.

    The first ``horizon-1`` rows are only partially populated and are returned
    with the shorter average rather than as ``NaN``: those are the opening days
    of the sample, the book is genuinely smaller then, and zeroing them would
    misstate the exposure rather than the returns.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if horizon == 1:
        return weights
    return weights.rolling(horizon, min_periods=1).mean()


def signal_to_book(
    score: pd.DataFrame,
    horizon: int = 5,
    quantile: float = 0.2,
    gross: float = 1.0,
    stagger: bool = True,
) -> pd.DataFrame:
    """Score -> target weights, with the staggered construction by default."""
    w = long_short_weights(score, quantile=quantile, gross=gross)
    return overlapping(w, horizon) if stagger else w


def neutralize(weights: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    """Project a set of unwanted exposures out of the book, cross-sectionally.

    ``exposures`` is a ``(tickers x factors)`` matrix — the PCA loadings, in
    practice. For each date the weight vector is regressed on the exposures and
    replaced by the residual, so the resulting book has (numerically) zero
    exposure to every column. This is how the write-up separates "the model
    found alpha" from "the model found a factor tilt": if the neutralized book
    keeps its return, the alpha was idiosyncratic; if it does not, it was beta
    with extra steps.

    The book is re-scaled back to its original gross exposure afterwards, so
    the comparison is between two books of the same size.
    """
    cols = weights.columns.intersection(exposures.index)
    w = weights[cols].to_numpy(dtype=float)
    b = exposures.reindex(cols).to_numpy(dtype=float)

    # One projection matrix for all dates: P = B (B'B)^-1 B'.
    btb = b.T @ b
    resid = w - (w @ b) @ np.linalg.solve(btb, b.T)

    out = pd.DataFrame(resid, index=weights.index, columns=cols)
    gross_before = weights[cols].abs().sum(axis=1)
    gross_after = out.abs().sum(axis=1).replace(0.0, np.nan)
    scaled = out.mul((gross_before / gross_after), axis=0).fillna(0.0)
    return scaled.reindex(columns=weights.columns).fillna(0.0)
