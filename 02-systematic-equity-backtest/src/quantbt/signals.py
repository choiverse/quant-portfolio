"""Cross-sectional signal construction.

Each signal maps a price/return history to a *score* per ticker on each date.
Scores are turned into dollar-neutral long/short target weights by
``long_short_weights``. All functions are vectorized over the whole panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """Classic 12-1 month momentum score.

    Score on day ``t`` is the trailing return from ``t-lookback`` to
    ``t-skip``. The one-month ``skip`` avoids the well-documented short-term
    reversal that contaminates raw 12-month momentum.
    """
    # price ratio P[t-skip] / P[t-lookback] - 1
    past = prices.shift(skip)
    base = prices.shift(lookback)
    score = past / base - 1.0
    return score


def short_term_reversal(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Short-term reversal score: the *negative* of the last-week return.

    Names that fell recently get a high (long) score, names that rose get a
    low (short) score.
    """
    past_return = prices / prices.shift(lookback) - 1.0
    return -past_return


def long_short_weights(
    score: pd.DataFrame,
    quantile: float = 0.2,
    gross: float = 1.0,
) -> pd.DataFrame:
    """Convert cross-sectional scores into dollar-neutral target weights.

    On each date, go long the top ``quantile`` of names and short the bottom
    ``quantile``, equal-weighted within each leg. Long and short legs each get
    ``gross/2`` of capital, so the book is dollar-neutral with gross exposure
    ``gross`` (e.g. gross=1.0 -> 50% long / 50% short).

    Rows with too few valid scores produce all-zero (flat) weights.
    """
    weights = pd.DataFrame(
        0.0, index=score.index, columns=score.columns, dtype=float
    )

    for dt, row in score.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n < 10:  # need a meaningful cross-section
            continue

        k = max(1, int(np.floor(quantile * n)))
        ranked = valid.sort_values()
        shorts = ranked.index[:k]
        longs = ranked.index[-k:]

        weights.loc[dt, longs] = (gross / 2.0) / k
        weights.loc[dt, shorts] = -(gross / 2.0) / k

    return weights


def long_only_benchmark(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight, fully-invested long-only benchmark weights.

    Every currently-listed name gets 1/N. Used as the passive comparison for
    the long/short strategies.
    """
    mask = prices.notna()
    counts = mask.sum(axis=1).replace(0, np.nan)
    weights = mask.div(counts, axis=0).fillna(0.0)
    return weights
