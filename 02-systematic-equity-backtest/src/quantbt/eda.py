"""Dataset profiling — what is actually in the file before any strategy touches it.

Every number a backtest produces inherits the defects of the panel underneath it,
so the panel gets audited first: how many names are listed on each day, where the
gaps are, how the daily cross-section is distributed, and which rows are
implausible enough to be data errors rather than market events.

These functions consume the *raw long* frame (one row per date-ticker) and the
*wide* matrices produced by :mod:`quantbt.data`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def panel_summary(raw: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Headline facts about the dataset, for the data card and the README."""
    span_days = (raw["date"].max() - raw["date"].min()).days
    return pd.Series(
        {
            "Rows (raw)": f"{len(raw):,}",
            "Unique tickers (raw)": f"{raw['Name'].nunique():,}",
            "Tickers after ≥98% presence filter": f"{prices.shape[1]:,}",
            "Trading days": f"{prices.shape[0]:,}",
            "Date range": f"{raw['date'].min():%Y-%m-%d} → {raw['date'].max():%Y-%m-%d}",
            "Calendar span": f"{span_days / 365.25:.1f} years",
            "Panel fill rate (raw)": f"{len(raw) / (raw['Name'].nunique() * raw['date'].nunique()):.1%}",
            "Missing OHLC cells": f"{int(raw[['open', 'high', 'low', 'close']].isna().sum().sum()):,}",
        },
        name="value",
    )


def listing_coverage(raw: pd.DataFrame) -> pd.Series:
    """Number of tickers with a price on each trading day.

    A rising line is the fingerprint of survivorship-biased construction: names
    enter as they list but nothing ever leaves, because the universe was frozen
    using end-of-sample membership.
    """
    return raw.groupby("date")["Name"].nunique().sort_index()


def ticker_completeness(raw: pd.DataFrame) -> pd.Series:
    """Fraction of the full calendar each ticker is present for, ascending."""
    n_days = raw["date"].nunique()
    return (raw.groupby("Name").size() / n_days).sort_values()


def missingness_by_column(raw: pd.DataFrame) -> pd.Series:
    """Null rate per column of the raw file."""
    return (raw.isna().sum() / len(raw)).sort_values(ascending=False)


def quality_flags(raw: pd.DataFrame) -> pd.DataFrame:
    """Row-level integrity checks with a count and a pass/fail verdict.

    These are the checks that catch the failure modes that silently poison a
    backtest: a ``high`` below the ``low`` means the OHLC bar is corrupt; a
    non-positive price breaks every log-return; a duplicated (date, ticker) pair
    silently doubles a name's weight in the cross-section.
    """
    ohlc = raw[["open", "high", "low", "close"]]
    checks = {
        "Duplicated (date, ticker) rows": int(raw.duplicated(["date", "Name"]).sum()),
        "Non-positive prices": int((ohlc <= 0).any(axis=1).sum()),
        "high < low": int((raw["high"] < raw["low"]).sum()),
        "close outside [low, high]": int(
            ((raw["close"] > raw["high"]) | (raw["close"] < raw["low"])).sum()
        ),
        "Missing close": int(raw["close"].isna().sum()),
        "Zero volume days": int((raw["volume"] == 0).sum()),
        "|daily move| > 50% (split/error suspects)": int(
            (raw.groupby("Name")["close"].pct_change().abs() > 0.5).sum()
        ),
    }
    out = pd.DataFrame({"count": pd.Series(checks)})
    out["share of rows"] = out["count"] / len(raw)
    # Only the structural checks are hard failures; large moves and zero volume
    # are legitimate market events that merely warrant a look.
    hard = [
        "Duplicated (date, ticker) rows",
        "Non-positive prices",
        "high < low",
        "close outside [low, high]",
        "Missing close",
    ]
    out["verdict"] = np.where(
        out.index.isin(hard), np.where(out["count"] == 0, "PASS", "FAIL"), "review"
    )
    return out


def integrity_violations(raw: pd.DataFrame) -> pd.DataFrame:
    """The individual rows that fail a structural OHLC check, with the reason.

    Small enough to commit and read: the point of a data card is to name the bad
    rows, not just count them.
    """
    bad_bar = raw["high"] < raw["low"]
    bad_close = (raw["close"] > raw["high"]) | (raw["close"] < raw["low"])
    missing = raw[["open", "high", "low", "close"]].isna().any(axis=1)

    reason = pd.Series("", index=raw.index, dtype=object)
    reason[bad_close] = "close outside [low, high]"
    reason[bad_bar] = "high < low"
    reason[missing] = "missing OHLC field"

    hit = reason != ""
    out = raw.loc[hit, ["date", "Name", "open", "high", "low", "close", "volume"]].copy()
    out["reason"] = reason[hit]
    return out.sort_values(["reason", "date"]).reset_index(drop=True)


def extreme_moves(raw: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Day-over-day close moves beyond ``threshold``, with the surrounding prices.

    In a *price-adjusted* series a ±50% single-day move is almost always a real
    event. In this file it is usually not: the closes are unadjusted for splits
    and spin-offs, so a corporate action shows up as a fabricated return that a
    momentum or reversal signal will happily trade on. Listing them explicitly is
    how you tell a genuine tail event from an artefact of the vendor's
    adjustment policy.
    """
    chg = raw.groupby("Name")["close"].pct_change()
    prev = raw.groupby("Name")["close"].shift(1)
    hit = chg.abs() > threshold
    out = raw.loc[hit, ["date", "Name", "close"]].copy()
    out["prev_close"] = prev[hit]
    out["pct_change"] = chg[hit]
    return out.sort_values("pct_change").reset_index(drop=True)


def cross_sectional_dispersion(returns: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional standard deviation and quartile spread of returns.

    Dispersion is the raw material a cross-sectional strategy trades: when every
    name moves together there is no relative bet to make, however good the
    signal. Plotting it explains *when* a factor could have worked at all.
    """
    return pd.DataFrame(
        {
            "std": returns.std(axis=1),
            "iqr": returns.quantile(0.75, axis=1) - returns.quantile(0.25, axis=1),
        }
    ).dropna()


def normal_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Written out rather than imported so the package keeps its NumPy/pandas-only
    dependency footprint. Accurate to ~1e-9 in the relative sense, which is far
    more than a Q-Q plot needs.
    """
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(p)

    lo = p < plow
    q = np.sqrt(-2 * np.log(np.where(lo, p, plow)))
    out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])[lo] / \
              ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)[lo]

    hi = p > phigh
    q = np.sqrt(-2 * np.log(np.where(hi, 1 - p, plow)))
    out[hi] = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])[hi] /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)[hi])

    mid = ~lo & ~hi
    q = np.where(mid, p, 0.5) - 0.5
    r = q * q
    out[mid] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q)[mid] / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)[mid]
    return out


def return_distribution_stats(returns: pd.DataFrame) -> pd.Series:
    """Pooled moments of the daily return panel, against the normal benchmark.

    Excess kurtosis far above zero is the standard finding and is the reason
    Sharpe ratios understate tail risk — worth stating explicitly rather than
    reporting Sharpe as if returns were Gaussian.
    """
    flat = returns.to_numpy().ravel()
    flat = flat[np.isfinite(flat)]
    mu, sd = flat.mean(), flat.std(ddof=1)
    z = (flat - mu) / sd
    return pd.Series(
        {
            "N observations": float(flat.size),
            "Mean (daily)": mu,
            "Std (daily)": sd,
            "Skewness": float((z ** 3).mean()),
            "Excess kurtosis": float((z ** 4).mean() - 3.0),
            "1st percentile": float(np.percentile(flat, 1)),
            "99th percentile": float(np.percentile(flat, 99)),
            "Worst day": float(flat.min()),
            "Best day": float(flat.max()),
        },
        name="value",
    )
