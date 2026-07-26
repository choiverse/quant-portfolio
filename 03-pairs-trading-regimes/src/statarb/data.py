"""Data loading and shaping for pair trading.

The raw file is the same Kaggle "S&P 500 daily OHLCV, 2013-2018" panel used by
project 02 (``all_stocks_5yr.csv``), and project 02's data card already audits
it. What changes here is what the shape has to support: pair screening needs
two aligned *price* histories with no gaps, and the strategy trades a spread in
**log** prices, because a log spread with a constant hedge ratio corresponds to
a fixed ratio of dollar exposures rather than a fixed share count.

The survivorship filter is the same idea as project 02's but bites much harder,
and the data card says so: requiring a complete history over the sample is
exactly the condition "this company did not blow up", which is the one thing a
pair trade is most exposed to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw long-format OHLCV file, sorted by (Name, date)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Price file not found: {csv_path}\n"
            "See data/README.md for how to obtain all_stocks_5yr.csv."
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df.sort_values(["Name", "date"]).reset_index(drop=True)


def to_price_matrix(
    df: pd.DataFrame,
    field: str = "close",
    min_obs_frac: float = 1.0,
) -> pd.DataFrame:
    """Pivot to a wide price matrix (index=date, cols=ticker).

    ``min_obs_frac`` defaults to 1.0 here, stricter than project 02's 0.98: a
    pair is only two names, so a single missing stretch in either leg silently
    changes what the hedge ratio was fitted on. Names without a complete
    history are dropped outright rather than patched.
    """
    wide = df.pivot(index="date", columns="Name", values=field).sort_index()
    n_days = len(wide)
    keep = wide.columns[wide.notna().sum() >= min_obs_frac * n_days]
    wide = wide[keep].ffill().dropna(how="any")
    wide.columns.name = "ticker"
    return wide


def to_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Natural log of the price matrix — the space the spread lives in."""
    return np.log(prices)


def to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Daily returns. ``simple`` for P&L aggregation, ``log`` for modelling."""
    if kind == "simple":
        rets = prices.pct_change()
    elif kind == "log":
        rets = np.log(prices).diff()
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    return rets.iloc[1:]


def split_walk_forward(
    index: pd.DatetimeIndex,
    formation_days: int = 252,
    trading_days: int = 126,
) -> list[tuple[slice, slice]]:
    """Non-overlapping walk-forward windows over a date index.

    Returns ``(formation, trading)`` index slices. Pairs are selected using
    *only* the formation window and then traded over the following window,
    which is the whole reason the screen can be trusted at all: selecting on
    the full sample and reporting performance on the full sample would make
    any set of series look cointegrated in hindsight.

    Windows do not overlap, so each trading day is traded by exactly one
    formation decision and the resulting return series can be concatenated.
    """
    n = len(index)
    windows = []
    start = 0
    while start + formation_days + trading_days <= n:
        f = slice(start, start + formation_days)
        t = slice(start + formation_days, start + formation_days + trading_days)
        windows.append((f, t))
        start += trading_days
    return windows


def market_return(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight average return across the universe — the market proxy.

    Used as the series whose volatility defines the regime. Equal weight
    rather than cap weight because the dataset carries no share counts; the
    data card records this as a known approximation.
    """
    out = returns.mean(axis=1)
    out.name = "market"
    return out
