"""Data loading and shaping.

The raw dataset is the Kaggle "S&P 500 daily OHLCV, 2013-2018" file
(``all_stocks_5yr.csv``). It is a long/tidy table with one
row per (date, ticker). Quant research almost always works on a *wide* price
matrix (dates x tickers), so the helpers here pivot into that shape and derive
returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw long-format OHLCV file.

    Returns a DataFrame with a parsed ``date`` column and a ``Name`` ticker
    column, sorted by (Name, date).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Price file not found: {csv_path}\n"
            "See data/README.md for how to obtain all_stocks_5yr.csv."
        )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values(["Name", "date"]).reset_index(drop=True)
    return df


def to_price_matrix(
    df: pd.DataFrame,
    field: str = "close",
    min_obs_frac: float = 0.98,
) -> pd.DataFrame:
    """Pivot the long table into a wide price matrix (index=date, cols=ticker).

    Tickers that are not present for at least ``min_obs_frac`` of the trading
    days are dropped — this removes names that IPO'd or were delisted mid-sample
    and would otherwise inject look-ahead / survivorship artefacts into the
    cross-section. Remaining gaps are forward-filled.
    """
    wide = df.pivot(index="date", columns="Name", values=field).sort_index()

    n_days = len(wide)
    keep = wide.columns[wide.notna().sum() >= min_obs_frac * n_days]
    wide = wide[keep]

    wide = wide.ffill().dropna(how="any")
    wide.columns.name = "ticker"
    return wide


def to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Convert a price matrix to a period (daily) return matrix.

    ``kind='simple'`` gives arithmetic returns (used for portfolio P&L
    aggregation across names); ``kind='log'`` gives log returns.
    """
    if kind == "simple":
        rets = prices.pct_change()
    elif kind == "log":
        rets = np.log(prices).diff()
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    return rets.iloc[1:]
