"""Data loading and shaping for the cross-sectional learning problem.

The raw file is the same Kaggle "S&P 500 daily OHLCV, 2013-2018" panel used by
projects 02 and 03 (``all_stocks_5yr.csv``); project 02's data card audits it
and this project's card records only what changes.

What changes is the shape the problem needs. Projects 02 and 03 consumed the
panel as *matrices* — dates x tickers, one number per cell. A supervised
learner needs the same information as a **stacked design matrix**: one row per
(date, ticker), one column per feature, plus a label. The conversion is
mechanical but it is also where leakage gets in, so it lives in one place with
one rule: ``stack_panel`` never sees a forward-looking column it did not
receive as the explicit ``target`` argument, and every row keeps the date it
was *known on*, never the date its label resolves.

Two matrices are carried through rather than one. ``close`` drives the
features and the labels; ``volume`` is needed for the liquidity features and
is the one field projects 02 and 03 never used.
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


def to_matrix(
    df: pd.DataFrame,
    field: str = "close",
    min_obs_frac: float = 1.0,
) -> pd.DataFrame:
    """Pivot one field to a wide matrix (index=date, cols=ticker).

    ``min_obs_frac=1.0`` keeps only names with a complete history, matching
    project 03 rather than project 02's 0.98. The reason here is different
    from project 03's: a learner trained on a panel with ragged coverage
    learns the *coverage pattern* as well as the signal, because a name that
    appears halfway through the sample carries a systematically different
    feature distribution (short histories mean fresher listings). Dropping the
    35 incomplete names costs universe breadth and buys a design matrix whose
    rows are exchangeable. The data card records what that filter deletes.
    """
    wide = df.pivot(index="date", columns="Name", values=field).sort_index()
    n_days = len(wide)
    keep = wide.columns[wide.notna().sum() >= min_obs_frac * n_days]
    wide = wide[keep].ffill()
    wide.columns.name = "ticker"
    return wide


def load_panel(csv_path: str | Path, min_obs_frac: float = 1.0) -> dict[str, pd.DataFrame]:
    """Load the raw file and return the aligned ``close`` / ``volume`` matrices.

    Both matrices are restricted to the *same* tickers and dates, so a feature
    built from volume and one built from price can never disagree about which
    cells exist.
    """
    raw = load_prices(csv_path)
    close = to_matrix(raw, "close", min_obs_frac=min_obs_frac)
    volume = to_matrix(raw, "volume", min_obs_frac=min_obs_frac)
    high = to_matrix(raw, "high", min_obs_frac=min_obs_frac)
    low = to_matrix(raw, "low", min_obs_frac=min_obs_frac)

    tickers = close.columns.intersection(volume.columns)
    tickers = tickers.intersection(high.columns).intersection(low.columns)
    dates = close.index
    out = {
        "close": close.loc[dates, tickers],
        "volume": volume.loc[dates, tickers],
        "high": high.loc[dates, tickers],
        "low": low.loc[dates, tickers],
    }
    complete = out["close"].notna().all(axis=1) if min_obs_frac >= 1.0 else slice(None)
    if min_obs_frac >= 1.0:
        for k in out:
            out[k] = out[k].loc[complete]
    return out


def to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Daily returns. ``simple`` for P&L aggregation, ``log`` for modelling."""
    if kind == "simple":
        rets = prices.pct_change()
    elif kind == "log":
        rets = np.log(prices).diff()
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    return rets.iloc[1:]


def forward_return(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """The label: simple return from ``t`` to ``t + horizon``.

    Indexed by ``t`` — the day the position would be taken — so that a row of
    the design matrix pairs features known at ``t`` with the return that
    follows. The last ``horizon`` rows are ``NaN`` by construction: their
    labels have not happened yet, and no amount of care elsewhere makes them
    usable.

    Note what this does *not* do. It does not shift the label back to make the
    panel look longer, and it does not fill the tail. Both are common and both
    silently train the model on returns it could not have observed.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return prices.shift(-horizon) / prices - 1.0


def market_return(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight average return across the universe — the market proxy.

    Equal weight rather than cap weight because the dataset carries no share
    counts; the data card records this as a known approximation, as it does in
    project 03.
    """
    out = returns.mean(axis=1)
    out.name = "market"
    return out


def stack_panel(
    features: dict[str, pd.DataFrame],
    target: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Melt a dict of wide feature matrices into a stacked design matrix.

    Returns a frame indexed by ``(date, ticker)`` with one column per feature
    and, if ``target`` is given, a ``y`` column. Rows with any missing feature
    or a missing label are dropped: an imputation rule is a modelling choice
    that would need its own validation, and the honest alternative — drop the
    row — costs only the burn-in period where the long lookbacks are undefined.

    The date in the index is always the date the features were *known*. It is
    the anchor every leakage control in this project keys off, so nothing
    downstream is permitted to re-index it.
    """
    if not features:
        raise ValueError("no features given")

    first = next(iter(features.values()))
    dates, tickers = first.index, first.columns

    # Built by raveling explicitly aligned matrices rather than by
    # ``DataFrame.stack``, whose NaN-dropping and its keyword have changed
    # across pandas 2.x/3.0. This is version-proof and, incidentally, faster.
    columns = {}
    for name, mat in features.items():
        columns[name] = mat.reindex(index=dates, columns=tickers).to_numpy().ravel()
    if target is not None:
        columns["y"] = target.reindex(index=dates, columns=tickers).to_numpy().ravel()

    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    design = pd.DataFrame(columns, index=index)
    return design.dropna(how="any")


def split_expanding(
    dates: pd.DatetimeIndex,
    initial_train: int = 504,
    test_size: int = 63,
) -> list[tuple[slice, slice]]:
    """Expanding-window walk-forward splits over a date index.

    Returns ``(train, test)`` positional slices. The training window grows —
    a learner with 30-odd parameters wants every observation it can get — while
    the test windows tile the remaining sample without overlap, so the
    out-of-sample predictions concatenate into one continuous series.

    Purging and embargoing the boundary between the two is *not* done here;
    it is ``crossval.purge_split``'s job, because it depends on the label
    horizon and this function does not know it.
    """
    n = len(dates)
    splits = []
    start_test = initial_train
    while start_test + test_size <= n:
        splits.append((slice(0, start_test), slice(start_test, start_test + test_size)))
        start_test += test_size
    return splits
