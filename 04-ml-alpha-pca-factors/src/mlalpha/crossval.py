"""Walk-forward splits that survive overlapping labels.

The label in this project is a 5-day forward return, so the observation dated
Monday is *still resolving* on Friday. Any split that puts Monday in training
and Wednesday in testing has therefore let three days of the test period's
returns into the training set — not through a mis-shifted feature, but through
the label itself. The model never sees a future feature and still learns from
future prices.

The standard k-fold makes this worse by shuffling, but a naive walk-forward
does not escape it either: the boundary between train and test is exactly
where the overlap lives. The fix is López de Prado's, and it has two parts.

**Purge.** Drop from the training set every observation whose label window
reaches into the test period. With a training set that ends at ``T0`` and a
horizon of ``h``, that means dropping the last ``h`` training dates.

**Embargo.** Drop a further ``e`` days before the test period. Purging handles
the mechanical overlap; the embargo handles serial correlation, which is the
part purging misses. A training row from the day before the test period has a
label that does not overlap, but it is drawn from a market state so close to
the test period's that it functions as a partial copy of it — features and
returns are both autocorrelated across a boundary this thin.

The cost is a handful of observations per fold, out of tens of thousands.
The benefit is that the out-of-sample number means what it says.
``validation.gate_purge`` checks the property directly: after splitting, no
training observation's label window may intersect the test window, and the
count of violations must be exactly zero.

Reference
---------
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """One walk-forward fold, as positional indices into the design matrix."""

    index: int
    train_rows: np.ndarray
    test_rows: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_purged: int

    @property
    def summary(self) -> dict:
        return {
            "fold": self.index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_train": len(self.train_rows),
            "n_test": len(self.test_rows),
            "n_purged": self.n_purged,
        }


def purged_walk_forward(
    row_dates: pd.DatetimeIndex,
    horizon: int,
    initial_train: int = 504,
    test_size: int = 63,
    embargo: int = 5,
    expanding: bool = True,
    min_test_size: int | None = None,
) -> list[Fold]:
    """Build purged, embargoed walk-forward folds over a stacked design matrix.

    ``row_dates`` is the date of each row (repeated across tickers, as it comes
    out of ``data.stack_panel``). All the window sizes are in *trading days* —
    positions in the sorted set of unique dates — not calendar days, so a fold
    is not silently shortened by a holiday.

    ``expanding=True`` grows the training window from the start of the sample;
    ``False`` makes it a rolling window of ``initial_train`` days, which is
    what you want if you believe the relationship drifts. The write-up reports
    both.

    ``min_test_size`` lets the final fold be shorter than ``test_size`` rather
    than being discarded. It defaults to ``test_size`` — no partial folds — but
    the report run sets it lower, because with a 252-day feature burn-in the
    sample is short enough that throwing away the trailing two months of
    out-of-sample data is a real cost. A short final fold is a fold with a
    noisier estimate, not a biased one; discarding it would quietly end the
    evaluation before the most recent market conditions in the sample.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    dates = pd.DatetimeIndex(row_dates)
    unique = pd.DatetimeIndex(np.sort(pd.unique(dates)))
    # Position of every row within the sorted unique dates — the whole split is
    # then integer comparisons rather than repeated date lookups.
    pos = pd.Series(np.arange(len(unique)), index=unique).reindex(dates).to_numpy()

    smallest = test_size if min_test_size is None else int(min_test_size)

    folds: list[Fold] = []
    start_test = initial_train
    k = 0
    while start_test + smallest <= len(unique):
        stop_test = min(start_test + test_size, len(unique))
        train_lo = 0 if expanding else max(0, start_test - initial_train)

        # A training row at position p carries a label resolving at p+horizon.
        # It is admissible only if that whole window closes before the test
        # period opens, and before the embargo band in front of it.
        cutoff = start_test - horizon - embargo
        raw_train = (pos >= train_lo) & (pos < start_test)
        keep_train = raw_train & (pos < cutoff)
        test_mask = (pos >= start_test) & (pos < stop_test)

        n_purged = int(raw_train.sum() - keep_train.sum())
        train_rows = np.flatnonzero(keep_train)
        test_rows = np.flatnonzero(test_mask)

        if len(train_rows) and len(test_rows):
            folds.append(
                Fold(
                    index=k,
                    train_rows=train_rows,
                    test_rows=test_rows,
                    train_start=unique[train_lo],
                    train_end=unique[max(train_lo, cutoff - 1)],
                    test_start=unique[start_test],
                    test_end=unique[stop_test - 1],
                    n_purged=n_purged,
                )
            )
            k += 1
        start_test = stop_test

    return folds


def overlap_violations(
    row_dates: pd.DatetimeIndex,
    fold: Fold,
    horizon: int,
) -> int:
    """Count training rows whose label window reaches into the test window.

    The direct statement of what purging is for, computed from the dates
    rather than from the splitting logic — so it would still catch the error
    if ``purged_walk_forward`` computed its cutoff wrongly. Must be zero.
    """
    dates = pd.DatetimeIndex(row_dates)
    unique = pd.DatetimeIndex(np.sort(pd.unique(dates)))
    pos = pd.Series(np.arange(len(unique)), index=unique).reindex(dates).to_numpy()

    test_lo = pos[fold.test_rows].min()
    test_hi = pos[fold.test_rows].max()
    train_pos = pos[fold.train_rows]
    label_end = train_pos + horizon
    return int(np.sum((label_end >= test_lo) & (train_pos <= test_hi)))


def fold_table(folds: list[Fold]) -> pd.DataFrame:
    """The folds as a table, for the write-up and the report."""
    return pd.DataFrame([f.summary for f in folds]).set_index("fold")
