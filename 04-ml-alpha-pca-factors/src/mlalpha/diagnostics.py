"""Signal diagnostics: does the prediction rank the cross-section?

The natural regression metric for this problem is R^2, and it is close to
useless here. A 5-day cross-sectional return is ~99% noise, so a model with
genuine skill still posts an out-of-sample R^2 within rounding distance of
zero — often *below* zero, because the small amount of signal it has does not
pay for the variance of its own predictions. That is a true statement about
the data, not a broken model, and §3 of the write-up reports it as one.

What the strategy actually needs is much weaker than magnitude accuracy: it
needs the *ordering* to be better than chance, because it buys the top quintile
and sells the bottom one and never cares by how much. That is what the
information coefficient measures — the cross-sectional rank correlation between
prediction and outcome, computed date by date and then treated as a time series
in its own right.

The number that matters is not the average IC but the **information ratio** of
the IC: its mean divided by its standard deviation, annualized. An average IC
of 0.02 that is positive on 55% of days is a strategy; an average IC of 0.06
that comes from four spectacular days out of a thousand is a story about four
days. Both are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rank_ic(signal: pd.DataFrame, forward: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman correlation between signal and outcome.

    Computed as a Pearson correlation of within-date ranks, vectorized across
    the whole panel — a per-date ``scipy.stats.spearmanr`` call over 1,000
    dates is the slowest possible way to get the same numbers.

    Dates with fewer than 10 valid pairs return ``NaN`` rather than a
    correlation computed on a handful of names.
    """
    common = signal.columns.intersection(forward.columns)
    s = signal[common].reindex(index=forward.index)
    f = forward[common]
    valid = s.notna() & f.notna()

    s = s.where(valid).rank(axis=1)
    f = f.where(valid).rank(axis=1)

    n = valid.sum(axis=1)
    s = s.sub(s.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)

    num = (s * f).sum(axis=1)
    den = np.sqrt((s ** 2).sum(axis=1) * (f ** 2).sum(axis=1))
    ic = (num / den.replace(0, np.nan)).where(n >= 10)
    return ic.rename("ic")


def ic_summary(ic: pd.Series, horizon: int = 1) -> pd.Series:
    """Mean IC, its dispersion, the information ratio, and a t-statistic.

    ``horizon`` matters. Overlapping labels make consecutive daily ICs
    positively correlated, so the naive ``t = mean/se * sqrt(n)`` overstates
    significance by roughly ``sqrt(horizon)``. The correction applied here is
    the standard one for overlapping observations — divide the effective
    sample size by the horizon — which is conservative and, unlike a
    Newey-West correction on the same series, cannot be accused of being
    tuned. Both the raw and corrected t-statistics are returned so the size of
    the adjustment is visible rather than buried.
    """
    x = ic.dropna()
    n = len(x)
    if n < 2:
        return pd.Series(dtype=float)

    mean, sd = float(x.mean()), float(x.std(ddof=1))
    t_raw = mean / sd * np.sqrt(n) if sd > 0 else np.nan
    t_adj = t_raw / np.sqrt(horizon) if np.isfinite(t_raw) else np.nan
    return pd.Series(
        {
            "IC mean": mean,
            "IC std": sd,
            "IC IR": mean / sd if sd > 0 else np.nan,
            "IC IR (ann.)": mean / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
            "t-stat (naive)": t_raw,
            f"t-stat (/sqrt {horizon})": t_adj,
            "hit rate": float((x > 0).mean()),
            "days": float(n),
        },
        name=ic.name or "ic",
    )


def ic_decay(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    horizons=(1, 2, 3, 5, 10, 21, 42, 63),
) -> pd.DataFrame:
    """Mean IC of the same signal against forward returns at several horizons.

    The shape of this curve says what kind of signal it is. A signal that
    peaks at one day and is gone by five is microstructure and cannot survive
    a one-day execution lag; one that holds out to a month is slower, cheaper
    to trade, and probably a known risk premium.
    """
    rows = []
    for h in horizons:
        fwd = prices.shift(-h) / prices - 1.0
        fwd = fwd.sub(fwd.mean(axis=1), axis=0)
        ic = rank_ic(signal, fwd)
        rows.append(
            {
                "horizon": h,
                "IC mean": float(ic.mean()),
                "IC IR": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else np.nan,
                "hit rate": float((ic.dropna() > 0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def quantile_returns(
    signal: pd.DataFrame,
    forward: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Mean forward return by signal quantile, plus the top-minus-bottom spread.

    The monotonicity of this table is the check the IC cannot give. An IC of
    0.03 is consistent with a signal that works only in the tails and with one
    that orders the whole cross-section; only the quantile profile
    distinguishes them, and only the second kind is robust to how many names
    the book holds.
    """
    common = signal.columns.intersection(forward.columns)
    s = signal[common].reindex(index=forward.index)
    f = forward[common]
    valid = s.notna() & f.notna()

    ranks = s.where(valid).rank(axis=1, pct=True)
    out = {}
    edges = np.linspace(0, 1, n_quantiles + 1)
    for q in range(n_quantiles):
        lo, hi = edges[q], edges[q + 1]
        mask = (ranks > lo) & (ranks <= hi) if q else (ranks >= 0) & (ranks <= hi)
        out[f"Q{q + 1}"] = f.where(mask & valid).mean(axis=1)

    frame = pd.DataFrame(out)
    frame["Q{}-Q1".format(n_quantiles)] = frame[f"Q{n_quantiles}"] - frame["Q1"]
    return frame


def feature_ic_table(
    features: dict[str, pd.DataFrame],
    forward: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """IC of every raw feature on its own, sorted by information ratio.

    The reference point for everything the models do. If the best single
    feature has an IR of 0.05 and the boosted tree has 0.06, the boosted tree
    has bought 0.01 for 300 trees' worth of complexity and a much less stable
    signal — and that comparison is the project's central question.
    """
    rows = []
    for name, mat in features.items():
        ic = rank_ic(mat, forward)
        s = ic_summary(ic, horizon=horizon)
        if s.empty:
            continue
        rows.append(
            {
                "feature": name,
                "IC mean": s["IC mean"],
                "IC IR": s["IC IR"],
                "t-stat": s[f"t-stat (/sqrt {horizon})"],
                "hit rate": s["hit rate"],
            }
        )
    table = pd.DataFrame(rows).set_index("feature")
    return table.reindex(table["IC IR"].abs().sort_values(ascending=False).index)


def ic_split_table(
    features: dict[str, pd.DataFrame],
    forward: pd.DataFrame,
    split: pd.Timestamp,
    family: dict[str, str] | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Each feature's IC in the training period and in the test period.

    The table behind the project's central observation. It lives here rather
    than in a script because two scripts need it and a second copy of the
    definition is a second chance for the two halves of the write-up to
    disagree about what "training period" means.

    ``dates`` restricts every feature to a common window, and the callers pass
    the design matrix's own dates. Without it each feature is scored over its
    own availability — 12-1 momentum needs a 252-day burn-in, 5-day reversal
    needs five — so the training-period information ratios would be measured
    over windows ranging from 505 to 752 days and across different market
    conditions. Comparing them would then be partly a comparison of calendars.
    Restricting to the design matrix's dates makes every number in this table
    describe exactly the data the models were shown.
    """
    rows = []
    for name, mat in features.items():
        if dates is not None:
            mat = mat.reindex(index=dates)
        ic = rank_ic(mat, forward)
        tr, oos = ic.loc[:split].dropna(), ic.loc[split:].dropna()
        if len(tr) < 2 or len(oos) < 2:
            continue
        rows.append(
            {
                "feature": name,
                "family": (family or {}).get(name, ""),
                "IC_train": float(tr.mean()),
                "IR_train": float(tr.mean() / tr.std(ddof=1)),
                "IC_oos": float(oos.mean()),
                "IR_oos": float(oos.mean() / oos.std(ddof=1)),
                "sign_held": bool(np.sign(tr.mean()) == np.sign(oos.mean())),
                "days_train": len(tr),
                "days_oos": len(oos),
            }
        )
    table = pd.DataFrame(rows).set_index("feature")
    return table.reindex(table["IR_train"].abs().sort_values(ascending=False).index)


def turnover_of(signal: pd.DataFrame) -> pd.Series:
    """Day-to-day rank instability of a signal, in [0, 1].

    Mean absolute change in each name's percentile rank. A model-free measure
    of how fast a signal moves, computed before any portfolio construction —
    so it separates "the signal is jumpy" from "the sizing rule is jumpy".
    """
    ranks = signal.rank(axis=1, pct=True)
    return (ranks - ranks.shift(1)).abs().mean(axis=1).rename("rank_turnover")
