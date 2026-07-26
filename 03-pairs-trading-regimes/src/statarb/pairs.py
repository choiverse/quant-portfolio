"""Pair selection and the spread trading rule.

The screen is a two-stage funnel, and the reason is both practical and
statistical.

*Practical*: 505 names make 127,260 pairs. An Engle-Granger test with
AIC lag selection is ~34 least-squares fits per pair, so testing all of them
in every walk-forward window is several million regressions — hours of work to
produce a table that is mostly noise anyway.

*Statistical*: it is the same problem from the other end. Testing 127,260
pairs at the 5% level produces roughly 6,400 rejections **when no pair is
cointegrated at all**. A screen that reports "we found 6,000 cointegrated
pairs" has found nothing; it has measured its own significance level. Stage 1
therefore cuts the field on a criterion that does not consume a hypothesis
test — the sum of squared deviations between normalised price paths, the
distance metric of Gatev et al. (2006) — and only the survivors are formally
tested.

This does not make the multiple-testing problem go away, and the code does not
pretend it does: ``screen_window`` reports the number of tests it ran and the
false positives expected at the chosen level, so the write-up can put the two
numbers next to each other.

Everything is fitted on a formation window and applied to the *following*
trading window. The hedge ratio, the spread mean and the spread standard
deviation are all frozen at the end of formation. Re-estimating them during
the trading window using the trading window's own data is the single easiest
way to manufacture a profitable pairs backtest, and it is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from . import cointegration as coint


@dataclass
class PairSpec:
    """Everything about a pair that is frozen at the end of formation."""

    y: str                  # the leg held long when the spread is cheap
    x: str                  # the hedge leg
    alpha: float            # cointegrating intercept
    beta: float             # hedge ratio, units of x per unit of y
    mu: float               # formation-window spread mean
    sigma: float            # formation-window spread standard deviation
    stat: float             # Engle-Granger statistic
    pvalue: float
    half_life: float
    ssd: float              # stage-1 distance

    @property
    def name(self) -> str:
        return f"{self.y}/{self.x}"

    def weights(self) -> tuple[float, float]:
        """Dollar weights on ``(y, x)`` for a long-spread position of gross 1.

        The spread is in logs, so ``d(spread) ~= r_y - beta*r_x``: one dollar
        of ``y`` against ``beta`` dollars of ``x``. Dividing by ``1 + |beta|``
        normalises every pair to the same gross exposure, so a pair with a
        hedge ratio of 3 does not silently get four times the capital of a
        pair with a hedge ratio of 1.
        """
        g = 1.0 + abs(self.beta)
        return 1.0 / g, -self.beta / g


# --------------------------------------------------------------------------
# Stage 1 — distance
# --------------------------------------------------------------------------


def distance_matrix(log_prices: pd.DataFrame) -> np.ndarray:
    """Pairwise sum of squared deviations between normalised price paths.

    Each column is rebased to start at zero and scaled to unit standard
    deviation, so the metric compares *shapes* rather than levels. Computed
    from the Gram matrix rather than pair by pair — ``||a-b||^2 = ||a||^2 +
    ||b||^2 - 2 a.b`` turns 127,260 pairwise loops into one matrix product.
    """
    z = log_prices - log_prices.iloc[0]
    sd = z.std(ddof=1).replace(0.0, np.nan)
    z = (z / sd).to_numpy()

    gram = z.T @ z
    sq = np.diag(gram)
    d = sq[:, None] + sq[None, :] - 2.0 * gram
    return np.maximum(d, 0.0)


def distance_screen(log_prices: pd.DataFrame, top_k: int = 1000) -> pd.DataFrame:
    """Rank all pairs by distance and keep the closest ``top_k``."""
    tickers = list(log_prices.columns)
    d = distance_matrix(log_prices)

    iu = np.triu_indices(len(tickers), k=1)
    flat = d[iu]
    valid = np.isfinite(flat)
    order = np.argsort(flat[valid])[:top_k]

    idx_i, idx_j = iu[0][valid][order], iu[1][valid][order]
    return pd.DataFrame(
        {
            "a": [tickers[i] for i in idx_i],
            "b": [tickers[j] for j in idx_j],
            "ssd": flat[valid][order],
        }
    )


def n_possible_pairs(n_tickers: int) -> int:
    """``C(n, 2)`` — the size of the field stage 1 is cutting down."""
    return n_tickers * (n_tickers - 1) // 2


# --------------------------------------------------------------------------
# Stage 2 — cointegration
# --------------------------------------------------------------------------


@dataclass
class ScreenResult:
    """Selected pairs plus the bookkeeping needed to judge the selection."""

    pairs: list[PairSpec]
    table: pd.DataFrame              # every tested candidate, passed or not
    n_universe: int
    n_possible: int
    n_tested: int
    level: float

    @property
    def n_passed(self) -> int:
        return int(self.table["passed"].sum())

    @property
    def expected_false_positives(self) -> float:
        """Rejections expected from the tests actually run, under the null."""
        return self.n_tested * self.level

    @property
    def expected_false_positives_untargeted(self) -> float:
        """Rejections expected had every possible pair been tested."""
        return self.n_possible * self.level


def screen_window(
    log_prices: pd.DataFrame,
    top_k: int = 1000,
    level: float = 0.05,
    max_lag: int = 10,
    max_half_life: float | None = 60.0,
    min_half_life: float = 1.0,
    max_pairs: int | None = 20,
) -> ScreenResult:
    """Select tradable pairs from one formation window.

    A candidate has to clear three hurdles, not one:

    1. survive the distance cut,
    2. reject no-cointegration at ``level``,
    3. have an Ornstein-Uhlenbeck half-life inside
       ``[min_half_life, max_half_life]``.

    The half-life bounds are there to exclude spreads that revert too slowly
    to trade inside the window and spreads whose apparent reversion is just
    bid-ask bounce. On this dataset the *upper* bound turns out to be almost
    entirely non-binding, and that is worth knowing rather than hiding: over a
    252-day formation window the Engle-Granger test only rejects for
    fast-reverting spreads in the first place, so of 2,293 rejections across
    the sample the slowest had a half-life of 19 days and only 10 were removed
    by the bounds at all — all of them at the lower end.

    Ranking the survivors by the strength of the evidence then pushes further
    in the same direction: the traded pairs have a median half-life of about 4
    days against 6 for all rejections, because the statistic is largest
    exactly where reversion is fastest. That is a selection effect working
    against the strategy, not for it — a spread that halves in 4 days is the
    one most likely to be microstructure noise, and the one least capturable
    with a one-day execution lag.
    """
    cand = distance_screen(log_prices, top_k=top_k)

    rows, specs = [], []
    for a, b, ssd in cand.itertuples(index=False):
        try:
            res = coint.engle_granger(log_prices[a], log_prices[b], max_lag=max_lag)
        except (ValueError, np.linalg.LinAlgError):
            continue

        # engle_granger tries both orderings and reports which one it kept.
        y_name, x_name = (b, a) if res.swapped else (a, b)

        passed = res.rejects(level)
        hl_ok = (
            min_half_life <= res.half_life <= max_half_life
            if max_half_life is not None
            else res.half_life >= min_half_life
        )
        rows.append(
            {
                "y": y_name, "x": x_name, "ssd": ssd,
                "stat": res.stat, "pvalue": res.pvalue,
                "crit_5pct": res.crit[0.05], "beta": res.beta,
                "half_life": res.half_life, "passed": passed,
                # "eligible" means it cleared the tests, not that it was
                # traded — only the top ``max_pairs`` of these are.
                "half_life_ok": hl_ok, "eligible": passed and hl_ok,
            }
        )
        if passed and hl_ok:
            spread = res.spread
            specs.append(
                PairSpec(
                    y=y_name, x=x_name, alpha=res.alpha, beta=res.beta,
                    mu=float(spread.mean()), sigma=float(spread.std(ddof=1)),
                    stat=res.stat, pvalue=res.pvalue,
                    half_life=res.half_life, ssd=ssd,
                )
            )

    table = pd.DataFrame(rows)
    # Rank by the strength of the evidence, not by distance: the distance was
    # only ever a device for making the test count tractable.
    specs.sort(key=lambda s: s.stat)
    if max_pairs is not None:
        specs = specs[:max_pairs]

    return ScreenResult(
        pairs=specs,
        table=table,
        n_universe=log_prices.shape[1],
        n_possible=n_possible_pairs(log_prices.shape[1]),
        n_tested=len(table),
        level=level,
    )


# --------------------------------------------------------------------------
# The trading rule
# --------------------------------------------------------------------------


def spread_series(log_prices: pd.DataFrame, spec: PairSpec) -> pd.Series:
    """The spread, using the hedge ratio frozen at formation."""
    s = log_prices[spec.y] - spec.alpha - spec.beta * log_prices[spec.x]
    return s.rename(f"spread_{spec.name}")


def zscore(spread: pd.Series, spec: PairSpec) -> pd.Series:
    """Standardise by the *formation* mean and sd, never the trading window's.

    Standardising a trading window by its own moments guarantees the spread
    looks mean-reverting over that window, because it has been centred on its
    own realised mean. That is the look-ahead this whole module is arranged to
    prevent.
    """
    if spec.sigma <= 0:
        return pd.Series(np.nan, index=spread.index, name="z")
    return ((spread - spec.mu) / spec.sigma).rename("z")


def pair_positions(
    z: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 4.0,
) -> pd.Series:
    """Convert a z-score path into a spread position in ``{-1, 0, +1}``.

    Enter against the deviation once ``|z| > entry``, close as it comes back
    inside ``exit``, and abandon the pair for the rest of the window if ``|z|``
    ever exceeds ``stop``.

    The stop is not risk-management decoration. A spread that reaches four
    formation standard deviations is evidence that the relationship estimated
    during formation has broken — a merger, a guidance cut, a sector rotation.
    Averaging into it is the trade that has historically ended pairs desks, so
    once stopped the pair is not re-entered even if the z-score comes back.

    The position on day ``t`` is decided from ``z_t``, i.e. the close of ``t``;
    applying it to returns is the caller's job and must lag by one day.
    """
    if entry <= exit:
        raise ValueError("entry threshold must exceed the exit threshold")

    vals = z.to_numpy()
    out = np.zeros(len(vals))
    pos = 0.0
    stopped = False

    for t, zt in enumerate(vals):
        if not np.isfinite(zt):
            out[t] = pos
            continue
        if stopped:
            out[t] = 0.0
            continue
        if abs(zt) > stop:
            pos, stopped = 0.0, True
        elif pos == 0.0:
            if zt > entry:
                pos = -1.0          # spread rich: short y, long x
            elif zt < -entry:
                pos = 1.0           # spread cheap: long y, short x
        elif abs(zt) < exit:
            pos = 0.0
        out[t] = pos

    return pd.Series(out, index=z.index, name="position")


def window_weights(
    log_prices: pd.DataFrame,
    specs: list[PairSpec],
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target dollar weights per ticker over a trading window.

    Capital is split equally across the selected pairs and each pair is run at
    gross exposure 1, so the book's gross exposure is at most 1 and is lower
    whenever some pairs are flat. Idle capital is left idle rather than
    levering the active pairs up to a constant gross — otherwise a window in
    which only one pair triggers would quietly become a single-pair bet at
    full size.

    Returns ``(weights, positions)``: weights indexed by date x ticker, and
    the ``{-1,0,1}`` spread position of every pair for diagnostics.
    """
    dates = log_prices.index
    weights = pd.DataFrame(0.0, index=dates, columns=log_prices.columns)
    positions = pd.DataFrame(0.0, index=dates, columns=[s.name for s in specs])

    if not specs:
        return weights, positions

    share = 1.0 / len(specs)
    for spec in specs:
        if spec.y not in log_prices.columns or spec.x not in log_prices.columns:
            continue
        z = zscore(spread_series(log_prices, spec), spec)
        pos = pair_positions(z, entry=entry, exit=exit, stop=stop)
        positions[spec.name] = pos

        wy, wx = spec.weights()
        weights[spec.y] += share * pos * wy
        weights[spec.x] += share * pos * wx

    return weights, positions


def all_pairs(tickers) -> list[tuple[str, str]]:
    """Every unordered pair — used to state the size of the untested field."""
    return list(combinations(sorted(tickers), 2))
