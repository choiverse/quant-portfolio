"""The backtest engine: a panel of target weights against a panel of returns.

Structurally the same engine as projects 02 and 03, and that is deliberate —
three strategies scored by three different backtesters would not be
comparable, and the whole point of quoting a Sharpe of 0.4 here is that it
means what the 0.10 in project 03 meant.

The execution convention is the one thing worth restating, because it is the
thing most easily got wrong:

    weights decided from the close of day t  ->  earn the return of day t+1

Everything downstream is defined on the *executed* book, ``weights.shift(1)``,
so a turnover charge can never be attributed to a day on which the position
was not yet held. In this project the convention is doing more work than in
the previous two: a machine-learned signal changes every day, so a
one-day-early execution assumption would hand the strategy a full day of the
move it was trying to predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class BacktestResult:
    """Container for a single backtest run."""

    returns: pd.Series          # net daily portfolio returns
    gross_returns: pd.Series    # before transaction costs
    weights: pd.DataFrame       # the book actually held each day
    turnover: pd.Series         # one-way turnover traded each day
    costs: pd.Series            # cost charged each day
    name: str = "strategy"

    @property
    def equity(self) -> pd.Series:
        return metrics.equity_curve(self.returns)

    @property
    def gross_exposure(self) -> pd.Series:
        return self.weights.abs().sum(axis=1).rename("gross_exposure")

    @property
    def net_exposure(self) -> pd.Series:
        return self.weights.sum(axis=1).rename("net_exposure")

    def summary(self, **kwargs) -> pd.Series:
        return metrics.performance_summary(self.returns, name=self.name, **kwargs)

    def breakeven_cost_bps(self) -> float:
        """Cost, in bps of one-way turnover, at which the strategy earns zero.

        ``mean(gross) = bps/1e4 * mean(turnover)`` solved for ``bps``. The
        single most informative number about a high-turnover strategy: it
        converts "is there an edge" and "can it be captured" into one figure
        that can be compared against what execution actually costs.
        """
        t = self.turnover.mean()
        if t <= 0:
            return np.inf
        return float(1e4 * self.gross_returns.mean() / t)

    def holding_period(self) -> float:
        """Average days a unit of capital stays in place: gross / turnover.

        Reported alongside turnover because it is the interpretable form. A
        turnover of 0.4 is hard to reason about; "the average position is held
        for two and a half days, against a signal trained to predict five" is
        the sentence that shows a problem.
        """
        t = self.turnover.mean()
        g = self.gross_exposure.mean()
        return float(g / t) if t > 0 else np.inf


class Backtester:
    """Run a panel of target weights against a return panel.

    Parameters
    ----------
    returns : DataFrame
        Daily simple returns, index=date, columns=ticker.
    cost_bps : float
        Charged per unit of one-way turnover, in basis points. 10 bps is the
        default used throughout the portfolio, matching projects 02 and 03.
    """

    def __init__(self, returns: pd.DataFrame, cost_bps: float = 10.0):
        self.returns = returns
        self.cost_bps = cost_bps
        self.cost_rate = cost_bps / 1e4

    def run(self, target_weights: pd.DataFrame, name: str = "strategy") -> BacktestResult:
        rets = self.returns
        tw = target_weights.reindex(index=rets.index, columns=rets.columns).fillna(0.0)

        # The book actually held during day t was decided at the close of t-1.
        held = tw.shift(1).fillna(0.0)

        gross = (held * rets).sum(axis=1)

        # Turnover on day t is the change in the *held* book, so it is charged
        # on the same day the new position starts earning.
        turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
        costs = turnover * self.cost_rate

        net = gross - costs
        net.name = gross.name = name

        # Trim the leading and trailing all-flat days: the staggered book ramps
        # in over its first `horizon` days and the signal stops before the
        # return panel does. Leaving them in pads the series with exact zeros,
        # which deflates the volatility and inflates every ratio built on it.
        #
        # A day counts as live if the book was held *or* if it traded, and the
        # second half matters: the day the book unwinds to flat holds nothing
        # but pays the full cost of closing. Trimming on holdings alone would
        # drop that row and quietly hand the strategy a free exit.
        active = (held.abs().sum(axis=1) > 0) | (turnover > 0)
        if active.any():
            lo, hi = active.idxmax(), active[::-1].idxmax()
            keep = slice(lo, hi)
            gross, net = gross.loc[keep], net.loc[keep]
            turnover, costs = turnover.loc[keep], costs.loc[keep]
            held = held.loc[keep]

        return BacktestResult(
            returns=net,
            gross_returns=gross,
            weights=held,
            turnover=turnover.rename("turnover"),
            costs=costs.rename("costs"),
            name=name,
        )

    def cost_sweep(self, target_weights: pd.DataFrame, bps_grid, name: str = "strategy") -> pd.DataFrame:
        """Re-price the same trades at a range of cost assumptions."""
        base = self.run(target_weights, name=name)
        return cost_sweep_from(base.gross_returns, base.turnover, bps_grid)


def cost_sweep_from(gross_returns: pd.Series, turnover: pd.Series, bps_grid) -> pd.DataFrame:
    """Cost sensitivity computed from an existing gross/turnover pair.

    The positions are identical across the grid — only the bill changes — so
    this isolates cost sensitivity from any strategy response to it.
    """
    rows = []
    for bps in bps_grid:
        net = gross_returns - turnover * (bps / 1e4)
        rows.append(
            {
                "cost_bps": bps,
                "Sharpe": metrics.sharpe_ratio(net),
                "CAGR": metrics.cagr(net),
                "Ann. Volatility": metrics.ann_volatility(net),
                "Max Drawdown": metrics.max_drawdown(net),
            }
        )
    return pd.DataFrame(rows).set_index("cost_bps")


def block_bootstrap_sharpe(
    returns: pd.Series,
    n_boot: int = 2_000,
    block: int = 21,
    seed: int = 0,
    level: float = 0.90,
) -> tuple[float, float]:
    """Block-bootstrap confidence interval for the Sharpe ratio.

    Blocks of ``block`` days rather than individual days, because daily
    strategy returns are not independent — a staggered book holds the same
    positions for a week by construction, so an i.i.d. bootstrap would
    understate the uncertainty by exactly the amount the overlap creates.

    Same procedure as project 03's, so the intervals are comparable.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < block * 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))

    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.concatenate([r[s: s + block] for s in starts[i]])[:n]
        sd = sample.std(ddof=1)
        sharpes[i] = sample.mean() / sd * np.sqrt(252) if sd > 0 else np.nan

    lo = float(np.nanquantile(sharpes, (1 - level) / 2))
    hi = float(np.nanquantile(sharpes, 1 - (1 - level) / 2))
    return lo, hi
