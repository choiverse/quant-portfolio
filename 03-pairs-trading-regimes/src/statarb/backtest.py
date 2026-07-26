"""A daily-rebalanced backtester for a book of pair positions.

Project 02's engine rebalanced on a fixed calendar grid, which is right for a
cross-sectional factor that is re-ranked monthly. A pair book is different:
positions open and close when a z-score crosses a threshold, on whatever day
that happens. So this engine takes a full panel of target weights and assumes
the book is brought to target every day — which for a pair strategy is not an
assumption at all, since the weights only change on the days a pair actually
trades.

The one thing that carries over unchanged is the execution convention, because
it is the thing most easily got wrong:

    weights decided from the close of day t  ->  earn the return of day t+1

Everything downstream is defined on the *executed* book, ``weights.shift(1)``,
so a turnover charge can never be attributed to a day on which the position
was not yet held.
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

        ``mean(gross) = bps/1e4 * mean(turnover)`` solved for ``bps``. A number
        below what the strategy would actually pay is the whole verdict: the
        edge exists, it is just smaller than the bill for capturing it.
        """
        t = self.turnover.mean()
        if t <= 0:
            return np.inf
        return float(1e4 * self.gross_returns.mean() / t)


class PairBacktester:
    """Run a panel of target weights against a return panel.

    Parameters
    ----------
    returns : DataFrame
        Daily simple returns, index=date, columns=ticker.
    cost_bps : float
        Charged per unit of one-way turnover, in basis points. 10 bps is the
        default used throughout, matching project 02 so the two are comparable.
    """

    def __init__(self, returns: pd.DataFrame, cost_bps: float = 10.0):
        self.returns = returns
        self.cost_bps = cost_bps
        self.cost_rate = cost_bps / 1e4

    def run(self, target_weights: pd.DataFrame, name: str = "pairs") -> BacktestResult:
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

        return BacktestResult(
            returns=net,
            gross_returns=gross,
            weights=held,
            turnover=turnover.rename("turnover"),
            costs=costs.rename("costs"),
            name=name,
        )

    def cost_sweep(
        self,
        target_weights: pd.DataFrame,
        bps_grid,
        name: str = "pairs",
    ) -> pd.DataFrame:
        """Re-price the same trades at a range of cost assumptions."""
        base = self.run(target_weights, name=name)
        return cost_sweep_from(base.gross_returns, base.turnover, bps_grid)


def cost_sweep_from(
    gross_returns: pd.Series,
    turnover: pd.Series,
    bps_grid,
) -> pd.DataFrame:
    """Cost sensitivity computed from an existing gross/turnover pair.

    The positions are identical across the grid — only the bill changes — so
    this isolates cost sensitivity from any strategy response to it.

    Taking the two series as arguments rather than re-running the engine is
    what keeps the sweep consistent with the headline. Re-running on a
    date-restricted weight panel silently re-expands it against the full
    return index, pads the ends with zero-return days, and produces a Sharpe
    at the charged cost that does not match the one reported everywhere else.
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
