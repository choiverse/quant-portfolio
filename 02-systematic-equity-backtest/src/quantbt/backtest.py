"""A small vectorized backtester with periodic rebalancing and costs.

Design
------
A *signal function* produces target weights for every date. The backtester
only actually trades on ``rebalance_every``-th day; between rebalances the
book drifts with the market. Crucially, weights decided using data up to the
close of day ``t`` are applied to the return of day ``t+1`` (one-day
execution lag), which removes look-ahead bias. Transaction costs are charged
on the traded turnover at each rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class BacktestResult:
    """Container for a single backtest run."""

    returns: pd.Series          # net daily portfolio returns
    gross_returns: pd.Series    # before transaction costs
    weights: pd.DataFrame       # realized weights held each day
    turnover: pd.Series         # one-way turnover charged at rebalances
    name: str = "strategy"

    @property
    def equity(self) -> pd.Series:
        return metrics.equity_curve(self.returns)

    def summary(self, **kwargs) -> pd.Series:
        return metrics.performance_summary(self.returns, name=self.name, **kwargs)


class Backtester:
    """Run a target-weight strategy against a return panel.

    Parameters
    ----------
    returns : DataFrame
        Daily simple returns, index=date, columns=ticker.
    rebalance_every : int
        Trade every N trading days (21 ~= monthly).
    cost_bps : float
        Round-trip cost is applied as ``cost_bps`` basis points per unit of
        one-way turnover (e.g. 10 bps).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        rebalance_every: int = 21,
        cost_bps: float = 10.0,
    ):
        self.returns = returns
        self.rebalance_every = rebalance_every
        self.cost_rate = cost_bps / 1e4

    def run(
        self,
        target_weights: pd.DataFrame,
        name: str = "strategy",
    ) -> BacktestResult:
        """Backtest a full panel of target weights.

        ``target_weights`` is sampled on rebalance dates; the sampled weights
        are held (fixed, not drift-adjusted) until the next rebalance. Applied
        to next-day returns to avoid look-ahead.
        """
        rets = self.returns
        tw = target_weights.reindex(index=rets.index, columns=rets.columns).fillna(0.0)

        dates = rets.index
        rebal_dates = dates[:: self.rebalance_every]

        # Held weights: take the target only on rebalance dates and forward-fill
        # it across the days in between (weights held fixed until next rebalance).
        held = tw.loc[rebal_dates].reindex(dates).ffill().fillna(0.0)

        # One-day execution lag: weights set at close of t act on return t+1.
        held_lagged = held.shift(1).fillna(0.0)

        gross_ret = (held_lagged * rets).sum(axis=1)

        # Turnover = sum |w_new - w_old| at each rebalance date.
        prev = held.shift(1).fillna(0.0)
        turnover_full = (held - prev).abs().sum(axis=1)
        turnover = turnover_full[turnover_full.index.isin(rebal_dates)]
        cost = turnover_full * self.cost_rate

        net_ret = gross_ret - cost
        net_ret.name = name
        gross_ret.name = name

        return BacktestResult(
            returns=net_ret,
            gross_returns=gross_ret,
            weights=held_lagged,
            turnover=turnover,
            name=name,
        )
