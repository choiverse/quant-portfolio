"""statarb — cointegration-based pair trading, conditioned on volatility regimes.

A dependency-light library (numpy/scipy/pandas/matplotlib only) for the
question "does statistical arbitrage on S&P 500 pairs survive transaction
costs, and does the answer depend on the volatility regime?". The econometrics
is built rather than imported, so the critical values, the likelihoods and the
EM recursions are all in this package and all checked against known answers by
``scripts/validate.py``.

Public API
----------
- data:          load_prices, to_price_matrix, to_log_prices, split_walk_forward
- cointegration: adf_test, engle_granger, ou_half_life, mackinnon_crit
- volatility:    realized_vol, ewma_vol, fit_garch11
- regimes:       GaussianHMM, fit_regimes, fit_causal, vol_tercile_regimes
- pairs:         screen_window, pair_positions, window_weights, PairSpec
- strategy:      run_walk_forward, run_strategy
- backtest:      PairBacktester, BacktestResult
- metrics:       performance_summary and the individual metric functions
- attribution:   regime_table, contribution, turnover_profile
- validation:    the correctness gates, shared by the script and the write-up
- plotting:      the report figures
- style:         shared matplotlib theme and validated palette
"""

from . import (
    attribution,
    backtest,
    cointegration,
    data,
    metrics,
    pairs,
    regimes,
    strategy,
    style,
    validation,
    volatility,
)
from .backtest import BacktestResult, PairBacktester
from .metrics import performance_summary
from .pairs import PairSpec
from .regimes import GaussianHMM

__all__ = [
    "attribution",
    "backtest",
    "cointegration",
    "data",
    "metrics",
    "pairs",
    "regimes",
    "strategy",
    "style",
    "validation",
    "volatility",
    "BacktestResult",
    "PairBacktester",
    "PairSpec",
    "GaussianHMM",
    "performance_summary",
]

__version__ = "0.1.0"
