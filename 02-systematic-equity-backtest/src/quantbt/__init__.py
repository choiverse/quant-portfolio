"""quantbt — a small, dependency-light backtesting engine for cross-sectional
equity strategies.

pandas/NumPy data wrangling, signal construction and empirical evaluation,
built as a quantitative-research workflow. Everything here is vectorized and
reproducible.

Public API
----------
- data:        load_prices, to_price_matrix, to_returns
- eda:         panel profiling and data-integrity checks on the raw file
- signals:     cross_sectional_momentum, short_term_reversal, long_short_weights
- backtest:    Backtester, BacktestResult
- metrics:     performance_summary and the individual metric functions
- diagnostics: information_coefficient, quantile_performance, cost_sensitivity
- plotting:    tearsheet and the report figures
- style:       shared matplotlib theme and validated palette
"""

from . import backtest, data, diagnostics, eda, metrics, signals, style
from .backtest import Backtester, BacktestResult
from .metrics import performance_summary

__all__ = [
    "backtest",
    "data",
    "diagnostics",
    "eda",
    "metrics",
    "signals",
    "style",
    "Backtester",
    "BacktestResult",
    "performance_summary",
]

__version__ = "0.2.0"
