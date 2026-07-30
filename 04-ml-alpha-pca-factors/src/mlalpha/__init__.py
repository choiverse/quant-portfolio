"""mlalpha — machine-learned cross-sectional alpha, and a PCA factor model to judge it against.

A dependency-light library (numpy/scipy/pandas/matplotlib only) for the
question "can a machine-learning model find tradable cross-sectional alpha in
S&P 500 daily data — and if it appears to, is it alpha or is it a factor
exposure?". The learners and the factor model are built rather than imported,
so the split search, the ridge filter, the eigenvalue bulk and the purging
rules are all in this package and all checked against known answers by
``scripts/validate.py``.

Public API
----------
- data:        load_panel, to_returns, forward_return, stack_panel
- features:    build_features, build_target, the cross-sectional transforms
- pca:         fit_pca, fit_significant, mp_edges, rolling_residuals, PCAModel
- models:      RidgeRegression, DecisionTreeRegressor, GradientBoostingRegressor
- crossval:    purged_walk_forward, overlap_violations
- pipeline:    run_walk_forward, BestFeatureModel, WalkForwardResult
- signals:     long_short_weights, overlapping, signal_to_book, neutralize
- backtest:    Backtester, BacktestResult, cost_sweep_from
- diagnostics: rank_ic, ic_summary, ic_decay, quantile_returns
- attribution: factor_attribution, decompose, newey_west_ols
- metrics:     performance_summary and the individual metric functions
- validation:  the correctness gates, shared by the script and the write-up
- plotting:    the report figures
- style:       shared matplotlib theme and validated palette
"""

from . import (
    attribution,
    backtest,
    crossval,
    data,
    diagnostics,
    features,
    metrics,
    models,
    pca,
    pipeline,
    signals,
    style,
    validation,
)
from .backtest import Backtester, BacktestResult
from .metrics import performance_summary
from .models import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    QuantileBinner,
    RidgeRegression,
)
from .pca import PCAModel
from .pipeline import BestFeatureModel, WalkForwardResult

__all__ = [
    "attribution",
    "backtest",
    "crossval",
    "data",
    "diagnostics",
    "features",
    "metrics",
    "models",
    "pca",
    "pipeline",
    "signals",
    "style",
    "validation",
    "Backtester",
    "BacktestResult",
    "BestFeatureModel",
    "DecisionTreeRegressor",
    "GradientBoostingRegressor",
    "PCAModel",
    "QuantileBinner",
    "RidgeRegression",
    "WalkForwardResult",
    "performance_summary",
]

__version__ = "0.1.0"
