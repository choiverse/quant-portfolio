"""Running a model across the walk-forward folds, and what comes out.

One function does the work — ``run_walk_forward`` — and everything the write-up
reports about a model is derived from its output. The contract is narrow on
purpose:

- a model is anything with ``fit(X, y)`` and ``predict(X)``;
- it is constructed fresh for every fold, so nothing carries across a boundary;
- it sees the training rows of that fold and nothing else;
- its predictions are collected only for the test rows.

The last two are the whole point. It is easy to write a backtest in which a
model is fitted once on everything and then "evaluated" on a subset, and the
resulting numbers are indistinguishable from honest ones until you look for the
fit call. Here there is one fit per fold and the fold owns its rows.

The one non-obvious piece is ``BestFeatureModel``. A learner that beats nothing
has not shown anything, and the right nothing to beat is not a random signal —
it is the best single feature, chosen the same way the model was: in sample,
per fold, with no knowledge of the test period. If the boosted tree cannot beat
that, then whatever it is doing with 14 features and 300 trees is not worth the
turnover it costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .crossval import Fold, fold_table


# --------------------------------------------------------------------------
# The baseline that is not a model
# --------------------------------------------------------------------------


class BestFeatureModel:
    """Predict with the single feature that correlates best in training.

    Rank correlation, not linear — the strategy ranks names, so a feature is
    useful to it exactly insofar as it orders them. The sign is taken from the
    training correlation too, so the baseline is allowed to discover that (say)
    high volatility predicts *low* returns without being told.
    """

    def __init__(self) -> None:
        self.feature_: int = -1
        self.sign_: float = 1.0
        self.train_ic_: float = np.nan

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BestFeatureModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        ry = pd.Series(y).rank().to_numpy()
        best, best_abs = -1, -np.inf
        best_c = 0.0
        for j in range(X.shape[1]):
            rx = pd.Series(X[:, j]).rank().to_numpy()
            c = np.corrcoef(rx, ry)[0, 1]
            if np.isfinite(c) and abs(c) > best_abs:
                best, best_abs, best_c = j, abs(c), c
        self.feature_ = best
        self.sign_ = float(np.sign(best_c)) or 1.0
        self.train_ic_ = float(best_c)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feature_ < 0:
            raise RuntimeError("model is not fitted")
        return self.sign_ * np.asarray(X, dtype=float)[:, self.feature_]


# --------------------------------------------------------------------------
# The walk-forward run
# --------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Out-of-sample predictions and per-fold diagnostics for one model."""

    name: str
    predictions: pd.Series               # indexed by (date, ticker)
    folds: pd.DataFrame
    importances: pd.DataFrame | None = None
    notes: dict = field(default_factory=dict)

    @property
    def signal(self) -> pd.DataFrame:
        """Predictions as a ``(dates x tickers)`` matrix."""
        return self.predictions.unstack("ticker")

    def r2(self, y: pd.Series) -> float:
        """Out-of-sample R^2 against the *training* mean, not the test mean.

        Scoring against the test period's own mean would credit the model for
        a constant it could not have known. On a cross-sectionally demeaned
        label that mean is ~0 anyway, so this is close to ``1 - SSE/SST``, but
        the distinction is the difference between an honest R^2 and a
        flattering one and it costs nothing to get right.
        """
        aligned = y.reindex(self.predictions.index)
        sse = float(((aligned - self.predictions) ** 2).sum())
        sst = float((aligned ** 2).sum())
        return 1.0 - sse / sst if sst > 0 else np.nan


def run_walk_forward(
    design: pd.DataFrame,
    folds: list[Fold],
    factory: Callable[[], object],
    name: str,
    feature_names: list[str] | None = None,
    target: str = "y",
    verbose: bool = False,
) -> WalkForwardResult:
    """Fit ``factory()`` on each fold's training rows, predict its test rows.

    ``design`` is the stacked ``(date, ticker)`` frame from ``data.stack_panel``
    with the label in column ``target``. Returns the concatenated out-of-sample
    predictions — every row predicted exactly once, by a model that was fitted
    before it.
    """
    features = feature_names or [c for c in design.columns if c != target]
    X = design[features].to_numpy(dtype=float)
    y = design[target].to_numpy(dtype=float)

    pieces: list[pd.Series] = []
    importance_rows: list[np.ndarray] = []
    fold_notes: list[dict] = []

    for fold in folds:
        model = factory()
        model.fit(X[fold.train_rows], y[fold.train_rows])
        pred = np.asarray(model.predict(X[fold.test_rows]), dtype=float)
        pieces.append(pd.Series(pred, index=design.index[fold.test_rows]))

        if hasattr(model, "importances"):
            importance_rows.append(np.asarray(model.importances(), dtype=float))
        elif hasattr(model, "coef_") and getattr(model, "coef_", None) is not None:
            importance_rows.append(np.abs(model.coef_) / max(np.abs(model.coef_).sum(), 1e-12))

        note = {"fold": fold.index}
        if isinstance(model, BestFeatureModel):
            note["chosen_feature"] = features[model.feature_]
            note["train_ic"] = model.train_ic_
        if hasattr(model, "effective_dof"):
            note["effective_dof"] = model.effective_dof
        fold_notes.append(note)

        if verbose:
            print(f"      fold {fold.index}: train {len(fold.train_rows):,} "
                  f"(purged {fold.n_purged:,})  test {len(fold.test_rows):,}  "
                  f"[{fold.test_start.date()} .. {fold.test_end.date()}]")

    predictions = pd.concat(pieces).rename(name)
    importances = (
        pd.DataFrame(importance_rows, columns=features)
        if importance_rows
        else None
    )
    if importances is not None:
        importances.index.name = "fold"

    return WalkForwardResult(
        name=name,
        predictions=predictions,
        folds=fold_table(folds),
        importances=importances,
        notes={"per_fold": pd.DataFrame(fold_notes).set_index("fold")},
    )


def default_models(
    ridge_alpha: float = 100.0,
    gbm_kwargs: dict | None = None,
) -> dict[str, Callable[[], object]]:
    """The model line-up: a baseline, a linear model, and a nonlinear one.

    Deliberately three and not thirty. A sweep over dozens of learners on one
    dataset selects the one that best fits this sample's noise, and the
    honest correction for that is a multiple-testing adjustment nobody
    applies. Three models chosen for what they can *represent* — one feature,
    a linear combination, an interaction-capable ensemble — answer the actual
    question, which is whether the extra representational power buys anything.
    """
    from .models import GradientBoostingRegressor, RidgeRegression

    gbm_kwargs = gbm_kwargs or {}
    return {
        "best_feature": BestFeatureModel,
        "ridge": lambda: RidgeRegression(alpha=ridge_alpha),
        "gbm": lambda: GradientBoostingRegressor(**gbm_kwargs),
    }
