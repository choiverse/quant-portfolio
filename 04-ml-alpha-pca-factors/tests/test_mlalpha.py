"""Correctness tests for mlalpha.

The learners and the factor model in this package are written from scratch, so
the tests are not "does it run" but "does it give the answer that is already
known": does a ridge shrink by the factor the algebra says, does a tree find
the split brute force finds, does a PCA return the eigenvalues that were put
in, does a purged fold exclude the rows whose labels overlap the test period,
does a position decided on day t earn day t+1's return and not day t's.

The leakage tests are the ones worth reading. Every one of them takes the form
"change the future, and nothing about the past may move" — because that is the
only property that distinguishes an honest backtest from a flattering one, and
it cannot be checked by looking at the results.

Anything requiring a long Monte Carlo lives in ``scripts/validate.py`` instead,
so this suite stays fast enough to run on every edit.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlalpha import (  # noqa: E402
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
    validation,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel():
    """A synthetic OHLCV panel with factor structure and no predictability."""
    return validation.synthetic_panel(n_days=700, n_names=30, seed=3)


@pytest.fixture(scope="module")
def predictable_panel():
    """The same, with a reversal effect planted in it."""
    return validation.synthetic_panel(n_days=700, n_names=30, seed=4,
                                      alpha_strength=0.9)


@pytest.fixture(scope="module")
def feats(panel):
    return features.build_features(panel)


@pytest.fixture(scope="module")
def design(panel, feats):
    target = features.build_target(panel["close"], horizon=5)
    return data.stack_panel(feats, target)


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 5))
    beta = np.array([1.0, -2.0, 0.5, 3.0, -0.25])
    y = 0.75 + x @ beta + rng.standard_normal(600) * 0.3
    return x, y, beta


# --------------------------------------------------------------------------
# Ridge — against closed forms, not against itself
# --------------------------------------------------------------------------


def test_ridge_matches_normal_equations(linear_data):
    x, y, _ = linear_data
    fit = models.RidgeRegression(alpha=2.5, standardize=False).fit(x, y)

    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    expected = np.linalg.solve(xc.T @ xc + 2.5 * np.eye(x.shape[1]), xc.T @ yc)
    assert np.allclose(fit.coef_, expected, atol=1e-10)


def test_ridge_at_zero_is_ols(linear_data):
    x, y, _ = linear_data
    fit = models.RidgeRegression(alpha=0.0).fit(x, y)
    lstsq = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    assert np.allclose(fit.intercept_, lstsq[0], atol=1e-10)
    assert np.allclose(fit.coef_, lstsq[1:], atol=1e-10)


def test_ridge_recovers_known_coefficients(linear_data):
    x, y, beta = linear_data
    fit = models.RidgeRegression(alpha=1.0).fit(x, y)
    assert np.allclose(fit.coef_, beta, atol=0.05)


def test_ridge_shrinks_monotonically(linear_data):
    x, y, _ = linear_data
    norms = [
        np.linalg.norm(models.RidgeRegression(alpha=a).fit(x, y).coef_)
        for a in (0.0, 1.0, 10.0, 100.0, 1e5)
    ]
    assert all(a > b for a, b in zip(norms, norms[1:]))


def test_effective_dof_spans_zero_to_p(linear_data):
    x, y, _ = linear_data
    assert models.RidgeRegression(0.0).fit(x, y).effective_dof == pytest.approx(5.0)
    assert models.RidgeRegression(1e12).fit(x, y).effective_dof < 1e-3


def test_ridge_is_scale_invariant_when_standardized(linear_data):
    """Rescaling a column must not change the fitted values."""
    x, y, _ = linear_data
    base = models.RidgeRegression(alpha=5.0).fit(x, y).predict(x)
    scaled = x.copy()
    scaled[:, 2] *= 1000.0
    other = models.RidgeRegression(alpha=5.0).fit(scaled, y).predict(scaled)
    assert np.allclose(base, other, atol=1e-8)


def test_ridge_rejects_negative_alpha():
    with pytest.raises(ValueError):
        models.RidgeRegression(alpha=-1.0)


# --------------------------------------------------------------------------
# Binning and trees
# --------------------------------------------------------------------------


def test_binner_is_monotone_in_the_feature():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 2))
    binner = models.QuantileBinner(32)
    b = binner.fit_transform(x)
    order = np.argsort(x[:, 0])
    assert np.all(np.diff(b[order, 0]) >= 0)


def test_binner_threshold_matches_the_split_it_encodes():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((400, 1))
    binner = models.QuantileBinner(16)
    b = binner.fit_transform(x)
    for k in range(len(binner.edges_[0])):
        cut = binner.threshold(0, k)
        assert np.array_equal(b[:, 0] <= k, x[:, 0] <= cut)


def test_tree_split_matches_brute_force():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((400, 3))
    y = 3.0 * (x[:, 2] > -0.4) + rng.standard_normal(400) * 0.5
    binner = models.QuantileBinner(64)
    b = binner.fit_transform(x)
    tree = models.DecisionTreeRegressor(max_depth=1, min_samples_leaf=10)
    tree.fit(b, y, binner.bin_counts)

    best = validation._brute_force_gain(x, y, 10, thresholds=binner.edges_)
    assert tree.root_.gain == pytest.approx(best, rel=1e-10)
    assert tree.root_.feature == 2


def test_tree_leaf_value_is_the_group_mean():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((300, 2))
    y = rng.standard_normal(300)
    binner = models.QuantileBinner(16)
    b = binner.fit_transform(x)
    tree = models.DecisionTreeRegressor(max_depth=2, min_samples_leaf=10).fit(
        b, y, binner.bin_counts
    )
    pred = tree.predict(b)
    for value in np.unique(pred):
        group = y[pred == value]
        assert group.mean() == pytest.approx(value, abs=1e-12)


def test_tree_respects_min_samples_leaf():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((300, 2))
    y = rng.standard_normal(300)
    binner = models.QuantileBinner(64)
    b = binner.fit_transform(x)
    tree = models.DecisionTreeRegressor(max_depth=4, min_samples_leaf=40).fit(
        b, y, binner.bin_counts
    )
    counts = pd.Series(tree.predict(b)).value_counts()
    assert counts.min() >= 40


def test_tree_depth_zero_predicts_the_mean():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((200, 2))
    y = rng.standard_normal(200)
    binner = models.QuantileBinner(8)
    b = binner.fit_transform(x)
    tree = models.DecisionTreeRegressor(max_depth=0).fit(b, y, binner.bin_counts)
    assert np.allclose(tree.predict(b), y.mean())


# --------------------------------------------------------------------------
# Gradient boosting
# --------------------------------------------------------------------------


def test_boosting_reduces_training_loss_monotonically():
    """Under squared error each stage fits the residual, so loss cannot rise."""
    rng = np.random.default_rng(9)
    x = rng.uniform(size=(800, 4))
    y = np.sin(3 * x[:, 0]) + x[:, 1] ** 2 + rng.standard_normal(800) * 0.1
    gbm = models.GradientBoostingRegressor(
        n_estimators=40, learning_rate=0.1, max_depth=2,
        min_samples_leaf=10, subsample=1.0, seed=1,
    ).fit(x, y)
    losses = np.array(gbm.train_loss_)
    assert np.all(np.diff(losses) <= 1e-12)


def test_boosting_beats_ridge_on_an_interaction():
    rng = np.random.default_rng(10)
    x_tr = rng.uniform(-1, 1, (2_000, 2))
    y_tr = x_tr[:, 0] * x_tr[:, 1] + rng.standard_normal(2_000) * 0.05
    x_te = rng.uniform(-1, 1, (2_000, 2))
    y_te = x_te[:, 0] * x_te[:, 1]

    gbm = models.GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.1, max_depth=3,
        min_samples_leaf=10, subsample=1.0, seed=2,
    ).fit(x_tr, y_tr)
    ridge = models.RidgeRegression(alpha=1.0).fit(x_tr, y_tr)

    mse_gbm = float(np.mean((y_te - gbm.predict(x_te)) ** 2))
    mse_ridge = float(np.mean((y_te - ridge.predict(x_te)) ** 2))
    assert mse_gbm < 0.3 * mse_ridge


def test_boosting_is_deterministic_given_a_seed():
    rng = np.random.default_rng(11)
    x = rng.uniform(size=(400, 3))
    y = x[:, 0] + rng.standard_normal(400) * 0.1
    kw = dict(n_estimators=20, learning_rate=0.1, max_depth=2,
              min_samples_leaf=10, subsample=0.7, seed=42)
    a = models.GradientBoostingRegressor(**kw).fit(x, y).predict(x)
    b = models.GradientBoostingRegressor(**kw).fit(x, y).predict(x)
    assert np.array_equal(a, b)


def test_boosting_ignores_pure_noise_features():
    rng = np.random.default_rng(12)
    x = rng.uniform(size=(2_000, 6))
    y = 5 * x[:, 0] + rng.standard_normal(2_000) * 0.2
    gbm = models.GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=2,
        min_samples_leaf=20, subsample=1.0, seed=3,
    ).fit(x, y)
    imp = gbm.importances()
    assert imp[0] > 0.8
    assert imp[1:].sum() < 0.2


def test_staged_predict_ends_at_the_full_prediction():
    rng = np.random.default_rng(13)
    x = rng.uniform(size=(300, 3))
    y = x[:, 1] + rng.standard_normal(300) * 0.1
    gbm = models.GradientBoostingRegressor(
        n_estimators=15, learning_rate=0.1, max_depth=2,
        min_samples_leaf=10, subsample=1.0, seed=4,
    ).fit(x, y)
    n, last = list(gbm.staged_predict(x))[-1]
    assert n == 15
    assert np.allclose(last, gbm.predict(x))


# --------------------------------------------------------------------------
# PCA and the Marchenko-Pastur law
# --------------------------------------------------------------------------


def test_loadings_are_orthonormal(panel):
    rets = data.to_returns(panel["close"])
    model = pca.fit_pca(rets, n_components=5)
    gram = model.loadings.T @ model.loadings
    assert np.allclose(gram, np.eye(5), atol=1e-10)


def test_eigenvalues_of_a_correlation_matrix_sum_to_n(panel):
    rets = data.to_returns(panel["close"])
    model = pca.fit_pca(rets)
    assert model.eigenvalues.sum() == pytest.approx(rets.shape[1], rel=1e-10)


def test_full_reconstruction_is_exact(panel):
    rets = data.to_returns(panel["close"])
    model = pca.fit_pca(rets, n_components=rets.shape[1])
    assert np.allclose(model.reconstruct(rets).to_numpy(), rets.to_numpy(), atol=1e-10)


def test_residual_plus_reconstruction_is_the_original(panel):
    rets = data.to_returns(panel["close"])
    model = pca.fit_pca(rets, n_components=3)
    total = model.residuals(rets) + model.reconstruct(rets)
    assert np.allclose(total.to_numpy(), rets.to_numpy(), atol=1e-12)


def test_mp_edges_match_the_closed_form():
    lo, hi = pca.mp_edges(1_000, 250)
    q = 0.25
    assert lo == pytest.approx((1 - np.sqrt(q)) ** 2)
    assert hi == pytest.approx((1 + np.sqrt(q)) ** 2)


def test_mp_density_integrates_to_one():
    grid = np.linspace(0, 4, 40_001)
    dens = pca.mp_density(grid, 1_000, 300)
    assert np.trapezoid(dens, grid) == pytest.approx(1.0, abs=2e-3)


def test_no_factors_found_in_pure_noise():
    """Across several noise panels the method must find essentially nothing.

    Not "exactly zero on every panel": the largest eigenvalue of a finite
    random matrix fluctuates around the edge with Tracy-Widom tails, so a
    single panel crossing it by a hair is expected behaviour rather than a
    false factor. What must not happen is a *systematic* count above zero.
    """
    counts = []
    for seed in range(6):
        rng = np.random.default_rng(14 + seed)
        noise = pd.DataFrame(rng.standard_normal((900, 150)))
        counts.append(pca.fit_pca(noise).n_significant())
    assert max(counts) <= 1
    assert np.mean(counts) < 0.5


def test_noise_variance_is_one_minus_the_market_share():
    rng = np.random.default_rng(15)
    x = pd.DataFrame(rng.standard_normal((600, 80)))
    model = pca.fit_pca(x)
    expected = 1.0 - model.eigenvalues[0] / 80
    assert model.noise_variance() == pytest.approx(expected, rel=1e-10)


def test_sign_convention_is_stable_across_refits(panel):
    """PC1 must not flip sign when one row is appended."""
    rets = data.to_returns(panel["close"])
    a = pca.fit_pca(rets.iloc[:-1], n_components=3).loadings[:, 0]
    b = pca.fit_pca(rets, n_components=3).loadings[:, 0]
    assert np.corrcoef(a, b)[0, 1] > 0.9


def test_fit_pca_rejects_missing_values(panel):
    rets = data.to_returns(panel["close"]).copy()
    rets.iloc[3, 2] = np.nan
    with pytest.raises(ValueError):
        pca.fit_pca(rets)


def test_rolling_residuals_are_causal(panel):
    """Appending a future row cannot change any earlier residual."""
    rets = data.to_returns(panel["close"])
    a, _ = pca.rolling_residuals(rets.iloc[:-30], window=252, step=21, n_components=3)
    b, _ = pca.rolling_residuals(rets, window=252, step=21, n_components=3)
    common = a.index[:-1]
    # The final partial block of `a` is refitted differently once more data
    # arrives, so compare only the blocks that were complete in both runs.
    n = (len(common) // 21) * 21
    assert np.allclose(a.iloc[:n].to_numpy(), b.iloc[:n].to_numpy(),
                       equal_nan=True, atol=1e-12)


# --------------------------------------------------------------------------
# Features — all causal, all cross-sectional
# --------------------------------------------------------------------------


def test_cs_rank_is_uniform_within_each_date(feats):
    ranked = feats["mom_12_1"].dropna(how="all")
    row = ranked.iloc[-1].dropna()
    assert row.min() == pytest.approx(-0.5)
    assert row.max() == pytest.approx(0.5)
    assert row.mean() == pytest.approx(0.0, abs=1e-12)


def test_cs_zscore_is_standardized_within_each_date():
    rng = np.random.default_rng(16)
    frame = pd.DataFrame(rng.standard_normal((50, 40)) * 5 + 3)
    z = features.cs_zscore(frame)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=1, ddof=1), 1.0, atol=1e-8)


def test_target_is_cross_sectionally_neutral(panel):
    target = features.build_target(panel["close"], horizon=5)
    assert np.allclose(target.mean(axis=1).dropna(), 0.0, atol=1e-14)


def test_forward_return_matches_a_hand_computation(panel):
    close = panel["close"]
    fwd = data.forward_return(close, horizon=5)
    expected = close.iloc[105, 0] / close.iloc[100, 0] - 1.0
    assert fwd.iloc[100, 0] == pytest.approx(expected)


def test_forward_return_leaves_the_tail_missing(panel):
    fwd = data.forward_return(panel["close"], horizon=5)
    assert fwd.iloc[-5:].isna().all().all()


def test_features_use_only_past_data(panel):
    """Rewriting the last bar cannot change any earlier feature value."""
    tampered = {k: v.copy() for k, v in panel.items()}
    for key in tampered:
        tampered[key].iloc[-1] *= 1.5

    base = features.build_features(panel)
    after = features.build_features(tampered)
    for name in base:
        a = base[name].iloc[:-1].to_numpy()
        b = after[name].iloc[:-1].to_numpy()
        assert np.allclose(a, b, equal_nan=True, atol=1e-12), name


def test_momentum_skips_the_recent_month(panel):
    close = panel["close"]
    mom = features.momentum(close, lookback=252, skip=21)
    expected = close.iloc[300 - 21, 0] / close.iloc[300 - 252, 0] - 1.0
    assert mom.iloc[300, 0] == pytest.approx(expected)


def test_rolling_beta_recovers_a_known_beta():
    rng = np.random.default_rng(17)
    idx = pd.bdate_range("2015-01-01", periods=800)
    mkt = pd.Series(rng.standard_normal(800) * 0.01, index=idx)
    rets = pd.DataFrame({"A": 1.8 * mkt + rng.standard_normal(800) * 0.001}, index=idx)
    beta, idio = features.rolling_beta_idio(rets, mkt, window=252)
    assert beta["A"].dropna().mean() == pytest.approx(1.8, abs=0.03)
    assert idio["A"].dropna().mean() < 0.1


def test_winsorize_clips_to_the_stated_quantiles():
    rng = np.random.default_rng(18)
    frame = pd.DataFrame(rng.standard_normal((20, 200)))
    w = features.winsorize(frame, limit=0.05)
    assert (w.max(axis=1) <= frame.quantile(0.95, axis=1) + 1e-12).all()
    assert (w.min(axis=1) >= frame.quantile(0.05, axis=1) - 1e-12).all()


# --------------------------------------------------------------------------
# The design matrix
# --------------------------------------------------------------------------


def test_stack_panel_preserves_values(panel, feats):
    target = features.build_target(panel["close"], horizon=5)
    design = data.stack_panel(feats, target)
    date, ticker = design.index[5_000]
    assert design.loc[(date, ticker), "mom_12_1"] == pytest.approx(
        feats["mom_12_1"].loc[date, ticker]
    )
    assert design.loc[(date, ticker), "y"] == pytest.approx(target.loc[date, ticker])


def test_stack_panel_drops_incomplete_rows(design):
    assert not design.isna().to_numpy().any()


def test_stack_panel_needs_features():
    with pytest.raises(ValueError):
        data.stack_panel({})


# --------------------------------------------------------------------------
# Purged walk-forward
# --------------------------------------------------------------------------


def test_purging_removes_exactly_the_overlapping_dates():
    dates = pd.bdate_range("2015-01-01", periods=400)
    rows = pd.DatetimeIndex(np.repeat(dates.to_numpy(), 5))
    folds = crossval.purged_walk_forward(rows, horizon=5, initial_train=200,
                                         test_size=50, embargo=3)
    # horizon 5 + embargo 3 = 8 dates x 5 tickers dropped from each training set
    assert all(f.n_purged == 8 * 5 for f in folds)


def test_no_fold_leaks_a_label(design):
    dates = design.index.get_level_values(0)
    folds = crossval.purged_walk_forward(dates, horizon=5, initial_train=200,
                                         test_size=60, embargo=5)
    assert folds
    for fold in folds:
        assert crossval.overlap_violations(dates, fold, 5) == 0


def test_train_always_precedes_test(design):
    dates = design.index.get_level_values(0)
    folds = crossval.purged_walk_forward(dates, horizon=5, initial_train=200,
                                         test_size=60, embargo=5)
    for fold in folds:
        assert dates[fold.train_rows].max() < dates[fold.test_rows].min()


def test_test_folds_tile_without_overlapping(design):
    dates = design.index.get_level_values(0)
    folds = crossval.purged_walk_forward(dates, horizon=5, initial_train=200,
                                         test_size=60, embargo=5)
    seen = np.concatenate([f.test_rows for f in folds])
    assert len(seen) == len(np.unique(seen))


def test_rolling_window_does_not_grow(design):
    dates = design.index.get_level_values(0)
    folds = crossval.purged_walk_forward(dates, horizon=5, initial_train=200,
                                         test_size=60, embargo=5, expanding=False)
    sizes = [len(f.train_rows) for f in folds]
    assert max(sizes) - min(sizes) < 0.05 * max(sizes)


def test_expanding_window_grows(design):
    dates = design.index.get_level_values(0)
    folds = crossval.purged_walk_forward(dates, horizon=5, initial_train=200,
                                         test_size=60, embargo=5, expanding=True)
    sizes = [len(f.train_rows) for f in folds]
    assert all(a < b for a, b in zip(sizes, sizes[1:]))


def test_purged_walk_forward_validates_arguments():
    dates = pd.bdate_range("2015-01-01", periods=100)
    with pytest.raises(ValueError):
        crossval.purged_walk_forward(dates, horizon=0)
    with pytest.raises(ValueError):
        crossval.purged_walk_forward(dates, horizon=5, embargo=-1)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def test_every_test_row_is_predicted_exactly_once(design):
    dates = design.index.get_level_values(0)
    folds = crossval.purged_walk_forward(dates, horizon=5, initial_train=200,
                                         test_size=60, embargo=5)
    result = pipeline.run_walk_forward(
        design, folds, lambda: models.RidgeRegression(10.0), "ridge"
    )
    expected = sum(len(f.test_rows) for f in folds)
    assert len(result.predictions) == expected
    assert not result.predictions.index.duplicated().any()


def test_best_feature_model_picks_the_planted_feature():
    rng = np.random.default_rng(19)
    x = rng.standard_normal((2_000, 4))
    y = 2.0 * x[:, 3] + rng.standard_normal(2_000) * 0.1
    m = pipeline.BestFeatureModel().fit(x, y)
    assert m.feature_ == 3
    assert m.sign_ == 1.0


def test_best_feature_model_learns_a_negative_sign():
    rng = np.random.default_rng(20)
    x = rng.standard_normal((2_000, 3))
    y = -3.0 * x[:, 1] + rng.standard_normal(2_000) * 0.1
    m = pipeline.BestFeatureModel().fit(x, y)
    assert m.feature_ == 1
    assert m.sign_ == -1.0
    assert np.corrcoef(m.predict(x), y)[0, 1] > 0.9


def test_pipeline_finds_a_planted_signal(predictable_panel):
    feats = features.build_features(predictable_panel)
    target = features.build_target(predictable_panel["close"], horizon=5)
    design = data.stack_panel(feats, target)
    dates = design.index.get_level_values(0)
    folds = crossval.purged_walk_forward(dates, horizon=5, initial_train=180,
                                         test_size=60, embargo=5, min_test_size=20)
    result = pipeline.run_walk_forward(
        design, folds, lambda: models.RidgeRegression(100.0), "ridge"
    )
    ic = diagnostics.rank_ic(result.signal, target)
    assert ic.mean() > 0.05


def test_unfitted_models_raise():
    with pytest.raises(RuntimeError):
        models.RidgeRegression().predict(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        models.GradientBoostingRegressor().predict(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        pipeline.BestFeatureModel().predict(np.zeros((2, 2)))


# --------------------------------------------------------------------------
# Signals and portfolio construction
# --------------------------------------------------------------------------


def test_long_short_weights_are_dollar_neutral():
    rng = np.random.default_rng(21)
    score = pd.DataFrame(rng.standard_normal((50, 100)))
    w = signals.long_short_weights(score, quantile=0.2)
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(w.abs().sum(axis=1), 1.0, atol=1e-12)


def test_long_short_weights_go_long_the_top_names():
    score = pd.DataFrame(
        [np.arange(100.0)], index=pd.bdate_range("2020-01-01", periods=1)
    )
    w = signals.long_short_weights(score, quantile=0.2)
    assert (w.iloc[0, 80:] > 0).all()
    assert (w.iloc[0, :20] < 0).all()
    assert (w.iloc[0, 20:80] == 0).all()


def test_thin_cross_sections_stay_flat():
    """Only the thin dates go flat, and the others are untouched.

    Written with a mix of thin and wide rows on purpose. A single-row frame
    passes this whether the ``min_names`` mask is aligned on the index or
    silently broadcast across the columns, so it would not catch the
    alignment bug it is meant to guard against.
    """
    rng = np.random.default_rng(39)
    score = pd.DataFrame(rng.standard_normal((3, 40)))
    score.iloc[1, :35] = np.nan          # only 5 valid names on the middle date
    w = signals.long_short_weights(score, min_names=20)
    gross = w.abs().sum(axis=1)
    assert gross.iloc[0] == pytest.approx(1.0)
    assert gross.iloc[1] == pytest.approx(0.0)
    assert gross.iloc[2] == pytest.approx(1.0)


def test_overlapping_reduces_turnover():
    rng = np.random.default_rng(22)
    score = pd.DataFrame(rng.standard_normal((300, 80)),
                         index=pd.bdate_range("2018-01-01", periods=300))
    raw = signals.long_short_weights(score)
    staggered = signals.overlapping(raw, 5)
    t_raw = raw.diff().abs().sum(axis=1).mean()
    t_stag = staggered.diff().abs().sum(axis=1).mean()
    assert t_stag < 0.5 * t_raw


def test_overlapping_with_horizon_one_is_a_no_op():
    rng = np.random.default_rng(23)
    w = pd.DataFrame(rng.standard_normal((20, 5)))
    assert signals.overlapping(w, 1) is w


def test_neutralize_removes_the_exposure():
    rng = np.random.default_rng(24)
    tickers = [f"T{i}" for i in range(60)]
    exposures = pd.DataFrame(rng.standard_normal((60, 3)), index=tickers)
    score = pd.DataFrame(rng.standard_normal((30, 60)), columns=tickers)
    w = signals.long_short_weights(score)
    neutral = signals.neutralize(w, exposures)
    loadings = neutral.to_numpy() @ exposures.to_numpy()
    assert np.abs(loadings).max() < 1e-10


def test_neutralize_preserves_gross_exposure():
    rng = np.random.default_rng(25)
    tickers = [f"T{i}" for i in range(60)]
    exposures = pd.DataFrame(rng.standard_normal((60, 2)), index=tickers)
    score = pd.DataFrame(rng.standard_normal((20, 60)), columns=tickers)
    w = signals.long_short_weights(score)
    neutral = signals.neutralize(w, exposures)
    assert np.allclose(neutral.abs().sum(axis=1), w.abs().sum(axis=1), atol=1e-10)


# --------------------------------------------------------------------------
# The backtester
# --------------------------------------------------------------------------


def test_position_earns_the_next_days_return():
    idx = pd.bdate_range("2020-01-01", periods=4)
    rets = pd.DataFrame({"A": [0.10, 0.20, 0.30, 0.40]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0]}, index=idx)
    run = backtest.Backtester(rets, cost_bps=0.0).run(weights)
    # decided at the close of day 0 -> earns day 1's 20%, not day 0's 10%
    assert run.gross_returns.iloc[0] == pytest.approx(0.20)


def test_costs_are_charged_on_the_day_the_trade_lands():
    idx = pd.bdate_range("2020-01-01", periods=4)
    rets = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0]}, index=idx)
    run = backtest.Backtester(rets, cost_bps=100.0).run(weights)
    assert run.turnover.iloc[0] == pytest.approx(1.0)      # opening the position
    assert run.turnover.iloc[1] == pytest.approx(0.0)      # holding it
    assert run.turnover.iloc[2] == pytest.approx(1.0)      # closing it
    assert run.returns.iloc[0] == pytest.approx(-0.01)


def test_zero_cost_run_equals_gross():
    rng = np.random.default_rng(26)
    idx = pd.bdate_range("2020-01-01", periods=200)
    rets = pd.DataFrame(rng.standard_normal((200, 10)) * 0.01, index=idx)
    w = signals.long_short_weights(pd.DataFrame(rng.standard_normal((200, 10)),
                                                index=idx), min_names=5)
    run = backtest.Backtester(rets, cost_bps=0.0).run(w)
    assert np.allclose(run.returns, run.gross_returns)


def test_breakeven_cost_zeroes_the_strategy():
    rng = np.random.default_rng(27)
    idx = pd.bdate_range("2020-01-01", periods=300)
    rets = pd.DataFrame(rng.standard_normal((300, 20)) * 0.01, index=idx)
    w = signals.long_short_weights(pd.DataFrame(rng.standard_normal((300, 20)),
                                                index=idx), min_names=10)
    run = backtest.Backtester(rets, cost_bps=10.0).run(w)
    be = run.breakeven_cost_bps()
    net = run.gross_returns - run.turnover * (be / 1e4)
    assert net.mean() == pytest.approx(0.0, abs=1e-15)


def test_cost_sweep_is_monotone_in_cost():
    rng = np.random.default_rng(28)
    idx = pd.bdate_range("2020-01-01", periods=300)
    gross = pd.Series(rng.standard_normal(300) * 0.005 + 0.0004, index=idx)
    turnover = pd.Series(np.abs(rng.standard_normal(300)) * 0.1, index=idx)
    sweep = backtest.cost_sweep_from(gross, turnover, [0, 5, 10, 20])
    assert sweep["Sharpe"].is_monotonic_decreasing


def test_backtest_ignores_a_future_return_shock():
    """The classic look-ahead check, at the engine level."""
    rng = np.random.default_rng(29)
    idx = pd.bdate_range("2020-01-01", periods=100)
    rets = pd.DataFrame(rng.standard_normal((100, 8)) * 0.01, index=idx)
    w = signals.long_short_weights(pd.DataFrame(rng.standard_normal((100, 8)),
                                                index=idx), min_names=4)
    base = backtest.Backtester(rets, cost_bps=10.0).run(w)

    shocked = rets.copy()
    shocked.iloc[-1] *= 10.0
    after = backtest.Backtester(shocked, cost_bps=10.0).run(w)
    assert np.allclose(base.returns.iloc[:-1], after.returns.iloc[:-1])


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def test_rank_ic_is_one_for_a_perfect_signal():
    rng = np.random.default_rng(30)
    fwd = pd.DataFrame(rng.standard_normal((40, 60)))
    ic = diagnostics.rank_ic(fwd, fwd)
    assert np.allclose(ic.dropna(), 1.0)


def test_rank_ic_is_minus_one_for_an_inverted_signal():
    rng = np.random.default_rng(31)
    fwd = pd.DataFrame(rng.standard_normal((40, 60)))
    ic = diagnostics.rank_ic(-fwd, fwd)
    assert np.allclose(ic.dropna(), -1.0)


def test_rank_ic_matches_scipy_spearman():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(32)
    sig = pd.DataFrame(rng.standard_normal((5, 50)))
    fwd = pd.DataFrame(rng.standard_normal((5, 50)))
    ours = diagnostics.rank_ic(sig, fwd)
    for i in range(5):
        expected = scipy_stats.spearmanr(sig.iloc[i], fwd.iloc[i]).statistic
        assert ours.iloc[i] == pytest.approx(expected)


def test_rank_ic_skips_thin_cross_sections():
    sig = pd.DataFrame(np.arange(6.0).reshape(1, 6))
    fwd = pd.DataFrame(np.arange(6.0).reshape(1, 6))
    assert diagnostics.rank_ic(sig, fwd).isna().all()


def test_ic_summary_adjusts_for_overlap():
    rng = np.random.default_rng(33)
    ic = pd.Series(rng.standard_normal(500) * 0.1 + 0.01)
    s = diagnostics.ic_summary(ic, horizon=5)
    assert s["t-stat (/sqrt 5)"] == pytest.approx(
        s["t-stat (naive)"] / np.sqrt(5)
    )


def test_quantile_returns_are_ordered_for_a_perfect_signal():
    rng = np.random.default_rng(34)
    fwd = pd.DataFrame(rng.standard_normal((60, 100)))
    q = diagnostics.quantile_returns(fwd, fwd, n_quantiles=5).mean()
    assert list(q.index[:5]) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert q["Q1"] < q["Q2"] < q["Q3"] < q["Q4"] < q["Q5"]
    assert q["Q5-Q1"] == pytest.approx(q["Q5"] - q["Q1"])


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


def test_newey_west_matches_ols_at_zero_lags():
    rng = np.random.default_rng(35)
    x = rng.standard_normal((400, 2))
    y = 0.5 + x @ np.array([1.5, -0.8]) + rng.standard_normal(400)
    res = attribution.newey_west_ols(y, x, lags=0)

    xd = np.column_stack([np.ones(400), x])
    beta = np.linalg.lstsq(xd, y, rcond=None)[0]
    assert np.allclose(res.params, beta, atol=1e-10)


def test_newey_west_widens_errors_under_autocorrelation():
    rng = np.random.default_rng(36)
    e = rng.standard_normal(1_000)
    for t in range(1, 1_000):
        e[t] = 0.8 * e[t - 1] + e[t]          # heavily autocorrelated residuals
    x = rng.standard_normal((1_000, 1))
    y = 0.3 * x[:, 0] + e
    plain = attribution.newey_west_ols(y, x, lags=0)
    robust = attribution.newey_west_ols(y, x, lags=20)
    assert robust.stderr[0] > 1.5 * plain.stderr[0]


def test_decomposition_adds_up():
    rng = np.random.default_rng(37)
    idx = pd.bdate_range("2019-01-01", periods=500)
    factors = pd.DataFrame(rng.standard_normal((500, 3)) * 0.01,
                           index=idx, columns=["PC1", "PC2", "PC3"])
    strat = (factors @ np.array([0.5, -0.2, 0.1]) + rng.standard_normal(500) * 0.002)
    strat.name = "s"
    d = attribution.decompose(strat, factors)
    assert d["Factor-explained"] + d["Residual alpha"] == pytest.approx(
        d["Total ann. return"], abs=1e-10
    )


def test_a_pure_factor_bet_has_no_alpha():
    rng = np.random.default_rng(38)
    idx = pd.bdate_range("2019-01-01", periods=800)
    factors = pd.DataFrame(rng.standard_normal((800, 2)) * 0.01,
                           index=idx, columns=["PC1", "PC2"])
    strat = (factors["PC1"] * 1.3).rename("s")
    d = attribution.decompose(strat, factors)
    assert d["R-squared"] > 0.999
    assert abs(d["Residual alpha"]) < 1e-8


# --------------------------------------------------------------------------
# Metrics — identical to projects 02 and 03, so the identities must hold
# --------------------------------------------------------------------------


def test_sharpe_of_a_flat_series_is_undefined():
    assert np.isnan(metrics.sharpe_ratio(pd.Series([0.0] * 100)))


def test_equity_curve_compounds():
    r = pd.Series([0.1, -0.1, 0.05])
    eq = metrics.equity_curve(r)
    assert eq.iloc[-1] == pytest.approx(1.1 * 0.9 * 1.05)


def test_max_drawdown_of_a_rising_series_is_zero():
    r = pd.Series([0.01] * 50)
    assert metrics.max_drawdown(r) == pytest.approx(0.0)


def test_cagr_matches_a_hand_computation():
    r = pd.Series([0.001] * 252)
    assert metrics.cagr(r) == pytest.approx(1.001 ** 252 - 1)


# --------------------------------------------------------------------------
# The gates themselves are importable and self-consistent
# --------------------------------------------------------------------------


def test_gate_pass_logic():
    assert validation.Gate("x", 1.0, 2.0, "<", "u").passed
    assert not validation.Gate("x", 3.0, 2.0, "<", "u").passed
    assert validation.Gate("x", 3.0, 2.0, ">", "u").passed


def test_quick_gate_suite_runs_and_passes():
    """The fast settings must still pass — a quick mode that fails is useless."""
    table = validation.validation_gates(quick=True)
    assert len(table) == len(validation.GATES)
    failed = table.index[~table["passed"]].tolist()
    assert not failed, f"failing gates: {failed}"
