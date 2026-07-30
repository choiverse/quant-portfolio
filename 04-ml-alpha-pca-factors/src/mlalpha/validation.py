"""Correctness gates: every estimator, checked against an answer known in advance.

Each gate constructs data whose true answer is fixed by construction — an exact
algebraic identity, a limiting distribution from a theorem, a function with
written-down parameters, a panel with a planted signal — and compares the
implementation's output against it. Nothing here reads market data, so the
whole suite runs in CI without the 28 MB panel and fails the build if the maths
breaks.

Three of the ten are worth flagging because they are doing something the other
seven are not.

**Gate 2 is the load-bearing one.** It generates a panel of pure independent
noise and compares the *whole* empirical eigenvalue distribution against the
Marchenko-Pastur law by a Kolmogorov-Smirnov distance. Every claim in this
project about how many factors are real rests on that law being implemented
correctly; if the edge were mis-scaled, the factor count would be wrong and the
"residual" returns would still contain factor structure.

**Gate 8 carries its own control.** Checking that the purged splits contain
zero label overlaps is only meaningful if the same check *would* have found
overlaps in an unpurged split. The gate therefore reports both, and passes only
when the unpurged count is positive and the purged count is zero. A test that
cannot fail is not a test.

**Gates 9 and 10 are a size/power pair on the same machinery.** Gate 9 shocks
the last bar of a panel and demands that no earlier P&L changes — no future
information can flow backwards. Gate 10 plants a real cross-sectional signal in
a panel and demands that the pipeline finds it. A pipeline that passes only the
first is one that predicts nothing at all, safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import backtest as bt
from . import crossval, data, diagnostics, features, models, pca, pipeline, signals


@dataclass
class Gate:
    """One correctness check."""

    name: str
    statistic: float
    tolerance: float
    comparator: str      # "<" -> pass when statistic < tolerance, ">" -> above
    units: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.statistic < self.tolerance
            if self.comparator == "<"
            else self.statistic > self.tolerance
        )


# --------------------------------------------------------------------------
# Gates 1-3: the factor model
# --------------------------------------------------------------------------


def gate_pca_spectrum(n_obs: int = 400, n_assets: int = 30, seed: int = 11) -> Gate:
    """PCA must return a spectrum that was written down, to machine precision.

    Constructed so the answer is exact rather than asymptotic. Start from a
    matrix ``Z`` with exactly orthonormal columns; its sample covariance is
    exactly ``I``. Then ``W = Z @ C^(1/2)`` has sample covariance exactly
    ``C``, whose eigenvalues are chosen by hand. No Monte Carlo convergence is
    involved, so the tolerance can be 1e-10 and a real error cannot hide
    inside sampling noise.
    """
    rng = np.random.default_rng(seed)
    target = np.linspace(10.0, 0.5, n_assets)

    q, _ = np.linalg.qr(rng.standard_normal((n_assets, n_assets)))
    cov = q @ np.diag(target) @ q.T
    root = q @ np.diag(np.sqrt(target)) @ q.T

    a = rng.standard_normal((n_obs, n_assets))
    a -= a.mean(axis=0)
    z, _ = np.linalg.qr(a)
    z *= np.sqrt(n_obs - 1)          # sample covariance of z is now exactly I

    panel = pd.DataFrame(z @ root, index=pd.RangeIndex(n_obs))
    model = pca.fit_pca(panel, standardize=False)

    worst = float(np.max(np.abs(np.sort(model.eigenvalues)[::-1] - np.sort(target)[::-1])))
    recon = float(np.abs(np.cov(panel.to_numpy(), rowvar=False) - cov).max())
    return Gate(
        name="PCA recovers a constructed covariance spectrum",
        statistic=worst,
        tolerance=1e-10,
        comparator="<",
        units="max abs eigenvalue error",
        detail=f"{n_assets} eigenvalues from {target[0]:.1f} down to {target[-1]:.1f}; "
               f"constructed covariance error {recon:.2e}",
    )


def _mp_cdf(x: np.ndarray, n_obs: int, n_assets: int, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur CDF by fine numerical integration of the density."""
    lo, hi = pca.mp_edges(n_obs, n_assets, sigma2)
    grid = np.linspace(lo, hi, 20_001)
    dens = pca.mp_density(grid, n_obs, n_assets, sigma2)
    cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(np.asarray(x, dtype=float), grid, cdf, left=0.0, right=1.0)


def gate_mp_law(n_obs: int = 1_200, n_assets: int = 300, n_reps: int = 12, seed: int = 22) -> Gate:
    """The eigenvalues of a pure-noise correlation matrix must follow the MP law.

    Not just the edge — the whole distribution, by a Kolmogorov-Smirnov
    distance between the pooled empirical spectrum and the theoretical CDF.
    This is the analogue of project 03's simulated-ADF-critical-values gate:
    the project generates its own null distribution and checks it against the
    published theory, so that a mis-scaled implementation cannot quietly move
    every factor-count decision in the study.
    """
    rng = np.random.default_rng(seed)
    pooled = []
    for _ in range(n_reps):
        x = pd.DataFrame(rng.standard_normal((n_obs, n_assets)))
        pooled.append(pca.fit_pca(x).eigenvalues)
    lam = np.sort(np.concatenate(pooled))

    emp = np.arange(1, len(lam) + 1) / len(lam)
    theo = _mp_cdf(lam, n_obs, n_assets)
    ks = float(np.max(np.abs(emp - theo)))

    _, hi = pca.mp_edges(n_obs, n_assets)
    return Gate(
        name="Eigenvalue distribution vs Marchenko-Pastur",
        statistic=ks,
        tolerance=0.02,
        comparator="<",
        units="KS distance",
        detail=f"{n_reps} panels of {n_obs}x{n_assets}; theoretical edge {hi:.3f}, "
               f"observed max {lam[-1]:.3f}",
    )


def gate_factor_count(n_obs: int = 600, n_assets: int = 150, k_true: int = 3, seed: int = 33) -> Gate:
    """Zero factors on noise, exactly ``k_true`` on a panel built with ``k_true``.

    Reported as the worse of the two errors. The noise half is what stops the
    method inventing structure; the planted half is what stops it being
    conservative to the point of uselessness. Either failure alone would make
    the residual returns wrong, in opposite directions.
    """
    rng = np.random.default_rng(seed)

    noise = pd.DataFrame(rng.standard_normal((n_obs, n_assets)))
    false_positives = pca.fit_pca(noise).n_significant(adjust_noise=True)

    f = rng.standard_normal((n_obs, k_true))
    loadings = rng.uniform(0.4, 1.0, size=(k_true, n_assets)) * rng.choice(
        [-1.0, 1.0], size=(k_true, n_assets), p=[0.2, 0.8]
    )
    planted = pd.DataFrame(f @ loadings + rng.standard_normal((n_obs, n_assets)))
    found = pca.fit_pca(planted).n_significant(adjust_noise=True)

    worst = max(false_positives, abs(found - k_true))
    return Gate(
        name="Factor count: none on noise, k on a planted k-factor panel",
        statistic=float(worst),
        tolerance=1.0,
        comparator="<",
        units="max factor-count error",
        detail=f"noise panel found {false_positives} factors (want 0); "
               f"{k_true}-factor panel found {found}",
    )


def gate_residual_is_neutral(
    n_days: int = 900,
    n_assets: int = 120,
    k: int = 3,
    n_panels: int = 6,
    seed: int = 34,
) -> Gate:
    """Residualization must not manufacture the signal it is meant to isolate.

    The panel is built with a factor structure and **nothing else**: every
    stock has the same expected return, and tomorrow's idiosyncratic return is
    independent of anything observable today. Momentum, reversal and
    volatility features scored against the forward *residual* return must
    therefore all come back with an information coefficient of zero. The gate
    reports the worst of them.

    The drift-free construction is deliberate and it took a wrong version of
    this gate to notice why. Give each stock its own constant drift and
    momentum becomes genuinely predictive — of the total return *and*, more
    strongly, of the residual, because removing the factor variance leaves the
    drift a larger share of what remains. That is a real effect, not an
    artefact, so a null containing it cannot test for artefacts.

    The IC is averaged over ``n_panels`` independent panels because a single
    panel is not precise enough to test against. Across seeds the probe IRs
    scatter with a standard deviation of roughly 0.04 to 0.08 — a one-panel
    gate at a tolerance of 0.15 fails perfectly clean code about one run in
    ten, purely on Monte Carlo error. Averaging six panels cuts that spread by
    ``sqrt(6)`` and makes the tolerance mean the same thing every time.

    This gate exists because the first residualizer failed it at **−1.34**. It
    subtracted each stock's trailing-window mean return as part of
    un-standardizing, so every past winner carried a mechanically negative
    forward residual and momentum "predicted" it with an IC information ratio
    of −1.3 on data with nothing whatsoever to predict. That bug passed every
    other gate in this file — the eigenvalues were right, the factor counts
    were right, no future data was used anywhere — and on the real panel it
    produced a beautifully stable signal that would have been this project's
    headline result. Only a null with a known answer catches it.
    """
    idx = pd.bdate_range("2014-01-01", periods=n_days)
    names = [f"S{i:03d}" for i in range(n_assets)]
    per_panel: list[dict[str, float]] = []

    for panel_seed in range(seed, seed + n_panels):
        rng = np.random.default_rng(panel_seed)
        f = rng.standard_normal((n_days, k)) * 0.01
        loadings = rng.uniform(-1.0, 1.0, (k, n_assets))
        r = f @ loadings + rng.standard_normal((n_days, n_assets)) * 0.012 + 0.0004

        rets = pd.DataFrame(r, index=idx, columns=names)
        close = 50.0 * np.exp(rets.cumsum())

        resid, _ = pca.rolling_residuals(rets, window=252, step=21, n_components=k)
        fwd = resid.rolling(5).sum().shift(-5)
        fwd = fwd.sub(fwd.mean(axis=1), axis=0)

        probes = {
            "mom_12_1": features.momentum(close, 252, 21),
            "rev_5": features.reversal(close, 5),
            "rev_21": features.reversal(close, 21),
            "vol_21": features.realized_vol(rets, 21),
        }
        row = {}
        for name, mat in probes.items():
            ic = diagnostics.rank_ic(features.cs_rank(mat), fwd)
            sd = ic.std(ddof=1)
            row[name] = float(ic.mean() / sd) if sd > 0 else 0.0
        per_panel.append(row)

    averaged = pd.DataFrame(per_panel).mean()
    worst_name = str(averaged.abs().idxmax())
    worst_ir = float(averaged[worst_name])

    return Gate(
        name="Residualization manufactures no signal on a null panel",
        statistic=abs(worst_ir),
        tolerance=0.12,
        comparator="<",
        units="abs IC information ratio",
        detail=f"worst of 4 probe features over {n_panels} null panels: "
               f"{worst_name} at IR {worst_ir:+.3f} "
               f"(centring version of the residualizer scores -1.34)",
    )


# --------------------------------------------------------------------------
# Gates 4-5: the linear model
# --------------------------------------------------------------------------


def gate_ridge_identity(n_obs: int = 400, n_features: int = 8, seed: int = 44) -> Gate:
    """On an orthonormal design, ridge must equal OLS divided by ``1 + alpha``.

    An exact algebraic identity, not an approximation: when ``X'X = I`` the
    ridge solution is ``(I + alpha*I)^-1 X'y``. Checking it pins the shrinkage
    to the right place in the SVD — an implementation that penalised the
    intercept, or scaled ``alpha`` by the sample size, would fail here and
    could otherwise be tuned away by picking a different ``alpha``.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_obs, n_features))
    a -= a.mean(axis=0)                    # zero-mean columns => centring is a no-op
    q, _ = np.linalg.qr(a)
    y = rng.standard_normal(n_obs)

    base = models.RidgeRegression(0.0, standardize=False).fit(q, y).coef_
    worst = 0.0
    for alpha in (0.25, 1.0, 4.0, 25.0):
        got = models.RidgeRegression(alpha, standardize=False).fit(q, y).coef_
        worst = max(worst, float(np.abs(got - base / (1.0 + alpha)).max()))

    return Gate(
        name="Ridge shrinkage identity on an orthonormal design",
        statistic=worst,
        tolerance=1e-12,
        comparator="<",
        units="max abs deviation",
        detail="checked at alpha = 0.25, 1, 4, 25",
    )


def gate_ridge_recovery(n_obs: int = 4_000, n_features: int = 10, seed: int = 55) -> Gate:
    """Ridge must recover coefficients that were put in by hand.

    With a small penalty and a well-conditioned design the bias is negligible
    and the estimates should land on the true values. Also checks that
    ``alpha=0`` reproduces the least-squares solution to machine precision,
    which is the property that makes the ridge a strict generalisation of the
    baseline rather than a different model.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_obs, n_features))
    true = np.linspace(-2.0, 3.0, n_features)
    y = 1.5 + x @ true + rng.standard_normal(n_obs) * 0.5

    fit = models.RidgeRegression(alpha=1.0).fit(x, y)
    coef_err = float(np.abs(fit.coef_ - true).max())

    lstsq = np.linalg.lstsq(np.column_stack([np.ones(n_obs), x]), y, rcond=None)[0]
    ols = models.RidgeRegression(alpha=0.0).fit(x, y)
    ols_err = max(
        float(np.abs(ols.coef_ - lstsq[1:]).max()),
        float(abs(ols.intercept_ - lstsq[0])),
    )

    return Gate(
        name="Ridge coefficient recovery (and OLS at alpha=0)",
        statistic=max(coef_err, ols_err * 1e6),
        tolerance=0.05,
        comparator="<",
        units="max abs error",
        detail=f"coefficients within {coef_err:.4f} of truth; "
               f"alpha=0 matches lstsq to {ols_err:.2e}",
    )


# --------------------------------------------------------------------------
# Gates 6-7: the trees
# --------------------------------------------------------------------------


def _brute_force_gain(
    x: np.ndarray,
    y: np.ndarray,
    min_samples_leaf: int,
    thresholds: list[np.ndarray] | None = None,
) -> float:
    """Best variance reduction over every admissible threshold, by exhaustion.

    ``thresholds=None`` searches every distinct value of every feature — the
    unrestricted optimum. Passing the binner's edges instead restricts the
    search to the splits a binned tree is able to represent.
    """
    total, n = float(y.sum()), len(y)
    parent = total ** 2 / n
    best = 0.0
    for j in range(x.shape[1]):
        cuts = np.unique(x[:, j]) if thresholds is None else thresholds[j]
        for cut in cuts:
            mask = x[:, j] <= cut
            n_left = int(mask.sum())
            n_right = n - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            s_left = float(y[mask].sum())
            gain = (
                s_left ** 2 / n_left
                + (total - s_left) ** 2 / n_right
                - parent
            )
            best = max(best, gain)
    return best


def gate_tree_split(n_reps: int = 25, n_obs: int = 500, seed: int = 66) -> Gate:
    """The binned split search must find the exact optimum it is able to represent.

    The histogram search is the one performance shortcut in the package, and
    this checks that it is not also an accuracy shortcut. The claim being
    tested is precise: among the thresholds the binning *can* express, the
    search must find the best one exactly — not approximately, and not usually.
    So the brute-force comparison is run over the binner's own edges, and the
    two must agree to floating-point tolerance.

    The looser question — what does binning itself cost? — is answered in the
    detail line rather than the statistic, because it is a property of the bin
    count and not of the search. Quantising 500 continuous rows into 64 bins
    puts roughly 8 rows in each, so the best achievable threshold can sit a few
    rows away from the unrestricted optimum. That costs a small fraction of the
    available gain at the root and nothing at all in the model, where the
    features are cross-sectional ranks with far fewer distinct values than bins.
    """
    rng = np.random.default_rng(seed)
    worst_exact, worst_cost = 0.0, 0.0
    for _ in range(n_reps):
        x = rng.standard_normal((n_obs, 4))
        y = (
            2.0 * (x[:, 1] > 0.3)
            + 0.7 * x[:, 0]
            + rng.standard_normal(n_obs) * 0.8
        )
        binner = models.QuantileBinner(n_bins=64)
        binned = binner.fit_transform(x)
        tree = models.DecisionTreeRegressor(max_depth=1, min_samples_leaf=5)
        tree.fit(binned, y, binner.bin_counts)

        representable = _brute_force_gain(x, y, 5, thresholds=binner.edges_)
        unrestricted = _brute_force_gain(x, y, 5, thresholds=None)

        worst_exact = max(
            worst_exact,
            abs(tree.root_.gain - representable) / max(representable, 1e-12),
        )
        worst_cost = max(
            worst_cost,
            (unrestricted - representable) / max(unrestricted, 1e-12),
        )

    return Gate(
        name="Binned split search is exact over representable splits",
        statistic=worst_exact,
        tolerance=1e-9,
        comparator="<",
        units="max relative gain shortfall",
        detail=f"{n_reps} random datasets of {n_obs} rows x 4 features; "
               f"64-bin quantisation itself costs at most "
               f"{worst_cost:.2%} of the unrestricted best split",
    )


def _friedman1(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Friedman's benchmark: two interactions, a quadratic, two linear terms,
    and five pure-noise features that a working model must ignore."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 10))
    y = (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20 * (x[:, 2] - 0.5) ** 2
        + 10 * x[:, 3]
        + 5 * x[:, 4]
        + rng.standard_normal(n)
    )
    return x, y


def gate_gbm_nonlinear(n_train: int = 4_000, n_test: int = 4_000) -> Gate:
    """Boosting must learn a nonlinear function that ridge provably cannot.

    Friedman #1 contains an ``x1*x2`` interaction and a quadratic term, so a
    linear model has a ceiling on it no amount of data removes. The gate
    requires the boosted trees to clear a high out-of-sample R^2 *and* to beat
    ridge by a wide margin — the second half being the part that would catch a
    "boosting" implementation that had quietly collapsed to a constant plus a
    linear fit.
    """
    x_tr, y_tr = _friedman1(n_train, seed=77)
    x_te, y_te = _friedman1(n_test, seed=78)

    gbm = models.GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=3,
        min_samples_leaf=10, subsample=0.8, seed=7,
    ).fit(x_tr, y_tr)
    ridge = models.RidgeRegression(alpha=1.0).fit(x_tr, y_tr)

    def r2(pred):
        return 1.0 - float(np.mean((y_te - pred) ** 2)) / float(np.var(y_te))

    r2_gbm, r2_ridge = r2(gbm.predict(x_te)), r2(ridge.predict(x_te))
    noise_share = float(gbm.importances()[5:].sum())

    return Gate(
        name="Boosting recovers Friedman #1 nonlinearity",
        statistic=r2_gbm,
        tolerance=0.85,
        comparator=">",
        units="out-of-sample R^2",
        detail=f"ridge on the same data: {r2_ridge:.3f}; "
               f"the 5 pure-noise features take {noise_share:.1%} of importance",
    )


# --------------------------------------------------------------------------
# Gate 8: the split
# --------------------------------------------------------------------------


def gate_purge(n_days: int = 500, n_names: int = 20, horizon: int = 5) -> Gate:
    """Purged folds must contain zero label overlap — and unpurged ones must not.

    The control is the point. "No overlaps found" is worthless unless the same
    counter finds overlaps when purging is switched off, so the gate reports
    both and passes only if the unpurged count is strictly positive while the
    purged count is exactly zero.
    """
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    row_dates = pd.DatetimeIndex(np.repeat(dates.to_numpy(), n_names))

    purged = crossval.purged_walk_forward(
        row_dates, horizon=horizon, initial_train=200, test_size=60, embargo=5
    )
    unpurged = crossval.purged_walk_forward(
        row_dates, horizon=horizon, initial_train=200, test_size=60, embargo=0
    )
    # Rebuild the unpurged folds with the purge removed as well, by extending
    # each training block back to the test boundary.
    naive = []
    for fold in unpurged:
        extra = np.flatnonzero(
            (row_dates >= fold.train_start) & (row_dates < fold.test_start)
        )
        naive.append(
            crossval.Fold(
                index=fold.index, train_rows=extra, test_rows=fold.test_rows,
                train_start=fold.train_start, train_end=fold.test_start,
                test_start=fold.test_start, test_end=fold.test_end, n_purged=0,
            )
        )

    n_purged_violations = sum(
        crossval.overlap_violations(row_dates, f, horizon) for f in purged
    )
    n_naive_violations = sum(
        crossval.overlap_violations(row_dates, f, horizon) for f in naive
    )

    ok = n_purged_violations == 0 and n_naive_violations > 0
    return Gate(
        name="Purged folds leak no labels (with an unpurged control)",
        statistic=1.0 if ok else 0.0,
        tolerance=0.5,
        comparator=">",
        units="pass flag",
        detail=f"purged: {n_purged_violations} overlapping training rows across "
               f"{len(purged)} folds; unpurged control: {n_naive_violations}",
    )


# --------------------------------------------------------------------------
# Gates 9-10: the pipeline, end to end
# --------------------------------------------------------------------------


def synthetic_panel(
    n_days: int = 900,
    n_names: int = 40,
    seed: int = 88,
    alpha_strength: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """A price panel with a market factor, sector blocks, and optional planted alpha.

    ``alpha_strength=0`` gives a panel with realistic factor structure and no
    predictability — the right null for the look-ahead gate, which must hold
    whether or not there is anything to predict. A positive value makes next
    week's idiosyncratic return depend on this week's reversal signal, which is
    what gate 10 requires the pipeline to find.

    Returns the same dict of matrices as ``data.load_panel`` so the gates
    exercise exactly the production code path.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2013-01-01", periods=n_days)
    names = [f"S{i:03d}" for i in range(n_names)]

    market = rng.standard_normal(n_days) * 0.009
    n_sectors = 4
    sector_of = np.arange(n_names) % n_sectors
    sector = rng.standard_normal((n_days, n_sectors)) * 0.006
    beta = rng.uniform(0.6, 1.4, n_names)

    idio = rng.standard_normal((n_days, n_names)) * 0.013
    rets = (
        market[:, None] * beta
        + sector[:, sector_of]
        + idio
    )

    if alpha_strength > 0:
        # Last week's loser outperforms next week, by construction, on top of
        # everything else. Cross-sectionally demeaned so it adds no market
        # return — exactly the kind of effect the strategy is built to harvest.
        for t in range(10, n_days - 5):
            past = rets[t - 5: t].sum(axis=0)
            tilt = -(past - past.mean())
            tilt /= max(np.abs(tilt).std(), 1e-9)
            rets[t: t + 5] += alpha_strength * tilt / 5.0

    close = pd.DataFrame(
        50.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=names
    )
    spread = 0.004 + 0.004 * rng.random((n_days, n_names))
    volume = pd.DataFrame(
        np.exp(rng.normal(13.0, 0.6, (n_days, n_names))), index=idx, columns=names
    )
    return {
        "close": close,
        "volume": volume,
        "high": close * (1.0 + spread),
        "low": close * (1.0 - spread),
    }


def _run_mini_pipeline(
    panel: dict[str, pd.DataFrame],
    horizon: int = 5,
    alpha: float = 100.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Features -> design -> purged folds -> ridge -> book -> P&L. The whole chain."""
    feats = features.build_features(panel)
    target = features.build_target(panel["close"], horizon=horizon)
    design = data.stack_panel(feats, target)
    row_dates = design.index.get_level_values(0)

    folds = crossval.purged_walk_forward(
        row_dates, horizon=horizon, initial_train=180, test_size=60,
        embargo=5, min_test_size=20,
    )
    result = pipeline.run_walk_forward(
        design, folds, lambda: models.RidgeRegression(alpha), "ridge"
    )
    book = signals.signal_to_book(result.signal, horizon=horizon, quantile=0.2)
    rets = panel["close"].pct_change().iloc[1:]
    run = bt.Backtester(rets, cost_bps=10.0).run(book, name="ridge")
    return run.returns, result.signal


def gate_lookahead(horizon: int = 5) -> Gate:
    """Rewriting the last bar must leave every earlier day of P&L untouched.

    The strongest single check in the suite, and the same one project 03 uses.
    Features, the label, the fold boundaries, the model fit, the ranking and
    the weights are all estimated from data, so a single misplaced ``shift``
    anywhere in that chain lets tomorrow's price influence today's position.
    Re-running the entire pipeline on a panel whose final row has been
    multiplied by 1.5, and demanding bit-identical earlier returns, catches it
    regardless of where it hides.
    """
    panel = synthetic_panel()
    base, _ = _run_mini_pipeline(panel, horizon=horizon)

    tampered = {k: v.copy() for k, v in panel.items()}
    for k in tampered:
        tampered[k].iloc[-1] = tampered[k].iloc[-1] * 1.5
    after, _ = _run_mini_pipeline(tampered, horizon=horizon)

    common = base.index.intersection(after.index)[:-1]
    worst = float(np.max(np.abs(base.loc[common].to_numpy() - after.loc[common].to_numpy())))
    return Gate(
        name="No look-ahead anywhere in the pipeline",
        statistic=worst,
        tolerance=1e-15,
        comparator="<",
        units="max abs P&L difference",
        detail=f"{len(common)} earlier days compared after a +50% shock to the last bar",
    )


def gate_signal_recovery(horizon: int = 5, strength: float = 0.9) -> Gate:
    """The pipeline must find a cross-sectional signal that was planted in the panel.

    The power half of the pair with gate 9. A pipeline that leaks nothing but
    also learns nothing would pass every other gate in this file; this one
    requires the out-of-sample information coefficient to be clearly positive
    on a panel where next week's return really does depend on last week's
    reversal.
    """
    panel = synthetic_panel(alpha_strength=strength, seed=99)
    _, signal = _run_mini_pipeline(panel, horizon=horizon)

    fwd = features.build_target(panel["close"], horizon=horizon)
    ic = diagnostics.rank_ic(signal, fwd)
    mean_ic = float(ic.mean())
    ir = float(mean_ic / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else np.nan

    return Gate(
        name="Pipeline recovers a planted cross-sectional signal",
        statistic=mean_ic,
        tolerance=0.05,
        comparator=">",
        units="out-of-sample mean IC",
        detail=f"daily IC information ratio {ir:.3f} over {int(ic.notna().sum())} "
               f"out-of-sample days",
    )


# --------------------------------------------------------------------------


GATES: list[Callable[..., Gate]] = [
    gate_pca_spectrum,
    gate_mp_law,
    gate_factor_count,
    gate_residual_is_neutral,
    gate_ridge_identity,
    gate_ridge_recovery,
    gate_tree_split,
    gate_gbm_nonlinear,
    gate_purge,
    gate_lookahead,
    gate_signal_recovery,
]


def validation_gates(quick: bool = False) -> pd.DataFrame:
    """Run every gate and return the results as a table.

    ``quick=True`` shrinks the Monte Carlo gates for a fast smoke run. The
    committed result and the CI run both use the full settings; the quick mode
    exists so a broken edit is caught in seconds rather than a minute.
    """
    results = []
    for fn in GATES:
        if quick and fn is gate_mp_law:
            gate = fn(n_obs=600, n_assets=150, n_reps=4)
        elif quick and fn is gate_gbm_nonlinear:
            gate = fn(n_train=1_500, n_test=1_500)
        elif quick and fn is gate_tree_split:
            gate = fn(n_reps=6, n_obs=300)
        else:
            gate = fn()
        results.append(gate)

    return pd.DataFrame(
        [
            {
                "statistic": g.statistic,
                "tolerance": g.tolerance,
                "comparator": g.comparator,
                "units": g.units,
                "detail": g.detail,
                "passed": g.passed,
            }
            for g in results
        ],
        index=[g.name for g in results],
    )
