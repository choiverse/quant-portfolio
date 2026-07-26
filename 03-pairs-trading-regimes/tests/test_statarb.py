"""Correctness tests for statarb.

The econometrics in this package is written from scratch, so the tests are not
"does it run" but "does it give the answer that is already known": does the ADF
regression reject what it should, does EM increase the likelihood it claims to
maximise, does a hedge ratio come back as the number that was put in, does a
position decided on day t earn day t+1's return and not day t's.

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

from statarb import (  # noqa: E402
    attribution,
    backtest,
    cointegration as coint,
    data,
    metrics,
    pairs,
    regimes,
    strategy,
    validation,
    volatility as vol,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def dates():
    return pd.bdate_range("2018-01-01", periods=600)


@pytest.fixture
def random_walk(dates):
    rng = np.random.default_rng(0)
    return pd.Series(np.cumsum(rng.standard_normal(len(dates))), index=dates)


@pytest.fixture
def ar1(dates):
    """A stationary AR(1) with phi = 0.8 — mean-reverting by construction."""
    rng = np.random.default_rng(1)
    e = rng.standard_normal(len(dates))
    y = np.empty(len(dates))
    y[0] = e[0]
    for t in range(1, len(dates)):
        y[t] = 0.8 * y[t - 1] + e[t]
    return pd.Series(y, index=dates)


@pytest.fixture
def coint_pair(dates):
    """``y = 0.5 + 1.4x + s`` with x a random walk and s a tight AR(1)."""
    return validation._cointegrated_pair(len(dates), beta=1.4, phi=0.9, seed=11)


@pytest.fixture
def panel():
    """Synthetic price panel with planted cointegrated pairs plus noise."""
    return validation.synthetic_panel(n_days=700, n_pairs=6, n_noise=6, seed=5)


# --------------------------------------------------------------------------
# Cointegration — against known answers, not against itself
# --------------------------------------------------------------------------
def test_adf_does_not_reject_a_random_walk(random_walk):
    res = coint.adf_test(random_walk.to_numpy(), max_lag=4)
    assert not res.rejects(0.05)


def test_adf_rejects_a_stationary_ar1(ar1):
    res = coint.adf_test(ar1.to_numpy(), max_lag=4)
    assert res.rejects(0.01)


def test_adf_statistic_is_more_negative_for_stationary(random_walk, ar1):
    rw = coint.adf_test(random_walk.to_numpy(), max_lag=4).stat
    st = coint.adf_test(ar1.to_numpy(), max_lag=4).stat
    assert st < rw


def test_mackinnon_critical_values_are_ordered():
    """1% must be the most negative and 10% the least, at any sample size."""
    crit = coint.mackinnon_crit("c", 1, nobs=500)
    assert crit[0.01] < crit[0.05] < crit[0.10]


def test_mackinnon_converges_to_asymptotic_value():
    finite = coint.mackinnon_crit("c", 1, nobs=100)[0.05]
    asymptotic = coint.MACKINNON[("c", 1)][0.05][0]
    huge = coint.mackinnon_crit("c", 1, nobs=10_000_000)[0.05]
    assert huge == pytest.approx(asymptotic, abs=1e-5)
    assert finite < asymptotic          # small samples need a harder threshold


def test_residual_based_critical_values_are_stricter():
    """A residual from an estimated regression needs a lower bar than raw data.

    Regressing y on x already minimises the residual variance, so the residual
    looks more stationary than it is. If this ordering ever reversed, every
    pair in the study would be tested against too easy a threshold.
    """
    plain = coint.mackinnon_crit("c", 1, nobs=500)[0.05]
    residual = coint.mackinnon_crit("c", 2, nobs=500)[0.05]
    assert residual < plain


def test_unsupported_critical_value_case_raises():
    with pytest.raises(ValueError):
        coint.mackinnon_crit("n", 2, nobs=500)


def test_hedge_ratio_recovers_known_slope(coint_pair):
    y, x = coint_pair
    _, beta = coint.hedge_ratio(y, x)
    assert beta == pytest.approx(1.4, abs=0.05)


def test_engle_granger_finds_a_planted_pair(coint_pair):
    y, x = coint_pair
    res = coint.engle_granger(y, x)
    assert res.rejects(0.05)
    assert np.isfinite(res.half_life)


def test_engle_granger_rejects_nothing_for_independent_walks(dates):
    rng = np.random.default_rng(7)
    a = pd.Series(np.cumsum(rng.standard_normal(len(dates))), index=dates)
    b = pd.Series(np.cumsum(rng.standard_normal(len(dates))), index=dates)
    assert not coint.engle_granger(a, b).rejects(0.05)


def test_engle_granger_spread_matches_reported_orientation(coint_pair):
    """``swapped`` must describe the spread that was actually returned.

    ``pairs.screen_window`` uses this flag to decide which leg is held long,
    so a wrong value would silently invert every position for that pair.
    """
    y, x = coint_pair
    res = coint.engle_granger(y, x)
    lhs, rhs = (x, y) if res.swapped else (y, x)
    expected = lhs - res.alpha - res.beta * rhs
    assert np.allclose(res.spread.to_numpy(), expected.to_numpy())


def test_ou_half_life_is_infinite_for_an_explosive_series():
    """Nothing pulls an explosive series back, so there is no half-life."""
    s = 1.01 ** np.arange(200.0)
    assert np.isinf(coint.ou_half_life(s))


def test_ou_half_life_of_a_random_walk_is_long_not_infinite(random_walk):
    """A random walk *sample* reverts slightly, and the estimator must say so.

    The population half-life is infinite, but OLS of ``ds`` on ``s`` is biased
    downward on a finite sample — the same Dickey-Fuller bias the ADF critical
    values exist to correct. So the estimate comes back large and finite, not
    infinite, and the half-life filter in ``pairs.screen_window`` is what keeps
    such a series out rather than a test for ``inf``.
    """
    hl = coint.ou_half_life(random_walk)
    assert hl > len(random_walk) / 10


def test_ou_half_life_matches_the_ar1_coefficient():
    """phi = 0.9 halves a shock in ln2 / -ln(0.9) = 6.58 days, not 6.93."""
    rng = np.random.default_rng(2)
    n = 20_000
    s = np.empty(n)
    s[0] = rng.standard_normal()
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.standard_normal()
    assert coint.ou_half_life(s) == pytest.approx(np.log(2) / -np.log(0.9), rel=0.05)


def test_adf_rejects_too_short_a_series():
    with pytest.raises(ValueError):
        coint.adf_test(np.arange(5.0))


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------
def test_ewma_matches_the_recursion_by_hand():
    idx = pd.bdate_range("2020-01-01", periods=8)
    r = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005, -0.01, 0.02, -0.005], index=idx)
    lam, init = 0.94, 3
    got = vol.ewma_vol(r, lam=lam, annualize=False, init=init)

    assert got.iloc[: init - 1].isna().all()
    var = r.iloc[:init].var(ddof=1)
    assert got.iloc[init - 1] == pytest.approx(np.sqrt(var))
    for t in range(init, len(r)):
        var = lam * var + (1 - lam) * r.iloc[t - 1] ** 2
        assert got.iloc[t] == pytest.approx(np.sqrt(var))


def test_ewma_seed_does_not_span_the_whole_sample():
    """The seed must come from the head of the series, not all of it."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.Series(np.full(60, 0.001), index=idx)
    r.iloc[-1] = 0.5                      # a huge move, far in the future
    assert np.isfinite(vol.ewma_vol(r, annualize=False).iloc[20])
    assert vol.ewma_vol(r, annualize=False).iloc[20] < 0.01


def test_ewma_uses_only_past_returns():
    """Changing the last return must not change any earlier volatility value."""
    idx = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 50), index=idx)
    bumped = r.copy()
    bumped.iloc[-1] = 0.5

    a = vol.ewma_vol(r)
    b = vol.ewma_vol(bumped)
    pd.testing.assert_series_equal(a.iloc[:-1], b.iloc[:-1])


def test_ewma_rejects_a_bad_decay():
    r = pd.Series(np.zeros(10))
    with pytest.raises(ValueError):
        vol.ewma_vol(r, lam=1.0)


def test_realized_vol_annualizes():
    idx = pd.bdate_range("2020-01-01", periods=40)
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 40), index=idx)
    daily = vol.realized_vol(r, window=21, annualize=False)
    ann = vol.realized_vol(r, window=21, annualize=True)
    assert ann.dropna().iloc[0] == pytest.approx(daily.dropna().iloc[0] * np.sqrt(252))


def test_garch_recovers_persistence():
    x = vol.simulate_garch11(3000, omega=0.02, alpha=0.08, beta=0.90, seed=9) / 100.0
    fit = vol.fit_garch11(pd.Series(x))
    assert fit.persistence == pytest.approx(0.98, abs=0.03)
    assert fit.long_run_var > 0


def test_garch_half_life_follows_from_persistence():
    x = vol.simulate_garch11(2000, omega=0.02, alpha=0.05, beta=0.90, seed=10) / 100.0
    fit = vol.fit_garch11(pd.Series(x))
    assert fit.half_life == pytest.approx(np.log(0.5) / np.log(fit.persistence))


def test_simulate_garch_rejects_explosive_parameters():
    with pytest.raises(ValueError):
        vol.simulate_garch11(100, omega=0.1, alpha=0.5, beta=0.6)


# --------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------
@pytest.fixture
def two_regime_series():
    rng = np.random.default_rng(12)
    n = 1200
    state = np.zeros(n, dtype=int)
    A = np.array([[0.97, 0.03], [0.08, 0.92]])
    for t in range(1, n):
        state[t] = rng.choice(2, p=A[state[t - 1]])
    x = rng.normal([0.0005, -0.001][0] * 0 + np.where(state == 0, 0.0005, -0.001),
                   np.where(state == 0, 0.006, 0.02))
    return pd.Series(x, index=pd.bdate_range("2015-01-01", periods=n)), state


def test_em_never_decreases_the_log_likelihood(two_regime_series):
    """The defining property of EM. If this fails, the M-step is wrong."""
    x, _ = two_regime_series
    model = regimes.GaussianHMM(n_states=2)
    model._init(x.to_numpy(), seed=0)

    lls = []
    for _ in range(15):
        lls.append(model.score(x.to_numpy()))
        model.fit(x.to_numpy(), n_iter=1, tol=-np.inf, seed=0)
    # fit() re-initialises, so drive it directly for a monotonicity check
    model = regimes.GaussianHMM(n_states=2)
    prev = -np.inf
    for n_iter in range(1, 12):
        model.fit(x.to_numpy(), n_iter=n_iter, tol=-np.inf, seed=0)
        assert model.loglik >= prev - 1e-8
        prev = model.loglik


def test_states_are_ordered_by_variance(two_regime_series):
    x, _ = two_regime_series
    res = regimes.fit_regimes(x, seed=0)
    assert res.model.variances[0] < res.model.variances[1]


def test_posteriors_are_probability_distributions(two_regime_series):
    x, _ = two_regime_series
    res = regimes.fit_regimes(x, seed=0)
    for probs in (res.model.filter(x.to_numpy()), res.model.smooth(x.to_numpy())):
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all()


def test_transition_matrix_rows_sum_to_one(two_regime_series):
    x, _ = two_regime_series
    res = regimes.fit_regimes(x, seed=0)
    assert np.allclose(res.model.transmat.sum(axis=1), 1.0)


def test_viterbi_path_covers_the_input(two_regime_series):
    x, _ = two_regime_series
    res = regimes.fit_regimes(x, seed=0)
    assert len(res.state) == len(x)
    assert set(np.unique(res.state)) <= {0, 1}


def test_expected_duration_matches_the_stay_probability(two_regime_series):
    x, _ = two_regime_series
    m = regimes.fit_regimes(x, seed=0).model
    assert np.allclose(m.expected_duration, 1.0 / (1.0 - np.diag(m.transmat)))


def test_stationary_distribution_is_a_fixed_point(two_regime_series):
    x, _ = two_regime_series
    m = regimes.fit_regimes(x, seed=0).model
    pi = m.stationary
    assert np.allclose(pi @ m.transmat, pi, atol=1e-8)
    assert pi.sum() == pytest.approx(1.0)


def test_filtered_probabilities_ignore_the_future(two_regime_series):
    """The filter is causal; the smoother is not. This is the difference."""
    x, _ = two_regime_series
    m = regimes.fit_regimes(x, seed=0).model

    bumped = x.copy()
    bumped.iloc[-1] = 1.0

    f_a, f_b = m.filter(x.to_numpy()), m.filter(bumped.to_numpy())
    assert np.allclose(f_a[:-1], f_b[:-1])

    s_a, s_b = m.smooth(x.to_numpy()), m.smooth(bumped.to_numpy())
    assert not np.allclose(s_a[:-1], s_b[:-1])


def test_causal_fit_leaves_the_burn_in_unlabelled(two_regime_series):
    x, _ = two_regime_series
    p = regimes.fit_causal(x, burn_in=400, seed=0)
    assert p.iloc[:400].isna().all()
    assert p.iloc[400:].notna().all()


def test_causal_fit_rejects_too_short_a_series():
    x = pd.Series(np.zeros(100), index=pd.bdate_range("2020-01-01", periods=100))
    with pytest.raises(ValueError):
        regimes.fit_causal(x, burn_in=504)


def test_vol_terciles_use_expanding_quantiles():
    """Appending future data must not relabel a day that is already classified."""
    idx = pd.bdate_range("2015-01-01", periods=500)
    rng = np.random.default_rng(13)
    v = pd.Series(np.abs(rng.normal(0.15, 0.05, 500)), index=idx)

    full = regimes.vol_tercile_regimes(v, min_periods=100)
    truncated = regimes.vol_tercile_regimes(v.iloc[:400], min_periods=100)
    common = truncated.dropna().index
    pd.testing.assert_series_equal(full.loc[common], truncated.loc[common])


# --------------------------------------------------------------------------
# Pair selection and the trading rule
# --------------------------------------------------------------------------
def test_distance_matrix_matches_a_brute_force_loop():
    rng = np.random.default_rng(14)
    lp = pd.DataFrame(
        np.cumsum(rng.standard_normal((120, 5)), axis=0),
        columns=list("ABCDE"),
        index=pd.bdate_range("2020-01-01", periods=120),
    )
    d = pairs.distance_matrix(lp)

    z = lp - lp.iloc[0]
    z = z / z.std(ddof=1)
    for i, a in enumerate(lp.columns):
        for j, b in enumerate(lp.columns):
            expected = float(((z[a] - z[b]) ** 2).sum())
            assert d[i, j] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_distance_screen_returns_the_closest_pairs():
    rng = np.random.default_rng(15)
    base = np.cumsum(rng.standard_normal(200))
    lp = pd.DataFrame(
        {
            "A": base,
            "B": base + rng.normal(0, 0.01, 200),   # nearly identical to A
            "C": np.cumsum(rng.standard_normal(200)),
        },
        index=pd.bdate_range("2020-01-01", periods=200),
    )
    top = pairs.distance_screen(lp, top_k=1)
    assert set(top.iloc[0][["a", "b"]]) == {"A", "B"}


def test_n_possible_pairs():
    assert pairs.n_possible_pairs(470) == 110_215
    assert pairs.n_possible_pairs(2) == 1


def test_positions_enter_exit_and_stop_on_a_hand_built_path():
    z = pd.Series(
        [0.0, 2.5, 1.5, 0.2, -2.5, -1.0, -0.1, 0.0],
        index=pd.bdate_range("2020-01-01", periods=8),
    )
    pos = pairs.pair_positions(z, entry=2.0, exit=0.5, stop=4.0)
    assert list(pos) == [0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_position_is_never_reopened_after_a_stop():
    z = pd.Series(
        [0.0, 2.5, 5.0, 2.5, 0.0, -3.0],
        index=pd.bdate_range("2020-01-01", periods=6),
    )
    pos = pairs.pair_positions(z, entry=2.0, exit=0.5, stop=4.0)
    assert list(pos) == [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]


def test_positions_require_a_sane_threshold_pair():
    z = pd.Series(np.zeros(5), index=pd.bdate_range("2020-01-01", periods=5))
    with pytest.raises(ValueError):
        pairs.pair_positions(z, entry=0.5, exit=2.0)


def test_pair_weights_are_gross_one_and_hedged():
    spec = pairs.PairSpec(
        y="A", x="B", alpha=0.0, beta=1.4, mu=0.0, sigma=1.0,
        stat=-4.0, pvalue=0.01, half_life=10.0, ssd=1.0,
    )
    wy, wx = spec.weights()
    assert abs(wy) + abs(wx) == pytest.approx(1.0)
    assert wx / wy == pytest.approx(-1.4)


def test_zscore_uses_formation_moments_not_the_window_moments():
    """Standardising by the trading window's own mean would be look-ahead."""
    idx = pd.bdate_range("2020-01-01", periods=50)
    lp = pd.DataFrame({"A": np.linspace(0, 1, 50) + 10.0, "B": np.zeros(50)}, index=idx)
    spec = pairs.PairSpec(
        y="A", x="B", alpha=10.0, beta=0.0, mu=0.0, sigma=0.5,
        stat=-4.0, pvalue=0.01, half_life=10.0, ssd=1.0,
    )
    z = pairs.zscore(pairs.spread_series(lp, spec), spec)
    assert z.mean() != pytest.approx(0.0)          # not re-centred on itself
    assert z.iloc[0] == pytest.approx(0.0)         # spread starts at mu


def test_screen_reports_its_own_multiplicity(panel):
    lp = data.to_log_prices(panel).iloc[:300]
    res = pairs.screen_window(lp, top_k=40, max_pairs=5)
    assert res.n_tested <= 40
    assert res.n_possible == pairs.n_possible_pairs(lp.shape[1])
    assert res.expected_false_positives == pytest.approx(res.n_tested * 0.05)
    assert len(res.pairs) <= 5


def test_screen_finds_the_planted_pairs(panel):
    """The synthetic panel plants cointegrated pairs named P<i>A / P<i>B."""
    lp = data.to_log_prices(panel).iloc[:400]
    res = pairs.screen_window(lp, top_k=60, max_pairs=10)
    found = {tuple(sorted((s.y, s.x))) for s in res.pairs}
    planted = {tuple(sorted((f"P{i}A", f"P{i}B"))) for i in range(6)}
    assert found & planted


# --------------------------------------------------------------------------
# Backtest mechanics
# --------------------------------------------------------------------------
@pytest.fixture
def returns_panel():
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(16)
    return pd.DataFrame(
        rng.normal(0.0004, 0.01, (100, 4)), index=idx, columns=list("ABCD")
    )


def test_flat_book_earns_nothing(returns_panel):
    w = pd.DataFrame(0.0, index=returns_panel.index, columns=returns_panel.columns)
    res = backtest.PairBacktester(returns_panel).run(w)
    assert res.returns.abs().sum() == pytest.approx(0.0)
    assert res.turnover.sum() == pytest.approx(0.0)


def test_constant_long_book_reproduces_the_asset(returns_panel):
    w = pd.DataFrame(0.0, index=returns_panel.index, columns=returns_panel.columns)
    w["A"] = 1.0
    res = backtest.PairBacktester(returns_panel, cost_bps=0.0).run(w)
    # Day 0 is unfunded (the book was set at the close of day 0), so compare from day 1.
    assert np.allclose(res.returns.iloc[1:], returns_panel["A"].iloc[1:])


def test_weights_are_applied_with_a_one_day_lag(returns_panel):
    """A weight set at the close of t must earn t+1, never t."""
    w = pd.DataFrame(0.0, index=returns_panel.index, columns=returns_panel.columns)
    w.iloc[5, w.columns.get_loc("A")] = 1.0
    res = backtest.PairBacktester(returns_panel, cost_bps=0.0).run(w)
    assert res.returns.iloc[5] == pytest.approx(0.0)
    assert res.returns.iloc[6] == pytest.approx(returns_panel["A"].iloc[6])


def test_turnover_is_charged_when_the_position_starts_earning(returns_panel):
    w = pd.DataFrame(0.0, index=returns_panel.index, columns=returns_panel.columns)
    w.iloc[5, w.columns.get_loc("A")] = 1.0
    res = backtest.PairBacktester(returns_panel, cost_bps=10.0).run(w)
    # One unit in on day 6 and back out on day 7: turnover 1.0 on each.
    assert res.turnover.iloc[6] == pytest.approx(1.0)
    assert res.costs.iloc[6] == pytest.approx(10.0 / 1e4)
    assert res.returns.iloc[6] == pytest.approx(
        returns_panel["A"].iloc[6] - 10.0 / 1e4
    )


def test_costs_reduce_returns_monotonically(returns_panel):
    rng = np.random.default_rng(17)
    w = pd.DataFrame(
        rng.normal(0, 0.2, returns_panel.shape),
        index=returns_panel.index, columns=returns_panel.columns,
    )
    sweep = backtest.PairBacktester(returns_panel).cost_sweep(w, [0, 5, 10, 20])
    assert sweep["Sharpe"].is_monotonic_decreasing


def test_breakeven_cost_solves_the_right_equation(returns_panel):
    rng = np.random.default_rng(18)
    w = pd.DataFrame(
        rng.normal(0, 0.2, returns_panel.shape),
        index=returns_panel.index, columns=returns_panel.columns,
    )
    res = backtest.PairBacktester(returns_panel).run(w)
    be = res.breakeven_cost_bps()
    net = res.gross_returns - res.turnover * (be / 1e4)
    assert net.mean() == pytest.approx(0.0, abs=1e-15)


def test_gross_exposure_tracks_the_held_book(returns_panel):
    w = pd.DataFrame(0.0, index=returns_panel.index, columns=returns_panel.columns)
    w["A"], w["B"] = 0.5, -0.5
    res = backtest.PairBacktester(returns_panel).run(w)
    assert res.gross_exposure.iloc[-1] == pytest.approx(1.0)
    assert res.net_exposure.iloc[-1] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------
def test_sharpe_stderr_shrinks_with_sample_size():
    rng = np.random.default_rng(19)
    short = pd.Series(rng.normal(0.0005, 0.01, 100))
    long = pd.Series(rng.normal(0.0005, 0.01, 2000))
    assert attribution.sharpe_stderr(short) > attribution.sharpe_stderr(long)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(20)
    r = pd.Series(rng.normal(0.0006, 0.008, 800))
    boot = attribution.block_bootstrap_sharpe(r, n_boot=400, seed=0)
    assert boot["lo"] < boot["sharpe"] < boot["hi"]


def test_regime_table_splits_the_days_it_was_given():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(21)
    net = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    gross = net + 0.0001
    reg = pd.Series(["calm"] * 120 + ["turbulent"] * 80, index=idx)

    tab = attribution.regime_table(net, gross, reg, n_boot=200)
    assert set(tab.index) == {"calm", "turbulent"}
    assert tab["days"].sum() == 200
    # gross is net plus a constant here, so the drag is that constant, annualised
    assert np.allclose(tab["cost_drag_ann"], 0.0001 * metrics.TRADING_DAYS)
    assert tab["net_sharpe_boot_lo"].notna().all()


def test_contribution_shares_sum_to_one():
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(22)
    net = pd.Series(rng.normal(0.001, 0.01, 100), index=idx)
    reg = pd.Series(["calm"] * 60 + ["turbulent"] * 40, index=idx)
    out = attribution.contribution(net, reg)
    assert out["day_share"].sum() == pytest.approx(1.0)
    assert out["pnl_share"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The walk-forward protocol — the tests the whole study rests on
# --------------------------------------------------------------------------
def test_walk_forward_windows_do_not_overlap():
    idx = pd.bdate_range("2015-01-01", periods=1000)
    windows = data.split_walk_forward(idx, formation_days=252, trading_days=126)
    assert windows
    for (f, t) in windows:
        assert f.stop == t.start                   # formation ends where trading starts
    for (_, t1), (_, t2) in zip(windows, windows[1:]):
        assert t1.stop == t2.start                 # trading segments tile the sample


def test_walk_forward_needs_enough_history():
    lp = data.to_log_prices(validation.synthetic_panel(n_days=200, n_pairs=2, n_noise=2))
    with pytest.raises(ValueError):
        strategy.run_walk_forward(lp, formation_days=252, trading_days=126)


def test_no_position_is_taken_during_the_first_formation_window(panel):
    lp = data.to_log_prices(panel)
    wf = strategy.run_walk_forward(lp, top_k=40, max_pairs=5, max_lag=4)
    first_trade = wf.windows[0].trading_start
    assert (wf.weights.loc[:first_trade].iloc[:-1].abs().sum().sum()
            == pytest.approx(0.0))


def test_rewriting_the_last_day_cannot_change_earlier_pnl(panel):
    """The look-ahead test. Nothing about tomorrow may reach today.

    Selection, hedge ratios, spread moments and z-scores are all estimated
    from data, so one misplaced shift anywhere in that chain would show up
    here and nowhere else.
    """
    kwargs = dict(formation_days=252, trading_days=126, top_k=40,
                  max_pairs=5, max_lag=4)
    base, _ = strategy.run_strategy(panel, **kwargs)

    tampered = panel.copy()
    tampered.iloc[-1] *= 1.5
    after, _ = strategy.run_strategy(tampered, **kwargs)

    pd.testing.assert_series_equal(base.returns.iloc[:-1], after.returns.iloc[:-1])


def test_strategy_book_stays_within_its_gross_budget(panel):
    res, _ = strategy.run_strategy(panel, top_k=40, max_pairs=5, max_lag=4)
    assert res.gross_exposure.max() <= 1.0 + 1e-9


def test_window_table_records_the_audit_trail(panel):
    lp = data.to_log_prices(panel)
    wf = strategy.run_walk_forward(lp, top_k=40, max_pairs=5, max_lag=4)
    tab = wf.window_table
    assert len(tab) == len(wf.windows)
    assert (tab["pairs_tested"] <= tab["pairs_possible"]).all()
    assert (tab["pairs_traded"] <= tab["pairs_passed"]).all()
    assert (tab["expected_false_positives"] > 0).all()


# --------------------------------------------------------------------------
# Data shaping
# --------------------------------------------------------------------------
def test_log_prices_and_returns_are_consistent(panel):
    lp = data.to_log_prices(panel)
    log_ret = data.to_returns(panel, kind="log")
    assert np.allclose(lp.diff().iloc[1:].to_numpy(), log_ret.to_numpy())


def test_price_matrix_drops_incomplete_histories():
    idx = pd.bdate_range("2020-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": list(idx) * 2,
            "Name": ["A"] * 100 + ["B"] * 100,
            "close": list(np.arange(100.0)) + [np.nan] * 30 + list(np.arange(70.0)),
        }
    )
    wide = data.to_price_matrix(df, min_obs_frac=1.0)
    assert list(wide.columns) == ["A"]


def test_missing_file_raises_a_useful_error():
    with pytest.raises(FileNotFoundError, match="data/README"):
        data.load_prices("no/such/file.csv")


def test_market_return_is_the_cross_sectional_mean(returns_panel):
    m = data.market_return(returns_panel)
    assert np.allclose(m.to_numpy(), returns_panel.mean(axis=1).to_numpy())


# --------------------------------------------------------------------------
# The gates themselves stay wired up
# --------------------------------------------------------------------------
def test_validation_gates_run_and_report(monkeypatch):
    """A cheap subset, so a broken gate definition fails here too, in seconds."""
    monkeypatch.setattr(
        validation, "GATES",
        [validation.gate_hedge_ratio, validation.gate_half_life],
    )
    gates = validation.validation_gates()
    assert list(gates.columns) == [
        "statistic", "tolerance", "comparator", "units", "detail", "passed"
    ]
    assert gates["passed"].all()
