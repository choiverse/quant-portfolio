"""Correctness tests for quantbt.

The tests that matter in a backtester are not "does it run" but "does it lie":
does a weight decided on day t get applied to day t's return (look-ahead), does
turnover get charged on trades that did not happen, does a metric agree with its
closed form. Each test below pins one of those down on data where the right
answer is known by construction.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantbt import data, diagnostics, eda, metrics, signals  # noqa: E402
from quantbt.backtest import Backtester  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-01", periods=300)


@pytest.fixture
def panel(dates):
    """A synthetic 20-name price panel with a deterministic seed.

    The width matters: ``long_short_weights`` refuses to trade a cross-section
    with fewer than 10 valid scores, so a narrower panel would make the
    long/short tests pass vacuously against an all-zero book.
    """
    rng = np.random.default_rng(42)
    tickers = [f"T{i}" for i in range(20)]
    steps = rng.normal(0.0004, 0.012, size=(len(dates), len(tickers)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers)
    prices.columns.name = "ticker"
    return prices


# --------------------------------------------------------------------------
# Metrics — check against closed forms, not against themselves
# --------------------------------------------------------------------------
def test_equity_curve_matches_compounding(dates):
    r = pd.Series(0.01, index=dates[:10])
    assert metrics.equity_curve(r).iloc[-1] == pytest.approx(1.01 ** 10)


def test_cagr_of_constant_growth():
    """A series returning exactly 10%/yr for 2 years must report CAGR = 10%."""
    daily = 1.10 ** (1 / 252) - 1
    r = pd.Series(daily, index=pd.bdate_range("2020-01-01", periods=504))
    assert metrics.cagr(r) == pytest.approx(0.10, rel=1e-6)


def test_sharpe_matches_manual_formula(dates):
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_is_zero_for_zero_mean(dates):
    r = pd.Series([0.01, -0.01] * (len(dates) // 2), index=dates)
    assert metrics.sharpe_ratio(r) == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_known_path():
    """1 -> 1.2 -> 0.6: the worst peak-to-trough is exactly -50%."""
    r = pd.Series([0.2, -0.5], index=pd.bdate_range("2020-01-01", periods=2))
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_drawdown_never_positive(dates):
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, len(dates)), index=dates)
    assert (metrics.drawdown_series(r) <= 1e-12).all()


def test_sortino_ignores_upside(dates):
    """Adding a purely positive day must not increase downside deviation."""
    r = pd.Series([0.01, -0.01] * 50, index=dates[:100])
    boosted = r.copy()
    boosted.iloc[0] = 0.05
    assert metrics.sortino_ratio(boosted) > metrics.sortino_ratio(r)


# --------------------------------------------------------------------------
# Data shaping
# --------------------------------------------------------------------------
def test_price_matrix_drops_sparse_tickers():
    """A name present for well under 98% of days must not enter the universe."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    rows = [{"date": d, "close": 10.0, "Name": "FULL"} for d in dates]
    rows += [{"date": d, "close": 20.0, "Name": "SPARSE"} for d in dates[:50]]
    df = pd.DataFrame(rows)
    wide = data.to_price_matrix(df, field="close")
    assert list(wide.columns) == ["FULL"]


def test_returns_are_correct_and_lose_one_row(panel):
    rets = data.to_returns(panel)
    assert len(rets) == len(panel) - 1
    expected = panel.iloc[1, 0] / panel.iloc[0, 0] - 1
    assert rets.iloc[0, 0] == pytest.approx(expected)


def test_log_returns_sum_to_total_log_growth(panel):
    lr = data.to_returns(panel, kind="log")
    total = np.log(panel.iloc[-1, 0] / panel.iloc[0, 0])
    assert lr.iloc[:, 0].sum() == pytest.approx(total)


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
def test_momentum_skips_the_recent_month(panel):
    """The 12-1 score must not contain any information from the last `skip` days."""
    score = signals.cross_sectional_momentum(panel, lookback=100, skip=20)
    tampered = panel.copy()
    tampered.iloc[-10:] *= 1.5          # violently change only the last 10 days
    score2 = signals.cross_sectional_momentum(tampered, lookback=100, skip=20)
    assert score.iloc[-1].equals(score2.iloc[-1])


def test_reversal_is_the_negated_past_return(panel):
    score = signals.short_term_reversal(panel, lookback=5)
    past = panel / panel.shift(5) - 1.0
    pd.testing.assert_frame_equal(score, -past)


def test_long_short_weights_are_dollar_neutral(panel):
    score = signals.cross_sectional_momentum(panel, lookback=50, skip=5)
    w = signals.long_short_weights(score, quantile=0.3, gross=1.0)
    active = w.loc[w.abs().sum(axis=1) > 0]
    assert len(active) > 0, "book never traded — the assertions below would be vacuous"
    assert np.allclose(active.sum(axis=1), 0.0, atol=1e-12)      # dollar neutral
    assert np.allclose(active.abs().sum(axis=1), 1.0, atol=1e-12)  # gross = 1


def test_long_short_goes_long_the_top_scores():
    score = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
        index=[pd.Timestamp("2020-01-01")],
        columns=list("ABCDEFGHIJ"),
    )
    w = signals.long_short_weights(score, quantile=0.2)
    assert w.loc[:, "J"].iloc[0] > 0 and w.loc[:, "I"].iloc[0] > 0
    assert w.loc[:, "A"].iloc[0] < 0 and w.loc[:, "B"].iloc[0] < 0


def test_benchmark_is_fully_invested(panel):
    w = signals.long_only_benchmark(panel)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all().all()


# --------------------------------------------------------------------------
# Backtester — the look-ahead guarantees
# --------------------------------------------------------------------------
def test_no_lookahead_future_returns_cannot_change_the_past(panel):
    """Rewriting the *last* day's returns must leave every earlier P&L untouched."""
    rets = data.to_returns(panel)
    w = signals.long_only_benchmark(panel)
    bt = Backtester(rets, rebalance_every=5, cost_bps=10)

    base = bt.run(w).returns
    tampered_rets = rets.copy()
    tampered_rets.iloc[-1] = 99.0
    tampered = Backtester(tampered_rets, rebalance_every=5, cost_bps=10).run(w).returns

    pd.testing.assert_series_equal(base.iloc[:-1], tampered.iloc[:-1])


def test_weights_are_lagged_by_exactly_one_day(panel):
    """Day-1 P&L must be zero: the first weights are only knowable at that close."""
    rets = data.to_returns(panel)
    w = signals.long_only_benchmark(panel)
    res = Backtester(rets, rebalance_every=1, cost_bps=0).run(w)
    assert res.returns.iloc[0] == pytest.approx(0.0)


def test_costs_only_reduce_returns(panel):
    rets = data.to_returns(panel)
    score = signals.cross_sectional_momentum(panel, lookback=50, skip=5)
    w = signals.long_short_weights(score, quantile=0.3)

    free = Backtester(rets, rebalance_every=10, cost_bps=0).run(w)
    costly = Backtester(rets, rebalance_every=10, cost_bps=25).run(w)

    assert (costly.returns <= free.returns + 1e-15).all()
    assert metrics.total_return(costly.returns) < metrics.total_return(free.returns)


def test_zero_cost_means_gross_equals_net(panel):
    rets = data.to_returns(panel)
    w = signals.long_only_benchmark(panel)
    res = Backtester(rets, rebalance_every=7, cost_bps=0).run(w)
    pd.testing.assert_series_equal(res.returns, res.gross_returns)


def test_flat_book_earns_nothing(panel):
    rets = data.to_returns(panel)
    flat = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    res = Backtester(rets, rebalance_every=5, cost_bps=50).run(flat)
    assert res.returns.abs().max() == pytest.approx(0.0)
    assert res.turnover.abs().max() == pytest.approx(0.0)


def test_turnover_of_a_static_book_is_paid_once(panel):
    """Holding a constant weight vector should trade on day one and never again."""
    rets = data.to_returns(panel)
    w = pd.DataFrame(1.0 / panel.shape[1], index=panel.index, columns=panel.columns)
    res = Backtester(rets, rebalance_every=10, cost_bps=10).run(w)
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.turnover.iloc[1:].abs().max() == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def test_forward_returns_look_forward(panel):
    rets = data.to_returns(panel)
    fwd = diagnostics.forward_returns(rets, horizon=5)
    manual = (1 + rets.iloc[1:6, 0]).prod() - 1
    assert fwd.iloc[0, 0] == pytest.approx(manual)


def test_perfect_foresight_scores_ic_of_one(panel):
    """A score that *is* the forward return must have rank IC = 1 every day."""
    rets = data.to_returns(panel)
    fwd = diagnostics.forward_returns(rets, horizon=5)
    ic = diagnostics.information_coefficient(fwd, rets, horizon=5, min_names=3)
    assert ic.mean() == pytest.approx(1.0)


def test_random_score_has_ic_near_zero(panel):
    rng = np.random.default_rng(7)
    rets = data.to_returns(panel)
    noise = pd.DataFrame(rng.normal(size=rets.shape), index=rets.index, columns=rets.columns)
    ic = diagnostics.information_coefficient(noise, rets, horizon=5, min_names=3)
    assert abs(ic.mean()) < 0.15


def test_quantile_weights_partition_the_cross_section(panel):
    score = signals.cross_sectional_momentum(panel, lookback=50, skip=5)
    buckets = diagnostics.quantile_weights(score, n_quantiles=3)
    assert len(buckets) == 3
    stacked = sum((b > 0).astype(int) for b in buckets)
    live = score.notna().sum(axis=1) > 0
    # Each live name lands in exactly one bucket.
    assert stacked.loc[live].max().max() <= 1


def test_breakeven_cost_is_interpolated_correctly():
    sens = pd.DataFrame({"sharpe": [1.0, -1.0], "cagr": [0.1, -0.1]},
                        index=pd.Index([0.0, 10.0], name="cost_bps"))
    assert diagnostics.breakeven_cost(sens) == pytest.approx(5.0)


def test_breakeven_is_nan_when_never_crossing():
    sens = pd.DataFrame({"sharpe": [1.0, 0.5]}, index=pd.Index([0.0, 10.0], name="cost_bps"))
    assert np.isnan(diagnostics.breakeven_cost(sens))


# --------------------------------------------------------------------------
# EDA helpers
# --------------------------------------------------------------------------
def test_normal_ppf_matches_known_quantiles():
    assert eda.normal_ppf(np.array([0.5]))[0] == pytest.approx(0.0, abs=1e-9)
    assert eda.normal_ppf(np.array([0.975]))[0] == pytest.approx(1.959964, abs=1e-4)
    assert eda.normal_ppf(np.array([0.025]))[0] == pytest.approx(-1.959964, abs=1e-4)
    assert eda.normal_ppf(np.array([0.001]))[0] == pytest.approx(-3.090232, abs=1e-3)


def test_quality_flags_detect_a_broken_bar():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "open": [10.0, 10.0], "high": [9.0, 11.0], "low": [10.5, 9.0],
        "close": [10.0, 10.0], "volume": [100, 100], "Name": ["A", "A"],
    })
    flags = eda.quality_flags(df)
    assert flags.loc["high < low", "count"] == 1
    assert flags.loc["high < low", "verdict"] == "FAIL"
    assert flags.loc["Duplicated (date, ticker) rows", "verdict"] == "PASS"


def test_extreme_moves_flags_a_split():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "close": [100.0, 50.0], "Name": ["A", "A"],
    })
    out = eda.extreme_moves(df, threshold=0.4)
    assert len(out) == 1
    assert out["pct_change"].iloc[0] == pytest.approx(-0.5)
