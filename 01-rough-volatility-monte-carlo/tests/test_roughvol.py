"""Correctness tests for the rough-volatility engine.

``scripts/validate.py`` runs the five statistical gates on large path counts.
These are the fast, deterministic unit tests underneath them: closed-form
identities, structural invariants of the simulator, and regression tests for
bugs that have actually been found here.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roughvol import analysis  # noqa: E402
from roughvol.black_scholes import bs_price, bs_vega, implied_vol  # noqa: E402
from roughvol.mc_pricer import _standard_error, price_european, price_strikes  # noqa: E402
from roughvol.models import GBM, RoughBergomi  # noqa: E402


# --------------------------------------------------------------------------
# Black-Scholes closed form
# --------------------------------------------------------------------------
def test_put_call_parity():
    S0, K, T, r, sigma = 100.0, 95.0, 0.75, 0.03, 0.25
    call = float(bs_price(S0, K, T, r, sigma, "call"))
    put = float(bs_price(S0, K, T, r, sigma, "put"))
    assert call - put == pytest.approx(S0 - K * np.exp(-r * T), abs=1e-10)


def test_zero_vol_gives_discounted_intrinsic():
    """With sigma = 0 the option is worth its forward intrinsic value."""
    assert float(bs_price(100.0, 90.0, 1.0, 0.0, 0.0, "call")) == pytest.approx(10.0)
    assert float(bs_price(100.0, 90.0, 1.0, 0.0, 0.0, "put")) == pytest.approx(0.0)


def test_call_price_is_monotone_in_vol():
    prices = [float(bs_price(100, 100, 1.0, 0.0, s, "call")) for s in (0.1, 0.2, 0.4, 0.8)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_price_respects_no_arbitrage_bounds():
    S0, K, T, r, sigma = 100.0, 110.0, 1.0, 0.02, 0.3
    c = float(bs_price(S0, K, T, r, sigma, "call"))
    assert max(S0 - K * np.exp(-r * T), 0.0) < c < S0


def test_vega_is_positive_and_peaks_near_the_money():
    atm = bs_vega(100, 100, 1.0, 0.0, 0.2)
    otm = bs_vega(100, 150, 1.0, 0.0, 0.2)
    assert atm > otm > 0


def test_implied_vol_inverts_the_pricer():
    """Round trip: price at a known sigma, invert, recover the same sigma."""
    for sigma in (0.08, 0.2, 0.55, 1.2):
        price = float(bs_price(100, 105, 0.5, 0.01, sigma, "call"))
        assert implied_vol(price, 100, 105, 0.5, 0.01, "call") == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_returns_nan_outside_arbitrage_bounds():
    assert np.isnan(implied_vol(200.0, 100, 105, 0.5, 0.0, "call"))
    assert np.isnan(implied_vol(-1.0, 100, 105, 0.5, 0.0, "call"))


# --------------------------------------------------------------------------
# The antithetic standard error (regression test for a real bug)
# --------------------------------------------------------------------------
def test_standard_error_uses_antithetic_pairs():
    """Perfectly anti-correlated pairs average to a constant -> zero error.

    Before this was fixed the estimator treated all N draws as independent, so
    it reported a large standard error for an estimator that is in fact exact.
    """
    values = np.array([1.0, 3.0, 5.0, 3.0, 1.0, -1.0])  # pairs (1,3),(3,1),(5,-1) -> mean 2
    assert _standard_error(values, antithetic=True) == pytest.approx(0.0, abs=1e-12)
    assert _standard_error(values, antithetic=False) > 0.5


def test_standard_error_falls_back_for_odd_counts():
    values = np.array([1.0, 2.0, 3.0])
    expected = values.std(ddof=1) / np.sqrt(3)
    assert _standard_error(values, antithetic=True) == pytest.approx(expected)


def test_antithetic_actually_reduces_the_error():
    """The whole point of the technique — must show up in the reported SE."""
    gbm = GBM(S0=100, r=0.02, T=1.0, sigma=0.2)
    plain = price_european(gbm, 105, "call", n_paths=200_000, seed=2,
                           antithetic=False, control_variate=False)
    anti = price_european(gbm, 105, "call", n_paths=200_000, seed=2,
                          antithetic=True, control_variate=False)
    assert anti.std_error < plain.std_error


def test_control_variate_reduces_the_error_more():
    gbm = GBM(S0=100, r=0.02, T=1.0, sigma=0.2)
    plain = price_european(gbm, 105, "call", n_paths=200_000, seed=2,
                           antithetic=False, control_variate=False)
    cv = price_european(gbm, 105, "call", n_paths=200_000, seed=2,
                        antithetic=False, control_variate=True)
    assert plain.std_error / cv.std_error > 1.5


# --------------------------------------------------------------------------
# Monte Carlo pricer vs the closed form
# --------------------------------------------------------------------------
def test_gbm_monte_carlo_matches_black_scholes():
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.02, 0.2
    gbm = GBM(S0=S0, r=r, T=T, sigma=sigma)
    est = price_european(gbm, K, "call", n_paths=200_000, seed=11)
    exact = float(bs_price(S0, K, T, r, sigma, "call"))
    assert abs(est.price - exact) < 4 * est.std_error


def test_gbm_terminal_price_is_a_martingale():
    gbm = GBM(S0=100.0, r=0.05, T=2.0, sigma=0.3)
    S_T = gbm.simulate_terminal(200_000, seed=3)
    se = S_T.std(ddof=1) / np.sqrt(len(S_T))
    assert abs(S_T.mean() - 100.0 * np.exp(0.05 * 2.0)) < 4 * se


def test_price_strikes_matches_pricing_one_at_a_time():
    """Common random numbers must not change what each strike is worth."""
    gbm = GBM(S0=100.0, r=0.0, T=1.0, sigma=0.2)
    strikes = [90.0, 100.0, 110.0]
    batch = price_strikes(gbm, strikes, option="call", n_paths=50_000, seed=7)
    for K, est in zip(strikes, batch):
        single = price_european(gbm, K, "call", n_paths=50_000, seed=7)
        assert est.price == pytest.approx(single.price, rel=1e-12)


def test_prices_decrease_with_strike():
    gbm = GBM(S0=100.0, r=0.0, T=1.0, sigma=0.2)
    ests = price_strikes(gbm, [80, 90, 100, 110, 120], option="call",
                         n_paths=100_000, seed=9)
    prices = [e.price for e in ests]
    assert all(b < a for a, b in zip(prices, prices[1:]))


# --------------------------------------------------------------------------
# Rough Bergomi
# --------------------------------------------------------------------------
def test_volterra_variance_matches_the_theoretical_law():
    """Var(Y_t) must equal t^{2H} — this is what the hybrid scheme guarantees."""
    H = 0.15
    rb = RoughBergomi(S0=100, r=0.0, T=1.0, H=H, eta=1.5, rho=-0.7, xi0=0.04, n_steps=100)
    _, v, t = rb.simulate_paths(40_000, seed=1)
    Y = analysis.recover_volterra(rb, v, t)
    rel_err = np.abs(Y.var(axis=0) - t ** (2 * H))[5:] / (t ** (2 * H))[5:]
    assert rel_err.max() < 0.06


def test_roughness_is_recovered_from_the_paths():
    """Estimate H from the simulated increments; it must match what went in."""
    for H in (0.1, 0.35):
        _, fitted = analysis.roughness_scaling(H, n_paths=4_000, n_steps=300, seed=5)
        assert fitted == pytest.approx(H, abs=0.02)


def test_rough_bergomi_is_a_martingale():
    rb = RoughBergomi(S0=100, r=0.0, T=1.0, H=0.1, eta=1.9, rho=-0.9, xi0=0.04, n_steps=100)
    S_T, _, _ = rb.simulate_paths(150_000, seed=3)
    se = S_T.std(ddof=1) / np.sqrt(len(S_T))
    assert abs(S_T.mean() - 100.0) < 4 * se


def test_zero_vol_of_vol_collapses_to_black_scholes():
    """eta -> 0 freezes variance at xi0, so the model must become Black-Scholes."""
    sigma = 0.2
    rb = RoughBergomi(S0=100, r=0.02, T=1.0, H=0.1, eta=1e-8, rho=-0.7,
                      xi0=sigma ** 2, n_steps=100)
    est = price_european(rb, 105.0, "call", n_paths=200_000, seed=4)
    exact = float(bs_price(100, 105, 1.0, 0.02, sigma, "call"))
    assert abs(est.price - exact) < 4 * est.std_error


def test_variance_paths_are_strictly_positive():
    """The exponential form rules out negative variance — unlike naive Heston."""
    rb = RoughBergomi(S0=100, r=0.0, T=1.0, H=0.1, eta=2.5, rho=-0.9, xi0=0.04, n_steps=200)
    _, v, _ = rb.simulate_paths(2_000, seed=6)
    assert (v > 0).all()


def test_expected_variance_is_the_forward_variance():
    """E[v_t] = xi0 by construction; the drift term exists to make that true."""
    rb = RoughBergomi(S0=100, r=0.0, T=1.0, H=0.1, eta=1.5, rho=-0.9, xi0=0.04, n_steps=100)
    _, v, _ = rb.simulate_paths(80_000, seed=8)
    assert v.mean(axis=0) == pytest.approx(np.full(v.shape[1], 0.04), rel=0.06)


def test_antithetic_paths_come_in_mirrored_pairs():
    """The pairing the standard-error calculation relies on must really exist."""
    rb = RoughBergomi(S0=100, r=0.0, T=1.0, H=0.2, eta=1.0, rho=-0.5,
                      xi0=0.04, n_steps=50)
    _, v, _ = rb.simulate_paths(1_000, seed=12, antithetic=True)
    half = 500
    # Y is antisymmetric across the pairing, so log(v) deviations must mirror.
    Y = np.log(v / 0.04)
    drift = Y.mean(axis=0)
    assert np.allclose((Y[:half] - drift) + (Y[half:] - drift), 0.0, atol=1e-9)


def test_rough_model_has_a_steeper_short_skew_than_a_smooth_one():
    """The economic claim of the whole project, as a test."""
    def factory(H):
        return lambda T: RoughBergomi(S0=100.0, r=0.0, T=T, H=H, eta=1.9,
                                      rho=-0.9, xi0=0.04, n_steps=max(50, int(200 * T)))

    from roughvol.smile import atm_skew
    rough = abs(atm_skew(factory(0.1), 0.1, n_paths=150_000, seed=21))
    smooth = abs(atm_skew(factory(0.45), 0.1, n_paths=150_000, seed=21))
    assert rough > smooth


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def test_same_seed_gives_identical_output():
    rb = RoughBergomi(S0=100, r=0.0, T=1.0, H=0.1, eta=1.9, rho=-0.9, xi0=0.04, n_steps=50)
    a, _, _ = rb.simulate_paths(5_000, seed=99)
    b, _, _ = rb.simulate_paths(5_000, seed=99)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_output():
    rb = RoughBergomi(S0=100, r=0.0, T=1.0, H=0.1, eta=1.9, rho=-0.9, xi0=0.04, n_steps=50)
    a, _, _ = rb.simulate_paths(5_000, seed=1)
    b, _, _ = rb.simulate_paths(5_000, seed=2)
    assert not np.array_equal(a, b)
