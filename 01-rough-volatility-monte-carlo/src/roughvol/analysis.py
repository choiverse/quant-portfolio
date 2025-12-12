"""Measurements that turn the simulator into evidence.

The engine is only interesting if its output can be *checked*. Each function here
extracts a quantity whose true value is known from theory, so the simulated
result can be held against it:

- :func:`roughness_scaling` recovers the Hurst exponent back out of the simulated
  paths (the scheme is told ``H``; the paths must independently exhibit it).
- :func:`convergence_study` measures how the Monte Carlo standard error falls
  with the path count, which must be the textbook ``N^{-1/2}``, and by how much
  each variance-reduction technique shifts that line down.
- :func:`implied_vol_surface` builds the full ``(maturity, moneyness)`` grid.
- :func:`skew_sensitivity` sweeps one model parameter at a time, which is how you
  find out which knob actually controls the short-dated skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .black_scholes import bs_price, implied_vol
from .mc_pricer import price_european, price_strikes
from .models import RoughBergomi
from .smile import atm_skew


def recover_volterra(model: RoughBergomi, variance_paths: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Invert ``v_t = xi0 exp(eta Y_t - 0.5 eta^2 t^{2H})`` back to ``Y``."""
    return (np.log(variance_paths / model.xi0) + 0.5 * model.eta ** 2 * t ** (2 * model.H)) / model.eta


def roughness_scaling(
    H: float,
    n_paths: int = 20_000,
    n_steps: int = 400,
    T: float = 1.0,
    max_lag: int = 60,
    seed: int = 5,
) -> tuple[pd.DataFrame, float]:
    """Measure the roughness of the *simulated* paths, independent of the input ``H``.

    For a process with ``Var(Y_t) = t^{2H}`` the second moment of an increment
    scales as ``E[(Y_{t+Δ} − Y_t)^2] ∝ Δ^{2H}``, so a log-log regression of the
    mean squared increment on the lag has slope ``2H``. Fitting that slope is a
    genuine test: it reads the Hurst exponent off the output rather than trusting
    the parameter that went in.

    Returns ``(DataFrame[lag, mean_sq_increment], fitted_H)``.
    """
    model = RoughBergomi(S0=100.0, r=0.0, T=T, H=H, eta=1.0, rho=-0.9,
                         xi0=0.04, n_steps=n_steps)
    _, v, t = model.simulate_paths(n_paths, seed=seed, antithetic=False)
    Y = recover_volterra(model, v, t)

    dt = model.dt
    lags = np.unique(np.round(np.geomspace(1, max_lag, 24)).astype(int))
    rows = []
    for lag in lags:
        incr = Y[:, lag:] - Y[:, :-lag]
        rows.append({"lag": lag, "delta": lag * dt, "msq": float((incr ** 2).mean())})
    df = pd.DataFrame(rows).set_index("lag")

    slope = np.polyfit(np.log(df["delta"]), np.log(df["msq"]), 1)[0]
    return df, float(slope / 2.0)


def convergence_study(
    model,
    K: float,
    path_counts=(2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000),
    seed: int = 13,
) -> pd.DataFrame:
    """Standard error vs path count for four estimator variants.

    ``plain`` is crude Monte Carlo; ``antithetic`` adds mirrored draws;
    ``control variate`` regresses the payoff on the discounted terminal price
    (whose mean is known exactly); the last combines both. All four must fall as
    ``N^{-1/2}`` — only the intercept should differ.

    The fourth variant is included precisely because the combination is *not*
    the best of both. The control variate removes the component of the payoff
    that is linear in the terminal price, and that is the same odd-in-``z``
    component antithetic sampling targets. What survives is an even function of
    ``z``, for which an antithetic pair is positively rather than negatively
    correlated — so pairing on top of the control variate adds variance back.
    """
    variants = {
        "plain": dict(antithetic=False, control_variate=False),
        "antithetic": dict(antithetic=True, control_variate=False),
        "control variate": dict(antithetic=False, control_variate=True),
        "antithetic + control variate": dict(antithetic=True, control_variate=True),
    }
    rows = []
    for n in path_counts:
        for label, kw in variants.items():
            est = price_european(model, K, "call", n_paths=n, seed=seed, **kw)
            rows.append({"n_paths": n, "variant": label,
                         "price": est.price, "std_error": est.std_error})
    return pd.DataFrame(rows)


def implied_vol_surface(
    model_factory,
    maturities,
    log_moneyness,
    S0: float = 100.0,
    n_paths: int = 200_000,
    seed: int = 17,
) -> pd.DataFrame:
    """Implied-vol surface on a ``(maturity x log-moneyness)`` grid.

    Each maturity is priced from a single simulation shared across all strikes
    (common random numbers), which keeps the smile smooth for a given ``T``.
    Out-of-the-money contracts are used on each wing, and points whose Monte
    Carlo price is not three standard errors clear of zero are dropped rather
    than inverted into a meaningless implied vol.
    """
    surface = pd.DataFrame(index=pd.Index(maturities, name="T"),
                           columns=pd.Index(log_moneyness, name="log_moneyness"),
                           dtype=float)
    for T in maturities:
        model = model_factory(T)
        F = S0 * np.exp(model.r * T)
        strikes = S0 * np.exp(log_moneyness)
        opts = ["put" if K < F else "call" for K in strikes]
        ests = price_strikes(model, strikes, option=opts, n_paths=n_paths, seed=seed)
        for k, opt, est in zip(log_moneyness, opts, ests):
            K = S0 * np.exp(k)
            if est.price > 3 * est.std_error:
                surface.loc[T, k] = implied_vol(est.price, S0, K, T, model.r, option=opt)
    return surface


def skew_sensitivity(
    base: dict,
    param: str,
    values,
    T: float = 0.1,
    n_paths: int = 150_000,
    seed: int = 23,
) -> pd.DataFrame:
    """ATM skew at a fixed maturity as one model parameter is swept.

    ``base`` holds the reference parameters (``H``, ``eta``, ``rho``, ``xi0``);
    ``param`` names the one that varies. Answers "which knob sets the skew?" —
    which matters because a model that can only fit the skew by distorting the
    volatility level is not usable for calibration.
    """
    rows = []
    for val in values:
        params = dict(base)
        params[param] = val

        def factory(T_, p=params):
            return RoughBergomi(S0=100.0, r=0.0, T=T_, n_steps=max(50, int(200 * T_)), **p)

        skew = atm_skew(factory, T, n_paths=n_paths, seed=seed)
        # Level as well as slope: report the ATM implied vol at the same point.
        model = factory(T)
        est = price_strikes(model, [100.0], option="call", n_paths=n_paths, seed=seed)[0]
        atm_iv = implied_vol(est.price, 100.0, 100.0, T, 0.0, "call")
        rows.append({param: val, "atm_skew": skew, "atm_iv": atm_iv})
    return pd.DataFrame(rows).set_index(param)


def validation_gates(seed_offset: int = 0) -> pd.DataFrame:
    """Run the five correctness gates and return them as a table.

    Mirrors ``scripts/validate.py`` but returns structured results so the same
    numbers can be plotted and committed rather than only printed.
    """
    from .models import GBM

    rows = []

    # 1. Scheme: Var(Y_t) -> t^{2H}.
    H = 0.1
    rb = RoughBergomi(S0=100, r=0.0, T=1.0, H=H, eta=1.5, rho=-0.7, xi0=0.04, n_steps=100)
    _, v, t = rb.simulate_paths(60_000, seed=1 + seed_offset)
    Y = recover_volterra(rb, v, t)
    rel_err = np.abs(Y.var(axis=0) - t ** (2 * H))[5:] / (t ** (2 * H))[5:]
    rows.append({"gate": "Volterra variance -> t^2H", "statistic": rel_err.max(),
                 "tolerance": 0.05, "units": "max rel. error", "passed": rel_err.max() < 0.05})

    # 2. Pricer: GBM Monte Carlo == Black-Scholes.
    S0, r, T, sigma, K = 100, 0.02, 1.0, 0.2, 105
    gbm = GBM(S0=S0, r=r, T=T, sigma=sigma)
    est = price_european(gbm, K, "call", n_paths=400_000, seed=2 + seed_offset)
    bs = float(bs_price(S0, K, T, r, sigma, "call"))
    z = abs(est.price - bs) / est.std_error
    rows.append({"gate": "GBM MC == Black-Scholes", "statistic": z,
                 "tolerance": 3.5, "units": "|z|", "passed": z < 3.5})

    # 3. Efficiency: antithetic sampling alone beats crude Monte Carlo.
    #    Compared like-for-like (both without the control variate), and with the
    #    standard error computed over antithetic *pairs* rather than over the
    #    correlated draws — otherwise the technique appears to do nothing.
    plain = price_european(gbm, K, "call", n_paths=400_000, seed=2 + seed_offset,
                           antithetic=False, control_variate=False)
    anti = price_european(gbm, K, "call", n_paths=400_000, seed=2 + seed_offset,
                          antithetic=True, control_variate=False)
    anti_ratio = plain.std_error / anti.std_error
    rows.append({"gate": "Antithetic cuts SE vs crude MC", "statistic": anti_ratio,
                 "tolerance": 1.1, "units": "SE reduction x", "passed": anti_ratio > 1.1})

    # 4. Efficiency: the control variate is the bigger win, and stands alone.
    cv = price_european(gbm, K, "call", n_paths=400_000, seed=2 + seed_offset,
                        antithetic=False, control_variate=True)
    cv_ratio = plain.std_error / cv.std_error
    rows.append({"gate": "Control variate cuts SE vs crude MC", "statistic": cv_ratio,
                 "tolerance": 1.5, "units": "SE reduction x", "passed": cv_ratio > 1.5})

    # 4. Model: discounted price is a martingale.
    S_T, _, _ = rb.simulate_paths(200_000, seed=3 + seed_offset)
    se = S_T.std(ddof=1) / np.sqrt(len(S_T))
    zm = abs(S_T.mean() - rb.S0 * np.exp(rb.r * rb.T)) / se
    rows.append({"gate": "RB martingale: E[S_T] = S0 e^(rT)", "statistic": zm,
                 "tolerance": 4.0, "units": "|z|", "passed": zm < 4.0})

    # 5. Model: eta -> 0 collapses to Black-Scholes.
    rb0 = RoughBergomi(S0=100, r=r, T=T, H=0.1, eta=1e-6, rho=-0.7, xi0=sigma ** 2, n_steps=100)
    est0 = price_european(rb0, K, "call", n_paths=400_000, seed=4 + seed_offset)
    z0 = abs(est0.price - float(bs_price(S0, K, T, r, sigma, "call"))) / est0.std_error
    rows.append({"gate": "RB eta -> 0 == Black-Scholes", "statistic": z0,
                 "tolerance": 4.0, "units": "|z|", "passed": z0 < 4.0})

    return pd.DataFrame(rows).set_index("gate")
