"""Volatility estimators: realized, EWMA, and a GARCH(1,1) fitted by MLE.

Three estimators with genuinely different assumptions, kept side by side
because the regime study needs a defensible answer to "how volatile was the
market on day t" and the three disagree in informative ways:

- **Realized** — a rolling standard deviation. No model, but it reacts to a
  shock only gradually and then, ``window`` days later, drops it abruptly.
- **EWMA** — the RiskMetrics recursion, an exponentially weighted variance
  with no mean-reversion term. One parameter, no fitting.
- **GARCH(1,1)** — adds mean reversion toward a long-run variance, so a shock
  decays at a rate the data chooses rather than one that is assumed.

All three are causal: the value at ``t`` uses only data up to and including
``t``. That is not a stylistic preference — the regime labels feed a
performance attribution, and a two-sided filter would leak the future into it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

TRADING_DAYS = 252


def realized_vol(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Rolling standard deviation of returns."""
    vol = returns.rolling(window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol.rename("realized_vol")


def ewma_vol(
    returns: pd.Series,
    lam: float = 0.94,
    annualize: bool = True,
    init: int = 21,
) -> pd.Series:
    """RiskMetrics exponentially weighted volatility.

    ``sigma2_t = lam*sigma2_{t-1} + (1-lam)*r_{t-1}^2``. The lagged squared
    return is deliberate: ``sigma_t`` is the volatility *forecast* for day
    ``t`` made at the close of ``t-1``, so it can be used to classify day
    ``t`` without seeing it.

    ``lam=0.94`` is the RiskMetrics daily default (a ~ 25-day half-life).

    The recursion has to start somewhere, and where it starts is not a detail.
    Seeding it with the variance of the whole series — the obvious choice, and
    a common one — makes every value in the output depend on every return in
    the input, including returns from years later. The series would then look
    causal, carry a docstring saying it was causal, and quietly leak the
    future into the regime labels built from it. Instead the recursion is
    seeded from the first ``init`` observations only, and the days inside that
    window are returned as NaN because no causal estimate exists for them.

    ``lam=1`` and ``lam=0`` are rejected: the first never updates, the second
    never remembers.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    if init < 2:
        raise ValueError("init must cover at least 2 observations")

    r = np.asarray(returns, dtype=float)
    n = len(r)
    var = np.full(n, np.nan, dtype=float)

    if n >= init:
        var[init - 1] = np.nanvar(r[:init], ddof=1)
        for t in range(init, n):
            prev = r[t - 1] if np.isfinite(r[t - 1]) else 0.0
            var[t] = lam * var[t - 1] + (1.0 - lam) * prev**2

    out = np.sqrt(var)
    if annualize:
        out = out * np.sqrt(TRADING_DAYS)
    return pd.Series(out, index=returns.index, name="ewma_vol")


# --------------------------------------------------------------------------
# GARCH(1,1)
# --------------------------------------------------------------------------


@dataclass
class GarchResult:
    """Fitted GARCH(1,1) parameters and the conditional volatility path."""

    omega: float
    alpha: float
    beta: float
    mu: float
    loglik: float
    sigma: pd.Series          # conditional volatility, same units as the input
    converged: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_var(self) -> float:
        """Unconditional variance ``omega / (1 - alpha - beta)``."""
        p = self.persistence
        return np.inf if p >= 1.0 else self.omega / (1.0 - p)

    @property
    def half_life(self) -> float:
        """Days for a variance shock to decay halfway to the long-run level."""
        p = self.persistence
        return np.inf if p >= 1.0 else float(np.log(0.5) / np.log(p))


def _garch_recursion(
    eps: np.ndarray, omega: float, alpha: float, beta: float
) -> np.ndarray:
    """Conditional variance path. ``sigma2_0`` is seeded at the sample variance."""
    n = len(eps)
    sigma2 = np.empty(n, dtype=float)
    sigma2[0] = eps.var(ddof=1)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def _garch_negloglik(params: np.ndarray, x: np.ndarray) -> float:
    mu, omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e12
    eps = x - mu
    sigma2 = _garch_recursion(eps, omega, alpha, beta)
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e12
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
    return -ll if np.isfinite(ll) else 1e12


def fit_garch11(
    returns: pd.Series,
    annualize: bool = True,
) -> GarchResult:
    """Fit a Gaussian GARCH(1,1) by maximum likelihood.

    Model: ``r_t = mu + eps_t``, ``eps_t = sigma_t z_t``, ``z_t ~ N(0,1)``,
    ``sigma2_t = omega + alpha*eps_{t-1}^2 + beta*sigma2_{t-1}``.

    Daily returns are rescaled to percent before optimising and the parameters
    are scaled back afterwards. This is not cosmetic: on decimal returns
    ``omega`` is order ``1e-6`` while ``alpha`` and ``beta`` are order ``0.1``,
    and the six orders of magnitude between them make the numerical gradient
    of the likelihood useless. In percent units every parameter is order
    ``0.01`` to ``1``.

    Constraints ``omega > 0``, ``alpha, beta >= 0`` and ``alpha + beta < 1``
    are imposed directly: the last one is what makes the variance process
    stationary, and without it the optimiser will happily wander into an
    explosive fit that has a higher in-sample likelihood.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        raise ValueError(f"need at least 100 observations, got {len(r)}")

    scale = 100.0
    x = r * scale
    var0 = x.var(ddof=1)

    # Several starts: the likelihood is flat along the alpha+beta ridge, and a
    # single start lands on whichever end of it the initial guess was nearest.
    starts = [
        (x.mean(), var0 * 0.05, 0.05, 0.90),
        (x.mean(), var0 * 0.20, 0.10, 0.70),
        (x.mean(), var0 * 0.01, 0.02, 0.97),
    ]
    bounds = [(None, None), (1e-10, None), (0.0, 1.0), (0.0, 1.0)]
    constraint = {"type": "ineq", "fun": lambda p: 0.9999 - p[2] - p[3]}

    best = None
    for p0 in starts:
        res = optimize.minimize(
            _garch_negloglik,
            np.array(p0, dtype=float),
            args=(x,),
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if best is None or res.fun < best.fun:
            best = res

    assert best is not None
    mu_s, omega_s, alpha, beta = best.x
    sigma2_s = _garch_recursion(x - mu_s, omega_s, alpha, beta)

    sigma = np.sqrt(sigma2_s) / scale
    if annualize:
        sigma = sigma * np.sqrt(TRADING_DAYS)

    return GarchResult(
        omega=float(omega_s / scale**2),
        alpha=float(alpha),
        beta=float(beta),
        mu=float(mu_s / scale),
        loglik=float(-best.fun),
        sigma=pd.Series(sigma, index=returns.index[-len(r):], name="garch_vol"),
        converged=bool(best.success),
    )


def simulate_garch11(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    mu: float = 0.0,
    seed: int = 0,
    burn: int = 500,
) -> np.ndarray:
    """Simulate a GARCH(1,1) path. Used to check that the fit recovers truth."""
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be < 1 for a stationary process")
    rng = np.random.default_rng(seed)
    total = n + burn
    z = rng.standard_normal(total)
    eps = np.empty(total)
    sigma2 = omega / (1.0 - alpha - beta)
    for t in range(total):
        eps[t] = np.sqrt(sigma2) * z[t]
        sigma2 = omega + alpha * eps[t] ** 2 + beta * sigma2
    return mu + eps[burn:]
