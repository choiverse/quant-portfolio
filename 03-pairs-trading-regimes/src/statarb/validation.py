"""Correctness gates: every estimator in the package, checked against a known answer.

Each gate constructs data whose true answer is known by construction — a
published critical value, a simulated process with the parameters written
down, an algebraic identity — and compares the estimator's output against it.
Nothing here reads market data, so the whole suite runs in CI without the
28 MB panel and fails the build if the maths breaks.

Gate 6 deserves a note, because its tolerance looks loose. It checks GARCH
*persistence* (``alpha + beta``) rather than the two parameters separately.
That is not to make the gate easier to pass: repeated fits on simulated paths
with 4,000 observations recover ``alpha + beta = 0.9794 +- 0.0039`` against a
true 0.98, while ``alpha`` itself lands at ``0.086 +- 0.004`` against a true
0.08 and only tightens slowly with sample size. The likelihood is nearly flat
along the ridge that trades ``alpha`` against ``beta``, so the split is weakly
identified at any sample size a daily equity series can offer. A gate on
``alpha`` alone would be testing luck; the persistence is the quantity the
data actually pins down, and it is also the one the volatility half-life
depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import cointegration as coint
from . import regimes as rg
from . import strategy
from . import volatility as vol


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
# Gate 1-3: the unit-root machinery
# --------------------------------------------------------------------------


def gate_adf_critical_values(n_reps: int = 40_000, nobs: int = 1_000) -> Gate:
    """Simulated Dickey-Fuller quantiles must reproduce MacKinnon (2010).

    This is the load-bearing gate. If the ADF regression, the deterministic
    handling or the residual-based adjustment were wrong, the simulated null
    would sit somewhere other than the published one and every cointegration
    p-value in the project would be quietly mis-scaled.
    """
    worst, where = 0.0, ""
    for regression, n_series in (("c", 1), ("ct", 1), ("c", 2)):
        stats = coint.simulate_df_stats(
            nobs=nobs, n_reps=n_reps, regression=regression,
            n_series=n_series, seed=101,
        )
        published = coint.mackinnon_crit(regression, n_series, nobs)
        for level, crit in published.items():
            diff = abs(float(np.quantile(stats, level)) - crit)
            if diff > worst:
                worst, where = diff, f"{regression}/N={n_series} @ {level:.0%}"
    return Gate(
        name="ADF critical values vs MacKinnon (2010)",
        statistic=worst,
        tolerance=0.05,
        comparator="<",
        units="max abs deviation",
        detail=f"worst case {where}",
    )


def gate_adf_size(n_reps: int = 1_000, nobs: int = 500) -> Gate:
    """On true random walks the test must reject at its nominal 5%, not more.

    A test that over-rejects manufactures cointegration out of noise, which is
    precisely the failure this project is about.

    The gate is expressed in standard errors of the rejection rate rather than
    as a raw deviation, because the rate is itself a Monte Carlo estimate: at
    200 replications one standard error *is* 1.5 percentage points, so a fixed
    tolerance of "within 1.5pp" would fail a perfectly calibrated test a third
    of the time. Scaling by ``sqrt(p(1-p)/n)`` makes the gate mean the same
    thing at every setting.
    """
    rng = np.random.default_rng(202)
    rejects = 0
    for _ in range(n_reps):
        y = np.cumsum(rng.standard_normal(nobs))
        res = coint.adf_test(y, max_lag=4, regression="c", n_series=1)
        rejects += res.rejects(0.05)

    rate = rejects / n_reps
    se = np.sqrt(0.05 * 0.95 / n_reps)
    return Gate(
        name="ADF size on random walks",
        statistic=abs(rate - 0.05) / se,
        tolerance=3.0,
        comparator="<",
        units="sigma from nominal 5%",
        detail=f"rejection rate {rate:.3f} over {n_reps} walks (1 s.e. = {se:.4f})",
    )


def gate_adf_power(n_reps: int = 1_000, nobs: int = 500, phi: float = 0.9) -> Gate:
    """On a stationary AR(1) the test must actually reject."""
    rng = np.random.default_rng(303)
    rejects = 0
    for _ in range(n_reps):
        e = rng.standard_normal(nobs)
        y = np.empty(nobs)
        y[0] = e[0]
        for t in range(1, nobs):
            y[t] = phi * y[t - 1] + e[t]
        res = coint.adf_test(y, max_lag=4, regression="c", n_series=1)
        rejects += res.rejects(0.05)
    rate = rejects / n_reps
    return Gate(
        name=f"ADF power on AR(1), phi={phi}",
        statistic=rate,
        tolerance=0.80,
        comparator=">",
        units="rejection rate",
        detail=f"{rejects}/{n_reps} rejected",
    )


# --------------------------------------------------------------------------
# Gate 4-5: cointegration and the spread
# --------------------------------------------------------------------------


def _cointegrated_pair(
    n: int, beta: float, phi: float, seed: int, alpha: float = 0.5
) -> tuple[pd.Series, pd.Series]:
    """``y = alpha + beta*x + s`` with ``x`` a random walk and ``s`` AR(1)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n)) * 0.01 + 4.0
    s = np.empty(n)
    s[0] = rng.standard_normal() * 0.01
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.standard_normal() * 0.01
    y = alpha + beta * x + s
    idx = pd.RangeIndex(n)
    return pd.Series(y, index=idx, name="y"), pd.Series(x, index=idx, name="x")


def gate_hedge_ratio(n: int = 1_500, beta: float = 1.4, phi: float = 0.95) -> Gate:
    """Engle-Granger must recover a hedge ratio that was put in by hand."""
    y, x = _cointegrated_pair(n, beta=beta, phi=phi, seed=404)
    res = coint.engle_granger(y, x)
    got = res.beta if not res.swapped else 1.0 / res.beta
    return Gate(
        name="Engle-Granger hedge-ratio recovery",
        statistic=abs(got - beta),
        tolerance=0.05,
        comparator="<",
        units="abs error",
        detail=f"true {beta:.3f}, fitted {got:.3f}, EG stat {res.stat:.2f}",
    )


def gate_half_life(n: int = 4_000, phi: float = 0.95) -> Gate:
    """The OU half-life must match the AR(1) coefficient it was generated from."""
    rng = np.random.default_rng(505)
    s = np.empty(n)
    s[0] = rng.standard_normal()
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.standard_normal()
    true_hl = np.log(2.0) / -np.log(phi)
    got = coint.ou_half_life(s)
    return Gate(
        name="OU half-life recovery",
        statistic=abs(got - true_hl) / true_hl,
        tolerance=0.15,
        comparator="<",
        units="relative error",
        detail=f"true {true_hl:.2f}d, fitted {got:.2f}d",
    )


# --------------------------------------------------------------------------
# Gate 6-7: the volatility and regime models
# --------------------------------------------------------------------------


def gate_garch_persistence(n: int = 4_000) -> Gate:
    """MLE must recover the persistence of a simulated GARCH(1,1)."""
    omega, alpha, beta = 0.02, 0.08, 0.90
    x = vol.simulate_garch11(n, omega=omega, alpha=alpha, beta=beta, seed=606) / 100.0
    fit = vol.fit_garch11(pd.Series(x))
    true_p = alpha + beta
    return Gate(
        name="GARCH(1,1) persistence recovery",
        statistic=abs(fit.persistence - true_p),
        tolerance=0.02,
        comparator="<",
        units="abs error",
        detail=(
            f"true {true_p:.4f}, fitted {fit.persistence:.4f} "
            f"(alpha {fit.alpha:.3f} vs {alpha}, beta {fit.beta:.3f} vs {beta})"
        ),
    )


def gate_hmm_recovery(n: int = 3_000) -> Gate:
    """Viterbi must recover the hidden states of a simulated 2-regime path."""
    rng = np.random.default_rng(707)
    A = np.array([[0.98, 0.02], [0.06, 0.94]])
    mu = np.array([0.0005, -0.0010])
    sd = np.array([0.006, 0.020])

    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = rng.choice(2, p=A[state[t - 1]])
    x = rng.normal(mu[state], sd[state])

    res = rg.fit_regimes(pd.Series(x, index=pd.RangeIndex(n)), seed=1)
    acc = float((res.state.to_numpy() == state).mean())
    fitted_vol = np.sqrt(res.model.variances) * np.sqrt(252)
    return Gate(
        name="HMM state recovery",
        statistic=acc,
        tolerance=0.90,
        comparator=">",
        units="classification accuracy",
        detail=(
            f"fitted ann vols {fitted_vol[0]:.3f}/{fitted_vol[1]:.3f} "
            f"vs true {sd[0]*np.sqrt(252):.3f}/{sd[1]*np.sqrt(252):.3f}"
        ),
    )


# --------------------------------------------------------------------------
# Gate 8: look-ahead
# --------------------------------------------------------------------------


def synthetic_panel(
    n_days: int = 800, n_pairs: int = 8, n_noise: int = 8, seed: int = 808
) -> pd.DataFrame:
    """A price panel containing genuine cointegrated pairs plus pure noise.

    Used by the look-ahead gate so it needs no external data, and useful on
    its own: a screen that cannot find the planted pairs, or that "finds"
    cointegration among the noise columns at more than the nominal rate, is
    broken in a way no market backtest would reveal.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    cols: dict[str, np.ndarray] = {}

    for i in range(n_pairs):
        x = np.cumsum(rng.standard_normal(n_days)) * 0.012 + 4.0
        s = np.empty(n_days)
        s[0] = 0.0
        for t in range(1, n_days):
            s[t] = 0.96 * s[t - 1] + rng.standard_normal() * 0.012
        beta = rng.uniform(0.7, 1.3)
        cols[f"P{i}A"] = np.exp(0.2 + beta * x + s)
        cols[f"P{i}B"] = np.exp(x)

    for j in range(n_noise):
        cols[f"N{j}"] = np.exp(np.cumsum(rng.standard_normal(n_days)) * 0.012 + 4.0)

    return pd.DataFrame(cols, index=idx)


def gate_lookahead() -> Gate:
    """Rewriting the last day must leave every earlier day of P&L untouched.

    The strongest single check in the suite. Selection, hedge ratios, spread
    moments and z-scores are all estimated from data, so a single misplaced
    ``shift`` anywhere in that chain lets tomorrow's price influence today's
    position. Re-running the entire pipeline on a panel whose final row has
    been replaced, and demanding bit-identical earlier returns, catches that
    regardless of where in the chain it hides.
    """
    prices = synthetic_panel()
    kwargs = dict(
        formation_days=252, trading_days=126, top_k=60,
        max_pairs=10, max_lag=4,
    )
    base, _ = strategy.run_strategy(prices, **kwargs)

    tampered = prices.copy()
    tampered.iloc[-1] = tampered.iloc[-1] * 1.5      # a 50% move on the last day

    after, _ = strategy.run_strategy(tampered, **kwargs)

    a = base.returns.iloc[:-1]
    b = after.returns.iloc[:-1]
    worst = float(np.max(np.abs(a.to_numpy() - b.to_numpy())))
    return Gate(
        name="No look-ahead in the pipeline",
        statistic=worst,
        tolerance=1e-15,
        comparator="<",
        units="max abs P&L difference",
        detail=f"{len(a)} earlier days compared after a +50% shock to the last bar",
    )


# --------------------------------------------------------------------------


GATES: list[Callable[[], Gate]] = [
    gate_adf_critical_values,
    gate_adf_size,
    gate_adf_power,
    gate_hedge_ratio,
    gate_half_life,
    gate_garch_persistence,
    gate_hmm_recovery,
    gate_lookahead,
]


def validation_gates(quick: bool = False) -> pd.DataFrame:
    """Run every gate and return the results as a table.

    ``quick=True`` shrinks the Monte Carlo gates for a fast smoke run. The
    committed result and the CI run both use the full settings; the quick mode
    exists so that a broken edit is caught in seconds rather than a minute.
    """
    results = []
    for fn in GATES:
        if quick and fn is gate_adf_critical_values:
            # Still enough replications that the 1% quantile's own Monte Carlo
            # error stays well inside the tolerance being tested.
            gate = fn(n_reps=20_000, nobs=500)
        elif quick and fn in (gate_adf_size, gate_adf_power):
            gate = fn(n_reps=400, nobs=300)
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
