"""Is it alpha, or is it a factor you could have bought?

A long/short book built from a machine-learned signal produces a return series.
The question this module answers is what that series *is*. Regressing it on the
principal components of the same universe splits it in two:

    r_t = alpha + sum_k beta_k * F_kt + e_t

``beta_k * F_kt`` is the part a portfolio of the statistical factors would have
delivered — the model discovered an exposure, not an edge. ``alpha`` is what
remains. A strategy whose return is 90% explained by PC1 has not found
anything: it has found leverage on the market, at a cost of 40 bps a year in
trading.

Standard errors are Newey-West, and that is not a formality. The book here is
staggered over a 5-day horizon by construction, so its daily returns are
autocorrelated by design; ordinary OLS standard errors on such a series
understate the uncertainty by a factor that grows with the overlap, and an
alpha t-statistic of 2.5 can become 1.4 once the correlation is accounted for.
The whole point of computing a t-statistic here is to decide whether to believe
the alpha, so it has to be the right one.

Reference
---------
- Newey, W. and West, K. (1987). *A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.*
  Econometrica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class RegressionResult:
    """OLS fit with heteroskedasticity- and autocorrelation-robust errors."""

    names: list[str]
    params: np.ndarray
    stderr: np.ndarray
    tstat: np.ndarray
    r_squared: float
    n_obs: int
    lags: int

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coef": self.params, "std err": self.stderr, "t": self.tstat},
            index=self.names,
        )

    @property
    def alpha_daily(self) -> float:
        return float(self.params[0])

    @property
    def alpha_annual(self) -> float:
        """Annualized intercept. Additive, not compounded — it is a mean."""
        return float(self.params[0]) * TRADING_DAYS

    @property
    def alpha_t(self) -> float:
        return float(self.tstat[0])


def newey_west_ols(
    y: np.ndarray,
    X: np.ndarray,
    names: list[str] | None = None,
    lags: int | None = None,
) -> RegressionResult:
    """OLS with a Newey-West covariance matrix. An intercept is added.

    ``lags=None`` uses the common ``floor(4*(n/100)^(2/9))`` rule of thumb.
    The Bartlett weights ``1 - l/(L+1)`` are what make the estimator positive
    semi-definite; a plain truncated sum is not, and can produce a negative
    variance on exactly the kind of series being fitted here.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(y):
        X = X.T
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    k = Xd.shape[1]

    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lags = max(0, min(lags, n - 1))

    xtx_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = xtx_inv @ (Xd.T @ y)
    resid = y - Xd @ beta

    u = Xd * resid[:, None]
    s = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = u[lag:].T @ u[:-lag]
        s += w * (gamma + gamma.T)

    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    labels = ["alpha"] + (names or [f"x{i + 1}" for i in range(k - 1)])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)

    return RegressionResult(
        names=labels, params=beta, stderr=se, tstat=t,
        r_squared=r2, n_obs=n, lags=lags,
    )


def factor_attribution(
    strategy: pd.Series,
    factors: pd.DataFrame,
    lags: int | None = None,
) -> RegressionResult:
    """Regress a strategy's daily returns on the factor return panel."""
    df = pd.concat([strategy.rename("y"), factors], axis=1).dropna()
    if df.empty:
        raise ValueError("no overlapping dates between strategy and factors")
    return newey_west_ols(
        df["y"].to_numpy(),
        df.drop(columns="y").to_numpy(),
        names=list(factors.columns),
        lags=lags,
    )


def decompose(strategy: pd.Series, factors: pd.DataFrame, lags: int | None = None) -> pd.Series:
    """Split the annualized mean return into a factor part and a residual part.

    The two components sum to the total by construction — this is an exact
    decomposition of the realized mean, not an approximation — so the table it
    produces cannot quietly fail to add up.
    """
    df = pd.concat([strategy.rename("y"), factors], axis=1).dropna()
    res = factor_attribution(df["y"], df.drop(columns="y"), lags=lags)

    betas = res.params[1:]
    factor_part = float((df.drop(columns="y").to_numpy() @ betas).mean()) * TRADING_DAYS
    total = float(df["y"].mean()) * TRADING_DAYS

    return pd.Series(
        {
            "Total ann. return": total,
            "Factor-explained": factor_part,
            "Residual alpha": res.alpha_annual,
            "Alpha t-stat (NW)": res.alpha_t,
            "R-squared": res.r_squared,
            "NW lags": float(res.lags),
        },
        name=strategy.name or "strategy",
    )


def rolling_exposure(
    strategy: pd.Series,
    factor: pd.Series,
    window: int = 126,
) -> pd.Series:
    """Rolling beta of the strategy to one factor.

    A single full-sample beta hides the case that matters most: a book whose
    market exposure is zero on average because it was long in one half of the
    sample and short in the other. That book is not market-neutral, it is
    market-timing, and it should not be described as the former.
    """
    cov = strategy.rolling(window, min_periods=window).cov(factor)
    var = factor.rolling(window, min_periods=window).var()
    return (cov / var.replace(0, np.nan)).rename(f"beta_{factor.name}")


def compare_models(
    results: dict[str, pd.Series],
    factors: pd.DataFrame,
    lags: int | None = None,
) -> pd.DataFrame:
    """Run the decomposition for several strategies and stack the answers."""
    return pd.DataFrame({name: decompose(r, factors, lags=lags) for name, r in results.items()}).T
