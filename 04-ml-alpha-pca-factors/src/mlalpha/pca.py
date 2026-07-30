"""Principal component analysis of the return panel, and how much of it is noise.

A cross-section of 470 correlated stocks is not 470 independent bets. Most of
the daily variation is a handful of common movements — the market, and a few
things that look like sectors — and any alpha signal that is really an
exposure to one of those is not alpha. This module extracts that structure and,
crucially, decides **how much of it is real**.

The decision is not made by a scree-plot eyeball or by "keep 80% of variance".
It is made against the Marchenko-Pastur law, which gives the exact limiting
distribution of the eigenvalues of a sample correlation matrix computed from
*pure noise*. With ``N`` assets and ``T`` observations and ``q = N/T`` fixed,
the eigenvalues of a correlation matrix of independent series fill the interval

    [ (1 - sqrt(q))^2 , (1 + sqrt(q))^2 ]

and nothing lies outside it. So an eigenvalue above the upper edge is
structure a random matrix of the same shape could not have produced, and one
inside the bulk is indistinguishable from noise no matter how suggestive the
scree plot looks. This is the single most useful thing in the module: at
``N=470, T=252`` the upper edge sits at 5.6, which is a far higher bar than
"the first twenty components look important".

Everything is computed by SVD of the standardized panel rather than by
eigendecomposition of a formed covariance matrix. Forming ``X'X`` squares the
condition number, and with 470 columns and daily data that is a real loss of
precision on the small eigenvalues — which are exactly the ones being compared
against a theoretical edge.

References
----------
- Marchenko, V. and Pastur, L. (1967). *Distribution of eigenvalues for some
  sets of random matrices.* Mathematics of the USSR-Sbornik.
- Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (1999). *Noise
  dressing of financial correlation matrices.* Physical Review Letters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# The Marchenko-Pastur law
# --------------------------------------------------------------------------


def mp_edges(n_obs: int, n_assets: int, sigma2: float = 1.0) -> tuple[float, float]:
    """The ``(lower, upper)`` support of the Marchenko-Pastur bulk.

    ``sigma2`` is the variance of the underlying noise; it is 1 for a
    correlation matrix by construction, which is why this project standardizes
    before decomposing. The width of the bulk is driven entirely by
    ``q = n_assets / n_obs`` — more assets, or a shorter window, and pure noise
    produces a wider spread of eigenvalues. That is the whole reason a scree
    plot cannot be read without it.
    """
    if n_obs <= 0 or n_assets <= 0:
        raise ValueError("n_obs and n_assets must be positive")
    q = n_assets / n_obs
    root = np.sqrt(q)
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def mp_density(x: np.ndarray, n_obs: int, n_assets: int, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density, evaluated on ``x``. Zero outside the bulk.

    Normalised as a density over the eigenvalues of the correlation matrix, so
    it can be overlaid directly on a histogram of them (with ``density=True``).
    When ``q > 1`` the distribution also carries a point mass at zero — there
    are more assets than observations, so the matrix is singular — and that
    mass is *not* represented here, only the continuous part.
    """
    x = np.asarray(x, dtype=float)
    q = n_assets / n_obs
    lo, hi = mp_edges(n_obs, n_assets, sigma2)
    out = np.zeros_like(x)
    inside = (x > lo) & (x < hi)
    xi = x[inside]
    out[inside] = np.sqrt((hi - xi) * (xi - lo)) / (2.0 * np.pi * sigma2 * q * xi)
    return out


def n_significant(
    eigenvalues: np.ndarray,
    n_obs: int,
    n_assets: int,
    sigma2: float = 1.0,
) -> int:
    """How many eigenvalues sit above the Marchenko-Pastur upper edge."""
    _, hi = mp_edges(n_obs, n_assets, sigma2)
    return int(np.sum(np.asarray(eigenvalues) > hi))


def fit_noise_variance(
    eigenvalues: np.ndarray,
    n_assets: int,
    n_remove: int = 1,
) -> float:
    """Laloux et al.'s noise variance: the variance left once the market is removed.

    The plain edge assumes the whole correlation matrix is noise, which on
    equity data is badly wrong in a specific direction: the market mode alone
    carries ~30% of the total variance here, and that variance is *not*
    available to the noise bulk. Since the eigenvalues of a correlation matrix
    sum to exactly ``N``, removing the top ``n_remove`` of them and dividing
    the remainder by ``N`` gives the share of variance the bulk actually has:

        sigma2 = (sum of eigenvalues, excluding the top n_remove) / N
               = 1 - lambda_1 / N        (for n_remove = 1)

    which is exactly the correction in Laloux et al. (1999).

    Note the direction, because it is the opposite of what "correcting for
    noise" sounds like: shrinking ``sigma2`` narrows the bulk and *lowers* the
    edge, so **more** components come out significant, not fewer. The
    unadjusted edge is the conservative one — it hands the entire variance of
    the market mode to the noise and then asks what could not be noise, which
    systematically under-counts factors. On this panel the adjustment moves
    the count from 11 to 15 over a 252-day window. Both are reported in the
    write-up rather than one being presented as the answer.

    It is applied once, not iterated to a fixed point. Iterating is tempting
    and wrong: each pass reclassifies more eigenvalues as deviating, which
    lowers ``sigma2``, which narrows the edge, which reclassifies more — and on
    real equity data the recursion does not converge to anything meaningful,
    it collapses until nearly the whole spectrum is called significant. The
    one-step version is the published one and it is the one used here.
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    if len(lam) == 0 or n_assets <= 0:
        return 1.0
    keep = lam[max(0, int(n_remove)):]
    return float(np.clip(keep.sum() / n_assets, 1e-6, 1.0))


# --------------------------------------------------------------------------
# The decomposition
# --------------------------------------------------------------------------


@dataclass
class PCAModel:
    """A fitted PCA: the loadings, the spectrum, and the standardization used.

    ``loadings`` is ``(n_assets, n_components)`` with orthonormal columns.
    ``eigenvalues`` covers the *whole* spectrum, not just the retained
    components, because the discarded tail is what the Marchenko-Pastur
    comparison is about.
    """

    loadings: np.ndarray
    eigenvalues: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    tickers: pd.Index
    n_obs: int
    standardized: bool

    @property
    def n_components(self) -> int:
        return self.loadings.shape[1]

    @property
    def n_assets(self) -> int:
        return self.loadings.shape[0]

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Share of total variance carried by each *retained* component."""
        return self.eigenvalues[: self.n_components] / self.eigenvalues.sum()

    @property
    def loadings_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.loadings,
            index=self.tickers,
            columns=[f"PC{i + 1}" for i in range(self.n_components)],
        )

    def mp_upper_edge(self, sigma2: float | None = None) -> float:
        if sigma2 is None:
            sigma2 = 1.0 if self.standardized else float(np.mean(self.scale ** 2))
        return mp_edges(self.n_obs, self.n_assets, sigma2)[1]

    def noise_variance(self, n_remove: int = 1) -> float:
        return fit_noise_variance(self.eigenvalues, self.n_assets, n_remove=n_remove)

    def n_significant(self, adjust_noise: bool = False) -> int:
        sigma2 = self.noise_variance() if adjust_noise else 1.0
        return n_significant(self.eigenvalues, self.n_obs, self.n_assets, sigma2)

    # -- application to data ------------------------------------------------

    def _rescale(self, returns: pd.DataFrame) -> np.ndarray:
        """Scale a return matrix by the stored volatilities — and *not* centre it.

        This is the single subtlest point in the module, and getting it wrong
        manufactures a signal out of nothing.

        Fitting the model centres the panel, because a covariance matrix is
        defined on centred data. Applying it must not, because the stored mean
        is the average daily return of each stock over the *fitting* window. If
        ``reconstruct`` adds that mean back, then ``residuals`` subtracts it,
        and every stock's idiosyncratic return is quietly reduced by its own
        trailing 252-day drift. A past winner then has a mechanically negative
        forward residual, and any momentum feature "predicts" it with an
        information ratio around −1.0 — on data containing no predictability
        whatsoever. ``validation.gate_residual_is_neutral`` plants exactly that
        null and requires the IC to come back at zero; the earlier, centring
        version of this method fails it at −0.99.

        Removing the centring makes the residual a pure linear function of the
        same day's cross-section, which is what a factor residual is supposed
        to be: ``r_t - B f_t``, no drift term anywhere.
        """
        x = returns.reindex(columns=self.tickers).to_numpy(dtype=float)
        return x / self.scale

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Factor scores: the daily return of each principal component."""
        scores = self._rescale(returns) @ self.loadings
        return pd.DataFrame(
            scores,
            index=returns.index,
            columns=[f"PC{i + 1}" for i in range(self.n_components)],
        )

    def reconstruct(self, returns: pd.DataFrame) -> pd.DataFrame:
        """The part of ``returns`` spanned by the retained components."""
        z = self._rescale(returns)
        approx = (z @ self.loadings) @ self.loadings.T
        return pd.DataFrame(
            approx * self.scale,
            index=returns.index,
            columns=self.tickers,
        )

    def residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """``returns`` with the common factor structure projected out.

        These are the idiosyncratic returns — what is left of a stock once the
        market and the other retained factors have been removed. A signal that
        predicts *these* is predicting something a factor portfolio does not
        already give you for free.
        """
        return returns.reindex(columns=self.tickers) - self.reconstruct(returns)


def _fix_signs(loadings: np.ndarray) -> np.ndarray:
    """Pin the sign of each component so repeated fits are comparable.

    SVD determines each loading vector only up to sign, so an unpinned PC1 can
    flip between two fits of nearly identical data and turn a stable factor
    exposure into a series that changes sign at a window boundary. The
    convention here: the sum of the loadings is non-negative, which makes PC1
    point the same way as the market rather than against it. For components
    whose loadings sum to ~0 by construction — every component after the
    first, being orthogonal to something market-like — fall back to fixing the
    sign of the largest-magnitude loading, which is well defined.
    """
    out = loadings.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        total = col.sum()
        ref = total if abs(total) > 1e-8 * np.abs(col).sum() else col[np.argmax(np.abs(col))]
        if ref < 0:
            out[:, j] = -col
    return out


def fit_pca(
    returns: pd.DataFrame,
    n_components: int | None = None,
    standardize: bool = True,
) -> PCAModel:
    """Decompose a ``(dates x tickers)`` return matrix by SVD.

    With ``standardize=True`` (the default) this is a PCA of the *correlation*
    matrix. That is the right choice for a factor model of equity returns: on
    the covariance matrix, a handful of high-volatility names dominate the
    leading components purely because they are volatile, and the "factors" that
    come out are largely a list of the noisiest stocks. It also makes the
    Marchenko-Pastur comparison exact, because the theoretical edge is stated
    for unit-variance data.

    ``n_components=None`` keeps the full spectrum.
    """
    if returns.isna().to_numpy().any():
        raise ValueError("fit_pca requires a complete return matrix (no NaNs)")
    x = returns.to_numpy(dtype=float)
    n_obs, n_assets = x.shape
    if n_obs < 2:
        raise ValueError("need at least 2 observations")

    mean = x.mean(axis=0)
    if standardize:
        scale = x.std(axis=0, ddof=1)
        scale = np.where(scale > 0, scale, 1.0)
    else:
        scale = np.ones(n_assets)
    z = (x - mean) / scale

    # Eigenvalues of the (co)variance matrix are the squared singular values of
    # the panel scaled by 1/sqrt(T-1) — obtained without ever forming Z'Z.
    _, sv, vt = np.linalg.svd(z / np.sqrt(n_obs - 1), full_matrices=False)
    eigenvalues = sv ** 2
    loadings = _fix_signs(vt.T)

    k = n_assets if n_components is None else int(n_components)
    k = max(1, min(k, loadings.shape[1]))

    return PCAModel(
        loadings=loadings[:, :k],
        eigenvalues=eigenvalues,
        mean=mean,
        scale=scale,
        tickers=returns.columns,
        n_obs=n_obs,
        standardized=standardize,
    )


def fit_significant(returns: pd.DataFrame, adjust_noise: bool = True) -> PCAModel:
    """Fit a PCA keeping exactly the components the MP edge calls significant.

    The number of factors is therefore chosen by the data and a theorem rather
    than by hand. At least one component is always kept: a universe of equities
    with no common mode at all would be a finding about the universe, not about
    the method, and silently returning a zero-factor model would hide it.
    """
    full = fit_pca(returns, n_components=None, standardize=True)
    k = max(1, full.n_significant(adjust_noise=adjust_noise))
    return fit_pca(returns, n_components=k, standardize=True)


# --------------------------------------------------------------------------
# Interpreting the factors
# --------------------------------------------------------------------------


def factor_portfolio_returns(returns: pd.DataFrame, model: PCAModel) -> pd.DataFrame:
    """Factor scores rescaled to read as the return of a tradable portfolio.

    A raw factor score is in units of standardized returns and has no natural
    size. Rescaling the implied weights to gross exposure 1 makes the series
    directly comparable with the strategy returns it is later regressed
    against, so an R^2 against PC1 means "this much of the strategy is the
    market" rather than "this much of the strategy is an unnamed multiple of
    the market".
    """
    weights = model.loadings / model.scale[:, None]
    gross = np.abs(weights).sum(axis=0)
    gross = np.where(gross > 0, gross, 1.0)
    x = returns.reindex(columns=model.tickers).to_numpy(dtype=float)
    return pd.DataFrame(
        x @ (weights / gross),
        index=returns.index,
        columns=[f"PC{i + 1}" for i in range(model.n_components)],
    )


def market_correlation(returns: pd.DataFrame, model: PCAModel) -> pd.Series:
    """Correlation of each factor score with the equal-weight market return.

    The check that PC1 is what everyone assumes it is. It usually comes back
    above 0.98, and that is worth stating rather than assuming: it means the
    first component is not an alpha factor, it is beta, and a "signal" that
    loads on it is a leveraged index position.
    """
    scores = model.transform(returns)
    mkt = returns.reindex(columns=model.tickers).mean(axis=1)
    return scores.corrwith(mkt)


def rolling_residuals(
    returns: pd.DataFrame,
    window: int = 252,
    step: int = 21,
    n_components: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Causally residualize the panel: fit on the trailing window, apply forward.

    Returns ``(residual_returns, n_factors_used)``. The model in force on day
    ``t`` was estimated on ``[t-window, t)`` and refreshed every ``step`` days;
    no return from ``t`` onward touches the loadings applied to ``t``. Fitting
    one PCA on the whole sample and calling the leftovers "idiosyncratic" is
    the standard shortcut here, and it leaks: the loadings are chosen partly to
    fit the very returns being residualized, which shrinks the residuals toward
    zero exactly where the model happens to fit best.

    The first ``window`` rows have no fitted model and are returned as ``NaN``.
    """
    idx = returns.index
    resid = pd.DataFrame(np.nan, index=idx, columns=returns.columns)
    counts = pd.Series(np.nan, index=idx, dtype=float)

    start = window
    while start < len(idx):
        train = returns.iloc[start - window: start]
        model = (
            fit_pca(train, n_components=n_components)
            if n_components is not None
            else fit_significant(train)
        )
        stop = min(start + step, len(idx))
        block = returns.iloc[start:stop]
        resid.iloc[start:stop] = model.residuals(block).to_numpy()
        counts.iloc[start:stop] = model.n_components
        start = stop

    return resid, counts
