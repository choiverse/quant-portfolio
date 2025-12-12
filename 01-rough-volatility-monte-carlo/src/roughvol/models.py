"""Risk-neutral asset-price models as simulatable objects.

Each model exposes ``simulate_terminal(n_paths, seed, antithetic)`` returning an
array of terminal prices S_T under the risk-neutral measure, which the Monte
Carlo pricer turns into option values. The rough Bergomi model additionally
exposes its simulated variance paths for diagnostics.

Models
------
- GBM            : Black-Scholes geometric Brownian motion (validation anchor)
- RoughBergomi   : rough stochastic volatility (Bayer-Friz-Gatheral 2016),
                   simulated with the hybrid scheme (Bennedsen-Lunde-Pakkanen
                   2017 / McCrickerd-Pakkanen 2018), kappa = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


class Model:
    """Base class: a risk-neutral price process on [0, T]."""

    S0: float
    r: float
    T: float

    def simulate_terminal(self, n_paths, seed=None, antithetic=True):
        raise NotImplementedError


@dataclass
class GBM(Model):
    """Geometric Brownian motion: dS = r S dt + sigma S dW."""

    S0: float
    r: float
    T: float
    sigma: float

    def simulate_terminal(self, n_paths, seed=None, antithetic=True):
        rng = np.random.default_rng(seed)
        z = _draw_normals(rng, n_paths, antithetic)
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * z
        return self.S0 * np.exp(drift + diffusion)


@dataclass
class RoughBergomi(Model):
    """Rough Bergomi model.

        v_t = xi0 * exp( eta * Y_t - 0.5 * eta^2 * t^{2H} ),   Var(Y_t) = t^{2H}
        dS_t / S_t = r dt + sqrt(v_t) dZ_t,   dZ = rho dW + sqrt(1-rho^2) dW^perp

    where ``Y`` is the (normalized) Riemann-Liouville fractional process driven
    by the same ``W`` as the price. ``H`` in (0, 1/2) makes volatility "rough".
    """

    S0: float
    r: float
    T: float
    H: float
    eta: float
    rho: float
    xi0: float          # flat forward variance (= v0)
    n_steps: int = 100

    def __post_init__(self):
        self.alpha = self.H - 0.5
        self.dt = self.T / self.n_steps

    # -- hybrid-scheme building blocks -------------------------------------
    def _cov_last_cell(self):
        """2x2 covariance of (plain increment dW, weighted increment W2) over
        one cell, where W2 = int_0^dt (dt - s)^alpha dW_s."""
        a, dt = self.alpha, self.dt
        c11 = dt
        c22 = dt ** (2 * a + 1) / (2 * a + 1)
        c12 = dt ** (a + 1) / (a + 1)
        return np.array([[c11, c12], [c12, c22]])

    def _optimal_weights(self):
        """Kernel weights g_k = (b*_k * dt)^alpha for cells k = 2..n that are
        more than one step from the evaluation point (optimal discretization
        point b*_k of Bennedsen-Lunde-Pakkanen)."""
        a, dt, n = self.alpha, self.dt, self.n_steps
        k = np.arange(2, n + 1)
        b_star = ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1.0 / a)
        return (b_star * dt) ** a  # length n-1, aligns to lags 2..n

    def _simulate_volterra(self, dW1, W2):
        """Build the normalized Volterra process Y at grid times t_1..t_n.

        ``dW1`` (plain increments) and ``W2`` (weighted last-cell increments)
        each have shape (n_paths, n_steps). Returns Y with the same shape and
        Var(Y_i) -> t_i^{2H} by construction.
        """
        g = self._optimal_weights()                    # weights for lags 2..n
        kernel = np.concatenate(([0.0], g))            # index 0 -> lag1 (=0, in W2)
        # Causal convolution of the plain increments with the kernel, vectorized
        # over all paths at once via FFT (O(n log n) per path instead of a
        # Python-level loop).
        conv = fftconvolve(dW1, kernel[None, :], mode="full", axes=1)[:, : self.n_steps]
        Y_hat = W2 + conv                              # raw Volterra
        return np.sqrt(2 * self.H) * Y_hat             # normalize -> Var=t^{2H}

    def simulate_paths(self, n_paths, seed=None, antithetic=True):
        """Return (S_T, variance_paths, time_grid) for diagnostics/pricing."""
        rng = np.random.default_rng(seed)
        n = self.n_steps

        if antithetic:
            half = (n_paths + 1) // 2
            base = rng.standard_normal((half, n, 2))
            perp = rng.standard_normal((half, n))
            base = np.concatenate([base, -base], axis=0)[:n_paths]
            perp = np.concatenate([perp, -perp], axis=0)[:n_paths]
        else:
            base = rng.standard_normal((n_paths, n, 2))
            perp = rng.standard_normal((n_paths, n))

        # Correlate the two Gaussians in each cell via Cholesky of the 2x2 cov.
        L = np.linalg.cholesky(self._cov_last_cell())
        corr = base @ L.T
        dW1 = corr[:, :, 0]                             # plain increment ~ N(0,dt)
        W2 = corr[:, :, 1]                              # weighted last-cell increment

        Y = self._simulate_volterra(dW1, W2)           # (n_paths, n) at t_1..t_n

        t = np.arange(1, n + 1) * self.dt
        v = self.xi0 * np.exp(self.eta * Y - 0.5 * self.eta**2 * t ** (2 * self.H))

        # Variance is predictable over each step: use left endpoint (v0 = xi0).
        v_left = np.concatenate([np.full((n_paths, 1), self.xi0), v[:, :-1]], axis=1)

        # Price BM increment on each cell, correlated with dW1.
        dB = np.sqrt(self.dt) * perp
        dZ = self.rho * dW1 + np.sqrt(1 - self.rho**2) * dB

        log_incr = (self.r - 0.5 * v_left) * self.dt + np.sqrt(v_left) * dZ
        logS = np.log(self.S0) + np.cumsum(log_incr, axis=1)
        S = np.exp(logS)
        S_T = S[:, -1]
        return S_T, v, t

    def simulate_terminal(self, n_paths, seed=None, antithetic=True):
        S_T, _, _ = self.simulate_paths(n_paths, seed, antithetic)
        return S_T


def _draw_normals(rng, n_paths, antithetic):
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal(half)
        return np.concatenate([z, -z])[:n_paths]
    return rng.standard_normal(n_paths)
