"""Volatility regimes: a two-state Gaussian HMM, and a simpler baseline.

A regime model answers "was the market calm or turbulent on day t". Two
answers are offered here because the harder one needs something to be checked
against:

- **``GaussianHMM``** — states are latent, with Gaussian emissions whose mean
  and variance are state-specific, fitted by Baum-Welch (EM). Transitions are
  Markov, so the model has an explicit notion of regime *persistence* and
  will not flip on a single bad day.
- **``vol_tercile_regimes``** — bucket an EWMA volatility series by its own
  expanding terciles. No latent structure, no fitting, no persistence. If the
  HMM does not beat this, the HMM is not earning its complexity.

Filtered vs smoothed, and why it matters here
---------------------------------------------
``filter`` returns ``P(state_t | x_1..x_t)`` — only past data. ``smooth``
returns ``P(state_t | x_1..x_T)`` — the whole sample, including the future.
Smoothed labels are sharper and make prettier charts, and they are also
completely unusable for anything that touches a trading decision. The rule
this module follows: **smoothed for description, filtered for attribution.**

Even the filtered path is not fully causal, because the *parameters* were
estimated on the whole sample. ``fit_causal`` handles that case properly by
estimating on a burn-in window and filtering forward from there; the ordinary
``fit`` is honest about being in-sample and is what the descriptive figures
use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_VAR_FLOOR = 1e-12


def _safe_normalize(gamma: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Row-normalise posteriors, falling back where the density underflowed.

    An observation far enough into the tails of *every* state — a 25-sigma
    return, say — has an emission density that underflows to zero in all of
    them, and the row sums to zero. Dividing then produces NaN and poisons the
    rest of the path. Such a point carries no information about which state
    generated it, so the filtered estimate is the right thing to keep.
    """
    total = gamma.sum(axis=1, keepdims=True)
    good = np.isfinite(total) & (total > 0)
    out = np.divide(gamma, total, out=np.zeros_like(gamma), where=good)
    return np.where(good, out, fallback)


@dataclass
class GaussianHMM:
    """Discrete-state, Gaussian-emission hidden Markov model.

    Parameters are ordered so that state 0 always has the *smallest* emission
    variance. EM has no idea which state is "calm" — relabelling the states
    gives an identical likelihood — so without a fixed convention the meaning
    of ``state 1`` would change from run to run and every downstream
    comparison would be nonsense.
    """

    n_states: int = 2
    startprob: np.ndarray = field(default_factory=lambda: np.array([]))
    transmat: np.ndarray = field(default_factory=lambda: np.array([]))
    means: np.ndarray = field(default_factory=lambda: np.array([]))
    variances: np.ndarray = field(default_factory=lambda: np.array([]))
    loglik: float = np.nan
    n_iter: int = 0
    converged: bool = False

    # -- emissions ---------------------------------------------------------

    def _emission(self, x: np.ndarray) -> np.ndarray:
        """Emission densities, shape (T, n_states)."""
        d = x[:, None] - self.means[None, :]
        var = np.maximum(self.variances, _VAR_FLOOR)[None, :]
        return np.exp(-0.5 * d**2 / var) / np.sqrt(2.0 * np.pi * var)

    # -- inference ---------------------------------------------------------

    def _forward(self, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Scaled forward pass. Returns (filtered posteriors, scales, loglik).

        Rescaling to sum 1 at every step is what keeps the recursion alive:
        unscaled, the joint density of a thousand observations underflows to
        zero within a few hundred steps and every posterior becomes 0/0. The
        scale factors are the one-step predictive likelihoods, so their logs
        sum to the log-likelihood — the normalisation is not thrown away, it
        *is* the answer.
        """
        T, k = b.shape
        alpha = np.empty((T, k))
        scale = np.empty(T)

        a = self.startprob * b[0]
        s = a.sum()
        if s <= 0:
            a, s = np.full(k, 1.0 / k), 1.0
        alpha[0], scale[0] = a / s, s

        for t in range(1, T):
            a = (alpha[t - 1] @ self.transmat) * b[t]
            s = a.sum()
            if s <= 0:
                a, s = np.full(k, 1.0 / k), 1.0
            alpha[t], scale[t] = a / s, s

        return alpha, scale, float(np.sum(np.log(scale)))

    def _backward(self, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
        T, k = b.shape
        beta = np.ones((T, k))
        for t in range(T - 2, -1, -1):
            beta[t] = self.transmat @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
        return beta

    def filter(self, x) -> np.ndarray:
        """``P(state_t | x_1..x_t)`` — causal, safe for attribution."""
        b = self._emission(np.asarray(x, dtype=float))
        alpha, _, _ = self._forward(b)
        return alpha

    def smooth(self, x) -> np.ndarray:
        """``P(state_t | x_1..x_T)`` — uses the future; description only."""
        b = self._emission(np.asarray(x, dtype=float))
        alpha, scale, _ = self._forward(b)
        beta = self._backward(b, scale)
        gamma = alpha * beta
        return _safe_normalize(gamma, fallback=alpha)

    def score(self, x) -> float:
        """Log-likelihood of ``x`` under the fitted parameters."""
        b = self._emission(np.asarray(x, dtype=float))
        return self._forward(b)[2]

    def viterbi(self, x) -> np.ndarray:
        """Most likely *joint* state path, in logs.

        Not the same thing as taking the argmax of the smoothed posteriors
        one day at a time: that can produce a path the transition matrix
        assigns zero probability to. Viterbi maximises the path as a whole.
        """
        x = np.asarray(x, dtype=float)
        b = np.log(np.maximum(self._emission(x), 1e-300))
        A = np.log(np.maximum(self.transmat, 1e-300))
        T, k = b.shape

        delta = np.empty((T, k))
        psi = np.zeros((T, k), dtype=int)
        delta[0] = np.log(np.maximum(self.startprob, 1e-300)) + b[0]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + A          # (from, to)
            psi[t] = cand.argmax(axis=0)
            delta[t] = cand.max(axis=0) + b[t]

        path = np.empty(T, dtype=int)
        path[-1] = int(delta[-1].argmax())
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    # -- fitting -----------------------------------------------------------

    def _init(self, x: np.ndarray, seed: int) -> None:
        """Seed EM by splitting the sample on the magnitude of the deviation.

        Random starts are the usual approach and are a bad fit here: the
        likelihood surface has a well-known degenerate corner where one state
        collapses onto a single point with zero variance, and a random start
        finds it often enough to matter. Splitting on ``|x - mean|`` starts EM
        already near the calm/turbulent structure it is meant to find.
        """
        rng = np.random.default_rng(seed)
        k = self.n_states
        dev = np.abs(x - x.mean())
        cuts = np.quantile(dev, np.linspace(0, 1, k + 1)[1:-1])
        labels = np.digitize(dev, cuts)

        means, variances = [], []
        for j in range(k):
            grp = x[labels == j]
            if len(grp) < 2:
                grp = x
            means.append(grp.mean())
            variances.append(max(grp.var(ddof=1), _VAR_FLOOR))
        self.means = np.array(means)
        self.variances = np.array(variances)

        self.startprob = np.full(k, 1.0 / k)
        # Start persistent: regimes that last are the hypothesis being tested.
        stay = 0.95
        self.transmat = np.full((k, k), (1.0 - stay) / (k - 1)) if k > 1 else np.ones((1, 1))
        np.fill_diagonal(self.transmat, stay)
        self.transmat += rng.uniform(0, 1e-3, size=(k, k))
        self.transmat /= self.transmat.sum(axis=1, keepdims=True)

    def fit(
        self,
        x,
        n_iter: int = 500,
        tol: float = 1e-8,
        seed: int = 0,
    ) -> "GaussianHMM":
        """Baum-Welch (EM). Iterates until the log-likelihood gain falls below ``tol``."""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 50:
            raise ValueError(f"need at least 50 observations, got {len(x)}")

        self._init(x, seed)
        k = self.n_states
        prev = -np.inf

        for it in range(1, n_iter + 1):
            b = self._emission(x)
            alpha, scale, loglik = self._forward(b)
            beta = self._backward(b, scale)

            gamma = _safe_normalize(alpha * beta, fallback=alpha)

            # xi[t, i, j] = P(state_t = i, state_{t+1} = j | x)
            xi_num = (
                alpha[:-1, :, None]
                * self.transmat[None, :, :]
                * (b[1:] * beta[1:])[:, None, :]
                / scale[1:, None, None]
            )
            xi = xi_num / xi_num.sum(axis=(1, 2), keepdims=True)

            self.startprob = gamma[0].copy()
            denom = gamma[:-1].sum(axis=0)[:, None]
            self.transmat = xi.sum(axis=0) / np.maximum(denom, 1e-300)
            self.transmat /= self.transmat.sum(axis=1, keepdims=True)

            w = gamma.sum(axis=0)
            self.means = (gamma * x[:, None]).sum(axis=0) / np.maximum(w, 1e-300)
            dev2 = (x[:, None] - self.means[None, :]) ** 2
            self.variances = np.maximum(
                (gamma * dev2).sum(axis=0) / np.maximum(w, 1e-300), _VAR_FLOOR
            )

            self.loglik, self.n_iter = loglik, it
            if loglik - prev < tol:
                self.converged = True
                break
            prev = loglik

        self._order_by_variance()
        return self

    def _order_by_variance(self) -> None:
        order = np.argsort(self.variances)
        self.means = self.means[order]
        self.variances = self.variances[order]
        self.startprob = self.startprob[order]
        self.transmat = self.transmat[np.ix_(order, order)]

    # -- derived quantities ------------------------------------------------

    @property
    def expected_duration(self) -> np.ndarray:
        """Mean run length of each state, ``1/(1 - a_ii)`` days."""
        stay = np.diag(self.transmat)
        return 1.0 / np.maximum(1.0 - stay, 1e-300)

    @property
    def stationary(self) -> np.ndarray:
        """Long-run share of time spent in each state.

        The left eigenvector of the transition matrix for eigenvalue 1.
        """
        vals, vecs = np.linalg.eig(self.transmat.T)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        v = np.abs(v)
        return v / v.sum()


# --------------------------------------------------------------------------
# Convenience wrappers producing labelled pandas output
# --------------------------------------------------------------------------

REGIME_NAMES = {0: "calm", 1: "turbulent"}


@dataclass
class RegimeResult:
    """Fitted regime model plus the per-day classification."""

    model: GaussianHMM
    prob_turbulent: pd.Series     # filtered P(state = high variance)
    state: pd.Series              # Viterbi path, integer coded
    label: pd.Series              # human-readable regime name
    smoothed: pd.Series           # smoothed P(turbulent); description only

    def summary(self) -> pd.DataFrame:
        """One row per state: emission parameters and persistence."""
        m = self.model
        return pd.DataFrame(
            {
                "mean_daily": m.means,
                "ann_vol": np.sqrt(m.variances) * np.sqrt(252),
                "stay_prob": np.diag(m.transmat),
                "expected_duration_days": m.expected_duration,
                "long_run_share": m.stationary,
                "realized_share": [
                    float((self.state == j).mean()) for j in range(m.n_states)
                ],
            },
            index=[REGIME_NAMES.get(j, f"state_{j}") for j in range(m.n_states)],
        )


def fit_regimes(
    returns: pd.Series,
    n_states: int = 2,
    seed: int = 0,
) -> RegimeResult:
    """Fit a Gaussian HMM to a return series and classify every day."""
    r = returns.dropna()
    model = GaussianHMM(n_states=n_states).fit(r.to_numpy(), seed=seed)

    filt = model.filter(r.to_numpy())
    smooth = model.smooth(r.to_numpy())
    path = model.viterbi(r.to_numpy())
    high = n_states - 1

    return RegimeResult(
        model=model,
        prob_turbulent=pd.Series(filt[:, high], index=r.index, name="p_turbulent"),
        state=pd.Series(path, index=r.index, name="state"),
        label=pd.Series(
            [REGIME_NAMES.get(int(s), f"state_{int(s)}") for s in path],
            index=r.index, name="regime",
        ),
        smoothed=pd.Series(smooth[:, high], index=r.index, name="p_turbulent_smoothed"),
    )


def fit_causal(
    returns: pd.Series,
    burn_in: int = 504,
    n_states: int = 2,
    seed: int = 0,
) -> pd.Series:
    """Filtered ``P(turbulent)`` with parameters estimated only on a burn-in.

    The strictly-honest version: parameters come from the first ``burn_in``
    days and are then held fixed while the filter runs forward over the rest.
    Nothing after the burn-in influences any label. Days inside the burn-in
    are returned as NaN rather than back-filled, because there is no
    out-of-sample label for them and pretending otherwise is the exact error
    this function exists to avoid.
    """
    r = returns.dropna()
    if len(r) <= burn_in + 50:
        raise ValueError(
            f"series of {len(r)} days is too short for a {burn_in}-day burn-in"
        )
    model = GaussianHMM(n_states=n_states).fit(r.iloc[:burn_in].to_numpy(), seed=seed)
    filt = model.filter(r.to_numpy())[:, n_states - 1]

    out = pd.Series(filt, index=r.index, name="p_turbulent_causal")
    out.iloc[:burn_in] = np.nan
    return out


def causal_labels(
    returns: pd.Series,
    burn_in: int = 504,
    threshold: float = 0.5,
    n_states: int = 2,
    seed: int = 0,
) -> pd.Series:
    """Causal regime *labels*, the form the attribution actually consumes.

    Wraps ``fit_causal`` and hardens the probability into a name. The burn-in
    stays NaN and is therefore dropped by every attribution function, which is
    the intended behaviour: those days have no out-of-sample label, so they
    contribute to no regime rather than silently joining the calm one.
    """
    p = fit_causal(returns, burn_in=burn_in, n_states=n_states, seed=seed)
    label = p.map(lambda v: np.nan if pd.isna(v)
                  else ("turbulent" if v > threshold else "calm"))
    return label.rename("regime")


def vol_tercile_regimes(vol: pd.Series, min_periods: int = 252) -> pd.Series:
    """Baseline regimes: expanding terciles of a volatility series.

    Expanding rather than full-sample quantiles, so the cut points on day
    ``t`` use only days up to ``t``. Full-sample terciles would label 2015 by
    a threshold computed partly from 2018.
    """
    v = vol.dropna()
    lo = v.expanding(min_periods).quantile(1 / 3)
    hi = v.expanding(min_periods).quantile(2 / 3)
    out = pd.Series(np.nan, index=v.index, dtype=object, name="vol_tercile")
    out[v <= lo] = "low"
    out[(v > lo) & (v <= hi)] = "mid"
    out[v > hi] = "high"
    return out
