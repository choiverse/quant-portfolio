"""Monte Carlo European-option pricing with variance reduction.

Given a model's simulated terminal prices, price vanilla options and report a
standard error. A control variate (the terminal price itself, whose risk-
neutral mean S0*e^{rT} is known exactly) sharply cuts the estimator variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PriceEstimate:
    price: float
    std_error: float
    n_paths: int

    def __repr__(self):
        return f"PriceEstimate(price={self.price:.4f}, se={self.std_error:.4f}, n={self.n_paths})"


def _payoff(S_T, K, option):
    if option == "call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)


def _standard_error(values, antithetic):
    """Standard error of the mean, respecting the antithetic pairing.

    Antithetic sampling deliberately makes path ``i`` and path ``i + N/2``
    *negatively correlated*. Treating all ``N`` draws as independent and
    reporting ``std(values)/sqrt(N)`` therefore measures the wrong thing: it
    ignores the negative covariance that is the entire point of the technique,
    and reports essentially the crude-Monte-Carlo error no matter how much
    variance the pairing actually removed.

    The correct estimator has ``N/2`` independent observations, each being the
    *average of an antithetic pair*. The point estimate is unchanged — the mean
    of the pair means equals the mean of all draws — but the error bar is now
    the honest one.
    """
    values = np.asarray(values)
    n = len(values)
    if antithetic and n % 2 == 0:
        half = n // 2
        pair_means = 0.5 * (values[:half] + values[half:])
        return float(pair_means.std(ddof=1) / np.sqrt(half))
    return float(values.std(ddof=1) / np.sqrt(n))


def price_european(
    model,
    K,
    option="call",
    n_paths=100_000,
    seed=None,
    antithetic=True,
    control_variate=True,
):
    """Price a European option on ``model``. Returns a PriceEstimate.

    ``control_variate`` regresses the discounted payoff on the (centered)
    terminal price to remove the component explained by S_T, whose mean is
    known — a textbook control-variate scheme.
    """
    S_T = model.simulate_terminal(n_paths, seed=seed, antithetic=antithetic)
    disc = np.exp(-model.r * model.T)
    payoff = disc * _payoff(S_T, K, option)

    if control_variate:
        # Control: discounted terminal price, known mean = S0 (since E[disc*S_T]=S0).
        control = disc * S_T
        mean_control = model.S0
        cov = np.cov(payoff, control, ddof=1)
        beta = cov[0, 1] / cov[1, 1]
        adjusted = payoff - beta * (control - mean_control)
    else:
        adjusted = payoff

    price = adjusted.mean()
    se = _standard_error(adjusted, antithetic)
    return PriceEstimate(price=float(price), std_error=se, n_paths=len(adjusted))


def price_strikes(model, strikes, option="call", n_paths=200_000, seed=None,
                  antithetic=True, control_variate=True):
    """Price many strikes from a *single* simulation (common random numbers).

    ``option`` may be a single string ("call"/"put") applied to every strike, or
    a list aligned with ``strikes`` (e.g. out-of-the-money: puts below the
    forward, calls above). Returns a list of PriceEstimate aligned with strikes.
    Reusing one terminal sample is faster and yields a smoother smile than
    pricing each strike on independent paths.
    """
    S_T = model.simulate_terminal(n_paths, seed=seed, antithetic=antithetic)
    disc = np.exp(-model.r * model.T)
    control = disc * S_T
    mean_control = model.S0

    options = [option] * len(strikes) if isinstance(option, str) else list(option)

    estimates = []
    for K, opt in zip(strikes, options):
        payoff = disc * _payoff(S_T, K, opt)
        if control_variate:
            cov = np.cov(payoff, control, ddof=1)
            beta = cov[0, 1] / cov[1, 1]
            adjusted = payoff - beta * (control - mean_control)
        else:
            adjusted = payoff
        estimates.append(
            PriceEstimate(float(adjusted.mean()),
                          _standard_error(adjusted, antithetic),
                          len(adjusted))
        )
    return estimates
